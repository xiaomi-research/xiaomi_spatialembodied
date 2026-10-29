"""EIEA cue encoder: Mamba-2 interpreter + structured-cue encoders + projection.

Reference implementation of the Efficient Image-Embodied Adapter described in
the XEmbodied paper (Sec. 3 / Appendix B) and clarified in the rebuttal.

Canonical sequence fed to the Mamba-2 interpreter (causal scan):

    [ text tokens | image tokens | depth tokens | <PHY>
      | box tokens | <MOD> | BEV-map tokens | (<MOD>; BEV-flow tokens)
      | R_1 ... R_K ]

``<PHY>``/``<MOD>`` are the paper supplement's learned modality markers (they
can be disabled with ``use_modality_tokens=False``).  ``R_1 ... R_K`` are the
K learnable latent reasoning queries.  They are placed
**last** so that their causal hidden states have already seen the question and
every available physical cue.  Only the hidden states at these K positions are
kept; they are mapped to the VLM embedding space by :class:`CueProjector`:

    Z_phy = Linear(GELU(Linear(RMSNorm(H_tor))))  in R^{K x d_VLM}

The order follows the paper supplement (``[X_T; <PHY>; ...]``, text first) and
the rebuttal Q&A (``[X_T; <PHY>; Z_map; <MOD>; Z_flow; <MOD>; Z_box; R_1..R_10]``,
latent tokens last).  An earlier version of this file placed the latent
queries *before* the text; that is wrong for a causal interpreter because the
queries would never see the question.

Missing cues are represented by *omitting their tokens*.  This is what the
training-time random masking in :mod:`model.eiea.corruption` simulates, and it
lets the interpreter degrade gracefully when a detection/map/depth source is
unavailable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EIEAConfig:
    """Shape/architecture configuration for the public EIEA implementation.

    ``image_dim`` / ``depth_dim`` are the CLIP ViT-L and SigLIP-SO400M hidden
    sizes used by the MIDI interpreter.  ``hidden_size`` is the Mamba-2 hidden
    size (mamba-130m uses 768).  ``d_vlm`` is the downstream VLM word-embedding
    size: 2048 for Qwen3-VL-30B-A3B and 3584 for the Qwen2.5-VL-7B proxy.
    """

    image_dim: int = 1024
    depth_dim: int = 1152
    hidden_size: int = 768
    d_vlm: int = 2048
    num_tor: int = 10
    bev_channels: int = 5
    flow_channels: int = 2
    num_box_classes: int = 11
    box_features: int = 10
    max_boxes: int = 100
    dropout: float = 0.0
    # Paper Appendix B: the Mamba-2 body is frozen; only the input/output
    # projections are fine-tuned.  Set False to train the body end-to-end.
    freeze_mamba: bool = True
    normalize_boxes: bool = True
    # Paper supplement: a <PHY> start marker before the physical block and
    # <MOD> separators between heterogeneous modalities.
    use_modality_tokens: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


# Scales for the 10 numeric box features:
# x, y, z, length, width, height, yaw, vx, vy, distance.
BOX_FEATURE_SCALES: Tuple[float, ...] = (
    50.0, 50.0, 10.0, 10.0, 10.0, 10.0, math.pi, 20.0, 20.0, 50.0,
)


# ---------------------------------------------------------------------------
# Sub-encoders
# ---------------------------------------------------------------------------
class BoxEncoder(nn.Module):
    """Encode 3D boxes as one token per box.

    Each box is ``[class_id, cx, cy, cz, length, width, height, yaw, vx, vy,
    dist]`` in the ego frame (same layout produced by
    ``experiments/build_eiea_data.py``).  Numeric features are divided by fixed
    physical scales before the MLP so that metres, radians and m/s enter the
    network on comparable ranges.  The paper specifies a capacity of
    ``L_3D = 100`` box tokens per scene; ``max_boxes`` enforces that cap.
    """

    def __init__(self, cfg: EIEAConfig):
        super().__init__()
        self.cfg = cfg
        self.class_emb = nn.Embedding(cfg.num_box_classes, 16)
        in_dim = 16 + cfg.box_features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, cfg.hidden_size),
        )
        scales = torch.tensor(BOX_FEATURE_SCALES[: cfg.box_features], dtype=torch.float32)
        self.register_buffer("feature_scales", scales, persistent=False)

    def forward(self, boxes: Optional[torch.Tensor]) -> torch.Tensor:
        """Args:
            boxes: ``[N, 1 + box_features]`` float tensor, or ``None``/empty.
        Returns:
            ``[N, hidden_size]``; ``[0, hidden_size]`` when no box is present.
        """
        w = self.class_emb.weight
        if boxes is None or boxes.numel() == 0:
            return torch.zeros(0, self.cfg.hidden_size, device=w.device, dtype=w.dtype)

        boxes = boxes[: self.cfg.max_boxes].to(device=w.device)
        cls = boxes[:, 0].long().clamp(0, self.cfg.num_box_classes - 1)
        feats = boxes[:, 1: 1 + self.cfg.box_features].to(dtype=w.dtype)
        if feats.shape[1] < self.cfg.box_features:
            pad = self.cfg.box_features - feats.shape[1]
            feats = torch.cat([feats, feats.new_zeros(feats.shape[0], pad)], dim=1)
        if self.cfg.normalize_boxes:
            feats = feats / self.feature_scales.to(device=feats.device, dtype=feats.dtype)
        tok = torch.cat([self.class_emb(cls), feats], dim=-1)
        return self.mlp(tok)


class BEVEncoder(nn.Module):
    """Encode a BEV raster into spatial tokens.

    The public paper encoder is DiffMap-style (768-d output, projected to the
    shared Mamba width).  This compact CNN keeps the same interface and emits
    ``tokens`` (default 64) spatial tokens, matching the 8x8 token grid used in
    the EIEA documentation.
    """

    def __init__(self, cfg: EIEAConfig, in_channels: Optional[int] = None,
                 tokens: int = 64):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels if in_channels is not None else cfg.bev_channels
        self.tokens = tokens
        side = int(math.sqrt(tokens))
        if side * side != tokens:
            raise ValueError("tokens must be a perfect square")
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((side, side)),
        )
        self.proj = nn.Linear(128, cfg.hidden_size)

    def forward(self, bev: Optional[torch.Tensor],
                empty_if_zero: bool = True) -> torch.Tensor:
        """Args:
            bev: ``[C, H, W]`` or ``[1, C, H, W]`` float raster.
        Returns:
            ``[tokens, hidden_size]``; ``[0, hidden_size]`` when absent.
        """
        empty = torch.zeros(0, self.cfg.hidden_size, device=self.conv[0].weight.device)
        if bev is None:
            return empty
        if bev.dim() == 4:
            if bev.shape[0] != 1:
                raise ValueError("BEVEncoder expects a single raster")
            bev = bev[0]
        if bev.dim() != 3:
            raise ValueError(f"BEV raster must be [C,H,W], got {tuple(bev.shape)}")
        if bev.shape[0] != self.in_channels:
            raise ValueError(
                f"BEV channels {bev.shape[0]} != configured {self.in_channels}")
        if empty_if_zero and float(bev.detach().abs().sum()) == 0.0:
            return empty
        bev = bev.to(device=self.conv[0].weight.device, dtype=self.conv[0].weight.dtype)
        f = self.conv(bev.unsqueeze(0))                 # [1,128,s,s]
        f = f.flatten(2).transpose(1, 2)                # [1,tokens,128]
        return self.proj(f).squeeze(0)                  # [tokens,hidden]


class FlowEncoder(BEVEncoder):
    """BEV-flow encoder (paper's PriorMotion modality).

    The public data pipeline does not ship flow rasters, so this module is
    optional; it is included so the implementation covers the paper's three
    physical modalities (map, flow, box) without changing the API.
    """

    def __init__(self, cfg: EIEAConfig, in_channels: Optional[int] = None,
                 tokens: int = 64):
        super().__init__(
            cfg,
            in_channels=in_channels if in_channels is not None else cfg.flow_channels,
            tokens=tokens,
        )


class CueProjector(nn.Module):
    """Map Mamba hidden states (d_mamba) to VLM word-embedding space (d_VLM)."""

    def __init__(self, d_mamba: int = 768, d_vlm: int = 2048,
                 dropout: float = 0.0):
        super().__init__()
        self.rms = nn.RMSNorm(d_mamba, eps=1e-6)
        self.fc1 = nn.Linear(d_mamba, d_vlm)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_vlm, d_vlm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(self.rms(x)))))


class MIDIBackbone(nn.Module):
    """Thin wrapper around the SSR MIDI Mamba-2 backbone.

    ``mamba_model`` is a ``MambaForCausalLM`` (or any module exposing
    ``get_input_embeddings`` and a causal ``forward(inputs_embeds=...)``).
    The image/depth projections mirror SSR's MIDI so the public SSR-MIDI
    checkpoint can initialise them; the Mamba body can be frozen per the paper.
    """

    def __init__(self, mamba_model: nn.Module, cfg: EIEAConfig):
        super().__init__()
        self.mamba = mamba_model
        self.cfg = cfg
        self.image_proj = nn.Sequential(
            nn.Linear(cfg.image_dim, cfg.hidden_size), nn.GELU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        self.depth_proj = nn.Sequential(
            nn.Linear(cfg.depth_dim, cfg.hidden_size), nn.GELU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        if cfg.dropout:
            self.image_proj.append(nn.Dropout(cfg.dropout))
            self.depth_proj.append(nn.Dropout(cfg.dropout))

    def text_embeds(self, text_ids: torch.Tensor) -> torch.Tensor:
        return self.mamba.get_input_embeddings()(text_ids)


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------
class EIEACueEncoder(nn.Module):
    """Full EIEA cue encoder producing ``[1, K, d_VLM]`` physical-cue tokens."""

    def __init__(self, mamba_model: nn.Module, cfg: Optional[EIEAConfig] = None):
        super().__init__()
        self.cfg = cfg or EIEAConfig()
        self.midi = MIDIBackbone(mamba_model, self.cfg)
        self.box_enc = BoxEncoder(self.cfg)
        self.bev_enc = BEVEncoder(self.cfg)
        self.flow_enc = FlowEncoder(self.cfg)
        self.proj = CueProjector(self.cfg.hidden_size, self.cfg.d_vlm,
                                 dropout=self.cfg.dropout)
        self.queries = nn.Parameter(
            torch.randn(self.cfg.num_tor, self.cfg.hidden_size) * 0.02)
        self.phy_token = nn.Parameter(
            torch.randn(1, 1, self.cfg.hidden_size) * 0.02)
        self.mod_token = nn.Parameter(
            torch.randn(1, 1, self.cfg.hidden_size) * 0.02)
        if self.cfg.freeze_mamba:
            self.set_mamba_frozen(True)

    # -- construction / persistence ----------------------------------------
    @classmethod
    def from_midi(cls, midi: nn.Module, cfg: Optional[EIEAConfig] = None,
                  copy_weights: bool = True) -> "EIEACueEncoder":
        """Build an encoder around a loaded SSR ``MIDI`` instance.

        ``midi`` is the full SSR MIDI module (``midi.mamba``, ``midi.image_proj``,
        ``midi.depth_proj``).  Only those three components are copied; the
        teacher LLM and ``tor_proj`` (which target the 7B teacher width) are not
        part of the public EIEA path.
        """
        enc = cls(midi.mamba, cfg)
        if copy_weights:
            with torch.no_grad():
                src = midi.state_dict()
                for name, dst in enc.midi.state_dict().items():
                    if name in src and src[name].shape == dst.shape:
                        dst.copy_(src[name])
                # SSR-MIDI already ships a pretrained rationale-to-3584
                # projector.  Reuse its two linear layers when the downstream
                # width matches (the 7B proxy); the 30B path needs a 768->2048
                # map instead, so only the compatible layers are copied.
                tor_proj = getattr(midi, "tor_proj", None)
                if tor_proj is not None and hasattr(tor_proj, "__getitem__"):
                    try:
                        lin1, lin2 = tor_proj[0], tor_proj[-1]
                        if lin1.weight.shape == enc.proj.fc1.weight.shape:
                            enc.proj.fc1.weight.copy_(lin1.weight)
                            if lin1.bias is not None and enc.proj.fc1.bias is not None:
                                enc.proj.fc1.bias.copy_(lin1.bias)
                        if lin2.weight.shape == enc.proj.fc2.weight.shape:
                            enc.proj.fc2.weight.copy_(lin2.weight)
                            if lin2.bias is not None and enc.proj.fc2.bias is not None:
                                enc.proj.fc2.bias.copy_(lin2.bias)
                    except (IndexError, AttributeError):
                        pass
        return enc

    def save_checkpoint(self, path: str, extra: Optional[Dict] = None) -> None:
        payload = {
            "format": "xembodied-eiea-cue-v1",
            "config": self.cfg.to_dict(),
            "state_dict": self.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(cls, path: str, mamba_model: nn.Module,
                        map_location: str = "cpu",
                        strict: bool = True) -> Tuple["EIEACueEncoder", Dict]:
        ck = torch.load(path, map_location=map_location, weights_only=False)
        cfg = EIEAConfig(**ck["config"]) if "config" in ck else EIEAConfig()
        enc = cls(mamba_model, cfg)
        enc.load_state_dict(ck["state_dict"], strict=strict)
        return enc, ck

    # -- training controls -------------------------------------------------
    def set_mamba_frozen(self, frozen: bool = True) -> None:
        self._mamba_frozen = frozen
        for p in self.midi.mamba.parameters():
            p.requires_grad_(not frozen)

    def train(self, mode: bool = True) -> "EIEACueEncoder":
        super().train(mode)
        if getattr(self, "_mamba_frozen", False):
            self.midi.mamba.eval()
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        image_embeds: Optional[torch.Tensor] = None,
        depth_embeds: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        bev: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """Encode one sample's cues.

        Args:
            image_embeds: ``[1, N_img, image_dim]`` CLIP features of the
                camera view that feeds EIEA (normally the front view).
            depth_embeds: ``[1, N_dep, depth_dim]`` SigLIP features of the
                DepthPro map, or ``None`` when depth is missing.
            boxes: ``[N, 11]`` ego-frame boxes, or ``None``/empty.
            bev: ``[C, H, W]`` BEV raster, or ``None``.
            flow: ``[C, H, W]`` BEV-flow raster, or ``None``.
            text_ids: ``[1, L_text]`` question token ids for the Mamba
                interpreter (no TOR markers; the K latent queries are added
                internally).
            return_aux: also return sequence metadata for debugging.

        Returns:
            ``[1, K, d_vlm]`` cue tokens (and optionally an aux dict).
        """
        device = self.queries.device
        dtype = self.queries.dtype
        parts: List[torch.Tensor] = []
        present: Dict[str, bool] = {}

        if text_ids is not None:
            te = self.midi.text_embeds(text_ids.to(device)).to(dtype)
            parts.append(te)
            present["text"] = True
        if image_embeds is not None:
            parts.append(self.midi.image_proj(image_embeds.to(dtype)).to(dtype))
            present["image"] = True
        if depth_embeds is not None:
            parts.append(self.midi.depth_proj(depth_embeds.to(dtype)).to(dtype))
            present["depth"] = True
        physical = []
        if boxes is not None and boxes.numel() > 0:
            bt = self.box_enc(boxes.to(device))
            if bt.shape[0] > 0:
                physical.append(("box", bt.unsqueeze(0).to(dtype)))
                present["box"] = True
        if bev is not None:
            bv = self.bev_enc(bev.to(device))
            if bv.shape[0] > 0:
                physical.append(("bev", bv.unsqueeze(0).to(dtype)))
                present["bev"] = True
        if flow is not None:
            fv = self.flow_enc(flow.to(device))
            if fv.shape[0] > 0:
                physical.append(("flow", fv.unsqueeze(0).to(dtype)))
                present["flow"] = True

        if physical and self.cfg.use_modality_tokens:
            parts.append(self.phy_token.to(device=device, dtype=dtype))
            for i, (_, tok) in enumerate(physical):
                if i:
                    parts.append(self.mod_token.to(device=device, dtype=dtype))
                parts.append(tok)
        else:
            parts.extend(tok for _, tok in physical)

        parts.append(self.queries.unsqueeze(0).to(device=device, dtype=dtype))
        seq = torch.cat(parts, dim=1)
        mask = torch.ones(1, seq.shape[1], dtype=torch.long, device=seq.device)
        out = self.midi.mamba(
            inputs_embeds=seq, attention_mask=mask, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        # The K latent queries are the last K positions by construction.
        tor_hidden = hidden[:, -self.cfg.num_tor:, :]
        cue = self.proj(tor_hidden.squeeze(0)).unsqueeze(0)

        if return_aux:
            aux = {
                "seq_len": int(seq.shape[1]),
                "modalities": present,
                "tor_hidden": tor_hidden,
            }
            return cue, aux
        return cue


__all__ = [
    "EIEAConfig", "BoxEncoder", "BEVEncoder", "FlowEncoder",
    "CueProjector", "MIDIBackbone", "EIEACueEncoder",
]
