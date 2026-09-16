"""
Qwen3-VL-30B-A3B MoE + VGGT 3D token fusion
(Q-Former main connector + deepstack multi-connector with MLP-add fusion).

Architecture overview:
- Main connector: language-initialized Q-Former that selects K geometry tokens
  from VGGT spatial embeddings and CONCATs them before native visual tokens.
- Deepstack connectors: lightweight MLP-based mergers (spatial merge + MLP +
  add fusion) that fuse VGGT intermediate-layer features into Qwen3's deepstack
  features, one connector per deepstack level.
- Base model: Qwen3VLMoeForConditionalGeneration (30B-A3B MoE).

The Q-Former uses nn.MultiheadAttention; _init_weights skips MHA to avoid
ZeRO-3 partitioned-parameter crashes in xavier_uniform_.

Registration lives in: plugin/qwen3_vl_moe_vggt_register.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen3VLMoeForConditionalGeneration
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeCausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

# ==============================================================================
# Helpers
# ==============================================================================

_DEFAULT_VGGT_REPO_DIR = str(Path(__file__).resolve().parent.parent / "vggt")


def _parse_torch_dtype(x: Optional[Union[str, torch.dtype]]) -> torch.dtype:
    if x is None:
        return torch.float32
    if isinstance(x, torch.dtype):
        return x
    s = str(x).lower().strip()
    if s in ("fp16", "float16", "torch.float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32", "torch.float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported VGGT dtype: {x}")


def _normalize_cuda_device(dev: torch.device) -> torch.device:
    if dev.type == "cuda" and dev.index is None:
        return torch.device("cuda:0")
    return dev


# ==============================================================================
# VGGT spatial encoder (3D tokens)
# ==============================================================================
@dataclass
class VGGTSpatialEncoderConfig:
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    spatial_embeds_layer_idx: int = -1


class VGGTSpatialEncoder(nn.Module):
    """
    Wrap the official VGGT repo model, exposing per-block spatial embeddings.
    Each Tensor is shaped [S, P_total, 2*D] where P_total includes camera,
    register, and patch tokens.
    """

    def __init__(
        self,
        cfg: VGGTSpatialEncoderConfig,
        *,
        vggt_repo_dir: str = _DEFAULT_VGGT_REPO_DIR,
        vggt_device: Optional[Union[str, torch.device]] = None,
        vggt_dtype: Optional[Union[str, torch.dtype]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.vggt_device = torch.device(vggt_device) if vggt_device is not None else torch.device("cpu")
        self.vggt_device = _normalize_cuda_device(self.vggt_device)
        self.vggt_dtype = _parse_torch_dtype(vggt_dtype)

        if vggt_repo_dir not in sys.path:
            sys.path.insert(0, vggt_repo_dir)

        from vggt.models.vggt import VGGT

        self.vggt_model = VGGT(img_size=cfg.img_size, patch_size=cfg.patch_size, embed_dim=cfg.embed_dim).eval()

        for head_name in ('camera_head', 'point_head', 'depth_head', 'track_head'):
            if hasattr(self.vggt_model, head_name):
                delattr(self.vggt_model, head_name)

    def _resolve_execution_device(self) -> torch.device:
        aggregator = getattr(self.vggt_model, "aggregator", None)
        hook = getattr(aggregator, "_hf_hook", None) if aggregator is not None else None
        exec_dev = getattr(hook, "execution_device", None)
        if exec_dev is not None:
            try:
                return _normalize_cuda_device(torch.device(exec_dev))
            except Exception:
                pass
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None and torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return self.vggt_device

    def _resolve_execution_dtype(self) -> torch.dtype:
        aggregator = getattr(self.vggt_model, "aggregator", None)
        for m in (aggregator, self.vggt_model):
            if m is None:
                continue
            try:
                return next(m.parameters()).dtype
            except (StopIteration, Exception):
                continue
        return self.vggt_dtype

    @torch.no_grad()
    def forward(
        self,
        image_tchw: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Tuple[List[List[torch.Tensor]], List[int]]:
        if isinstance(image_tchw, torch.Tensor):
            if image_tchw.dim() == 4:
                image_tchw_list = [image_tchw]
            elif image_tchw.dim() == 5:
                image_tchw_list = [image_tchw[i] for i in range(image_tchw.shape[0])]
            else:
                raise ValueError(f"image_tchw must be 4D or 5D, got {tuple(image_tchw.shape)}")
        elif isinstance(image_tchw, list):
            image_tchw_list = image_tchw
        else:
            raise TypeError(f"image_tchw must be Tensor or List[Tensor], got {type(image_tchw)}")

        for i, t in enumerate(image_tchw_list):
            if not isinstance(t, torch.Tensor) or t.dim() != 4:
                raise ValueError(f"image_tchw[{i}] must be 4D [T,C,H,W]")

        group_map = {}
        for idx, v in enumerate(image_tchw_list):
            group_map.setdefault(tuple(v.shape), []).append((idx, v))

        final_outputs: List[List[torch.Tensor]] = [None] * len(image_tchw_list)  # type: ignore[list-item]
        final_indices: List[int] = [None] * len(image_tchw_list)  # type: ignore[list-item]

        for shape, items in group_map.items():
            indices = [i for i, _ in items]
            tensors = [t for _, t in items]
            batch_input = torch.stack(tensors, dim=0)
            exec_device = self._resolve_execution_device()
            exec_dtype = self._resolve_execution_dtype()
            batch_input = batch_input.to(device=exec_device, dtype=exec_dtype)
            try:
                aggregator = getattr(self.vggt_model, "aggregator", None)
                hook = getattr(aggregator, "_hf_hook", None) if aggregator is not None else None
                if aggregator is not None and hook is None:
                    aggregator.to(device=exec_device, dtype=exec_dtype)
            except Exception:
                pass
            batch_out, patch_start_idx = self.vggt_model.aggregator(batch_input)
            for j, real_idx in enumerate(indices):
                final_outputs[real_idx] = [layer_tensor[j] for layer_tensor in batch_out]
                final_indices[real_idx] = int(patch_start_idx)

        return final_outputs, final_indices


# ==============================================================================
# Q-Former connector (main connector)
# ==============================================================================
@dataclass
class VGGTQFormerConnectorConfig:
    input_dim: int
    output_dim: int
    spatial_embeds_layer_idx: int = -1
    qwen_temporal_patch_size: int = 2
    qwen_spatial_merge_size: int = 2
    num_query_tokens: int = 8
    qformer_num_heads: int = 8
    qformer_num_layers: int = 1
    use_camera_token: bool = False
    use_register_tokens: bool = False


class _QFormerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.self_attn_ln = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_attn_ln_q = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.cross_attn_ln_kv = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ffn_ln = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_ln = self.self_attn_ln(q)
        q = q + self.self_attn(q_ln, q_ln, q_ln, need_weights=False)[0]
        q_ln = self.cross_attn_ln_q(q)
        kv_ln = self.cross_attn_ln_kv(kv)
        q = q + self.cross_attn(q_ln, kv_ln, kv_ln, need_weights=False)[0]
        q = q + self.ffn(self.ffn_ln(q))
        return q


class VGGTQFormerConnector(nn.Module):
    """
    Language-initialized Q-Former geometry selector:
    1) Build VGGT memory tokens (e3d by default; ec/register optional).
    2) Initialize query tokens from language seed.
    3) Q-Former cross-attends to memory and selects K geometry tokens.
    4) Concatenate geometry tokens before each visual token block.
    """

    def __init__(self, config: VGGTQFormerConnectorConfig) -> None:
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.spatial_embeds_layer_idx = config.spatial_embeds_layer_idx
        self.qwen_temporal_patch_size = config.qwen_temporal_patch_size
        self.qwen_spatial_merge_size = config.qwen_spatial_merge_size
        self.num_query_tokens = config.num_query_tokens
        self.use_camera_token = config.use_camera_token
        self.use_register_tokens = config.use_register_tokens

        self.memory_proj = nn.Linear(self.input_dim, self.output_dim)
        self.query_proj = nn.Linear(self.output_dim, self.num_query_tokens * self.output_dim)
        self.qformer_blocks = nn.ModuleList(
            [_QFormerBlock(self.output_dim, config.qformer_num_heads) for _ in range(config.qformer_num_layers)]
        )
        self.out_ln = Qwen2RMSNorm(self.output_dim, eps=1e-6)

    def _split_units_by_grid(
        self,
        *,
        media_grid_thw: torch.LongTensor,
        media_tchw: List[torch.Tensor],
    ) -> List[List[Tuple[int, int]]]:
        per_sample_units: List[List[Tuple[int, int]]] = []
        grid_cursor = 0
        tp = int(self.qwen_temporal_patch_size)

        for tchw in media_tchw:
            t_total = int(tchw.shape[0])
            acc = 0
            units: List[Tuple[int, int]] = []
            while acc < t_total:
                if grid_cursor >= int(media_grid_thw.shape[0]):
                    raise ValueError("media_grid_thw rows are insufficient for media_tchw in connector.")
                grid_t = int(media_grid_thw[grid_cursor, 0].item())
                unit_frames = grid_t * tp
                units.append((acc, acc + unit_frames))
                acc += unit_frames
                grid_cursor += 1
            if acc != t_total:
                raise ValueError(f"Cannot align media_tchw T={t_total} with media_grid_thw rows.")
            per_sample_units.append(units)
        return per_sample_units

    def _unit_token_len(self, grid_row: torch.Tensor) -> int:
        t, h, w = [int(x) for x in grid_row.tolist()]
        return int((t * h * w) // (self.qwen_spatial_merge_size ** 2))

    def _build_memory_for_unit(
        self,
        *,
        spatial_embeds: torch.Tensor,
        patch_start_idx: int,
        frame_start: int,
        frame_end: int,
    ) -> torch.Tensor:
        ec = spatial_embeds[frame_start:frame_end, :1, :]
        er = spatial_embeds[frame_start:frame_end, 1:patch_start_idx, :]
        e3d = spatial_embeds[frame_start:frame_end, patch_start_idx:, :]

        mem_parts = [e3d]
        if self.use_camera_token:
            mem_parts.append(ec)
        if self.use_register_tokens:
            mem_parts.append(er)
        mem = torch.cat(mem_parts, dim=1)
        mem = mem.reshape(-1, mem.shape[-1])
        return mem

    def _select_geometry_tokens(
        self,
        *,
        memory_tokens: torch.Tensor,
        query_seed: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        kv = self.memory_proj(memory_tokens.to(device=device, dtype=dtype)).unsqueeze(0)
        q0 = self.query_proj(query_seed.to(device=device, dtype=dtype).unsqueeze(0)).view(
            1, self.num_query_tokens, self.output_dim
        )
        q = q0
        for blk in self.qformer_blocks:
            q = blk(q, kv)
        return self.out_ln(q).squeeze(0)

    def forward(
        self,
        image_embeds: torch.Tensor,
        *,
        spatial_embeds_list: List[List[torch.Tensor]],
        patch_start_idx: List[int],
        image_grid_thw: torch.Tensor,
        image_tchw: List[torch.Tensor],
        query_seeds: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        unit_ranges = self._split_units_by_grid(media_grid_thw=image_grid_thw, media_tchw=image_tchw)

        out_chunks: List[torch.Tensor] = []
        grid_cursor = 0
        visual_cursor = 0

        for b_idx, units in enumerate(unit_ranges):
            spatial_layers = spatial_embeds_list[b_idx]
            spatial_embeds = spatial_layers[self.spatial_embeds_layer_idx]
            if spatial_embeds.dim() != 3:
                raise ValueError(f"Unexpected VGGT token dim: {spatial_embeds.shape}")
            ps = int(patch_start_idx[b_idx])
            if ps <= 1 or ps > int(spatial_embeds.shape[1]):
                raise ValueError(f"Invalid patch_start_idx={ps}, P_total={spatial_embeds.shape[1]}")

            if query_seeds.shape[0] != len(unit_ranges):
                raise ValueError(
                    f"query_seeds has {query_seeds.shape[0]} rows but the batch carries "
                    f"{len(unit_ranges)} media samples; the language-conditioned query seeds "
                    "are not aligned with the VGGT streams."
                )
            query_seed_b = query_seeds[b_idx]
            for frame_start, frame_end in units:
                if grid_cursor >= int(image_grid_thw.shape[0]):
                    raise ValueError("image_grid_thw rows exhausted in connector.")
                token_len = self._unit_token_len(image_grid_thw[grid_cursor])
                native_chunk = image_embeds[visual_cursor : visual_cursor + token_len]
                visual_cursor += token_len

                memory_tokens = self._build_memory_for_unit(
                    spatial_embeds=spatial_embeds,
                    patch_start_idx=ps,
                    frame_start=frame_start,
                    frame_end=frame_end,
                )
                geom_chunk = self._select_geometry_tokens(
                    memory_tokens=memory_tokens,
                    query_seed=query_seed_b,
                    device=device,
                    dtype=dtype,
                ).to(native_chunk.device, native_chunk.dtype)

                out_chunks.append(torch.cat([geom_chunk, native_chunk], dim=0))
                grid_cursor += 1

        if visual_cursor != int(image_embeds.shape[0]):
            raise ValueError(
                f"Native visual token cursor mismatch: used={visual_cursor}, total={int(image_embeds.shape[0])}"
            )
        if grid_cursor != int(image_grid_thw.shape[0]):
            raise ValueError(
                f"image_grid_thw cursor mismatch: used={grid_cursor}, total={int(image_grid_thw.shape[0])}"
            )
        return torch.cat(out_chunks, dim=0)


# ==============================================================================
# Feature fusion module (for deepstack connectors)
# ==============================================================================
@dataclass
class FeatureFusionConfig:
    fusion_method: str = "add"
    hidden_size: int = 2048
    num_heads: int = 8
    dropout: float = 0.1
    num_layers: int = 1


class CrossAttentionBlock(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1_query = nn.LayerNorm(hidden_size)
        self.norm1_key = nn.LayerNorm(hidden_size)
        self.norm1_value = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size), nn.Dropout(dropout),
        )

    def _get_2d_sincos_pos_embed(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        embed_dim = self.hidden_size
        assert embed_dim % 4 == 0
        half = embed_dim // 2
        omega = torch.arange(half // 2, dtype=torch.float32, device=device)
        omega = 1.0 / (10000.0 ** (omega / (half / 2.0)))
        grid_h = torch.arange(height, dtype=torch.float32, device=device)
        grid_w = torch.arange(width, dtype=torch.float32, device=device)
        out_h = grid_h.unsqueeze(1) * omega.unsqueeze(0)
        emb_h = torch.cat([out_h.sin(), out_h.cos()], dim=1)
        out_w = grid_w.unsqueeze(1) * omega.unsqueeze(0)
        emb_w = torch.cat([out_w.sin(), out_w.cos()], dim=1)
        pos = torch.zeros(height, width, embed_dim, device=device, dtype=dtype)
        pos[:, :, :half] = emb_h.unsqueeze(1).expand(-1, width, -1)
        pos[:, :, half:] = emb_w.unsqueeze(0).expand(height, -1, -1)
        return pos.view(height * width, embed_dim)

    def forward(self, features_2d, features_3d, num_images, h_grid, w_grid):
        query = self.norm1_query(features_2d)
        key = self.norm1_key(features_3d)
        value = self.norm1_value(features_3d)
        if query.dim() == 2:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        single_pos_embed = self._get_2d_sincos_pos_embed(h_grid, w_grid, query.device, query.dtype)
        pos_embed = single_pos_embed.repeat(num_images, 1)
        query = query + pos_embed.unsqueeze(0)
        key = key + pos_embed.unsqueeze(0)
        attn_output, _ = self.cross_attention(query, key, value)
        if squeeze_output:
            attn_output = attn_output.squeeze(0)
        x = features_2d + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class FeatureFusionModule(nn.Module):

    SUPPORTED_METHODS = ("add", "concat", "gated", "weighted", "cross_attention")

    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method
        self.hidden_size = config.hidden_size
        if self.fusion_method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown fusion method '{self.fusion_method}'. Supported: {self.SUPPORTED_METHODS}")
        self._build_fusion_layers()

    def _build_fusion_layers(self):
        if self.fusion_method == "concat":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
        elif self.fusion_method == "cross_attention":
            self.cross_attn_blocks = nn.ModuleList([
                CrossAttentionBlock(self.hidden_size, self.config.num_heads, self.config.dropout)
                for _ in range(self.config.num_layers)
            ])
        elif self.fusion_method == "gated":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.gate_projection = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Sigmoid())
        elif self.fusion_method == "weighted":
            self.weight_2d = nn.Parameter(torch.tensor(0.5))
            self.weight_3d = nn.Parameter(torch.tensor(0.5))

    def forward(self, features_2d, features_3d, *, sample_token_counts=None, sample_nhw=None):
        if self.fusion_method == "add":
            return features_2d + features_3d
        if self.fusion_method == "concat":
            return self.projection(torch.cat([self.norm1(features_2d), self.norm2(features_3d)], dim=-1))
        if self.fusion_method == "cross_attention":
            if sample_token_counts is None:
                sample_token_counts = [features_2d.shape[0]]
            q_chunks = list(torch.split(features_2d, sample_token_counts, dim=0))
            kv_chunks = list(torch.split(features_3d, sample_token_counts, dim=0))
            out_chunks: List[torch.Tensor] = []
            for q_c, kv_c, (ni, hg, wg) in zip(q_chunks, kv_chunks, sample_nhw):
                x = q_c
                for block in self.cross_attn_blocks:
                    x = block(x.unsqueeze(0), kv_c.unsqueeze(0), ni, hg, wg).squeeze(0)
                out_chunks.append(x)
            return torch.cat(out_chunks, dim=0)
        if self.fusion_method == "gated":
            f2d, f3d = self.norm1(features_2d), self.norm2(features_3d)
            gate = self.gate_projection(torch.cat([f2d, f3d], dim=-1))
            return gate * f2d + (1 - gate) * f3d
        if self.fusion_method == "weighted":
            ws = self.weight_2d + self.weight_3d
            return (self.weight_2d / ws) * features_2d + (self.weight_3d / ws) * features_3d
        raise ValueError(f"Unknown fusion method: {self.fusion_method}")


# ==============================================================================
# Deepstack connector (simple MLP + add)
# ==============================================================================
@dataclass
class VGGTDeepstackConnectorConfig:
    input_dim: int
    output_dim: int
    spatial_embeds_layer_idx: int = -1
    vggt_patch_size: int = 14
    qwen_temporal_patch_size: int = 2
    qwen_spatial_merge_size: int = 2
    merger_hidden_dim: int = 4096
    fusion_method: str = "add"
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.1
    fusion_num_layers: int = 1


class VGGTDeepstackConnector(nn.Module):
    """
    Lightweight MLP-based connector for deepstack features.
    Resamples VGGT spatial tokens to match the Qwen3 grid, applies spatial
    merge + MLP projection, then fuses with Qwen3 deepstack features via
    FeatureFusionModule (default: element-wise add).
    """

    def __init__(self, config: VGGTDeepstackConnectorConfig) -> None:
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.spatial_embeds_layer_idx = config.spatial_embeds_layer_idx
        self.vggt_patch_size = config.vggt_patch_size
        self.qwen_temporal_patch_size = config.qwen_temporal_patch_size
        self.qwen_spatial_merge_size = config.qwen_spatial_merge_size
        self.merger_hidden_dim = config.merger_hidden_dim

        self.token_merge_in_dim = self.input_dim * (self.qwen_spatial_merge_size ** 2)
        self.token_merge_ln_q = Qwen2RMSNorm(self.input_dim, eps=1e-6)
        self.token_merge_mlp = nn.Sequential(
            nn.Linear(self.token_merge_in_dim, self.merger_hidden_dim),
            nn.GELU(),
            nn.Linear(self.merger_hidden_dim, self.output_dim),
        )

        fusion_cfg = FeatureFusionConfig(
            fusion_method=config.fusion_method,
            hidden_size=config.output_dim,
            num_heads=config.fusion_num_heads,
            dropout=config.fusion_dropout,
            num_layers=config.fusion_num_layers,
        )
        self.fusion = FeatureFusionModule(fusion_cfg)

    def _infer_vggt_hw(self, image_tchw: torch.Tensor) -> Tuple[int, int]:
        h, w = int(image_tchw.shape[-2]), int(image_tchw.shape[-1])
        if h % self.vggt_patch_size != 0 or w % self.vggt_patch_size != 0:
            raise ValueError(f"VGGT input H,W must be multiple of {self.vggt_patch_size}, got {h}x{w}")
        return h // self.vggt_patch_size, w // self.vggt_patch_size

    def _interp_one_image(self, vggt_frames, *, h_v, w_v, h_q, w_q):
        t, p_v, dd = vggt_frames.shape
        if p_v != h_v * w_v:
            raise ValueError(f"VGGT P mismatch: P={p_v}, H_v*W_v={h_v * w_v}")
        x = vggt_frames.view(t, h_v, w_v, dd).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(h_q, w_q), mode="nearest-exact")
        x = x.permute(0, 2, 3, 1).contiguous().view(t, h_q * w_q, dd)
        return x

    def forward(
        self,
        image_embeds: torch.Tensor,
        *,
        spatial_embeds_list: List[List[torch.Tensor]],
        patch_start_idx: List[int],
        image_grid_thw: torch.Tensor,
        image_tchw: List[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        spatial_embeds_layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        all_spatial_tokens: List[torch.Tensor] = []
        sample_token_counts: List[int] = []
        sample_nhw: List[Tuple[int, int, int]] = []
        grid_idx = 0

        for b_idx, spatial_layers in enumerate(spatial_embeds_list):
            layer_idx = spatial_embeds_layer_idx if spatial_embeds_layer_idx is not None else self.spatial_embeds_layer_idx
            spatial_embeds = spatial_layers[layer_idx]
            if spatial_embeds.dim() != 3:
                raise ValueError(f"Unexpected VGGT token dim: {spatial_embeds.shape}")

            ps = int(patch_start_idx[b_idx])
            if ps <= 1 or ps > spatial_embeds.shape[1]:
                raise ValueError(f"Invalid patch_start_idx={ps} for P_total={spatial_embeds.shape[1]}")

            patch_tokens = spatial_embeds[:, ps:, :]
            h_v, w_v = self._infer_vggt_hw(image_tchw[b_idx])

            t_total = int(image_tchw[b_idx].shape[0])

            if patch_tokens.shape[0] != t_total:
                raise ValueError(f"VGGT temporal S mismatch: got {patch_tokens.shape[0]}, expect {t_total}")

            remaining_grids = len(image_grid_thw) - grid_idx
            if t_total <= remaining_grids:
                num_images = t_total
                frames_per_image = 1
            else:
                frames_per_image = self.qwen_temporal_patch_size
                num_images = t_total // frames_per_image

            sample_spatial_per_image: List[torch.Tensor] = []
            sample_hw: List[Tuple[int, int]] = []
            for img_i in range(num_images):
                if grid_idx >= len(image_grid_thw):
                    raise ValueError("Not enough image_grid_thw rows for batch image_tchw")
                t_q, h_q, w_q = [int(x) for x in image_grid_thw[grid_idx].tolist()]
                if t_q != 1:
                    raise ValueError(f"Unsupported image grid_t={t_q} for images (expect 1).")
                vggt_frames = patch_tokens[img_i * frames_per_image : (img_i + 1) * frames_per_image]
                vggt_frames = self._interp_one_image(vggt_frames, h_v=h_v, w_v=w_v, h_q=h_q, w_q=w_q)
                sample_spatial_per_image.append(vggt_frames)
                sample_hw.append((h_q, w_q))
                grid_idx += 1

            # Require an identical (H, W) grid -- not merely the same token count -- for every
            # image in the sample. Two grids such as 56x100 and 100x56 share the product, but
            # reshaping by the first image's shape would silently scramble the KV spatial order.
            # (This variant previously had no per-sample grid check at all.)
            h_q, w_q = sample_hw[0]
            for other_h, other_w in sample_hw[1:]:
                if (other_h, other_w) != (h_q, w_q):
                    raise ValueError(
                        "Per-sample images have different Qwen3 patch grids "
                        f"({h_q}x{w_q} vs {other_h}x{other_w}); not supported yet."
                    )
            spatial_q = torch.cat(sample_spatial_per_image, dim=0)
            s, p_q, dd = spatial_q.shape
            if p_q != h_q * w_q:
                raise ValueError(f"Qwen3 P mismatch after interp: {p_q} vs {h_q*w_q}")

            if frames_per_image > 1:
                spatial_q = spatial_q.view(num_images, frames_per_image, h_q, w_q, dd)
                spatial_q = spatial_q.mean(dim=1)
            else:
                spatial_q = spatial_q.view(num_images, h_q, w_q, dd)

            ms = self.qwen_spatial_merge_size
            merged_h = h_q // ms
            merged_w = w_q // ms
            if h_q % ms != 0 or w_q % ms != 0:
                raise ValueError(f"Qwen3 grid H/W must be divisible by merge_size={ms}, got {h_q}x{w_q}")
            spatial_q = spatial_q.view(num_images, merged_h, ms, merged_w, ms, dd)
            spatial_q = spatial_q.permute(0, 1, 3, 2, 4, 5).contiguous()
            spatial_q = spatial_q.view(-1, ms ** 2, self.input_dim)
            all_spatial_tokens.append(spatial_q)

            sample_token_counts.append(num_images * merged_h * merged_w)
            sample_nhw.append((num_images, merged_h, merged_w))

        spatial_cat = torch.cat(all_spatial_tokens, dim=0)
        spatial_cat = spatial_cat.to(device=device, dtype=dtype)
        spatial_normed = self.token_merge_ln_q(spatial_cat)
        spatial_flat = spatial_normed.view(-1, self.token_merge_in_dim)
        projected_3d_tokens = self.token_merge_mlp(spatial_flat)
        projected_3d_tokens = projected_3d_tokens.to(device=image_embeds.device, dtype=image_embeds.dtype)

        if projected_3d_tokens.shape[0] != image_embeds.shape[0]:
            raise ValueError(
                f"Token count mismatch before fusion: 3D={projected_3d_tokens.shape[0]} vs 2D={image_embeds.shape[0]}"
            )

        return self.fusion(
            image_embeds, projected_3d_tokens,
            sample_token_counts=sample_token_counts, sample_nhw=sample_nhw,
        )


# ==============================================================================
# Main model class
# ==============================================================================
class Qwen3VLMoeVGGTForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):
    """
    Qwen3-VL-30B-A3B MoE + VGGT with Q-Former main connector and
    MLP-add deepstack connectors.
    """

    def __init__(self, config):
        super().__init__(config)

        spatial_cfg = getattr(config, "vggt_spatial_config", None) or {}
        self.vggt_spatial_config = VGGTSpatialEncoderConfig(**spatial_cfg)

        self.vggt_repo_dir = getattr(config, "vggt_repo_dir", _DEFAULT_VGGT_REPO_DIR)
        self.vggt_device = getattr(config, "vggt_device", "cpu")
        self.vggt_dtype = getattr(config, "vggt_dtype", "float32")

        self.spatial_encoder = VGGTSpatialEncoder(
            self.vggt_spatial_config,
            vggt_repo_dir=self.vggt_repo_dir,
            vggt_device=self.vggt_device,
            vggt_dtype=self.vggt_dtype,
        )

        # --- Q-Former main connector ---
        connector_cfg = getattr(config, "vggt_connector_config", None) or {}
        spatial_embeds_layer_idx = connector_cfg.get(
            "spatial_embeds_layer_idx", self.vggt_spatial_config.spatial_embeds_layer_idx
        )
        self.num_query_tokens = int(connector_cfg.get("num_query_tokens", 8))
        qformer_config = VGGTQFormerConnectorConfig(
            input_dim=2 * self.vggt_spatial_config.embed_dim,
            output_dim=config.text_config.hidden_size,
            spatial_embeds_layer_idx=spatial_embeds_layer_idx,
            qwen_temporal_patch_size=config.vision_config.temporal_patch_size,
            qwen_spatial_merge_size=config.vision_config.spatial_merge_size,
            num_query_tokens=self.num_query_tokens,
            qformer_num_heads=int(connector_cfg.get("qformer_num_heads", 8)),
            qformer_num_layers=int(connector_cfg.get("qformer_num_layers", 1)),
            use_camera_token=bool(connector_cfg.get("use_camera_token", False)),
            use_register_tokens=bool(connector_cfg.get("use_register_tokens", False)),
        )
        self.connector = VGGTQFormerConnector(qformer_config)

        # --- Deepstack connectors (simple MLP + add) ---
        num_deepstack_layers = len(self.model.visual.deepstack_merger_list)
        deepstack_config = VGGTDeepstackConnectorConfig(
            input_dim=2 * self.vggt_spatial_config.embed_dim,
            output_dim=config.text_config.hidden_size,
            spatial_embeds_layer_idx=spatial_embeds_layer_idx,
            vggt_patch_size=self.vggt_spatial_config.patch_size,
            qwen_temporal_patch_size=config.vision_config.temporal_patch_size,
            qwen_spatial_merge_size=config.vision_config.spatial_merge_size,
            merger_hidden_dim=int(connector_cfg.get("merger_hidden_dim", 4096)),
            fusion_method=connector_cfg.get("deepstack_fusion_method", "add"),
            fusion_num_heads=int(connector_cfg.get("fusion_num_heads", 8)),
            fusion_dropout=float(connector_cfg.get("fusion_dropout", 0.1)),
            fusion_num_layers=int(connector_cfg.get("fusion_num_layers", 1)),
        )
        self.deepstack_connectors = nn.ModuleList([
            VGGTDeepstackConnector(deepstack_config) for _ in range(num_deepstack_layers)
        ])
        self.vggt_intermediate_layer_indices = connector_cfg.get(
            "vggt_intermediate_layer_indices", [5, 11, 17]
        )

        # VGGT is a frozen feature extractor: it runs under `torch.no_grad()` and is never
        # optimized. Marking it non-trainable keeps its parameters out of the optimizer state
        # and DeepSpeed ZeRO partitions, and avoids DDP "unused parameter" reduction errors.
        self.spatial_encoder.requires_grad_(False)

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.MultiheadAttention):
            return
        super()._init_weights(module)

    @property
    def vggt_embedding_merger(self) -> VGGTQFormerConnector:
        return self.connector

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if any(k.startswith("vggt_embedding_merger.") for k in state_dict.keys()):
            state_dict = dict(state_dict)
            for key in list(state_dict.keys()):
                if not key.startswith("vggt_embedding_merger."):
                    continue
                new_key = "connector." + key[len("vggt_embedding_merger."):]
                state_dict.setdefault(new_key, state_dict[key])
                state_dict.pop(key)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    # ------------------------------------------------------------------
    # Static / internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_image_tchw(image_tchw):
        if image_tchw is None:
            return []
        if isinstance(image_tchw, torch.Tensor):
            if image_tchw.dim() == 4:
                return [image_tchw]
            if image_tchw.dim() == 5:
                return [image_tchw[i] for i in range(image_tchw.shape[0])]
            raise ValueError(f"image_tchw must be 4D or 5D, got {tuple(image_tchw.shape)}")
        if isinstance(image_tchw, list):
            return image_tchw
        raise TypeError(f"image_tchw must be Tensor or List[Tensor], got {type(image_tchw)}")

    @staticmethod
    def _expand_video_grid_thw(video_grid_thw):
        if video_grid_thw is None:
            raise ValueError("video_grid_thw is None")
        expanded = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        expanded = expanded.clone()
        expanded[:, 0] = 1
        return expanded

    def _build_language_query_seeds(self, *, input_ids, inputs_embeds, attention_mask=None, media_token_id=None):
        """
        Build one language-conditioned query seed per sample.

        When `media_token_id` is given, only samples that actually contain that token
        (image or video) are returned. The VGGT streams are gathered only for samples
        that carry media, so without this filter a batch mixing text-only and multimodal
        samples would index the seeds by position in the *whole* batch and hand a
        multimodal sample the seed derived from a text-only one.
        """
        if input_ids is None:
            return inputs_embeds.new_zeros((inputs_embeds.shape[0], inputs_embeds.shape[-1]))

        text_mask = (input_ids != self.config.image_token_id) & (input_ids != self.config.video_token_id)
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2:
            text_mask = text_mask & attention_mask.to(dtype=torch.bool, device=text_mask.device)
        text_mask = text_mask.to(inputs_embeds.device)

        seeds: List[torch.Tensor] = []
        for b in range(inputs_embeds.shape[0]):
            mask_b = text_mask[b]
            if int(mask_b.sum().item()) == 0:
                seeds.append(inputs_embeds[b].mean(dim=0))
            else:
                seeds.append(inputs_embeds[b][mask_b].mean(dim=0))
        seeds = torch.stack(seeds, dim=0)

        if media_token_id is not None:
            keep = (input_ids == int(media_token_id)).any(dim=-1)
            seeds = seeds[keep]
        return seeds

    def _prepend_zero_geom_to_deepstack(
        self,
        *,
        deepstack_embeds: List[torch.Tensor],
        media_grid_thw: torch.LongTensor,
        media_tchw_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        tp = int(self.config.vision_config.temporal_patch_size)
        ms2 = int(self.config.vision_config.spatial_merge_size ** 2)

        per_sample_units: List[List[Tuple[int, int]]] = []
        grid_cursor = 0
        for tchw in media_tchw_list:
            t_total = int(tchw.shape[0])
            acc = 0
            units: List[Tuple[int, int]] = []
            while acc < t_total:
                if grid_cursor >= int(media_grid_thw.shape[0]):
                    raise ValueError("media_grid_thw rows insufficient for deepstack alignment.")
                grid_t = int(media_grid_thw[grid_cursor, 0].item())
                unit_frames = grid_t * tp
                units.append((acc, acc + unit_frames))
                acc += unit_frames
                grid_cursor += 1
            if acc != t_total:
                raise ValueError("Cannot align media_tchw and media_grid_thw for deepstack alignment.")
            per_sample_units.append(units)

        aligned_layers: List[torch.Tensor] = []
        for layer_embed in deepstack_embeds:
            chunks: List[torch.Tensor] = []
            grid_cursor = 0
            token_cursor = 0
            for units in per_sample_units:
                for _ in units:
                    t, h, w = [int(x) for x in media_grid_thw[grid_cursor].tolist()]
                    native_len = int((t * h * w) // ms2)
                    native_chunk = layer_embed[token_cursor : token_cursor + native_len]
                    token_cursor += native_len
                    zeros = native_chunk.new_zeros((self.num_query_tokens, native_chunk.shape[-1]))
                    chunks.append(torch.cat([zeros, native_chunk], dim=0))
                    grid_cursor += 1
            if token_cursor != int(layer_embed.shape[0]):
                raise ValueError(
                    f"Deepstack native token cursor mismatch: used={token_cursor}, total={int(layer_embed.shape[0])}"
                )
            aligned_layers.append(torch.cat(chunks, dim=0))
        return aligned_layers

    @staticmethod
    def _collapse_temporal_spatial(
        spatial_embeds_list: List[List[torch.Tensor]],
        tp: int,
    ) -> List[List[torch.Tensor]]:
        """Average every ``tp`` consecutive VGGT frames for deepstack connector input."""
        collapsed = []
        for spatial_layers in spatial_embeds_list:
            collapsed_layers = []
            for layer_tensor in spatial_layers:
                S = layer_tensor.shape[0]
                if tp > 1 and S % tp == 0 and S > 0:
                    c = layer_tensor.view(S // tp, tp, *layer_tensor.shape[1:]).mean(dim=1)
                else:
                    c = layer_tensor
                collapsed_layers.append(c)
            collapsed.append(collapsed_layers)
        return collapsed

    # ------------------------------------------------------------------
    # Qwen3 visual feature extraction
    # ------------------------------------------------------------------
    def _get_qwen3_image_embeds(self, *, input_ids, inputs_embeds, pixel_values, image_grid_thw):
        image_embeds_list, deepstack_image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        if not isinstance(deepstack_image_embeds, list):
            deepstack_image_embeds = [deepstack_image_embeds]
        return image_embeds, deepstack_image_embeds

    def _get_qwen3_video_embeds(self, *, pixel_values_videos, video_grid_thw, inputs_embeds):
        video_embeds_list, deepstack_video_embeds = self.model.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        if not isinstance(deepstack_video_embeds, list):
            deepstack_video_embeds = [deepstack_video_embeds]
        return video_embeds, deepstack_video_embeds

    # ------------------------------------------------------------------
    # Inject fused embeds (Q-Former + deepstack)
    # ------------------------------------------------------------------
    def _inject_fused_image_embeds(
        self,
        *,
        input_ids,
        inputs_embeds,
        attention_mask,
        image_embeds,
        deepstack_image_embeds,
        image_grid_thw,
        image_tchw_list,
    ):
        spatial_embeds_list, patch_start_idx = self.spatial_encoder(image_tchw_list)

        query_seeds = self._build_language_query_seeds(
            input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            media_token_id=self.config.image_token_id,
        )

        fused_image_embeds = self.connector(
            image_embeds,
            spatial_embeds_list=spatial_embeds_list,
            patch_start_idx=patch_start_idx,
            image_grid_thw=image_grid_thw,
            image_tchw=image_tchw_list,
            query_seeds=query_seeds,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

        image_mask, _ = self.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=fused_image_embeds,
        )
        fused_image_embeds = fused_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        new_inputs_embeds = inputs_embeds.masked_scatter(image_mask, fused_image_embeds)

        # Collapse temporal repeat in VGGT spatial embeds for deepstack connectors
        tp = int(self.config.vision_config.temporal_patch_size)
        collapsed_spatial = self._collapse_temporal_spatial(spatial_embeds_list, tp) if tp > 1 else spatial_embeds_list
        collapsed_tchw_list = [t[::tp] for t in image_tchw_list] if tp > 1 else image_tchw_list

        fused_deepstack_embeds = []
        for ds_idx, (deep_feat, ds_connector, vggt_layer_idx) in enumerate(
            zip(deepstack_image_embeds, self.deepstack_connectors, self.vggt_intermediate_layer_indices)
        ):
            fused_ds = ds_connector(
                deep_feat,
                spatial_embeds_list=collapsed_spatial,
                patch_start_idx=patch_start_idx,
                image_grid_thw=image_grid_thw,
                image_tchw=collapsed_tchw_list,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                spatial_embeds_layer_idx=vggt_layer_idx,
            )
            fused_deepstack_embeds.append(fused_ds.to(new_inputs_embeds.device, new_inputs_embeds.dtype))

        fused_deepstack_embeds = self._prepend_zero_geom_to_deepstack(
            deepstack_embeds=fused_deepstack_embeds,
            media_grid_thw=image_grid_thw,
            media_tchw_list=image_tchw_list,
        )
        return new_inputs_embeds, image_mask, fused_deepstack_embeds

    def _inject_fused_video_embeds(
        self,
        *,
        input_ids,
        inputs_embeds,
        attention_mask,
        video_embeds,
        deepstack_video_embeds,
        video_grid_thw,
        video_tchw_list,
    ):
        spatial_embeds_list, patch_start_idx = self.spatial_encoder(video_tchw_list)

        query_seeds = self._build_language_query_seeds(
            input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            media_token_id=self.config.video_token_id,
        )

        # Q-Former connector uses original video_grid_thw (units span grid_t * tp frames)
        fused_video_embeds = self.connector(
            video_embeds,
            spatial_embeds_list=spatial_embeds_list,
            patch_start_idx=patch_start_idx,
            image_grid_thw=video_grid_thw,
            image_tchw=video_tchw_list,
            query_seeds=query_seeds,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

        _, video_mask = self.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=fused_video_embeds,
        )
        fused_video_embeds = fused_video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        new_inputs_embeds = inputs_embeds.masked_scatter(video_mask, fused_video_embeds)

        # Deepstack connectors use expanded grid (grid_t=1 per row) + collapsed spatial
        tp = int(self.config.vision_config.temporal_patch_size)
        video_grid_thw_exp = self._expand_video_grid_thw(video_grid_thw)
        collapsed_spatial = self._collapse_temporal_spatial(spatial_embeds_list, tp) if tp > 1 else spatial_embeds_list
        collapsed_tchw_list = [t[::tp] for t in video_tchw_list] if tp > 1 else video_tchw_list

        fused_deepstack_embeds = []
        for ds_idx, (deep_feat, ds_connector, vggt_layer_idx) in enumerate(
            zip(deepstack_video_embeds, self.deepstack_connectors, self.vggt_intermediate_layer_indices)
        ):
            fused_ds = ds_connector(
                deep_feat,
                spatial_embeds_list=collapsed_spatial,
                patch_start_idx=patch_start_idx,
                image_grid_thw=video_grid_thw_exp,
                image_tchw=collapsed_tchw_list,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                spatial_embeds_layer_idx=vggt_layer_idx,
            )
            fused_deepstack_embeds.append(fused_ds.to(new_inputs_embeds.device, new_inputs_embeds.dtype))

        fused_deepstack_embeds = self._prepend_zero_geom_to_deepstack(
            deepstack_embeds=fused_deepstack_embeds,
            media_grid_thw=video_grid_thw,
            media_tchw_list=video_tchw_list,
        )
        return new_inputs_embeds, video_mask, fused_deepstack_embeds

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_tchw: Optional[List[torch.FloatTensor]] = None,
        video_tchw: Optional[List[torch.FloatTensor]] = None,
        **kwargs,
    ):
        if image_tchw is None and "images_input" in kwargs:
            image_tchw = kwargs.pop("images_input")
        if video_tchw is None and "videos_input" in kwargs:
            video_tchw = kwargs.pop("videos_input")

        if pixel_values is None and pixel_values_videos is None:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels,
                pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                cache_position=cache_position, logits_to_keep=logits_to_keep, **kwargs,
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            if image_tchw is None:
                raise ValueError("`image_tchw` must be provided when `pixel_values` is not None for VGGT fusion.")
            if image_grid_thw is None:
                raise ValueError("`image_grid_thw` must be provided when `pixel_values` is not None.")
            image_embeds, deepstack_image_embeds = self._get_qwen3_image_embeds(
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            )
            image_tchw_list = self._normalize_image_tchw(image_tchw)
            inputs_embeds, image_mask, deepstack_image_embeds = self._inject_fused_image_embeds(
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                attention_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
                image_embeds=image_embeds, deepstack_image_embeds=deepstack_image_embeds,
                image_grid_thw=image_grid_thw, image_tchw_list=image_tchw_list,
            )

        if pixel_values_videos is not None:
            if video_tchw is None:
                raise ValueError("`video_tchw` must be provided when `pixel_values_videos` is not None for VGGT fusion.")
            if video_grid_thw is None:
                raise ValueError("`video_grid_thw` must be provided when `pixel_values_videos` is not None.")
            video_embeds, deepstack_video_embeds = self._get_qwen3_video_embeds(
                pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
            )
            video_tchw_list = self._normalize_image_tchw(video_tchw)
            inputs_embeds, video_mask, deepstack_video_embeds = self._inject_fused_video_embeds(
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                attention_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
                video_embeds=video_embeds, deepstack_video_embeds=deepstack_video_embeds,
                video_grid_thw=video_grid_thw, video_tchw_list=video_tchw_list,
            )

        # Aggregate visual masks and deepstack embeds
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask_2d = image_mask.to(inputs_embeds.device)[..., 0]
            video_mask_2d = video_mask.to(inputs_embeds.device)[..., 0]
            visual_pos_masks = image_mask_2d | video_mask_2d
            deepstack_visual_embeds = []
            image_mask_joint = image_mask_2d[visual_pos_masks]
            video_mask_joint = video_mask_2d[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):  # type: ignore[arg-type]
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint.to(inputs_embeds.device, inputs_embeds.dtype))
        elif image_mask is not None:
            visual_pos_masks = image_mask.to(inputs_embeds.device)[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds  # type: ignore[assignment]
        elif video_mask is not None:
            visual_pos_masks = video_mask.to(inputs_embeds.device)[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds  # type: ignore[assignment]

        if attention_mask is not None:
            if isinstance(attention_mask, dict):
                attention_mask = {
                    k: v.to(inputs_embeds.device) if isinstance(v, torch.Tensor) else v
                    for k, v in attention_mask.items()
                }
            elif isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(inputs_embeds.device)

        # Q-Former concat introduces extra geometry tokens; use monotonic text-style
        # position IDs to avoid native multimodal RoPE/grid mismatches.
        if position_ids is None:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            self.model.rope_deltas = torch.zeros(
                (batch_size, 1), device=inputs_embeds.device, dtype=position_ids.dtype
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )
        hidden_states = outputs[0]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.model.rope_deltas,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        cache_position = model_inputs.get("cache_position", None)
        if cache_position is not None and cache_position[0] != 0:
            model_inputs["image_tchw"] = None
            model_inputs["video_tchw"] = None
            model_inputs["pixel_values"] = None
            model_inputs["image_grid_thw"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["video_grid_thw"] = None
        return model_inputs


__all__ = [
    "VGGTSpatialEncoderConfig",
    "VGGTSpatialEncoder",
    "VGGTQFormerConnectorConfig",
    "VGGTQFormerConnector",
    "FeatureFusionConfig",
    "FeatureFusionModule",
    "VGGTDeepstackConnectorConfig",
    "VGGTDeepstackConnector",
    "Qwen3VLMoeVGGTForConditionalGeneration",
]

# Backward-compatible aliases.
VGGTEmbeddingMerger = VGGTQFormerConnector
VGGTEmbeddingMergerConfig = VGGTQFormerConnectorConfig
Qwen3VLVGGTForConditionalGeneration = Qwen3VLMoeVGGTForConditionalGeneration
