"""
Qwen3-VL-30B-A3B + VGGT 3D token fusion + TOR Embedding injection interface.
Preserves full VGGT functionality, adds TOR adapter (aligned with Qwen3-VL-MoE TOR logic).
Fix: visual_pos_masks undefined error.
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
        try:
            from vggt.models.vggt import VGGT
        except ImportError:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "3DA", "my_qwen3_vggt_xattnv2"))
            from vggt.models.vggt import VGGT
        self.vggt_model = VGGT(
            img_size=cfg.img_size, 
            patch_size=cfg.patch_size, 
            embed_dim=cfg.embed_dim
        ).eval()

        if hasattr(self.vggt_model, 'camera_head'):
            del self.vggt_model.camera_head
            self.vggt_model.camera_head = None
        if hasattr(self.vggt_model, 'point_head'):
            del self.vggt_model.point_head
            self.vggt_model.point_head = None
        if hasattr(self.vggt_model, 'depth_head'):
            del self.vggt_model.depth_head
            self.vggt_model.depth_head = None
        if hasattr(self.vggt_model, 'track_head'):
            del self.vggt_model.track_head
            self.vggt_model.track_head = None

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
        if aggregator is not None:
            try:
                return next(aggregator.parameters()).dtype
            except StopIteration:
                pass
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
                raise ValueError(
                    f"image_tchw tensor must be 4D [T,C,H,W] or 5D [B,T,C,H,W], got {tuple(image_tchw.shape)}"
                )
        elif isinstance(image_tchw, list):
            image_tchw_list = image_tchw
        else:
            raise TypeError(f"image_tchw must be a Tensor or List[Tensor], got {type(image_tchw)}")

        for i, t in enumerate(image_tchw_list):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"image_tchw[{i}] must be a torch.Tensor, got {type(t)}")
            if t.dim() != 4:
                raise ValueError(f"image_tchw[{i}] must be 4D [T,C,H,W], got {tuple(t.shape)}")

        group_map = {}
        for idx, v in enumerate(image_tchw_list):
            group_map.setdefault(tuple(v.shape), []).append((idx, v))

        final_outputs: List[List[torch.Tensor]] = [None] * len(image_tchw_list)
        final_indices: List[int] = [None] * len(image_tchw_list)

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


@dataclass
class VGGTEmbeddingMergerConfig:
    input_dim: int
    output_dim: int
    spatial_embeds_layer_idx: int = -1
    vggt_patch_size: int = 14
    qwen_temporal_patch_size: int = 2
    qwen_spatial_merge_size: int = 2


class VGGTEmbeddingMerger(nn.Module):
    def __init__(self, config: VGGTEmbeddingMergerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.spatial_embeds_layer_idx = config.spatial_embeds_layer_idx
        self.vggt_patch_size = config.vggt_patch_size
        self.qwen_temporal_patch_size = config.qwen_temporal_patch_size
        self.qwen_spatial_merge_size = config.qwen_spatial_merge_size

        self.token_merge_in_dim = (
            self.input_dim * self.qwen_temporal_patch_size * (self.qwen_spatial_merge_size**2)
        )
        self.token_merge_ln_q = Qwen2RMSNorm(self.token_merge_in_dim, eps=1e-6)
        self.token_merge_mlp = nn.Sequential(
            nn.Linear(self.token_merge_in_dim, self.token_merge_in_dim),
            nn.GELU(),
            nn.Linear(self.token_merge_in_dim, self.output_dim),
        )
        num_heads = max(1, self.output_dim // 128)
        while self.output_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.cross_attn_num_heads = num_heads
        self.cross_attn_q_norm = Qwen2RMSNorm(self.output_dim, eps=1e-6)
        self.cross_attn_kv_norm = Qwen2RMSNorm(self.output_dim, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.output_dim,
            num_heads=self.cross_attn_num_heads,
            batch_first=True,
        )

    def _infer_vggt_hw(self, image_tchw: torch.Tensor) -> Tuple[int, int]:
        h, w = int(image_tchw.shape[-2]), int(image_tchw.shape[-1])
        if h % self.vggt_patch_size != 0 or w % self.vggt_patch_size != 0:
            raise ValueError(f"VGGT input H,W must be multiple of {self.vggt_patch_size}, got {h}x{w}")
        return h // self.vggt_patch_size, w // self.vggt_patch_size

    def _interp_one_image(
        self,
        vggt_frames: torch.Tensor,
        *,
        h_v: int,
        w_v: int,
        h_q: int,
        w_q: int,
    ) -> torch.Tensor:
        t, p_v, dd = vggt_frames.shape
        if p_v != h_v * w_v:
            raise ValueError(f"VGGT P mismatch: P={p_v}, H_v*W_v={h_v*w_v}")
        x = vggt_frames.view(t, h_v, w_v, dd).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(h_q, w_q), mode="bilinear", align_corners=False)
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
    ) -> torch.Tensor:
        all_spatial_tokens: List[torch.Tensor] = []
        sample_token_counts: List[int] = []
        grid_idx = 0

        for b_idx, spatial_layers in enumerate(spatial_embeds_list):
            spatial_embeds = spatial_layers[self.spatial_embeds_layer_idx]
            if spatial_embeds.dim() != 3:
                raise ValueError(f"Unexpected VGGT token dim: {spatial_embeds.shape}")

            ps = int(patch_start_idx[b_idx])
            if ps <= 1 or ps > spatial_embeds.shape[1]:
                raise ValueError(
                    f"Invalid patch_start_idx={ps} for VGGT tokens with P_total={spatial_embeds.shape[1]}"
                )

            camera_tokens = spatial_embeds[:, :1, :]
            register_tokens = spatial_embeds[:, 1:ps, :]
            patch_tokens = spatial_embeds[:, ps:, :]
            del camera_tokens, register_tokens

            h_v, w_v = self._infer_vggt_hw(image_tchw[b_idx])
            sample_spatial_per_image: List[torch.Tensor] = []

            t_total = int(image_tchw[b_idx].shape[0])
            if t_total % self.qwen_temporal_patch_size != 0:
                raise ValueError(f"image_tchw.T must be multiple of {self.qwen_temporal_patch_size}, got {t_total}")
            num_images = t_total // self.qwen_temporal_patch_size

            if patch_tokens.shape[0] != t_total:
                raise ValueError(f"VGGT temporal S mismatch: got {patch_tokens.shape[0]}, expect {t_total}")

            for img_i in range(num_images):
                if grid_idx >= len(image_grid_thw):
                    raise ValueError("Not enough image_grid_thw rows for batch image_tchw")
                t_q, h_q, w_q = [int(x) for x in image_grid_thw[grid_idx].tolist()]
                if t_q != 1:
                    raise ValueError(f"Unsupported image grid_t={t_q} for images (expect 1).")

                vggt_frames = patch_tokens[
                    img_i * self.qwen_temporal_patch_size : (img_i + 1) * self.qwen_temporal_patch_size
                ]
                vggt_frames = self._interp_one_image(vggt_frames, h_v=h_v, w_v=w_v, h_q=h_q, w_q=w_q)
                sample_spatial_per_image.append(vggt_frames)
                grid_idx += 1

            hq0 = sample_spatial_per_image[0].shape[1]
            for x in sample_spatial_per_image[1:]:
                if x.shape[1] != hq0:
                    raise ValueError("Per-sample images have different Qwen3 patch grids; not supported yet.")
            spatial_q = torch.cat(sample_spatial_per_image, dim=0)

            s, p_q, dd = spatial_q.shape
            h_q = int(image_grid_thw[grid_idx - num_images][1].item())
            w_q = int(image_grid_thw[grid_idx - num_images][2].item())
            if p_q != h_q * w_q:
                raise ValueError(f"Qwen3 P mismatch after interp: {p_q} vs {h_q*w_q}")

            spatial_q = spatial_q.view(num_images, self.qwen_temporal_patch_size, h_q, w_q, dd)
            spatial_q = spatial_q.permute(0, 2, 3, 4, 1).contiguous()

            ms = self.qwen_spatial_merge_size
            if h_q % ms != 0 or w_q % ms != 0:
                raise ValueError(f"Qwen3 grid H/W must be divisible by merge_size={ms}, got {h_q}x{w_q}")
            spatial_q = spatial_q.view(num_images, h_q // ms, ms, w_q // ms, ms, dd, self.qwen_temporal_patch_size)
            spatial_q = spatial_q.permute(0, 1, 3, 2, 4, 5, 6).contiguous()
            spatial_q = spatial_q.view(num_images * (h_q // ms) * (w_q // ms), ms * ms * dd * self.qwen_temporal_patch_size)

            all_spatial_tokens.append(spatial_q)
            sample_token_counts.append(int(spatial_q.shape[0]))

        spatial_tokens = torch.cat(all_spatial_tokens, dim=0)
        spatial_tokens = spatial_tokens.to(device=device, dtype=dtype)
        spatial_tokens = self.token_merge_ln_q(spatial_tokens)
        projected_3d_tokens = self.token_merge_mlp(spatial_tokens)
        projected_3d_tokens = projected_3d_tokens.to(device=image_embeds.device, dtype=image_embeds.dtype)

        if projected_3d_tokens.shape[0] != image_embeds.shape[0]:
            raise ValueError(
                f"Token count mismatch before xattn: 3D={projected_3d_tokens.shape[0]} vs 2D={image_embeds.shape[0]}"
            )

        q_chunks = list(torch.split(image_embeds, sample_token_counts, dim=0))
        kv_chunks = list(torch.split(projected_3d_tokens, sample_token_counts, dim=0))
        xattn_out_chunks: List[torch.Tensor] = []
        for q_chunk, kv_chunk in zip(q_chunks, kv_chunks):
            q = self.cross_attn_q_norm(q_chunk).unsqueeze(0)
            kv = self.cross_attn_kv_norm(kv_chunk).unsqueeze(0)
            xattn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
            xattn_out_chunks.append(xattn_out.squeeze(0))

        xattn_out = torch.cat(xattn_out_chunks, dim=0).to(device=image_embeds.device, dtype=image_embeds.dtype)
        return image_embeds + xattn_out


# ==============================================================================
# Core: VGGT + TOR fusion model (with TOR interface + fix for undefined variable)
# ==============================================================================
class Qwen3VLMoeVGGTForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):
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

        connector_cfg = getattr(config, "vggt_connector_config", None) or {}
        spatial_embeds_layer_idx = connector_cfg.get(
            "spatial_embeds_layer_idx", self.vggt_spatial_config.spatial_embeds_layer_idx
        )
        merger_config = VGGTEmbeddingMergerConfig(
            input_dim=2 * self.vggt_spatial_config.embed_dim,
            output_dim=config.text_config.hidden_size,
            spatial_embeds_layer_idx=spatial_embeds_layer_idx,
            vggt_patch_size=self.vggt_spatial_config.patch_size,
            qwen_temporal_patch_size=config.vision_config.temporal_patch_size,
            qwen_spatial_merge_size=config.vision_config.spatial_merge_size,
        )
        self.connector = VGGTEmbeddingMerger(merger_config)
        self.post_init()

    @property
    def vggt_embedding_merger(self) -> VGGTEmbeddingMerger:
        return self.connector

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if any(k.startswith("vggt_embedding_merger.") for k in state_dict.keys()):
            state_dict = dict(state_dict)
            for key in list(state_dict.keys()):
                if not key.startswith("vggt_embedding_merger."):
                    continue
                new_key = "connector." + key[len("vggt_embedding_merger.") :]
                state_dict.setdefault(new_key, state_dict[key])
                state_dict.pop(key)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
        
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_image_tchw(image_tchw: Optional[Union[torch.Tensor, List[torch.Tensor]]]) -> List[torch.Tensor]:
        if image_tchw is None:
            return []
        if isinstance(image_tchw, torch.Tensor):
            if image_tchw.dim() == 4:
                return [image_tchw]
            if image_tchw.dim() == 5:
                return [image_tchw[i] for i in range(image_tchw.shape[0])]
            raise ValueError(f"image_tchw must be 4D [T,C,H,W] or 5D [B,T,C,H,W], got {tuple(image_tchw.shape)}")
        if isinstance(image_tchw, list):
            return image_tchw
        raise TypeError(f"image_tchw must be a Tensor or List[Tensor], got {type(image_tchw)}")

    @staticmethod
    def _expand_video_grid_thw(video_grid_thw: torch.LongTensor) -> torch.LongTensor:
        if video_grid_thw is None:
            raise ValueError("video_grid_thw is None")
        expanded = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        expanded = expanded.clone()
        expanded[:, 0] = 1
        return expanded

    def _get_qwen3_image_embeds(
        self,
        *,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        image_embeds_list, deepstack_image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        if not isinstance(deepstack_image_embeds, list):
            deepstack_image_embeds = [deepstack_image_embeds]
        return image_embeds, deepstack_image_embeds

    def _get_qwen3_video_embeds(
        self,
        *,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        video_embeds_list, deepstack_video_embeds = self.model.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        if not isinstance(deepstack_video_embeds, list):
            deepstack_video_embeds = [deepstack_video_embeds]
        return video_embeds, deepstack_video_embeds

    def _inject_fused_image_embeds(
        self,
        *,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_embeds: torch.Tensor,
        deepstack_image_embeds: List[torch.Tensor],
        image_grid_thw: torch.LongTensor,
        image_tchw_list: List[torch.Tensor],
    ) -> Tuple[torch.FloatTensor, torch.Tensor, List[torch.Tensor]]:
        spatial_embeds_list, patch_start_idx = self.spatial_encoder(image_tchw_list)
        fused_image_embeds = self.connector(
            image_embeds,
            spatial_embeds_list=spatial_embeds_list,
            patch_start_idx=patch_start_idx,
            image_grid_thw=image_grid_thw,
            image_tchw=image_tchw_list,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

        image_mask, _ = self.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=fused_image_embeds,
        )

        fused_image_embeds = fused_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        new_inputs_embeds = inputs_embeds.masked_scatter(image_mask, fused_image_embeds)

        deepstack_image_embeds = [emb.to(new_inputs_embeds.device, new_inputs_embeds.dtype) for emb in deepstack_image_embeds]
        return new_inputs_embeds, image_mask, deepstack_image_embeds

    def _inject_fused_video_embeds(
        self,
        *,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        video_embeds: torch.Tensor,
        deepstack_video_embeds: List[torch.Tensor],
        video_grid_thw: torch.LongTensor,
        video_tchw_list: List[torch.Tensor],
    ) -> Tuple[torch.FloatTensor, torch.Tensor, List[torch.Tensor]]:
        spatial_embeds_list, patch_start_idx = self.spatial_encoder(video_tchw_list)
        video_grid_thw_exp = self._expand_video_grid_thw(video_grid_thw)

        fused_video_embeds = self.connector(
            video_embeds,
            spatial_embeds_list=spatial_embeds_list,
            patch_start_idx=patch_start_idx,
            image_grid_thw=video_grid_thw_exp,
            image_tchw=video_tchw_list,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

        _, video_mask = self.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=fused_video_embeds,
        )
        fused_video_embeds = fused_video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        new_inputs_embeds = inputs_embeds.masked_scatter(video_mask, fused_video_embeds)

        deepstack_video_embeds = [emb.to(new_inputs_embeds.device, new_inputs_embeds.dtype) for emb in deepstack_video_embeds]
        return new_inputs_embeds, video_mask, deepstack_video_embeds

    # ------------------------------------------------------------------
    # TOR Embedding injection logic + fix for undefined variable
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
        # ===================== TOR new parameters =====================
        tor_embeds: Optional[torch.FloatTensor] = None,
        tor_token_id: int = None,
        **kwargs,
    ):
        # Backward-compatible aliases
        if image_tchw is None and "images_input" in kwargs:
            image_tchw = kwargs.pop("images_input")
        if video_tchw is None and "videos_input" in kwargs:
            video_tchw = kwargs.pop("videos_input")

        # 1. Initialize word embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # ===================== TOR embedding injection (core logic) =====================
        if tor_embeds is not None and tor_token_id is not None:
            if input_ids is None:
                raise ValueError(
                    "tor_embeds were supplied without `input_ids`; the TOR positions cannot be "
                    "located. Pass `input_ids` alongside the embeddings."
                )
            mask = input_ids == tor_token_id
            num_tokens = int(mask.sum().item())
            if num_tokens > 0 and tor_embeds.shape[1] == num_tokens:
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                tor_mask = mask_expanded.to(inputs_embeds.device)
                tor_embeds = tor_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(tor_mask, tor_embeds.squeeze(0))
            else:
                # Do not fail silently: this used to be an `if` with no `else`, so a prompt that
                # carried the wrong number of TOR tokens quietly ran as plain 3DA inference.
                print(
                    f"[EIEA] TOR embedding injection SKIPPED: found {num_tokens} position(s) "
                    f"with token id {tor_token_id}, but {int(tor_embeds.shape[1])} embedding(s) "
                    "were supplied. The two must match exactly."
                )

        # Fix: initialize visual variables early to avoid undefined error
        visual_pos_masks = None
        deepstack_visual_embeds = None

        # 2. VGGT 3D fusion logic (unchanged)
        if pixel_values is not None or pixel_values_videos is not None:
            image_mask = None
            video_mask = None
            deepstack_image_embeds = None
            deepstack_video_embeds = None

            if pixel_values is not None:
                if image_tchw is None or image_grid_thw is None:
                    raise ValueError("`image_tchw`/`image_grid_thw` must not be None")
                image_embeds, deepstack_image_embeds = self._get_qwen3_image_embeds(
                    input_ids=input_ids, inputs_embeds=inputs_embeds, pixel_values=pixel_values, image_grid_thw=image_grid_thw
                )
                image_tchw_list = self._normalize_image_tchw(image_tchw)
                inputs_embeds, image_mask, deepstack_image_embeds = self._inject_fused_image_embeds(
                    input_ids=input_ids, inputs_embeds=inputs_embeds, image_embeds=image_embeds, 
                    deepstack_image_embeds=deepstack_image_embeds, image_grid_thw=image_grid_thw, image_tchw_list=image_tchw_list
                )

            if pixel_values_videos is not None:
                if video_tchw is None or video_grid_thw is None:
                    raise ValueError("`video_tchw`/`video_grid_thw` must not be None")
                video_embeds, deepstack_video_embeds = self._get_qwen3_video_embeds(
                    pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, inputs_embeds=inputs_embeds
                )
                video_tchw_list = self._normalize_image_tchw(video_tchw)
                inputs_embeds, video_mask, deepstack_video_embeds = self._inject_fused_video_embeds(
                    input_ids=input_ids, inputs_embeds=inputs_embeds, video_embeds=video_embeds, 
                    deepstack_video_embeds=deepstack_video_embeds, video_grid_thw=video_grid_thw, video_tchw_list=video_tchw_list
                )

            # Aggregate visual masks
            if image_mask is not None and video_mask is not None:
                image_mask_2d = image_mask.to(inputs_embeds.device)[..., 0]
                video_mask_2d = video_mask.to(inputs_embeds.device)[..., 0]
                visual_pos_masks = image_mask_2d | video_mask_2d
                deepstack_visual_embeds = []
                image_mask_joint = image_mask_2d[visual_pos_masks]
                video_mask_joint = video_mask_2d[visual_pos_masks]
                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint.to(inputs_embeds.device, inputs_embeds.dtype))
            elif image_mask is not None:
                visual_pos_masks = image_mask.to(inputs_embeds.device)[..., 0]
                deepstack_visual_embeds = deepstack_image_embeds
            elif video_mask is not None:
                visual_pos_masks = video_mask.to(inputs_embeds.device)[..., 0]
                deepstack_visual_embeds = deepstack_video_embeds

        # 3. Language model forward
        if attention_mask is not None:
            if isinstance(attention_mask, dict):
                attention_mask = {k: v.to(inputs_embeds.device) if isinstance(v, torch.Tensor) else v for k, v in attention_mask.items()}
            elif isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None:
            attention_mask_tensor = attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            past_len = 0 if past_key_values is None else past_key_values.get_seq_length()
            is_prefill = (cache_position is not None and cache_position[0] == 0) or past_len == 0
            if is_prefill or self.model.rope_deltas is None:
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask_tensor
                )
                self.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None, position_ids=position_ids, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, cache_position=cache_position, visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds, **kwargs,
        )
        hidden_states = outputs[0]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLMoeCausalLMOutputWithPast(
            loss=loss, aux_loss=None, logits=logits, past_key_values=outputs.past_key_values, rope_deltas=self.model.rope_deltas
        )

    # ------------------------------------------------------------------
    # Fix: pass TOR parameters during generation
    # ------------------------------------------------------------------
    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        cache_position = model_inputs.get("cache_position", None)
        
        # Clear visual inputs during decode, preserve TOR parameters
        if cache_position is not None and cache_position[0] != 0:
            model_inputs["image_tchw"] = None
            model_inputs["video_tchw"] = None
            model_inputs["pixel_values"] = None
            model_inputs["image_grid_thw"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["video_grid_thw"] = None
        
        # Critical: pass TOR parameters
        model_inputs["tor_embeds"] = kwargs.get("tor_embeds")
        model_inputs["tor_token_id"] = kwargs.get("tor_token_id")
        return model_inputs


__all__ = [
    "VGGTSpatialEncoderConfig", "VGGTSpatialEncoder",
    "VGGTEmbeddingMergerConfig", "VGGTEmbeddingMerger",
    "Qwen3VLMoeVGGTForConditionalGeneration",
]

MLPAddConnectorInterp = VGGTEmbeddingMerger
Qwen3VLVGGTMoeForConditionalGeneration = Qwen3VLMoeVGGTForConditionalGeneration