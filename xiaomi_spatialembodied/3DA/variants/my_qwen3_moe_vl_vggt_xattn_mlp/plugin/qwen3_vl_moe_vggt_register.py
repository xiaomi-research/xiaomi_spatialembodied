"""
MS-Swift custom registration file for Qwen3-VL-MoE-30B + VGGT xattn-MLP deepstack.

This file is meant to be passed via `--custom_register_path`.
It registers:
- a template that prepares `image_tchw` for VGGT
- a model type that points to the FULL fused checkpoint directory
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
from PIL import Image

from swift.model.model_arch import ModelArch
from swift.model.model_meta import Model, ModelGroup, ModelMeta
from swift.model.models.qwen import Qwen3VLLoader
from swift.model.register import ModelLoader as SwiftModelLoader, register_model
from swift.template.base import Template
from swift.template.register import register_template
from swift.template.template_inputs import StdTemplateInputs
from swift.template.templates.qwen import Qwen3VLTemplate, QwenTemplateMeta
from swift.template.utils import findall

# ==============================================================================
# Paths
# ==============================================================================
_PLUGIN_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _PLUGIN_DIR.parent  # .../my_qwen3_moe_vl_vggt_xattn_mlp
_DEFAULT_VGGT_REPO_DIR = str(Path(__file__).resolve().parents[3] / "vggt")

if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


# ==============================================================================
# VGGT env knobs
# ==============================================================================
@dataclass(frozen=True)
class _VGGTEnv:
    repo_dir: str
    device: str
    dtype: str


def _get_vggt_env() -> _VGGTEnv:
    return _VGGTEnv(
        repo_dir=os.getenv("VGGT_REPO_DIR", _DEFAULT_VGGT_REPO_DIR),
        device=os.getenv("VGGT_DEVICE", "cpu"),
        dtype=os.getenv("VGGT_DTYPE", "float32"),
    )


# ==============================================================================
# Image helpers
# ==============================================================================
def _to_pil_rgb(x: Union[str, Image.Image]) -> Image.Image:
    """Normalize input into an RGB PIL.Image."""
    if isinstance(x, str):
        img = Image.open(x)
    elif isinstance(x, Image.Image):
        # Always use the PIL object that was handed to us, rather than re-opening
        # `x.filename`. On modern Pillow `convert()` and `resize()` clear `filename`, so
        # re-opening is a no-op on the MS-Swift path; it only fires for callers that hand us
        # an image still carrying a filename, and there it would silently feed the VGGT
        # stream the on-disk pixels instead of the in-memory ones the Qwen stream sees.
        img = x
    else:
        raise TypeError(f"Unsupported image type for VGGT preprocessing: {type(x)}")
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert("RGB")


def _preprocess_image_to_vggt_tensor(x: Union[str, Image.Image]) -> torch.Tensor:
    """Aspect-ratio resize to long-side 518, short-side rounded to multiple of 14."""
    from torchvision import transforms as TF

    target_size = 518
    patch_size = 14
    to_tensor = TF.ToTensor()

    img = _to_pil_rgb(x)
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / patch_size) * patch_size
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / patch_size) * patch_size

    new_height = max(int(new_height), patch_size)
    new_width = max(int(new_width), patch_size)

    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img = to_tensor(img)
    return img.contiguous()


def _make_image_tchw(
    images: List[Union[str, Image.Image]],
    *,
    temporal_patch_size: int,
) -> torch.Tensor:
    if not images:
        raise ValueError("images is empty for VGGT preprocessing")

    images_nchw: List[torch.Tensor] = []
    for x in images:
        images_nchw.append(_preprocess_image_to_vggt_tensor(x))

    images_nchw = torch.stack(images_nchw, dim=0)
    images_ntchw = images_nchw.unsqueeze(1).repeat(1, int(temporal_patch_size), 1, 1, 1)
    return images_ntchw.reshape(-1, images_ntchw.shape[2], images_ntchw.shape[3], images_ntchw.shape[4]).contiguous()


def _grid_hw_all_equal(media_grid_thw: torch.Tensor) -> bool:
    if media_grid_thw is None or int(media_grid_thw.shape[0]) <= 1:
        return True
    h0, w0 = int(media_grid_thw[0, 1].item()), int(media_grid_thw[0, 2].item())
    for i in range(1, int(media_grid_thw.shape[0])):
        h_i, w_i = int(media_grid_thw[i, 1].item()), int(media_grid_thw[i, 2].item())
        if h_i != h0 or w_i != w0:
            return False
    return True


def _select_target_grid_hw(media_grid_thw: torch.Tensor) -> Tuple[int, int]:
    counts: Dict[Tuple[int, int], int] = {}
    order: List[Tuple[int, int]] = []
    for i in range(int(media_grid_thw.shape[0])):
        key = (int(media_grid_thw[i, 1].item()), int(media_grid_thw[i, 2].item()))
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1
    best = order[0]
    best_count = counts[best]
    for key in order[1:]:
        cur = counts[key]
        if cur > best_count:
            best = key
            best_count = cur
    return best


def _resize_image_to_target_grid(
    x: Union[str, Image.Image],
    *,
    target_h: int,
    target_w: int,
    patch_size: int,
) -> Image.Image:
    img = _to_pil_rgb(x)
    target_h_px = int(target_h) * int(patch_size)
    target_w_px = int(target_w) * int(patch_size)
    if img.height == target_h_px and img.width == target_w_px:
        return img

    scale = min(target_w_px / float(img.width), target_h_px / float(img.height))
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (target_w_px, target_h_px), (255, 255, 255))
    left = (target_w_px - new_w) // 2
    top = (target_h_px - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _align_images_to_same_grid_if_needed(
    *,
    images: List[Union[str, Image.Image]],
    image_processor: Any,
) -> Tuple[List[Union[str, Image.Image]], Dict[str, Any], torch.Tensor]:
    # `do_resize=False` makes the image processor reshape raw pixels into patches, which
    # requires both sides to be exact multiples of patch_size * merge_size (32 for Qwen3-VL).
    # Otherwise it fails deep inside the processor with an opaque
    # "shape ... invalid for input of size ..." error. Check up front and say what to fix.
    patch_size = int(getattr(image_processor, "patch_size", 16))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    grid_unit = patch_size * merge_size
    misaligned = []
    for i, x in enumerate(images):
        if isinstance(x, str):
            with Image.open(x) as fh:
                width, height = fh.size
        else:
            width, height = x.size
        if height % grid_unit or width % grid_unit:
            misaligned.append(f"[{i}] {width}x{height}")
    if misaligned:
        raise ValueError(
            "VGGT stream images must have both sides divisible by "
            f"patch_size*merge_size={grid_unit}, got: {', '.join(misaligned)}. "
            "MS-Swift's `fetch_image` normally guarantees this; when building inputs by "
            "hand, resize with `qwen_vl_utils.smart_resize` first."
        )

    media_inputs = image_processor(images=images, return_tensors="pt", do_resize=False)
    media_grid_thw = media_inputs["image_grid_thw"]
    if _grid_hw_all_equal(media_grid_thw):
        return images, media_inputs, media_grid_thw

    target_h, target_w = _select_target_grid_hw(media_grid_thw)
    aligned_images: List[Union[str, Image.Image]] = []
    for i, x in enumerate(images):
        h_i = int(media_grid_thw[i, 1].item())
        w_i = int(media_grid_thw[i, 2].item())
        if h_i == target_h and w_i == target_w:
            aligned_images.append(x)
            continue
        aligned_images.append(
            _resize_image_to_target_grid(
                x,
                target_h=target_h,
                target_w=target_w,
                patch_size=patch_size,
            )
        )

    aligned_media_inputs = image_processor(images=aligned_images, return_tensors="pt", do_resize=False)
    aligned_grid_thw = aligned_media_inputs["image_grid_thw"]
    if not _grid_hw_all_equal(aligned_grid_thw):
        raise ValueError("Failed to align per-sample multi-image grid_thw to a consistent target grid.")
    return aligned_images, aligned_media_inputs, aligned_grid_thw


def _decode_video_frames(
    video: Any,
    *,
    num_frames: int,
) -> List[Image.Image]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")

    if isinstance(video, torch.Tensor):
        x = video.detach().cpu()
        if x.ndim != 4:
            raise ValueError(f"Video tensor must be 4D (T,H,W,C) or (T,C,H,W), got shape {tuple(x.shape)}")
        if x.shape[-1] == 3:
            thwc = x
        elif x.shape[1] == 3:
            thwc = x.permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError(f"Cannot infer channel dim for video frames: shape {tuple(x.shape)}")
        if thwc.dtype != torch.uint8:
            if thwc.is_floating_point():
                thwc = (thwc * 255.0).clamp(0, 255)
            thwc = thwc.to(torch.uint8)
        n = int(thwc.shape[0])
        if n <= 0:
            raise ValueError("Video tensor has no frames.")
        idx = torch.linspace(0, n - 1, steps=num_frames).round().to(torch.int64)
        return [Image.fromarray(thwc[int(i)].numpy()).convert("RGB") for i in idx.tolist()]

    if isinstance(video, list) and all(isinstance(x, Image.Image) for x in video):
        if len(video) == 0:
            raise ValueError("Empty frame list for video input.")
        if len(video) == num_frames:
            return [x.convert("RGB") for x in video]
        idx = torch.linspace(0, len(video) - 1, steps=num_frames).round().to(torch.int64)
        return [video[int(i)].convert("RGB") for i in idx.tolist()]

    if isinstance(video, str):
        from decord import VideoReader, cpu

        vr = VideoReader(video, ctx=cpu(0))
        n = len(vr)
        if n <= 0:
            raise ValueError(f"Video has no frames: {video}")
        idx = torch.linspace(0, n - 1, steps=num_frames).round().to(torch.int64).tolist()
        frames = vr.get_batch(idx).asnumpy()
        return [Image.fromarray(frames[i]).convert("RGB") for i in range(frames.shape[0])]

    raise TypeError(f"Unsupported video input type for VGGT preprocessing: {type(video)}")


def _make_video_tchw(
    videos: List[Any],
    *,
    video_grid_thw: torch.LongTensor,
    temporal_patch_size: int,
) -> torch.Tensor:
    if not videos:
        raise ValueError("videos is empty for VGGT preprocessing")
    if video_grid_thw is None:
        raise ValueError("video_grid_thw is required to build video_tchw")
    if int(video_grid_thw.shape[0]) != len(videos):
        raise ValueError(f"video_grid_thw rows ({int(video_grid_thw.shape[0])}) != num videos ({len(videos)})")

    frames_nchw: List[torch.Tensor] = []
    for i, v in enumerate(videos):
        grid_t = int(video_grid_thw[i, 0].item())
        decoded = _decode_video_frames(v, num_frames=int(grid_t))
        for fr in decoded:
            frames_nchw.append(_preprocess_image_to_vggt_tensor(fr))

    return torch.stack(frames_nchw, dim=0).contiguous()


# ==============================================================================
# Template
# ==============================================================================
class Qwen3VLVGGTTemplate(Qwen3VLTemplate):
    version = "v3"
    images_input = None
    videos_input = None

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        processor = self.processor
        input_ids = encoded["input_ids"]
        labels = encoded["labels"]
        loss_scale = encoded.get("loss_scale", None)

        for media_type in ["images", "videos"]:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == "images":
                    media_token = self.image_token_id
                    aligned_images, media_inputs, media_grid_thw = _align_images_to_same_grid_if_needed(
                        images=list(mm_data),
                        image_processor=processor.image_processor,
                    )
                    self.images_input = aligned_images
                    image_tchw = _make_image_tchw(
                        aligned_images,
                        temporal_patch_size=int(self.config.vision_config.temporal_patch_size),
                    )
                    encoded["image_tchw"] = [image_tchw]
                else:
                    split_token = self._tokenize("\n")[0]
                    media_inputs = processor(
                        text=["\n".join(["<|vision_start|><|video_pad|><|vision_end|>"] * len(mm_data))],
                        videos=mm_data,
                        return_tensors="pt",
                        do_resize=False,
                        **inputs.mm_processor_kwargs,
                    )
                    splited_tokens = self._split_list(media_inputs["input_ids"][0].tolist(), split_token)
                    media_grid_thw = media_inputs["video_grid_thw"]
                    self.videos_input = mm_data
                    media_inputs.pop("input_ids", None)
                    media_inputs.pop("attention_mask", None)
                    media_token = self.video_token_id

                    video_tchw = _make_video_tchw(
                        mm_data,
                        video_grid_thw=media_grid_thw,
                        temporal_patch_size=int(self.config.vision_config.temporal_patch_size),
                    )
                    encoded["video_tchw"] = [video_tchw]

                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    if media_type == "images":
                        token_len = media_grid_thw[i].prod() // merge_length
                        return [media_token] * int(token_len)
                    return splited_tokens[i]

                input_ids, labels, loss_scale = self._extend_tokens(
                    input_ids, labels, loss_scale, idx_list, _get_new_tokens
                )
                encoded.update(media_inputs)

        encoded["input_ids"] = input_ids
        encoded["labels"] = labels
        encoded["loss_scale"] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        image_tchw = self.gather_list(batch, "image_tchw")
        if image_tchw:
            res["image_tchw"] = image_tchw
        video_tchw = self.gather_list(batch, "video_tchw")
        if video_tchw:
            res["video_tchw"] = video_tchw
        return res

    def generate(self, model, *args, **kwargs):
        if kwargs.get("image_tchw") is None and self.images_input:
            kwargs["image_tchw"] = [
                _make_image_tchw(
                    self.images_input,
                    temporal_patch_size=int(self.config.vision_config.temporal_patch_size),
                )
            ]
        if kwargs.get("video_tchw") is None and self.videos_input:
            video_grid_thw = kwargs.get("video_grid_thw", None)
            if video_grid_thw is None:
                media_inputs = self.processor(
                    text=["\n".join(["<|vision_start|><|video_pad|><|vision_end|>"] * len(self.videos_input))],
                    videos=self.videos_input,
                    return_tensors="pt",
                    do_resize=False,
                )
                video_grid_thw = media_inputs["video_grid_thw"]
            kwargs["video_tchw"] = [
                _make_video_tchw(
                    self.videos_input,
                    video_grid_thw=video_grid_thw,
                    temporal_patch_size=int(self.config.vision_config.temporal_patch_size),
                )
            ]
        return super().generate(model, *args, **kwargs)


# ==============================================================================
# Loader
# ==============================================================================
class Qwen3VLMoeVGGTLoader(Qwen3VLLoader):
    def _check_qwen_vl_utils(self):
        from swift.model.models.qwen import compat_qwen_vl_utils
        from transformers.utils.versions import require_version

        require_version("qwen_vl_utils>=0.0.14")
        compat_qwen_vl_utils(image_patch_size=16)

    def get_model(self, model_dir: str, config, processor, model_kwargs):
        from modeling_qwen3_vl_moe_vggt import Qwen3VLMoeVGGTForConditionalGeneration

        self.auto_model_cls = Qwen3VLMoeVGGTForConditionalGeneration

        world_size = int(os.getenv("WORLD_SIZE", "1") or "1")
        if world_size > 1:
            model_kwargs = dict(model_kwargs or {})
            model_kwargs.pop("device_map", None)
            model_kwargs.pop("max_memory", None)

        env = _get_vggt_env()
        config.vggt_repo_dir = env.repo_dir if "VGGT_REPO_DIR" in os.environ else (getattr(config, "vggt_repo_dir", None) or env.repo_dir)
        config.vggt_device = env.device if "VGGT_DEVICE" in os.environ else (getattr(config, "vggt_device", None) or env.device)
        config.vggt_dtype = env.dtype if "VGGT_DTYPE" in os.environ else (getattr(config, "vggt_dtype", None) or env.dtype)
        config.vggt_spatial_config = getattr(config, "vggt_spatial_config", None) or {
            "img_size": 518,
            "patch_size": 14,
            "embed_dim": 1024,
            "spatial_embeds_layer_idx": -1,
        }
        config.vggt_connector_config = getattr(config, "vggt_connector_config", None) or {
            "spatial_embeds_layer_idx": -1,
        }

        model = SwiftModelLoader.get_model(self, model_dir, config, processor, model_kwargs)

        from swift.model.patcher import patch_get_input_embeddings
        base = model.model if hasattr(model, 'model') else model
        if hasattr(base, 'visual') and not hasattr(base.visual, '_original_get_input_embeddings'):
            patch_get_input_embeddings(base.visual, 'patch_embed')

        return model


# ==============================================================================
# Register
# ==============================================================================
register_template(QwenTemplateMeta("qwen3_vl_moe_vggt_xattn_mlp", template_cls=Qwen3VLVGGTTemplate, default_system=None))

register_model(
    ModelMeta(
        "qwen3_vl_moe_vggt_xattn_mlp",
        [
            ModelGroup([
                Model(
                    model_path="/path/to/models/Qwen3-VL-30B-A3B-Instruct",
                    hf_model_id="Qwen/Qwen3-30B-A3B-Instruct",
                )
            ]),
        ],
        Qwen3VLMoeVGGTLoader,
        template="qwen3_vl_moe_vggt_xattn_mlp",
        model_arch=ModelArch.qwen3_vl,
        architectures=["Qwen3VLMoeVGGTForConditionalGeneration"],
        is_multimodal=True,
        requires=["transformers>=4.57", "qwen_vl_utils>=0.0.14", "decord"],
        tags=["vision", "video"],
    )
)


__all__ = [
    "Qwen3VLVGGTTemplate",
    "Qwen3VLMoeVGGTLoader",
]

# ── Optional: type-grouped batch sampler for ZeRO-3 ──
if os.environ.get("SWIFT_GROUP_BY_DATA_TYPE", "").strip() in ("1", "true", "True"):
    from type_grouped_sampler import patch_dataloader_mixin
    patch_dataloader_mixin()
