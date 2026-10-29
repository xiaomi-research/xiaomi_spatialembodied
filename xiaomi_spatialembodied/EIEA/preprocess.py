"""VGGT image preprocessing without importing the MS-Swift register.

The 30B model consumes two visual streams: the native Qwen image processor and
the VGGT spatial encoder.  The latter expects a letterboxed square tensor for
every image, repeated ``temporal_patch_size`` times.  The Swift register has
this helper inline; pulling it in would import Swift (and its optional
dependencies), so training/evaluation use this small standalone copy.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _to_pil_rgb(x: Union[str, Image.Image]) -> Image.Image:
    img = Image.open(x) if isinstance(x, str) else x
    if not isinstance(img, Image.Image):
        raise TypeError(f"unsupported image type: {type(x)}")
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def _to_tensor(img: Image.Image) -> torch.Tensor:
    try:
        from torchvision.transforms.functional import to_tensor
        return to_tensor(img)
    except Exception:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def preprocess_image_to_vggt_tensor(
    x: Union[str, Image.Image],
    target_size: int = 518,
    patch_size: int = 14,
    fill: float = 1.0,
) -> torch.Tensor:
    """Letterbox one image to ``target_size`` square, keeping aspect ratio.

    The released weights used white (``fill=1.0``) letterboxing; changing this
    invalidates the trained 3DA connector.
    """
    img = _to_pil_rgb(x)
    w, h = img.size
    if w >= h:
        new_w = target_size
        new_h = max(patch_size, round(h * (new_w / w) / patch_size) * patch_size)
    else:
        new_h = target_size
        new_w = max(patch_size, round(w * (new_h / h) / patch_size) * patch_size)
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    ten = _to_tensor(img)
    pad_h = target_size - ten.shape[1]
    pad_w = target_size - ten.shape[2]
    if pad_h > 0 or pad_w > 0:
        top, left = pad_h // 2, pad_w // 2
        ten = F.pad(ten, (left, pad_w - left, top, pad_h - top),
                    mode="constant", value=fill)
    return ten.contiguous()


def make_image_tchw(
    images: Sequence[Union[str, Image.Image]],
    temporal_patch_size: int = 2,
    target_size: int = 518,
    patch_size: int = 14,
) -> torch.Tensor:
    """Build the ``[N*tps, 3, S, S]`` tensor expected by ``image_tchw``."""
    if len(images) == 0:
        raise ValueError("images is empty")
    nchw = torch.stack([
        preprocess_image_to_vggt_tensor(
            im, target_size=target_size, patch_size=patch_size)
        for im in images
    ])
    ntchw = nchw.unsqueeze(1).repeat(1, int(temporal_patch_size), 1, 1, 1)
    return ntchw.reshape(-1, *ntchw.shape[2:]).contiguous()


__all__ = ["preprocess_image_to_vggt_tensor", "make_image_tchw"]
