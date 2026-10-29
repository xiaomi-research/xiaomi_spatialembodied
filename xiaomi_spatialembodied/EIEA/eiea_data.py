"""Shared data / prompt helpers for EIEA training and evaluation.

The public EIEA trainer consumes the same JSON rows as the rebuttal
proof-of-concept (``experiments/build_eiea_data.py`` and
``experiments/build_maplm_data.py``):

    {
      "question": "...", "answer": "A", "options": ["A. ...", ...],
      "images": [front_left, front, ...],       # or "image": path
      "depth": "/path/to/depth.npy",            # optional
      "boxes": [[class, x, y, z, l, w, h, yaw, vx, vy, dist], ...],
      "bev": "/path/to/bev.npy"                 # optional
    }

Everything here is dependency-light (PIL/numpy/torch); the VLM-specific
processor is passed in by the trainer/evaluator.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


VIEW_ORDER_6 = ["Front_Left", "Front", "Front_Right",
                "Back_Right", "Back", "Back_Left"]
VIEW_PROMPT_6 = ("Camera views are provided in this order: "
                 "Front Left, Front, Front Right, Back Right, Back, Back Left.")


# ---------------------------------------------------------------------------
# Rows / files
# ---------------------------------------------------------------------------
def load_rows(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = [json.loads(line) for line in open(path) if line.strip()]
    else:
        rows = json.load(open(path))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list or JSONL")
    return rows[:limit] if limit else rows


def _resolve_path(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(str(path)))


def front_image_path(row: Dict[str, Any]) -> str:
    if row.get("images"):
        return _resolve_path(row["images"][1] if len(row["images"]) > 1 else row["images"][0])
    if row.get("image"):
        return _resolve_path(row["image"])
    raise KeyError("row has neither 'images' nor 'image'")


def view_image_paths(row: Dict[str, Any], n_views: int = 1) -> List[str]:
    if row.get("images"):
        paths = [_resolve_path(p) for p in row["images"]]
    elif row.get("image"):
        paths = [_resolve_path(row["image"])]
    else:
        raise KeyError("row has neither 'images' nor 'image'")
    if n_views <= len(paths) and n_views > 1:
        return paths[:n_views]
    if n_views >= 6 and len(paths) >= 6:
        return paths[:6]
    # EIEA cue is generated from the front camera; the VLM can still see all
    # views when n_views > 1.  Fall back to the front view for short rows.
    return [paths[1] if len(paths) > 1 else paths[0]]


def load_view_images(row: Dict[str, Any], n_views: int = 1) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in view_image_paths(row, n_views)]


def load_depth(row: Dict[str, Any]) -> Optional[np.ndarray]:
    """Load a precomputed DepthPro map as uint8 ``[H,W,3]`` (or ``None``)."""
    path = row.get("depth")
    if path is None or path == "":
        return None
    path = _resolve_path(path)
    if not os.path.exists(path):
        return None
    dep = np.load(path) if path.endswith(".npy") else np.asarray(Image.open(path).convert("L"))
    return ensure_depth_uint8(dep)


def ensure_depth_uint8(dep: np.ndarray) -> np.ndarray:
    dep = np.asarray(dep)
    if dep.dtype != np.uint8:
        dep = dep.astype(np.float32)
        if dep.size and float(dep.max()) <= 1.0:
            dep = dep * 255.0
        dep = np.clip(dep, 0, 255).astype(np.uint8)
    if dep.ndim == 2:
        dep = np.stack([dep] * 3, axis=-1)
    if dep.ndim == 3 and dep.shape[-1] == 1:
        dep = np.repeat(dep, 3, axis=-1)
    if dep.ndim != 3 or dep.shape[-1] != 3:
        raise ValueError(f"depth map must be [H,W] or [H,W,3], got {dep.shape}")
    return dep


def load_bev(row: Dict[str, Any], expected_channels: Optional[int] = None) -> Optional[np.ndarray]:
    """Load a BEV raster as float32 ``[C,H,W]`` (or ``None``)."""
    path = row.get("bev")
    if path is None or path == "":
        return None
    path = _resolve_path(path)
    if not os.path.exists(path):
        return None
    if path.endswith(".npy"):
        bev = np.load(path).astype(np.float32)
    else:
        # MapLM-v2 ships a point-cloud BEV JPEG; treat it as an RGB raster.
        bev = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        bev = np.transpose(bev, (2, 0, 1))
    if bev.ndim == 2:
        bev = bev[None]
    if bev.ndim != 3:
        raise ValueError(f"BEV raster must be 3-D, got {bev.shape}")
    # Accept [H,W,C] when it is unambiguous.
    if expected_channels and bev.shape[0] != expected_channels and bev.shape[-1] == expected_channels:
        bev = np.transpose(bev, (2, 0, 1))
    if expected_channels and bev.shape[0] != expected_channels:
        raise ValueError(
            f"BEV channels {bev.shape[0]} != expected {expected_channels} "
            f"(shape {bev.shape}, path {path})")
    return np.ascontiguousarray(bev, dtype=np.float32)


def boxes_from_row(row: Dict[str, Any], feature_dim: int = 11) -> np.ndarray:
    boxes = row.get("boxes")
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, feature_dim), dtype=np.float32)
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < feature_dim:
        arr = np.pad(arr, ((0, 0), (0, feature_dim - arr.shape[1])))
    return arr[:, :feature_dim]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def question_text(row: Dict[str, Any], n_views: int = 1) -> str:
    """Question + options, with literal ``<image>`` placeholders removed.

    The Ego3D rows embed ``<image>`` markers in the question text.  Qwen's
    chat template inserts its own vision tokens for the images passed in
    ``content``; leaving the literals in place would duplicate/confuse the
    view layout, especially when EIEA trains on the front view only.
    """
    q = str(row.get("question", ""))
    q = re.sub(r"<image>", "", q)
    if n_views == 1:
        q = re.sub(r"\bThese\s+are\s+six\s+camera\s+views\b",
                   "This is one camera view", q, flags=re.I)
        q = re.sub(r"\bsix\s+camera\s+views\b", "one camera view", q, flags=re.I)
    if row.get("options"):
        q = q.rstrip() + "\nOptions:\n" + "\n".join(str(o) for o in row["options"])
    return q.strip()


def prompt_text(row: Dict[str, Any], n_views: int = 1,
                tor_str: str = "<|endoftext|>", num_tor: int = 10,
                terse: Optional[str] = None) -> str:
    parts = []
    if n_views > 1:
        parts.append(VIEW_PROMPT_6)
    parts.append(question_text(row, n_views=n_views))
    text = "\n".join(p for p in parts if p)
    if terse:
        text = text.rstrip() + terse
    return tor_str * int(num_tor) + "\n" + text


def mamba_text_ids(tokenizer, row: Dict[str, Any],
                   max_length: Optional[int] = None) -> torch.Tensor:
    """Token ids for the Mamba interpreter (question only, no TOR markers)."""
    text = question_text(row, n_views=1)
    enc = tokenizer(text, add_special_tokens=False, return_tensors="pt",
                    truncation=max_length is not None, max_length=max_length or 1024)
    return enc["input_ids"]


def cap_processor_pixels(processor, max_tokens: int = 512,
                         min_tokens: int = 4) -> None:
    """Bound the native Qwen image token budget.

    An uncapped 1600x900 camera frame expands to ~2k image tokens; EIEA
    training/evaluation uses 512 (cost setting) or 1024 (paper evaluation).
    The 16x16x4 scale matches the budget convention used by the public
    evaluation scripts.
    """
    try:
        ip = processor.image_processor
        ip.max_pixels = int(max_tokens) * 16 * 16 * 4
        ip.min_pixels = int(min_tokens) * 16 * 16 * 4
    except Exception as e:
        print(f"WARNING: could not cap image pixels: {e}")


def ensure_tor_token(processor, token: str = "<tor>") -> int:
    """Register ``<tor>`` on the 7B proxy tokenizer and return its id."""
    tok = processor.tokenizer
    added = tok.add_tokens([token], special_tokens=True)
    if added:
        # The 7B path only injects embeddings at these positions; the
        # placeholder token's own embedding is never read.  No resize needed
        # because SSR's tokenizer already has the marker in its vocabulary in
        # the reference setup, but a fresh Qwen tokenizer needs this branch.
        pass
    return tok.convert_tokens_to_ids(token)


def _content(images: Sequence[Image.Image], text: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "image", "image": im} for im in images
    ]
    content.append({"type": "text", "text": text})
    return content


def prepare_vlm_batch(
    processor,
    row: Dict[str, Any],
    device: torch.device,
    *,
    n_views: int = 1,
    tor_str: str = "<|endoftext|>",
    num_tor: int = 10,
    answer: Optional[str] = None,
    terse: Optional[str] = None,
    with_tor: bool = True,
) -> Dict[str, Any]:
    """Build one VLM training/eval batch with TOR markers and answer labels.

    Returns a dict with model kwargs (on ``device``), ``labels``, ``images``
    (PIL list for the VGGT stream), and ``tor_count``.
    """
    images = load_view_images(row, n_views=n_views)
    text = prompt_text(row, n_views=n_views,
                       tor_str=(tor_str if with_tor else ""),
                       num_tor=(num_tor if with_tor else 0), terse=terse)
    user = [{"role": "user", "content": _content(images, text)}]
    prompt_prefix = processor.apply_chat_template(
        user, tokenize=False, add_generation_prompt=True)
    full_msgs = user + ([{"role": "assistant", "content": answer}]
                        if answer is not None else [])
    prompt_full = processor.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=(answer is None))

    enc = processor(text=[prompt_full], images=images, videos=None,
                    padding=True, return_tensors="pt")
    prefix_enc = processor(text=[prompt_prefix], images=images, videos=None,
                           padding=True, return_tensors="pt")
    n_prefix = int(prefix_enc["input_ids"].shape[1])
    input_ids = enc["input_ids"]
    labels = torch.full_like(input_ids, -100)
    if answer is not None:
        if n_prefix >= input_ids.shape[1]:
            raise ValueError(
                f"assistant prefix consumed the full sequence ({n_prefix} >= "
                f"{input_ids.shape[1]}); check the chat template / answer")
        labels[:, n_prefix:] = input_ids[:, n_prefix:]

    kw = {k: v.to(device) for k, v in enc.items()
          if k in ("input_ids", "attention_mask", "pixel_values",
                   "pixel_values_videos", "image_grid_thw", "video_grid_thw")}
    kw["labels"] = labels.to(device)
    tor_id = int(processor.tokenizer.convert_tokens_to_ids(tor_str))
    tor_count = int((input_ids == tor_id).sum()) if with_tor else 0
    if with_tor and tor_count != int(num_tor):
        raise ValueError(
            f"expected {num_tor} TOR markers in the prompt, found {tor_count}; "
            f"tor_str={tor_str!r} id={tor_id}. The cue injection would be "
            "silently skipped.")
    kw["tor_token_id"] = tor_id
    kw["tor_count"] = tor_count
    kw["with_tor"] = with_tor
    kw["images"] = images
    kw["prefix_len"] = n_prefix
    return kw


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def is_numeric(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)
    return (m.group(1) if m else text).strip()


def score_answer(pred: str, gt: Any) -> Any:
    """Return 1/0 for multiple choice and an absolute error for numeric GT.

    Mirrors the rebuttal scoring used by ``evaluations/synergy_2x2.py`` and
    ``tools/paired_stats.py`` so results are comparable across scripts.
    """
    gu = str(gt).strip().upper()
    up = str(pred).strip().upper()
    if re.fullmatch(r"[A-D]", gu):
        if re.fullmatch(r"[\(\[]?[A-D][\)\].:]?", up.strip()):
            return int(up.strip("().[]: ")[0] == gu)
        m = re.search(r"(?:answer is|answer\s*[:：])\s*[\(\[]?([A-D])\b", up)
        if m:
            return int(m.group(1) == gu)
        letters = re.findall(r"(?:^|[\s:：])([A-D])[\.\):：]", up)
        return int((letters[-1] if letters else up[:1]) == gu)
    if is_numeric(gt):
        nums = re.findall(r"-?\d+(?:\.\d+)?", pred)
        return abs(float(nums[0]) - float(gt)) if nums else None
    return int(gu in up)


__all__ = [
    "VIEW_ORDER_6", "load_rows", "front_image_path", "view_image_paths",
    "load_view_images", "load_depth", "ensure_depth_uint8", "load_bev",
    "boxes_from_row", "question_text", "prompt_text", "mamba_text_ids",
    "cap_processor_pixels", "ensure_tor_token", "prepare_vlm_batch",
    "is_numeric", "extract_answer", "score_answer",
]
