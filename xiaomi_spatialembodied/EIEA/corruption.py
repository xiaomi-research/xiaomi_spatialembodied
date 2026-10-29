"""Training-time cue corruption for EIEA robustness.

The paper's EIEA is trained on curated physical cues, but at deployment the
detector, BEV map and depth source can be missing, partial or noisy.  The
rebuttal (W1 / LSwS) asks for evidence that EIEA degrades gracefully and still
exploits whatever cues remain.  This module implements the two interventions
used for that training recipe:

* **random masking** -- drop a whole modality, drop a fraction of boxes, or
  mask blocks of the BEV/depth raster;
* **noise** -- jitter box geometry/velocity, add Gaussian noise to BEV or
  depth, or flatten depth.

Corruption is applied to the *raw* cue tensors, before the modality encoders,
so the encoder sees the same kind of failure it will see at inference time.
Everything is numpy-based and deterministic given a ``random.Random`` instance,
which keeps training reproducible and lets evaluation replay fixed conditions.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CorruptionConfig:
    """Probabilities / magnitudes for cue corruption.

    Defaults are the "standard" preset used for EIEA training.  Set all
    probabilities to zero for a clean run.
    """

    # whole-modality dropout
    p_drop_box: float = 0.15
    p_drop_bev: float = 0.15
    p_drop_depth: float = 0.15
    p_drop_all: float = 0.02
    # partial-cue curriculum: keep exactly one of box/BEV/depth
    p_keep_one: float = 0.10

    # box corruption
    p_box_element_drop: float = 0.15
    box_element_drop_ratio: float = 0.50
    p_box_jitter: float = 0.20
    box_xy_noise_m: float = 1.0
    box_z_noise_m: float = 0.5
    box_size_noise_ratio: float = 0.10
    box_yaw_noise_rad: float = 0.10
    box_vel_noise_mps: float = 0.50

    # BEV corruption
    p_bev_noise: float = 0.20
    bev_noise_std: float = 0.10          # fraction of the per-channel range
    p_bev_mask: float = 0.20
    bev_mask_ratio: float = 0.50
    p_bev_channel_drop: float = 0.05

    # depth corruption
    p_depth_noise: float = 0.20
    depth_noise_std: float = 0.10        # fraction of the depth range
    p_depth_mask: float = 0.15
    depth_mask_ratio: float = 0.50
    p_depth_flat: float = 0.05

    # raster patch size used for block masking
    patch_size: int = 8

    # ------------------------------------------------------------------
    @classmethod
    def preset(cls, name: str) -> "CorruptionConfig":
        name = (name or "standard").lower()
        if name in ("none", "clean", "off"):
            return cls(
                p_drop_box=0.0, p_drop_bev=0.0, p_drop_depth=0.0,
                p_drop_all=0.0, p_keep_one=0.0,
                p_box_element_drop=0.0, p_box_jitter=0.0,
                p_bev_noise=0.0, p_bev_mask=0.0, p_bev_channel_drop=0.0,
                p_depth_noise=0.0, p_depth_mask=0.0, p_depth_flat=0.0,
            )
        if name == "light":
            return cls(
                p_drop_box=0.08, p_drop_bev=0.08, p_drop_depth=0.08,
                p_drop_all=0.0, p_keep_one=0.03,
                p_box_element_drop=0.08, box_element_drop_ratio=0.30,
                p_box_jitter=0.10, box_xy_noise_m=0.5,
                box_yaw_noise_rad=0.05, box_vel_noise_mps=0.25,
                p_bev_noise=0.10, bev_noise_std=0.05,
                p_bev_mask=0.10, bev_mask_ratio=0.30, p_bev_channel_drop=0.0,
                p_depth_noise=0.10, depth_noise_std=0.05,
                p_depth_mask=0.08, depth_mask_ratio=0.30, p_depth_flat=0.02,
            )
        if name == "heavy":
            return cls(
                p_drop_box=0.25, p_drop_bev=0.25, p_drop_depth=0.25,
                p_drop_all=0.05, p_keep_one=0.20,
                p_box_element_drop=0.25, box_element_drop_ratio=0.60,
                p_box_jitter=0.30, box_xy_noise_m=1.5, box_z_noise_m=0.75,
                box_size_noise_ratio=0.20, box_yaw_noise_rad=0.20,
                box_vel_noise_mps=1.0,
                p_bev_noise=0.30, bev_noise_std=0.20,
                p_bev_mask=0.30, bev_mask_ratio=0.60, p_bev_channel_drop=0.10,
                p_depth_noise=0.30, depth_noise_std=0.20,
                p_depth_mask=0.25, depth_mask_ratio=0.60, p_depth_flat=0.10,
            )
        if name != "standard":
            raise ValueError(
                f"unknown corruption preset {name!r}; "
                "choose none/light/standard/heavy")
        return cls()

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CorruptedCues:
    """Output of :func:`apply_corruption`."""

    boxes: np.ndarray
    bev: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    meta: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.meta:
            return "clean"
        return ",".join(f"{k}={v}" for k, v in sorted(self.meta.items()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_rng(rng) -> random.Random:
    if isinstance(rng, random.Random):
        return rng
    return random.Random(rng)


def _as_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _empty_boxes(boxes: np.ndarray) -> np.ndarray:
    n_features = boxes.shape[1] if boxes is not None and boxes.ndim == 2 else 11
    return np.zeros((0, n_features), dtype=np.float32)


def _box_span(boxes: np.ndarray) -> float:
    if boxes is None or len(boxes) == 0:
        return 1.0
    return max(float(np.abs(boxes[:, 1:4]).max()), 1.0)


def corrupt_boxes(boxes: np.ndarray, cfg: CorruptionConfig,
                  rng: random.Random, meta: Dict) -> np.ndarray:
    """Apply element drop + geometric jitter.  Returns a new array."""
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    n = boxes.shape[0]
    if n == 0:
        return boxes

    if cfg.p_box_element_drop > 0 and rng.random() < cfg.p_box_element_drop:
        ratio = min(max(cfg.box_element_drop_ratio, 0.0), 1.0)
        keep = np.array([rng.random() >= ratio for _ in range(n)], dtype=bool)
        if not keep.any():
            keep[rng.randrange(n)] = True
        boxes = boxes[keep]
        meta["box_element_drop"] = f"{int((~keep).sum())}/{n}"
        n = boxes.shape[0]

    if cfg.p_box_jitter > 0 and rng.random() < cfg.p_box_jitter:
        # columns: class, x,y,z, l,w,h, yaw, vx,vy, dist
        boxes[:, 1] += rng.gauss(0.0, cfg.box_xy_noise_m)
        boxes[:, 2] += rng.gauss(0.0, cfg.box_xy_noise_m)
        boxes[:, 3] += rng.gauss(0.0, cfg.box_z_noise_m)
        for col in (4, 5, 6):  # l, w, h
            boxes[:, col] *= max(0.1, 1.0 + rng.gauss(0.0, cfg.box_size_noise_ratio))
            boxes[:, col] = np.maximum(boxes[:, col], 0.05)
        boxes[:, 7] = (boxes[:, 7] + rng.gauss(0.0, cfg.box_yaw_noise_rad) + np.pi) % (2 * np.pi) - np.pi
        boxes[:, 8] += rng.gauss(0.0, cfg.box_vel_noise_mps)
        boxes[:, 9] += rng.gauss(0.0, cfg.box_vel_noise_mps)
        # keep the distance feature consistent with the jittered centre
        boxes[:, 10] = np.sqrt(boxes[:, 1] ** 2 + boxes[:, 2] ** 2)
        meta["box_jitter"] = f"{cfg.box_xy_noise_m}m"
    return boxes


def corrupt_bev(bev: np.ndarray, cfg: CorruptionConfig,
                rng: random.Random, meta: Dict) -> np.ndarray:
    """Block-mask / Gaussian-noise / channel-drop a BEV raster."""
    bev = np.asarray(bev, dtype=np.float32).copy()
    if bev.ndim != 3:
        raise ValueError(f"BEV raster must be [C,H,W], got {bev.shape}")

    if cfg.p_bev_noise > 0 and rng.random() < cfg.p_bev_noise:
        # Per-channel range keeps map probabilities and metric channels usable.
        for c in range(bev.shape[0]):
            span = float(bev[c].max() - bev[c].min())
            if span <= 0:
                span = 1.0
            bev[c] += np.random.default_rng(rng.randrange(2**31)).normal(
                0.0, cfg.bev_noise_std * span, bev[c].shape).astype(np.float32)
        meta["bev_noise"] = f"std={cfg.bev_noise_std}"

    if cfg.p_bev_mask > 0 and rng.random() < cfg.p_bev_mask:
        h, w = bev.shape[-2:]
        p = cfg.patch_size
        ih, iw = max(1, h // p), max(1, w // p)
        blocks = [(i, j) for i in range(ih) for j in range(iw)]
        n_mask = int(round(cfg.bev_mask_ratio * len(blocks)))
        for i, j in rng.sample(blocks, min(n_mask, len(blocks))):
            bev[:, i * p:(i + 1) * p, j * p:(j + 1) * p] = 0.0
        meta["bev_mask"] = f"{cfg.bev_mask_ratio:.0%}"

    if cfg.p_bev_channel_drop > 0 and rng.random() < cfg.p_bev_channel_drop:
        c = rng.randrange(bev.shape[0])
        bev[c] = 0.0
        meta["bev_channel_drop"] = c
    return bev


def corrupt_depth(depth: np.ndarray, cfg: CorruptionConfig,
                  rng: random.Random, meta: Dict) -> np.ndarray:
    """Gaussian-noise / block-mask / flatten a depth map (uint8 or float)."""
    depth = np.asarray(depth).copy()
    is_uint8 = depth.dtype == np.uint8
    work = depth.astype(np.float32)

    span = float(work.max() - work.min()) if work.size else 1.0
    if span <= 0:
        span = 1.0

    if cfg.p_depth_noise > 0 and rng.random() < cfg.p_depth_noise:
        noise = np.random.default_rng(rng.randrange(2**31)).normal(
            0.0, cfg.depth_noise_std * span, work.shape).astype(np.float32)
        work = work + noise
        meta["depth_noise"] = f"std={cfg.depth_noise_std}"

    if cfg.p_depth_mask > 0 and rng.random() < cfg.p_depth_mask:
        h, w = work.shape[:2]
        p = cfg.patch_size
        ih, iw = max(1, h // p), max(1, w // p)
        blocks = [(i, j) for i in range(ih) for j in range(iw)]
        n_mask = int(round(cfg.depth_mask_ratio * len(blocks)))
        fill = float(work.mean())
        for i, j in rng.sample(blocks, min(n_mask, len(blocks))):
            work[i * p:(i + 1) * p, j * p:(j + 1) * p] = fill
        meta["depth_mask"] = f"{cfg.depth_mask_ratio:.0%}"

    if cfg.p_depth_flat > 0 and rng.random() < cfg.p_depth_flat:
        work[...] = float(work.mean())
        meta["depth_flat"] = True

    if is_uint8:
        return np.clip(work, 0, 255).astype(np.uint8)
    return work


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def apply_corruption(
    boxes=None,
    bev=None,
    depth=None,
    cfg: Optional[CorruptionConfig] = None,
    rng=None,
) -> CorruptedCues:
    """Randomly mask / noise the three cue sources of one training sample.

    Args:
        boxes: ``[N, 11]`` array (or tensor).
        bev: ``[C, H, W]`` array (or tensor), or ``None``.
        depth: ``[H, W]`` / ``[H, W, 3]`` array (or tensor), or ``None``.
        cfg: corruption config (defaults to the standard preset).
        rng: ``random.Random`` or integer seed.

    Returns:
        :class:`CorruptedCues` with numpy arrays (``None`` for dropped rasters)
        and a ``meta`` dict recording which interventions fired.
    """
    cfg = cfg or CorruptionConfig()
    rng = _as_rng(rng)
    meta: Dict[str, object] = {}

    boxes = _empty_boxes(None) if boxes is None else _as_numpy(boxes).astype(np.float32)
    if boxes.ndim != 2:
        boxes = _empty_boxes(boxes)
    bev = _as_numpy(bev)
    depth = _as_numpy(depth)

    # 1. whole-modality dropout -------------------------------------------
    if cfg.p_drop_all > 0 and rng.random() < cfg.p_drop_all:
        meta["drop_all"] = True
        return CorruptedCues(_empty_boxes(boxes), None, None, meta)
    if boxes is not None and boxes.shape[0] and cfg.p_drop_box > 0 and rng.random() < cfg.p_drop_box:
        boxes = _empty_boxes(boxes)
        meta["drop_box"] = True
    if bev is not None and cfg.p_drop_bev > 0 and rng.random() < cfg.p_drop_bev:
        bev = None
        meta["drop_bev"] = True
    if depth is not None and cfg.p_drop_depth > 0 and rng.random() < cfg.p_drop_depth:
        depth = None
        meta["drop_depth"] = True

    # 2. partial-cue curriculum: keep exactly one modality ----------------
    if cfg.p_keep_one > 0 and rng.random() < cfg.p_keep_one:
        present = [name for name, value in
                   (("box", boxes if boxes is not None and boxes.shape[0] else None),
                    ("bev", bev), ("depth", depth)) if value is not None]
        if present:
            keep = rng.choice(present)
            if keep != "box":
                boxes = _empty_boxes(boxes)
            if keep != "bev":
                bev = None
            if keep != "depth":
                depth = None
            meta["keep_only"] = keep

    # 3. per-modality corruption ------------------------------------------
    if boxes is not None and boxes.shape[0]:
        boxes = corrupt_boxes(boxes, cfg, rng, meta)
    if bev is not None:
        bev = corrupt_bev(bev, cfg, rng, meta)
    if depth is not None:
        depth = corrupt_depth(depth, cfg, rng, meta)

    return CorruptedCues(boxes=boxes, bev=bev, depth=depth, meta=meta)


# ---------------------------------------------------------------------------
# Fixed evaluation conditions
# ---------------------------------------------------------------------------
EVAL_CONDITIONS = [
    "clean",
    "no_box", "drop50_box", "jitter_box",
    "no_bev", "bev_mask50", "bev_noise",
    "no_depth", "depth_noise", "depth_mask",
    "no_physical",
    "box_only", "bev_only", "depth_only",
]


def apply_eval_condition(
    name: str,
    boxes=None,
    bev=None,
    depth=None,
    rng=None,
) -> CorruptedCues:
    """Deterministic stress-test condition used by the rebuttal evaluation.

    ``wrong_scene`` is handled by the caller (it needs a donor sample); every
    other condition is constructed here.
    """
    rng = _as_rng(rng)
    boxes = _empty_boxes(None) if boxes is None else _as_numpy(boxes).astype(np.float32)
    if boxes.ndim != 2:
        boxes = _empty_boxes(boxes)
    bev = _as_numpy(bev)
    depth = _as_numpy(depth)
    meta: Dict[str, object] = {"condition": name}

    def empty():
        return CorruptedCues(_empty_boxes(boxes), None, None, dict(meta))

    if name == "clean":
        return CorruptedCues(boxes.copy(), None if bev is None else bev.copy(),
                             None if depth is None else depth.copy(), meta)
    if name == "no_box":
        return CorruptedCues(_empty_boxes(boxes), None if bev is None else bev.copy(),
                             None if depth is None else depth.copy(), meta)
    if name == "drop50_box":
        cfg = CorruptionConfig(p_box_element_drop=1.0, box_element_drop_ratio=0.5)
        return CorruptedCues(corrupt_boxes(boxes, cfg, rng, meta),
                             None if bev is None else bev.copy(),
                             None if depth is None else depth.copy(), meta)
    if name == "jitter_box":
        cfg = CorruptionConfig(p_box_jitter=1.0, box_xy_noise_m=1.0,
                               box_z_noise_m=0.5, box_size_noise_ratio=0.0,
                               box_yaw_noise_rad=0.0, box_vel_noise_mps=0.0)
        return CorruptedCues(corrupt_boxes(boxes, cfg, rng, meta),
                             None if bev is None else bev.copy(),
                             None if depth is None else depth.copy(), meta)
    if name == "no_bev":
        return CorruptedCues(boxes.copy(), None,
                             None if depth is None else depth.copy(), meta)
    if name == "bev_mask50":
        cfg = CorruptionConfig(p_bev_mask=1.0, bev_mask_ratio=0.5,
                               p_bev_noise=0.0, p_bev_channel_drop=0.0)
        return CorruptedCues(boxes.copy(), corrupt_bev(bev, cfg, rng, meta) if bev is not None else None,
                             None if depth is None else depth.copy(), meta)
    if name == "bev_noise":
        cfg = CorruptionConfig(p_bev_noise=1.0, bev_noise_std=0.10,
                               p_bev_mask=0.0, p_bev_channel_drop=0.0)
        return CorruptedCues(boxes.copy(), corrupt_bev(bev, cfg, rng, meta) if bev is not None else None,
                             None if depth is None else depth.copy(), meta)
    if name == "no_depth":
        return CorruptedCues(boxes.copy(), None if bev is None else bev.copy(), None, meta)
    if name == "depth_noise":
        cfg = CorruptionConfig(p_depth_noise=1.0, depth_noise_std=0.10,
                               p_depth_mask=0.0, p_depth_flat=0.0)
        return CorruptedCues(boxes.copy(), None if bev is None else bev.copy(),
                             corrupt_depth(depth, cfg, rng, meta) if depth is not None else None, meta)
    if name == "depth_mask":
        cfg = CorruptionConfig(p_depth_mask=1.0, depth_mask_ratio=0.5,
                               p_depth_noise=0.0, p_depth_flat=0.0)
        return CorruptedCues(boxes.copy(), None if bev is None else bev.copy(),
                             corrupt_depth(depth, cfg, rng, meta) if depth is not None else None, meta)
    if name == "no_physical":
        return empty()
    if name == "box_only":
        return CorruptedCues(boxes.copy(), None, None, meta)
    if name == "bev_only":
        return CorruptedCues(_empty_boxes(boxes), None if bev is None else bev.copy(), None, meta)
    if name == "depth_only":
        return CorruptedCues(_empty_boxes(boxes), None,
                             None if depth is None else depth.copy(), meta)
    raise ValueError(f"unknown eval condition {name!r}; choose from {EVAL_CONDITIONS}")


__all__ = [
    "CorruptionConfig", "CorruptedCues", "apply_corruption",
    "apply_eval_condition", "EVAL_CONDITIONS",
    "corrupt_boxes", "corrupt_bev", "corrupt_depth",
]
