"""Debug VGGT <-> Qwen3-VL token-grid geometric alignment under the
"no letterbox + nearest-exact" configuration.

⚠️ This script models the *alternative* configuration, NOT the defaults of the released
   main model. `my_qwen3_vggt_xattnv2` ships with `letterbox=True` (VGGT padded to a
   518x518 white square) and `interp_mode="bilinear"`; the numbers printed here
   (post-merge relative error ~0.26 token, 0% torn) were computed without the padding
   and with nearest-exact resampling, so they do NOT describe the released checkpoints.
   To reproduce them, set
       VGGT_LETTERBOX=0 VGGT_INTERP_MODE=nearest-exact
   when running the model. See `alignment_debug/VGGT_QWEN_ALIGNMENT_ANALYSIS.md`.

For each test image we:
  1. Mimic the new register's `_preprocess_image_to_vggt_tensor` to find the
     real VGGT input resolution (h_real_v, w_real_v) and patch grid (h_v, w_v).
  2. Call the Qwen3-VL image processor to find Qwen's pre-merge patch grid
     (h_q, w_q) under do_resize=False (matches how the runtime template calls).
  3. Reproduce the connector's `nearest-exact` resampling formula and compute,
     for every Qwen pre-merge patch (i_q, j_q), which VGGT token (i_v, j_v) it
     pulls from.
  4. Render an overlay on the original image (VGGT grid in red, Qwen grid in
     blue, sampling arrows in green) and dump per-image alignment statistics.

Outputs are written next to this script under ``alignment_debug/``.
"""
# UPDATE: Replace the /path/to/... placeholders below with your actual local paths.

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

# We don't want to depend on the swift project; use HF processor directly.
from transformers import AutoImageProcessor

QWEN_MODEL_DIR = "/path/to/models/Qwen3-VL-8B-Instruct"
OUT_DIR = Path(__file__).resolve().parent / "alignment_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VGGT_LONG_SIDE = 518
VGGT_PATCH = 14
QWEN_PATCH = 16  # ViT patch (pre-merge)
QWEN_MERGE = 2

TEST_IMAGES: List[Tuple[str, str]] = [
    (
        "FRONT_1600x900",
        "/path/to/xiaomi_spatialembodied/data/nuscenes/nuScenes/samples/CAM_FRONT/"
        "n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984491412460.jpg",
    ),
    (
        "FRONT_LEFT_1600x900",
        "/path/to/xiaomi_spatialembodied/data/nuscenes/nuScenes/samples/CAM_FRONT_LEFT/"
        "n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984491404844.jpg",
    ),
    (
        "BACK_1600x900",
        "/path/to/xiaomi_spatialembodied/data/nuscenes/nuScenes/samples/CAM_BACK/"
        "n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984491437525.jpg",
    ),
]


@dataclass
class AlignmentStats:
    name: str
    orig_size: Tuple[int, int]
    orig_aspect: float

    vggt_input_hw: Tuple[int, int]
    vggt_grid_hw: Tuple[int, int]
    vggt_aspect: float

    qwen_input_hw: Tuple[int, int]
    qwen_grid_hw_premerge: Tuple[int, int]
    qwen_grid_hw_postmerge: Tuple[int, int]
    qwen_aspect: float

    # Pre-merge alignment (Qwen 16-px patch level, what `_interp_one_image` operates on)
    err_pre_mean_px: float
    err_pre_p99_px: float
    err_pre_max_px: float
    qwen_patch_size_pre_px: float
    # Post-merge alignment (Qwen 32-px token level, what the LLM actually sees)
    err_post_mean_px: float
    err_post_p99_px: float
    err_post_max_px: float
    qwen_patch_size_post_px: float

    # Adjacency: per 2x2 pre-merge group, max manhattan distance between any two
    # of the 4 sampled VGGT tokens. 0 means all 4 hit the same VGGT token; 1
    # means strictly adjacent in the VGGT grid; ≥2 means torn across cells.
    adj_max_mean: float
    adj_max_max: int
    adj_pct_torn: float  # fraction of post-merge cells with adj_max ≥ 2

    def rel_pre_mean(self) -> float:
        return self.err_pre_mean_px / self.qwen_patch_size_pre_px

    def rel_pre_p99(self) -> float:
        return self.err_pre_p99_px / self.qwen_patch_size_pre_px

    def rel_post_mean(self) -> float:
        return self.err_post_mean_px / self.qwen_patch_size_post_px

    def rel_post_p99(self) -> float:
        return self.err_post_p99_px / self.qwen_patch_size_post_px


def vggt_preprocess_shape(W_orig: int, H_orig: int) -> Tuple[int, int]:
    """Mirror the new register: long side -> 518, short side rounded to 14."""
    if W_orig >= H_orig:
        new_w = VGGT_LONG_SIDE
        new_h = round(H_orig * (new_w / W_orig) / VGGT_PATCH) * VGGT_PATCH
    else:
        new_h = VGGT_LONG_SIDE
        new_w = round(W_orig * (new_h / H_orig) / VGGT_PATCH) * VGGT_PATCH
    new_h = max(int(new_h), VGGT_PATCH)
    new_w = max(int(new_w), VGGT_PATCH)
    return int(new_h), int(new_w)


def qwen_grid_for_image(processor, img: Image.Image) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Run Qwen3-VL image processor and return (input_hw, grid_hw_pre_merge).

    grid_hw_pre_merge is `image_grid_thw[0, 1:3]` — number of 16-pixel patches
    along (H, W). After the spatial merge the LLM-visible grid is half of this.

    Note: we use Qwen's default ``do_resize=True`` smart_resize. The runtime
    register passes ``do_resize=False`` but only after the image dimensions are
    already aligned to ``patch_size * merge_size = 32``; for images whose raw
    shape is not 32-aligned (e.g. nuScenes 1600x900) ``do_resize=False`` raises
    a reshape error. The smart_resize result we get here matches what the
    template ultimately produces.
    """
    out = processor(images=[img], return_tensors="pt")
    grid = out["image_grid_thw"][0]  # (t, h, w)
    t_q, h_q, w_q = [int(x) for x in grid.tolist()]
    h_qwen_input = h_q * QWEN_PATCH
    w_qwen_input = w_q * QWEN_PATCH
    return (h_qwen_input, w_qwen_input), (h_q, w_q)


def build_nearest_exact_map(h_v: int, w_v: int, h_q: int, w_q: int) -> np.ndarray:
    """For each Qwen pre-merge position (i_q, j_q), return the VGGT (i_v, j_v)
    chosen by `F.interpolate(mode='nearest-exact')`.
    """
    src = np.zeros((h_q, w_q, 2), dtype=np.int64)
    for i_q in range(h_q):
        # nearest-exact center alignment formula (matches PyTorch impl)
        sy = (i_q + 0.5) * h_v / h_q - 0.5
        i_v = int(np.clip(np.round(sy), 0, h_v - 1))
        for j_q in range(w_q):
            sx = (j_q + 0.5) * w_v / w_q - 0.5
            j_v = int(np.clip(np.round(sx), 0, w_v - 1))
            src[i_q, j_q] = (i_v, j_v)
    return src


def alignment_pixel_error(
    src_grid: np.ndarray, h_v: int, w_v: int, h_q: int, w_q: int,
    H_orig: int, W_orig: int,
) -> np.ndarray:
    """For each Qwen pre-merge token, distance (in original pixels) between the
    Qwen-token center and the picked VGGT-token center, both projected back to
    original image coordinates via normalized space.
    """
    errs = np.zeros(h_q * w_q, dtype=np.float64)
    k = 0
    for i_q in range(h_q):
        qy_n = (i_q + 0.5) / h_q
        for j_q in range(w_q):
            qx_n = (j_q + 0.5) / w_q
            i_v, j_v = src_grid[i_q, j_q]
            vy_n = (int(i_v) + 0.5) / h_v
            vx_n = (int(j_v) + 0.5) / w_v
            dy = (qy_n - vy_n) * H_orig
            dx = (qx_n - vx_n) * W_orig
            errs[k] = (dy * dy + dx * dx) ** 0.5
            k += 1
    return errs


def _render_dense(
    img: Image.Image, h_v: int, w_v: int, h_q: int, w_q: int,
    src_grid: np.ndarray, out_path: Path,
) -> None:
    """Full-density overlay (every grid line) — useful for global inspection."""
    W_orig, H_orig = img.size
    overlay = img.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)

    for i in range(h_v + 1):
        y = int(round(i * H_orig / h_v))
        draw.line([(0, y), (W_orig, y)], fill=(255, 60, 60), width=2)
    for j in range(w_v + 1):
        x = int(round(j * W_orig / w_v))
        draw.line([(x, 0), (x, H_orig)], fill=(255, 60, 60), width=2)

    for i in range(h_q + 1):
        y = int(round(i * H_orig / h_q))
        draw.line([(0, y), (W_orig, y)], fill=(80, 160, 255), width=1)
    for j in range(w_q + 1):
        x = int(round(j * W_orig / w_q))
        draw.line([(x, 0), (x, H_orig)], fill=(80, 160, 255), width=1)

    for i in range(0, h_q + 1, QWEN_MERGE):
        y = int(round(i * H_orig / h_q))
        draw.line([(0, y), (W_orig, y)], fill=(255, 220, 0), width=2)
    for j in range(0, w_q + 1, QWEN_MERGE):
        x = int(round(j * W_orig / w_q))
        draw.line([(x, 0), (x, H_orig)], fill=(255, 220, 0), width=2)

    overlay.save(out_path, quality=92)


def _render_sparse(
    img: Image.Image, h_v: int, w_v: int, h_q: int, w_q: int,
    src_grid: np.ndarray, out_path: Path,
) -> None:
    """Sparse overlay — only post-merge Qwen tokens (yellow) + VGGT grid (red),
    plus arrows showing the post-merge → VGGT center mapping for ~25 tokens.
    """
    W_orig, H_orig = img.size
    overlay = img.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)

    h_post = h_q // QWEN_MERGE
    w_post = w_q // QWEN_MERGE

    # VGGT grid (red, thick)
    for i in range(h_v + 1):
        y = int(round(i * H_orig / h_v))
        draw.line([(0, y), (W_orig, y)], fill=(255, 80, 80), width=2)
    for j in range(w_v + 1):
        x = int(round(j * W_orig / w_v))
        draw.line([(x, 0), (x, H_orig)], fill=(255, 80, 80), width=2)

    # Qwen post-merge grid (yellow)
    for i in range(h_post + 1):
        y = int(round(i * H_orig / h_post))
        draw.line([(0, y), (W_orig, y)], fill=(255, 220, 0), width=2)
    for j in range(w_post + 1):
        x = int(round(j * W_orig / w_post))
        draw.line([(x, 0), (x, H_orig)], fill=(255, 220, 0), width=2)

    # Arrows: for ~5x5 sampled post-merge cells, show centroid -> avg of 4 sampled VGGT centers.
    step_i = max(1, h_post // 5)
    step_j = max(1, w_post // 6)
    for i_post in range(step_i // 2, h_post, step_i):
        for j_post in range(step_j // 2, w_post, step_j):
            # 4 sub-cells in pre-merge grid
            sub_v = []
            for di in range(QWEN_MERGE):
                for dj in range(QWEN_MERGE):
                    i_q = i_post * QWEN_MERGE + di
                    j_q = j_post * QWEN_MERGE + dj
                    sub_v.append(src_grid[i_q, j_q])
            sub_v = np.array(sub_v)
            # Post-merge cell center (Qwen)
            qy = int(round((i_post + 0.5) * H_orig / h_post))
            qx = int(round((j_post + 0.5) * W_orig / w_post))
            # Average of the 4 sampled VGGT centers (in original-pixel space)
            vy_norm = (sub_v[:, 0] + 0.5).mean() / h_v
            vx_norm = (sub_v[:, 1] + 0.5).mean() / w_v
            vy = int(round(vy_norm * H_orig))
            vx = int(round(vx_norm * W_orig))
            # Draw the bounding box of the 4 VGGT cells in cyan
            min_iv = int(sub_v[:, 0].min())
            max_iv = int(sub_v[:, 0].max())
            min_jv = int(sub_v[:, 1].min())
            max_jv = int(sub_v[:, 1].max())
            box_y0 = int(round(min_iv * H_orig / h_v))
            box_y1 = int(round((max_iv + 1) * H_orig / h_v))
            box_x0 = int(round(min_jv * W_orig / w_v))
            box_x1 = int(round((max_jv + 1) * W_orig / w_v))
            draw.rectangle([box_x0, box_y0, box_x1, box_y1], outline=(0, 255, 255), width=3)
            # Arrow from Qwen post-merge centroid to VGGT centroid
            draw.line([(qx, qy), (vx, vy)], fill=(50, 230, 50), width=3)
            draw.ellipse([qx - 6, qy - 6, qx + 6, qy + 6], fill=(255, 220, 0), outline=(0, 0, 0))
            draw.ellipse([vx - 6, vy - 6, vx + 6, vy + 6], fill=(255, 80, 80), outline=(0, 0, 0))

    overlay.save(out_path, quality=92)


def render_overlay(
    img: Image.Image, h_v: int, w_v: int, h_q: int, w_q: int,
    src_grid: np.ndarray, out_path_dense: Path, out_path_sparse: Path,
) -> None:
    _render_dense(img, h_v, w_v, h_q, w_q, src_grid, out_path_dense)
    _render_sparse(img, h_v, w_v, h_q, w_q, src_grid, out_path_sparse)


def post_merge_alignment(
    src_grid: np.ndarray, h_v: int, w_v: int, h_q: int, w_q: int,
    H_orig: int, W_orig: int,
):
    """Compute post-merge alignment.

    The post-merge cell at (i_post, j_post) covers the 2x2 pre-merge cells
    [i_post*2 .. i_post*2+1] x [j_post*2 .. j_post*2+1]. We compare its
    Qwen-side center to the average of the 4 sampled VGGT-token centers,
    plus the manhattan adjacency span among those 4 sampled VGGT positions.
    """
    h_post = h_q // QWEN_MERGE
    w_post = w_q // QWEN_MERGE

    err_post = np.zeros(h_post * w_post, dtype=np.float64)
    adj_max = np.zeros(h_post * w_post, dtype=np.int64)
    k = 0
    for i_post in range(h_post):
        qy_n = (i_post + 0.5) / h_post
        for j_post in range(w_post):
            qx_n = (j_post + 0.5) / w_post
            sub = []
            for di in range(QWEN_MERGE):
                for dj in range(QWEN_MERGE):
                    i_q = i_post * QWEN_MERGE + di
                    j_q = j_post * QWEN_MERGE + dj
                    sub.append(src_grid[i_q, j_q])
            sub = np.array(sub, dtype=np.int64)
            vy_n = (sub[:, 0].mean() + 0.5) / h_v
            vx_n = (sub[:, 1].mean() + 0.5) / w_v
            dy = (qy_n - vy_n) * H_orig
            dx = (qx_n - vx_n) * W_orig
            err_post[k] = (dy * dy + dx * dx) ** 0.5
            adj_max[k] = max(
                int(sub[:, 0].max() - sub[:, 0].min()),
                int(sub[:, 1].max() - sub[:, 1].min()),
            )
            k += 1
    return err_post, adj_max


def run_single(name: str, path: str, processor) -> AlignmentStats:
    img = Image.open(path).convert("RGB")
    W_orig, H_orig = img.size
    orig_aspect = W_orig / H_orig

    # VGGT side
    h_real_v, w_real_v = vggt_preprocess_shape(W_orig, H_orig)
    h_v, w_v = h_real_v // VGGT_PATCH, w_real_v // VGGT_PATCH
    vggt_aspect = w_v / h_v if h_v > 0 else float("nan")

    # Qwen side
    (h_qwen_input, w_qwen_input), (h_q, w_q) = qwen_grid_for_image(processor, img)
    qwen_aspect = w_q / h_q if h_q > 0 else float("nan")

    src_grid = build_nearest_exact_map(h_v, w_v, h_q, w_q)
    errs_pre = alignment_pixel_error(src_grid, h_v, w_v, h_q, w_q, H_orig, W_orig)
    errs_post, adj_max = post_merge_alignment(src_grid, h_v, w_v, h_q, w_q, H_orig, W_orig)

    qwen_patch_size_pre_px = (16 * H_orig / h_qwen_input + 16 * W_orig / w_qwen_input) / 2
    qwen_patch_size_post_px = qwen_patch_size_pre_px * QWEN_MERGE

    out_dense = OUT_DIR / f"alignment_{name}_dense.jpg"
    out_sparse = OUT_DIR / f"alignment_{name}_sparse.jpg"
    render_overlay(img, h_v, w_v, h_q, w_q, src_grid, out_dense, out_sparse)

    return AlignmentStats(
        name=name,
        orig_size=(W_orig, H_orig),
        orig_aspect=orig_aspect,
        vggt_input_hw=(h_real_v, w_real_v),
        vggt_grid_hw=(h_v, w_v),
        vggt_aspect=vggt_aspect,
        qwen_input_hw=(h_qwen_input, w_qwen_input),
        qwen_grid_hw_premerge=(h_q, w_q),
        qwen_grid_hw_postmerge=(h_q // QWEN_MERGE, w_q // QWEN_MERGE),
        qwen_aspect=qwen_aspect,
        err_pre_mean_px=float(errs_pre.mean()),
        err_pre_p99_px=float(np.percentile(errs_pre, 99)),
        err_pre_max_px=float(errs_pre.max()),
        qwen_patch_size_pre_px=qwen_patch_size_pre_px,
        err_post_mean_px=float(errs_post.mean()),
        err_post_p99_px=float(np.percentile(errs_post, 99)),
        err_post_max_px=float(errs_post.max()),
        qwen_patch_size_post_px=qwen_patch_size_post_px,
        adj_max_mean=float(adj_max.mean()),
        adj_max_max=int(adj_max.max()),
        adj_pct_torn=float((adj_max >= 2).mean() * 100.0),
    )


def main() -> int:
    print(f"=== Loading Qwen3-VL processor from {QWEN_MODEL_DIR} ===")
    processor = AutoImageProcessor.from_pretrained(QWEN_MODEL_DIR, trust_remote_code=True)

    all_stats: List[AlignmentStats] = []
    for name, path in TEST_IMAGES:
        if not Path(path).exists():
            print(f"[skip] {name}: file not found at {path}")
            continue
        try:
            stats = run_single(name, path, processor)
        except Exception as exc:
            print(f"[error] {name}: {exc!r}")
            continue
        all_stats.append(stats)

    print()
    print("=" * 96)
    print(f"{'image':<22} | {'orig':<11} | {'VGGT in':<11} VGGT grid | {'Qwen in':<11} Qwen pre / post-merge")
    print("-" * 96)
    for s in all_stats:
        print(
            f"{s.name:<22} | {s.orig_size[0]:>4}x{s.orig_size[1]:<5} | "
            f"{s.vggt_input_hw[1]:>4}x{s.vggt_input_hw[0]:<5} {s.vggt_grid_hw[1]:>3}x{s.vggt_grid_hw[0]:<3} | "
            f"{s.qwen_input_hw[1]:>4}x{s.qwen_input_hw[0]:<5} "
            f"{s.qwen_grid_hw_premerge[1]:>3}x{s.qwen_grid_hw_premerge[0]:<3} / "
            f"{s.qwen_grid_hw_postmerge[1]:>3}x{s.qwen_grid_hw_postmerge[0]:<3}"
        )

    print()
    print("=" * 110)
    print(f"{'image':<22} | aspect (orig / VGGT / Qwen) | pre-merge err mean/p99/max (px, vs 16-px patch)")
    print("-" * 110)
    for s in all_stats:
        print(
            f"{s.name:<22} | "
            f"{s.orig_aspect:.4f} / {s.vggt_aspect:.4f} / {s.qwen_aspect:.4f}  | "
            f"{s.err_pre_mean_px:>5.2f} / {s.err_pre_p99_px:>5.2f} / {s.err_pre_max_px:>5.2f}  "
            f"({s.rel_pre_mean():.2f} / {s.rel_pre_p99():.2f})"
        )

    print()
    print("=" * 110)
    print(f"{'image':<22} | post-merge err mean/p99/max (px, vs 32-px LLM-token) | adj_max mean/max | torn%")
    print("-" * 110)
    for s in all_stats:
        print(
            f"{s.name:<22} | "
            f"{s.err_post_mean_px:>5.2f} / {s.err_post_p99_px:>5.2f} / {s.err_post_max_px:>5.2f}  "
            f"({s.rel_post_mean():.2f} / {s.rel_post_p99():.2f})  | "
            f"{s.adj_max_mean:.2f} / {s.adj_max_max} | {s.adj_pct_torn:.1f}%"
        )

    print()
    print(f"=== overlays saved to: {OUT_DIR} ===")
    print("    *_dense.jpg : full grid view (VGGT red, Qwen pre-merge light-blue, Qwen post-merge yellow)")
    print("    *_sparse.jpg: post-merge view (VGGT red, Qwen post-merge yellow,")
    print("                  cyan box = bbox of 4 sampled VGGT tokens for one Qwen post-merge cell,")
    print("                  green arrow = Qwen post-merge centroid -> sampled VGGT centroid)")
    print()

    summary = [
        {
            "name": s.name,
            "orig_size": list(s.orig_size),
            "orig_aspect": s.orig_aspect,
            "vggt_input_hw": list(s.vggt_input_hw),
            "vggt_grid_hw": list(s.vggt_grid_hw),
            "vggt_aspect": s.vggt_aspect,
            "qwen_input_hw": list(s.qwen_input_hw),
            "qwen_grid_hw_premerge": list(s.qwen_grid_hw_premerge),
            "qwen_grid_hw_postmerge": list(s.qwen_grid_hw_postmerge),
            "qwen_aspect": s.qwen_aspect,
            "err_pre_mean_px": s.err_pre_mean_px,
            "err_pre_p99_px": s.err_pre_p99_px,
            "err_pre_max_px": s.err_pre_max_px,
            "qwen_patch_size_pre_px": s.qwen_patch_size_pre_px,
            "rel_pre_mean": s.rel_pre_mean(),
            "rel_pre_p99": s.rel_pre_p99(),
            "err_post_mean_px": s.err_post_mean_px,
            "err_post_p99_px": s.err_post_p99_px,
            "err_post_max_px": s.err_post_max_px,
            "qwen_patch_size_post_px": s.qwen_patch_size_post_px,
            "rel_post_mean": s.rel_post_mean(),
            "rel_post_p99": s.rel_post_p99(),
            "adj_max_mean": s.adj_max_mean,
            "adj_max_max": s.adj_max_max,
            "adj_pct_torn": s.adj_pct_torn,
        }
        for s in all_stats
    ]
    json_path = OUT_DIR / "alignment_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"=== summary JSON: {json_path} ===")

    return 0


if __name__ == "__main__":
    sys.exit(main())
