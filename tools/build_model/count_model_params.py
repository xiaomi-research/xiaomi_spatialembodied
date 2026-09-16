#!/usr/bin/env python3
# Copyright (c) ms-swift custom example.
"""
Count parameters for MS-Swift *registered* Qwen3-VL + VGGT models (method-1 / method-2).

Workflow:
  1) import the `plugin/*_register.py` (same as --custom_register_path in training)
  2) load the checkpoint with `swift.model.register.get_model_processor`
  3) print total / trainable / per-name-prefix stats

Default load uses `device_map="meta"` so large checkpoints do not allocate real memory.

Usage:
  conda activate vla4d
  export PYTHONPATH="/path/to/ms-swift:$PYTHONPATH"  # repository root (contains the `swift` package)

  python count_registered_model_params.py \
    --model /path/to/hf-style-checkpoint \
    --register_path /path/to/.../plugin/qwen3_vl_moe_vggt_register.py \
    --model_type qwen3_vl_vggt_fuse2d_all

  # Optional: finer breakdown (deeper name prefixes)
  python count_registered_model_params.py ... --depth 3 --per-tensor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

# Ensure ms-swift is importable when run as `python path/to/this_script.py`
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MS_SWIFT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
_SWIFT_SRC = os.path.join(_MS_SWIFT_ROOT, "swift")
if os.path.isdir(_SWIFT_SRC) and _MS_SWIFT_ROOT not in sys.path:
    sys.path.insert(0, _MS_SWIFT_ROOT)

def _import_register_file(path: str) -> None:
    from swift.utils.utils import import_external_file

    import_external_file(path)

@dataclass
class ParamStats:
    name: str
    numel: int
    numel_trainable: int
    is_buffer: bool

def _iter_param_stats(model: nn.Module) -> List[ParamStats]:
    out: List[ParamStats] = []
    for name, t in model.named_parameters():
        n = t.numel()
        nt = n if t.requires_grad else 0
        out.append(ParamStats(name=name, numel=n, numel_trainable=nt, is_buffer=False))
    for name, t in model.named_buffers():
        out.append(ParamStats(name=name, numel=t.numel(), numel_trainable=0, is_buffer=True))
    return out

def _prefix_key(name: str, depth: int) -> str:
    parts = name.split(".")
    if depth <= 0 or len(parts) <= depth:
        return name
    return ".".join(parts[:depth])

def _group_by_prefix(
    stats: List[ParamStats],
    depth: int,
    *,
    include_buffers: bool,
) -> "OrderedDict[str, Tuple[int, int]]":
    """
    Returns ordered mapping prefix -> (total_numel, trainable_numel).
    Buffers are included in first column when include_buffers, with trainable=0.
    """
    m: "OrderedDict[str, Tuple[int, int]]" = OrderedDict()
    for s in stats:
        if s.is_buffer and not include_buffers:
            continue
        key = _prefix_key(s.name, depth)
        tot, tr = m.get(key, (0, 0))
        if s.is_buffer:
            m[key] = (tot + s.numel, tr)
        else:
            m[key] = (tot + s.numel, tr + s.numel_trainable)
    return m

def _fmt(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.4f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.4f}M"
    if n >= 1_000:
        return f"{n / 1_000:.4f}K"
    return str(n)

def _load_model(
    model_path: str,
    *,
    model_type: Optional[str],
    device_map: str,
) -> nn.Module:
    from swift.model.register import get_model_processor

    if device_map in ("none", "None", ""):
        device_map = None

    # download_model False: use local directory as-is
    model, _processor = get_model_processor(
        model_path,
        model_type=model_type,
        load_model=True,
        download_model=False,
        model_kwargs={"device_map": device_map, "low_cpu_mem_usage": True},
    )
    if model is None:
        raise RuntimeError("get_model_processor returned model=None")
    return model

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Count parameters for a Swift-registered VL+VGGT checkpoint "
        "given a register file (custom_register_path) and model_type.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="HuggingFace-style directory (config + weights) or local finetune output path.",
    )
    p.add_argument(
        "--register_path",
        required=True,
        help="Path to plugin/qwen3_vl_moe_vggt_register.py (or equivalent register).",
    )
    p.add_argument(
        "--model_type",
        default=None,
        help="Must match the register_model(...) id (e.g. qwen3_vl_vggt_fuse2d_all). "
        "If omitted, Swift tries to infer (may fail for custom local dirs).",
    )
    p.add_argument(
        "--device_map",
        default="meta",
        help="Transformers `device_map` for loading. Default 'meta' counts params without real weight RAM. "
        "If loading fails, retry with e.g. 'cpu' (needs enough RAM) or a single GPU id.",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Group stats by the first D dot-separated name segments (default: 2).",
    )
    p.add_argument(
        "--per-tensor",
        action="store_true",
        help="Also print one line per parameter/buffer (can be very long).",
    )
    p.add_argument(
        "--include-buffers",
        action="store_true",
        help="Count buffers in totals and in grouped prefix output.",
    )
    p.add_argument(
        "--json",
        default=None,
        help="If set, write a JSON file with total / grouped / optional per-tensor data.",
    )
    args = p.parse_args(argv)

    if not os.path.isdir(args.model):
        raise SystemExit(f"--model is not a directory: {args.model}")
    if not os.path.isfile(args.register_path):
        raise SystemExit(f"--register_path is not a file: {args.register_path}")

    _import_register_file(args.register_path)
    try:
        model = _load_model(
            args.model,
            model_type=args.model_type,
            device_map=args.device_map,
        )
    except Exception as e:  # noqa: BLE001
        if args.device_map == "meta":
            print(
                f"[warn] device_map=meta load failed: {e}\n"
                "       Retrying with device_map=cpu (needs enough host RAM) ...",
                file=sys.stderr,
            )
            model = _load_model(
                args.model,
                model_type=args.model_type,
                device_map="cpu",
            )
        else:
            raise
    if not isinstance(model, nn.Module):
        raise SystemExit(f"Expected nn.Module, got {type(model)}")

    stats = _iter_param_stats(model)
    param_only = [s for s in stats if not s.is_buffer]
    buffer_only = [s for s in stats if s.is_buffer]

    p_total = sum(s.numel for s in param_only)
    p_train = sum(s.numel_trainable for s in param_only)
    b_total = sum(s.numel for s in buffer_only)

    print(f"model_path:    {os.path.abspath(args.model)}")
    print(f"register_path: {os.path.abspath(args.register_path)}")
    resolved_type = getattr(getattr(model, "model_info", None), "model_type", None)
    print(
        f"model_type:    {args.model_type!r} (if None, Swift infers; resolved={resolved_type!r})",
    )
    print(f"device_map:    {args.device_map}")
    print()
    print("=== Summary ===")
    print(f"Params total:      {p_total}  ({_fmt(p_total)})")
    print(f"Params trainable:  {p_train}  ({_fmt(p_train)})")
    print(f"Params frozen:     {p_total - p_train}  ({_fmt(p_total - p_train)})")
    if args.include_buffers or buffer_only:
        print(f"Buffers (total):  {b_total}  ({_fmt(b_total)})  [non-trainable]")

    grouped = _group_by_prefix(
        stats,
        depth=max(1, int(args.depth)),
        include_buffers=bool(args.include_buffers),
    )
    print()
    print(f"=== By name prefix (depth={max(1, int(args.depth))}) ===")
    print(f"{'prefix':<72} {'all':>18} {'trainable':>18}")
    print("-" * 110)
    for k, (t, tr) in sorted(grouped.items(), key=lambda x: -x[1][0]):
        print(f"{k:<72} {t:>18,} {tr:>18,}")
    if args.include_buffers:
        print("\n(Notice) In grouped rows, the first column includes buffers; trainable is parameters only.")

    if args.per_tensor:
        print()
        print("=== Per tensor ===")
        print(f"{'name':<90} {'numel':>16} {'trainable':>10} {'kind':>8}")
        print("-" * 128)
        for s in sorted(stats, key=lambda x: -x.numel):
            tr = s.numel_trainable if not s.is_buffer else 0
            kind = "buffer" if s.is_buffer else "param"
            if not args.include_buffers and s.is_buffer:
                continue
            mark = "yes" if tr > 0 else "no"
            print(f"{s.name:<90} {s.numel:>16,} {mark:>10} {kind:>8}")

    if args.json:
        payload: Dict[str, Any] = {
            "model_path": os.path.abspath(args.model),
            "register_path": os.path.abspath(args.register_path),
            "model_type_arg": args.model_type,
            "device_map": args.device_map,
            "summary": {
                "params_total": p_total,
                "params_trainable": p_train,
                "params_frozen": p_total - p_train,
                "buffers_total": b_total,
            },
            "grouped_by_prefix_depth": {
                "depth": max(1, int(args.depth)),
                "include_buffers": bool(args.include_buffers),
                "groups": {k: {"all": t, "trainable": tr} for k, (t, tr) in grouped.items()},
            },
        }
        if args.per_tensor:
            payload["per_tensor"] = [
                {
                    "name": s.name,
                    "numel": s.numel,
                    "numel_trainable": s.numel_trainable,
                    "is_buffer": s.is_buffer,
                }
                for s in stats
                if args.include_buffers or not s.is_buffer
            ]
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print()
        print(f"Wrote JSON: {os.path.abspath(args.json)}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
