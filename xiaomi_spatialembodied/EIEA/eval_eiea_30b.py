#!/usr/bin/env python3
"""Evaluate a trained EIEA cue encoder on frozen Qwen3-VL-30B-A3B.

The evaluation mirrors the rebuttal stress test: regenerate the K cue tokens
for every condition while keeping the VLM, prompt and decoding fixed, then
report multiple-choice accuracy / numeric MAE and the paired change relative
to clean cues.  Conditions include whole-modality removal, partial cues,
box drop/jitter, BEV mask/noise, and depth noise/mask, plus ``no_cue`` (no
injection at all) as the "is the cue being used?" control.

Example:
    python evaluation/eval_eiea_30b.py \
        --model_dir /dev/shm/models/xembodied-v0 \
        --ckpt results/eiea_box/eiea_30b_box_bev_depth.pt \
        --data results/eiea_box/val_bev.json \
        --limit 100 --out results/eiea_box/eval_eiea_30b.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

EIEA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EIEA_DIR.parent
if str(EIEA_DIR) not in sys.path:
    sys.path.insert(0, str(EIEA_DIR))

from corruption import EVAL_CONDITIONS, apply_eval_condition
from cue_encoder import EIEACueEncoder
from preprocess import make_image_tchw
from eiea_data import (
    boxes_from_row, cap_processor_pixels, extract_answer, is_numeric, load_bev,
    load_depth, load_rows, load_view_images, mamba_text_ids,
    prepare_vlm_batch, score_answer,
)

DEFAULT_MODEL_DIR = os.environ.get("XEMB_MODEL_DIR", "/dev/shm/models/xembodied-v0")
DEFAULT_TOR_STR = "<|endoftext|>"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/eiea_box/eval_eiea_30b.json")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--n_views", type=int, default=1)
    ap.add_argument("--max_image_tokens", type=int, default=1024)
    ap.add_argument("--bev_channels", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu_layers", type=int, default=22)
    ap.add_argument("--tor_str", default=DEFAULT_TOR_STR)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help=f"default: {EVAL_CONDITIONS + ['wrong_scene', 'no_cue']}")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def mc_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    mc = [r for r in records if not is_numeric(r["gt"])]
    num = [r for r in records if is_numeric(r["gt"]) and isinstance(r["score"], float)]
    return {
        "n": len(records),
        "mc_n": len(mc),
        "mc_acc": (sum(r["score"] for r in mc) / len(mc)) if mc else None,
        "num_n": len(num),
        "num_mae": (sum(abs(r["score"]) for r in num) / len(num)) if num else None,
    }


def paired_delta(records: Dict[str, List[Dict[str, Any]]],
                 cond: str, ref: str = "clean") -> Dict[str, Any]:
    """Paired MC delta vs the clean condition (same qids)."""
    if ref not in records or cond not in records:
        return {}
    a = {r["qid"]: r for r in records[ref]}
    b = {r["qid"]: r for r in records[cond]}
    qids = [q for q in a if q in b and not is_numeric(a[q]["gt"])]
    if not qids:
        return {}
    da = [int(a[q]["score"]) for q in qids]
    db = [int(b[q]["score"]) for q in qids]
    diff = 100.0 * (sum(db) - sum(da)) / len(qids)
    # normal-approximation 95% CI on the paired difference
    deltas = np.asarray(db, dtype=float) - np.asarray(da, dtype=float)
    se = float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
    changed = sum(1 for q in qids if a[q].get("pred") != b[q].get("pred"))
    return {"mc_n": len(qids), "delta_pts": diff,
            "ci95": [diff - 1.96 * 100 * se, diff + 1.96 * 100 * se],
            "changed": changed}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
    torch.cuda.set_device(torch.device(args.device))
    random.seed(args.seed)
    np.random.seed(args.seed)

    from ssr_eiea_common import load_midi, load_visual_encoders
    from train_eiea_30b import load_eiea_vlm

    rows = load_rows(args.data, limit=args.limit)
    conditions = args.conditions or (EVAL_CONDITIONS + ["wrong_scene", "no_cue"])
    print(f"{len(rows)} rows; conditions={conditions}", flush=True)

    model, proc, tps, d_vlm = load_eiea_vlm(args.model_dir, args.gpu_layers)
    model.eval()
    model.config.use_cache = True
    cap_processor_pixels(proc, args.max_image_tokens)

    midi, mtok = load_midi(args.device)
    encoder, ck = EIEACueEncoder.from_checkpoint(
        args.ckpt, midi.mamba, map_location="cpu")
    encoder = encoder.to(args.device).eval()
    clip_proc, clip, sig_proc, sig = load_visual_encoders(args.device)
    if encoder.cfg.d_vlm != d_vlm:
        raise ValueError(
            f"checkpoint d_vlm={encoder.cfg.d_vlm} does not match model d_vlm={d_vlm}")

    rng = random.Random(args.seed)

    def cue_for(row, boxes, bev, depth):
        img = load_view_images(row, 1)[0]
        with torch.no_grad():
            ie = clip(**clip_proc(images=img, return_tensors="pt").to(args.device)).last_hidden_state.float()
            de = None
            if depth is not None:
                de = sig(**sig_proc(images=depth, return_tensors="pt").to(args.device)).last_hidden_state.float()
            ids = mamba_text_ids(mtok, row).to(args.device)
            bt = torch.from_numpy(boxes).to(args.device)
            bv = None if bev is None else torch.from_numpy(bev).to(args.device)
            c = encoder(ie, de, bt, bv, None, ids)
        return c.to(torch.bfloat16)

    def generate(row, cue, with_tor: bool):
        kw = prepare_vlm_batch(
            proc, row, torch.device(args.device), n_views=args.n_views,
            tor_str=args.tor_str, num_tor=encoder.cfg.num_tor,
            answer=None, with_tor=with_tor)
        tchw = make_image_tchw(kw["images"], tps).to(args.device)
        model_kwargs = {k: v for k, v in kw.items()
                        if k in ("input_ids", "attention_mask", "pixel_values",
                                 "image_grid_thw")}
        gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                      stop_strings=["</answer>"], tokenizer=proc.tokenizer)
        with torch.no_grad():
            if cue is not None and with_tor:
                out = model.generate(**model_kwargs, image_tchw=[tchw],
                                     tor_embeds=cue,
                                     tor_token_id=kw["tor_token_id"], **gen_kw)
            else:
                out = model.generate(**model_kwargs, image_tchw=[tchw], **gen_kw)
        raw = proc.tokenizer.decode(out[0][kw["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
        return extract_answer(raw), raw

    results: Dict[str, List[Dict[str, Any]]] = {c: [] for c in conditions}
    for i, row in enumerate(rows):
        boxes = boxes_from_row(row)
        bev = load_bev(row, args.bev_channels)
        depth = load_depth(row)
        # donor for the cross-scene condition
        donor = rows[(i + 1) % len(rows)]
        for cond in conditions:
            if cond == "no_cue":
                pred, raw = generate(row, None, with_tor=False)
            elif cond == "wrong_scene":
                c = cue_for(row, boxes_from_row(donor),
                            load_bev(donor, args.bev_channels),
                            load_depth(donor))
                pred, raw = generate(row, c, with_tor=True)
            else:
                corr = apply_eval_condition(
                    cond, boxes, bev, depth,
                    rng=random.Random(args.seed * 100003 + i))
                c = cue_for(row, corr.boxes, corr.bev, corr.depth)
                pred, raw = generate(row, c, with_tor=True)
            results[cond].append({
                "qid": str(row.get("qid", i)),
                "gt": row.get("answer"),
                "pred": pred,
                "raw": raw,
                "score": score_answer(pred, row.get("answer")),
            })
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)

    summary = {c: mc_summary(recs) for c, recs in results.items()}
    deltas = {c: paired_delta(results, c) for c in conditions if c != "clean"}
    clean_pred = {r["qid"]: r["pred"] for r in results.get("clean", [])}
    nocue_pred = {r["qid"]: r["pred"] for r in results.get("no_cue", [])}
    cue_used = any(clean_pred.get(q) != nocue_pred.get(q) for q in clean_pred)
    if not cue_used:
        print("WARNING: clean and no_cue produced identical predictions on all "
              "questions; the cue may not be injected. Check TOR count / shapes.",
              flush=True)

    out = {"summary": summary, "deltas_vs_clean": deltas,
           "cue_changes_output": cue_used,
           "checkpoint": args.ckpt, "data": args.data,
           "predictions": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    with open(os.path.splitext(args.out)[0] + ".jsonl", "w") as f:
        for cond, recs in results.items():
            for r in recs:
                f.write(json.dumps({"condition": cond, "qid": r["qid"],
                                    "gt": r["gt"], "pred": r["pred"],
                                    "score": r["score"]}) + "\n")
    print(json.dumps({"summary": summary, "deltas": deltas,
                      "cue_changes_output": cue_used}, indent=1))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
