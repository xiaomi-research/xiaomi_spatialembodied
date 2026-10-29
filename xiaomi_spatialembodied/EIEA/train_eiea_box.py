#!/usr/bin/env python3
"""Train EIEA-lite on the 7B SSR proxy (box + BEV map + depth).

This is the runnable counterpart of :mod:`training.train_eiea_30b`: the same
cue encoder, corruption recipe and answer-CE objective, but the frozen
backbone is SSR-VLM (Qwen2.5-VL-7B + SSR LoRA) and the cue width is 3584.
It is intended for ablation / robustness checks on 2x4090-class hardware.

Corruption is applied to the raw cue sources every step: whole-modality
dropout, partial-cue dropping (keep only one of box/BEV/depth), box element
drop / jitter, BEV block mask / noise, and depth noise / mask / flatten.

Example:
    python training/train_eiea_box.py \
        --data results/eiea_box/train_bev.json \
        --save results/eiea_box/eiea_7b_box_bev_depth.pt \
        --steps 800 --corruption standard
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

EIEA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EIEA_DIR.parent
if str(EIEA_DIR) not in sys.path:
    sys.path.insert(0, str(EIEA_DIR))

from corruption import CorruptionConfig, apply_corruption
from cue_encoder import EIEAConfig, EIEACueEncoder
from eiea_data import (
    boxes_from_row, cap_processor_pixels, load_bev, load_depth, load_rows,
    load_view_images, mamba_text_ids, prepare_vlm_batch,
)

DEFAULT_TOR_STR = "<|endoftext|>"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_midi", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bev_channels", type=int, default=5)
    ap.add_argument("--num_tor", type=int, default=10)
    ap.add_argument("--hidden_size", type=int, default=768)
    ap.add_argument("--d_vlm", type=int, default=3584)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--train_mamba", action="store_true")
    ap.add_argument("--corruption", default="standard",
                    choices=["none", "light", "standard", "heavy"])
    ap.add_argument("--image_dropout", type=float, default=0.0,
                    help="probability of replacing the VLM image with a gray "
                         "frame, forcing the answer to rely on the cue tokens")
    ap.add_argument("--tor_str", default=DEFAULT_TOR_STR)
    ap.add_argument("--max_image_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EIEA training requires CUDA")
    torch.cuda.set_device(torch.device(args.device))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    from ssr_eiea_common import (
        VLM_BASE, load_midi, load_visual_encoders, load_vlm,
    )

    rows = load_rows(args.data)
    corr_cfg = CorruptionConfig.preset(args.corruption)
    print(f"loaded {len(rows)} rows; corruption={args.corruption}", flush=True)

    midi, mtok = load_midi(args.device)
    midi.eval()
    for p in midi.parameters():
        p.requires_grad_(False)

    cfg = EIEAConfig(
        hidden_size=args.hidden_size, d_vlm=args.d_vlm,
        num_tor=args.num_tor, bev_channels=args.bev_channels,
        freeze_mamba=not args.train_mamba, dropout=args.dropout,
    )
    encoder = EIEACueEncoder.from_midi(midi, cfg).to(args.device)
    encoder.train()
    clip_proc, clip, sig_proc, sig = load_visual_encoders(args.device)

    if args.dry_run:
        rng = random.Random(args.seed)
        for i, r in enumerate(rows[:2]):
            img = load_view_images(r, 1)[0]
            corr = apply_corruption(boxes_from_row(r), load_bev(r, args.bev_channels),
                                    load_depth(r), corr_cfg, rng)
            with torch.no_grad():
                ie = clip(**clip_proc(images=img, return_tensors="pt").to(args.device)).last_hidden_state.float()
                de = None
                if corr.depth is not None:
                    de = sig(**sig_proc(images=corr.depth, return_tensors="pt").to(args.device)).last_hidden_state.float()
                ids = mamba_text_ids(mtok, r).to(args.device)
                cue, aux = encoder(ie, de, torch.from_numpy(corr.boxes).to(args.device),
                                   None if corr.bev is None else torch.from_numpy(corr.bev).to(args.device),
                                   None, ids, return_aux=True)
            print(f"dry-run {i}: cue={tuple(cue.shape)} aux={aux['modalities']} "
                  f"corruption={corr.summary()}", flush=True)
        print("dry-run OK", flush=True)
        return

    vlm = load_vlm(args.device)
    vlm.train()
    vlm.config.use_cache = False
    for p in vlm.parameters():
        p.requires_grad_(False)
    vlm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    proc = __import__("transformers").AutoProcessor.from_pretrained(VLM_BASE)
    proc.tokenizer.add_tokens([args.tor_str], special_tokens=True)
    cap_processor_pixels(proc, args.max_image_tokens)

    midi_names = ("midi.image_proj", "midi.depth_proj")
    midi_params = [p for n, p in encoder.named_parameters()
                   if n.startswith(midi_names) and p.requires_grad]
    new_params = [p for n, p in encoder.named_parameters()
                  if not n.startswith(midi_names) and p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": midi_params, "lr": args.lr_midi, "weight_decay": args.weight_decay},
        {"params": new_params, "lr": args.lr, "weight_decay": args.weight_decay},
    ])

    rng = random.Random(args.seed)
    data_order = list(range(len(rows)))
    random.Random(args.seed + 1).shuffle(data_order)
    gray_path = None
    if args.image_dropout > 0:
        from PIL import Image
        gray_path = "/tmp/eiea_gray_256.png"
        Image.new("RGB", (256, 256), (128, 128, 128)).save(gray_path)
    t0 = time.time()
    for step in range(args.steps):
        r = rows[data_order[step % len(data_order)]]
        img = load_view_images(r, 1)[0]
        corr = apply_corruption(boxes_from_row(r), load_bev(r, args.bev_channels),
                                load_depth(r), corr_cfg, rng)
        with torch.no_grad():
            ie = clip(**clip_proc(images=img, return_tensors="pt").to(args.device)).last_hidden_state.float()
            de = None
            if corr.depth is not None:
                de = sig(**sig_proc(images=corr.depth, return_tensors="pt").to(args.device)).last_hidden_state.float()
        ids = mamba_text_ids(mtok, r).to(args.device)
        cue = encoder(ie, de, torch.from_numpy(corr.boxes).to(args.device),
                      None if corr.bev is None else torch.from_numpy(corr.bev).to(args.device),
                      None, ids).to(torch.bfloat16)
        r_vlm = r
        if gray_path is not None and rng.random() < args.image_dropout:
            r_vlm = dict(r)
            r_vlm["images"] = [gray_path]
        kw = prepare_vlm_batch(
            proc, r_vlm, torch.device(args.device), n_views=1,
            tor_str=args.tor_str, num_tor=args.num_tor,
            answer=str(r.get("answer", "")))
        keep = ("input_ids", "attention_mask", "pixel_values",
                "image_grid_thw", "labels")
        out = vlm(**{k: v for k, v in kw.items() if k in keep},
                  tor_embeds=cue, tor_token_id=kw["tor_token_id"])
        loss = out.loss / args.grad_accum
        loss.backward()
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(encoder.trainable_parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0:
            print(f"{step}/{args.steps} loss={out.loss.item():.4f} "
                  f"corrupt={corr.summary()} {time.time() - t0:.0f}s", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            encoder.save_checkpoint(args.save, extra={
                "step": step + 1, "args": vars(args),
                "corruption": corr_cfg.to_dict()})
        if args.smoke and step + 1 >= args.smoke:
            break

    encoder.save_checkpoint(args.save, extra={
        "step": step + 1, "args": vars(args),
        "corruption": corr_cfg.to_dict()})
    print(f"saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
