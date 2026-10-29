#!/usr/bin/env python3
"""Train the EIEA cue encoder on a frozen Qwen3-VL-30B-A3B backbone.

Training recipe (paper Stage 4 / rebuttal EIEA-lite):

* cues: 3D boxes + BEV map + depth (optionally BEV flow);
* interpreter: Mamba-2 / SSR-MIDI, body frozen, input/output projections and
  the box/BEV encoders trained;
* loss: answer cross-entropy through the frozen VLM with the K cue tokens
  injected at ``<TOR>`` positions;
* robustness: every step independently applies random modality masking,
  partial-cue dropping, box jitter and BEV/depth noise
  (:mod:`model.eiea.corruption`).

The 30B model is large; the default device map follows the public evaluation
scripts (22 decoder layers on GPU 0, the rest on GPU 1).  Use ``--dry-run`` to
validate the data / cue encoder / corruption pipeline without loading the VLM.

Example (2x46 GB):
    python training/train_eiea_30b.py \
        --model_dir /dev/shm/models/xembodied-v0 \
        --data results/eiea_box/train_bev.json \
        --save results/eiea_box/eiea_30b_box_bev_depth.pt \
        --steps 2000 --corruption standard
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

EIEA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EIEA_DIR.parent
if str(EIEA_DIR) not in sys.path:
    sys.path.insert(0, str(EIEA_DIR))

from corruption import CorruptionConfig, apply_corruption
from cue_encoder import EIEAConfig, EIEACueEncoder
from preprocess import make_image_tchw
from eiea_data import (
    boxes_from_row, cap_processor_pixels, load_bev, load_depth, load_rows,
    load_view_images, mamba_text_ids, prepare_vlm_batch,
)

DEFAULT_MODEL_DIR = os.environ.get("XEMB_MODEL_DIR", "/dev/shm/models/xembodied-v0")
DEFAULT_TOR_STR = "<|endoftext|>"  # pad token used as the TOR placeholder


# ---------------------------------------------------------------------------
# Model / encoder loading
# ---------------------------------------------------------------------------
def load_eiea_vlm(model_dir: str, gpu_layers: int, dtype=torch.bfloat16):
    """Load the repo's Qwen3-VL-30B + VGGT + TOR model with 2-GPU split."""
    from transformers import AutoConfig, AutoProcessor

    vggt_dir = os.environ.get(
        "VGGT_REPO_DIR", str(PROJECT_ROOT / "xiaomi_spatialembodied" / "3DA" / "my_qwen3_vggt_xattnv2"))
    os.environ.setdefault("VGGT_REPO_DIR", vggt_dir)
    os.environ.setdefault("VGGT_DEVICE", "cuda:0")
    os.environ.setdefault("VGGT_DTYPE", "bfloat16")

    modeling = EIEA_DIR / "models" / "qwen3_vl_moe_vggt.py"
    spec = importlib.util.spec_from_file_location("xembodied_eiea_modeling", modeling)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["xembodied_eiea_modeling"] = mod
    spec.loader.exec_module(mod)
    cls = mod.Qwen3VLMoeVGGTForConditionalGeneration

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    n_layers = int(cfg.text_config.num_hidden_layers)
    gpu_layers = max(0, min(gpu_layers, n_layers))
    dm = {f"model.language_model.layers.{i}": (0 if i < gpu_layers else 1)
          for i in range(n_layers)}
    dm.update({
        "model.language_model.embed_tokens": 0,
        "model.language_model.norm": 1,
        "model.language_model.rotary_emb": 0,
        "model.visual": 0,
        "spatial_encoder": 0,
        "connector": 0,
        "lm_head": 1,
    })
    cfg.vggt_device = os.environ["VGGT_DEVICE"]
    cfg.vggt_dtype = os.environ["VGGT_DTYPE"]
    cfg.vggt_repo_dir = vggt_dir
    model = cls.from_pretrained(model_dir, config=cfg, dtype=dtype,
                                device_map=dm, trust_remote_code=True)
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    tps = int(cfg.vision_config.temporal_patch_size)
    d_vlm = int(cfg.text_config.hidden_size)
    return model, proc, tps, d_vlm


def load_cue_encoder(args, device: str, d_vlm: int,
                     vis_device: str) -> tuple:
    """Load MIDI + CLIP/SigLIP and build the trainable EIEA cue encoder."""
    from ssr_eiea_common import load_midi, load_visual_encoders

    midi, mtok = load_midi(device)
    midi.eval()
    for p in midi.parameters():
        p.requires_grad_(False)

    cfg = EIEAConfig(
        image_dim=args.image_dim,
        depth_dim=args.depth_dim,
        hidden_size=args.hidden_size,
        d_vlm=d_vlm,
        num_tor=args.num_tor,
        bev_channels=args.bev_channels,
        freeze_mamba=not args.train_mamba,
        dropout=args.dropout,
    )
    encoder = EIEACueEncoder.from_midi(midi, cfg).to(device)
    encoder.train()
    clip_proc, clip, sig_proc, sig = load_visual_encoders(vis_device)
    for m in (clip, sig):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return encoder, mtok, clip_proc, clip, sig_proc, sig


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--data", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="only batch_size=1 is supported (cue encoder is per-sample)")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="learning rate for new cue modules")
    ap.add_argument("--lr_midi", type=float, default=2e-5,
                    help="learning rate for the MIDI projections")
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vis_device", default="cuda:0",
                    help="device for CLIP/SigLIP + cue encoder")
    ap.add_argument("--gpu_layers", type=int, default=22,
                    help="decoder layers on cuda:0; the rest go to cuda:1")
    ap.add_argument("--n_views", type=int, default=1,
                    help="images given to the VLM (EIEA cues always use the front view)")
    ap.add_argument("--max_image_tokens", type=int, default=512,
                    help="native Qwen image-token budget per image")
    ap.add_argument("--tor_str", default=DEFAULT_TOR_STR)
    ap.add_argument("--num_tor", type=int, default=10)
    ap.add_argument("--hidden_size", type=int, default=768)
    ap.add_argument("--image_dim", type=int, default=1024)
    ap.add_argument("--depth_dim", type=int, default=1152)
    ap.add_argument("--bev_channels", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--train_mamba", action="store_true",
                    help="also train the Mamba body (paper freezes it)")
    ap.add_argument("--corruption", default="standard",
                    choices=["none", "light", "standard", "heavy"])
    ap.add_argument("--image_dropout", type=float, default=0.0,
                    help="probability of replacing the VLM image with a gray "
                         "frame, forcing the answer to rely on the cue tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--dry_run", action="store_true",
                    help="build cues + corruptions without loading the 30B model")
    ap.add_argument("--smoke", type=int, default=0,
                    help="stop after N optimizer steps (debug)")
    return ap.parse_args()


def _filter_model_kwargs(kw: Dict) -> Dict:
    keep = ("input_ids", "attention_mask", "pixel_values",
            "pixel_values_videos", "image_grid_thw", "video_grid_thw",
            "labels")
    return {k: v for k, v in kw.items() if k in keep}


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("EIEA cue encoder currently supports batch_size=1")
    if not torch.cuda.is_available():
        raise RuntimeError("EIEA training requires CUDA (MIDI/VLM are on GPU)")
    torch.cuda.set_device(torch.device(args.device))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rows = load_rows(args.data)
    if not rows:
        raise ValueError(f"no rows in {args.data}")
    print(f"loaded {len(rows)} rows from {args.data}", flush=True)

    corr_cfg = CorruptionConfig.preset(args.corruption)
    print(f"corruption preset={args.corruption}: {corr_cfg.to_dict()}", flush=True)

    # ---- dry run: data + cue encoder only --------------------------------
    if args.dry_run:
        from ssr_eiea_common import load_midi, load_visual_encoders
        midi, mtok = load_midi(args.device)
        cfg = EIEAConfig(d_vlm=2048, num_tor=args.num_tor,
                         bev_channels=args.bev_channels,
                         freeze_mamba=not args.train_mamba)
        encoder = EIEACueEncoder.from_midi(midi, cfg).to(args.device)
        encoder.eval()
        cp, clip, sp, sig = load_visual_encoders(args.device)
        rng = random.Random(args.seed)
        for i, r in enumerate(rows[: min(3, len(rows))]):
            img = load_view_images(r, 1)[0]
            dep = load_depth(r)
            boxes = boxes_from_row(r)
            bev = load_bev(r, args.bev_channels)
            corr = apply_corruption(boxes, bev, dep, corr_cfg, rng)
            with torch.no_grad():
                ie = clip(**cp(images=img, return_tensors="pt").to(args.device)).last_hidden_state.float()
                de = None
                if corr.depth is not None:
                    de = sig(**sp(images=corr.depth, return_tensors="pt").to(args.device)).last_hidden_state.float()
                ids = mamba_text_ids(mtok, r).to(args.device)
                bt = torch.from_numpy(corr.boxes).to(args.device)
                bv = None if corr.bev is None else torch.from_numpy(corr.bev).to(args.device)
                cue = encoder(ie, de, bt, bv, None, ids, return_aux=True)
            print(f"dry-run {i}: cue={tuple(cue[0].shape)} aux={cue[1]['modalities']} "
                  f"corruption={corr.summary()}", flush=True)
        print("dry-run OK", flush=True)
        return

    # ---- load frozen VLM + trainable cue encoder ------------------------
    model, proc, tps, d_vlm = load_eiea_vlm(
        args.model_dir, args.gpu_layers)
    cap_processor_pixels(proc, args.max_image_tokens)
    if args.save and not os.path.isabs(args.save):
        args.save = str(PROJECT_ROOT / args.save)
    encoder, mtok, clip_proc, clip, sig_proc, sig = load_cue_encoder(
        args, args.vis_device, d_vlm, args.vis_device)
    if d_vlm != encoder.cfg.d_vlm:
        raise ValueError(f"d_vlm mismatch: model={d_vlm}, encoder={encoder.cfg.d_vlm}")

    # MIDI projections use a lower LR; new cue modules use args.lr.
    midi_names = ("midi.image_proj", "midi.depth_proj")
    midi_params = [p for n, p in encoder.named_parameters()
                   if n.startswith(midi_names) and p.requires_grad]
    new_params = [p for n, p in encoder.named_parameters()
                  if not n.startswith(midi_names) and p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": midi_params, "lr": args.lr_midi, "weight_decay": args.weight_decay},
        {"params": new_params, "lr": args.lr, "weight_decay": args.weight_decay},
    ])
    print(f"trainable params: midi_proj={sum(p.numel() for p in midi_params):,} "
          f"new={sum(p.numel() for p in new_params):,} "
          f"(mamba frozen={encoder.cfg.freeze_mamba})", flush=True)

    rng = random.Random(args.seed)
    data_order = list(range(len(rows)))
    random.Random(args.seed + 1).shuffle(data_order)
    gray_path = None
    if args.image_dropout > 0:
        from PIL import Image
        gray_path = "/tmp/eiea_gray_256.png"
        Image.new("RGB", (256, 256), (128, 128, 128)).save(gray_path)
    t0 = time.time()
    running = 0.0
    step = 0
    gn = None
    for step in range(args.steps):
        r = rows[data_order[step % len(data_order)]]
        img = load_view_images(r, 1)[0]
        dep = load_depth(r)
        boxes = boxes_from_row(r)
        bev = load_bev(r, args.bev_channels)
        corr = apply_corruption(boxes, bev, dep, corr_cfg, rng)

        with torch.no_grad():
            ie = clip(**clip_proc(images=img, return_tensors="pt").to(args.vis_device)).last_hidden_state.float()
            de = None
            if corr.depth is not None:
                de = sig(**sig_proc(images=corr.depth, return_tensors="pt").to(args.vis_device)).last_hidden_state.float()

        ids = mamba_text_ids(mtok, r).to(args.device)
        bt = torch.from_numpy(corr.boxes).to(args.device)
        bv = None if corr.bev is None else torch.from_numpy(corr.bev).to(args.device)
        cue = encoder(ie, de, bt, bv, None, ids)

        r_vlm = r
        if gray_path is not None and rng.random() < args.image_dropout:
            n_vis = max(1, min(args.n_views, len(r.get("images", [])) or 1))
            r_vlm = dict(r)
            r_vlm["images"] = [gray_path] * n_vis
        kw = prepare_vlm_batch(
            proc, r_vlm, torch.device(args.device),
            n_views=args.n_views, tor_str=args.tor_str,
            num_tor=args.num_tor, answer=str(r.get("answer", "")))
        tchw = make_image_tchw(kw["images"], tps).to(args.device)
        out = model(**_filter_model_kwargs(kw), image_tchw=[tchw],
                    tor_embeds=cue.to(torch.bfloat16),
                    tor_token_id=kw["tor_token_id"])
        loss = out.loss / args.grad_accum
        loss.backward()
        running += float(out.loss.item())
        if (step + 1) % args.grad_accum == 0:
            gn = torch.nn.utils.clip_grad_norm_(encoder.trainable_parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0:
            print(f"{step}/{args.steps} loss={out.loss.item():.4f} "
                  f"avg={running / (step % args.log_every + 1):.4f} "
                  f"grad={float(gn) if gn is not None else float('nan'):.3f} "
                  f"corrupt={corr.summary()} {time.time() - t0:.0f}s", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            encoder.save_checkpoint(args.save, extra={
                "step": step + 1, "args": vars(args),
                "corruption": corr_cfg.to_dict()})
            print(f"saved {args.save} at step {step + 1}", flush=True)
        if args.smoke and step + 1 >= args.smoke:
            break

    encoder.save_checkpoint(args.save, extra={
        "step": step + 1, "args": vars(args),
        "corruption": corr_cfg.to_dict()})
    print(f"saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
