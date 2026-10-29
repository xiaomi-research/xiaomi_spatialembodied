# EIEA Training: Cues, Masking and Noise

This document records the public EIEA training recipe implemented in
[`train_eiea_30b.py`](../train_eiea_30b.py) (frozen 30B) and
[`train_eiea_box.py`](../train_eiea_box.py) (frozen 7B SSR
proxy). It follows Stage 4 of the paper (answer cross-entropy with plug-and-play
physical-cue tokens) and the robustness experiments in the rebuttal (W1 / LSwS:
unreliable, missing or partial cues).

## 1. Cues

| Cue | Source | Encoder | Public data field |
|-----|--------|---------|-------------------|
| 3D boxes | detector / dataset GT annotations in the ego frame | class embedding + 10 numeric features → MLP | `boxes` |
| BEV map | HD-map / occupancy raster (5 channels for nuScenes map; 3 for MapLM point-cloud BEV) | CNN → 8×8 pool → Linear | `bev` |
| Depth | DepthPro map from the front camera | SigLIP-SO400M → `depth_proj` | `depth` |
| BEV flow (optional) | PriorMotion-style motion raster | CNN → 8×8 pool → Linear | `flow` |

The interpreter also receives the CLIP front-view features and the question
text. The canonical sequence and retained positions are specified in
[`EIEA_ARCHITECTURE.md`](EIEA_ARCHITECTURE.md).

`boxes`, `bev` and `depth` are all optional per row: if a field is missing the
corresponding tokens are omitted (the same state the model sees when random
masking drops it). The Ego3D rows built for the rebuttal contain boxes + depth
for every row; only a small `train_bev.json` subset has BEV rasters. To render
BEV for the full set:

```bash
python xiaomi_spatialembodied/EIEA/build_eiea_bev.py --split train \
  --data results/eiea_box/train.json --out results/eiea_box
python xiaomi_spatialembodied/EIEA/build_eiea_bev.py --split val \
  --data results/eiea_box/val.json --out results/eiea_box
```

This writes `results/eiea_box/{split}_bev.json` with 5-channel 200×200
ego-centric rasters (`NUSCENES_ROOT` points at the nuScenes v1.0-trainval
metadata).

The paper's modality table lists BEV map, BEV flow and 3D box. The public
training recipe in this repository uses box + BEV map + depth, matching the
rebuttal's EIEA-lite and the data fields that are actually released; the flow
encoder is present behind an optional argument but no public flow raster is
shipped.

## 2. Trainable modules

- Frozen: the Qwen3-VL-30B-A3B backbone (including VGGT/3DA), CLIP, SigLIP.
- Frozen per paper Appendix B: the Mamba-2 body (`--train_mamba` overrides).
- Trained: `image_proj`, `depth_proj`, `BoxEncoder`, `BEVEncoder`,
  `FlowEncoder` (if used), `CueProjector`, and the `K` latent queries.
- Loss: Eq. (8), answer cross-entropy through the frozen VLM with the `K` cue
  tokens injected at `<TOR>` positions.

Default optimizer: AdamW, weight decay 0.05, LR `2e-4` for new modules and
`2e-5` for the MIDI projections. These are the rebuttal EIEA-lite settings;
the paper's Stage 4 table lists `1e-5` for the LoRA/RL stage, which is not the
same module set.

## 3. Random masking + noise

Every training sample independently draws from
[`corruption.py`](../corruption.py). Presets:
`none`, `light`, `standard` (default), `heavy`.

| Group | Intervention | Standard probability / magnitude |
|-------|--------------|----------------------------------|
| modality | drop all boxes | 0.15 |
| modality | drop BEV | 0.15 |
| modality | drop depth | 0.15 |
| modality | drop every cue | 0.02 |
| partial | keep exactly one of box/BEV/depth | 0.10 |
| box | drop a fraction of boxes (50%) | 0.15 |
| box | jitter x/y (1.0 m), z (0.5 m), yaw (0.1 rad), velocity (0.5 m/s) | 0.20 |
| BEV | Gaussian noise (10% of per-channel range) | 0.20 |
| BEV | 8×8 block mask (50%) | 0.20 |
| BEV | drop one channel | 0.05 |
| depth | Gaussian noise (10% of range) | 0.20 |
| depth | 8×8 block mask (50%) | 0.15 |
| depth | flatten to mean | 0.05 |

Why this combination: whole-modality dropout teaches the interpreter to keep
working when a detector/map/depth source is absent; the partial-cue curriculum
teaches it to exploit whichever source remains; noise/jitter prevents the cue
encoder from over-fitting exact annotation values. This is the training-side
counterpart of the rebuttal's inference-time stress test, so the same
conditions are used in [`eval_eiea_30b.py`](../eval_eiea_30b.py).

There is also an optional `--image_dropout p` knob: with probability `p` the
VLM's own image is replaced by a gray frame while the cue is generated from the
real image, forcing the answer to rely on the cue tokens. It defaults to 0
(paper-style training). The rebuttal's end-to-end MIDI experiment found that
answer CE can otherwise ignore box/BEV tokens when the VLM image is sufficient
(§12.4), so use `--image_dropout 0.1-0.3` when the goal is to *measure cue
utilisation* rather than only clean accuracy.

## 4. Commands

### 30B (paper backbone; 2×46 GB with the public device split)

```bash
python xiaomi_spatialembodied/EIEA/train_eiea_30b.py \
  --model_dir /dev/shm/models/xembodied-v0 \
  --data results/eiea_box/train_bev.json \
  --save results/eiea_box/eiea_30b_box_bev_depth.pt \
  --steps 2000 --corruption standard
```

Validate the data/cue path without loading the 30B model:

```bash
python xiaomi_spatialembodied/EIEA/train_eiea_30b.py \
  --data results/eiea_box/train_bev.json \
  --save /tmp/unused.pt --dry_run
```

### 7B SSR proxy (rebuttal EIEA-lite; runnable on 2×4090)

```bash
python xiaomi_spatialembodied/EIEA/train_eiea_box.py \
  --data results/eiea_box/train_bev.json \
  --save results/eiea_box/eiea_7b_box_bev_depth.pt \
  --steps 800 --corruption standard
```

Required checkpoints are configured through the `SSR_*` environment variables
in [`ssr_eiea_common.py`](../ssr_eiea_common.py):
`SSR_ROOT`, `SSR_MODEL_ROOT`, `SSR_MIDI_DIR`, `SSR_MAMBA_DIR`,
`SSR_CLIP_DIR`, `SSR_SIGLIP_DIR`, `SSR_VLM_BASE`, `SSR_VLM_ADAPTER`.

### Evaluation

```bash
python xiaomi_spatialembodied/EIEA/eval_eiea_30b.py \
  --model_dir /dev/shm/models/xembodied-v0 \
  --ckpt results/eiea_box/eiea_30b_box_bev_depth.pt \
  --data results/eiea_box/val_bev.json \
  --limit 100 --out results/eiea_box/eval_eiea_30b.json
```

The evaluator reports clean accuracy / MAE, the paired change for each
condition, and a `cue_changes_output` flag. If clean and `no_cue` produce
identical predictions on every question, the cue was not injected (or the VLM
ignored it); treat that run as invalid.

## 5. Alignment notes / known limitations

- The paper trains EIEA in Stage 4 on the Stage-3 checkpoint and adds GRPO.
  This repository implements the supervised cue-alignment objective (Eq. 8);
  GRPO is part of the main training pipeline, not this EIEA reference trainer.
- The 30B trainer follows the public 2-GPU device split used by the evaluation
  scripts. It has not been run end-to-end on the 2×4090 development machine
  (61 GB bf16 weights plus activations); the 7B proxy trainer is the runnable
  path for robustness ablations.
- The rebuttal EIEA-lite numbers (box/BEV removal degrades numeric RMSE by
  ~30%; 50% box drop / 1 m jitter costs ~2.9 MC points) come from the 7B
  proxy and small held-out sets. They are proof-of-concept evidence for cue
  utilisation, not paper-scale 30B results.
- The rebuttal's strongest *cue-use* evidence for box/BEV came from a
  frozen-MIDI + cross-attention adapter, while the end-to-end Mamba training
  could ignore extra tokens when the VLM image was sufficient (§12.4 of the
  rebuttal). This repository implements the paper's Mamba interpreter; use
  `--image_dropout` and the `box_only`/`bev_only`/`depth_only` evaluation
  conditions to verify that the trained cue is actually consumed, and compare
  against the cross-attention adapter if cue utilisation is the primary claim.
- The public data builder emits at most 30 boxes per frame; the supplement's
  `L_3D = 100` is enforced as a capacity cap, not by padding with fake boxes.
