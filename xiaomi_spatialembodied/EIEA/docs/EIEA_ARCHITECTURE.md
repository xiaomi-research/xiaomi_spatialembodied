# EIEA Architecture: Forward Pass and Design Clarifications

This document specifies the public EIEA (Efficient Image-Embodied Adapter)
forward pass and records the points where the paper, the technical supplement
and the rebuttal need to be read together. The executable reference is
[`cue_encoder.py`](../cue_encoder.py).

## Mamba-2 Input Sequence

The Mamba-2 interpreter performs one causal scan over

```
[X_T; X_img; X_depth; <PHY>; Z_box; <MOD>; Z_BEVmap; (<MOD>; Z_BEVflow); R_1 ... R_K]
```

| Segment | Source | Shape | Description |
|---------|--------|-------|-------------|
| `X_T` | question tokenizer + Mamba embedding | `[1, L_text, 768]` | Question/options text |
| `X_img` | CLIP ViT-L last hidden states → `image_proj` | `[1, N_img, 768]` | Front-view image features |
| `X_depth` | SigLIP-SO400M last hidden states → `depth_proj` | `[1, N_dep, 768]` | DepthPro depth features |
| `<PHY>` | learnable marker | `[1, 768]` | Starts the physical-cue block |
| `Z_box` | `BoxEncoder` | `[N_box, 768]` | Class embedding + 10 geometric/kinematic features → MLP |
| `<MOD>` | learnable separator | `[1, 768]` | Separates heterogeneous physical modalities |
| `Z_BEVmap` | `BEVEncoder` | `[64, 768]` | BEV raster → CNN → 8×8 spatial pool → Linear |
| `Z_BEVflow` | `FlowEncoder` (optional) | `[64, 768]` | BEV-flow raster (paper's PriorMotion modality) |
| `R_1..R_K` | learnable latent queries | `[K, 768]` | `K = 10` in the public implementation |

`<PHY>`/`<MOD>` are the paper supplement's symbolic markers; they are inserted
only around physical cues that are actually present, so a missing modality
simply removes its separator as well. Set `EIEAConfig.use_modality_tokens =
False` to reproduce the simpler `[X_T; X_img; X_depth; Z_box; Z_BEVmap; R]`
sequence used by the early SSR-MIDI prototype.

All segments are projected to `d_mamba = 768` before concatenation. The
supplement's design table says `d = 1024`, but the released SSR-MIDI checkpoint
is built on `mamba-130m`, whose native hidden size is 768; the public code uses
768 so the pretrained MIDI weights load without re-shaping. Missing cues are
represented by omitting their tokens; this is the same state the model sees when
the training-time random masking in
[`corruption.py`](../corruption.py) removes a modality.

### Ordering

The paper supplement writes the interpreter input as
`[X_T; <PHY>; Z_BEVmap; <MOD>; Z_BEVflow; <MOD>; Z_3Dbox]` (text first,
physical cues after it). The rebuttal Q&A writes
`[X_T; <PHY>; Z_map; <MOD>; Z_flow; <MOD>; Z_box; R_1..R_10]` and explains that
the ten latent tokens are **last** so their causal state has already seen the
whole question and all physical blocks. The public implementation follows the
rebuttal ordering. The earlier version of this document placed the latent
queries *before* the question text; that is inconsistent with a causal
interpreter and has been corrected.

## Retained Positions

Only the hidden states at the `K` latent-query positions are retained:

```python
tor_hidden = hidden_states[:, -K:, :]          # queries are appended last
```

Extracting by offset (`N_img + N_dep + N_box + ...`) is brittle because the
box/BEV/flow token counts vary with the sample and with missing modalities;
the implementation therefore takes the last `K` positions by construction.

## Projection to VLM Space

```python
Z_phy = Linear(GELU(Linear(RMSNorm(tor_hidden))))   # [K, d_VLM]
```

For the 30B model, `d_VLM = 2048` (Qwen3-VL-30B-A3B word-embedding size). For
the 7B SSR proxy (rebuttal EIEA-lite), `d_VLM = 3584`.

## TOR Token Injection

The `K` projected cue tokens replace the embeddings of `K` designated
placeholder tokens in the frozen VLM's prompt:

```python
mask = input_ids == tor_token_id          # or an explicit tor_positions mask
inputs_embeds = inputs_embeds.masked_scatter(
    mask.unsqueeze(-1).expand_as(inputs_embeds), tor_embeds.reshape(-1, d_vlm))
```

The public 30B path uses `tor_token_id = 151643` (`<|endoftext|>`) as the
placeholder, exactly `K` times in the prompt. The model now **raises** if the
marker count does not equal the cue length, instead of silently skipping the
injection. During decoding there are no prompt positions left, so injection
is intentionally skipped.

The prompt builder places the `K` placeholders immediately before the question
(`<TOR>...<TOR>\n<question>`), matching the rebuttal's evaluation scripts. The
paper schematic draws `<PHY>` after `<TEXT>`; because the VLM is a decoder, both
orders expose the cue to the question, and the public/rebuttal implementation is
the one used here.

## Missing / Partial Cues

EIEA is trained with random modality masking and noise (see
[`EIEA_TRAINING.md`](EIEA_TRAINING.md)):

- whole-modality drop of box, BEV or depth;
- partial-cue curriculum keeping only one of the three;
- per-box element drop and geometric jitter;
- BEV block masking, Gaussian noise and channel drop;
- depth Gaussian noise, block masking and flattening.

At inference the same interventions are applied to produce the stress-test
conditions `no_box`, `drop50_box`, `jitter_box`, `no_bev`, `bev_mask50`,
`bev_noise`, `no_depth`, `depth_noise`, `depth_mask`, `no_physical`,
`box_only`, `bev_only`, `depth_only`, `wrong_scene` and `no_cue`.

## Dimension / Count Clarifications

The public checkpoint and rebuttal use **`K = 10`** latent cue tokens and
`d_VLM = 2048` for Qwen3-VL-30B-A3B. The technical supplement's
`L_compact = 64` and `d_VLM = 8192` describe an earlier configuration and do
not match the released 30B word-embedding size; they should not be used to
size the public model. This document previously called the 64 a
typographical error; the safer statement is that 10/2048 is what the public
code, the rebuttal experiments and the 30B checkpoint actually use.

The supplement specifies a capacity of `L_3D = 100` box tokens per scene. The
public data builder emits at most 30 boxes per frame (all annotations within
50 m), and `BoxEncoder.max_boxes` defaults to 100 to enforce the paper's cap
without padding short scenes with fake boxes.

## Modality Encoders

The paper lists DiffMap (BEV map, 768-d), PriorMotion (BEV flow, 512-d) and
SG-Net (3D box, 256-d) as the modality encoders. Those pretrained encoders are
not part of the public release. This repository ships lightweight stand-ins
(CNN for map/flow, MLP for boxes) with the same token interface and shape, so
the interpreter / projection / injection path is paper-faithful while the
encoder weights can be swapped in later. The public training recipe uses
box + BEV map + depth, matching the data fields that are actually shipped; the
paper's BEV-flow modality is available through `FlowEncoder` when flow rasters
are supplied.
