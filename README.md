<div align="center">
<h1><b>Xiaomi-SpatialEmbodied</b></h1>
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2604.18484-B31B1B.svg?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.18484)
[![ModelScope](https://img.shields.io/badge/ModelScope-Collection-624aff.svg?logo=modelscope&amp;logoColor=white)](https://www.modelscope.cn/models/xzhong/xembodied-v0)

</div>

<div align="center">

**[<a href="README_CN.md">中文版</a>]**
**[<a href="https://www.modelscope.cn/models/xzhong/xembodied-v0">Model Weights</a>]**
**[<a href="https://arxiv.org/abs/2604.18484">Paper</a>]**

</div>

<div align="center">
<strong>A Xiaomi Autodrive and Robotics Team research project, developed with academic collaborators.</strong>
</div>

Official implementation of **Xiaomi-SpatialEmbodied**, the released codebase for the paper **XEmbodied: A Foundation Model with Enhanced Geometric and Physical Cues for Large-Scale Embodied Environments** ([arXiv 2604.18484](https://arxiv.org/abs/2604.18484)).

The model is designed for cloud-side autonomous-driving and robotics VQA in data closed-loop systems. It improves domain semantics and 3D understanding while preserving general VLM capabilities. The implementation follows the paper's two complementary enhancement paths: a geometric adapter for intrinsic spatial representations and an embodied adapter for compact physical cues.

**Key Contributions:**
- Present Xiaomi-SpatialEmbodied, a cloud-side embodied closed-loop VQA generalist fusing intrinsic geometric representations with physical cue interaction.
- Propose 3DA and EIEA-powered implicit physical cue alignment for adaptive geometric priors injection and efficient physical-augmented reasoning.
- Develop a progressive domain curriculum for robust adaptation with reduced forgetting, validated on 18 benchmarks to demonstrate consistent embodied understanding gains.

## Model Introduction

Xiaomi-SpatialEmbodied uses Qwen3-VL-30B-A3B-Instruct as its multimodal language backbone and adds trainable spatial and embodied interfaces. Images are processed by the native Qwen vision tower and a VGGT 3D foundation model; the resulting features are aligned before language reasoning, so spatial evidence participates in the model's internal representation rather than being appended as text or an auxiliary depth map.

- **3D Adapter (3DA):** VGGT extracts multi-layer geometric features from the same camera image. `VGGTEmbeddingMerger` resamples the VGGT patch grid to Qwen's image-token grid, applies token merging, normalization, and MLP projection, then injects the result through a cross-attention residual (image tokens query VGGT tokens). This preserves the original sequence length while adding depth, layout, and relative-position cues.
- **Efficient Image-Embodied Adapter (EIEA):** Outputs from embodied tools such as occupancy, 3D boxes, and map cues are distilled into a compact set of TOR embeddings. These embeddings are inserted at designated positions in the MLLM context, providing physical affordance priors without exposing raw tool traces to the language model. This repository ships the inference consumer and precomputed caches.
- **Progressive domain curriculum:** Training proceeds from general multimodal capability to spatial/3D supervision and then embodied driving and manipulation data. The staged recipe reduces catastrophic forgetting and yields consistent gains across the 18 benchmarks reported in the paper.

The released main checkpoint is the 3DA variant under `xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2/`; the connector variants in `xiaomi_spatialembodied/3DA/variants/` are provided for ablation studies.

### Model Architecture

<div align="center">
  <img src="demo/model_architecture.png" alt="Xiaomi-SpatialEmbodied model architecture" width="85%">
  <br>
  <em>Architecture of Xiaomi-SpatialEmbodied.</em>
</div>

### Benchmark Overview

<div align="center">
  <img src="demo/radar_results.png" alt="Benchmark results radar chart" width="90%">
  <br>
  <em>Performance across spatial and 3D understanding, semantic reasoning, and embodied affordance benchmarks.</em>
</div>

## News
- 2026/09/14: Xiaomi-SpatialEmbodied code officially released
- 2026/04/20: Xiaomi-SpatialEmbodied paper available on arXiv

## Results

### Table 1: Spatial & 3D Understanding
The best results among the listed models are **bolded**.

| Model | Ego3DBench ACC | Ego3DBench RMSE | SURDS | VLADBench | STRIDE-QA |
|---|---|---|---|---|---|
| **Proprietary Models** | | | | | |
| GPT-4o [67] | 52.70 | 19.20 | 13.31 | 56.00 | 14.70 |
| Gemini-1.5 [82] | 53.50 | 19.62 | 32.77 | 54.23 | - |
| **Open-Source Models** | | | | | |
| Qwen2.5-VL-7B [5] | 41.10 | 30.36 | 12.61 | 50.75 | 1.41 |
| Qwen2.5-VL-32B [5] | 57.30 | 15.87 | 38.82 | 61.33 | 6.58 |
| Qwen3-VL-A3B-30B [4] | 53.15 | 13.84 | 38.69 | 59.81 | 8.69 |
| Qwen3-VL-32B [4] | **59.93** | 15.70 | 41.53 | 61.64 | 6.00 |
| **Spatial Models** | | | | | |
| UniVG-R1-7B [6] | 46.82 | 53.85 | 1.96 | 46.51 | 4.61 |
| PR1-OCR-2B [101] | 38.88 | 11.60 | 13.81 | 44.65 | 8.18 |
| PR1-Detection-3B [101] | 36.74 | 62.61 | 33.84 | 56.08 | 4.35 |
| PR1-Counting-2B [101] | 40.32 | 51.95 | 13.69 | 48.52 | 1.94 |
| PR1-Grounding-2B [101] | 39.54 | 12.34 | 13.96 | 43.67 | 7.15 |
| **Embodied Models** | | | | | |
| DriveMM [36] | 49.73 | 12.68 | 8.73 | 47.45 | 1.12 |
| Cosmos-R1 [2] | 45.62 | 23.41 | 10.05 | 55.93 | 1.03 |
| Mimo-Embodied [29] | 53.62 | 16.57 | 24.04 | 55.82 | 7.90 |
| **Our Model** | | | | | |
| Xiaomi-SpatialEmbodied (Best) | 55.28 | **9.25** | **83.83** | **68.61** | **27.76** |

### Table 2: Semantic & Reasoning
The best results among the listed models are **bolded**.

| Model | DriveBench | DriveLMM-o1 | MapLM-v2 | LingoQA | Omnidrive |
|---|---|---|---|---|---|
| **Proprietary Models** | | | | | |
| GPT-4o [67] | 47.97 | 57.84 | 57.81 | 53.58 | 4.54 |
| Gemini-1.5 [82] | - | - | 58.44 | 61.30 | 2.73 |
| **Open-Source Models** | | | | | |
| Qwen2.5-VL-7B [5] | 43.29 | 37.81 | 55.10 | 53.20 | 2.53 |
| Qwen2.5-VL-32B [5] | 53.06 | 55.04 | 57.47 | 43.70 | 0.00 |
| Qwen3-VL-A3B-30B [4] | 46.22 | 57.17 | 53.63 | 47.00 | 0.31 |
| Qwen3-VL-32B [4] | 51.70 | 55.29 | 46.77 | 54.30 | 0.00 |
| **Spatial Models** | | | | | |
| UniVG-R1-7B [6] | 48.75 | 51.42 | 39.13 | 39.20 | 1.20 |
| PR1-OCR-2B [101] | 50.73 | 47.68 | 44.05 | 48.50 | 4.07 |
| PR1-Detection-3B [101] | 41.35 | 44.81 | 62.47 | 56.08 | 0.10 |
| PR1-Counting-2B [101] | 39.72 | 50.94 | 47.43 | 48.52 | 2.00 |
| PR1-Grounding-2B [101] | 52.03 | 53.83 | 51.00 | 50.30 | 4.85 |
| **Embodied Models** | | | | | |
| DriveMM [36] | 44.50 | 65.91 | 52.18 | 37.70 | 1.10 |
| Cosmos-R1 [2] | 35.80 | 54.66 | 55.33 | 52.70 | 1.23 |
| Mimo-Embodied [29] | 52.95 | 40.31 | 65.23 | **68.60** | 4.90 |
| **Our Model** | | | | | |
| Xiaomi-SpatialEmbodied (Best) | **53.18** | **77.01** | **78.55** | 65.70 | **25.43** |

### Table 3: Embodied & Affordance
The best results among the listed models are **bolded**.

| Model | Affordance-2K | Robo-Afford | Cosmos-R1 | Embodied-R1 | RoboRefitBench | VABench-Point | Where2place |
|---|---|---|---|---|---|---|---|
| **Proprietary Models** | | | | | | | |
| GPT-4o [67] | 2.70 | 3.80 | 67.40 | 0.35 | 13.40 | 0.40 | 0.44 |
| Gemini-1.5 [82] | 5.19 | 4.20 | 71.10 | - | 35.50 | 0.80 | 0.89 |
| **Open-Source Models** | | | | | | | |
| Qwen2.5-VL-7B [5] | 8.70 | 3.00 | 72.50 | 0.25 | 76.40 | 0.75 | 0.90 |
| Qwen2.5-VL-32B [5] | 9.10 | 3.25 | 72.50 | 0.00 | 77.60 | 1.45 | 1.25 |
| Qwen3-VL-A3B-30B [4] | 8.85 | 3.65 | 74.50 | 0.05 | **83.95** | 1.10 | 1.51 |
| Qwen3-VL-32B [4] | 6.90 | 4.00 | 75.50 | 0.05 | 83.15 | 1.84 | 1.50 |
| **Embodied Models** | | | | | | | |
| Cosmos-R1 [2] | 6.63 | 1.95 | 72.00 | 0.05 | 34.75 | 0.35 | 0.45 |
| Mimo-Embodied [29] | 8.85 | 3.25 | **81.00** | 0.00 | 78.35 | 1.85 | 1.25 |
| **Our Model** | | | | | | | |
| Xiaomi-SpatialEmbodied (Best) | **78.50** | **4.35** | 76.00 | **3.80** | 87.15 | **3.50** | **2.30** |

## TODO

- [x] Release model code
- [x] Release model weights
- [x] Release inference code
- [ ] Release training code
- [x] Release evaluation code (`tools/eval_tools/`; the Table 3 embodied benchmarks that have no released scorer are marked in the table above)
- [ ] Release dataset download

## Finetuned Models

| Model | Base Model | Code | Weights |
|-------|-----------|------|---------|
| Xiaomi-SpatialEmbodied (main) | Qwen3-VL-30B-A3B-Instruct | `xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2/` | [ModelScope](https://www.modelscope.cn/models/xzhong/xembodied-v0) |

The two experimental connector variants under `xiaomi_spatialembodied/3DA/variants/` are ablation studies and have no released weights.

## Project Structure

```
BaseModel-open/
├── xiaomi_spatialembodied/
│   ├── 3DA/                                 # 3D Adapter (see xiaomi_spatialembodied/3DA/README.md)
│   │   ├── my_qwen3_vggt_xattnv2/           # ★ MAIN: Qwen3-VL-MoE-VGGT (3DA geometric connector)
│   │   │   ├── modeling_qwen3_vl_moe_vggt.py  # Core geometric-enhanced model definition
│   │   │   ├── configuration_qwen3_vl_moe.py
│   │   │   ├── modular_qwen3_vl_moe.py
│   │   │   ├── plugin/
│   │   │   │   ├── qwen3_vl_moe_vggt_register.py  # MS-Swift custom model registration
│   │   │   │   └── save_qwen3_vl_moe_vggt.py      # Merge VGGT weights into fused checkpoint
│   │   │   ├── scripts/                     # 4-stage SFT + inference script family
│   │   │   └── vggt/                        # VGGT backbone code
│   │   └── variants/                        # Experimental connector variants (not released)
│   │       ├── my_qwen3_moe_vl_vggt_xattn_mlp/    # Configurable fusion + deepstack connectors
│   │       └── my_qwen3_moe_vl_vggt_qformer/      # Q-Former main connector + deepstack MLP
│   ├── EIEA/                                # Inference side only; see xiaomi_spatialembodied/EIEA/README.md
│   │   ├── README.md                        # Scope, quick start, known limitations
│   │   ├── infer_with_EIEA_3D.py            # Loads the shipped TOR cache and runs inference
│   │   ├── models/
│   │   │   ├── qwen3_vl_vggt_register.py    # Registration with TOR physical embedding auto-injection
│   │   │   ├── qwen3_vl_moe_vggt.py         # Model definition with physical-geometric dual enhancement
│   │   │   └── ...
│   │   ├── tor_embeds_cache_fb.pt           # Pre-cached TOR embeddings (loaded by qwen3_vl_vggt_register.py)
│   │   └── tor_embeds_cache.pt              # Second, different TOR cache (see note below)
│   └── infer_demo.py                        # Basic VLM inference entry
├── demo/                                    # Synthetic sample image for the quick start
├── model_tools/                             # VGGT <-> Qwen grid-alignment debug tooling
├── requirements.txt / .gitignore
└── tools/                                   # Data processing & evaluation toolkit (see tools/README_TOOLS.md)
    ├── convert_data_tools/                 # Dataset format conversion scripts
    ├── parse_io_tools/                     # Raw dataset parsing (Parquet/Arrow/PKL)
    ├── infer_tools/                        # Inference helpers and output post-processing
    │   └── infer_demo_show.py              # Visual inference demo script
    ├── eval_tools/                         # Benchmark evaluation and score aggregation
    ├── vis/                                # Result visualization
    ├── data_statistics/                    # Data statistics, filtering, and sampling
    ├── build_model/                        # Model parameter counting and checkpoint utilities
    ├── reward_tools/                       # Reward model utilities for GRPO/RL training
    ├── trainer_tools/                      # Training debug and launch scripts
    └── planningQA/                         # Planning QA data preparation
```

The 4-stage training scripts live in `xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2/scripts/`
(see `xiaomi_spatialembodied/3DA/README.md` for what each one does).

**Documentation map:**

| Document | Covers |
|---|---|
| [`README.md`](README.md) | Overview, results, environment, inference quick start |
| [`xiaomi_spatialembodied/3DA/README.md`](xiaomi_spatialembodied/3DA/README.md) | 3DA layout, main model vs. variants, VGGT/Qwen grid alignment |
| [`xiaomi_spatialembodied/EIEA/README.md`](xiaomi_spatialembodied/EIEA/README.md) | EIEA scope, quick start, known limitations |
| [`tools/README_TOOLS.md`](tools/README_TOOLS.md) | Data conversion / evaluation / visualization scripts |

## Model Architecture

Xiaomi-SpatialEmbodied is a unified foundation model framework for autonomous driving and embodied intelligence, where the geometric enhancement backbone (3DA) and physical cue injection paradigm (EIEA) work in deep synergy. Both are built on Qwen3-VL-30B-A3B-Instruct:

### 1. 3DA: Geometric Spatial Enhancement Backbone

The 3D Adapter (3DA) is a geometric connector that injects representations from a 3D foundation model (VGGT) into the MLLM, aligning 3D structure with language reasoning so the VLM can understand 3D geometry without relying on external depth models. Unlike approaches that merely append depth or point clouds as auxiliary inputs, 3DA fuses the VGGT features directly into the native visual token stream.

Key components:

- **VGGTSpatialEncoder**: Loads the VGGT aggregator to extract multi-layer spatial features from images. Unnecessary heads (camera/point/depth/track) are removed to reduce memory consumption.
- **VGGTEmbeddingMerger**: Aligns and fuses 3D spatial tokens with native 2D visual tokens through interpolation, token merging, RMSNorm + MLP projection, and Cross-Attention, injecting 3D geometric cues into the vision-language model.

Registered with MS-Swift for one-stop fine-tuning and inference, serving as the foundational carrier of Xiaomi-SpatialEmbodied's geometric perception capability.

> **Note on token bookkeeping.** The released main model does **not** add any new tokens to the
> sequence: the connector resamples the VGGT patch grid onto Qwen3's image-token grid and adds a
> cross-attention residual back into the existing image embeddings. Only the experimental
> Q-Former variant (under `variants/`, not released) introduces dedicated 3D tokens.

#### VGGT / Qwen grid alignment

The two streams are resampled onto a common grid, so how the VGGT input is framed matters. The
released checkpoints were trained with the VGGT stream **letterboxed to a white 518×518 square**
while the Qwen stream keeps the image's aspect ratio, and the connector resamples with
**bilinear** interpolation. For non-square images the two grids therefore agree only near the
image centre. Both settings are configurable and default to the released behaviour:

```python
vggt_connector_config = {
    "letterbox": True,          # or export VGGT_LETTERBOX=0
    "interp_mode": "bilinear",  # or export VGGT_INTERP_MODE=nearest-exact
}
```

Changing either one invalidates weights trained with the other, so only do it when retraining.
See [`xiaomi_spatialembodied/3DA/README.md`](xiaomi_spatialembodied/3DA/README.md) for details.

#### Implementation Variants

`xiaomi_spatialembodied/3DA/` contains the released main model plus two experimental connector variants, kept for ablation comparison:

| Directory | MS-Swift `model_type` | Main connector | Deepstack connectors |
|---|---|---|---|
| `my_qwen3_vggt_xattnv2/` ★ **main, released** | `qwen3_vl_moe_vggt` | `VGGTEmbeddingMerger` — interpolate → token merge → RMSNorm+MLP → cross-attention residual (Q=image tokens, K/V=VGGT 3D tokens) | ✗ (Qwen3 deepstack features pass through; VGGT only enters the image-token stream) |
| `variants/my_qwen3_moe_vl_vggt_xattn_mlp/` | `qwen3_vl_moe_vggt_xattn_mlp` | Same merger, but `fusion_method` is configurable: `add` / `concat` / `gated` / `weighted` / `cross_attention` | ✓ one `VGGTEmbeddingMerger` per level, `deepstack_fusion_method="add"` |
| `variants/my_qwen3_moe_vl_vggt_qformer/` | `qwen3_vl_moe_vggt_qformer` | `VGGTQFormerConnector` — a language-initialized Q-Former selects K=8 geometry tokens and concatenates them before the native visual tokens | ✓ lightweight MLP `VGGTDeepstackConnector` per level, fused by element-wise add |

Only one variant can be mounted per `swift` run via `--custom_register_path`. See [`xiaomi_spatialembodied/3DA/README.md`](xiaomi_spatialembodied/3DA/README.md) for the full layout, per-variant scripts, and VGGT path resolution (`VGGT_REPO_DIR`).

### 2. EIEA: Efficient Image-Embodied Adapter

While prior work leverages embodied-specific tools (e.g., occupancy grids, 3D boxes, HD map cues) to explicitly guide VQA reasoning, such explicit tool invocation suffers from low efficiency and poor alignment between tool outputs and language reasoning processes. To address these limitations, EIEA distills raw tool outputs into compact token summaries via TOR (Textual Object Rationale) embeddings, which are then seamlessly reinserted into the MLLM context via `masked_scatter`, circumventing the interpretability burden on LLMs.

EIEA shares the model's underlying structure with 3DA, together forming Xiaomi-SpatialEmbodied's complete embodied perception capability.

> **Scope of this release**: the repository ships a pre-computed TOR embedding cache
> (`xiaomi_spatialembodied/EIEA/tor_embeds_cache*.pt`) together with the inference code that loads and
> injects it. **The EIEA pipeline that produces such a cache is not open-sourced at this time.**

## Environment Setup

### 1. Base Environment

```bash
conda create -n xiaomi_spatialembodied python=3.10
conda activate xiaomi_spatialembodied

# Install PyTorch
pip install torch>=2.4.0 torchvision>=0.19.0
```

### 2. MS-Swift (Training & Inference Framework)

```bash
pip install ms-swift
pip install qwen_vl_utils==0.0.14
pip install trl -U
pip install deepspeed
pip install transformers>=4.57.1
pip install decord
```

> ⚠️ **`transformers` must be ≥ 4.57.** Qwen3-VL (including the MoE variant) was introduced
> in transformers 4.57, and the model code imports
> `transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe`. The MS-Swift registration files
> declare the same constraint (`requires=["transformers>=4.57", ...]`).
>
> `deepspeed` is required by every script under `xiaomi_spatialembodied/3DA/*/scripts/` (they all pass
> `--deepspeed zero2|zero3`), and `decord` is needed to decode video samples.
>
> vLLM is optional and only used for the evaluation tooling. It ships its own torch build, so
> installing it here would silently override the pins above — use a separate environment.

See [`requirements.txt`](requirements.txt) for the full pinned list.

### 3. EIEA Dependencies (for EIEA inference)

```bash
pip install peft==0.14.0
pip install aenum sentencepiece protobuf
```

## Quick Start: Path Configuration

Before running any inference script, configure the paths to your local model checkpoints. The repository uses relative paths by default, so you only need to update the specified placeholder paths.

### Required Models & Weights

| Item | Default Location | Notes |
|------|------------------|-------|
| Qwen3-VL-30B-A3B-Instruct | `Qwen/Qwen3-VL-30B-A3B-Instruct` | Base model, supports auto-download |
| VGGT-1B weights (`model.pt`) | anywhere on disk; pass it to `save_qwen3_vl_moe_vggt.py` via `--vggt_ckpt_path` | Download from [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B). **Review the VGGT licence before redistributing any fused checkpoint.** |
| VGGT backbone code | directory that *contains* the `vggt/` package, set via `VGGT_REPO_DIR` | A copy ships at `xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2/vggt/`, so `VGGT_REPO_DIR=xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2` works. The built-in default does **not** resolve — always export it. |
| Trained model weights | Kaggle Model Scope | Full trainable weights: https://www.kaggle.com/models/zhongyangtony/xiaomi_spatialembodied/ |
| VGGT register file | `xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2/plugin/qwen3_vl_moe_vggt_register.py` | MS-Swift model registration file (main model; variants live in `xiaomi_spatialembodied/3DA/variants/`) |
| EIEA register file | `xiaomi_spatialembodied/EIEA/models/qwen3_vl_vggt_register.py` | Registration with physical embedding injection |
| TOR embedding cache | `xiaomi_spatialembodied/EIEA/tor_embeds_cache_fb.pt` | Pre-computed cache shipped with the repo. The EIEA pipeline that generates it is not open-sourced yet. |
| second TOR cache | `xiaomi_spatialembodied/EIEA/tor_embeds_cache.pt` | Different contents from `_fb.pt`. `qwen3_vl_vggt_register.py` loads **only** `_fb.pt` (`TOR_EMBEDS_PATH`, line 32); swap that constant if you want to run the other cache. |

### Setup Steps

1. **Download the base VLM**: Download [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) or let MS-Swift auto-download it.
2. **Download VGGT-1B** and export the two environment variables the code reads at load time:
   ```bash
   export VGGT_REPO_DIR=/path/to/xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2  # parent of the vggt/ package
   export VGGT_DEVICE=cuda:0     # default is cpu, which is unusable for a 30B model
   export VGGT_DTYPE=bfloat16    # default is float32
   ```
3. **Build a fused checkpoint** containing the VGGT weights (required — the model code never
   loads VGGT weights on its own):
   ```bash
   python xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2/plugin/save_qwen3_vl_moe_vggt.py \
     --base_model_dir  /path/to/Qwen3-VL-30B-A3B-Instruct \
     --vggt_ckpt_path  /path/to/VGGT-1B/model.pt \
     --vggt_repo_dir   /path/to/xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2 \
     --output_dir      xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2/base_ckpt
   ```
   Alternatively, point `CUSTOM_MODEL_CFG["model_path"]` in `xiaomi_spatialembodied/infer_demo.py` at the
   checkpoint you downloaded from Kaggle — that one is already fused.
4. **EIEA physical embedding**: The repository includes pre-cached TOR embedding files — no extra pretrained model downloads required.
5. **Training**: the 4-stage recipe is encoded in `xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2/scripts/sft_ddp_s{1..4}*.sh`; each stage's `--model` points at the previous stage's merged checkpoint.

## Inference

### Method 1: 3DA Geometric Enhancement Inference

Use `xiaomi_spatialembodied/infer_demo.py` to run Xiaomi-SpatialEmbodied's geometric enhancement backbone via MS-Swift.
The script ships with a synthetic sample image (`demo/sample.jpg`) so it runs without any
dataset download:

```bash
# From the repository root
export VGGT_REPO_DIR=$PWD/xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2   # parent of the vggt/ package
export VGGT_DEVICE=cuda:0
export VGGT_DTYPE=bfloat16

python xiaomi_spatialembodied/infer_demo.py
```

Script configuration:

```python
# Enable Xiaomi-SpatialEmbodied geometric enhancement model
USE_CUSTOM_MODEL = True

# Custom model config (use Kaggle-downloaded trained weights, or a locally fused base_ckpt)
CUSTOM_MODEL_CFG = {
    "model_path": "Path to the *fused* checkpoint (VGGT weights already merged in)",
    "model_type": "qwen3_vl_moe_vggt",
    "register_file": "<repo>/xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2/plugin/qwen3_vl_moe_vggt_register.py",
}

# Native model config
NATIVE_MODEL_CFG = {
    "model_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "model_type": "qwen3_vl_moe",
}

# Inference parameters
INFER_CFG = {
    "max_tokens": 512,
    "temperature": 0,
    "user_prompt": "Your prompt here",
    "image_paths": ["<image_path>"]
}
```

### Method 2: EIEA Physical-Geometric Dual Enhancement Inference

Use `xiaomi_spatialembodied/EIEA/infer_with_EIEA_3D.py` to load the shipped TOR cache and run inference with physical cue injection:

```bash
cd xiaomi_spatialembodied/EIEA
python infer_with_EIEA_3D.py
```

This script shows how to load the TOR embedding cache shipped with the repository and inject it, combining the physical branch with the VGGT geometric branch for dual-cue inference. No extra pretrained models need to be downloaded.

> ⚠️ The repository provides **one pre-computed TOR embedding cache** and the inference code that loads and injects it; **the EIEA pipeline that generates such a cache is not open-sourced at this time**. The demo therefore injects a **fixed soft prompt** rather than deriving physical priors per input image.

## Tools

See [tools/README_TOOLS.md](tools/README_TOOLS.md) for detailed documentation of each tool, including functionality descriptions and usage examples.

| Directory/Script | Description |
|-----------------|-------------|
| `convert_data_tools/` | Convert embodied datasets (NuScenes, OmniDrive, LingoQA, MapLM-v2, DriveBench, etc.) to MS-Swift training JSONL format |
| `parse_io_tools/` | Parse raw datasets (Parquet/Arrow/PKL) to structured JSON/JSONL format |
| `infer_tools/` | Inference helpers, output post-processing, and demo scripts |
| `eval_tools/` | Benchmark evaluation and score aggregation (18+ benchmarks) |
| `vis/` | Evaluation result and dataset sample visualization |
| `data_statistics/` | Data statistics, filtering, sampling, and quality checks |
| `build_model/` | Model parameter counting and checkpoint utilities |
| `reward_tools/` | Reward model utilities for GRPO/RL training |
| `trainer_tools/` | Training debug and launch scripts |
| `planningQA/` | Planning QA data preparation |

## Authors

Kangan Qian<sup>\*</sup>, ChuChu Xie<sup>\*</sup>, Yang Zhong<sup>✉,\*</sup>, Jingrui Pang, Siwen Jiao, Sicong Jiang, Zilin Huang, Yunlong Wang, Kun Jiang<sup>†</sup>, Mengmeng Yang, Hao Ye<sup>✉,†</sup>, Guanghao Zhang, Hangjun Ye, Guang Chen, Long Chen, Diange Yang<sup>†</sup>

**Affiliations:** Automotive and Robotics, Xiaomi Corporation &nbsp; Tsinghua University &nbsp; National University of Singapore &nbsp; McGill University &nbsp; University of Wisconsin-Madison

✉ Project Leader &nbsp; \* Equal contribution &nbsp; † Corresponding author

## Citation

If this project is useful in your work, we'd appreciate a citation:

```bibtex
@misc{qian2026xembodiedfoundationmodelenhanced,
      title={XEmbodied: A Foundation Model with Enhanced Geometric and Physical Cues for Large-Scale Embodied Environments},
      author={Kangan Qian and ChuChu Xie and Yang Zhong and Jingrui Pang and Siwen Jiao and Sicong Jiang and Zilin Huang and Yunlong Wang and Kun Jiang and Mengmeng Yang and Hao Ye and Guanghao Zhang and Hangjun Ye and Guang Chen and Long Chen and Diange Yang},
      year={2026},
      eprint={2604.18484},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.18484},
}
```
