# Tools Directory

Utility scripts for data conversion, evaluation, inference, visualization, and model management in the Xiaomi-SpatialEmbodied project.

> **Note:** All scripts use relative or placeholder paths by default. If a script contains a `# UPDATE:` comment at the top, replace the placeholder paths with your actual local paths before running.
>
> **Scope:** these are **one-off ETL / debugging scripts written against external datasets and
> benchmark repositories**. Nothing under `tools/` imports anything from `xiaomi_spatialembodied/`, and the
> `data/`, `checkpoints/` and `results/` directories they reference are **not shipped with this
> repository** — bring your own data, and expect to adjust paths. Some scripts are historical and
> target dataset versions or helper repositories that are not part of this release; where that is
> the case it is called out below.
>
> ⚠️ **Some scripts rewrite their input in place.** Read a script before running it on data you
> care about. See the per-script notes and `# UPDATE:` headers.

### External (non-PyPI) dependencies

Several scripts import helpers that ship with the **benchmark repositories**, not with
Xiaomi-SpatialEmbodied. They are listed in `requirements.txt` under "NOT on PyPI", and each import raises a
message naming the repository to clone:

| Import | Comes from | Used by |
|---|---|---|
| `strideqa_bench` | STRIDE-QA-Bench | `convert_strideqa_bench.py`, `convert_strideqa_mini.py`, `benchmark_sv2.py` |
| `evalcap` | EvalCap / RAG-Driver | `eval_BDD-X.py`, `eval_omnidrive.py` |
| `utils` (Ego3D) | Ego3D-Bench | `eval_ego3dbenchr1.py` |
| `utilsv2` | MAPLM | `eval_maplmv2.py` |
| `evaluate_utils_v1` | VLADBench | `eval_vladbenchv2.py` |
| `evaluate.*` | DriveBench toolkit | `eval_drivebench_withqwenv3.py` |
| `judge` | LingoQA | `eval_lingoQAv1.py` |
| `eval.eval_utils.preprocessor` | upstream eval harness | most `eval_tools/` scripts — falls back to an identity function **with a warning**; scores may then differ from the official benchmark |

---

## Directory Structure

```
tools/
├── build_model/          # Model building, parameter counting, and checkpoint utilities
├── convert_data_tools/   # Dataset format conversion scripts (to MS-Swift JSONL)
├── data_statistics/      # Data statistics, filtering, and sampling
├── eval_tools/           # Benchmark evaluation and score aggregation
├── infer_tools/          # Inference helpers and output post-processing
├── parse_io_tools/       # Raw dataset parsing (Parquet/Arrow/JSON to structured format)
├── planningQA/           # Planning QA data preparation
├── reward_tools/         # Reward model utilities (GRPO / RL)
├── trainer_tools/        # Training debug and launch scripts
└── vis/                  # Visualization tools for evaluation results and datasets
```

---

## build_model

Model parameter counting and checkpoint utilities for MS-Swift registered models.

| Script | Description |
|--------|-------------|
| `count_model_params.py` | Count total / trainable / per-prefix parameters for MS-Swift registered Qwen3-VL + VGGT models. Uses `device_map="meta"` to avoid GPU memory allocation. |
| `count_model_params_demo.sh` | Shell wrapper for `count_model_params.py`. |

### Usage Example

```bash
python count_model_params.py \
    --model /path/to/hf-style-checkpoint \
    --register_path /path/to/plugin/qwen3_vl_moe_vggt_register.py \
    --model_type qwen3_vl_vggt_fuse2d_all
```

---

## convert_data_tools

Convert various embodied / driving VQA datasets into the MS-Swift training JSONL format.

| Script | Target Dataset |
|--------|---------------|
| `convert_maplmv2.py` | MapLM-v2 (driving map understanding) |
| `batch_convert_maplmv2.py` | MapLM-v2 batch conversion |
| `conver_maplmv2_.py` | MapLM-v2 alternative converter |
| `convert_omnidrive.py` | OmniDrive (multi-view driving VQA) |
| `convert_lingoqa_val.py` | LingoQA validation set |
| `convert_drivebench_val.py` | DriveBench validation (corruption evaluation) |
| `convert_drivebench_valv1.py` | DriveBench validation v1 |
| `convert_drivelmm_test.py` | DriveLMM-o1 test set |
| `convert_blink_val.py` | BLINK visual relation benchmark |
| `convert_cosmos_bench.py` | Cosmos-R1 benchmark |
| `convert_cosmos_rl.py` | Cosmos-R1 RL data |
| `convert_cosmos_sft.py` | Cosmos-R1 SFT data |
| `convert_ego3d_bench.py` | Ego3D-Bench |
| `convert_ego3d_benchv1.py` | Ego3D-Bench v1 |
| `convert_embodiedr1.py` | Embodied-R1 |
| `convert_nuscenes_planning.py` | NuScenes planning QA |
| `convert_partafford.py` | Part Affordance dataset |
| `convert_roboafford.py` | Robo-Affordance dataset |
| `convert_roborefit.py` | RoboRefit dataset |
| `convert_roborefit_bench.py` | RoboRefit benchmark |
| `convert_strideqa_bench.py` | STRIDE-QA benchmark |
| `convert_strideqa_mini.py` | STRIDE-QA mini subset |
| `convert_vabench_point_bbox.py` | VABench point & bbox data |
| `convert_vabench_visual_trace.py` | VABench visual trace data |
| `convert_vladbench.py` | VLADBench |
| `convert_vsi-bench.py` | VSI-Bench |
| `convert_vsi.py` | VSI dataset |
| `convert_where2place.py` | Where2Place dataset |
| `convert2r1_type_data.py` | Convert standard SFT data to R1-type (with `<think>` tags) |
| `convert2fix_vladbench_val.py` | Fix VLADBench validation format |
| `batch_convert_vabench_point_bbox.py` | Batch VABench conversion |

### Usage Example

```bash
python convert_maplmv2.py \
    --input_file data/maplm_v2/data/train_v2.json \
    --output_dir data/preprocess_data/MapLMv2 \
    --data_split train \
    --image_base_path data/maplm_v2/data/images/
```

---

## data_statistics

Data quality checks, filtering, sampling, and debugging utilities.

| Script | Description |
|--------|-------------|
| `check_multimodal_data_exists.py` | Validate that image/video/audio files referenced in JSONL data exist on disk; count single-image, multi-image, and video QA samples. |
| `data_statistics_val.py` | Sample 20% of training data to create a validation split. |
| `debug_data_processor.py` | Debug tool for data processing pipeline. |
| `filter_partafford.py` | Filter Part Affordance dataset samples. |
| `filter_rl_data.py` | Filter RL training data. |
| `filter_where2place.py` | Filter Where2Place dataset samples. |
| `find_drivelmm.py` | Find specific patterns in DriveLMM data. |
| `find_drivelmm_no_answer_tag.py` | Find DriveLMM samples missing `<answer>` tags. |
| `fix_drivelmm.py` | Fix DriveLMM answer format issues. |
| `fix_drivelmm_option.py` | Fix DriveLMM option format issues. |
| `rl_data_sample_debug.py` | Debug RL data sampling. |

---

## eval_tools

Benchmark evaluation scripts that compute metrics from model inference outputs.

| Script | Benchmark | Key Metrics |
|--------|-----------|-------------|
| `benchmark_sv2.py` | SURDS / STRIDE-QA (unified) | Localization accuracy, direction/distance scoring |
| `eval_BDD-X.py` | BDD-X | Action description accuracy |
| `eval_DriveLMMo1v2.py` | DriveLMM-o1 v2 | MCQ accuracy, BLEU, ROUGE-L |
| `eval_drivingvqa_v1.py` | DrivingVQA v1 | Image-filename matched evaluation |
| `eval_blink_v0.py` | BLINK | Binary choice accuracy |
| `eval_cosmos.py` | Cosmos-R1 | Yes/no semantic accuracy, F1 |
| `eval_drivebench_withqwenv3.py` | DriveBench | Dual-layer weighted score (MCQ 0.6 + VQA 0.4), Qwen LLM-as-judge scoring |
| `eval_ego3dbenchr1.py` | Ego3D-Bench | ACC, RMSE per task category |
| `eval_lingoQAv1.py` | LingoQA | GPT-based scoring |
| `eval_maplmv2.py` | MapLM-v2 | BLEU score |
| `eval_omnidrive.py` | OmniDrive | CIDEr, BLEU, ROUGE-L |
| `eval_robot_grounding.py` | Robot Grounding | IoU, bbox accuracy |
| `eval_SURDSv2.py` | SURDS v2 | Point-in-bbox, yaw/depth/direction accuracy |
| `eval_vladbenchv2.py` | VLADBench v2 | Multi-task localization metrics |
| `drivebench_aggregrator.py` | DriveBench | Aggregate results across distortion modes |
| `unified_benchmark_system_prompt.py` | All benchmarks | Generate unified system prompts for evaluation |

### Usage Example

```bash
# Evaluate on SURDS benchmark
python eval_SURDSv2.py \
    --infer_file_path data/infer_results/surds/output.jsonl \
    --dataset_path data/preprocess_data/SURDS/surds_val.json \
    --output_dir data/eval_results/SURDS

# Evaluate on DriveBench
python eval_drivebench_withqwenv3.py \
    --infer_file_path data/infer_results/drivebench/output.jsonl \
    --dataset_path data/preprocess_data/DriveBench/meta.jsonl
```

---

## infer_tools

Inference helpers, output post-processing, and demo scripts.

| Script | Description |
|--------|-------------|
| `infer_demo_show.py` | Demo script: run native Qwen3-VL or custom Xiaomi-SpatialEmbodied model inference and print results. |
| `infer_qwen_235B.py` | Inference via SGLang server API for Qwen3-VL models. Encodes images as base64 and sends chat completion requests. |
| `infer_lingoQA.sh` | Shell wrapper for MS-Swift inference on the LingoQA dataset. |
| `pt_debug.sh` | Shell wrapper for MS-Swift debug inference. |
| `filter_think_format.py` | Remove `<think>` reasoning sections from R1-type model outputs; extract final answers. |
| `option_parser.py` | Parse model completions into choice indices for MCQ evaluation. Supports `<answer>` tag extraction and fuzzy matching. |
| `box_match.py` | Regex patterns for matching bbox coordinate formats like `(x1,y1),(x2,y2)` in `<answer>` tags. |

### Usage Example

```bash
# Run inference demo
python infer_demo_show.py

# Inference via SGLang server
python infer_qwen_235B.py \
    --input_file data/infer_input/samples.jsonl \
    --output_file results/output.jsonl \
    --server_url http://localhost:8000/v1/chat/completions

# Filter think tags from R1 outputs
from filter_think_format import preprocess_model_output
clean_text = preprocess_model_output(model_output)
```

---

## parse_io_tools

Parse raw datasets (Parquet, Arrow, PKL formats) into structured JSON/JSONL for downstream conversion.

| Script | Source Format | Description |
|--------|--------------|-------------|
| `parse_ego3d_bench.py` | Arrow IPC | Parse Ego3D-Bench test data to JSON |
| `parse_embodiedr1.py` | Parquet | Parse Embodied-R1 point data with image extraction |
| `parse_embodiedr1_visual.py` | Parquet | Parse Embodied-R1 visual trace data with image extraction |
| `parse_partafford.py` | Parquet | Parse Part-Affordance-2K training data |
| `parse_roborefit.py` | Parquet | Parse RoboRefit dataset |
| `parse_roborefit_bench.py` | Parquet | Parse RoboRefit benchmark data |
| `parse_vabench.py` | Parquet | Parse VABench point-bbox data |
| `read_pkl.py` | PKL | Read and inspect NuScenes ego info PKL files |

### Usage Example

```bash
# Parse Ego3D-Bench data
python parse_ego3d_bench.py

# Parse Part Affordance data
python parse_partafford.py
```

---

## planningQA

Data preparation tools for planning QA tasks.

| Script | Description |
|--------|-------------|
| `navsim2json_navtrain_only_traj.py` | Convert NavSim navigation training data to JSON format with trajectory analysis (speed, acceleration, steering). |

---

## reward_tools

Utilities for reward model debugging and content extraction in GRPO / RL training.

| Script | Description |
|--------|-------------|
| `extract_content.py` | Extract answers from model responses using `<answer>` tag parsing, `Answer(s):` patterns, and multiple regex fallback strategies. |

### Usage Example

```python
from extract_content import _extract_answer
answer = _extract_answer("The result is <answer>B</answer>")
```

---

## trainer_tools

Training debug and launch scripts for MS-Swift based fine-tuning.

| Script | Description |
|--------|-------------|
| `sft_debug.py` | Supervised fine-tuning debug script using MS-Swift with LoRA configuration. |
| `sft_debug.sh` | Shell wrapper for launching SFT debug training. |

### Usage Example

```bash
# Launch SFT debug training
bash sft_debug.sh
```

---

## vis

Visualization tools for evaluation results and dataset samples.

| Script | Description |
|--------|-------------|
| `vis_embodied.py` | Visualize bounding boxes and points from Embodied-R1 data with sampling support. |
| `vis_eval_result.py` | Aggregate and visualize evaluation results from JSON score files. |
| `vis_partafford.py` | Visualize Part-Affordance-2K data with category-based color coding. |
| `vis_roboafford.py` | Visualize Robo-Affordance dataset samples. |
| `vis_roborefit.py` | Visualize RoboRefit dataset samples. |
| `vis_roborefit_bench.py` | Visualize RoboRefit benchmark samples. |
| `vis_vabench.py` | Visualize VABench pointing and bounding box data. |
| `vis_where2place.py` | Visualize Where2Place dataset samples. |

### Usage Example

```bash
# Visualize evaluation results  (note: -i/--input and -o/--output)
python vis_eval_result.py -i data/eval_results -o data/vis

# Visualize Part Affordance data.
# NOTE: vis_partafford.py has no CLI -- edit the paths at the bottom of the file
# (`if __name__ == "__main__":`) and run `python vis_partafford.py`.
```
