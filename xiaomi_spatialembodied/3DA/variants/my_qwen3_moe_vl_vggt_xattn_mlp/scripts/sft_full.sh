#!/usr/bin/env bash
# UPDATE: Replace the /path/to/... placeholders below with your actual local paths.
#
# Minimal single-file demo for the configurable-fusion + deepstack variant.
#
# ⚠️ `--model` MUST point to a checkpoint that already contains the VGGT weights
#    (i.e. the output of `plugin/save_qwen3_vl_moe_vggt.py`), NOT to a bare
#    Qwen3-VL base model. This variant's `modeling_*.py` constructs `VGGT(...)`
#    without loading any weights, so pointing `--model` at a bare base model would
#    silently train against a randomly-initialised, permanently frozen 3D branch.
#    The loss still goes down (the connector learns to suppress the noise), so the
#    mistake is invisible in the logs.
#
# ⚠️ Also export `VGGT_REPO_DIR` (the directory that CONTAINS the `vggt/` package)
#    and `VGGT_DEVICE` / `VGGT_DTYPE`; the defaults are `cpu` / `float32`.
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && { echo "[ERROR] SLURM_JOB_ID set but CUDA_VISIBLE_DEVICES empty." 1>&2; exit 1; }
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
fi
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_PORT="${MASTER_PORT:-29520}"
export VGGT_REPO_DIR="${VGGT_REPO_DIR:-/path/to/vggt-main}"
export VGGT_DEVICE="${VGGT_DEVICE:-cuda:0}"
export VGGT_DTYPE="${VGGT_DTYPE:-bfloat16}"
export SWIFT_GROUP_BY_DATA_TYPE="${SWIFT_GROUP_BY_DATA_TYPE:-1}"

swift sft \
  --model "${BASE_CKPT:-/path/to/checkpoints/qwen3_moe_vggt_xattn_mlp_base_ckpt}" \
  --model_type qwen3_vl_moe_vggt_xattn_mlp \
  --custom_register_path xiaomi_spatialembodied/3DA/variants/my_qwen3_moe_vl_vggt_xattn_mlp/plugin/qwen3_vl_moe_vggt_register.py \
  --dataset "${DATASET:-/path/to/data/omnidrive_vqa_val_ms_swift.jsonl}#5" \
  --train_type full \
  --freeze_parameters_ratio 1.0 \
  --trainable_parameters connector deepstack_connectors \
  --gradient_checkpointing true \
  --torch_dtype bfloat16 \
  --remove_unused_columns false \
  --learning_rate 1e-4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1 \
  --save_steps 9999 \
  --save_total_limit 1 \
  --max_steps 10 \
  --max_length 16096 \
  --output_dir "${OUTPUT_DIR:-/path/to/workspace/sft_checkpoint/qwen3_moe_vggt_xattn_mlp/test_zero3}" \
  --run_name test_zero3_moe_xattn_mlp \
  --deepspeed zero3 \
  --use_logits_to_keep true
