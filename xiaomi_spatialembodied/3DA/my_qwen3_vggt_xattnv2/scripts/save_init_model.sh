# UPDATE: Replace placeholder paths with your actual paths.
#!/usr/bin/env bash

set -euo pipefail
#  --base_model_dir  checkpoints/Qwen3-VL-30B-A3B-Instruct \
python plugin/save_qwen3_vl_moe_vggt.py \
    --base_model_dir checkpoints/qwen3_v1_public_0202/v0-20260202-182912/checkpoint-1335 \
    --output_dir checkpoints/base_ckpt1 \
    --vggt_ckpt_path checkpoints/vggt/model.pt \
    --vggt_repo_dir "${VGGT_REPO_DIR:-.}" \
  --device_map cuda:0 \
  --torch_dtype bfloat16 \
  --max_shard_size 4GB \
  --safe_serialization 1
