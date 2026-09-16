#!/bin/bash
# UPDATE: Replace placeholder paths with your actual paths.
export PYTHONPATH="./ms-swift:${PYTHONPATH}"
export VGGT_REPO_DIR="./vggt"
export VGGT_DEVICE=cpu
export VGGT_DTYPE=float32

python /path/to/tools/build_model/count_model_params.py \
  --model /CKPT \
  --register_path /register_path \
  --model_type qwen3_vl_vggt_fuse2d_all \
  --device_map cpu \
  --depth 2
