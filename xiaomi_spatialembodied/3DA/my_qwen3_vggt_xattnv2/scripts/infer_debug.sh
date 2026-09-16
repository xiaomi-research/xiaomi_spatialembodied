#!/bin/bash
# Inference debug script using MS-Swift
# UPDATE: Replace placeholder paths with your actual checkpoint/dataset/result paths.
export VGGT_DTYPE="${VGGT_DTYPE:-bfloat16}"
CUDA_VISIBLE_DEVICES=0

swift infer \
  --model checkpoints/qwen3_vl_moe_vggt_xattn_merged \
  --model_type qwen3_vl_moe_vggt \
  --external_plugins xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2/plugin/qwen3_vl_moe_vggt_register.py \
  --infer_backend transformers \
  --torch_dtype bfloat16 \
  --val_dataset data/cosmos_r1_robofail.jsonl#10 \
  --max_batch_size 1 \
  --max_new_tokens 2048 \
  --temperature 0 \
  --result_path results/qwen3_moe_vggt_xattn_infer.jsonl
