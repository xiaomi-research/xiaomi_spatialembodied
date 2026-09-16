#!/bin/bash
# Inference on LingoQA dataset using MS-Swift
# UPDATE: Replace placeholder paths with your actual model/dataset/result paths.
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 2048 \
    --val_dataset data/lingoqa_swift_val.jsonl \
    --max_batch_size 2 \
    --result_path results/lingoQA/Qwen2_5_VL-3B/val.jsonl
