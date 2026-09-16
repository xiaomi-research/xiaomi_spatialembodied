#!/bin/bash
# Debug inference script using MS-Swift
# UPDATE: Replace placeholder paths with your actual model/dataset/result paths.
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model 'checkpoints/diy_model' \
    --custom_register_path xiaomi_spatialembodied/3DA/my_qwen3_vggt_xattnv2/plugin/qwen3_vl_moe_vggt_register.py \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 2048 \
    --val_dataset 'data/bdd100k_labels_images_train_weather_qwen_v1.json' \
    --max_batch_size 2 \
    --result_path results/infer_debug
