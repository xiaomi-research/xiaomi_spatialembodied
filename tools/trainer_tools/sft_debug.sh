#!/bin/bash
# UPDATE: Replace placeholder paths with your actual model/dataset/output paths.
#
# NOTE: the two registration files below (`debug_dataset.py`, `diy_model.py`) are MS-Swift
# registration *examples* that are NOT shipped with this repository -- they were part of an
# upstream workspace. Point these paths at your own registration files, or drop the
# `--custom_register_path` argument entirely if you are not registering custom code.
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift sft \
    --model '/path/to/your/model' \
    --dataset '/path/to/your/dataset.json' \
    --remove_unused_columns false \
    --custom_register_path /path/to/custom/debug_dataset.py \
                            /path/to/custom/diy_model.py \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 16 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir /path/to/output/trainer_debug \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4
