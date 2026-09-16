# UPDATE: Replace placeholder paths with your actual paths.
python plugin/save_qwen3_vl_moe_vggt.py \
    --base_model_dir checkpoints/Qwen3-VL-30B-A3B-Instruct \
     --output_dir checkpoints/base_ckpt \
    --vggt_ckpt_path checkpoints/vggt/model.pt \
    --vggt_repo_dir "${VGGT_REPO_DIR:-.}" \
  --device_map cuda:0 \
  --torch_dtype bfloat16 \
  --max_shard_size 4GB \
  --safe_serialization 1
