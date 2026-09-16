# UPDATE: Replace placeholder paths with your actual paths.
NPROC_PER_NODE=$RESOURCE_GPU
nnodes=$WORLD_SIZE
NNODES=$WORLD_SIZE
NODE_RANK=$RANK
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT
export VGGT_DTYPE="${VGGT_DTYPE:-bfloat16}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"


swift infer \
  --model checkpoints/fm_qwen3_vl_moe_vggt_xattn_full/v1-20260217-010636/checkpoint-29500 \
  --model_type qwen3_vl_moe_vggt \
  --external_plugins plugin/qwen3_vl_moe_vggt_register.py \
  --infer_backend transformers \
  --torch_dtype bfloat16 \
  --val_dataset data/all_eval_data/cosmos_r1/cosmos_r1_robofail.jsonl  \
  --max_batch_size 1 \
  --max_new_tokens 2048 \
  --temperature 0 \
  --result_path results/infer_results/qwen3_moe_vggt_xattn_infer.jsonl

# checkpoints/qwen3_v1_public_0202/v0-20260202-182912/checkpoint-1335
# checkpoints/fm_qwen3_vl_moe_vggt_mlp_full/v0-20260217-022917/checkpoint-26656
