# UPDATE: Replace placeholder paths with your actual paths.
import argparse
from swift.llm import sft_main, TrainArguments, BaseArguments

# Create argument parser
parser = argparse.ArgumentParser(description='SFT Training Script')

# Define model path argument
parser.add_argument('--model', type=str,
                    default='checkpoints/Qwen2.5-0.5B-Instruct',
                    help='Name or path of the pre-trained model')

# Define dataset path argument
parser.add_argument('--dataset', type=str,
                    default='data/bdd100k_labels_images_train_weather_qwen_v1.json',
                    help='Path to training dataset file')

# Define output directory argument
parser.add_argument('--output_dir', type=str,
                    default='results/trainer_debug',
                    help='Directory for saving model checkpoints')

# Parse command-line arguments
args = parser.parse_args()

# Execute supervised fine-tuning
result = sft_main(
    TrainArguments(
        model=args.model,
        custom_register_path=['checkpoints/debug_dataset.py'],  # Optional custom module paths
        # loss_type='mse',         # Custom loss type (MSE, etc)
        train_type='lora',          # Training method (lora/full/etc)
        dataset=[args.dataset],     # Dataset path
        torch_dtype='bfloat16',     # Tensor data type

        # Training hyperparameters
        num_train_epochs=20,          # Total training epochs
        per_device_train_batch_size=1, # Batch size per device during training
        per_device_eval_batch_size=1,  # Batch size per device during evaluation
        learning_rate=1e-4,            # Initial learning rate

        # LoRA-specific parameters
        lora_rank=8,              # Rank for LoRA approximation
        lora_alpha=32,            # Scaling factor for LoRA weights
        target_modules='all-linear', # Modules to apply LoRA (all linear layers)

        # Vision model configuration
        freeze_vit=True,           # Freeze vision transformer weights

        # Training optimization
        gradient_accumulation_steps=16, # Steps for gradient accumulation

        # Checkpointing and logging
        eval_steps=500,            # Evaluation interval (in steps)
        save_steps=500,            # Checkpoint saving interval (in steps)
        save_total_limit=5,        # Maximum number of checkpoints to keep
        logging_steps=5,           # Logging interval (in steps)

        # Resource management
        max_length=12800,          # Maximum sequence length
        output_dir=args.output_dir, # Output directory for results
        warmup_ratio=0.05,         # Warmup period ratio
        dataloader_num_workers=4,  # Number of data loader worker threads
        dataset_num_proc=4,        # Number of processes for dataset processing
    )
)
