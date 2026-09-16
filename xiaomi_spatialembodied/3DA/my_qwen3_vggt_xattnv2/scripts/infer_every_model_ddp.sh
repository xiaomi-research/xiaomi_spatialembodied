# UPDATE: Replace placeholder paths with your actual paths.
# <repo-root>/ms-swift-main/examples/custom/my_qwen3_vggt_mlp/scripts/infer_every_model_ddp.sh
#!/bin/bash
# Batch inference script: pass model name and path, batch execute inference on different datasets
# Usage: ./batch_infer.sh <model_name> <model_path>
# Example: ./batch_infer.sh qwen2.5-7b-instruct /path/to/qwen2.5-7b-instruct

# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=TRACE
NPROC_PER_NODE=$RESOURCE_GPU
nnodes=$WORLD_SIZE
NNODES=$WORLD_SIZE
NODE_RANK=$RANK
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT
export VGGT_DTYPE="${VGGT_DTYPE:-bfloat16}"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export SWIFT_SINGLE_DEVICE_MODE=1
# export DISABLE_MP_DDP=1
export CLI_TYPE="VAL"

# Parameter check
if [ $# -lt 2 ]; then
    echo "Error: Please provide model name and path parameters"
    echo "Usage: $0 <MODEL_NAME> <MODEL_PATH> [MODEL_TYPE]"
    echo "Example: $0 qwen2.5-7b-instruct /path/to/model"
    echo "Example: $0 Qwen2-VL-UniVG-R1 /path/to/model"
    echo "Example: $0 custom-model /path/to/model qwen2_vl"
    echo ""
    echo "If MODEL_TYPE is not specified, it will be automatically inferred from the model name"
    exit 1
fi

# Parameter settings
MODEL_NAME=$1
MODEL_PATH=$2
MODEL_TYPE=$3  # Optional third parameter, manually specify model type

echo "Model name: $MODEL_NAME"
echo "Model path: $MODEL_PATH"

# Model type auto-inference function
infer_model_type() {
    local model_name=$1

    # Convert to lowercase for comparison
    local lowercase_name=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')

    # Model type mapping rules
    # Format: model name keyword -> model type
    declare -A MODEL_TYPE_MAP=(
        # Qwen2-VL series
        ["qwen2.5-vl"]="qwen2_5_vl"
        ["qwen2-vl"]="qwen2_vl"
        ["qwen-vl"]="qwen_vl"
        ["qwen2vl"]="qwen2_vl"
        ["qwen2_5_vl"]="qwen2_5_vl"

        # UniVG series (based on Qwen2-VL)
        ["univg"]="qwen2_vl"
        ["univg-r1"]="qwen2_vl"
        ["univg-r2"]="qwen2_vl"

        # Qwen2.5 series
        ["qwen2.5"]="qwen2_5"
        ["qwen2_5"]="qwen2_5"

        # Qwen2 series
        ["qwen2"]="qwen2"

        # Other vision-language models
        ["llava"]="llava"
        ["llava-next"]="llava_next"
        ["llavanext"]="llava_next"
        ["llama-vision"]="llama_vision"
        ["minigpt"]="minigpt"
        ["instructblip"]="instructblip"
        ["blip2"]="blip2"

        # Pure language models
        ["llama"]="llama"
        ["baichuan"]="baichuan"
        ["chatglm"]="chatglm"
        ["yi"]="yi"
        ["internlm"]="internlm"
        ["deepseek"]="deepseek"
    )

    # Special model name matching
    declare -A SPECIAL_MODEL_MAP=(
        # Format: full or partial model name -> model type
        ["qwen2-vl-univg-r1"]="qwen2_vl"
        ["qwen2vl-univg-r1"]="qwen2_vl"
        ["univg-r1"]="qwen2_vl"
        ["qwen2.5-vl-7b-instruct"]="qwen2_5_vl"
        ["qwen2-vl-7b-instruct"]="qwen2_vl"
        ["qwen-vl-7b-instruct"]="qwen_vl"
        ["qwen2.5-7b-instruct"]="qwen2_5"
        ["qwen2-7b-instruct"]="qwen2"
    )

    # First check special mapping
    for key in "${!SPECIAL_MODEL_MAP[@]}"; do
        if [[ "$lowercase_name" == *"$key"* ]]; then
            echo "${SPECIAL_MODEL_MAP[$key]}"
            return 0
        fi
    done

    # Then check general mapping
    for key in "${!MODEL_TYPE_MAP[@]}"; do
        if [[ "$lowercase_name" == *"$key"* ]]; then
            echo "${MODEL_TYPE_MAP[$key]}"
            return 0
        fi
    done

    # If no match found, return default value or prompt
    echo "unknown"
    return 1
}

# If user did not specify model type, auto-infer
if [ -z "$MODEL_TYPE" ]; then
    echo "Model type not specified, auto-inferring from model name..."
    INFERRED_TYPE=$(infer_model_type "$MODEL_NAME")

    if [ "$INFERRED_TYPE" = "unknown" ]; then
        echo "Warning: Cannot infer model type from model name: $MODEL_NAME"
        echo "Please manually specify model type as the third parameter:"
        echo "Example: $0 $MODEL_NAME $MODEL_PATH qwen2_vl"
        echo ""
        echo "Common model types:"
        echo "  - qwen2_5_vl (Qwen2.5-VL series)"
        echo "  - qwen2_vl (Qwen2-VL series)"
        echo "  - qwen_vl (Qwen-VL series)"
        echo "  - qwen2_5 (Qwen2.5 series)"
        echo "  - qwen2 (Qwen2 series)"
        echo "  - llava (LLaVA series)"
        echo "  - llama (LLaMA series)"
        echo ""
        read -p "Please enter model type (or press Enter to use default qwen2_vl): " MODEL_TYPE
        MODEL_TYPE=${MODEL_TYPE:-"qwen2_vl"}
        echo "Using model type: $MODEL_TYPE"
    else
        MODEL_TYPE=$INFERRED_TYPE
        echo "Inferred model type: $MODEL_TYPE"
    fi
else
    echo "Specified model type: $MODEL_TYPE"
fi

# Validate model type
VALID_TYPES=("qwen2_5_vl" "qwen2_vl" "qwen_vl" "qwen2_5" "qwen2" "llava" "llava_next" "llama_vision" "minigpt" "instructblip" "blip2" "llama" "baichuan" "chatglm" "yi" "internlm" "deepseek")

if [[ ! " ${VALID_TYPES[@]} " =~ " ${MODEL_TYPE} " ]]; then
    echo "Warning: Model type '$MODEL_TYPE' is not in the known valid types list, but will continue using it."
    echo "Known valid types: ${VALID_TYPES[*]}"
fi

BASE_EVAL_PATH=data/all_eval_data
# Dataset configuration array
# Format: dataset name, dataset path
declare -a DATASETS=(
    "cosmos_r1 ${BASE_EVAL_PATH}/cosmos_r1"
    "drivebench ${BASE_EVAL_PATH}/drivebench"
    "drivelmm ${BASE_EVAL_PATH}/drivelmm"
    "drivingvqa ${BASE_EVAL_PATH}/drivingvqa"
    "ego3d-bench ${BASE_EVAL_PATH}/ego3d-bench"
    "embodied-r1 ${BASE_EVAL_PATH}/embodied-r1"
    "lingoqa ${BASE_EVAL_PATH}/lingoqa"
    "maplmv2 ${BASE_EVAL_PATH}/maplmv2"
    "omnidrive ${BASE_EVAL_PATH}/omnidrive"
    "Part-Affordance-2K ${BASE_EVAL_PATH}/Part-Affordance-2K"
    "roborefit-benchmark-dataset ${BASE_EVAL_PATH}/roborefit-benchmark-dataset"
    "strideqa ${BASE_EVAL_PATH}/strideqa"
    "SURDS ${BASE_EVAL_PATH}/SURDS"
    "vabench-point-bbox ${BASE_EVAL_PATH}/vabench-point-bbox"
    "vabench-visual-trace ${BASE_EVAL_PATH}/vabench-visual-trace"
    "vladbench ${BASE_EVAL_PATH}/vladbench"
    "where2place ${BASE_EVAL_PATH}/where2place"
    "RoboAfford ${BASE_EVAL_PATH}/RoboAfford"
    # More datasets can be added here
)

# Log file
LOG_DIR="results/exp_logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/${MODEL_NAME}_infer_$(date +%Y%m%d_%H%M%S).log"

# Statistics variables
TOTAL_DATASETS=${#DATASETS[@]}
PROCESSED_DATASETS=0
SKIPPED_DATASETS=0
FAILED_DATASETS=0
PROCESSED_FILES=0
SKIPPED_FILES=0
FAILED_FILES=0
TOTAL_SAMPLES=0
SUCCESS_SAMPLES=0

# Record start time
START_SCRIPT_TIME=$(date +%s)
echo "=== Batch inference started: $(date) ===" | tee -a $LOG_FILE
echo "Model: $MODEL_NAME" | tee -a $LOG_FILE
echo "Model path: $MODEL_PATH" | tee -a $LOG_FILE
echo "Model type: $MODEL_TYPE" | tee -a $LOG_FILE
echo "Total datasets: $TOTAL_DATASETS" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Iterate over all datasets
for dataset_info in "${DATASETS[@]}"; do
    # Split dataset name and path
    DATASET_NAME=$(echo $dataset_info | cut -d' ' -f1)
    DATASET_PATH=$(echo $dataset_info | cut -d' ' -f2)

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing dataset: $DATASET_NAME" | tee -a $LOG_FILE
    echo "[Dataset path]: $DATASET_PATH" | tee -a $LOG_FILE

    # Check if dataset path exists
    if [ ! -d "$DATASET_PATH" ]; then
        echo "[Warning]: Dataset path does not exist: $DATASET_PATH" | tee -a $LOG_FILE
        echo "[Skip] $DATASET_NAME" | tee -a $LOG_FILE
        ((SKIPPED_DATASETS++))
        continue
    fi

    # Find all jsonl files under the dataset directory
    JSONL_FILES=()

    # Pattern 1: Exact match at beginning of dataset name
    PATTERN1="${DATASET_NAME}_*.jsonl"
    if compgen -G "$DATASET_PATH/$PATTERN1" > /dev/null; then
        JSONL_FILES+=($DATASET_PATH/$PATTERN1)
    fi

    # Pattern 2: Filename contains dataset name
    PATTERN2="*${DATASET_NAME}*.jsonl"
    if compgen -G "$DATASET_PATH/$PATTERN2" > /dev/null; then
        for file in $DATASET_PATH/$PATTERN2; do
            if [[ ! " ${JSONL_FILES[@]} " =~ " ${file} " ]]; then
                JSONL_FILES+=("$file")
            fi
        done
    fi

    # Pattern 3: All jsonl files (if nothing found above)
    if [ ${#JSONL_FILES[@]} -eq 0 ]; then
        PATTERN3="*.jsonl"
        if compgen -G "$DATASET_PATH/$PATTERN3" > /dev/null; then
            JSONL_FILES+=($DATASET_PATH/$PATTERN3)
        fi
    fi

    # Check if jsonl files were found
    if [ ${#JSONL_FILES[@]} -eq 0 ]; then
        echo "[Warning]: No jsonl files found in $DATASET_PATH" | tee -a $LOG_FILE
        echo "[Skip] $DATASET_NAME" | tee -a $LOG_FILE
        ((SKIPPED_DATASETS++))
        continue
    fi

    echo "[Found files]: ${#JSONL_FILES[@]} jsonl files" | tee -a $LOG_FILE

    # Create output directory for current dataset
    RESULT_BASE_PATH="results/infer_results/$DATASET_NAME/$MODEL_NAME"
    mkdir -p $RESULT_BASE_PATH

    # Iterate over each jsonl file for inference
    DATASET_FILES_PROCESSED=0
    DATASET_FILES_SKIPPED=0
    DATASET_FILES_FAILED=0
    DATASET_SAMPLES=0

    for JSONL_FILE in "${JSONL_FILES[@]}"; do
        # Get filename (without path and extension)
        FILENAME=$(basename "$JSONL_FILE" .jsonl)

        # Construct output result file path
        RESULT_FILE="$RESULT_BASE_PATH/${FILENAME}_result.jsonl"

        # Check if result file already exists
        if [ -f "$RESULT_FILE" ]; then
            RESULT_SIZE=$(wc -l < "$RESULT_FILE" 2>/dev/null || echo "0")
            INPUT_SIZE=$(wc -l < "$JSONL_FILE" 2>/dev/null || echo "0")

            if [ "$RESULT_SIZE" -eq "$INPUT_SIZE" ] && [ "$INPUT_SIZE" -gt 0 ]; then
                echo "[Skip] $(basename $JSONL_FILE) already completed" | tee -a $LOG_FILE
                echo "  [Line count match] Input: $INPUT_SIZE, Output: $RESULT_SIZE" | tee -a $LOG_FILE
                ((DATASET_FILES_SKIPPED++))
                ((SKIPPED_FILES++))
                ((SUCCESS_SAMPLES+=INPUT_SIZE))
                continue
            else
                echo "[Warning] $(basename $JSONL_FILE) result file exists but line count mismatch" | tee -a $LOG_FILE
                echo "  [Line count] Input: $INPUT_SIZE, Existing result: $RESULT_SIZE" | tee -a $LOG_FILE
            fi
        fi

        echo "[Start inference] $(basename $JSONL_FILE)" | tee -a $LOG_FILE
        echo "  [Output file] $RESULT_FILE" | tee -a $LOG_FILE

        # Record inference start time
        START_TIME=$(date +%s)

        # Create temporary log file for capturing errors
        TEMP_LOG=$(mktemp)

        # Execute inference command, only log error information
        swift infer \
            --model $MODEL_PATH \
            --external_plugins plugin/qwen3_vl_moe_vggt_register.py \
            --infer_backend transformers \
            --torch_dtype bfloat16 \
            --temperature 0 \
            --max_batch_size 1 \
            --max_new_tokens 2048 \
            --model_type $MODEL_TYPE \
            --val_dataset $JSONL_FILE \
            --result_path $RESULT_FILE 2>$TEMP_LOG

        INFER_EXIT_CODE=$?

        # Record inference end time and duration
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        # Check if inference succeeded
        if [ $INFER_EXIT_CODE -eq 0 ] && [ -f "$RESULT_FILE" ]; then
            echo "[Inference completed] $(basename $JSONL_FILE)" | tee -a $LOG_FILE
            echo "  [Duration] ${DURATION}s" | tee -a $LOG_FILE

            # Count inference result lines
            if command -v wc &> /dev/null; then
                RESULT_LINES=$(wc -l < "$RESULT_FILE" 2>/dev/null || echo "unknown")
                INPUT_LINES=$(wc -l < "$JSONL_FILE" 2>/dev/null || echo "unknown")
                echo "  [Line count] Input: $INPUT_LINES, Output: $RESULT_LINES" | tee -a $LOG_FILE

                ((DATASET_SAMPLES+=INPUT_LINES))
                ((TOTAL_SAMPLES+=INPUT_LINES))
                ((SUCCESS_SAMPLES+=INPUT_LINES))
            fi

            ((DATASET_FILES_PROCESSED++))
            ((PROCESSED_FILES++))
        else
            echo "[Error] $(basename $JSONL_FILE) inference failed!" | tee -a $LOG_FILE
            echo "  [Exit code] $INFER_EXIT_CODE" | tee -a $LOG_FILE
            echo "  [Error message]:" | tee -a $LOG_FILE
            cat $TEMP_LOG | grep -E "error|Error|ERROR|fail|Fail|FAIL|exception|Exception" | tee -a $LOG_FILE
            ((DATASET_FILES_FAILED++))
            ((FAILED_FILES++))
        fi

        # Clean up temporary log file
        # rm -f $TEMP_LOG
    done

    # Dataset statistics
    echo "[Dataset statistics] $DATASET_NAME:" | tee -a $LOG_FILE
    echo "  [Processed files] $DATASET_FILES_PROCESSED files" | tee -a $LOG_FILE
    echo "  [Skipped files] $DATASET_FILES_SKIPPED files" | tee -a $LOG_FILE
    if [ $DATASET_FILES_FAILED -gt 0 ]; then
        echo "  [Failed files] $DATASET_FILES_FAILED files" | tee -a $LOG_FILE
    fi
    if [ $DATASET_SAMPLES -gt 0 ]; then
        echo "  [Total samples] $DATASET_SAMPLES samples" | tee -a $LOG_FILE
    fi
    echo "------------------------------" | tee -a $LOG_FILE

    ((PROCESSED_DATASETS++))

done

# Overall statistics
END_SCRIPT_TIME=$(date +%s)
TOTAL_DURATION=$((END_SCRIPT_TIME - START_SCRIPT_TIME))
TOTAL_FILES=$((PROCESSED_FILES + SKIPPED_FILES + FAILED_FILES))

# Output overall statistics
echo "" | tee -a $LOG_FILE
echo "=== Batch inference completed: $(date) ===" | tee -a $LOG_FILE
echo "====================== Statistics Summary ======================" | tee -a $LOG_FILE
echo "Total datasets: $TOTAL_DATASETS" | tee -a $LOG_FILE
echo "  Processed: $PROCESSED_DATASETS" | tee -a $LOG_FILE
echo "  Skipped: $SKIPPED_DATASETS" | tee -a $LOG_FILE
echo "  Failed: $FAILED_DATASETS" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Total files: $TOTAL_FILES" | tee -a $LOG_FILE
echo "  Successfully processed: $PROCESSED_FILES" | tee -a $LOG_FILE
echo "  Skipped (already exist): $SKIPPED_FILES" | tee -a $LOG_FILE
echo "  Failed: $FAILED_FILES" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Sample statistics:" | tee -a $LOG_FILE
echo "  Successfully inferred samples: $SUCCESS_SAMPLES" | tee -a $LOG_FILE
echo "  Skipped samples (already exist): $((TOTAL_SAMPLES - SUCCESS_SAMPLES))" | tee -a $LOG_FILE
echo "  Failed samples: 0" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Total duration: $(printf "%02d:%02d:%02d\n" $((TOTAL_DURATION/3600)) $(((TOTAL_DURATION%3600)/60)) $((TOTAL_DURATION%60)))" | tee -a $LOG_FILE
echo "Average duration per dataset: $(printf "%dm%02ds\n" $((TOTAL_DURATION/TOTAL_DATASETS/60)) $((TOTAL_DURATION/TOTAL_DATASETS%60)))" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Output list of successfully processed files
echo "================== Successfully Processed File List ==================" | tee -a $LOG_FILE
for dataset_info in "${DATASETS[@]}"; do
    DATASET_NAME=$(echo $dataset_info | cut -d' ' -f1)
    RESULT_BASE_PATH="results/infer_results/$DATASET_NAME/$MODEL_NAME"

    if [ -d "$RESULT_BASE_PATH" ]; then
        RESULT_FILES=$(find "$RESULT_BASE_PATH" -name "*_result.jsonl" 2>/dev/null | wc -l)
        if [ $RESULT_FILES -gt 0 ]; then
            echo "$DATASET_NAME:" | tee -a $LOG_FILE
            find "$RESULT_BASE_PATH" -name "*_result.jsonl" 2>/dev/null | xargs -I {} basename {} | sort | tee -a $LOG_FILE
            echo "" | tee -a $LOG_FILE
        fi
    fi
done

# Output failure statistics
if [ $FAILED_FILES -gt 0 ]; then
    echo "================== Failed File List ===================" | tee -a $LOG_FILE
    echo "$FAILED_FILES file(s) failed to process, please check the log file for details." | tee -a $LOG_FILE
fi

echo "Detailed log: $LOG_FILE" | tee -a $LOG_FILE
echo "All results saved in: results/infer_results/" | tee -a $LOG_FILE
echo "=====================================================" | tee -a $LOG_FILE
