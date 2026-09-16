# UPDATE: Replace placeholder paths with your actual paths.
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import random


def extract_dataset_info(input_file: str) -> Tuple[str, str, Path]:
    """
    Extract dataset info from input file path

    Args:
        input_file: Input file path

    Returns:
        dataset_name: Dataset name
        task_type: Task type
        video_root: Video file root directory
    """
    input_path = Path(input_file)

    # Determine which dataset from filename
    if "robofail" in input_path.name:
        dataset_name = "cosmos_r1_robofail"
        task_type = "success_prediction"
    elif "robovqa" in input_path.name:
        dataset_name = "cosmos_r1_robovqa"
        task_type = "video_qa"
    else:
        # Default: determine from parent directory name
        parent_name = input_path.parent.name.lower()
        if "robofail" in parent_name:
            dataset_name = "cosmos_r1_robofail"
            task_type = "success_prediction"
        elif "robovqa" in parent_name:
            dataset_name = "cosmos_r1_robovqa"
            task_type = "video_qa"
        else:
            dataset_name = "cosmos_r1_unknown"
            task_type = "unknown"

    # Determine video root directory
    # Videos may be in clips directory, but need to check
    possible_video_dirs = [
        input_path.parent / "clips",  # clips in same directory
        input_path.parent.parent / "clips",  # clips in parent directory
    ]

    video_root = None
    for dir_path in possible_video_dirs:
        if dir_path.exists():
            video_root = dir_path
            break

    if video_root is None:
        # If clips directory not found, use parent directory
        video_root = input_path.parent
        print(f"Warning: clips directory not found, using {video_root} as video root directory")

    return dataset_name, task_type, video_root


def get_system_prompt(task_type: str) -> str:
    """Get system prompt based on task type"""
    if task_type == "success_prediction":
        return "You are a helpful robot assistant. Watch the video and answer the question about whether the robot successfully completed a subtask."
    elif task_type == "video_qa":
        return "You are a helpful robot assistant. Watch the video and answer questions about what is happening in the video."
    else:
        return "You are a helpful robot assistant. Watch the video and answer the question."


def process_qa_pairs(qa_pairs: Dict[str, Any], task_type: str) -> Tuple[str, str, str, Dict]:
    """
    Process QA pairs, extract question and answer

    Args:
        qa_pairs: QA pairs dictionary
        task_type: Task type

    Returns:
        question: Question text
        answer_key: Answer key
        answer_text: Answer text
        meta_info: Meta info dictionary
    """
    # Extract info based on dataset structure
    if "question" in qa_pairs and "answer" in qa_pairs:
        # robofail format
        question = qa_pairs["question"]
        index2ans = qa_pairs.get("index2ans", {})
        answer_key = qa_pairs["answer"]
        answer_text = index2ans.get(answer_key, "")
        task_desc = qa_pairs.get("task", "")

        # Build options string
        if index2ans:
            options = "\n".join([f"{k}: {v}" for k, v in index2ans.items()])
            full_question = f"{question}\n{options}"
        else:
            full_question = question

    elif "q" in qa_pairs and "a" in qa_pairs:
        # Possible other format
        question = qa_pairs["q"]
        answer_key = qa_pairs["a"]
        answer_text = answer_key
        full_question = question

        index2ans = qa_pairs.get("index2ans", {})
        task_desc = qa_pairs.get("task", "")
    else:
        # Unknown format, try to extract
        question = str(qa_pairs)
        answer_key = ""
        answer_text = ""
        full_question = question
        index2ans = {}
        task_desc = ""

    meta_info = {
        "original_qa_pairs": qa_pairs,
        "index2ans": index2ans,
        "task_desc": task_desc
    }

    return full_question, answer_key, answer_text, meta_info


def convert_cosmos_r1_dataset(
    input_file: str,
    output_dir: str,
    convert_ratio: float = 1.0
) -> None:
    """
    Convert Cosmos-R1 benchmark dataset to ms-swift format

    Args:
        input_file: Original JSON file path
        output_dir: Output directory
        convert_ratio: Conversion ratio, for debugging
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract dataset info
    dataset_name, task_type, video_root = extract_dataset_info(input_file)

    # Build output file path
    output_file = output_dir / f"{dataset_name}.jsonl"

    # Read original data
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"Dataset: {dataset_name}")
    print(f"Task type: {task_type}")
    print(f"Video root directory: {video_root}")
    print(f"Original data entries: {len(raw_data)}")

    # Sample data based on conversion ratio
    if convert_ratio < 1.0:
        sample_size = int(len(raw_data) * convert_ratio)
        raw_data = random.sample(raw_data, sample_size)
        print(f"Sampled data entries: {len(raw_data)}")

    # Get system prompt
    system_prompt = get_system_prompt(task_type)

    # Prepare converted data
    converted_data = []
    processed_count = 0
    error_count = 0

    for i, item in enumerate(raw_data):
        try:
            # Extract video path
            if "video" in item:
                video_path = item["video"]
            elif "video_path" in item:
                video_path = item["video_path"]
            else:
                raise KeyError("Video path field not found")

            # Extract QA pairs
            if "qa_pairs" in item:
                qa_pairs = item["qa_pairs"]
            elif "qa" in item:
                qa_pairs = item["qa"]
            else:
                # Try using item directly as QA pairs
                qa_pairs = {k: v for k, v in item.items() if k != "video" and k != "video_path"}

            # Build absolute video path
            video_filename = Path(video_path).name
            video_abs_path = video_root / video_filename

            # Check if video file exists
            if not video_abs_path.exists():
                print(f"Warning: Video file does not exist: {video_abs_path}")
                # Can skip or use relative path
                video_path_to_use = str(video_abs_path)
            else:
                video_path_to_use = str(video_abs_path)

            # Process QA pairs
            question, answer_key, answer_text, qa_meta = process_qa_pairs(qa_pairs, task_type)

            # Build ms-swift format messages
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"<video>\n{question}"
                },
                {
                    "role": "assistant",
                    "content": answer_key
                }
            ]

            # Build complete data item
            converted_item = {
                "messages": messages,
                "videos": [video_path_to_use]
            }

            # Add metadata
            converted_item["meta_data"] = {
                "original_video_path": video_path,
                "dataset": dataset_name,
                "task_type": task_type,
                "question": question,
                "answer_key": answer_key,
                "answer_text": answer_text,
                "qa_meta": qa_meta,
                "data_idx": i
            }

            # Add other original fields to metadata
            for key, value in item.items():
                if key not in ["qa_pairs", "qa", "video", "video_path"]:
                    converted_item["meta_data"][f"original_{key}"] = value

            converted_data.append(converted_item)
            processed_count += 1

        except Exception as e:
            error_count += 1
            print(f"Error processing item {i}: {e}")
            print(f"Problem data: {item}")
            continue

    # Write JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + '\n')

    print(f"Conversion complete!")
    print(f"Successfully processed: {processed_count} items")
    print(f"Failed: {error_count} items")
    print(f"Output file: {output_file}")


def process_multiple_datasets(
    dataset_paths: List[str],
    output_base_dir: str,
    convert_ratio: float = 1.0
) -> None:
    """
    Batch process multiple datasets

    Args:
        dataset_paths: Dataset path list
        output_base_dir: Output base directory
        convert_ratio: Conversion ratio
    """
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            print(f"\nProcessing dataset: {dataset_path}")
            convert_cosmos_r1_dataset(
                input_file=dataset_path,
                output_dir=output_base_dir,
                convert_ratio=convert_ratio
            )
        else:
            print(f"File does not exist: {dataset_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Cosmos-R1 benchmark dataset to ms-swift format")

    # Set default values
    default_inputs = [
        "data/public_robo_datasets/Cosmos-R1/robovqa/robovqa_benchmark_qa_pairs.json",
        "data/public_robo_datasets/Cosmos-R1/robofail/robofail_benchmark_qa_pairs.json"
    ]

    default_output = "data/preprocess_data/all_eval_data/cosmos_r1"

    parser.add_argument(
        "--input_files",
        type=str,
        nargs='+',
        default=default_inputs,
        help="Original JSON file path list, supports multiple files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output,
        help="Output directory"
    )
    parser.add_argument(
        "--convert_ratio",
        type=float,
        default=1.0,
        help="Conversion ratio, for debugging (between 0-1)"
    )
    parser.add_argument(
        "--single_file",
        type=str,
        help="Process a single file, overrides input_files parameter"
    )

    args = parser.parse_args()

    # If a single file is specified, process it
    if args.single_file:
        convert_cosmos_r1_dataset(
            input_file=args.single_file,
            output_dir=args.output_dir,
            convert_ratio=args.convert_ratio
        )
    else:
        # Process multiple files
        process_multiple_datasets(
            dataset_paths=args.input_files,
            output_base_dir=args.output_dir,
            convert_ratio=args.convert_ratio
        )


if __name__ == "__main__":
    main()
