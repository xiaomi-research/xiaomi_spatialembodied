# UPDATE: Replace placeholder paths with your actual paths.
import json
import os
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any
import glob

def convert_drivebench_to_qwen_format(input_file: str, output_file: str) -> None:
    """
    Convert DriveBench JSONL format to Qwen-SFT format

    Input format:
    {
        "id": "...",
        "image": "path/to/image.png",
        "conversations": [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "answer"},
            {"from": "gpt", "value": "robust_qas"},
            {"from": "gpt", "value": [0]}
        ]
    }

    Output format:
    {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"}
        ],
        "images": ["path/to/image.png"],
        "question_type": "robust_qas",
        "answer_index": [0]
    }
    """

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Extract basic info
                conversations = data.get("conversations", [])
                image_path = data.get("image", "")

                if len(conversations) < 2:
                    print(f"Warning: conversations length insufficient at line {line_num}")
                    continue

                # Extract conversation content
                messages = []
                for conv in conversations[:2]:  # Only take first two (question and answer)
                    if conv.get("from") == "human":
                        messages.append({
                            "role": "user",
                            "content": conv.get("value", "")
                        })
                    elif conv.get("from") == "gpt":
                        messages.append({
                            "role": "assistant",
                            "content": conv.get("value", "")
                        })

                # Build output data
                output_data = {
                    "messages": messages,
                    "images": [image_path] if image_path else []
                }

                # Add extra info (if exists)
                if len(conversations) > 2:
                    # Third element is question type
                    if len(conversations) > 2 and conversations[2].get("from") == "gpt":
                        output_data["question_type"] = conversations[2].get("value", "")

                    # Fourth element is answer index
                    if len(conversations) > 3 and conversations[3].get("from") == "gpt":
                        output_data["answer_index"] = conversations[3].get("value", [])

                # Write output file
                outfile.write(json.dumps(output_data, ensure_ascii=False) + "\n")

            except json.JSONDecodeError as e:
                print(f"Error: line {line_num} JSON parse failed: {e}")
                continue
            except Exception as e:
                print(f"Error: line {line_num} processing failed: {e}")
                continue

def process_all_files(input_dir: str, output_dir: str) -> None:
    """
    Process all JSONL files in specified directory

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all JSONL files
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in directory {input_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files:")
    for file_path in jsonl_files:
        print(f"  - {os.path.basename(file_path)}")

    # Process each file
    for input_file in jsonl_files:
        filename = os.path.basename(input_file)
        output_filename = filename.replace(".jsonl", "_qwen.jsonl")
        output_file = os.path.join(output_dir, output_filename)

        print(f"\nProcessing file: {filename}")
        print(f"Output to: {output_filename}")

        convert_drivebench_to_qwen_format(input_file, output_file)

        # Verify conversion result
        with open(output_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            try:
                sample_data = json.loads(first_line)
                print(f"Conversion successful! Example output:")
                print(f"  messages: {len(sample_data.get('messages', []))} entries")
                print(f"  images: {len(sample_data.get('images', []))} images")
                print(f"  Extra fields: {list(sample_data.keys() - {'messages', 'images'})}")
            except:
                print("Conversion result verification failed")

def process_specific_file(input_path: str) -> None:
    """
    Process a specified single file
    """
    input_dir = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    output_filename = filename.replace(".jsonl", "_qwen.jsonl")
    output_path = os.path.join(input_dir, output_filename)

    print(f"Processing file: {input_path}")
    print(f"Output to: {output_path}")

    convert_drivebench_to_qwen_format(input_path, output_path)

    # Conversion complete
    print(f"\nConversion complete!")

    # Show before/after comparison
    print(f"\nBefore conversion example:")
    with open(input_path, 'r', encoding='utf-8') as f:
        for _ in range(2):  # Show first 2 lines
            line = f.readline()
            if line:
                data = json.loads(line)
                print(f"  ID: {data.get('id', 'N/A')}")
                print(f"  Image: {data.get('image', 'N/A')}")
                print(f"  Conversation count: {len(data.get('conversations', []))}")

    print(f"\nAfter conversion example:")
    with open(output_path, 'r', encoding='utf-8') as f:
        for _ in range(2):  # Show first 2 lines
            line = f.readline()
            if line:
                data = json.loads(line)
                print(f"  messages: {data.get('messages', [])}")
                print(f"  images: {data.get('images', [])}")
                print(f"  Extra fields: {list(data.keys() - {'messages', 'images'})}")

if __name__ == "__main__":
    # # Method 1: Process a specified single file
    # print("=" * 60)
    # print("Method 1: Process a specified single file")
    # print("=" * 60)

    # specific_file = "data/preprocess_data/data/bright.jsonl"
    # if os.path.exists(specific_file):
    #     process_specific_file(specific_file)
    # else:
    #     print(f"File does not exist: {specific_file}")
    #     print("Please check if the path is correct")

    # Method 2: Process entire directory
    print("\n" + "=" * 60)
    print("Method 2: Batch process entire directory")
    print("=" * 60)

    input_directory = "data/preprocess_data/data"
    output_directory = "data/preprocess_data/drivebench"

    if os.path.exists(input_directory):
        process_all_files(input_directory, output_directory)
    else:
        print(f"Input directory does not exist: {input_directory}")

    print("\n" + "=" * 60)
    print("Conversion script is ready!")
    print("=" * 60)
