# UPDATE: Replace placeholder paths with your actual paths.
"""
Data conversion script for VSI-Bench dataset to MS-Swift JSONL format.

Target format (MS-Swift JSONL):
1. Each record must contain a messages list with standard system, user, assistant roles.
2. For multimodal data, insert <image> or <video> tags in user content and provide
   corresponding absolute paths in the images or videos list.
3. Preserve useful metadata fields (e.g., token, timestamp) in a metadata dictionary.

Business logic:
1. System Prompt is hardcoded, defining the model role.
2. Multi-view handling (optional): if the dataset has multiple views, insert <image> tags
   with view descriptions (e.g., <image> (front view)), and ensure images list order
   matches tag order in content.
3. Task splitting: support splitting into VQA (single-turn) and Conversation (multi-turn).

Code requirements:
1. Object-oriented programming with a Converter class.
2. Use argparse for parameters:
   - --input_file: path to raw data
   - --output_dir: output directory
   - --image_dir: root directory for image/video data
   - --task_type: task type
   - --data_split: data split (val only)
   - --convert_ratio: conversion ratio (0.0 to 1.0) for debug sampling
3. Include exception handling: print warning and continue on conversion failure.
   Include progress printing.
"""
import os
import json
import argparse
import random
from tqdm import tqdm

class Converter:
    def __init__(self, input_file, output_dir, image_dir, task_type, data_split, convert_ratio):
        self.input_file = input_file
        self.output_dir = output_dir
        self.image_dir = image_dir
        self.task_type = task_type
        self.data_split = data_split
        self.convert_ratio = convert_ratio

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, f"{data_split}.jsonl")

    def convert(self):
        # Read input file
        with open(self.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Sample lines based on convert_ratio
        if self.convert_ratio < 1.0:
            random.seed(42)
            sampled_lines = random.sample(lines, int(len(lines) * self.convert_ratio))
        else:
            sampled_lines = lines

        # Process each line
        total_processed = 0
        with open(self.output_file, 'w', encoding='utf-8') as out_f:
            for line in tqdm(sampled_lines, desc="Converting data"):
                try:
                    entry = json.loads(line.strip())
                    converted_entry = self.process_entry(entry)
                    if converted_entry:
                        out_f.write(json.dumps(converted_entry, ensure_ascii=False) + '\n')
                        total_processed += 1
                except Exception as e:
                    print(f"Warning: Failed to process entry: {e}")
                    continue

        print(f"Conversion completed. Processed {total_processed} out of {len(sampled_lines)} entries.")

    def process_entry(self, entry):
        # Extract fields
        id_ = entry.get('id')
        dataset = entry.get('dataset')
        scene_name = entry.get('scene_name')
        question_type = entry.get('question_type')
        question = entry.get('question')
        ground_truth = entry.get('ground_truth')
        options = entry.get('options')

        # System prompt
        system_prompt = "You are a helpful assistant. Please answer the question according to the given images or videos. "

        # User message content
        user_content = question

        # Add options to user content if they exist
        if options:
            user_content += "\nOptions:"
            for option in options:
                user_content += f"\n{option}"

        # Check if there are videos
        videos = []
        # Construct video path based on dataset
        # Assuming image_dir is the root directory containing dataset folders
        video_dir = os.path.join(self.image_dir, dataset)
        if os.path.exists(video_dir):
            # For scannet and scannetpp, the video file might have a different extension or naming
            # Try different extensions
            extensions = ['.mp4', '.avi', '.mov']
            video_path = None
            for ext in extensions:
                temp_path = os.path.join(video_dir, f"{scene_name}{ext}")
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break

            if video_path:
                videos.append(video_path)
                # Add <video> tag to user content
                user_content += " <video>"
            else:
                print(f"Warning: Video file not found for {dataset}/{scene_name}")
        else:
            print(f"Warning: Video directory not found: {video_dir}")

        # Assistant message content
        if options:
            # Map ground truth (e.g., "C") to option text
            # Assuming options are a list like ["A. ...", "B. ...", ...]
            # Extract the index from ground truth (A=0, B=1, C=2, D=3)
            index = ord(ground_truth) - ord('A')
            if 0 <= index < len(options):
                assistant_content = options[index]
            else:
                assistant_content = ground_truth
        else:
            assistant_content = ground_truth

        # Construct messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]

        # Construct converted entry
        converted_entry = {
            "messages": messages,
            "metadata": {
                "id": id_,
                "dataset": dataset,
                "scene_name": scene_name,
                "question_type": question_type,
                "options": options,
            }
        }

        # Add videos list if not empty
        if videos:
            converted_entry["videos"] = videos

        return converted_entry

def main():
    parser = argparse.ArgumentParser(description="Convert VSI-Bench data to MS-Swift JSONL format")
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory for images/videos')
    parser.add_argument('--task_type', type=str, required=True, help='Task type')
    parser.add_argument('--data_split', type=str, required=True, help='Data split (e.g., val)')
    parser.add_argument('--convert_ratio', type=float, default=1.0, help='Convert ratio (0.0-1.0)')
    args = parser.parse_args()

    converter = Converter(
        input_file=args.input_file,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        task_type=args.task_type,
        data_split=args.data_split,
        convert_ratio=args.convert_ratio
    )
    converter.convert()

if __name__ == "__main__":
    main()
