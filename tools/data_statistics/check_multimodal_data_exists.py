# UPDATE: Replace placeholder paths with your actual paths.
"""Task: Write a Python script to check if given multimodal data files exist.
If a file does not exist, print a warning and return False; otherwise return True.
Aggregate all results. Also support per-dataset sample count statistics,
and provide counts for: single-image, multi-image, and video QA respectively."""

# Reference data file paths (replace with your actual paths):
# data/lingoqa_action_train_qwen_v0.json
# data/merged_surds_vqas_v0.jsonl
# data/preprocess_data/MapLMv2/maplm_v2_train_ms_swift.jsonl
# data/preprocess_data/MapLMv2/maplm_v2_train_ms_swift_multi.jsonl
# data/Drivevqa/filter_data/drivevqa_rel_filter_06.jsonl
# data/DriveLMMo1/filter_data/filtered_raw_data_drivelmm_06_v2.jsonl
# data/Omnidrive/filter_data/omnidrive_vqa_filtered_06.jsonl
# data/Omnidrive/filter_data/omnidrive_vqa_filtered_multi_06.jsonl
# data/RefCOCO/raw_data/refcoco_train_qwen_v1.jsonl
# data/Flickr/filter_data/flickr_filter_06.jsonl
# data/VQAv2/filter_data/vqa_filter_06.jsonl
# data/Gqa/filter_data/gqa_filter_06.jsonl
# data/bdd100k_labels_images_train_weather_qwen_v1.json
# data/Cosmosr1/filter_data/cosmosr1_filter_06.jsonl
# data/preprocess_data/cosmos_rl_data/robovqa_rl_qa_pairs_ms_swift.jsonl
# data/robovqa_understanding.jsonl
# data/vladbench_train_image_rel_qwen_v1.json
# data/visual_trace_rel_qwen_v0.jsonl
# data/embodied-r1_train_rel_qwen_v0.jsonl
# data/part_affordance_train_rel_qwen_v0.jsonl
# data/RoboAfford_train_rel_qwen_v0.jsonl
# data/vabench_bbox_no_reasoning_train_rel_qwen_v0.jsonl
# data/where2place_train_rel_qwen_v0.jsonl
# data/OpenSpaces/dataset/openspaces_train.jsonl
# data/convert_tools/SPAR-7m/scannet_merged_data_converted.jsonl
# data/convert_tools/SPAR-7m/structured3d_merged_data_converted.jsonl
# data/preprocess_data/vsi-590k/train.jsonl

# 1. Each record contains a messages list with standard system, user, assistant roles. JSONL format:
# {"messages": [{"role": "assistant", "content": "pretrained text here"}]}
# {"messages": [{"role": "assistant", "content": "<image>is a puppy, <image>is a kitten"}], "images": ["/xxx/x.jpg", "/xxx/x.png"]}
# {"messages": [{"role": "assistant", "content": "<audio>describes nice weather today"}], "audios": ["/xxx/x.wav"]}
# {"messages": [{"role": "assistant", "content": "<image>is an elephant, <video>is a running lion"}], "images": ["/xxx/x.jpg"], "videos": ["/xxx/x.mp4"]}
# 2. Multimodal alignment: For data containing images/videos, insert corresponding <image> or <video> tags
#    in the user content of messages, and provide corresponding absolute paths in the images or videos list.

import os
import json
from collections import defaultdict
from typing import List, Dict, Any

# Configure data file path list
FILE_LIST = [
    "data/lingoqa_action_train_qwen_v0.json",
    "data/merged_surds_vqas_v0.jsonl",
    "data/preprocess_data/MapLMv2/maplm_v2_train_ms_swift.jsonl",
    "data/preprocess_data/MapLMv2/maplm_v2_train_ms_swift_multi.jsonl",
    "data/Drivevqa/filter_data/drivevqa_rel_filter_06.jsonl",
    "data/DriveLMMo1/filter_data/filtered_raw_data_drivelmm_06_v2.jsonl",
    "data/Omnidrive/filter_data/omnidrive_vqa_filtered_06.jsonl",
    "data/Omnidrive/filter_data/omnidrive_vqa_filtered_multi_06.jsonl",
    "data/RefCOCO/raw_data/refcoco_train_qwen_v1.jsonl",
    "data/Flickr/filter_data/flickr_filter_06.jsonl",
    "data/VQAv2/filter_data/vqa_filter_06.jsonl",
    "data/Gqa/filter_data/gqa_filter_06.jsonl",
    "data/bdd100k_labels_images_train_weather_qwen_v1.json",
    "data/Cosmosr1/filter_data/cosmosr1_filter_06.jsonl",
    "data/preprocess_data/cosmos_rl_data/robovqa_rl_qa_pairs_ms_swift.jsonl",
    "data/robovqa_understanding.jsonl",
    "data/vladbench_train_image_rel_qwen_v1.json",
    "data/visual_trace_rel_qwen_v0.jsonl",
    "data/embodied-r1_train_rel_qwen_v0.jsonl",
    "data/part_affordance_train_rel_qwen_v0.jsonl",
    "data/RoboAfford_train_rel_qwen_v0.jsonl",
    "data/vabench_bbox_no_reasoning_train_rel_qwen_v0.jsonl",
    "data/where2place_train_rel_qwen_v0.jsonl",
    "data/OpenSpaces/dataset/openspaces_train.jsonl",
    "data/convert_tools/SPAR-7m/scannet_merged_data_converted.jsonl",
    "data/convert_tools/SPAR-7m/structured3d_merged_data_converted.jsonl",
    "data/preprocess_data/vsi-590k/train.jsonl"
]

def check_file_existence(file_path: str) -> bool:
    """Check if a file exists"""
    if os.path.exists(file_path):
        return True
    else:
        print(f"[Warning] File does not exist: {file_path}")
        return False

def load_data(file_path: str) -> List[Dict]:
    """Load data, automatically handle .json, .jsonl and suffix name errors"""
    data = []
    try:
        # Try reading as JSONL first (streaming), compatible with .json suffix that is actually JSONL
        # This is safer since standard JSON can also be parsed line by line (if format allows),
        # but for "Extra data" errors, direct line-by-line parsing is the best approach.
        is_jsonl_content = False
        bad_lines = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                        is_jsonl_content = True
                    except json.JSONDecodeError:
                        if not is_jsonl_content:
                            # The very first non-empty line is not JSON: this is probably a
                            # pretty-printed JSON file, so fall through to the standard reader.
                            data = []
                            break
                        # A later bad line must not discard everything parsed so far --
                        # the previous version reset `data` to [] here, so a single corrupt
                        # line wiped the statistics for the whole file.
                        bad_lines += 1
                        print(f"  [WARN] {file_path}:{lineno} is not valid JSON, skipped")

            if is_jsonl_content:
                if bad_lines:
                    print(f"  [WARN] {file_path}: {bad_lines} malformed line(s) skipped")
                return data

        # If line-by-line reading failed (or no data), try standard JSON reading
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)

            if isinstance(content, list):
                data = content
            elif isinstance(content, dict):
                if 'data' in content:
                    data = content['data']
                else:
                    data = [content]

    except json.JSONDecodeError as e:
        print(f"[Error] JSON parse failed {file_path}: {e}")
    except Exception as e:
        print(f"[Error] Failed to read file {file_path}: {e}")

    return data

def normalize_to_list(value):
    """Convert string or list to a unified list"""
    if isinstance(value, str):
        return [value]
    elif isinstance(value, list):
        return value
    return []

def analyze_dataset(file_path: str):
    """Analyze a single dataset"""
    dataset_name = os.path.basename(file_path)
    stats = {
        'total': 0,
        'single_image': 0,
        'multi_image': 0,
        'video_qa': 0,
        'missing_media': 0
    }

    data_list = load_data(file_path)
    stats['total'] = len(data_list)

    if stats['total'] == 0:
        # Only print when file exists but parses to empty, to avoid spam
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
             print(f"[Info] {dataset_name} parsed to empty data")
        return stats

    for item in data_list:
        # Get fields and normalize to list
        images = normalize_to_list(item.get('images'))
        videos = normalize_to_list(item.get('videos'))

        # Check if files exist
        media_files = images + videos
        has_missing = False
        for media_path in media_files:
            if not os.path.exists(media_path):
                stats['missing_media'] += 1
                has_missing = True
                break # Record only one missing per record

        # Statistics logic
        # Priority: check if it contains video
        if videos:
            stats['video_qa'] += 1
        elif images:
            if len(images) == 1:
                stats['single_image'] += 1
            else:
                stats['multi_image'] += 1

    return stats

def main():
    print("=" * 100)
    print("Starting multimodal dataset check...")
    print("=" * 100)

    global_stats = {
        'total_files': len(FILE_LIST),
        'existing_files': 0,
        'total_samples': 0,
        'total_single_image': 0,
        'total_multi_image': 0,
        'total_video_qa': 0
    }

    # Print header
    print(f"{'Dataset Name':<50} | {'Total':<8} | {'Single':<8} | {'Multi':<8} | {'Video':<8} | {'Missing':<8}")
    print("-" * 110)

    for file_path in FILE_LIST:
        if check_file_existence(file_path):
            global_stats['existing_files'] += 1
            stats = analyze_dataset(file_path)

            # Accumulate global statistics
            global_stats['total_samples'] += stats['total']
            global_stats['total_single_image'] += stats['single_image']
            global_stats['total_multi_image'] += stats['multi_image']
            global_stats['total_video_qa'] += stats['video_qa']

            # Print single line result
            print(f"{os.path.basename(file_path):<50} | {stats['total']:<8} | {stats['single_image']:<8} | {stats['multi_image']:<8} | {stats['video_qa']:<8} | {stats['missing_media']:<8}")

    print("-" * 110)
    print(f"{'Summary (SUM)':<50} | {global_stats['total_samples']:<8} | {global_stats['total_single_image']:<8} | {global_stats['total_multi_image']:<8} | {global_stats['total_video_qa']:<8}")
    print("=" * 100)
    print(f"File check complete: {global_stats['existing_files']}/{global_stats['total_files']} exist")

if __name__ == "__main__":
    main()
