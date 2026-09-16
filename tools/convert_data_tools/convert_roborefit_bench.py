# UPDATE: Replace placeholder paths with your actual paths.
import json
import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Any

def convert_roborefit_benchmark(args):
    """
    Convert Roborefit benchmark data to ms-swift format
    """
    # Read metadata.json
    metadata_path = args.input_file
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Determine dataset name
    dataset_name = "roborefit-benchmark"

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"data/preprocess_data/{dataset_name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if "bench" in metadata_path or "test" in metadata_path:
        output_file = output_dir / "roborefit_ms_swift_val.jsonl"
    else:
        output_file = output_dir / "train.jsonl"

    # Sample if needed
    if args.convert_ratio < 1.0 and len(metadata) > 0:
        sample_size = int(len(metadata) * args.convert_ratio)
        metadata = random.sample(metadata, sample_size)

    # Get base path for images
    metadata_dir = Path(metadata_path).parent

    # Convert data
    converted_data = []

    skipped = 0

    for item in metadata:
        # Validate the record before touching any of its fields. A malformed entry used to
        # abort the whole run with a KeyError/ValueError, and since the output file is only
        # written after the loop that meant an empty dataset.
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            skipped += 1
            print(f"  [SKIP] {item.get('id', '?')}: missing or malformed bbox ({bbox!r})")
            continue
        if not item.get("image_path"):
            skipped += 1
            print(f"  [SKIP] {item.get('id', '?')}: missing image_path")
            continue

        # Build image path
        image_path = metadata_dir / item["image_path"]

        # Build user prompt
        # Use concise instruction, focusing only on box localization
        ref_exp = item.get("ref_exp", "")
        # `ref_exp.split('the')[1]` raised IndexError for any instruction without the word
        # "the" (and split on "there"/"then" too, yielding a garbled object name). The
        # instruction is already self-contained, so just reuse it.
        user_prompt = f"<image>\nPlease locate the object according to the instruction: {ref_exp}\n\nPlease output the bounding box coordinates for the object described above in the format: <answer>[x1, y1, x2, y2]</answer>"

        # Build assistant response - only return box coordinates
        x1, y1, x2, y2 = bbox
        assistant_response = f"<answer>[{x1}, {y1}, {x2}, {y2}]</answer>"

        # Build meta_data
        meta_data = {
            "id": item.get("id", ""),
            "original_id": item.get("original_id", ""),
            "source_file": item.get("source_file", ""),
            "file_index": item.get("file_index", 0),
            "row_index": item.get("row_index", 0),
            "has_image": item.get("has_image", True),
            "has_annotation": item.get("has_annotation", True),
            "image_path": str(item.get("image_path", "")),
            "annotation_path": item.get("annotation_path", ""),
            "conversation_path": item.get("conversation_path", ""),
            "ref_exp": item.get("ref_exp", ""),
            "bbox": item.get("bbox", []),
            "normalized_bbox": item.get("normalized_bbox", []),
            "image_width": None,  # Can be read from image, set to None for now
            "image_height": None
        }

        # Build converted data
        converted_item = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. I need you to identify and locate objects in the image based on the instruction."
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": assistant_response
                }
            ],
            "images": [str(image_path)],
            "meta_data": meta_data
        }

        converted_data.append(converted_item)

    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Conversion complete! Converted {len(converted_data)} samples")
    if skipped:
        print(f"Skipped {skipped} malformed record(s).")
    print(f"Output file: {output_file}")

    # Save data statistics
    stats = {
        "total_samples": len(converted_data),
        "output_file": str(output_file),
        "input_file": args.input_file,
        "convert_ratio": args.convert_ratio
    }

    stats_file = output_dir / "conversion_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Statistics saved to: {stats_file}")

    return converted_data

def main():
    parser = argparse.ArgumentParser(description='Convert Roborefit benchmark data to ms-swift format')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input metadata.json file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory, default: data/preprocess_data/roborefit-benchmark')
    parser.add_argument('--convert_ratio', type=float, default=1.0,
                       help='Conversion ratio for debugging, default 1.0 (convert all)')

    args = parser.parse_args()

    # Execute conversion
    convert_roborefit_benchmark(args)

if __name__ == "__main__":
    main()
