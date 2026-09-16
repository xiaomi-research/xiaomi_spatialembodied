# UPDATE: Replace placeholder paths with your actual paths.

# python
# !/usr/bin/env python3

"""
Convert vabench-point-bbox dataset to ms-swift format with support for different coordinate normalization modes.
Modified to use box format instead of point format.
"""

import os
import json
import argparse
import random
from PIL import Image
from typing import List, Tuple, Dict, Any
from pathlib import Path

def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image width and height."""
    try:
        with Image.open(image_path) as img:
            return img.size  # returns (width, height)
    except Exception as e:
        print(f"Warning: Could not read image dimensions for {image_path}: {e}")
        return None, None

def normalize_coordinates(bbox: List[int], width: int, height: int, 
                         coord_mode: str) -> List[float]:
    """
    Normalize coordinates according to the specified mode.
    
    Args:
        bbox: [x1, y1, x2, y2] in absolute coordinates
        width: image width
        height: image height
        coord_mode: normalization mode, one of ['absolute', 'thousand', 'normalized']
    
    Returns:
        List of box coordinates [x1, y1, x2, y2] in the specified coordinate system
    """
    x1, y1, x2, y2 = bbox
    
    if coord_mode == 'absolute':
        # Use absolute coordinates directly
        return [float(x1), float(y1), float(x2), float(y2)]
    
    elif coord_mode == 'thousand':
        # Convert to thousand-level normalized coordinates (0-1000)
        if width and height:
            x1_norm = int(x1 / width * 1000)
            y1_norm = int(y1 / height * 1000)
            x2_norm = int(x2 / width * 1000)
            y2_norm = int(y2 / height * 1000)
            return [float(x1_norm), float(y1_norm), float(x2_norm), float(y2_norm)]
        else:
            # Fallback to absolute if image dimensions not available
            return [float(x1), float(y1), float(x2), float(y2)]
    
    elif coord_mode == 'normalized':
        # Convert to normalized coordinates (0-1) and round to 2 decimal places
        if width and height:
            x1_norm = round(x1 / width, 2)
            y1_norm = round(y1 / height, 2)
            x2_norm = round(x2 / width, 2)
            y2_norm = round(y2 / height, 2)
            return [x1_norm, y1_norm, x2_norm, y2_norm]
        else:
            # Fallback to absolute if image dimensions not available
            return [float(x1), float(y1), float(x2), float(y2)]
    
    else:
        raise ValueError(f"Unknown coord_mode: {coord_mode}")

def process_user_prompt(original_prompt: str, include_reasoning: bool) -> str:
    """
    Process the user prompt based on whether reasoning is included.
    
    When include_reasoning is False, remove the reasoning instruction and
    change it to "directly provide the final answer".
    Modify point format to box format in the instruction.
    """
    if include_reasoning:
        # Reasoning version: modify point to box format
        prompt = original_prompt
        # Replace references to points with bounding box
        prompt = prompt.replace("Use 2D points to mark", "Use a bounding box to mark")
        prompt = prompt.replace("several coordinate points", "bounding box coordinates")
        # Replace the example format
        prompt = prompt.replace("</think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>", 
                               "</think><answer>[x1, y1, x2, y2]</answer>")
        return prompt
    
    # No reasoning version
    lines = original_prompt.split('\n')
    processed_lines = []
    
    for i, line in enumerate(lines):
        if i == 0:
            # First line: task instruction - replace "2D points" with "bounding box"
            line = line.replace("Use 2D points to mark", "Use a bounding box to mark")
            processed_lines.append(line)
        elif "You FIRST think about the reasoning process" in line:
            # Replace with direct answer instruction for box format
            new_line = "You directly provide the final answer. The answer is enclosed within <answer> </answer> tags. The answer consists only of bounding box coordinates, with the overall format being: <answer>[x1, y1, x2, y2]</answer>"
            processed_lines.append(new_line)
        elif "The reasoning process and answer are enclosed" in line:
            # Skip this line
            continue
        elif "The answer consists only of several coordinate points" in line and "</think><answer>" in line:
            # Skip this line
            continue
        elif "with the overall format being:" in line and "point>" in line:
            # This line will be handled by the replacement above
            continue
        else:
            # For other lines, replace references to points
            line = line.replace("coordinate points", "bounding box coordinates")
            # Replace point format with box format
            if "<point>[[x1, y1], [x2, y2], ...]</point>" in line:
                line = line.replace("<point>[[x1, y1], [x2, y2], ...]</point>", 
                                   "[x1, y1, x2, y2]")
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def convert_vabench_to_swift(
    input_file: str,
    output_dir: str,
    convert_ratio: float = 1.0,
    coord_mode: str = 'thousand',
    system_prompt: str = None,
    include_reasoning: bool = True
) -> List[Dict]:
    """
    Convert vabench-point-bbox dataset to ms-swift format.
    
    Args:
        input_file: Path to metadata.json file
        output_dir: Output directory for converted data
        convert_ratio: Ratio of data to convert (0.0-1.0)
        coord_mode: Coordinate normalization mode 
                   ('absolute', 'thousand', or 'normalized')
        system_prompt: Custom system prompt (if None, use default)
        include_reasoning: Whether to include reasoning process in answer
    
    Returns:
        List of converted samples in ms-swift format
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original metadata
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Apply conversion ratio
    num_samples = int(len(original_data) * convert_ratio)
    samples_to_convert = original_data[:num_samples]
    
    # Default system prompt (in English) - updated for box format
    if system_prompt is None:
        system_prompt = (
            "You are a robot assistant performing robotic manipulation tasks. "
            "Please mark the target location with a bounding box based on the task instructions."
        )
    
    swift_data = []
    
    for i, sample in enumerate(samples_to_convert):
        if i % 100 == 0:
            print(f"Processing sample {i+1}/{num_samples}...")
        
        # Get image path and dimensions
        image_path = sample["image_path"]
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Get image dimensions
        width, height = get_image_dimensions(image_path)
        
        # Process bounding box coordinates
        bbox = sample["bbox"]
        box_coords = normalize_coordinates(bbox, width, height, coord_mode)
        
        # Generate reasoning process (in English)
        if include_reasoning:
            reasoning = (
                "I will analyze the task instruction and determine the target location. "
                "Based on scene understanding, I identify the target object that needs to be manipulated, "
                "and mark the target location with a bounding box according to the instruction requirements."
            )
            answer = f"<think>{reasoning}</think><answer>{box_coords}</answer>"
        else:
            answer = f"<answer>{box_coords}</answer>"
        
        # Process user prompt
        user_content = process_user_prompt(sample["problem"], include_reasoning)
        
        # Use absolute image path
        absolute_image_path = image_path
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
        
        # Create the swift format sample
        swift_sample = {
            "messages": messages,
            "images": [absolute_image_path]  # Use absolute path
        }
        
        # Format normalized_bbox to 2 decimal places if it exists
        normalized_bbox = sample.get("normalized_bbox", [])
        if normalized_bbox and len(normalized_bbox) == 4:
            normalized_bbox = [round(float(x), 2) for x in normalized_bbox]
        
        # Add metadata with all original information
        swift_sample["meta_data"] = {
            "id": sample["id"],
            "original_bbox": sample["bbox"],
            "normalized_bbox": normalized_bbox,
            "original_image_path": sample["image_path"],
            "image_dimensions": {"width": width, "height": height} if width and height else None,
            "coord_mode": coord_mode,
            "processed_box": box_coords,
            "include_reasoning": include_reasoning,
            "dataset": "vabench-point-bbox",
            "answer_format": "box"  # Mark the format as box instead of point
        }
        
        swift_data.append(swift_sample)
    
    # Save as JSONL file
    output_filename = f"vabench_bbox_{coord_mode}"
    if not include_reasoning:
        output_filename += "_no_reasoning"
    output_filename += "_ms_swift.jsonl"
    
    output_file = os.path.join(output_dir, output_filename)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in swift_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nConversion completed!")
    print(f"Converted {len(swift_data)}/{len(original_data)} samples")
    print(f"Coordinate mode: {coord_mode}")
    print(f"Include reasoning: {include_reasoning}")
    print(f"Answer format: bounding box [x1, y1, x2, y2] in <answer> tags")
    print(f"Output saved to: {output_file}")
    
    # Save conversion summary
    summary = {
        "dataset": "vabench-point-bbox",
        "total_samples": len(swift_data),
        "original_samples": len(original_data),
        "conversion_ratio": convert_ratio,
        "coord_mode": coord_mode,
        "include_reasoning": include_reasoning,
        "answer_format": "box [x1, y1, x2, y2] in <answer> tags",
        "output_format": "ms-swift",
        "image_count": len(swift_data),
        "features_preserved": ["id", "problem", "bbox", "normalized_bbox", "image_path"],
        "image_dimensions_included": width is not None and height is not None,
        "image_path_type": "absolute",
        "normalized_bbox_precision": 2
    }
    
    summary_file = os.path.join(output_dir, f"conversion_summary_{coord_mode}")
    if not include_reasoning:
        summary_file += "_no_reasoning"
    summary_file += ".json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to: {summary_file}")
    
    return swift_data

def verify_conversion(output_file: str, num_samples: int = 3):
    """Verify the converted data format."""
    
    print(f"\nVerifying conversion: {output_file}")
    
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:num_samples]
    
    for i, line in enumerate(lines):
        sample = json.loads(line.strip())
        
        print(f"\n--- Sample {i} ---")
        print(f"Keys: {list(sample.keys())}")
        
        # Check required keys
        assert "messages" in sample, "Missing 'messages' key"
        assert "images" in sample, "Missing 'images' key"
        assert "meta_data" in sample, "Missing 'meta_data' key"
        
        # Check messages structure
        messages = sample["messages"]
        assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"
        assert messages[0]["role"] == "system", "First message should be system"
        assert messages[1]["role"] == "user", "Second message should be user"
        assert messages[2]["role"] == "assistant", "Third message should be assistant"
        
        # Check system prompt is in English
        system_content = messages[0]["content"]
        assert isinstance(system_content, str) and len(system_content) > 0, "System prompt should be non-empty string"
        
        # Check that system prompt mentions bounding box
        assert "bounding box" in system_content.lower() or "box" in system_content.lower(), \
            f"System prompt should mention bounding box: {system_content[:50]}..."
        
        # Check image path is absolute
        image_path = sample["images"][0]
        assert os.path.isabs(image_path), f"Image path should be absolute: {image_path}"
        
        # Check user prompt mentions bounding box
        user_content = messages[1]["content"]
        assert "bounding box" in user_content.lower() or "box" in user_content.lower(), \
            f"User prompt should mention bounding box: {user_content[:100]}..."
        
        # Check assistant answer uses the correct format
        assistant_content = messages[2]["content"]
        if "include_reasoning" in sample["meta_data"] and sample["meta_data"]["include_reasoning"]:
            assert "</think><answer>" in assistant_content, \
                f"Assistant answer with reasoning should contain </think><answer>: {assistant_content[:100]}..."
        assert "<answer>" in assistant_content and "</answer>" in assistant_content, \
            f"Assistant answer should contain <answer> tags: {assistant_content[:100]}..."
        
        # Check meta_data
        meta = sample["meta_data"]
        assert "id" in meta, "Missing id in meta_data"
        assert "coord_mode" in meta, "Missing coord_mode in meta_data"
        assert "include_reasoning" in meta, "Missing include_reasoning in meta_data"
        assert "answer_format" in meta and meta["answer_format"] == "box", "Missing or incorrect answer_format"
        
        # Check processed_box has 4 coordinates
        if "processed_box" in meta:
            assert len(meta["processed_box"]) == 4, f"Processed box should have 4 coordinates: {meta['processed_box']}"
        
        # Check normalized_bbox precision
        if meta.get("normalized_bbox"):
            for val in meta["normalized_bbox"]:
                if isinstance(val, float):
                    # Check if it's formatted to 2 decimal places
                    decimal_str = str(val).split('.')[1] if '.' in str(val) else '0'
                    assert len(decimal_str) <= 2, f"normalized_bbox should have at most 2 decimal places: {val}"
        
        print(f"✓ Format correct")
        print(f"  System (English): {messages[0]['content'][:50]}...")
        print(f"  User prompt modified: {'bounding box' in messages[1]['content'].lower()}")
        print(f"  Assistant answer format: {messages[2]['content'][:100]}...")
        print(f"  Image (absolute): {image_path}")
        print(f"  Coord mode: {meta['coord_mode']}")
        print(f"  Include reasoning: {meta['include_reasoning']}")
        print(f"  Answer format: bounding box in <answer> tags")
    
    print(f"\n✅ All {num_samples} samples verified successfully!")

def main():
    parser = argparse.ArgumentParser(
        description="Convert vabench-point-bbox dataset to ms-swift format with coordinate normalization (box format)"
    )
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True,
        help="Path to the metadata.json file"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/preprocess_data/all_eval_data/vabench-point-bbox",
        help="Output directory for converted data"
    )
    
    parser.add_argument(
        "--convert_ratio", 
        type=float, 
        default=1.0,
        help="Ratio of data to convert (0.0-1.0, for debugging)"
    )
    
    parser.add_argument(
        "--coord_mode",
        type=str,
        default="thousand",
        choices=["absolute", "thousand", "normalized"],
        help="Coordinate normalization mode: 'absolute' (Qwen2.5-VL), 'thousand' (Qwen2/3-VL, InternVL2.5), or 'normalized' (0-1)"
    )
    
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Custom system prompt (default: English robot assistant prompt for bounding box)"
    )
    
    parser.add_argument(
        "--no_reasoning",
        action="store_true",
        help="Exclude reasoning process in assistant answer and modify user prompt to directly provide answer"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify conversion after completion"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    if args.convert_ratio <= 0 or args.convert_ratio > 1:
        raise ValueError("convert_ratio must be between 0.0 and 1.0")
    
    print("=" * 60)
    print("vabench-point-bbox to ms-swift Converter (Box Format)")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Conversion ratio: {args.convert_ratio}")
    print(f"Coordinate mode: {args.coord_mode}")
    print(f"Include reasoning: {not args.no_reasoning}")
    print(f"Answer format: Bounding Box [x1, y1, x2, y2] in <answer> tags")
    print(f"System prompt language: English")
    print(f"Image paths: Absolute")
    
    if args.coord_mode == "thousand":
        print("Note: Using thousand-level normalization (0-1000) for Qwen2/3-VL, InternVL2.5")
    elif args.coord_mode == "absolute":
        print("Note: Using absolute coordinates for Qwen2.5-VL")
    elif args.coord_mode == "normalized":
        print("Note: Using normalized coordinates (0-1) with 2 decimal places")
    
    if args.no_reasoning:
        print("Note: Modified user prompts to ask for direct answer")
    
    # Perform conversion
    converted_data = convert_vabench_to_swift(
        input_file=args.input_file,
        output_dir=args.output_dir,
        convert_ratio=args.convert_ratio,
        coord_mode=args.coord_mode,
        system_prompt=args.system_prompt,
        include_reasoning=not args.no_reasoning
    )
    
    # Verification
    if args.verify and converted_data:
        output_filename = f"vabench_bbox_{args.coord_mode}"
        if args.no_reasoning:
            output_filename += "_no_reasoning"
        output_filename += "_swift.jsonl"
        
        output_file = os.path.join(args.output_dir, output_filename)
        if os.path.exists(output_file):
            verify_conversion(output_file)
    
    print("\n✅ Conversion completed successfully!")

if __name__ == "__main__":
    main()
