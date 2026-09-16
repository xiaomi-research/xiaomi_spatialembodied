# UPDATE: Replace placeholder paths with your actual paths.
import argparse
import json
import os
import random
from PIL import Image, ImageDraw, ImageFont
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize bounding boxes and points from embodied-r1 JSONL data.")
    parser.add_argument("--input", type=str, default="data/preprocess_data/all_eval_data/embodied-r1/embodied-r1_ms_swift_test.jsonl",
                        help="Path to the input JSONL file.")
    parser.add_argument("--output", type=str, default="results/visualization_embodied",
                        help="Output directory for visualization images. Default: ./visualizations")
    parser.add_argument("--ratio", type=float, default=0.1,
                        help="Sampling ratio (0.0 to 1.0). Default: 0.1")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of samples to visualize. Default: 100")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. Default: 42")
    return parser.parse_args()

def draw_visualization(image_path, bbox, points, instruction, output_path, img_id):
    """
    Draw bounding box and points on the image and save.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image {image_path}: {e}")
        return
    draw = ImageDraw.Draw(img)

    # Draw bounding box (red)
    if bbox and len(bbox) == 4:
        draw.rectangle(bbox, outline="red", width=3)

    # Draw points (green circles)
    if points:
        for point in points:
            x, y = point
            draw.ellipse([x-4, y-4, x+4, y+4], fill="green")

    # Add instruction text (first 50 chars) at the top
    try:
        # Use default font, may need to adjust for different systems
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    # Truncate instruction if too long
    text = instruction[:100] + "..." if len(instruction) > 100 else instruction
    # Draw black background rectangle for text
    text_bbox = draw.textbbox((10, 10), text, font=font)
    draw.rectangle([text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5], fill="black")
    draw.text((10, 10), text, fill="white", font=font)

    # Save image
    img.save(output_path)
    print(f"Saved visualization for {img_id} to {output_path}")

def main():
    args = parse_args()
    random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Read JSONL file
    lines = []
    with open(args.input, 'r') as f:
        for line in f:
            lines.append(line.strip())
    total = len(lines)
    print(f"Total lines in JSONL: {total}")

    # Determine number of samples
    n_samples = min(int(total * args.ratio), args.max_samples, total)
    print(f"Sampling {n_samples} out of {total} lines.")

    # Randomly sample indices
    sampled_indices = random.sample(range(total), n_samples)

    for idx, line_idx in enumerate(sampled_indices):
        line = lines[line_idx]
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON at line {line_idx}: {e}")
            continue

        # Extract image path
        if "images" in data and isinstance(data["images"], list) and len(data["images"]) > 0:
            image_path = data["images"][0]  # assuming first image
        else:
            print(f"Line {line_idx}: No image path found.")
            continue

        # Extract bbox and points from meta_data
        bbox = None
        points = None
        instruction = ""

        if "meta_data" in data:
            meta = data["meta_data"]
            if "original_answer_parsed" in meta:
                parsed = meta["original_answer_parsed"]
                if parsed.get("type") == "fsd_free_point" and "data" in parsed:
                    data_dict = parsed["data"]
                    bbox = data_dict.get("bbox")
                    points = data_dict.get("free_points")
            # Also get id for filename
            img_id = meta.get("id", f"sample_{idx}")
        else:
            img_id = f"sample_{idx}"

        # Extract instruction from messages
        if "messages" in data:
            for msg in data["messages"]:
                if msg.get("role") == "user":
                    instruction = msg.get("content", "")
                    break

        # Check if bbox and points are valid
        if bbox is None and points is None:
            print(f"Line {line_idx}: No bbox or points found.")
            continue

        # Draw and save
        output_filename = f"{img_id}.jpg"
        output_path = os.path.join(args.output, output_filename)
        draw_visualization(image_path, bbox, points, instruction, output_path, img_id)

    print(f"Visualization completed. Saved {n_samples} images to {args.output}")

if __name__ == "__main__":
    main()
