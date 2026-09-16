# UPDATE: Replace placeholder paths with your actual paths.
#!/usr/bin/env python3
"""
RoboRefit-SFT dataset visualization script (with Box and Points drawing)
Function: Save dataset images and conversation content to local directory, and draw boxes and points
Dependencies: pip install pillow numpy pandas
"""

import argparse
import json
import os
import sys
import shutil
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

def create_output_dirs(output_dir):
    """Create output directory structure"""
    dirs = {
        'images': os.path.join(output_dir, 'images'),
        'conversations': os.path.join(output_dir, 'conversations'),
        'html': output_dir,
        'metadata': output_dir
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs

def parse_coordinates(text, divide_by_1000=False):
    """Parse coordinate strings, supporting multiple formats"""
    coordinates = []

    # Match multiple coordinate formats
    patterns = [
        r'\((\d+),(\d+)\)',  # (x,y)
        r'(\d+),(\d+)',      # x,y
        r'(\d+)\s*,\s*(\d+)' # May have spaces
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            break

    for match in matches:
        try:
            x, y = int(match[0]), int(match[1])
            if divide_by_1000:
                x, y = x / 1000, y / 1000
            coordinates.append((x, y))
        except (ValueError, IndexError):
            continue

    return coordinates

def parse_boxes_and_points(content, divide_by_1000=False):
    """Parse boxes and points from conversation content"""
    boxes = []
    points = []

    # Parse boxes
    box_patterns = [
        r'<box>\s*(.*?)\s*</box>',
        r'box:\s*(.*?)(?:\n|$)',
        r'box\s*\((.*?)\)'
    ]

    for pattern in box_patterns:
        box_matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        if box_matches:
            for box_match in box_matches:
                coords = parse_coordinates(box_match, divide_by_1000)
                if len(coords) >= 2:  # At least two points needed for rectangle
                    boxes.append(coords[:2])  # Take first two points
            break

    # Parse points
    point_patterns = [
        r'<points>\s*(.*?)\s*</points>',
        r'points:\s*(.*?)(?:\n|$)',
        r'point\s*\((.*?)\)',
        r'points\s*\((.*?)\)'
    ]

    for pattern in point_patterns:
        point_matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        if point_matches:
            for point_match in point_matches:
                coords = parse_coordinates(point_match, divide_by_1000)
                points.extend(coords)
            break

    return boxes, points

def get_conversation_summary(conversation_data, max_length=200):
    """Extract summary from conversation data"""
    summary = ""

    if isinstance(conversation_data, list):
        for i, turn in enumerate(conversation_data, 1):
            if isinstance(turn, dict):
                role = turn.get('from', turn.get('role', turn.get('speaker', 'unknown')))
                content = turn.get('value', turn.get('content', turn.get('text', '')))

                if isinstance(content, str):
                    # Extract box and points information
                    boxes, points = parse_boxes_and_points(content, divide_by_1000=False)
                    if boxes or points:
                        summary += f"[{role}]: "
                        if boxes:
                            summary += f"Boxes: {len(boxes)} "
                        if points:
                            summary += f"Points: {len(points)} "
                        # Truncate overly long content
                        if len(content) > 50:
                            content_preview = content[:50] + "..."
                        else:
                            content_preview = content
                        summary += f"{content_preview}\n"
                    else:
                        # Truncate overly long content
                        if len(content) > 50:
                            content_preview = content[:50] + "..."
                        else:
                            content_preview = content

                        summary += f"[{role}]: {content_preview}\n"
    elif isinstance(conversation_data, dict):
        # If dict, try to find key fields
        for key in ['conversation', 'dialogue', 'messages', 'text', 'content']:
            if key in conversation_data:
                value = str(conversation_data[key])
                if len(value) > 100:
                    summary = value[:100] + "..."
                else:
                    summary = value
                break

    if not summary and conversation_data:
        summary = str(conversation_data)[:200]

    return summary.strip()

def draw_boxes_and_points(image, boxes, points, conversation_summary=""):
    """Draw boxes and points on image"""
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    # Draw boxes (rectangles)
    for i, box in enumerate(boxes):
        if len(box) >= 2:
            # Ensure coordinates are integers
            x1, y1 = int(box[0][0]), int(box[0][1])
            x2, y2 = int(box[1][0]), int(box[1][1])

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width-1))
            y1 = max(0, min(y1, img_height-1))
            x2 = max(0, min(x2, img_width-1))
            y2 = max(0, min(y2, img_height-1))

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

            # Add box label
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()

            # Show number in top-left corner of box
            draw.text((x1+5, y1+5), f"Box{i+1}", fill='red', font=font)

    # Draw points (dots)
    for i, point in enumerate(points):
        if len(point) >= 2:
            # Ensure coordinates are integers
            x, y = int(point[0]), int(point[1])

            # Ensure coordinates are within image bounds
            x = max(0, min(x, img_width-1))
            y = max(0, min(y, img_height-1))

            # Draw point (small circle)
            radius = 5
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill='blue', outline='white', width=2)

            # Add point label
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                except:
                    font = ImageFont.load_default()

            # Show number next to point
            draw.text((x+10, y-10), f"P{i+1}", fill='blue', font=font)

    # Add annotation information at bottom of image
    if conversation_summary:
        # Calculate text area
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()

        # Split text into multiple lines
        max_width = img_width - 20
        lines = []
        words = conversation_summary.split()
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            # Estimate text width
            bbox = font.getbbox(test_line) if hasattr(font, 'getbbox') else (0, 0, len(test_line)*8, 20)
            text_width = bbox[2] - bbox[0] if len(bbox) > 2 else len(test_line)*8

            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # Add text area
        line_height = 20
        text_height = len(lines) * line_height + 20

        # Create new image (add space at bottom)
        new_img = Image.new('RGB', (img_width, img_height + text_height), color='white')
        new_img.paste(image, (0, 0))

        # Draw text
        draw = ImageDraw.Draw(new_img)
        for i, line in enumerate(lines):
            y_offset = img_height + 10 + i * line_height
            draw.text((10, y_offset), line, fill='black', font=font)

        return new_img

    return image

def save_conversation_as_text(conversation_data, filepath, sample_id, boxes=None, points=None):
    """Save conversation as text file, including box and points information"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Sample ID: {sample_id}\n")
        f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        if boxes or points:
            f.write("Parsed annotation information:\n")
            f.write("-" * 40 + "\n")
            if boxes:
                f.write(f"Boxes (total {len(boxes)}):\n")
                for i, box in enumerate(boxes, 1):
                    f.write(f"  Box{i}: {box}\n")
            if points:
                f.write(f"Points (total {len(points)}):\n")
                for i, point in enumerate(points, 1):
                    f.write(f"  Point{i}: {point}\n")
            f.write("\n")

        if isinstance(conversation_data, list):
            for i, turn in enumerate(conversation_data, 1):
                f.write(f"Turn {i}:\n")
                f.write("-" * 40 + "\n")

                if isinstance(turn, dict):
                    for key, value in turn.items():
                        if key == 'content' and isinstance(value, str):
                            # Parse and display boxes and points
                            parsed_boxes, parsed_points = parse_boxes_and_points(value, divide_by_1000=False)
                            f.write(f"{key}: {value}\n")
                            if parsed_boxes:
                                f.write(f"  -> Parsed boxes: {parsed_boxes}\n")
                            if parsed_points:
                                f.write(f"  -> Parsed points: {parsed_points}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                else:
                    f.write(f"{turn}\n")

                f.write("\n")
        elif isinstance(conversation_data, dict):
            for key, value in conversation_data.items():
                f.write(f"{key}: {value}\n")
        else:
            f.write(f"{conversation_data}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Conversation ended\n")

def add_text_to_image(image_path, text, output_path, max_width=30):
    """Add text description below image (split multi-line text)"""
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Calculate text area height
        # Approximately 20 pixels per line, plus margins
        lines = []
        words = text.split()
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            if len(test_line) <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        text_height = len(lines) * 20 + 20  # 20 pixels line height, 10 pixels top and bottom margins

        # Create new image
        new_img = Image.new('RGB', (img_width, img_height + text_height), color='white')
        new_img.paste(img, (0, 0))

        # Draw text
        draw = ImageDraw.Draw(new_img)

        # Try to use default font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()

        # Draw each line of text
        y_offset = img_height + 10
        for i, line in enumerate(lines):
            draw.text((10, y_offset + i*20), line, fill='black', font=font)

        new_img.save(output_path)
        return True
    except Exception as e:
        print(f"  [WARN] Unable to add text to image: {e}")
        # If adding text fails, copy original image directly
        try:
            shutil.copy2(image_path, output_path)
            return True
        except:
            return False

def process_sample(item, base_dir, output_dirs, index, total, divide_by_1000=False, add_text=False):
    """Process a single sample"""
    sample_id = item['id']
    print(f"[{index+1}/{total}] Processing sample: {sample_id}")

    result = {
        'id': sample_id,
        'index': index,
        'has_image': item.get('has_image', False),
        'has_conversation': item.get('has_conversation', False),
        'image_saved': False,
        'conversation_saved': False,
        'boxes_count': 0,
        'points_count': 0,
        'image_path': '',
        'conversation_path': '',
        'conversation_summary': ''
    }

    all_boxes = []
    all_points = []

    # Process conversation
    if result['has_conversation'] and 'conversation_path' in item:
        conv_source_path = os.path.join(base_dir, item['conversation_path'])
        conv_output_path = os.path.join(output_dirs['conversations'], f"{sample_id}.txt")

        if os.path.exists(conv_source_path):
            try:
                with open(conv_source_path, 'r', encoding='utf-8') as f:
                    conversation_data = json.load(f)

                # Extract boxes and points from all conversation content
                if isinstance(conversation_data, list):
                    for turn in conversation_data:
                        if isinstance(turn, dict) and 'content' in turn:
                            content = turn.get('content', '')
                            if isinstance(content, str):
                                boxes, points = parse_boxes_and_points(content, divide_by_1000)
                                all_boxes.extend(boxes)
                                all_points.extend(points)
                elif isinstance(conversation_data, dict):
                    # Handle dict format
                    for key, value in conversation_data.items():
                        if isinstance(value, str):
                            boxes, points = parse_boxes_and_points(value, divide_by_1000)
                            all_boxes.extend(boxes)
                            all_points.extend(points)

                result['boxes_count'] = len(all_boxes)
                result['points_count'] = len(all_points)

                print(f"  Parsed: Boxes={result['boxes_count']}, Points={result['points_count']}")

                # Save conversation as text file
                save_conversation_as_text(conversation_data, conv_output_path, sample_id, all_boxes, all_points)

                # Extract conversation summary
                result['conversation_summary'] = get_conversation_summary(conversation_data)
                print(f"  Conversation summary: {result['conversation_summary'][:100]}...")

                result['conversation_saved'] = True
                result['conversation_path'] = conv_output_path

            except json.JSONDecodeError as e:
                print(f"  [ERROR] Conversation JSON parse failed: {e}")
                # Try to copy file directly
                try:
                    shutil.copy2(conv_source_path, conv_output_path)
                    result['conversation_saved'] = True
                except:
                    pass
            except Exception as e:
                print(f"  [ERROR] Cannot process conversation {conv_source_path}: {e}")
        else:
            print(f"  [WARN] Conversation file does not exist: {conv_source_path}")

    # Process image
    if result['has_image'] and 'image_path' in item:
        img_source_path = os.path.join(base_dir, item['image_path'])
        img_output_path = os.path.join(output_dirs['images'], f"{sample_id}_annotated.jpg")
        img_original_path = os.path.join(output_dirs['images'], f"{sample_id}_original.jpg")

        if os.path.exists(img_source_path):
            try:
                # Open image and check basic info
                with Image.open(img_source_path) as img:
                    img = img.convert('RGB')  # Ensure RGB format
                    img_info = f"Size: {img.size}, Format: {img.format}, Mode: {img.mode}"
                    print(f"  Image: {img_info}")

                # Save original image
                shutil.copy2(img_source_path, img_original_path)
                print(f"  Saved original image: {img_original_path}")

                # If there is annotation information, draw on image
                if all_boxes or all_points:
                    # Reopen image for drawing
                    img = Image.open(img_source_path).convert('RGB')

                    # Draw boxes and points
                    if all_boxes or all_points:
                        img_with_annotations = draw_boxes_and_points(
                            img,
                            all_boxes,
                            all_points,
                            f"Boxes: {len(all_boxes)}, Points: {len(all_points)}"
                        )
                        img_with_annotations.save(img_output_path)
                        print(f"  Saved annotated image: {img_output_path}")
                    else:
                        # No annotation information, save directly
                        img.save(img_output_path)

                    result['image_saved'] = True
                    result['image_path'] = img_output_path
                else:
                    # No annotation information, copy directly
                    shutil.copy2(img_source_path, img_output_path)
                    result['image_saved'] = True
                    result['image_path'] = img_output_path
                    print(f"  No annotation info, saved original image: {img_output_path}")

            except Exception as e:
                print(f"  [ERROR] Cannot process image {img_source_path}: {e}")
        else:
            print(f"  [WARN] Image file does not exist: {img_source_path}")

    return result

def generate_html_report(results, output_dir, base_dir, metadata_path, start_idx, max_samples):
    """Generate HTML report"""
    html_path = os.path.join(output_dir, 'report.html')

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('<!DOCTYPE html>\n')
        f.write('<html>\n')
        f.write('<head>\n')
        f.write('<meta charset="UTF-8">\n')
        f.write('<title>RoboRefit-SFT Dataset Visualization Report (with Box/Points Annotations)</title>\n')
        f.write('<style>\n')
        f.write('body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }\n')
        f.write('h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }\n')
        f.write('.summary { background-color: #fff; padding: 20px; border-radius: 5px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }\n')
        f.write('.sample { background-color: #fff; padding: 20px; margin-bottom: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }\n')
        f.write('.sample-header { background-color: #4CAF50; color: white; padding: 10px; border-radius: 3px; margin-bottom: 10px; }\n')
        f.write('.image-container { margin-bottom: 10px; text-align: center; }\n')
        f.write('img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }\n')
        f.write('.conversation { background-color: #f9f9f9; padding: 15px; border-left: 4px solid #4CAF50; margin-top: 10px; white-space: pre-wrap; font-family: monospace; }\n')
        f.write('.stats { display: flex; justify-content: space-between; flex-wrap: wrap; }\n')
        f.write('.stat-item { background-color: #e8f5e9; padding: 10px; border-radius: 4px; margin: 5px; flex: 1; min-width: 150px; }\n')
        f.write('.legend { background-color: #fff; padding: 10px; border-radius: 4px; margin: 10px 0; border-left: 4px solid #f44336; }\n')
        f.write('.legend-item { display: inline-block; margin-right: 20px; }\n')
        f.write('.legend-box { display: inline-block; width: 20px; height: 5px; background-color: red; margin-right: 5px; vertical-align: middle; }\n')
        f.write('.legend-point { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background-color: blue; margin-right: 5px; vertical-align: middle; }\n')
        f.write('</style>\n')
        f.write('</head>\n')
        f.write('<body>\n')

        # Title and summary
        f.write('<h1>RoboRefit-SFT Dataset Visualization Report (with Box/Points Annotations)</h1>\n')
        f.write(f'<p>Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>\n')

        # Legend
        f.write('<div class="legend">\n')
        f.write('<strong>Legend:</strong>\n')
        f.write('<div class="legend-item"><span class="legend-box"></span> Box (red rectangle)</div>\n')
        f.write('<div class="legend-item"><span class="legend-point"></span> Point (blue dot)</div>\n')
        f.write('</div>\n')

        # Statistics
        total = len(results)
        images_saved = sum(1 for r in results if r['image_saved'])
        conversations_saved = sum(1 for r in results if r['conversation_saved'])
        both_saved = sum(1 for r in results if r['image_saved'] and r['conversation_saved'])
        total_boxes = sum(r['boxes_count'] for r in results)
        total_points = sum(r['points_count'] for r in results)

        f.write('<div class="summary">\n')
        f.write('<h2>Dataset Statistics</h2>\n')
        f.write(f'<p><strong>Base directory:</strong> {base_dir}</p>\n')
        f.write(f'<p><strong>Metadata file:</strong> {metadata_path}</p>\n')
        f.write(f'<p><strong>Processing range:</strong> index {start_idx} to {start_idx + total - 1}</p>\n')
        f.write('<div class="stats">\n')
        f.write(f'<div class="stat-item"><strong>Total samples:</strong> {total}</div>\n')
        f.write(f'<div class="stat-item"><strong>Images saved:</strong> {images_saved}</div>\n')
        f.write(f'<div class="stat-item"><strong>Conversations saved:</strong> {conversations_saved}</div>\n')
        f.write(f'<div class="stat-item"><strong>Both saved:</strong> {both_saved}</div>\n')
        f.write(f'<div class="stat-item"><strong>Total boxes:</strong> {total_boxes}</div>\n')
        f.write(f'<div class="stat-item"><strong>Total points:</strong> {total_points}</div>\n')
        f.write('</div>\n')
        f.write('</div>\n')

        # Sample details
        f.write('<h2>Sample Details</h2>\n')
        for i, result in enumerate(results):
            f.write('<div class="sample">\n')
            f.write(f'<div class="sample-header">Sample {i+1}/{total}: {result["id"]}</div>\n')

            f.write(f'<p><strong>Index:</strong> {result["index"]}</p>\n')
            f.write(f'<p><strong>Has image:</strong> {result["has_image"]}</p>\n')
            f.write(f'<p><strong>Has conversation:</strong> {result["has_conversation"]}</p>\n')
            f.write(f'<p><strong>Box count:</strong> {result["boxes_count"]}</p>\n')
            f.write(f'<p><strong>Point count:</strong> {result["points_count"]}</p>\n')

            if result['image_saved']:
                rel_img_path = os.path.relpath(result['image_path'], output_dir)
                f.write('<div class="image-container">\n')
                f.write(f'<p><strong>Annotated image:</strong></p>\n')
                f.write(f'<img src="{rel_img_path}" alt="{result["id"]}">\n')
                f.write('</div>\n')

            if result['conversation_saved']:
                conv_content = ""
                try:
                    with open(result['conversation_path'], 'r', encoding='utf-8') as conv_file:
                        conv_content = conv_file.read()
                except:
                    conv_content = "Cannot read conversation file"

                f.write('<div class="conversation-container">\n')
                f.write(f'<p><strong>Conversation summary:</strong> {result.get("conversation_summary", "No summary")}</p>\n')
                f.write(f'<p><strong>Full conversation:</strong></p>\n')
                f.write(f'<div class="conversation">{conv_content}</div>\n')
                f.write('</div>\n')

            f.write('</div>\n')

        f.write('</body>\n')
        f.write('</html>\n')

    return html_path

def save_dataset_visualization(metadata_path, base_dir, output_dir, start_idx=0, max_samples=None, divide_by_1000=False, add_text=False):
    """
    Save dataset visualization results to local directory

    Args:
        metadata_path: Full path to metadata.json file
        base_dir: Dataset base directory
        output_dir: Output directory
        start_idx: Start index
        max_samples: Maximum number of samples to process
        divide_by_1000: Whether to divide coordinates by 1000 (thousandth scaling)
        add_text: Whether to add conversation summary text to images
    """
    try:
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        print(f"Loaded {len(metadata)} samples")
        print(f"Thousandth scaling: {'enabled' if divide_by_1000 else 'disabled'}")

        # Determine processing range
        if max_samples is None:
            end_idx = len(metadata)
        else:
            end_idx = min(start_idx + max_samples, len(metadata))

        total_to_process = end_idx - start_idx
        print(f"Will process {total_to_process} samples from index {start_idx} to {end_idx-1}")
        print(f"Output directory: {output_dir}")

        # Create output directory
        output_dirs = create_output_dirs(output_dir)
        print(f"Created directories: {output_dirs}")

        # Process each sample
        results = []
        for i in range(start_idx, end_idx):
            item = metadata[i]
            result = process_sample(item, base_dir, output_dirs, i - start_idx, total_to_process, divide_by_1000, add_text)
            results.append(result)

        # Generate statistics
        print("\n" + "="*60)
        print("Processing complete! Statistics:")
        print("="*60)

        processed = len(results)
        images_saved = sum(1 for r in results if r['image_saved'])
        conversations_saved = sum(1 for r in results if r['conversation_saved'])
        both_saved = sum(1 for r in results if r['image_saved'] and r['conversation_saved'])
        total_boxes = sum(r['boxes_count'] for r in results)
        total_points = sum(r['points_count'] for r in results)

        print(f"Total samples: {processed}")
        print(f"Images saved: {images_saved} ({images_saved/processed*100:.1f}%)")
        print(f"Conversations saved: {conversations_saved} ({conversations_saved/processed*100:.1f}%)")
        print(f"Both saved: {both_saved} ({both_saved/processed*100:.1f}%)")
        print(f"Total boxes: {total_boxes}")
        print(f"Total points: {total_points}")
        print(f"Average boxes per sample: {total_boxes/processed if processed > 0 else 0:.1f}")
        print(f"Average points per sample: {total_points/processed if processed > 0 else 0:.1f}")

        # Generate HTML report
        html_path = generate_html_report(results, output_dir, base_dir, metadata_path, start_idx, total_to_process)
        print(f"\nHTML report generated: {html_path}")

        # Save processing results to CSV
        csv_path = os.path.join(output_dir, 'results.csv')
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Results CSV saved: {csv_path}")

        # Generate brief README
        readme_path = os.path.join(output_dir, 'README.txt')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("RoboRefit-SFT Dataset Visualization Output (with Box/Points Annotations)\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source data directory: {base_dir}\n")
            f.write(f"Metadata file: {metadata_path}\n")
            f.write(f"Sample range: {start_idx} to {end_idx-1} ({total_to_process} total)\n")
            f.write(f"Thousandth scaling: {'Yes' if divide_by_1000 else 'No'}\n\n")
            f.write("Directory structure:\n")
            f.write("  images/ - Image files (with annotations)\n")
            f.write("  conversations/ - Conversation text files\n")
            f.write("  report.html - HTML visualization report\n")
            f.write("  results.csv - Processing results CSV file\n")
            f.write("  README.txt - This file\n\n")
            f.write("Annotation description:\n")
            f.write("  - Red rectangles: Box annotations\n")
            f.write("  - Blue dots: Points annotations\n")
            f.write(f"Statistics:\n")
            f.write(f"  Total samples: {processed}\n")
            f.write(f"  Images saved: {images_saved}\n")
            f.write(f"  Conversations saved: {conversations_saved}\n")
            f.write(f"  Both saved: {both_saved}\n")
            f.write(f"  Total boxes: {total_boxes}\n")
            f.write(f"  Total points: {total_points}\n")

        print(f"README file: {readme_path}")
        print(f"\nAll files saved to: {output_dir}")

        return results

    except FileNotFoundError as e:
        print(f"[ERROR] File not found - {e}")
        print(f"Please check path: {metadata_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse failed - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='RoboRefit-SFT dataset visualization tool (with Box/Points annotations)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Basic usage, save to default output directory
  python visualize_roborefit_boxes.py

  # Save first 20 samples, enable thousandth scaling
  python visualize_roborefit_boxes.py --max 20 --divide_by_1000

  # Specify output directory and processing range
  python visualize_roborefit_boxes.py --output ./roborefit_boxes_viz --start 0 --max 20

  # Full example: start from sample 10, save 50 samples, enable thousandth scaling
  python visualize_roborefit_boxes.py --base_dir "/path/to/dataset" --output "./my_output" --start 10 --max 50 --divide_by_1000
        """
    )

    # Use provided paths as defaults
    default_base_dir = "data/preprocess_data/roborefit-sft-dataset-30k"
    default_metadata = os.path.join(default_base_dir, "metadata.json")

    parser.add_argument('--metadata', type=str, default=default_metadata,
                       help=f'metadata.json file path (default: {default_metadata})')
    parser.add_argument('--base_dir', type=str, default=default_base_dir,
                       help=f'Dataset base directory (default: {default_base_dir})')
    parser.add_argument('--output', type=str, default='results/vis/roborefit',
                       help='Output directory (default: results/vis/roborefit)')
    parser.add_argument('--start', type=int, default=0,
                       help='Start sample index (default: 0)')
    parser.add_argument('--max', type=int, default=100,
                       help='Maximum number of samples to process (default: 10)')
    parser.add_argument('--divide_by_1000', action='store_true',
                       help='Divide coordinates by 1000 (thousandth scaling)')
    parser.add_argument('--add_text', action='store_true',
                       help='Add conversation summary text to images')

    args = parser.parse_args()

    print("="*80)
    print("RoboRefit-SFT Dataset Visualization Tool (with Box/Points Annotations)")
    print("="*80)
    print(f"Metadata file: {args.metadata}")
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output}")
    print(f"Start index: {args.start}")
    print(f"Maximum samples: {args.max}")
    print(f"Thousandth scaling: {'Yes' if args.divide_by_1000 else 'No'}")
    print(f"Add text to images: {'Yes' if args.add_text else 'No'}")
    print("="*80)

    # Check if metadata file exists
    if not os.path.exists(args.metadata):
        print(f"[ERROR] Metadata file does not exist: {args.metadata}")
        sys.exit(1)

    # Check if base_dir exists
    if not os.path.exists(args.base_dir):
        print(f"[WARN] Base directory does not exist: {args.base_dir}")
        print("Some files may not be found")

    # If output directory already exists, ask whether to overwrite
    if os.path.exists(args.output):
        response = input(f"Output directory {args.output} already exists. Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Operation cancelled")
            sys.exit(0)

    # Start processing
    save_dataset_visualization(
        args.metadata,
        args.base_dir,
        args.output,
        args.start,
        args.max,
        args.divide_by_1000,
        args.add_text
    )

if __name__ == "__main__":
    main()
