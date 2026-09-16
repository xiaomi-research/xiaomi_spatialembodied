# UPDATE: Replace placeholder paths with your actual paths.
import json
import ast
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_bbox_on_image(image_path, bbox, output_path=None, line_width=3, color='red'):
    """
    Draw bounding box on image and save

    Args:
        image_path: Image path
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        output_path: Output image path, auto-generated if None
        line_width: Border line width
        color: Border color
    """
    # Open image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Parse bbox coordinates
    if isinstance(bbox, str):
        bbox = ast.literal_eval(bbox)

    # Draw rectangle
    draw.rectangle(bbox, outline=color, width=line_width)

    # Generate output path
    if output_path is None:
        # Add _bbox suffix to original image filename
        dir_name, file_name = os.path.split(image_path)
        name, ext = os.path.splitext(file_name)
        output_path = os.path.join(dir_name, f"{name}_bbox{ext}")

    # Save image
    img.save(output_path)
    print(f"Image saved to: {output_path}")

    return output_path


def draw_bbox_with_matplotlib(image_path, bbox, output_path=None, line_width=2, edgecolor='red', facecolor='none'):
    """
    Draw bounding box using matplotlib (supports more customization options)

    Args:
        image_path: Image path
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        output_path: Output image path
        line_width: Border line width
        edgecolor: Border color
        facecolor: Fill color
    """
    # Open image
    img = plt.imread(image_path)

    # Parse bbox coordinates
    if isinstance(bbox, str):
        bbox = ast.literal_eval(bbox)

    # Create figure
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    # Calculate rectangle width and height
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Create rectangle patch
    rect = patches.Rectangle(
        (x1, y1), width, height,
        linewidth=line_width,
        edgecolor=edgecolor,
        facecolor=facecolor
    )

    # Add to axes
    ax.add_patch(rect)

    # Remove axes
    ax.axis('off')

    # Generate output path
    if output_path is None:
        dir_name, file_name = os.path.split(image_path)
        name, ext = os.path.splitext(file_name)
        output_path = os.path.join(dir_name, f"{name}_bbox_matplotlib{ext}")

    # Save image
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"Image saved to: {output_path}")

    return output_path


def visualize_from_jsonl(jsonl_path, sample_index=1, use_matplotlib=True):
    """
    Read data from JSONL file and visualize

    Args:
        jsonl_path: JSONL file path
        sample_index: Data index to visualize (starting from 0)
        use_matplotlib: Whether to use matplotlib (otherwise use PIL)
    """
    # Read JSONL file
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    # Get data at specified index
    if sample_index < 0 or sample_index >= len(lines):
        print(f"Index {sample_index} out of range, file has {len(lines)} records")
        return

    data = json.loads(lines[sample_index])

    # Extract image path and bbox
    image_path = data['images'][0]
    bbox_str = data['meta_data']['original_bbox']

    print(f"Image path: {image_path}")
    print(f"Bounding Box: {bbox_str}")

    # Visualize
    if use_matplotlib:
        output_path = draw_bbox_with_matplotlib(
            image_path,
            bbox_str,
            edgecolor='lime',
            line_width=3
        )
    else:
        output_path = draw_bbox_on_image(
            image_path,
            bbox_str,
            color='red',
            line_width=3
        )

    return output_path


def visualize_specific_bbox(image_path, bbox_str, output_suffix="_visualized"):
    """
    Visualize specified image and bbox

    Args:
        image_path: Image path
        bbox_str: Bbox string, e.g. "[239.0, 340.0, 463.0, 402.0]"
        output_suffix: Output file suffix
    """
    if not os.path.exists(image_path):
        print(f"Image does not exist: {image_path}")
        return

    # Parse bbox
    try:
        bbox = ast.literal_eval(bbox_str)
        print(f"BBox coordinates: {bbox}")
    except:
        print(f"Cannot parse bbox string: {bbox_str}")
        return

    # Draw using PIL
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline='red', width=4)

    # Save image
    dir_name, file_name = os.path.split(image_path)
    name, ext = os.path.splitext(file_name)
    output_dir = "./debug"
    os.makedirs(os.path.join(output_dir, 'visualization_where2place'), exist_ok=True)
    output_path = os.path.join(output_dir, 'visualization_where2place', f"{name}{output_suffix}{ext}")
    img.save(output_path)

    print(f"Visualization result saved to: {output_path}")
    return output_path


def batch_visualize(jsonl_path, num_samples=5, output_dir="visualizations"):
    """
    Batch visualize multiple samples

    Args:
        jsonl_path: JSONL file path
        num_samples: Number of samples to visualize
        output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read data
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    num_samples = min(num_samples, len(lines))

    for i in range(num_samples):
        data = json.loads(lines[i])
        image_path = data['images'][0]
        bbox_str = data['meta_data']['original_bbox']

        # Parse bbox
        bbox = ast.literal_eval(bbox_str)

        # Draw using PIL
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline='red', width=3)

        # Save
        file_name = os.path.basename(image_path)
        name, ext = os.path.splitext(file_name)
        output_path = os.path.join(output_dir, f"{i}_{name}_bbox{ext}")
        img.save(output_path)

        print(f"Saved: {output_path}")


# Usage example
if __name__ == "__main__":
    # JSONL file path
    jsonl_path = "data/preprocess_data/where2place/where2place_ms_swift_absolute.jsonl"

    # Example 1: Visualize the second bbox
    print("Example 1: Visualize second sample")
    image_path = "data/public_robo_datasets/where2place/images/01.jpg"
    bbox_str = "[239.0, 340.0, 463.0, 402.0]"

    # Visualize and save
    output_path = visualize_specific_bbox(image_path, bbox_str)

    # Example 2: Read from JSONL file and visualize
    print("\nExample 2: Read from JSONL file and visualize")
    if os.path.exists(jsonl_path):
        visualize_from_jsonl(jsonl_path, sample_index=1, use_matplotlib=True)
    else:
        print(f"[SKIP] Example 2: {jsonl_path} does not exist")

    # Example 3: Batch visualize first 3 samples
    print("\nExample 3: Batch visualize first 3 samples")
    # `batch_visualize` used to be called unconditionally, so a missing JSONL raised
    # FileNotFoundError instead of the friendly skip above.
    if os.path.exists(jsonl_path):
        batch_visualize(jsonl_path, num_samples=3)
    else:
        print(f"[SKIP] Example 3: {jsonl_path} does not exist")
