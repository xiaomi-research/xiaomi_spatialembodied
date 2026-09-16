# UPDATE: Replace placeholder paths with your actual paths.
import json
import os
import ast
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm
import random
import re


class RoboAffordVisualizer:
    """RoboAfford data visualization tool"""

    def __init__(self, output_dir="debug/visualization_roboafford"):
        """
        Initialize visualization tool

        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Predefined color list for different bboxes
        self.bbox_colors = [
            'red', 'blue', 'green', 'purple', 'orange',
            'cyan', 'magenta', 'yellow', 'lime', 'pink',
            'gold', 'navy', 'teal', 'maroon', 'olive'
        ]

        # Try to load font
        try:
            self.font = ImageFont.truetype("arial.ttf", 18)
        except:
            self.font = ImageFont.load_default()

    def parse_bbox_list(self, bbox_data):
        """
        Parse bounding box list data

        Args:
            bbox_data: Can be string, list, or list of lists

        Returns:
            Bounding box list [[x1, y1, x2, y2], ...]
        """
        if isinstance(bbox_data, str):
            # Try to extract bbox list from string
            try:
                # Try direct parsing
                bboxes = ast.literal_eval(bbox_data)
                if isinstance(bboxes, list):
                    if len(bboxes) > 0 and isinstance(bboxes[0], (int, float)):
                        # If single bbox
                        return [bboxes]
                    else:
                        # If bbox list
                        return bboxes
            except:
                # Try to extract from <answer> tags
                match = re.search(r'<answer>\[(.*?)\]</answer>', bbox_data)
                if match:
                    bbox_str = match.group(1)
                    try:
                        bboxes = ast.literal_eval(f'[{bbox_str}]')
                        return bboxes
                    except:
                        pass

        elif isinstance(bbox_data, list):
            if len(bbox_data) > 0:
                if isinstance(bbox_data[0], (int, float)):
                    # Single bbox
                    return [bbox_data]
                elif isinstance(bbox_data[0], list):
                    # Bbox list -- `bbox_data` itself is the list. This used to return
                    # `bboxes`, a name that is only bound inside the `str` branch above, so
                    # every entry reaching here died with UnboundLocalError.
                    return bbox_data
        return []

    def extract_question_summary(self, question_text, max_words=8):
        """
        Extract summary from question text

        Args:
            question_text: Question text
            max_words: Maximum number of words

        Returns:
            Summary string
        """
        # Remove newlines
        text = question_text.replace('\n', ' ').strip()

        # Extract first few words
        words = text.split()[:max_words]
        summary = ' '.join(words)

        # If too long, add ellipsis
        if len(text.split()) > max_words:
            summary += '...'

        return summary

    def draw_multiple_bboxes_pil(self, image_path, bbox_list, question_text="",
                                 output_path=None, line_width=2,
                                 show_numbers=True, show_question=True):
        """
        Draw multiple bboxes on image using PIL

        Args:
            image_path: Image path
            bbox_list: Bounding box list
            question_text: Question text
            output_path: Output path
            line_width: Line width
            show_numbers: Whether to show bbox numbers
            show_question: Whether to show question summary
        """
        # Open image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Parse bbox list
        bboxes = self.parse_bbox_list(bbox_list)

        if not bboxes:
            print("No valid bbox data found")
            return None

        print(f"Found {len(bboxes)} bounding boxes")

        # Draw each bbox
        for i, bbox in enumerate(bboxes):
            if len(bbox) != 4:
                print(f"Skipping invalid bbox: {bbox}")
                continue

            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            # Select color
            color_index = i % len(self.bbox_colors)
            color = self.bbox_colors[color_index]

            # Draw rectangle
            draw.rectangle(bbox, outline=color, width=line_width)

            # Show number
            if show_numbers:
                # Show number in top-left corner
                text = f"{i+1}"
                text_bbox = draw.textbbox((x1, y1-20), text, font=self.font)
                # Draw text background
                draw.rectangle(text_bbox, fill=color)
                # Draw text
                draw.text((x1, y1-20), text, fill='black', font=self.font)

            # Show coordinates inside rectangle
            coord_text = f"({int(x1)},{int(y1)})-({int(x2)},{int(y2)})"
            coord_font = ImageFont.load_default()  # Use small font for coordinates
            # Calculate text position (try to place in top-right inside box)
            text_x = x1 + 5
            text_y = y1 + 5

            # Check if there is enough space
            if height > 20:  # If box height is sufficient
                # Draw semi-transparent background
                text_width = len(coord_text) * 6
                text_height = 10
                draw.rectangle(
                    [text_x, text_y, text_x + text_width, text_y + text_height],
                    fill=(0, 0, 0, 128)
                )
                draw.text((text_x, text_y), coord_text, fill='white', font=coord_font)

        # Show question summary
        if show_question and question_text:
            # Extract question summary
            question_summary = self.extract_question_summary(question_text, max_words=10)

            # Add question text at top of image
            # Create text background
            img_width, img_height = img.size
            text_bg_height = 40

            # Create new image including text area
            new_img = Image.new('RGB', (img_width, img_height + text_bg_height), 'black')
            new_img.paste(img, (0, text_bg_height))

            # Draw text on new image
            new_draw = ImageDraw.Draw(new_img)

            # Calculate text position
            text_x = 10
            text_y = 10

            # Draw text
            new_draw.text((text_x, text_y), f"Q: {question_summary}",
                         fill='white', font=self.font)

            # Update image
            img = new_img

        # Generate output path
        if output_path is None:
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)

            # Extract keywords from question as filename
            if question_text:
                # Extract first few words
                words = question_text.lower().split()[:4]
                keywords = "_".join(words)
                output_name = f"{name}_{keywords[:30]}{ext}"
            else:
                output_name = f"{name}_multi_bbox{ext}"

            output_path = os.path.join(self.output_dir, output_name)

        # Save image
        img.save(output_path)
        print(f"Saved: {output_path}")
        return output_path

    def visualize_jsonl_entry(self, jsonl_path, entry_index=0,
                             save_individual=True, show_question=True):
        """
        Visualize a single entry from a JSONL file

        Args:
            jsonl_path: JSONL file path
            entry_index: Entry index
            save_individual: Whether to save as individual file
            show_question: Whether to show question
        """
        # Read JSONL file
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()

        if entry_index >= len(lines):
            print(f"Index {entry_index} out of range, file has {len(lines)} entries")
            return

        # Parse data
        data = json.loads(lines[entry_index])

        # Extract information
        image_path = data['images'][0]
        bbox_data = data['messages'][1]['content']  # assistant's reply
        question_text = data['messages'][1]['content']  # user's question

        # Actually, the user question is data['messages'][0]['content']
        if 'messages' in data and len(data['messages']) > 0:
            question_text = data['messages'][0]['content']

        # Check if image path exists
        if not os.path.exists(image_path):
            print(f"Image does not exist: {image_path}")
            return

        # Print information
        print(f"=== Entry {entry_index} ===")
        print(f"Image path: {image_path}")
        print(f"Question: {question_text[:100]}...")
        print(f"Answer: {bbox_data[:100]}...")

        # Extract bbox list
        bboxes = self.parse_bbox_list(bbox_data)
        print(f"Parsed {len(bboxes)} bounding boxes:")
        for i, bbox in enumerate(bboxes):
            print(f"  BBox {i+1}: {bbox}")

        # Visualize
        output_path = self.draw_multiple_bboxes_pil(
            image_path, bboxes, question_text,
            show_question=show_question
        )

        return output_path

    def visualize_with_mask(self, image_path, mask_path, bbox_list,
                           question_text="", output_suffix="_with_mask"):
        """
        Visualize with mask and bbox together

        Args:
            image_path: Image path
            mask_path: Mask path
            bbox_list: Bounding box list
            question_text: Question text
            output_suffix: Output file suffix
        """
        if not os.path.exists(image_path):
            print(f"Image does not exist: {image_path}")
            return

        # Open image
        img = Image.open(image_path).convert("RGBA")

        # Parse bbox
        bboxes = self.parse_bbox_list(bbox_list)

        # If mask exists, load and overlay
        if mask_path and os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path)

                # If mask is binary, convert to RGBA
                if mask.mode != 'RGBA':
                    mask = mask.convert("L")
                    # Create colored mask
                    mask_rgba = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    mask_pixels = mask.load()
                    rgba_pixels = mask_rgba.load()

                    for x in range(img.width):
                        for y in range(img.height):
                            if mask_pixels[x, y] > 0:
                                # Use red semi-transparent
                                rgba_pixels[x, y] = (255, 0, 0, 100)

                    # Overlay mask
                    img = Image.alpha_composite(img, mask_rgba)
            except Exception as e:
                print(f"Failed to load mask: {e}")

        # Draw bbox
        draw = ImageDraw.Draw(img)

        for i, bbox in enumerate(bboxes):
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            color_index = i % len(self.bbox_colors)
            color = self.bbox_colors[color_index]

            # Draw rectangle
            draw.rectangle(bbox, outline=color, width=3)

            # Add number
            text = f"{i+1}"
            text_bbox = draw.textbbox((x1, y1-20), text, font=self.font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1-20), text, fill='black', font=self.font)

        # Save
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(self.output_dir, f"{name}{output_suffix}{ext}")

        img.convert("RGB").save(output_path)
        print(f"Mask visualization saved: {output_path}")
        return output_path

    def batch_visualize(self, jsonl_path, num_samples=10,
                       random_sample=False, show_progress=True):
        """
        Batch visualize multiple samples

        Args:
            jsonl_path: JSONL file path
            num_samples: Number of samples to visualize
            random_sample: Whether to randomly sample
            show_progress: Whether to show progress bar
        """
        # Read JSONL file
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()

        # Determine indices to visualize
        if random_sample:
            indices = random.sample(range(len(lines)), min(num_samples, len(lines)))
        else:
            indices = list(range(min(num_samples, len(lines))))

        # Create progress bar
        if show_progress:
            indices_iter = tqdm(indices, desc="Visualization progress")
        else:
            indices_iter = indices

        saved_paths = []

        for idx in indices_iter:
            try:
                data = json.loads(lines[idx])

                # Extract information
                image_path = data['images'][0]
                bbox_data = data['messages'][1]['content']

                # Extract question text
                question_text = ""
                if 'messages' in data and len(data['messages']) > 0:
                    question_text = data['messages'][0]['content']

                # Parse bbox
                bboxes = self.parse_bbox_list(bbox_data)

                if not bboxes or not os.path.exists(image_path):
                    continue

                # Visualize
                output_path = self.draw_multiple_bboxes_pil(
                    image_path, bboxes, question_text,
                    show_question=True
                )

                if output_path:
                    saved_paths.append(output_path)

            except Exception as e:
                print(f"Error processing entry {idx}: {e}")

        print(f"\nBatch visualization complete, processed {len(saved_paths)} samples")
        print(f"Output directory: {self.output_dir}")

        return saved_paths

    def create_comparison_grid(self, jsonl_path, num_rows=2, num_cols=3):
        """
        Create comparison grid showing different question annotations for the same image

        Args:
            jsonl_path: JSONL file path
            num_rows: Number of rows
            num_cols: Number of columns
        """
        # Read JSONL file
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()

        # Group by image
        image_groups = {}
        for i, line in enumerate(lines[:50]):  # Only check first 50
            try:
                data = json.loads(line)
                image_path = data['images'][0]

                if image_path not in image_groups:
                    image_groups[image_path] = []

                image_groups[image_path].append({
                    'data': data,
                    'index': i
                })
            except:
                continue

        # Find images with multiple questions
        multi_question_images = {k: v for k, v in image_groups.items() if len(v) > 1}

        if not multi_question_images:
            print("No images with multiple questions found")
            return

        # Select the first image with multiple questions
        selected_image = list(multi_question_images.keys())[0]
        entries = multi_question_images[selected_image]

        # Create figure
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))

        if num_rows == 1 and num_cols == 1:
            axes = [[axes]]
        elif num_rows == 1:
            axes = [axes]
        elif num_cols == 1:
            axes = [[ax] for ax in axes]

        # Visualize each question
        for i, entry in enumerate(entries[:num_rows*num_cols]):
            row = i // num_cols
            col = i % num_cols

            data = entry['data']

            # Extract information
            image_path = data['images'][0]
            bbox_data = data['messages'][1]['content']
            question_text = data['messages'][0]['content']

            # Load image
            img = plt.imread(image_path)
            axes[row][col].imshow(img)

            # Parse bbox
            bboxes = self.parse_bbox_list(bbox_data)

            # Draw each bbox
            for j, bbox in enumerate(bboxes):
                if len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1

                color_index = j % len(self.bbox_colors)
                color = self.bbox_colors[color_index]

                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                axes[row][col].add_patch(rect)

                # Add number
                axes[row][col].text(x1, y1-5, f"{j+1}",
                                   color=color, fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.3",
                                            facecolor=color, alpha=0.7))

            # Set title (question summary)
            title = self.extract_question_summary(question_text, max_words=6)
            axes[row][col].set_title(title, fontsize=10)
            axes[row][col].axis('off')

        # Hide extra subplots
        for i in range(len(entries), num_rows*num_cols):
            row = i // num_cols
            col = i % num_cols
            axes[row][col].axis('off')

        # Set overall title
        base_name = os.path.basename(selected_image)
        plt.suptitle(f"RoboAfford: Different questions for {base_name}", fontsize=14)
        plt.tight_layout()

        # Save
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(self.output_dir, f"{name}_comparison_grid.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.3, dpi=150)
        plt.close()

        print(f"Comparison grid saved: {output_path}")
        return output_path

    def visualize_heatmap_overlay(self, jsonl_path, image_index=0, output_suffix="_heatmap"):
        """
        Create heatmap overlay for all bboxes of the same image

        Args:
            jsonl_path: JSONL file path
            image_index: Image index
            output_suffix: Output suffix
        """
        # Read JSONL file
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()

        # Find all entries for specified image
        target_image = None
        all_bboxes = []

        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                image_path = data['images'][0]
                bbox_data = data['messages'][1]['content']

                if i == image_index:
                    target_image = image_path

                if image_path == target_image:
                    bboxes = self.parse_bbox_list(bbox_data)
                    all_bboxes.extend(bboxes)
            except:
                continue

        if not target_image or not os.path.exists(target_image):
            print(f"Image does not exist: {target_image}")
            return

        if not all_bboxes:
            print("No bounding box data found")
            return

        print(f"Found {len(all_bboxes)} bounding boxes for heatmap")

        # Load image
        img = Image.open(target_image)
        img_array = np.array(img)

        # Create heatmap
        heatmap = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=float)

        for bbox in all_bboxes:
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(x1, img_array.shape[1]-1))
            x2 = max(0, min(x2, img_array.shape[1]-1))
            y1 = max(0, min(y1, img_array.shape[0]-1))
            y2 = max(0, min(y2, img_array.shape[0]-1))

            heatmap[y1:y2, x1:x2] += 1

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Show original image
        axes[0].imshow(img_array)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Show heatmap
        im = axes[1].imshow(heatmap, cmap='hot', alpha=0.7)
        axes[1].imshow(img_array, alpha=0.3)
        axes[1].set_title("Bounding Box Heatmap")
        axes[1].axis('off')

        # Add colorbar
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save
        base_name = os.path.basename(target_image)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(self.output_dir, f"{name}{output_suffix}{ext}")

        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"Heatmap saved: {output_path}")
        return output_path


# Quick visualization function
def quick_roboafford_visualize(jsonl_path, output_dir, num_samples=5):
    """
    Quickly visualize RoboAfford data

    Args:
        jsonl_path: JSONL file path
        output_dir: Output directory
        num_samples: Number of samples to visualize
    """
    visualizer = RoboAffordVisualizer(output_dir)

    # Read JSONL file
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    # Process each sample
    for i, line in enumerate(lines[:num_samples]):
        try:
            data = json.loads(line)

            # Extract information
            image_path = data['images'][0]
            bbox_data = data['messages'][1]['content']
            question_text = data['messages'][0]['content']

            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Image does not exist: {image_path}")
                continue

            # Visualize
            output_path = visualizer.draw_multiple_bboxes_pil(
                image_path, bbox_data, question_text,
                show_question=True
            )

            if output_path:
                print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Error processing line {i}: {e}")


# Main program
if __name__ == "__main__":
    # Initialize visualization tool
    visualizer = RoboAffordVisualizer(
        output_dir="debug/visualization_roboafford"
    )

    # Placeholder JSONL file path
    jsonl_path = "/path/to/your/roboafford_data.jsonl"

    # Test data
    test_data = [
        {
            "images": ["data/public_robo_datasets/RoboAfford-Eval/images/00.jpg"],
            "messages": [
                {"role": "user", "content": "What part of a mug should be gripped to lift it? Your answer should be formatted as a list of bounding boxes, i.e. [[x1, y1, x2, y2], ...], where each bounding box contains the absolute pixel coordinates of the region that should be gripped. The coordinates should be integers."},
                {"role": "assistant", "content": "<answer>[[269.0, 268.0, 296.0, 311.0], [505.0, 331.0, 563.0, 373.0]]</answer>"}
            ],
            "meta_data": {
                "original_img": "00.jpg",
                "original_mask": "object_affordance_img_00_anno_1.png"
            }
        },
        {
            "images": ["data/public_robo_datasets/RoboAfford-Eval/images/00.jpg"],
            "messages": [
                {"role": "user", "content": "What part of a mug holds the liquid inside for drinking? Your answer should be formatted as a list of bounding boxes, i.e. [[x1, y1, x2, y2], ...], where each bounding box contains the absolute pixel coordinates of the region that should be gripped. The coordinates should be integers."},
                {"role": "assistant", "content": "<answer>[[193.0, 265.0, 278.0, 333.0], [473.0, 267.0, 553.0, 333.0]]</answer>"}
            ],
            "meta_data": {
                "original_img": "00.jpg",
                "original_mask": "object_affordance_img_00_anno_2.png"
            }
        }
    ]

    # Test visualization
    print("=== Test visualization ===")

    for i, data in enumerate(test_data):
        print(f"\nProcessing test data {i+1}:")

        image_path = data["images"][0]
        bbox_data = data["messages"][1]["content"]
        question_text = data["messages"][0]["content"]

        if os.path.exists(image_path):
            # Visualize multiple bboxes
            output_path = visualizer.draw_multiple_bboxes_pil(
                image_path, bbox_data, question_text,
                show_question=True
            )

            if output_path:
                print(f"[OK] Saved: {output_path}")

            # If mask exists, also visualize mask
            mask_path = os.path.join(
                os.path.dirname(image_path),
                "..",  # Parent directory
                "masks",  # masks directory
                data["meta_data"]["original_mask"]
            )
            mask_path = os.path.abspath(mask_path)

            if os.path.exists(mask_path):
                mask_output = visualizer.visualize_with_mask(
                    image_path, mask_path, bbox_data, question_text
                )
                if mask_output:
                    print(f"[OK] Mask visualization: {mask_output}")
            else:
                print(f"[FAIL] Mask does not exist: {mask_path}")
        else:
            print(f"[FAIL] Image does not exist: {image_path}")

    # If JSONL file exists, run full pipeline
    if os.path.exists(jsonl_path):
        print("\n=== Full visualization pipeline ===")

        # 1. Visualize single entry
        print("\n1. Visualize single entry:")
        visualizer.visualize_jsonl_entry(jsonl_path, entry_index=0)

        # 2. Batch visualize
        print("\n2. Batch visualization:")
        visualizer.batch_visualize(jsonl_path, num_samples=5, random_sample=True)

        # 3. Create comparison grid
        print("\n3. Create comparison grid:")
        visualizer.create_comparison_grid(jsonl_path, num_rows=2, num_cols=2)

        # 4. Create heatmap
        print("\n4. Create heatmap:")
        visualizer.visualize_heatmap_overlay(jsonl_path, image_index=0)
    else:
        print(f"\nJSONL file does not exist: {jsonl_path}")
        print("Please use an actual data file path")


# Simplified usage example
def simple_visualize_example():
    """
    Simplified usage example
    """
    visualizer = RoboAffordVisualizer()

    # Example data
    image_path = "data/public_robo_datasets/RoboAfford-Eval/images/00.jpg"
    bbox_data = "[[269.0, 268.0, 296.0, 311.0], [505.0, 331.0, 563.0, 373.0]]"
    question = "What part of a mug should be gripped to lift it?"

    if os.path.exists(image_path):
        output = visualizer.draw_multiple_bboxes_pil(
            image_path, bbox_data, question, show_question=True
        )
        print(f"Visualization result: {output}")
    else:
        print(f"Image does not exist: {image_path}")


# Run simplified example
if __name__ == "__main__":
    simple_visualize_example()
