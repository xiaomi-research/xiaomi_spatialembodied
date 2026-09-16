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


class VABenchVisualizer:
    """VABench data visualization tool"""

    def __init__(self, output_dir="debug/visualization_vabench"):
        """
        Initialize visualization tool

        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Task type color mapping
        self.task_colors = {
            'move': 'red',
            'grasp': 'blue',
            'push': 'green',
            'place': 'purple',
            'arrange': 'orange',
            'stack': 'cyan',
            'default': 'yellow'
        }

        # Try to load fonts
        try:
            self.font = ImageFont.truetype("arial.ttf", 16)
            self.title_font = ImageFont.truetype("arial.ttf", 20)
        except:
            self.font = ImageFont.load_default()
            self.title_font = ImageFont.load_default()

    def extract_task_type(self, instruction):
        """
        Extract task type from task instruction

        Args:
            instruction: Task instruction text

        Returns:
            Task type string
        """
        instruction_lower = instruction.lower()

        if 'move' in instruction_lower:
            return 'move'
        elif 'grasp' in instruction_lower:
            return 'grasp'
        elif 'push' in instruction_lower:
            return 'push'
        elif 'place' in instruction_lower or 'put' in instruction_lower:
            return 'place'
        elif 'arrange' in instruction_lower or 'between' in instruction_lower:
            return 'arrange'
        elif 'stack' in instruction_lower:
            return 'stack'
        else:
            return 'default'

    def extract_object_info(self, instruction):
        """
        Extract object information from task instruction

        Args:
            instruction: Task instruction text

        Returns:
            Object information string
        """
        # Use simple keyword extraction
        color_keywords = ['yellow', 'red', 'blue', 'green', 'orange', 'purple',
                         'white', 'black', 'brown', 'gray', 'grey', 'pink']
        object_keywords = ['block', 'cube', 'ball', 'object', 'item', 'thing',
                          'pot', 'pan', 'bowl', 'plate', 'cup', 'mug',
                          'spoon', 'fork', 'knife', 'bread', 'food']

        # Extract color
        color = None
        for keyword in color_keywords:
            if keyword in instruction.lower():
                color = keyword
                break

        # Extract object
        obj = None
        for keyword in object_keywords:
            if keyword in instruction.lower():
                obj = keyword
                break

        if color and obj:
            return f"{color} {obj}"
        elif obj:
            return obj
        elif color:
            return f"{color} object"
        else:
            return "target object"

    def parse_bbox(self, bbox_data):
        """
        Parse bounding box data

        Args:
            bbox_data: Can be a list, string, or tuple

        Returns:
            bbox coordinates (x1, y1, x2, y2)
        """
        if isinstance(bbox_data, str):
            # Try to extract coordinates from string
            if bbox_data.startswith('<answer>') and bbox_data.endswith('</answer>'):
                # Extract coordinate part
                match = re.search(r'<answer>\[(.*?)\]</answer>', bbox_data)
                if match:
                    coord_str = match.group(1)
                    try:
                        return ast.literal_eval(f'[{coord_str}]')
                    except:
                        pass
            else:
                # Try direct parsing
                try:
                    return ast.literal_eval(bbox_data)
                except:
                    pass
        elif isinstance(bbox_data, (list, tuple)):
            return list(bbox_data)

        return None

    def draw_bbox_with_task_info(self, image_path, bbox, instruction="",
                                 output_path=None, line_width=3,
                                 show_instruction=True, show_task_type=True):
        """
        Draw bbox on image and display task information

        Args:
            image_path: Image path
            bbox: Bounding box coordinates
            instruction: Task instruction
            output_path: Output path
            line_width: Line width
            show_instruction: Whether to show instruction
            show_task_type: Whether to show task type
        """
        # Open image
        img = Image.open(image_path)
        original_width, original_height = img.size

        # Parse bbox
        bbox_coords = self.parse_bbox(bbox)
        if bbox_coords is None:
            print(f"Cannot parse bbox: {bbox}")
            return None

        x1, y1, x2, y2 = bbox_coords

        # Extract task information
        task_type = self.extract_task_type(instruction)
        object_info = self.extract_object_info(instruction)
        color = self.task_colors.get(task_type, self.task_colors['default'])

        # Create drawing object
        draw = ImageDraw.Draw(img)

        # Draw bounding box
        draw.rectangle(bbox_coords, outline=color, width=line_width)

        # Display task type above bounding box
        if show_task_type:
            task_text = f"Task: {task_type.upper()}"
            text_bbox = draw.textbbox((x1, y1-25), task_text, font=self.font)
            # Draw text background
            draw.rectangle(text_bbox, fill=color)
            # Draw text
            draw.text((x1, y1-25), task_text, fill='black', font=self.font)

        # Display object information inside bounding box
        if object_info:
            # Calculate text size suitable for inside bounding box
            box_width = x2 - x1
            box_height = y2 - y1

            if box_width > 60 and box_height > 20:
                # Display object information inside bounding box
                obj_text = object_info[:10]  # Limit length
                # Calculate text position (centered)
                text_width = len(obj_text) * 7
                text_x = x1 + (box_width - text_width) // 2
                text_y = y1 + 5

                # Draw semi-transparent background
                draw.rectangle(
                    [text_x-5, text_y-2, text_x+text_width+5, text_y+12],
                    fill=(0, 0, 0, 128)
                )
                draw.text((text_x, text_y), obj_text, fill='white', font=self.font)

        # Display task instruction at top of image
        if show_instruction and instruction:
            # Extract simplified instruction
            instruction_summary = instruction[:80] + "..." if len(instruction) > 80 else instruction

            # Calculate required top space
            top_padding = 60
            new_img = Image.new('RGB', (original_width, original_height + top_padding), 'black')
            new_img.paste(img, (0, top_padding))

            # Draw on new image
            new_draw = ImageDraw.Draw(new_img)

            # Draw title
            new_draw.text((10, 10), "Task Instruction:", fill='white', font=self.title_font)

            # Draw instruction text (may need line wrapping)
            words = instruction_summary.split()
            lines = []
            current_line = ""

            for word in words:
                test_line = f"{current_line} {word}".strip()
                if len(test_line) * 6 <= original_width - 20:  # Simple width estimate
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            # Draw multiple lines of text
            for i, line in enumerate(lines[:3]):  # Show at most 3 lines
                new_draw.text((10, 35 + i*20), line, fill='yellow', font=self.font)

            img = new_img

        # Display coordinate information in bottom-right corner
        coord_text = f"({int(x1)},{int(y1)})-({int(x2)},{int(y2)})"
        draw.text((original_width-150, original_height-20), coord_text, fill='white', font=self.font)

        # Generate output path
        if output_path is None:
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)

            # Use task type and object information as filename
            safe_task = task_type.replace(' ', '_')
            safe_obj = object_info.replace(' ', '_')[:20]
            output_name = f"{name}_{safe_task}_{safe_obj}{ext}"
            output_path = os.path.join(self.output_dir, output_name)

        # Save image
        img.save(output_path)
        return output_path

    def visualize_jsonl_entry(self, jsonl_path, entry_index=0,
                             show_instruction=True, show_task_type=True):
        """
        Visualize a single entry from a JSONL file

        Args:
            jsonl_path: JSONL file path
            entry_index: Entry index
            show_instruction: Whether to show instruction
            show_task_type: Whether to show task type
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
        instruction = data['messages'][0]['content']  # user's task instruction

        # Can also get bbox from meta_data
        meta_bbox = data['meta_data'].get('original_bbox', None)

        # Check if image path exists
        if not os.path.exists(image_path):
            print(f"Image does not exist: {image_path}")
            return

        # Print information
        print(f"=== Entry {entry_index} ===")
        print(f"Image path: {image_path}")
        print(f"Task instruction: {instruction[:100]}...")

        # Prefer using bbox from meta_data as it is more reliable
        if meta_bbox is not None:
            bbox = meta_bbox
            print(f"Using bbox from meta_data: {bbox}")
        else:
            bbox = bbox_data
            print(f"Using bbox from assistant reply: {bbox}")

        # Visualize
        output_path = self.draw_bbox_with_task_info(
            image_path, bbox, instruction,
            show_instruction=show_instruction,
            show_task_type=show_task_type
        )

        if output_path:
            print(f"Visualization result saved to: {output_path}")

        return output_path

    def batch_visualize(self, jsonl_path, num_samples=10,
                       random_sample=False, show_progress=True,
                       show_instruction=False, show_task_type=True):
        """
        Batch visualize multiple samples

        Args:
            jsonl_path: JSONL file path
            num_samples: Number of samples to visualize
            random_sample: Whether to randomly sample
            show_progress: Whether to show progress bar
            show_instruction: Whether to show instruction
            show_task_type: Whether to show task type
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
                instruction = data['messages'][0]['content']

                # Get bbox
                bbox = data['meta_data'].get('original_bbox', None)
                if bbox is None:
                    bbox = data['messages'][1]['content']

                # Check if image exists
                if not os.path.exists(image_path):
                    continue

                # Visualize
                output_path = self.draw_bbox_with_task_info(
                    image_path, bbox, instruction,
                    show_instruction=show_instruction,
                    show_task_type=show_task_type
                )

                if output_path:
                    saved_paths.append(output_path)

            except Exception as e:
                print(f"Error processing entry {idx}: {e}")

        print(f"\nBatch visualization complete, processed {len(saved_paths)} samples")
        print(f"Output directory: {self.output_dir}")

        return saved_paths

    def create_task_comparison_grid(self, jsonl_path, num_images=6, grid_size=(2, 3)):
        """
        Create task comparison grid

        Args:
            jsonl_path: JSONL file path
            num_images: Number of images
            grid_size: Grid size (rows, cols)
        """
        # Read JSONL file
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()

        num_images = min(num_images, len(lines), grid_size[0] * grid_size[1])

        # Create figure
        fig, axes = plt.subplots(grid_size[0], grid_size[1],
                                figsize=(5*grid_size[1], 4*grid_size[0]))

        if grid_size[0] == 1 and grid_size[1] == 1:
            axes = [[axes]]
        elif grid_size[0] == 1:
            axes = [axes]
        elif grid_size[1] == 1:
            axes = [[ax] for ax in axes]

        for i in range(num_images):
            row = i // grid_size[1]
            col = i % grid_size[1]

            data = json.loads(lines[i])
            image_path = data['images'][0]
            instruction = data['messages'][0]['content']
            bbox = data['meta_data'].get('original_bbox', data['messages'][1]['content'])

            # Load image
            img = plt.imread(image_path)
            axes[row][col].imshow(img)

            # Parse bbox
            bbox_coords = self.parse_bbox(bbox)
            # Compute the per-task colour before the bbox branch: it is used for the title
            # below as well, and the previous version only bound it inside `if bbox_coords`,
            # so any sample without a bbox raised UnboundLocalError.
            task_type = self.extract_task_type(instruction)
            color = self.task_colors.get(task_type, self.task_colors['default'])

            if bbox_coords:
                x1, y1, x2, y2 = bbox_coords
                width = x2 - x1
                height = y2 - y1

                # Draw bounding box
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                axes[row][col].add_patch(rect)

            # Set title
            title = task_type.upper()
            axes[row][col].set_title(title, fontsize=12, color=color)
            axes[row][col].axis('off')

        # Hide extra subplots
        for i in range(num_images, grid_size[0] * grid_size[1]):
            row = i // grid_size[1]
            col = i % grid_size[1]
            axes[row][col].axis('off')

        plt.suptitle("VABench: Different Task Types", fontsize=16)
        plt.tight_layout()

        # Save
        output_path = os.path.join(self.output_dir, "task_comparison_grid.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2, dpi=150)
        plt.close(fig)

        print(f"Task comparison grid saved: {output_path}")
        return output_path

    def visualize_with_point_marker(self, image_path, bbox, instruction="",
                                    point_color='red', point_size=10):
        """
        Add point marker at bbox center

        Args:
            image_path: Image path
            bbox: Bounding box coordinates
            instruction: Task instruction
            point_color: Point color
            point_size: Point size
        """
        # Open image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Parse bbox
        bbox_coords = self.parse_bbox(bbox)
        if bbox_coords is None:
            return None

        x1, y1, x2, y2 = bbox_coords

        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Draw bounding box
        draw.rectangle(bbox_coords, outline='yellow', width=2)

        # Draw center point
        draw.ellipse(
            [center_x - point_size, center_y - point_size,
             center_x + point_size, center_y + point_size],
            fill=point_color, outline='white', width=2
        )

        # Add cross marker
        cross_size = 15
        draw.line([center_x - cross_size, center_y, center_x + cross_size, center_y],
                 fill='white', width=2)
        draw.line([center_x, center_y - cross_size, center_x, center_y + cross_size],
                 fill='white', width=2)

        # Save
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(self.output_dir, f"{name}_point_marker{ext}")
        img.save(output_path)

        return output_path

    def analyze_bbox_distribution(self, jsonl_path, num_samples=100):
        """
        Analyze bounding box distribution

        Args:
            jsonl_path: JSONL file path
            num_samples: Number of samples to analyze
        """
        # Read JSONL file
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()

        num_samples = min(num_samples, len(lines))

        bbox_sizes = []
        bbox_aspect_ratios = []
        image_centers = []

        for i in range(num_samples):
            data = json.loads(lines[i])
            bbox = data['meta_data'].get('original_bbox', None)

            if bbox is not None:
                bbox_coords = self.parse_bbox(bbox)
                if bbox_coords:
                    x1, y1, x2, y2 = bbox_coords

                    # Calculate size
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height

                    bbox_sizes.append(area)

                    # Calculate aspect ratio
                    if height > 0:
                        aspect_ratio = width / height
                        bbox_aspect_ratios.append(aspect_ratio)

                    # Calculate center point
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    image_centers.append((center_x, center_y))

        if not bbox_sizes:
            print("No valid bounding box data found")
            return

        # Create analysis charts
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Bounding box size distribution
        axes[0].hist(bbox_sizes, bins=20, alpha=0.7, color='blue')
        axes[0].set_xlabel('BBox Area (pixels)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Bounding Box Size Distribution')
        axes[0].grid(True, alpha=0.3)

        # 2. Aspect ratio distribution
        axes[1].hist(bbox_aspect_ratios, bins=20, alpha=0.7, color='green')
        axes[1].set_xlabel('Aspect Ratio (width/height)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Bounding Box Aspect Ratio Distribution')
        axes[1].grid(True, alpha=0.3)

        # 3. Center point distribution
        centers_x = [c[0] for c in image_centers]
        centers_y = [c[1] for c in image_centers]
        scatter = axes[2].scatter(centers_x, centers_y, alpha=0.6, c='red', s=20)
        axes[2].set_xlabel('X Coordinate')
        axes[2].set_ylabel('Y Coordinate')
        axes[2].set_title('Bounding Box Center Distribution')
        axes[2].grid(True, alpha=0.3)

        # Add statistics
        stats_text = f"""
        Statistics:
        Total bboxes: {len(bbox_sizes)}
        Average size: {np.mean(bbox_sizes):.1f} pixels
        Average aspect ratio: {np.mean(bbox_aspect_ratios):.2f}
        Min size: {np.min(bbox_sizes):.1f}
        Max size: {np.max(bbox_sizes):.1f}
        """

        plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.suptitle(f"VABench Bounding Box Analysis (n={len(bbox_sizes)})", fontsize=16)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

        # Save
        output_path = os.path.join(self.output_dir, "bbox_analysis.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        print(f"Bounding box analysis chart saved: {output_path}")

        return {
            'sizes': bbox_sizes,
            'aspect_ratios': bbox_aspect_ratios,
            'centers': image_centers
        }


# Quick visualization function
def quick_vabench_visualize(jsonl_path, output_dir, num_samples=5):
    """
    Quickly visualize VABench data

    Args:
        jsonl_path: JSONL file path
        output_dir: Output directory
        num_samples: Number of samples to visualize
    """
    visualizer = VABenchVisualizer(output_dir)

    # Read JSONL file
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    # Process each sample
    for i, line in enumerate(lines[:num_samples]):
        try:
            data = json.loads(line)

            # Extract information
            image_path = data['images'][0]
            instruction = data['messages'][0]['content']

            # Get bbox
            bbox = data['meta_data'].get('original_bbox', None)
            if bbox is None:
                bbox = data['messages'][1]['content']

            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Image does not exist: {image_path}")
                continue

            # Visualize
            output_path = visualizer.draw_bbox_with_task_info(
                image_path, bbox, instruction,
                show_instruction=True,
                show_task_type=True
            )

            if output_path:
                print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Error processing line {i}: {e}")


# Main program
if __name__ == "__main__":
    # Initialize visualization tool
    visualizer = VABenchVisualizer(
        output_dir="debug/visualization_vabench"
    )

    # JSONL file path
    jsonl_path = "data/preprocess_data/vabench-point-bbox/vabench_bbox_absolute_no_reasoning_swift.jsonl"

    if os.path.exists(jsonl_path):
        print(f"Found JSONL file: {jsonl_path}")

        # Example 1: Visualize first few samples
        print("\n=== Visualize first 3 samples ===")
        quick_vabench_visualize(jsonl_path, visualizer.output_dir, num_samples=3)

        # Example 2: Use full tool for visualization
        print("\n=== Use full tool for visualization ===")

        # 2.1 Visualize single entry
        print("\n1. Visualize single entry:")
        visualizer.visualize_jsonl_entry(jsonl_path, entry_index=0,
                                        show_instruction=True, show_task_type=True)

        # 2.2 Batch visualize
        print("\n2. Batch visualization:")
        visualizer.batch_visualize(jsonl_path, num_samples=5,
                                  random_sample=False, show_progress=True,
                                  show_instruction=False, show_task_type=True)

        # 2.3 Create task comparison grid
        print("\n3. Create task comparison grid:")
        visualizer.create_task_comparison_grid(jsonl_path, num_images=6, grid_size=(2, 3))

        # 2.4 Analyze bounding box distribution
        print("\n4. Analyze bounding box distribution:")
        visualizer.analyze_bbox_distribution(jsonl_path, num_samples=50)

        # 2.5 Test point marker visualization
        print("\n5. Test point marker visualization:")
        # Read first sample
        with open(jsonl_path, 'r') as f:
            first_line = f.readline()
            first_data = json.loads(first_line)

            image_path = first_data['images'][0]
            instruction = first_data['messages'][0]['content']
            bbox = first_data['meta_data'].get('original_bbox', first_data['messages'][1]['content'])

            if os.path.exists(image_path):
                point_output = visualizer.visualize_with_point_marker(
                    image_path, bbox, instruction
                )
                if point_output:
                    print(f"Point marker visualization: {point_output}")

    else:
        print(f"JSONL file does not exist: {jsonl_path}")
        print("Using example data for demonstration...")

        # Example data
        examples = [
            {
                "images": ["data/preprocess_data/vabench-point-bbox/images/image_00000.png"],
                "messages": [
                    {"role": "user", "content": "<image> You are currently a robot performing robotic manipulation tasks. The task instruction is: Move the yellow block in the middle of the table. Use a bounding box to mark the target location where the object you need to manipulate in the task should ultimately be moved.\nYou directly provide the final answer. The answer is enclosed within <answer> </answer> tags. The answer consists only of bounding box coordinates, with the overall format being: <answer>[x1, y1, x2, y2]</answer>"},
                    {"role": "assistant", "content": "<answer>[235.0, 106.0, 305.0, 160.0]</answer>"}
                ],
                "meta_data": {
                    "original_bbox": [235, 106, 305, 160],
                    "image_dimensions": {"width": 640, "height": 480}
                }
            },
            {
                "images": ["data/preprocess_data/vabench-point-bbox/images/image_00001.png"],
                "messages": [
                    {"role": "user", "content": "<image> You are currently a robot performing robotic manipulation tasks. The task instruction is: Moved the orange pot in between the spoon and the bread.. Use a bounding box to mark the target location where the object you need to manipulate in the task should ultimately be moved.\nYou directly provide the final answer. The answer is enclosed within <answer> </answer> tags. The answer consists only of bounding box coordinates, with the overall format being: <answer>[x1, y1, x2, y2]</answer>"},
                    {"role": "assistant", "content": "<answer>[192.0, 249.0, 314.0, 379.0]</answer>"}
                ],
                "meta_data": {
                    "original_bbox": [192, 249, 314, 379],
                    "image_dimensions": {"width": 640, "height": 480}
                }
            }
        ]

        for i, example in enumerate(examples):
            print(f"\nProcessing example {i+1}:")

            image_path = example["images"][0]
            instruction = example["messages"][0]["content"]
            bbox = example["meta_data"]["original_bbox"]

            # Check if image exists
            if os.path.exists(image_path):
                output_path = visualizer.draw_bbox_with_task_info(
                    image_path, bbox, instruction,
                    show_instruction=True,
                    show_task_type=True
                )

                if output_path:
                    print(f"Saved: {output_path}")
            else:
                print(f"Image does not exist: {image_path}")
                print("Creating mock image for demonstration...")

                # Create mock image
                width = 640
                height = 480
                img = Image.new('RGB', (width, height), color='gray')

                # Draw a simple scene
                draw = ImageDraw.Draw(img)
                draw.rectangle([100, 100, 540, 380], outline='white', width=2)
                draw.text((200, 200), "VABench Example", fill='white', font=visualizer.font)

                # Save mock image
                temp_image_path = os.path.join(visualizer.output_dir, f"example_{i}.png")
                img.save(temp_image_path)

                # Use mock image for visualization
                output_path = visualizer.draw_bbox_with_task_info(
                    temp_image_path, bbox, instruction,
                    show_instruction=True,
                    show_task_type=True
                )

                if output_path:
                    print(f"Saved mock visualization: {output_path}")


# Simplified usage script
def simple_vabench_visualizer():
    """
    Simplified VABench visualization script
    """
    import json
    from PIL import Image, ImageDraw
    import os

    # Configuration
    jsonl_path = "data/preprocess_data/vabench-point-bbox/vabench_bbox_absolute_no_reasoning_swift.jsonl"
    output_dir = "debug/visualization_vabench_simple"

    os.makedirs(output_dir, exist_ok=True)

    # Read JSONL file
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    # Process each sample
    for i, line in enumerate(lines[:5]):  # Only process first 5
        try:
            data = json.loads(line)

            # Extract information
            image_path = data['images'][0]
            instruction = data['messages'][0]['content']
            bbox = data['meta_data']['original_bbox']

            if not os.path.exists(image_path):
                continue

            # Open image
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)

            # Draw bounding box
            x1, y1, x2, y2 = bbox
            draw.rectangle(bbox, outline='red', width=3)

            # Add task type label
            task_type = "MOVE" if "move" in instruction.lower() else "OTHER"
            draw.text((x1, y1-25), task_type, fill='red')

            # Save
            base_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f"{i}_{base_name}")
            img.save(output_path)

            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Run simplified version
    # simple_vabench_visualizer()
    pass
