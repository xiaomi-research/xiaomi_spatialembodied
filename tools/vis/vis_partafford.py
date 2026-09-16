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


class PartAffordanceVisualizer:
    """Part-Affordance-2K data visualization tool"""

    def __init__(self, output_dir="debug/visualization_partafford"):
        """
        Initialize visualization tool

        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Color mapping, use different colors for different categories
        self.category_colors = {
            'mug': 'red',
            'bottle': 'blue',
            'knife': 'green',
            'scissor': 'purple',
            'cup': 'orange',
            'hammer': 'cyan',
            'screwdriver': 'magenta',
            'spoon': 'yellow',
            'fork': 'lime',
            'pan': 'pink',
            'default': 'white'
        }

        # Font
        try:
            self.font = ImageFont.truetype("arial.ttf", 20)
        except:
            self.font = ImageFont.load_default()

    def parse_bbox(self, bbox_data):
        """
        Parse bounding box data

        Args:
            bbox_data: Can be a list, string, or tuple

        Returns:
            bbox coordinates (x1, y1, x2, y2)
        """
        if isinstance(bbox_data, str):
            return ast.literal_eval(bbox_data)
        elif isinstance(bbox_data, (list, tuple)):
            return list(bbox_data)
        else:
            raise ValueError(f"Cannot parse bbox data: {bbox_data}")

    def draw_bbox_on_image_pil(self, image_path, bbox, category="default",
                               output_path=None, line_width=3,
                               show_label=True, show_mask=False, mask_path=None):
        """
        Draw bbox on image using PIL

        Args:
            image_path: Image path
            bbox: Bounding box coordinates
            category: Category
            output_path: Output path
            line_width: Line width
            show_label: Whether to show label
            show_mask: Whether to show mask
            mask_path: Mask path
        """
        # Open image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Parse bbox
        bbox = self.parse_bbox(bbox)
        x1, y1, x2, y2 = bbox

        # Get color
        color = self.category_colors.get(category, self.category_colors['default'])

        # Draw bbox
        draw.rectangle(bbox, outline=color, width=line_width)

        # Show label in top-left corner
        if show_label and category != "default":
            label_text = f"{category}"
            # Calculate text size
            text_bbox = draw.textbbox((x1, y1-30), label_text, font=self.font)
            # Draw text background
            draw.rectangle(text_bbox, fill=color)
            # Draw text
            draw.text((x1, y1-30), label_text, fill='black', font=self.font)

        # If showing mask and mask path exists
        if show_mask and mask_path and os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert("L")
                # Convert mask to RGBA
                mask_rgba = mask.copy()
                mask_rgba.putalpha(128)  # Set transparency

                # Create a colored mask
                color_mask = Image.new("RGBA", img.size, (255, 255, 255, 0))
                color_pixels = color_mask.load()

                for i in range(color_mask.width):
                    for j in range(color_mask.height):
                        if mask.getpixel((i, j)) > 0:
                            # Set color based on category
                            if category == 'mug':
                                color_pixels[i, j] = (255, 0, 0, 128)  # Red
                            elif category == 'bottle':
                                color_pixels[i, j] = (0, 0, 255, 128)  # Blue
                            else:
                                color_pixels[i, j] = (0, 255, 0, 128)  # Green

                # Overlay colored mask onto original image
                img = Image.alpha_composite(img.convert("RGBA"), color_mask)
            except Exception as e:
                print(f"Failed to load mask: {e}")

        # Generate output path
        if output_path is None:
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(self.output_dir, f"{name}_bbox{ext}")

        # Save image
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img.save(output_path)
        return output_path

    def visualize_jsonl_entry(self, jsonl_path, entry_index=0,
                             use_matplotlib=False, save_individual=True):
        """
        Visualize a single entry from a JSONL file

        Args:
            jsonl_path: JSONL file path
            entry_index: Entry index
            use_matplotlib: Whether to use matplotlib
            save_individual: Whether to save as individual file
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
        bbox = data['meta_data']['absolute_bbox']
        category = data['meta_data']['category_type']
        mask_path = data['meta_data']['mask_path']
        original_mask_path = os.path.join(os.path.dirname(os.path.dirname(image_path)),
                                          "Part-Affordance-2K", mask_path)

        # Check if image path exists
        if not os.path.exists(image_path):
            print(f"Image does not exist: {image_path}")
            return

        # Print information
        print(f"Entry index: {entry_index}")
        print(f"Image path: {image_path}")
        print(f"BBox: {bbox}")
        print(f"Category: {category}")
        print(f"Mask path: {original_mask_path}")
        print(f"Mask exists: {os.path.exists(original_mask_path)}")

        if use_matplotlib:
            output_path = self.draw_with_matplotlib(
                image_path, bbox, category, mask_path=original_mask_path
            )
        else:
            output_path = self.draw_bbox_on_image_pil(
                image_path, bbox, category,
                show_mask=True, mask_path=original_mask_path
            )

        print(f"Visualization result saved to: {output_path}")
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
                bbox = data['meta_data']['absolute_bbox']
                category = data['meta_data']['category_type']
                mask_path = data['meta_data']['mask_path']
                original_mask_path = os.path.join(os.path.dirname(os.path.dirname(image_path)),
                                                  "Part-Affordance-2K", mask_path)

                # Visualize
                output_path = self.draw_bbox_on_image_pil(
                    image_path, bbox, category,
                    show_mask=False,  # Do not show mask in batch processing to improve speed
                    mask_path=original_mask_path
                )

                saved_paths.append(output_path)

            except Exception as e:
                print(f"Error processing entry {idx}: {e}")

        print(f"\nBatch visualization complete, processed {len(saved_paths)} samples")
        print(f"Output directory: {self.output_dir}")

        return saved_paths

    def visualize_with_mask_overlay(self, image_path, mask_path, bbox=None,
                                    category="default", alpha=0.5):
        """
        Visualize with mask overlay

        Args:
            image_path: Image path
            mask_path: Mask path
            bbox: Bounding box
            category: Category
            alpha: Mask transparency
        """
        # Open image
        img = Image.open(image_path).convert("RGBA")

        # Open mask
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")

            # Create colored mask
            color_map = {
                'mug': (255, 0, 0, int(255 * alpha)),
                'bottle': (0, 0, 255, int(255 * alpha)),
                'knife': (0, 255, 0, int(255 * alpha)),
                'default': (255, 255, 0, int(255 * alpha))
            }

            color = color_map.get(category, color_map['default'])
            colored_mask = Image.new("RGBA", img.size, (0, 0, 0, 0))

            # Apply mask
            for x in range(img.width):
                for y in range(img.height):
                    if mask.getpixel((x, y)) > 0:
                        colored_mask.putpixel((x, y), color)

            # Merge image and mask
            result = Image.alpha_composite(img, colored_mask)

            # If bbox exists, draw bounding box
            if bbox is not None:
                draw = ImageDraw.Draw(result)
                bbox_coords = self.parse_bbox(bbox)
                draw.rectangle(bbox_coords, outline='white', width=3)

                # Add category label
                label = f"{category}"
                draw.text((bbox_coords[0], bbox_coords[1] - 25),
                         label, fill='white', font=self.font)
        else:
            result = img

        # Save result
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(self.output_dir, f"{name}_mask_overlay{ext}")

        result.convert("RGB").save(output_path)
        print(f"Mask overlay visualization saved to: {output_path}")
        return output_path

    def draw_with_matplotlib(self, image_path, bbox, category="default",
                             mask_path=None, figsize=(12, 9)):
        """
        Visualize using matplotlib

        Args:
            image_path: Image path
            bbox: Bounding box
            category: Category
            mask_path: Mask path
            figsize: Figure size
        """
        # Read image
        img = plt.imread(image_path)

        # Parse bbox
        bbox_coords = self.parse_bbox(bbox)
        x1, y1, x2, y2 = bbox_coords
        width = x2 - x1
        height = y2 - y1

        # Create figure
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(img)

        # Draw bounding box
        color = self.category_colors.get(category, self.category_colors['default'])
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Add label
        ax.text(x1, y1-10, category, color=color, fontsize=12,
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

        # If mask exists, draw mask
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask_array = np.array(mask)

            # Create mask overlay
            mask_overlay = np.zeros((*mask_array.shape, 4))
            mask_overlay[mask_array > 0] = [1.0, 0.0, 0.0, 0.3]  # Red, 30% transparency

            ax.imshow(mask_overlay, alpha=0.5)

        ax.axis('off')
        plt.tight_layout()

        # Save
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(self.output_dir, f"{name}_matplotlib{ext}")

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)

        return output_path

    def create_summary_grid(self, jsonl_path, num_images=9, grid_size=(3, 3)):
        """
        Create visualization summary grid

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
                                figsize=(4*grid_size[1], 4*grid_size[0]))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < num_images:
                data = json.loads(lines[i])
                image_path = data['images'][0]
                bbox = data['meta_data']['absolute_bbox']
                category = data['meta_data']['category_type']

                # Read image
                img = plt.imread(image_path)
                ax.imshow(img)

                # Draw bbox
                bbox_coords = self.parse_bbox(bbox)
                x1, y1, x2, y2 = bbox_coords
                width = x2 - x1
                height = y2 - y1

                color = self.category_colors.get(category, self.category_colors['default'])
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)

                # Add title
                ax.set_title(f"{category}", fontsize=10, color=color)
                ax.axis('off')
            else:
                ax.axis('off')

        plt.suptitle(f"Part-Affordance-2K Data Visualization ({num_images} samples)", fontsize=16)
        plt.tight_layout()

        # Save grid
        output_path = os.path.join(self.output_dir, "summary_grid.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2, dpi=200)
        plt.close(fig)

        print(f"Summary grid saved to: {output_path}")
        return output_path


# Usage example
if __name__ == "__main__":
    # Initialize visualization tool
    visualizer = PartAffordanceVisualizer(
        output_dir="debug/visualization_partafford"
    )

    # JSONL file path (placeholder path)
    jsonl_path = "data/preprocess_data/Part-Affordance-2K/ms-swift/part_affordance_converted.jsonl"

    # Check if file exists
    if not os.path.exists(jsonl_path):
        print(f"JSONL file does not exist: {jsonl_path}")
        # Use example data
        example_data = [
            {
                "images": ["data/preprocess_data/Part-Affordance-2K/images/10600.jpg"],
                "meta_data": {
                    "absolute_bbox": [253, 167, 273, 227],
                    "category_type": "mug",
                    "mask_path": "masks/10600.png"
                }
            },
            {
                "images": ["data/preprocess_data/Part-Affordance-2K/images/8315.jpg"],
                "meta_data": {
                    "absolute_bbox": [291, 213, 311, 265],
                    "category_type": "mug",
                    "mask_path": "masks/8315.png"
                }
            }
        ]

        # Example visualization
        for i, data in enumerate(example_data):
            image_path = data["images"][0]
            bbox = data["meta_data"]["absolute_bbox"]
            category = data["meta_data"]["category_type"]
            mask_path = os.path.join(os.path.dirname(os.path.dirname(image_path)),
                                    "Part-Affordance-2K",
                                    data["meta_data"]["mask_path"])

            print(f"\nProcessing example {i+1}:")
            print(f"Image: {image_path}")
            print(f"Category: {category}")
            print(f"BBox: {bbox}")

            # Visualize
            if os.path.exists(image_path):
                output = visualizer.draw_bbox_on_image_pil(
                    image_path, bbox, category,
                    show_mask=True, mask_path=mask_path
                )
                print(f"Saved: {output}")
            else:
                print(f"Image does not exist: {image_path}")
    else:
        # Visualize single entry
        print("=== Visualize single entry ===")
        visualizer.visualize_jsonl_entry(jsonl_path, entry_index=0)

        # Batch visualize
        print("\n=== Batch visualization ===")
        visualizer.batch_visualize(jsonl_path, num_samples=5, random_sample=False)

        # Create summary grid
        print("\n=== Create summary grid ===")
        visualizer.create_summary_grid(jsonl_path, num_images=9, grid_size=(3, 3))
