# UPDATE: Replace placeholder paths with your actual paths.
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib
from PIL import Image
import numpy as np
import random

# Set matplotlib to use Agg (no GUI backend)
matplotlib.use('Agg')

class RoboRefitVisualizer:
    def __init__(self, metadata_path, dataset_root, images_dir=None, annotations_dir=None):
        """
        Initialize the visualization tool

        Args:
            metadata_path: Path to metadata.json file
            dataset_root: Dataset root directory
            images_dir: Images directory (default: dataset_root/images)
            annotations_dir: Annotations directory (default: dataset_root/annotations)
        """
        self.metadata_path = metadata_path
        self.dataset_root = Path(dataset_root)

        # Set directory paths
        if images_dir:
            self.images_dir = Path(images_dir)
        else:
            self.images_dir = self.dataset_root / "images"

        if annotations_dir:
            self.annotations_dir = Path(annotations_dir)
        else:
            self.annotations_dir = self.dataset_root / "annotations"

        # Create output directory
        self.output_dir = self.dataset_root / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.load_metadata()

    def load_metadata(self):
        """Load metadata.json file"""
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"Loaded {len(self.metadata)} data entries")

    def get_image_path(self, image_path_from_metadata):
        """
        Get the full image path

        Args:
            image_path_from_metadata: Relative image path from metadata
        """
        # If already an absolute path, use directly
        if os.path.isabs(image_path_from_metadata):
            return Path(image_path_from_metadata)
        # Otherwise, relative to images_dir
        return self.images_dir / Path(image_path_from_metadata).name

    def draw_bbox_on_image(self, image, bbox, normalized_bbox=None, ref_exp="",
                          color='red', linewidth=2, alpha=0.3, show_text=True):
        """
        Draw bounding box on image

        Args:
            image: PIL Image or numpy array
            bbox: [x_min, y_min, x_max, y_max]
            normalized_bbox: Normalized bounding box
            ref_exp: Corresponding text description
            color: Box color
            linewidth: Line width
            alpha: Fill transparency
            show_text: Whether to show text
        """
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)

        # Draw bounding box
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        # Draw rectangle
        rect = Rectangle((x_min, y_min), width, height,
                         linewidth=linewidth, edgecolor=color,
                         facecolor=color, alpha=alpha)
        ax.add_patch(rect)

        # Add text description
        if show_text and ref_exp:
            # Add text above the box
            text_x = x_min
            text_y = y_min - 5 if y_min > 20 else y_max + 15
            ax.text(text_x, text_y, ref_exp,
                   fontsize=10, color='white',
                   bbox=dict(boxstyle="round,pad=0.3",
                            facecolor=color, alpha=0.8))

        # Hide axes
        ax.axis('off')
        plt.tight_layout()

        return fig, ax

    def visualize_single_entry(self, entry, save_path=None, show_normalized=False):
        """
        Visualize a single data entry

        Args:
            entry: Entry dictionary from metadata
            save_path: Save path (auto-generated if not specified)
            show_normalized: Whether to also show normalized coordinates
        """
        # Get image path
        image_path = self.get_image_path(entry['image_path'])

        if not image_path.exists():
            print(f"Image does not exist: {image_path}")
            return None

        # Load image
        try:
            image = Image.open(image_path)
            image = np.array(image)
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return None

        # Create figure
        fig, ax = self.draw_bbox_on_image(
            image=image,
            bbox=entry['bbox'],
            ref_exp=entry['ref_exp']
        )

        # Add title info
        title_parts = [
            f"ID: {entry['id']}",
            f"Original ID: {entry['original_id']}",
            f"Source: {entry['source_file']}"
        ]

        if show_normalized and 'normalized_bbox' in entry:
            norm_bbox = entry['normalized_bbox']
            title_parts.append(f"Norm: [{norm_bbox[0]:.3f}, {norm_bbox[1]:.3f}, {norm_bbox[2]:.3f}, {norm_bbox[3]:.3f}]")

        ax.set_title("\n".join(title_parts), fontsize=12, y=1.05)

        # Save image
        if save_path is None:
            save_path = self.output_dir / f"{entry['id']}_visualized.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        print(f"Saved visualization result: {save_path}")
        return save_path

    def visualize_random_samples(self, n_samples=5, show_normalized=False):
        """
        Randomly visualize n samples

        Args:
            n_samples: Number of samples
            show_normalized: Whether to show normalized coordinates
        """
        if n_samples > len(self.metadata):
            n_samples = len(self.metadata)

        # Randomly select samples
        indices = random.sample(range(len(self.metadata)), n_samples)

        saved_paths = []
        for idx in indices:
            entry = self.metadata[idx]
            save_path = self.output_dir / f"random_{entry['id']}_visualized.png"
            saved_path = self.visualize_single_entry(
                entry,
                save_path=save_path,
                show_normalized=show_normalized
            )
            if saved_path:
                saved_paths.append(saved_path)

        print(f"\nCompleted visualization of {len(saved_paths)}/{n_samples} samples")
        return saved_paths

    def visualize_all(self, batch_size=50, start_idx=0, end_idx=None,
                     show_normalized=False, skip_existing=True):
        """
        Visualize all data entries (process in batches to avoid memory issues)

        Args:
            batch_size: Batch size
            start_idx: Start index
            end_idx: End index
            show_normalized: Whether to show normalized coordinates
            skip_existing: Whether to skip existing files
        """
        if end_idx is None:
            end_idx = len(self.metadata)

        total = end_idx - start_idx
        print(f"Starting visualization of {total} samples...")

        saved_count = 0
        skipped_count = 0

        for i in range(start_idx, end_idx, batch_size):
            batch_end = min(i + batch_size, end_idx)
            print(f"Processing batch: {i+1}-{batch_end}/{end_idx}")

            for j in range(i, batch_end):
                entry = self.metadata[j]
                save_path = self.output_dir / f"{entry['id']}_visualized.png"

                # Check if already exists
                if skip_existing and save_path.exists():
                    skipped_count += 1
                    if skipped_count % 100 == 0:
                        print(f"  Skipped {skipped_count} existing files...")
                    continue

                saved = self.visualize_single_entry(
                    entry,
                    save_path=save_path,
                    show_normalized=show_normalized
                )
                if saved:
                    saved_count += 1

                # Show progress every 100 processed
                if (saved_count + skipped_count) % 100 == 0:
                    print(f"  Progress: {saved_count + skipped_count}/{total}")

            # Clean up memory
            plt.close('all')

        print(f"\nComplete! Saved {saved_count} files, skipped {skipped_count} existing files")

    def create_collage(self, entry_ids, n_cols=3, figsize=(20, 15),
                      save_name="collage.png"):
        """
        Create a collage containing multiple samples

        Args:
            entry_ids: List of entry IDs to include
            n_cols: Number of columns
            figsize: Figure size
            save_name: Save file name
        """
        n_samples = len(entry_ids)
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        for i, entry_id in enumerate(entry_ids):
            # Find entry
            entry = None
            for e in self.metadata:
                if e['id'] == entry_id:
                    entry = e
                    break

            if entry is None:
                axes[i].axis('off')
                axes[i].text(0.5, 0.5, f"ID not found:\n{entry_id}",
                           ha='center', va='center', fontsize=10)
                continue

            # Load image
            image_path = self.get_image_path(entry['image_path'])
            if not image_path.exists():
                axes[i].axis('off')
                axes[i].text(0.5, 0.5, f"Image not found:\n{image_path.name}",
                           ha='center', va='center', fontsize=10)
                continue

            try:
                image = Image.open(image_path)
                image = np.array(image)

                # Display in subplot
                axes[i].imshow(image)

                # Draw bounding box
                bbox = entry['bbox']
                x_min, y_min, x_max, y_max = bbox
                width = x_max - x_min
                height = y_max - y_min

                rect = Rectangle((x_min, y_min), width, height,
                               linewidth=2, edgecolor='red',
                               facecolor='red', alpha=0.3)
                axes[i].add_patch(rect)

                # Add text
                axes[i].text(x_min, max(y_min - 5, 10), entry['ref_exp'],
                           fontsize=8, color='white',
                           bbox=dict(boxstyle="round,pad=0.2",
                                    facecolor='red', alpha=0.8))

                # Set title
                axes[i].set_title(f"ID: {entry['id']}\nRef: {entry['ref_exp'][:30]}..."
                                if len(entry['ref_exp']) > 30 else f"ID: {entry['id']}",
                                fontsize=9)

            except Exception as e:
                axes[i].axis('off')
                axes[i].text(0.5, 0.5, f"Error: {str(e)[:20]}",
                           ha='center', va='center', fontsize=8)

            axes[i].axis('off')

        # Hide extra subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()

        # Save collage
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved collage: {save_path}")
        return save_path

    def generate_report(self, save_name="visualization_report.txt"):
        """
        Generate visualization report
        """
        report_path = self.output_dir / save_name

        with open(report_path, 'w') as f:
            f.write("RoboRefit Benchmark Visualization Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Data Statistics:\n")
            f.write(f"  Total entries: {len(self.metadata)}\n")

            # Count entries with images and annotations
            has_image = sum(1 for e in self.metadata if e.get('has_image', False))
            has_annotation = sum(1 for e in self.metadata if e.get('has_annotation', False))

            f.write(f"  Entries with images: {has_image}\n")
            f.write(f"  Entries with annotations: {has_annotation}\n\n")

            # Show a few examples
            f.write("Example entries:\n")
            for i, entry in enumerate(self.metadata[:5]):
                f.write(f"\nExample {i+1}:\n")
                f.write(f"  ID: {entry['id']}\n")
                f.write(f"  Text description: {entry['ref_exp']}\n")
                f.write(f"  BBox: {entry['bbox']}\n")
                if 'normalized_bbox' in entry:
                    f.write(f"  Norm BBox: {[f'{x:.4f}' for x in entry['normalized_bbox']]}\n")
                f.write(f"  Image path: {entry['image_path']}\n")

            f.write(f"\nVisualization output directory: {self.output_dir}\n")

        print(f"Generated report: {report_path}")
        return report_path


def main():
    """Main function example"""
    # Set paths
    metadata_path = "data/preprocess_data/roborefit-benchmark-dataset/metadata.json"
    dataset_root = "data/preprocess_data/roborefit-benchmark-dataset"

    # Initialize visualizer
    visualizer = RoboRefitVisualizer(metadata_path, dataset_root)

    # 1. Visualize random samples
    print("1. Visualizing random samples...")
    visualizer.visualize_random_samples(n_samples=5, show_normalized=True)

    # 2. Visualize specific entry
    print("\n2. Visualizing specific entry...")
    sample_entry = visualizer.metadata[0]  # First entry
    visualizer.visualize_single_entry(sample_entry, show_normalized=True)

    # 3. Create collage
    print("\n3. Creating collage...")
    # Select first 9 entries
    entry_ids = [visualizer.metadata[i]['id'] for i in range(min(9, len(visualizer.metadata)))]
    visualizer.create_collage(entry_ids, n_cols=3, save_name="sample_collage.png")

    # 4. Generate report
    print("\n4. Generating report...")
    visualizer.generate_report()

    print("\nVisualization complete! Results saved in:", visualizer.output_dir)


if __name__ == "__main__":
    main()
