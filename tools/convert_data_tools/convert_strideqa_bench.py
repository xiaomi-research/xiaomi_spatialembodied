# UPDATE: Replace placeholder paths with your actual paths.
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import yaml
from PIL import Image
import numpy as np
import os
import sys

# Add path to import functions from strideqa_bench
sys.path.append('eval/STRIDE-QA-Dataset-main/benchmarks/STRIDE-QA-Bench/src')

try:
    from strideqa_bench.inference.utils.som import (
        draw_som_on_image,
        resize_mask,
        restore_mask_from_rle,
    )
    from strideqa_bench.inference.utils.video import save_videov2 as save_video
except ImportError as e:
    # There is deliberately NO fallback here any more.
    #
    # The placeholder implementations that used to live here silently fabricated data:
    # `restore_mask_from_rle` returned a RANDOM boolean mask, and `save_video` wrote only the
    # first frame as a `.png` while returning True, so the caller recorded a `.mp4` path that
    # did not exist. Both produced training/benchmark data that looked well-formed and was
    # meaningless. Fail loudly instead.
    raise ImportError(
        "convert_strideqa_bench.py requires the STRIDE-QA benchmark utilities "
        "(`strideqa_bench.inference.utils.som` and `.video`).\n"
        "  They are not part of this repository. Clone the official STRIDE-QA dataset repo "
        "and point the `sys.path.append(...)` line at the top of this file at its "
        "`benchmarks/STRIDE-QA-Bench/src` directory.\n"
        "  This script will not run without them: the previous placeholder fallback emitted "
        "random masks and bogus video paths."
    ) from e

class STRIDEQABenchToMSSwiftConverter:
    """Converter for STRIDE-QA-Bench data to ms-swift format"""

    def __init__(
            self,
            input_path: Path,
            output_dir: Path,
            image_folder: Path,
            system_prompt: Optional[str] = None,
            target_size: tuple = (1024, 1024),
            save_mode: str = "image",  # "image" or "video"
            video_fps: int = 1,
            data_type: str = "bench",  # "mini" or "bench"
            create_subdir: bool = True,  # Whether to create subdirectory
        ):
        """
        Initialize converter

        Args:
            input_path: Input JSON file path
            output_dir: Output directory
            image_folder: Original image folder
            system_prompt: System prompt text
            target_size: Target image size
            save_mode: Save mode, "image" saves as multiple images, "video" saves as video
            video_fps: Video frame rate
            data_type: Data type, "mini" or "bench"
            create_subdir: Whether to create a subdirectory based on input filename
        """
        self.input_path = Path(input_path)
        self.original_output_dir = Path(output_dir)
        self.image_folder = Path(image_folder)
        self.target_size = target_size
        self.save_mode = save_mode
        self.video_fps = video_fps
        self.data_type = data_type

        # Set system prompt
        if system_prompt:
            self.system_prompt = system_prompt.strip()
        else:
            self.system_prompt = "You are a helpful driver, which is good at spatial and temporal analysis given a sequence of images or video."

        # Benchmark time bucket definition (consistent with original code)
        self.time_buckets = ["prev_3", "prev_2", "prev_1", "current"]


        # if '.json' in self.original_output_dir:
        #     self.original_output_dir =

        # Determine the final output directory
        if create_subdir:
            # Use the input filename (without extension) as subdirectory name
            subdir_name = self.input_path.stem
            self.output_dir = self.original_output_dir / subdir_name
        else:
            self.output_dir = self.original_output_dir



        # breakpoint()
        # Ensure output directory exists and is a directory
        # self.output_dir = ÷
        self._ensure_directory(self.output_dir)

        if save_mode == "image":
            self._ensure_directory(self.output_dir / "images")
        elif save_mode == "video":
            self._ensure_directory(self.output_dir / "videos")

        # Load original data
        with open(self.input_path, 'r', encoding='utf-8') as f:
            self.original_data = json.load(f)

        print(f"Loaded {len(self.original_data)} samples, data type: {self.data_type}")
        print(f"Output directory: {self.output_dir}")

    def _ensure_directory(self, path: Path) -> None:
        """Ensure the specified path is a directory"""
        if path.exists():
            if path.is_file():
                # If a file with the same name exists, delete it
                print(f"Warning: File with same name exists, will delete: {path}")
                try:
                    path.unlink()
                except Exception as e:
                    print(f"Failed to delete file {path}: {e}")
                    raise NotADirectoryError(f"Cannot create directory, file with same name exists: {path}")
            # If it's already a directory, return directly
            return
        # Create directory
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Failed to create directory {path}: {e}")
            raise

    def detect_data_type(self) -> str:
        """
        Auto-detect data type
        """
        if not self.original_data:
            return "unknown"

        sample = self.original_data[0]

        # Check if there is a conversations field (STRIDE-QA-Mini format)
        if "conversations" in sample:
            return "mini"
        # Check if there are question and gt fields (STRIDE-QA-Bench format)
        elif "question" in sample and "gt" in sample:
            return "bench"
        else:
            return "unknown"

    def load_image_frames(self, sample: Dict[str, Any]) -> List[Image.Image]:
        """
        Load image frames
        """
        images = []

        frame_paths = sample.get("images", [])
        if not frame_paths:
            print(f"Warning: Sample {sample.get('question_id', 'unknown')} has no image information")
            return images

        # Load each frame
        for i, frame_file in enumerate(frame_paths):
            frame_path = self.image_folder / frame_file

            if not frame_path.exists():
                # Try to find in subdirectories of image_folder
                found = False
                for root, dirs, files in os.walk(self.image_folder):
                    for file in files:
                        if file == Path(frame_file).name:
                            frame_path = Path(root) / file
                            found = True
                            break
                    if found:
                        break

                if not found:
                    print(f"Warning: Image does not exist {frame_path}")
                    continue

            try:
                # Load image
                img = Image.open(frame_path).convert("RGB")
                # Resize
                img = img.resize(self.target_size)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {frame_path}: {e}")

        return images

    def validate_question(self, q: dict) -> bool:
        """
        Validate whether question data is valid
        """
        question_id = q.get("question_id", "unknown")
        if "rle" not in q:
            print(f"Warning: No RLE data found for question_id {question_id}")
            return False
        rle_data = q["rle"]
        if "current" not in rle_data:
            print(f"Warning: No 'current' RLE data found for question_id {question_id}")
            return False
        return True

    def prepare_images_with_masks_bench(self, sample: Dict[str, Any]) -> List[Image.Image]:
        """
        Prepare images with masks for bench data
        Each frame has a corresponding mask (prev_3, prev_2, prev_1, current)
        """
        if not self.validate_question(sample):
            return []

        images = self.load_image_frames(sample)
        if not images:
            return []

        region_id = sample.get("region_id", 0)
        images_with_masks = []

        # Process corresponding mask for each frame
        for idx, img in enumerate(images):
            if idx < len(self.time_buckets):
                bucket = self.time_buckets[idx]
            else:
                # If more frames than time buckets, use the last one (current)
                bucket = self.time_buckets[-1]

            # Check if there is RLE data for the corresponding time bucket
            rle_dict = sample.get("rle", {})
            if bucket not in rle_dict:
                print(f"Warning: Sample {sample.get('question_id', 'unknown')} has no RLE data for {bucket}")
                images_with_masks.append(img)
                continue

            # Convert to numpy array
            image_rgb = np.array(img)

            # Get mask
            mask = restore_mask_from_rle(rle_dict[bucket])
            mask = resize_mask(mask, self.target_size)

            # Draw mask on image
            image_som_np = draw_som_on_image(
                image_rgb,
                [mask],
                [region_id],
                alpha=0.2,
                anno_mode=["Mask", "Mark"]
            )

            images_with_masks.append(Image.fromarray(image_som_np))

        return images_with_masks

    def prepare_images_with_masks_mini(self, sample: Dict[str, Any]) -> List[Image.Image]:
        """
        Prepare images with masks for mini data
        Only draw all masks on the last frame
        """
        images = self.load_image_frames(sample)
        if not images:
            return []

        # Check if there are RLE masks
        rle_list = sample.get("rle", [])
        region_list = sample.get("region", [])

        if not rle_list:
            return images

        # Prepare all masks
        masks = []
        region_ids = []

        for i, rle_data in enumerate(rle_list):
            mask = restore_mask_from_rle(rle_data)
            mask = resize_mask(mask, self.target_size)
            masks.append(mask)

            if i < len(region_list):
                region_info = region_list[i]
                if isinstance(region_info, list) and len(region_info) > 1:
                    region_id = str(region_info[1])
                else:
                    region_id = str(i)
            else:
                region_id = str(i)
            region_ids.append(region_id)

        # Only draw all masks on the last frame
        if masks and region_ids:
            target_image_np = np.array(images[-1])
            image_with_mask_np = draw_som_on_image(
                target_image_np,
                masks,
                region_ids,
                alpha=0.2,
                anno_mode=["Mask", "Mark"]
            )
            images[-1] = Image.fromarray(image_with_mask_np)

        return images

    def prepare_images_with_masks(self, sample: Dict[str, Any]) -> List[Image.Image]:
        """
        Prepare images with masks
        """
        if self.data_type == "mini":
            return self.prepare_images_with_masks_mini(sample)
        else:  # bench
            return self.prepare_images_with_masks_bench(sample)

    def build_conversation_for_mini(self, sample: Dict[str, Any]) -> List[Dict]:
        """
        Build conversation for mini data
        """
        conversations = []

        # Add system prompt
        if self.system_prompt:
            conversations.append({
                "role": "system",
                "content": self.system_prompt
            })

        # Convert original conversation format
        for conv in sample["conversations"]:
            role = conv["from"]
            content = conv["value"]

            # Convert role names
            if role == "human":
                ms_role = "user"
            elif role == "gpt":
                ms_role = "assistant"
            else:
                ms_role = role

            conversations.append({
                "role": ms_role,
                "content": content
            })

        return conversations

    def build_conversation_for_bench(self, sample: Dict[str, Any]) -> List[Dict]:
        """
        Build conversation for bench data
        """
        conversations = []

        # Add system prompt
        if self.system_prompt:
            conversations.append({
                "role": "system",
                "content": self.system_prompt
            })

        # Build single-turn conversation
        question = sample.get("question", "")
        answer = sample.get("gt", "")

        conversations.append({
            "role": "user",
            "content": question
        })

        conversations.append({
            "role": "assistant",
            "content": answer
        })

        return conversations

    def add_media_tags_to_conversation(
        self,
        conversations: List[Dict],
        num_images: int
    ) -> List[Dict]:
        """
        Add media tags to conversation
        """
        if num_images <= 0:
            return conversations

        # Find the first user message
        for i, conv in enumerate(conversations):
            if conv["role"] == "user":
                # Check if tags have already been added
                if not conv["content"].startswith("<"):
                    if self.save_mode == "video":
                        # Video mode: add <video> tag
                        conversations[i]["content"] = f"<video>\n{conv['content']}"
                    else:
                        # Image mode: add <image> tags based on frame count
                        image_tags = "".join(["<image>" for _ in range(num_images)])
                        conversations[i]["content"] = f"{image_tags}\n{conv['content']}"
                break

        return conversations

    def save_media(
        self,
        sample_id: str,
        images: List[Image.Image]
    ) -> List[str]:
        """
        Save media files
        """
        # Clean special characters from sample_id
        safe_sample_id = sample_id.replace('/', '_').replace('\\', '_')

        if self.save_mode == "image":
            # Save as multiple images
            media_paths = []
            for i, img in enumerate(images):
                # Generate filename
                if len(images) == 1:
                    img_filename = f"{safe_sample_id}.png"
                else:
                    # Use time bucket for naming
                    if i < len(self.time_buckets) and self.data_type == "bench":
                        time_bucket = self.time_buckets[i]
                        img_filename = f"{safe_sample_id}_{time_bucket}.png"
                    else:
                        img_filename = f"{safe_sample_id}_frame_{i:03d}.png"

                img_path = self.output_dir / "images" / img_filename
                img.save(img_path, "PNG")
                media_paths.append(str(img_path.absolute()))

            return media_paths

        elif self.save_mode == "video":
            # Save as video
            video_filename = f"{safe_sample_id}.mp4"
            video_path = self.output_dir / "videos" / video_filename

            # Ensure video directory exists
            self._ensure_directory(video_path.parent)

            # Save video
            try:
                save_video(images, video_path, fps=self.video_fps, align_macro_block_size=True)
                return [str(video_path.absolute())]
            except Exception as e:
                print(f"Failed to save video {video_path}: {e}")
                return []

    def get_sample_id(self, sample: Dict[str, Any]) -> str:
        """Get sample ID"""
        if self.data_type == "mini":
            return sample.get("id", f"mini_{abs(hash(str(sample))) % 10000:04d}")
        else:  # bench
            return sample.get("question_id", f"bench_{abs(hash(str(sample))) % 10000:04d}")

    def get_task_type(self, sample: Dict[str, Any]) -> str:
        """Get task type"""
        if self.data_type == "mini":
            filename = self.input_path.name.lower()
            if "spatial" in filename:
                return "spatial"
            elif "temporal" in filename:
                return "temporal"
            else:
                return "unknown"
        else:  # bench
            return sample.get("qa_category", "unknown")

    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single sample
        """
        sample_id = self.get_sample_id(sample)

        try:
            # 1. Prepare images with masks
            images = self.prepare_images_with_masks(sample)

            if not images:
                print(f"Warning: Sample {sample_id} has no images, skipping")
                return None

            # 2. Save media files
            media_paths = self.save_media(sample_id, images)
            if not media_paths:
                print(f"Warning: Sample {sample_id} failed to save media files")
                return None

            # 3. Build conversation
            if self.data_type == "mini":
                conversations = self.build_conversation_for_mini(sample)
            else:  # bench
                conversations = self.build_conversation_for_bench(sample)

            # 4. Add media tags
            conversations = self.add_media_tags_to_conversation(conversations, len(images))

            # 5. Build ms-swift format sample
            ms_sample = {
                "messages": conversations
            }

            # 6. Add media paths
            if self.save_mode == "image":
                ms_sample["images"] = media_paths
            elif self.save_mode == "video":
                ms_sample["videos"] = media_paths

            # 7. Add metadata
            ms_sample["metadata"] = {
                "id": sample_id,
                "data_type": self.data_type,
                "task_type": self.get_task_type(sample),
                "num_frames": len(images),
                "save_mode": self.save_mode
            }

            # 8. Add extra information for bench format
            if self.data_type == "bench":
                ms_sample["metadata"].update({
                    "group_id": sample.get("group_id", ""),
                    "qa_category": sample.get("qa_category", ""),
                    "object_class": sample.get("object_class", ""),
                    "region_id": sample.get("region_id", 0)
                })

            return ms_sample

        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def convert(self, sample_ratio: float = 1.0) -> List[Dict[str, Any]]:
        """
        Convert the entire dataset
        """
        if sample_ratio < 1.0:
            data_to_process = self.original_data[:int(len(self.original_data) * sample_ratio)]
        else:
            data_to_process = self.original_data

        print(f"Starting conversion of {len(data_to_process)} samples...")

        converted_samples = []
        for i, sample in enumerate(data_to_process):
            if i % 10 == 0 and i > 0:
                print(f"Processing progress: {i}/{len(data_to_process)}")

            converted_sample = self.process_sample(sample)
            if converted_sample:
                converted_samples.append(converted_sample)

        print(f"Successfully converted {len(converted_samples)} samples")

        return converted_samples

    def save_converted_data(
        self,
        converted_samples: List[Dict[str, Any]],
        output_filename: str = None
    ) -> Path:
        """
        Save converted data
        """
        if output_filename is None:
            input_stem = self.input_path.stem
            output_filename = f"{input_stem}_{self.data_type}_{self.save_mode}.jsonl"

        output_path = self.output_dir / output_filename

        # Ensure output directory exists
        self._ensure_directory(output_path.parent)

        # Save as JSONL format
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in converted_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"Converted data saved to: {output_path}")

        # Also save as JSON format for easier viewing
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(converted_samples, f, indent=2, ensure_ascii=False)

        print(f"JSON format data saved to: {json_path}")

        return output_path

def batch_convert_files(
    input_dir: str,
    output_dir: str,
    image_folder: str,
    system_prompt: Optional[str] = None,
    target_size: tuple = (532, 336),
    save_mode: str = "image",
    video_fps: int = 1,
    data_type: str = "bench",
    file_pattern: str = "strideqa_bench_*.json",
    sample_ratio: float = 1.0,
    create_subdir: bool = True,
):
    """
    Batch convert multiple files

    Args:
        input_dir: Input directory
        output_dir: Output directory
        image_folder: Image folder
        system_prompt: System prompt
        target_size: Target size
        save_mode: Save mode
        video_fps: Video frame rate
        data_type: Data type
        file_pattern: File matching pattern
        sample_ratio: Processing ratio
        create_subdir: Whether to create subdirectory
    """
    input_path = Path(input_dir)
    base_output_dir = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    # Ensure output directory exists
    if not base_output_dir.exists():
        base_output_dir.mkdir(parents=True, exist_ok=True)

    # Find matching files
    json_files = list(input_path.glob(file_pattern))
    if not json_files:
        print(f"Error: No files matching {file_pattern} found in directory {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files:")
    for f in json_files:
        print(f"  - {f.name}")

    all_converted_samples = []

    for json_file in json_files:
        print(f"\n{'='*60}")
        print(f"Processing file: {json_file.name}")
        print(f"{'='*60}")

        # breakpoint()
        # Create converter
        converter = STRIDEQABenchToMSSwiftConverter(
            input_path=json_file,
            output_dir=base_output_dir,
            image_folder=image_folder,
            system_prompt=system_prompt,
            target_size=target_size,
            save_mode=save_mode,
            video_fps=video_fps,
            data_type=data_type,
            create_subdir=create_subdir,
        )

        # Auto-detect data type
        if converter.data_type == "auto":
            detected_type = converter.detect_data_type()
            if detected_type == "unknown":
                print(f"Error: Cannot auto-detect data type: {json_file.name}")
                continue
            converter.data_type = detected_type
            print(f"Auto-detected data type: {detected_type}")

        # Convert data
        converted_samples = converter.convert(sample_ratio=sample_ratio)

        if not converted_samples:
            print(f"Warning: File {json_file.name} did not successfully convert any samples")
            continue

        # Save converted data
        file_output_path = converter.save_converted_data(converted_samples)

        all_converted_samples.extend(converted_samples)

    # Summary information
    if all_converted_samples:
        print(f"\n{'='*60}")
        print("Batch processing complete! Summary:")
        print(f"{'='*60}")
        print(f"Total samples: {len(all_converted_samples)}")
        print(f"Files processed: {len(json_files)}")

        # Count data type distribution
        data_types = {}
        for sample in all_converted_samples:
            data_type = sample["metadata"].get("data_type", "unknown")
            data_types[data_type] = data_types.get(data_type, 0) + 1

        if data_types:
            print("\nData type distribution:")
            for dt, count in data_types.items():
                print(f"  {dt}: {count}")

        # Count task type distribution
        task_types = {}
        for sample in all_converted_samples:
            task_type = sample["metadata"].get("task_type", "unknown")
            task_types[task_type] = task_types.get(task_type, 0) + 1

        if task_types:
            print("\nTask type distribution:")
            for tt, count in task_types.items():
                print(f"  {tt}: {count}")

        # Save all samples to a single file
        all_output_path = Path(output_dir) / f"all_samples_{data_type}_{save_mode}.jsonl"
        with open(all_output_path, 'w', encoding='utf-8') as f:
            for sample in all_converted_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"\nAll samples merged and saved to: {all_output_path}")

    return all_converted_samples

def main():
    parser = argparse.ArgumentParser(
        description="Convert STRIDE-QA data to ms-swift format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single STRIDE-QA file
  python convert_strideqa.py --input /path/to/strideqa_bench_t0.json --output_dir /path/to/preprocessed_data/STRIDE-QA-Bench --image_folder /path/to/STRIDE-QA-Bench/images --data_type bench --save_mode image

  # Batch process multiple files
  python convert_strideqa.py --input_dir /path/to/annotation_files --output_dir /path/to/preprocessed_data/STRIDE-QA-Bench --image_folder /path/to/STRIDE-QA-Bench/images --batch --data_type bench --save_mode image

  # Process as video format
  python convert_strideqa.py --input /path/to/strideqa_bench_t0.json --output_dir /path/to/preprocessed_data/STRIDE-QA-Bench --image_folder /path/to/STRIDE-QA-Bench/images --data_type bench --save_mode video --video_fps 5

  # Use custom system prompt
  python convert_strideqa.py --input /path/to/strideqa_bench_t0.json --output_dir /path/to/preprocessed_data/STRIDE-QA-Bench --image_folder /path/to/STRIDE-QA-Bench/images --system_prompt "You are a professional driving assistant, skilled at analyzing spatial and temporal information." --save_mode image
        """
    )

    # Input/output parameters
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str,
                           help="Single input JSON file path")
    input_group.add_argument("--input_dir", type=str,
                           help="Directory path containing multiple JSON files")

    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory path")
    parser.add_argument("--image_folder", type=str, required=True,
                       help="Original image folder path")

    # Batch processing parameters
    parser.add_argument("--batch", action="store_true",
                       help="Batch processing mode (automatically enabled when --input_dir is specified)")
    parser.add_argument("--file_pattern", type=str, default="strideqa_bench_*.json",
                       help="File matching pattern (only used in batch processing)")

    # Optional parameters
    parser.add_argument("--system_prompt", type=str,
                       default="You are a helpful driver, which is good at spatial and temporal analysis given a sequence of images or video.",
                       help="System prompt text")
    parser.add_argument("--data_type", type=str, default="auto",
                       choices=["auto", "mini", "bench"],
                       help="Data type: auto (auto-detect), mini, or bench")
    parser.add_argument("--save_mode", type=str, default="image",
                       choices=["image", "video"],
                       help="Save mode: image (multiple images) or video")
    parser.add_argument("--video_fps", type=int, default=1,
                       help="Video frame rate (only effective when save_mode=video)")
    parser.add_argument("--target_width", type=int, default=532,
                       help="Target image width")
    parser.add_argument("--target_height", type=int, default=336,
                       help="Target image height")
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                       help="Ratio of data to process, for testing (0.0-1.0)")
    parser.add_argument("--output_filename", type=str, default=None,
                       help="Output filename (auto-generated by default)")
    parser.add_argument("--no_subdir", action="store_true",
                       help="Do not create subdirectory (all output in the same directory)")

    args = parser.parse_args()

    # Check image folder
    image_folder = Path(args.image_folder)
    if not image_folder.exists():
        print(f"Warning: Image folder does not exist: {args.image_folder}")
        print("Attempting to create directory...")
        image_folder.mkdir(parents=True, exist_ok=True)

    # Create output directory
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir or args.batch:
        # Batch processing mode
        input_dir = args.input_dir if args.input_dir else Path(args.input).parent if args.input else None

        if not input_dir:
            print("Error: Batch processing requires --input_dir parameter")
            return

        all_samples = batch_convert_files(
            input_dir=input_dir,
            output_dir=output_dir,
            image_folder=image_folder,
            system_prompt=args.system_prompt,
            target_size=(args.target_width, args.target_height),
            save_mode=args.save_mode,
            video_fps=args.video_fps,
            data_type=args.data_type,
            file_pattern=args.file_pattern,
            sample_ratio=args.sample_ratio,
            create_subdir=not args.no_subdir,
        )

    else:
        # Single file processing mode
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file does not exist: {args.input}")
            return

        # Create converter
        converter = STRIDEQABenchToMSSwiftConverter(
            input_path=input_path,
            output_dir=output_dir,
            image_folder=image_folder,
            system_prompt=args.system_prompt,
            target_size=(args.target_width, args.target_height),
            save_mode=args.save_mode,
            video_fps=args.video_fps,
            data_type=args.data_type if args.data_type != "auto" else "auto",
            create_subdir=not args.no_subdir,
        )

        # If data_type is auto, auto-detect
        if converter.data_type == "auto":
            detected_type = converter.detect_data_type()
            if detected_type == "unknown":
                print("Error: Cannot auto-detect data type, please specify using --data_type parameter")
                return
            converter.data_type = detected_type
            print(f"Auto-detected data type: {detected_type}")

        # Convert data
        converted_samples = converter.convert(sample_ratio=args.sample_ratio)

        if not converted_samples:
            print("Error: No samples were successfully converted")
            return

        # Save converted data
        output_path = converter.save_converted_data(converted_samples, args.output_filename)

        # Print statistics
        print("\n" + "="*50)
        print("Conversion complete! Statistics:")
        print(f"Total samples: {len(converted_samples)}")
        print(f"Data type: {converter.data_type}")
        print(f"Save mode: {converter.save_mode}")
        print(f"Output directory: {converter.output_dir}")

        # Count samples by task type
        task_types = {}
        for sample in converted_samples:
            task_type = sample["metadata"].get("task_type", "unknown")
            task_types[task_type] = task_types.get(task_type, 0) + 1

        if task_types:
            print("\nTask type distribution:")
            for task_type, count in task_types.items():
                print(f"  {task_type}: {count}")

        # Print example output
        if converted_samples:
            sample = converted_samples[0]
            print(f"\nExample output (first 500 characters):")
            print(json.dumps(sample, indent=2, ensure_ascii=False)[:500] + "...")

            # Check media files
            if "images" in sample:
                print(f"\nImage save location: {converter.output_dir}/images/")
                print(f"Example image count: {len(sample.get('images', []))}")
            elif "videos" in sample:
                print(f"\nVideo save location: {converter.output_dir}/videos/")
                print(f"Example video: {sample.get('videos', [])[0] if sample.get('videos') else 'None'}")

if __name__ == "__main__":
    main()
