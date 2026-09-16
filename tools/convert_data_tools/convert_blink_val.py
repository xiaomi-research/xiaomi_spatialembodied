# UPDATE: Replace placeholder paths with your actual paths.
# Data location = "data/utils/public_3d_datasets/BLINK"
# Output location = "data/utils/public_3d_datasets/BLINK/converted_blink_data.jsonl"
# Image location = "data/utils/public_3d_datasets/BLINK/converted_images"
import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import io
import sys

def convert_blink_parquet_to_jsonl(root_dir, output_jsonl_path, image_output_root):
    """
    Convert all Parquet files in the BLINK dataset directory to the specified JSONL format.

    Args:
        root_dir: Root directory path of the BLINK dataset (containing subfolders like Art_Style).
        output_jsonl_path: Output JSONL file path.
        image_output_root: Root directory for saving converted images.
    """
    root_path = Path(root_dir)
    output_jsonl_path = Path(output_jsonl_path)
    image_output_root = Path(image_output_root)

    # Ensure image output root directory exists
    image_output_root.mkdir(parents=True, exist_ok=True)

    all_converted_data = []

    # Iterate through all subdirectories under root directory
    for task_dir in root_path.iterdir():
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        print(f"Processing task directory: {task_name}")

        # Find all .parquet files in this task folder
        parquet_files = list(task_dir.glob("val*.parquet"))
        if not parquet_files:
            print(f"  No parquet files found in {task_name}. Skipping.")
            continue

        for parquet_file in parquet_files:
            print(f"  Reading {parquet_file.name}...")
            try:
                df = pd.read_parquet(parquet_file)
            except Exception as e:
                print(f"    Error reading {parquet_file}: {e}")
                continue

            for idx, row in df.iterrows():
                # 1. Build images path list and save images
                image_paths = []
                # Assume image field names are image_1, image_2, image_3, image_4, some may be None
                for img_idx in range(1, 5):
                    col_name = f'image_{img_idx}'
                    if col_name not in row or pd.isna(row[col_name]):
                        # If column doesn't exist or value is empty, skip this image slot
                        continue

                    img_data = row[col_name]
                    # Assume img_data is a dict with a 'bytes' key
                    if isinstance(img_data, dict) and 'bytes' in img_data:
                        img_bytes = img_data['bytes']
                    elif isinstance(img_data, bytes):
                        img_bytes = img_data
                    else:
                        # If format is unexpected, skip this image
                        print(f"    Warning: Unexpected image data format in row {idx}, column {col_name}. Skipping this image.")
                        continue

                    try:
                        # Generate unique filename and save path for each image
                        # Use task name, filename, row index and image index to construct path
                        img_filename = f"{task_name}_{parquet_file.stem}_{idx}_{img_idx}.jpg"
                        # Can create subdirectory by task to avoid filename conflicts
                        task_image_dir = image_output_root / task_name
                        task_image_dir.mkdir(parents=True, exist_ok=True)

                        img_save_path = task_image_dir / img_filename
                        # img_save_path_f = os.path.join(blink_root_directory, str(img_save_path))
                        # print(img_save_path, 'img_save_path')
                        # Open from byte stream and save as JPEG
                        image = Image.open(io.BytesIO(img_bytes))
                        # Convert to RGB mode for compatibility
                        if image.mode in ('RGBA', 'LA', 'P'):
                            image = image.convert('RGB')
                        image.save(img_save_path, format='JPEG')

                        # Record saved path (as string)
                        image_paths.append(str(img_save_path))
                        # print(image_paths)
                    except Exception as e:
                        print(f"    Error processing image in row {idx}, column {col_name}: {e}")
                        # If image processing fails, can choose to skip or continue
                        continue

                # If no images were successfully processed, skip this sample
                if not image_paths:
                    print(f"    Warning: No valid images in row {idx}. Skipping this sample.")
                    continue

                # 2. Build messages conversation content
                # User message: use <image> placeholders + question text
                # Per example, place as many <image> placeholders as image paths
                image_placeholders = "".join(["<image>" for _ in range(len(image_paths))])

                # Get question text, assume column name is 'question'
                user_content = f"{image_placeholders}{row.get('prompt', '')}"
                # Assistant message: this is a key point, in the example 'answer' is 'hidden'.
                # You need to determine the source of the assistant answer based on your actual data.
                # Here are several possible handling approaches, you need to choose or modify:
                assistant_content = ""

                # Possibility A: answer is in 'choices' list, correct answer indicated by 'answer' field index (but in docs it's 'hidden')
                choices = row.get('choices', [])
                # for choice in choices:
                #     user_content += choice
                answer_key = row.get('answer')
                assistant_content = answer_key
                user_content = user_content.replace('(', '').replace(')', '.')
                assistant_content = assistant_content.strip('()')
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]

                meta_data = {"sub_task": row.get('sub_task'), "idx": row.get('idx')}

                # 4. Build final sample dictionary
                sample_dict = {
                    "messages": messages,
                    "images": image_paths,
                    "meta_data": meta_data
                }

                all_converted_data.append(sample_dict)
    # 5. Write all data to JSONL file
    print(f"\nWriting {len(all_converted_data)} samples to {output_jsonl_path}...")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_converted_data:
            print(item)
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("Conversion completed!")

# Usage example
if __name__ == "__main__":
    # Please modify the following parameters according to your actual paths
    blink_root_directory = "data/utils/public_3d_datasets/BLINK"
    output_jsonl_file = "data/utils/public_3d_datasets/BLINK/converted_blink_data.jsonl"
    converted_images_root = "data/utils/public_3d_datasets/BLINK/converted_images"  # Directory for saving converted images

    # Run conversion function
    convert_blink_parquet_to_jsonl(blink_root_directory, output_jsonl_file, converted_images_root)
