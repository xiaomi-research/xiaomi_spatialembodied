# UPDATE: Replace placeholder paths with your actual paths.
import pandas as pd
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
import io
import datetime

def parse_visual_trace_dataset(parquet_path, output_dir):
    """
    Parse visual_trace dataset

    Args:
        parquet_path: parquet file path
        output_dir: output directory

    Returns:
        List of parsed data
    """
    print("=== Start parsing visual_trace dataset ===")

    # Read data
    df = pd.read_parquet(parquet_path)
    print(f"Read {len(df)} records")

    # Create output directory
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    json_dir = output_dir / "annotations"

    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    parsed_data = []
    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        data_id = str(row.get('id', f"sample_{idx}"))
        record_idx = int(row.get('idx', idx))

        # Progress display
        if (idx + 1) % 10 == 0:
            print(f"Processing: {idx + 1}/{len(df)}")

        # Process image data
        image_path = None
        image_info = {}
        images_data = row.get('images')

        try:
            if isinstance(images_data, np.ndarray) and images_data.shape == (1,) and images_data.dtype == np.object_:
                # Extract byte data from (1,) shaped array
                byte_data = images_data.item(0)

                if isinstance(byte_data, bytes):
                    # Verify if JPEG
                    is_jpeg = False
                    if len(byte_data) >= 2:
                        # JPEG files start with 0xFFD8
                        if byte_data[:2] == b'\xff\xd8':
                            is_jpeg = True

                    # Decode byte data
                    with io.BytesIO(byte_data) as f:
                        try:
                            img = Image.open(f)
                            img.load()  # Ensure image is fully loaded

                            # Save image
                            if is_jpeg:
                                img_path = image_dir / f"{data_id}.jpg"
                                # If JPEG, keep original format
                                with open(img_path, 'wb') as f_img:
                                    f_img.write(byte_data)
                                img_format = 'JPEG'
                            else:
                                img_path = image_dir / f"{data_id}.png"
                                img.save(img_path, format='PNG')
                                img_format = img.format

                            image_path = str(img_path)

                            # Image info
                            image_info = {
                                'format': img_format,
                                'mode': img.mode,
                                'width': img.width,
                                'height': img.height,
                                'size_bytes': len(byte_data)
                            }

                            success_count += 1

                        except Exception as e:
                            print(f"  [Error] Image decode failed (ID: {data_id}): {e}")
                            fail_count += 1
                else:
                    print(f"  [Warning] Not byte data (ID: {data_id}): {type(byte_data)}")
                    fail_count += 1
            else:
                print(f"  [Warning] Unsupported images format (ID: {data_id}): shape={images_data.shape if hasattr(images_data, 'shape') else 'N/A'}, dtype={images_data.dtype if hasattr(images_data, 'dtype') else 'N/A'}")
                fail_count += 1

        except Exception as e:
            print(f"  [Error] Processing failed (ID: {data_id}): {e}")
            fail_count += 1

        # Parse answer field
        answer_info = {}
        answer_str = row.get('answer', '')

        if isinstance(answer_str, str) and answer_str:
            if answer_str.startswith('<type>'):
                end_type_tag = answer_str.find('</type>')
                if end_type_tag != -1:
                    answer_type = answer_str[6:end_type_tag]
                    json_str = answer_str[end_type_tag + 7:]

                    try:
                        answer_data = json.loads(json_str)
                        answer_info = {
                            'type': answer_type,
                            'data': answer_data
                        }
                    except json.JSONDecodeError as e:
                        print(f"  [Warning] JSON parse failed (ID: {data_id}): {e}")
                        answer_info = {
                            'type': answer_type,
                            'raw': json_str
                        }

        # Build record
        record = {
            "id": data_id,
            "idx": record_idx,
            "problem": str(row.get('problem', '')),
            "dataset_name": str(row.get('dataset_name', '')),
            "data_type": str(row.get('data_type', '')),
            "answer_raw": answer_str,
            "answer_parsed": answer_info,
            "image_path": image_path,
            "image_info": image_info
        }

        parsed_data.append(record)

    # Save JSON
    json_path = json_dir / "visual_trace_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Parsing complete!")
    print(f"Total records: {len(parsed_data)}")
    print(f"Successfully saved images: {success_count}")
    print(f"Save failures: {fail_count}")
    print(f"Data saved to: {json_path}")
    print(f"Images saved to: {image_dir}")

    return parsed_data

def test_image_decoding(parquet_path, output_dir, sample_indices=[0, 1, 2]):
    """
    Test image decoding

    Args:
        parquet_path: parquet file path
        output_dir: output directory
        sample_indices: sample indices to test
    """
    print("=== Test image decoding ===")

    df = pd.read_parquet(parquet_path)

    for idx in sample_indices:
        if idx >= len(df):
            continue

        row = df.iloc[idx]
        data_id = row.get('id', f"sample_{idx}")
        images_data = row.get('images')

        print(f"\nTest sample {idx} (ID: {data_id}):")

        if isinstance(images_data, np.ndarray) and images_data.shape == (1,) and images_data.dtype == np.object_:
            byte_data = images_data.item(0)

            if isinstance(byte_data, bytes):
                print(f"  Byte data length: {len(byte_data)}")

                # Check file header
                if len(byte_data) >= 4:
                    hex_header = byte_data[:4].hex()
                    print(f"  File header: {hex_header}")

                    # Identify format
                    if hex_header.startswith('ffd8'):
                        print(f"  Format: JPEG")
                    elif hex_header == '89504e47':
                        print(f"  Format: PNG")
                    elif hex_header.startswith('47494638'):
                        print(f"  Format: GIF")
                    elif hex_header.startswith('424d'):
                        print(f"  Format: BMP")
                    else:
                        print(f"  Format: Unknown")

                # Try to decode
                try:
                    with io.BytesIO(byte_data) as f:
                        img = Image.open(f)
                        img.verify()  # Verify image integrity

                        print(f"  Successfully decoded!")
                        print(f"    Format: {img.format}")
                        print(f"    Size: {img.size}")
                        print(f"    Mode: {img.mode}")

                        # Save test image
                        test_dir = Path(output_dir) / "test_images"
                        test_dir.mkdir(parents=True, exist_ok=True)

                        if img.format == 'JPEG':
                            test_path = test_dir / f"{data_id}_test.jpg"
                            with open(test_path, 'wb') as f_img:
                                f_img.write(byte_data)
                        else:
                            test_path = test_dir / f"{data_id}_test.png"
                            img.save(test_path, format='PNG')

                        print(f"    Test image saved to: {test_path}")

                        # Show first few bytes
                        print(f"    First 20 bytes (hex): {byte_data[:20].hex()}")

                except Exception as e:
                    print(f"  Decode failed: {e}")

                    # Save raw data for debugging
                    test_dir = Path(output_dir) / "debug"
                    test_dir.mkdir(parents=True, exist_ok=True)
                    debug_path = test_dir / f"{data_id}_raw.bin"
                    with open(debug_path, 'wb') as f:
                        f.write(byte_data)
                    print(f"    Raw data saved to: {debug_path}")
            else:
                print(f"  Not byte data: {type(byte_data)}")
        else:
            print(f"  Unsupported format: {type(images_data)}")

            if hasattr(images_data, 'shape'):
                print(f"    Shape: {images_data.shape}")
            if hasattr(images_data, 'dtype'):
                print(f"    dtype: {images_data.dtype}")

def create_readme(output_dir, dataset_info):
    """
    Create README file

    Args:
        output_dir: output directory
        dataset_info: dataset info dictionary
    """
    readme_path = Path(output_dir) / "README.md"

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Embodied-R1 Visual Trace Dataset Parse Results

## Basic Info
- Source data: {dataset_info.get('source_path', 'N/A')}
- Parse time: {dataset_info.get('parse_time', 'N/A')}
- Total samples: {dataset_info.get('total_samples', 0)}
- Saved images: {dataset_info.get('saved_images', 0)}

## Dataset Description
This is a visual trajectory tracking dataset, containing visual trajectory data of robots performing manipulation tasks.

## Data Structure
Each sample contains the following fields:

### 1. Problem
Contains task description, format: `"<image> [task description]..."`

### 2. Answer
Contains visual trajectory data, format: `"<type>fsd_visual_trace</type>{{"trajectory": [[x1, y1], [x2, y2], ...]}}"`

### 3. Images
Original JPEG format images, stored in numpy arrays

## Output Format
### JSON format example:
json
{{
  "id": "fsd_pointbridge_17150",
  "idx": 0,
  "problem": "<image> You are currently a robot performing...",
  "dataset_name": "FSD_visual_trace",
  "data_type": "fsd_visual_trace",
  "answer_raw": "<type>fsd_visual_trace</type>{{"trajectory": [[158, 59], ...]}}",
  "answer_parsed": {{
    "type": "fsd_visual_trace",
    "data": {{
      "trajectory": [[158, 59], [164, 64], ...]
    }}
  }},
  "image_path": "images/fsd_pointbridge_17150.jpg",
  "image_info": {{
    "format": "JPEG",
    "mode": "RGB",
    "width": 640,
    "height": 480,
    "size_bytes": 12345
  }}
}}

## File Structure

{output_dir}/
+-- images/              # Parsed image files
|   +-- fsd_pointbridge_17150.jpg
|   +-- fsd_pointbridge_22450.jpg
|   +-- ...
+-- annotations/         # Annotation info
|   +-- visual_trace_data.json
+-- test_images/         # Test images (optional)
+-- debug/              # Debug files (optional)
+-- README.md          # This document

## Usage Example
python
import json
from PIL import Image

# Load annotation data

with open('annotations/visual_trace_data.json', 'r') as f:
    data = json.load(f)

# View first sample

sample = data[0]
print(f"ID: {{sample['id']}}")
print(f"Trajectory point count: {{len(sample['answer_parsed']['data']['trajectory'])}}")

# Load image

if sample['image_path']:
    img = Image.open(sample['image_path'])
    print(f"Image size: {{img.size}}")

""")

    print(f"README created: {readme_path}")

def main():
    """Main function"""
    input_path = "data/public_robo_datasets/Embodied-R1-Eval/vabench_visual_trace.parquet"
    output_dir = "data/preprocess_data/embodied-r1-visual-trace"

    # Record start time
    start_time = datetime.datetime.now()

    # 1. Test image decoding first
    print("Step 1: Test image decoding")
    print("=" * 60)
    test_image_decoding(input_path, output_dir, sample_indices=[0, 1, 2])

    # 2. Parse entire dataset
    print(f"\n{'='*60}")
    print("Step 2: Parse entire dataset")
    print(f"{'='*60}")

    data = parse_visual_trace_dataset(input_path, output_dir)

    # 3. Display result examples
    print(f"\n{'='*60}")
    print("Step 3: Result examples")
    print(f"{'='*60}")

    if data and len(data) > 0:
        print(f"\nFirst 3 records:")
        for i, record in enumerate(data[:3]):
            print(f"\nRecord {i}:")
            print(f"  ID: {record['id']}")
            print(f"  Index: {record['idx']}")

            if record.get('problem'):
                problem = record['problem']
                if len(problem) > 100:
                    print(f"  Problem preview: {problem[:100]}...")
                else:
                    print(f"  Problem: {problem}")

            if record.get('image_path'):
                print(f"  Image: {Path(record['image_path']).name}")
                img_info = record.get('image_info', {})
                if img_info:
                    print(f"    Size: {img_info.get('width', '?')}x{img_info.get('height', '?')}")
                    print(f"    Format: {img_info.get('format', 'Unknown')}")
            else:
                print(f"  Image: None")

            if record.get('answer_parsed') and 'data' in record['answer_parsed']:
                ans_data = record['answer_parsed']['data']
                if 'trajectory' in ans_data:
                    trajectory = ans_data['trajectory']
                    print(f"  Trajectory point count: {len(trajectory)}")
                    if trajectory:
                        print(f"    First 3 points: {trajectory[:3]}")

    # 4. Create README
    print(f"\n{'='*60}")
    print("Step 4: Create documentation")
    print(f"{'='*60}")

    dataset_info = {
        'source_path': input_path,
        'parse_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(data) if data else 0,
        'saved_images': sum(1 for item in data if item.get('image_path')) if data else 0
    }

    create_readme(output_dir, dataset_info)

    # 5. Done
    end_time = datetime.datetime.now()
    elapsed = end_time - start_time

    print(f"\n{'='*60}")
    print(f"All done!")
    print(f"Total time: {elapsed.total_seconds():.1f} seconds")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
