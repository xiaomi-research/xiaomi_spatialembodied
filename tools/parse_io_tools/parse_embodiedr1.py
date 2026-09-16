# UPDATE: Replace placeholder paths with your actual paths.
import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import io
import cv2

def parse_embodied_r1_bytes(parquet_path, output_dir):
    """
    Parse Embodied-R1 dataset, specifically handling numpy arrays containing byte data
    """
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

    for idx, row in df.iterrows():
        data_id = str(row.get('id', f"sample_{idx}"))
        record_idx = int(row.get('idx', idx))

        # Progress display
        if (idx + 1) % 10 == 0:
            print(f"Processing: {idx + 1}/{len(df)}")

        # Process image data
        image_path = None
        image_info = None
        images_data = row.get('images')

        if isinstance(images_data, np.ndarray):
            try:
                # Case 1: numpy array containing byte data
                if images_data.shape == (1,) and images_data.dtype == np.object_:
                    # Extract byte data
                    byte_data = images_data.item(0)

                    if isinstance(byte_data, bytes):
                        # Create image from byte data
                        img = Image.open(io.BytesIO(byte_data))

                        # Save image
                        img_path = image_dir / f"{data_id}.jpg"

                        # Save based on format
                        if img.format in ['JPEG', 'PNG']:
                            img.save(img_path, format=img.format)
                        else:
                            img.save(img_path, format='JPEG')

                        image_path = str(img_path)

                        # Record image info
                        image_info = {
                            'format': img.format,
                            'mode': img.mode,
                            'width': img.width,
                            'height': img.height,
                            'bytes_size': len(byte_data),
                            'array_shape': images_data.shape,
                            'array_dtype': str(images_data.dtype)
                        }

                        print(f"  [OK] Saved image: {data_id}, format: {img.format}, size: {img.size}")
                    else:
                        print(f"  [FAIL] Non-byte data (ID: {data_id}): type: {type(byte_data)}")

                # Case 2: Direct image array
                elif len(images_data.shape) in [2, 3]:
                    # Process 2D or 3D array
                    img_array = images_data

                    if len(img_array.shape) == 3:  # (H, W, C)
                        if img_array.shape[2] == 3:  # RGB or BGR
                            if img_array.dtype == np.uint8:
                                # Assume BGR, convert to RGB
                                try:
                                    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                                    img = Image.fromarray(img_rgb)
                                except:
                                    img = Image.fromarray(img_array)
                            else:
                                img = Image.fromarray(img_array.astype(np.uint8))
                        elif img_array.shape[2] == 1:  # Single channel
                            img = Image.fromarray(img_array.squeeze(2).astype(np.uint8), mode='L')
                        else:
                            print(f"  Unsupported channel count: {img_array.shape[2]}")
                            continue

                    elif len(img_array.shape) == 2:  # Grayscale
                        img = Image.fromarray(img_array.astype(np.uint8), mode='L')

                    # Save image
                    img_path = image_dir / f"{data_id}.png"
                    img.save(img_path, format='PNG')
                    image_path = str(img_path)

                    # Record image info
                    image_info = {
                        'array_shape': img_array.shape,
                        'dtype': str(img_array.dtype),
                        'width': img.width,
                        'height': img.height,
                        'mode': img.mode
                    }

                else:
                    print(f"  Unsupported array shape (ID: {data_id}): {images_data.shape}")
                    continue

            except Exception as e:
                print(f"  Image processing failed (ID: {data_id}): {e}")
                import traceback
                traceback.print_exc()
        elif images_data is not None:
            print(f"  [WARN] Ignoring non-numpy data (ID: {data_id}): type: {type(images_data)}")

        # Parse answer field
        answer_info = {}
        answer_str = row.get('answer', '')

        if answer_str and isinstance(answer_str, str):
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
                    except json.JSONDecodeError:
                        # If JSON parsing fails, save raw data
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
    json_path = json_dir / "embodied_r1_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2, default=str)

    # Statistics
    saved_images = sum(1 for item in parsed_data if item.get('image_path') is not None)

    print(f"\n{'='*60}")
    print(f"Parsing complete!")
    print(f"Total records: {len(parsed_data)}")
    print(f"Successfully saved images: {saved_images}")
    print(f"Image save failures: {len(parsed_data) - saved_images}")
    print(f"Data saved to: {json_path}")
    print(f"Images saved to: {image_dir}")

    return parsed_data

def debug_images_data(parquet_path, num_samples=5):
    """
    Debug the images column data
    """
    df = pd.read_parquet(parquet_path)

    print("=== Debugging images column data ===")

    for idx in range(min(num_samples, len(df))):
        row = df.iloc[idx]
        data_id = row.get('id', f"sample_{idx}")
        images_data = row.get('images')

        print(f"\nSample {idx} (ID: {data_id}):")
        print(f"  Type: {type(images_data)}")

        if isinstance(images_data, np.ndarray):
            print(f"  Numpy array shape: {images_data.shape}")
            print(f"  Numpy dtype: {images_data.dtype}")

            if images_data.shape == (1,) and images_data.dtype == np.object_:
                byte_data = images_data.item(0)
                print(f"  Inner data type: {type(byte_data)}")

                if isinstance(byte_data, bytes):
                    print(f"  Byte data length: {len(byte_data)}")

                    # Check file header
                    if len(byte_data) >= 4:
                        hex_header = byte_data[:4].hex()
                        print(f"  File header (hex): {hex_header}")

                        # Common image format file headers
                        headers = {
                            'ffd8ffe0': 'JPEG/JFIF',
                            'ffd8ffe1': 'JPEG/Exif',
                            'ffd8ffe2': 'JPEG/Canon',
                            '89504e47': 'PNG',
                            '47494638': 'GIF',
                            '424d': 'BMP',
                        }

                        for header_hex, format_name in headers.items():
                            if hex_header.startswith(header_hex):
                                print(f"  Likely image format: {format_name}")
                                break

                        # Try to decode
                        try:
                            img = Image.open(io.BytesIO(byte_data))
                            print(f"  [OK] PIL can decode: {img.format}, size: {img.size}")
                        except Exception as e:
                            print(f"  [FAIL] PIL decode failed: {e}")

                # Show first few bytes
                if isinstance(byte_data, bytes) and len(byte_data) > 0:
                    print(f"  First 20 bytes: {byte_data[:20]}")
        else:
            print(f"  Value: {images_data}")

# Main program
if __name__ == "__main__":
    input_path = "data/public_robo_datasets/Embodied-R1-Eval/vabench_point.parquet"
    output_dir = "data/preprocess_data/embodied-r1"

    # 1. Debug data first
    print("Step 1: Debug data format")
    print("=" * 60)
    debug_images_data(input_path, num_samples=3)

    # 2. Parse dataset
    print(f"\n{'='*60}")
    print("Step 2: Parse dataset")
    print(f"{'='*60}")

    data = parse_embodied_r1_bytes(input_path, output_dir)

    # 3. Display results
    print(f"\n{'='*60}")
    print("Step 3: Result examples")
    print(f"{'='*60}")

    if data:
        print(f"\nFirst 3 records:")
        for i, record in enumerate(data[:3]):
            print(f"\nRecord {i}:")
            print(f"  ID: {record['id']}")
            print(f"  Index: {record['idx']}")
            print(f"  Dataset: {record['dataset_name']}")
            print(f"  Data type: {record['data_type']}")

            if record.get('image_path'):
                print(f"  Image path: {os.path.basename(record['image_path'])}")
                if record.get('image_info'):
                    info = record['image_info']
                    print(f"  Image info: {info.get('format', 'unknown')}, "
                          f"{info.get('width', '?')}x{info.get('height', '?')}")
            else:
                print(f"  Image: none")

            if record.get('answer_parsed'):
                ans = record['answer_parsed']
                if 'data' in ans and 'free_points' in ans['data']:
                    points = ans['data']['free_points']
                    print(f"  Point count: {len(points)}")
                elif 'data' in ans and 'bbox' in ans['data']:
                    bbox = ans['data']['bbox']
                    print(f"  Bounding box: {bbox}")
