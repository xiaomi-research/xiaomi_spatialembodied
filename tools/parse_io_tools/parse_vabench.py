# UPDATE: Replace placeholder paths with your actual paths.
import pandas as pd
import json
from pathlib import Path

# Quick parse script
def quick_parse(parquet_path, output_dir):
    # Read data
    df = pd.read_parquet(parquet_path)
    print(f"Read {len(df)} records")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)

    parsed_data = []

    for idx, row in df.iterrows():
        record_idx = row.get('idx', idx)

        # Process image
        image_obj = row.get('image', {})
        image_bytes = None
        image_path = None

        if isinstance(image_obj, dict) and 'bytes' in image_obj:
            image_bytes = image_obj['bytes']
        elif isinstance(image_obj, bytes):
            image_bytes = image_obj

        if image_bytes:
            img_path = image_dir / f"image_{record_idx:05d}.png"
            with open(img_path, 'wb') as f:
                f.write(image_bytes)
            image_path = str(img_path)

        # Parse bounding box
        bbox = row.get('bbox')
        normalized_bbox = row.get('normalized_bbox')

        if hasattr(bbox, 'tolist'):
            bbox = bbox.tolist()
        if hasattr(normalized_bbox, 'tolist'):
            normalized_bbox = normalized_bbox.tolist()

        # Build record
        record = {
            "id": int(record_idx),
            "problem": str(row.get('problem', '')),
            "image_path": image_path,
            "bbox": bbox,
            "normalized_bbox": normalized_bbox
        }

        parsed_data.append(record)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} records")

    # Save JSON
    json_path = output_dir / "data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)

    print(f"\nData saved to: {json_path}")
    print(f"Images saved to: {image_dir}")

    return parsed_data

# Usage example
if __name__ == "__main__":
    # Replace with your path
    data = quick_parse(
        parquet_path="data/public_robo_datasets/vabench-point-bbox/data/test-00000-of-00001.parquet",
        output_dir="data/preprocess_data/vabench-point-bbox"
    )

    # Print first 3 records info
    print("\nFirst 3 records:")
    for i, record in enumerate(data[:3]):
        print(f"\nRecord {i}:")
        print(f"  ID: {record['id']}")
        print(f"  Problem preview: {record['problem'][:100]}...")
        print(f"  Image path: {record['image_path']}")
        print(f"  BBox: {record['bbox']}")
        print(f"  Normalized BBox: {record['normalized_bbox']}")
