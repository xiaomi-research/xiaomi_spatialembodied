# UPDATE: Replace placeholder paths with your actual paths.
import pandas as pd
import os
import base64
from PIL import Image
from io import BytesIO
import numpy as np

# Configure paths
file_path = "data/public_robo_datasets/Part-Affordance-2K/data/train-00000-of-00001.parquet"
output_base_dir = "data/preprocess_data/Part-Affordance-2K/"

# Create output directories
os.makedirs(os.path.join(output_base_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, "masks"), exist_ok=True)
os.makedirs(os.path.join(output_base_dir, "metadata"), exist_ok=True)

# Read parquet file
df = pd.read_parquet(file_path)
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Store metadata
metadata_records = []

# Process each row
for idx, row in df.iterrows():
    try:
        image_id = row['question_id']

        # Extract and save image
        if 'image' in row and row['image'] is not None:
            if isinstance(row['image'], bytes):
                # Direct bytes data
                img_data = row['image']
            elif isinstance(row['image'], dict) and 'bytes' in row['image']:
                # Bytes data contained in a dictionary
                img_data = row['image']['bytes']
            else:
                print(f"Row {idx}: Unrecognized image format")
                continue

            # Convert to PIL image and save
            try:
                image = Image.open(BytesIO(img_data))
                # Convert to RGB mode (ensure JPG compatibility)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                image_path = os.path.join(output_base_dir, "images", f"{image_id}.jpg")
                image.save(image_path, "JPEG", quality=95)
            except Exception as e:
                print(f"Row {idx}: Image processing error - {e}")
                continue

        # Extract and save mask
        if 'mask' in row and row['mask'] is not None:
            if isinstance(row['mask'], bytes):
                mask_data = row['mask']
            elif isinstance(row['mask'], dict) and 'bytes' in row['mask']:
                mask_data = row['mask']['bytes']
            else:
                print(f"Row {idx}: Unrecognized mask format")
                continue

            # Process mask data
            try:
                mask_image = Image.open(BytesIO(mask_data))
                mask_path = os.path.join(output_base_dir, "masks", f"{image_id}.png")
                # Masks are typically saved as PNG (supports transparency)
                mask_image.save(mask_path, "PNG")
            except Exception as e:
                print(f"Row {idx}: Mask processing error - {e}")
                continue

        # Save metadata (excluding binary data)
        metadata = {
            'question_id': image_id,
            'problem': row.get('problem', ''),
            'category_type': row.get('category_type', ''),
            'image_path': f"images/{image_id}.jpg",
            'mask_path': f"masks/{image_id}.png"
        }
        metadata_records.append(metadata)

        if idx % 100 == 0:
            print(f"Processed {idx+1}/{len(df)} records")

    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        continue

# Save metadata as JSON
import json
metadata_path = os.path.join(output_base_dir, "metadata", "image_metadata.json")
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata_records, f, indent=2, ensure_ascii=False)

print(f"\nProcessing complete!")
print(f"Images saved to: {os.path.join(output_base_dir, 'images')}")
print(f"Masks saved to: {os.path.join(output_base_dir, 'masks')}")
print(f"Metadata saved to: {metadata_path}")
print(f"Successfully processed {len(metadata_records)}/{len(df)} records")

# Display metadata for first few records
print("\nMetadata for first 3 records:")
for i, meta in enumerate(metadata_records[:3]):
    print(f"\nRecord {i+1}:")
    for key, value in meta.items():
        print(f"  {key}: {value}")
