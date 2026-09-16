# UPDATE: Replace placeholder paths with your actual paths.
import pandas as pd
import os
import json
from PIL import Image
from io import BytesIO
import numpy as np

# --------------------------------
# 1. Configure paths
# --------------------------------

# Input: directory containing two parquet files
input_dir = "data/public_robo_datasets/roboreft"

# Define two input files
parquet_files = [
    "test-00000-of-00002.parquet",
    "test-00001-of-00002.parquet",
]

# Output directory
output_base_dir = "data/preprocess_data/roborefit-benchmark-dataset"

# Create output subdirectories
images_dir = os.path.join(output_base_dir, "images")
conversations_dir = os.path.join(output_base_dir, "conversations")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(conversations_dir, exist_ok=True)

# List for storing all metadata
all_metadata_records = []

# Global image counter for generating unique IDs
global_image_counter = 0

# Track failed rows
failed_rows = []

# --------------------------------
# Helper function: process image data
# --------------------------------
def process_image_data(img_data, image_id, parquet_file, row_idx):
    """Process image data in various formats"""
    try:
        if isinstance(img_data, np.ndarray):
            # Process numpy array
            if img_data.shape == (1,):
                # Array with shape (1,), take the first element
                element = img_data[0]

                if isinstance(element, dict) and 'bytes' in element:
                    # Dictionary containing bytes data
                    img_bytes = element['bytes']
                    image = Image.open(BytesIO(img_bytes))
                elif isinstance(element, bytes):
                    # Direct bytes data
                    image = Image.open(BytesIO(element))
                else:
                    print(f"File {parquet_file} row {row_idx}: Unsupported array element type: {type(element)}")
                    return False
            else:
                print(f"File {parquet_file} row {row_idx}: Unsupported numpy array shape: {img_data.shape}")
                return False
        elif isinstance(img_data, bytes):
            # Process bytes data
            image = Image.open(BytesIO(img_data))
        elif isinstance(img_data, dict) and 'bytes' in img_data:
            # Process dictionary containing bytes
            image = Image.open(BytesIO(img_data['bytes']))
        else:
            print(f"File {parquet_file} row {row_idx}: Unsupported image format - type: {type(img_data)}")
            return False

        # Convert to RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Save image
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        image.save(image_path, "JPEG", quality=95)
        return True

    except Exception as e:
        print(f"File {parquet_file} row {row_idx}: Image processing error - {e}")
        return False

# --------------------------------
# Helper function: process conversation data
# --------------------------------
def process_conversation_data(conv_data, image_id, parquet_file, row_idx):
    """Process conversation data"""
    try:
        conv_path = os.path.join(conversations_dir, f"{image_id}.json")

        # Handle different types of conversation data
        if isinstance(conv_data, np.ndarray):
            # Numpy array, convert to list
            conv_list = conv_data.tolist()
            with open(conv_path, 'w', encoding='utf-8') as f:
                json.dump(conv_list, f, indent=2, ensure_ascii=False)
        elif isinstance(conv_data, list):
            # Save list directly
            with open(conv_path, 'w', encoding='utf-8') as f:
                json.dump(conv_data, f, indent=2, ensure_ascii=False)
        elif isinstance(conv_data, dict):
            # Save dictionary directly
            with open(conv_path, 'w', encoding='utf-8') as f:
                json.dump(conv_data, f, indent=2, ensure_ascii=False)
        elif isinstance(conv_data, str):
            try:
                # Try to parse as JSON
                parsed_data = json.loads(conv_data)
                with open(conv_path, 'w', encoding='utf-8') as f:
                    json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                # If not JSON, save as text
                with open(conv_path, 'w', encoding='utf-8') as f:
                    json.dump({"text": conv_data}, f, indent=2, ensure_ascii=False)
        else:
            # Convert other types to string
            with open(conv_path, 'w', encoding='utf-8') as f:
                json.dump({"text": str(conv_data)}, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"File {parquet_file} row {row_idx}: Conversation save error - {e}")
        return False

# --------------------------------
# 2. Process each Parquet file
# --------------------------------

# Process each file in the list
for file_idx, parquet_file in enumerate(parquet_files):
    file_path = os.path.join(input_dir, parquet_file)
    print(f"\n{'='*60}")
    print(f"Processing file: {file_path}")
    print(f"{'='*60}")

    # Read parquet file
    df = pd.read_parquet(file_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Show first row data example to help understand data structure
    if len(df) > 0:
        first_row = df.iloc[0]
        print(f"\nFirst row data example:")
        print(f"conversations: {first_row['conversations']}")
        print(f"images type: {type(first_row['images'])}")
        if isinstance(first_row['images'], np.ndarray):
            print(f"images shape: {first_row['images'].shape}")
            if first_row['images'].shape[0] > 0:
                first_element = first_row['images'][0]
                print(f"images[0] type: {type(first_element)}")
                if isinstance(first_element, dict):
                    print(f"images[0] has 'bytes' key: {'bytes' in first_element}")

    # Process each row
    for idx, row in df.iterrows():
        try:
            # Generate unique image ID
            image_id = f"roborefit_{global_image_counter:08d}"

            has_image = False
            has_conversation = False

            # Extract and save image
            if 'images' in row and row['images'] is not None:
                img_data = row['images']
                has_image = process_image_data(img_data, image_id, parquet_file, idx)

            # Save conversation content
            if 'conversations' in row and row['conversations'] is not None:
                conv_data = row['conversations']
                has_conversation = process_conversation_data(conv_data, image_id, parquet_file, idx)

            # Save metadata
            metadata = {
                'id': image_id,
                'source_file': parquet_file,
                'file_index': file_idx,
                'row_index': idx,
                'has_image': has_image,
                'has_conversation': has_conversation,
                'image_path': f"images/{image_id}.jpg" if has_image else None,
                'conversation_path': f"conversations/{image_id}.json" if has_conversation else None
            }

            all_metadata_records.append(metadata)
            global_image_counter += 1

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx+1}/{len(df)} records")

        except Exception as e:
            error_msg = f"Error processing file {parquet_file} row {idx}: {e}"
            print(error_msg)
            failed_rows.append({
                'file': parquet_file,
                'row': idx,
                'error': str(e)
            })
            continue

    print(f"File {parquet_file} processing complete, processed {len(df)} records")

# --------------------------------
# 3. Save all metadata and statistics
# --------------------------------

# Save main metadata
metadata_path = os.path.join(output_base_dir, "metadata.json")
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(all_metadata_records, f, indent=2, ensure_ascii=False)

# Save failed records
if failed_rows:
    failed_path = os.path.join(output_base_dir, "failed_rows.json")
    with open(failed_path, 'w', encoding='utf-8') as f:
        json.dump(failed_rows, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"All processing complete!")
print(f"{'='*60}")
print(f"Images saved to: {images_dir}")
print(f"Conversations saved to: {conversations_dir}")
print(f"Metadata saved to: {metadata_path}")

if failed_rows:
    print(f"Failed records saved to: {failed_path}")

print(f"\nTotal processed {len(all_metadata_records)} records")

# Statistics
images_count = sum(1 for meta in all_metadata_records if meta['has_image'])
conversations_count = sum(1 for meta in all_metadata_records if meta['has_conversation'])
print(f"Successfully extracted images: {images_count}")
print(f"Successfully extracted conversations: {conversations_count}")

# If there are failed records
if failed_rows:
    print(f"\nFailed record count: {len(failed_rows)}")
    print("First 5 failed records:")
    for i, fail in enumerate(failed_rows[:5]):
        print(f"  {i+1}. File: {fail['file']}, Row: {fail['row']}, Error: {fail['error']}")

# Show first few records' metadata
print(f"\n{'='*60}")
print("First 3 records metadata:")
print(f"{'='*60}")
for i, meta in enumerate(all_metadata_records[:3]):
    print(f"\nRecord {i+1}:")
    for key, value in meta.items():
        if key not in ['image_path', 'conversation_path'] or value is not None:
            print(f"  {key}: {value}")

print(f"\n{'='*60}")
print("Processing complete!")
print(f"{'='*60}")
