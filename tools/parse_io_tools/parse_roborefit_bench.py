# UPDATE: Replace placeholder paths with your actual paths.
import pandas as pd
import os
import json
from PIL import Image
from io import BytesIO
import numpy as np
import base64

# --------------------------------
# 1. Configure paths
# --------------------------------

# Input: directory containing parquet files
input_dir = "data/public_robo_datasets/roboreft"

# Define input files
parquet_files = [
    "test-00000-of-00002.parquet",
    "test-00001-of-00002.parquet",
]

# Output directory
output_base_dir = "data/preprocess_data/roborefit-benchmark-dataset"

# Create output subdirectories
images_dir = os.path.join(output_base_dir, "images")
annotations_dir = os.path.join(output_base_dir, "annotations")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# List for storing all metadata
all_metadata_records = []

# Global counter
global_counter = 0

# Track failed rows
failed_rows = []

# --------------------------------
# Helper function: process image data
# --------------------------------
def process_image_data(img_data, image_id, parquet_file, row_idx):
    """Process image data from the roborefit dataset"""
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
        elif isinstance(img_data, dict) and 'bytes' in img_data:
            # Process dictionary containing bytes
            image = Image.open(BytesIO(img_data['bytes']))
        elif isinstance(img_data, bytes):
            # Process bytes data
            image = Image.open(BytesIO(img_data))
        elif isinstance(img_data, str):
            # Possibly base64 encoded string
            try:
                img_bytes = base64.b64decode(img_data)
                image = Image.open(BytesIO(img_bytes))
            except:
                # If not base64, try opening directly
                try:
                    image = Image.open(BytesIO(img_data.encode()))
                except:
                    print(f"File {parquet_file} row {row_idx}: Cannot decode image data")
                    return False
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
# Helper function: process bounding box
# --------------------------------
def process_bbox_data(row, image_id, parquet_file, row_idx):
    """Process annotation data from the roborefit dataset"""
    try:
        # Get ref_exp
        ref_exp = row.get('ref_exp', '')

        # Get bounding box
        bbox = row.get('bbox', None)
        normalized_bbox = row.get('normalized_bbox', None)

        # Process bbox data types
        if bbox is not None and hasattr(bbox, 'tolist'):
            bbox = bbox.tolist()
        elif bbox is not None and isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()

        if normalized_bbox is not None and hasattr(normalized_bbox, 'tolist'):
            normalized_bbox = normalized_bbox.tolist()
        elif normalized_bbox is not None and isinstance(normalized_bbox, np.ndarray):
            normalized_bbox = normalized_bbox.tolist()

        # Create annotation data
        annotation = {
            'id': row.get('id', image_id),
            'ref_exp': ref_exp,
            'bbox': bbox,
            'normalized_bbox': normalized_bbox
        }

        # Save annotation
        annotation_path = os.path.join(annotations_dir, f"{image_id}.json")
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"File {parquet_file} row {row_idx}: Annotation processing error - {e}")
        return False

# --------------------------------
# Helper function: generate conversation format
# --------------------------------
def generate_conversation_format(ref_exp, image_id, normalized_bbox=None):
    """Convert referring expression to conversation format"""
    if normalized_bbox is not None:
        bbox_text = f"Normalized bbox: {normalized_bbox}"
    else:
        bbox_text = "No bbox information available"

    conversation = [
        {
            "from": "human",
            "value": f"Please locate the object described: {ref_exp}"
        },
        {
            "from": "gpt",
            "value": f"The object described as '{ref_exp}' is located at {bbox_text}"
        }
    ]

    return conversation

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
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Failed to read file: {e}")
        continue

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Show first few rows as examples
    if len(df) > 0:
        print(f"\nFirst 3 rows example:")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            print(f"\nRow {i}:")
            print(f"  id: {row.get('id', 'N/A')}")
            print(f"  image type: {type(row.get('image', 'N/A'))}")

            # Check image data structure
            img_data = row.get('image', None)
            if img_data is not None:
                if isinstance(img_data, np.ndarray):
                    print(f"  image shape: {img_data.shape}")
                    if img_data.shape[0] > 0:
                        first_element = img_data[0]
                        print(f"  image[0] type: {type(first_element)}")
                        if isinstance(first_element, dict):
                            print(f"  image[0] has 'bytes' key: {'bytes' in first_element}")
                elif isinstance(img_data, dict):
                    print(f"  image dict has 'bytes' key: {'bytes' in img_data}")

            print(f"  ref_exp: {row.get('ref_exp', 'N/A')[:50]}...")
            print(f"  bbox: {row.get('bbox', 'N/A')}")
            print(f"  normalized_bbox: {row.get('normalized_bbox', 'N/A')}")

    # Process each row
    for idx, row in df.iterrows():
        try:
            # Generate unique ID
            sample_id = f"roborefit_{global_counter:08d}"

            has_image = False
            has_annotation = False

            # Extract and save image
            if 'image' in row and row['image'] is not None:
                img_data = row['image']
                has_image = process_image_data(img_data, sample_id, parquet_file, idx)

            # Save annotation info
            has_annotation = process_bbox_data(row, sample_id, parquet_file, idx)

            # Create conversation format (optional)
            if 'ref_exp' in row and row['ref_exp'] is not None:
                normalized_bbox = row.get('normalized_bbox', None)
                if hasattr(normalized_bbox, 'tolist'):
                    normalized_bbox = normalized_bbox.tolist()
                elif isinstance(normalized_bbox, np.ndarray):
                    normalized_bbox = normalized_bbox.tolist()

                conversation = generate_conversation_format(row['ref_exp'], sample_id, normalized_bbox)
                conversation_path = os.path.join(annotations_dir, f"{sample_id}_conversation.json")
                with open(conversation_path, 'w', encoding='utf-8') as f:
                    json.dump(conversation, f, indent=2, ensure_ascii=False)

            # Save metadata
            metadata = {
                'id': sample_id,
                'original_id': int(row.get('id', 0)) if hasattr(row.get('id'), '__int__') else row.get('id'),
                'source_file': parquet_file,
                'file_index': file_idx,
                'row_index': int(idx),
                'has_image': has_image,
                'has_annotation': has_annotation,
                'image_path': f"images/{sample_id}.jpg" if has_image else None,
                'annotation_path': f"annotations/{sample_id}.json" if has_annotation else None,
                'conversation_path': f"annotations/{sample_id}_conversation.json" if 'ref_exp' in row else None,
                'ref_exp': row.get('ref_exp', ''),
                'bbox': row.get('bbox', None),
                'normalized_bbox': row.get('normalized_bbox', None)
            }

            # Handle special data types
            for key in ['bbox', 'normalized_bbox']:
                if key in metadata and metadata[key] is not None:
                    if hasattr(metadata[key], 'tolist'):
                        metadata[key] = metadata[key].tolist()
                    elif isinstance(metadata[key], np.ndarray):
                        metadata[key] = metadata[key].tolist()
                    elif isinstance(metadata[key], np.generic):
                        metadata[key] = metadata[key].item()

            all_metadata_records.append(metadata)
            global_counter += 1

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx+1}/{len(df)} records")

        except Exception as e:
            error_msg = f"Error processing file {parquet_file} row {idx}: {e}"
            print(error_msg)
            failed_rows.append({
                'file': parquet_file,
                'row': int(idx),
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

# Save statistics
stats = {
    "total_samples": len(all_metadata_records),
    "samples_with_image": sum(1 for meta in all_metadata_records if meta['has_image']),
    "samples_with_annotation": sum(1 for meta in all_metadata_records if meta['has_annotation']),
    "failed_samples": len(failed_rows),
    "processed_files": len(parquet_files),
    "output_structure": {
        "images_dir": "images/",
        "annotations_dir": "annotations/",
        "metadata_file": "metadata.json"
    }
}

stats_path = os.path.join(output_base_dir, "dataset_statistics.json")
with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"All processing complete!")
print(f"{'='*60}")
print(f"Images saved to: {images_dir}")
print(f"Annotations saved to: {annotations_dir}")
print(f"Metadata saved to: {metadata_path}")
print(f"Statistics saved to: {stats_path}")

if failed_rows:
    print(f"Failed records saved to: {failed_path}")

print(f"\nTotal processed {len(all_metadata_records)} records")
print(f"Successfully extracted images: {stats['samples_with_image']}")
print(f"Successfully extracted annotations: {stats['samples_with_annotation']}")

# Show dataset structure
print(f"\n{'='*60}")
print("Dataset structure:")
print(f"{'='*60}")
print(f"Each sample contains:")
print(f"  1. Image file: images/{{sample_id}}.jpg")
print(f"  2. Annotation file: annotations/{{sample_id}}.json (contains ref_exp, bbox, normalized_bbox)")
print(f"  3. Conversation file: annotations/{{sample_id}}_conversation.json (optional, contains VLM format conversation)")

# Show first few records' metadata
print(f"\n{'='*60}")
print("First 3 records metadata:")
print(f"{'='*60}")
for i, meta in enumerate(all_metadata_records[:3]):
    print(f"\nRecord {i+1}:")
    print(f"  id: {meta.get('id', 'N/A')}")
    print(f"  ref_exp: {meta.get('ref_exp', 'N/A')[:50]}...")
    print(f"  bbox: {meta.get('bbox', 'N/A')}")
    print(f"  normalized_bbox: {meta.get('normalized_bbox', 'N/A')}")
    print(f"  has_image: {meta.get('has_image', False)}")
    print(f"  has_annotation: {meta.get('has_annotation', False)}")

print(f"\n{'='*60}")
print("Processing complete!")
print(f"{'='*60}")
