# UPDATE: Replace placeholder paths with your actual paths.
#!/usr/bin/env python3

import argparse
import json
import os
import ast
from pathlib import Path
import random

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Convert ego3d-bench dataset to training JSONL format')
    parser.add_argument('--input', type=str, default="data/Ego3d-Bench/ego3d_bench_converted.json", 
                       help='Input JSON file path containing ego3d-bench data')
    parser.add_argument('--output_dir', type=str, default="data/preprocess_data/all_eval_data",
                       help='Output directory for JSONL files')
    parser.add_argument('--image_root', type=str, 
                       default='data/Ego3d-Bench/images',
                       help='Root directory for images')
    parser.add_argument('--system_prompt', type=str,
                       default='You are a professional vision-language assistant capable of accurately understanding multi-view images and answering related questions.',
                       help='System prompt for the conversation')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                       help='Train/validation split ratio (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for splitting')
    parser.add_argument('--dataset_name', type=str, default="ego3d-bench")
    
    return parser.parse_args()

def parse_images_string(images_str):
    """Parse images field string to dictionary"""
    try:
        # Try to parse as dictionary directly
        if isinstance(images_str, dict):
            return images_str
        
        # Handle string format dictionary
        images_str = images_str.strip()
        if images_str.startswith('{') and images_str.endswith('}'):
            return ast.literal_eval(images_str)
        else:
            # If format is incorrect, return empty dictionary
            return {}
    except Exception as e:
        print(f"Warning: Failed to parse images string: {e}")
        return {}

def get_image_paths(images_dict, source, image_root):
    """Get complete image paths based on data source"""
    image_paths = []
    
    for view, filename in images_dict.items():
        if filename and filename.strip():  # Skip empty filenames
            # Construct full path - all files are directly in image_root, no subdirectories
            full_path = os.path.join(image_root, filename)
                
            # Check if file exists
            if not os.path.exists(full_path):
                print(f"Warning: Image file not found: {full_path}")
                # Continue anyway, training pipeline will handle missing files
            
            image_paths.append(full_path)
    
    return image_paths

def convert_to_training_format(record, system_prompt, image_root, record_idx):
    """Convert single record to training format"""
    # Parse images field
    images_dict = parse_images_string(record.get('images', {}))
    
    # Get image paths
    image_paths = get_image_paths(images_dict, record.get('source', ''), image_root)
    
    # Build messages list
    messages = []
    
    # Add system prompt
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Build user message with image references
    user_content = record.get('question', '')
    
    # Add assistant message
    assistant_content = record.get('answer', '')
    
    # Create conversation
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_content})
    
    # Get record index/id
    record_id = record.get('idx', record_idx)
    
    # Build training sample
    training_sample = {
        "messages": messages,
        "id": record_id
    }
    
    # Add optional metadata
    if 'category' in record:
        training_sample["category"] = record['category']
    if 'source' in record:
        training_sample["source"] = record['source']
    if 'sub_category' in record:
        training_sample["sub_category"] = record['sub_category']
    
    # Add images if available
    if image_paths:
        training_sample["images"] = image_paths
    
    return training_sample

def split_dataset(samples, train_ratio, seed=42):
    """Split dataset into train and validation sets"""
    random.seed(seed)
    
    # Shuffle samples
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    # Calculate split index
    split_idx = int(len(shuffled) * train_ratio)
    
    train_set = shuffled[:split_idx]
    val_set = shuffled[split_idx:]
    
    return train_set, val_set

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, args.dataset_name), exist_ok=True)
    
    # Read input data
    print(f"📥 Reading input file: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract records
    if isinstance(data, dict) and 'records' in data:
        records = data['records']
    elif isinstance(data, list):
        records = data
    else:
        records = []
    
    print(f"✅ Successfully read {len(records)} records")
    
    # Convert all records
    all_samples = []
    failed_records = 0
    
    for i, record in enumerate(records):
        try:
            sample = convert_to_training_format(record, args.system_prompt, args.image_root, i)
            all_samples.append(sample)
            
            # Print first 3 samples as examples
            if i < 3:
                print(f"📄 Sample {i+1}:")
                print(json.dumps(sample, ensure_ascii=False, indent=2))
                print("-" * 50)
                
        except Exception as e:
            print(f"⚠️ Error processing record {i}: {e}")
            failed_records += 1
            continue
    
    print(f"🔄 Successfully converted {len(all_samples)} records, failed: {failed_records}")
    
    # Split into train and validation sets
    train_samples, val_samples = split_dataset(all_samples, args.train_ratio, args.seed)
    
    # Save train set
    train_output = os.path.join(args.output_dir, args.dataset_name, "ego3d_bench_train.jsonl")
    with open(train_output, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Save validation set
    val_output = os.path.join(args.output_dir, args.dataset_name, "ego3d_bench_val.jsonl")
    with open(val_output, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Save full dataset
    full_output = os.path.join(args.output_dir, args.dataset_name, "ego3d_bench_full.jsonl")
    with open(full_output, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Generate statistics
    categories = set()
    sources = set()
    samples_with_images = 0
    
    for sample in all_samples:
        if 'category' in sample:
            categories.add(sample['category'])
        if 'source' in sample:
            sources.add(sample['source'])
        if 'images' in sample and sample['images']:
            samples_with_images += 1
    
    stats = {
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_ratio": args.train_ratio,
        "categories": sorted(list(categories)),
        "sources": sorted(list(sources)),
        "samples_with_images": samples_with_images,
        "conversion_date": data.get('metadata', {}).get('conversion_date') if isinstance(data, dict) else None,
        "seed": args.seed
    }
    
    # Save statistics
    stats_output = os.path.join(args.output_dir, args.dataset_name, "conversion_stats.json")
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 Conversion completed!")
    print(f"📊 Statistics:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Train set: {stats['train_samples']}")
    print(f"   Validation set: {stats['val_samples']}")
    print(f"   Train ratio: {stats['train_ratio']}")
    print(f"   Categories: {', '.join(stats['categories'])}")
    print(f"   Data sources: {', '.join(stats['sources'])}")
    print(f"   Samples with images: {stats['samples_with_images']}")
    print(f"\n📁 Output files:")
    print(f"   Train set: {train_output}")
    print(f"   Validation set: {val_output}")
    print(f"   Full dataset: {full_output}")
    print(f"   Statistics: {stats_output}")

if __name__ == "__main__":
    main()