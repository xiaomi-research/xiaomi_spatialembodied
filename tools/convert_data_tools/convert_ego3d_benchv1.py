# UPDATE: Replace placeholder paths with your actual paths.
#!/usr/bin/env python3

import argparse
import json
import os
import ast
import re
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
                       default='You are a professional vision-language assistant capable of accurately understanding multi-view images and answering related questions. Output format requirement:\nPlease provide your answer in the format: "Answer: the answer".',
                       help='System prompt for the conversation')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                       help='Train/validation split ratio (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for splitting')
    parser.add_argument('--dataset_name', type=str, default="ego3d-bench")
    parser.add_argument('--convert_ratio', type=float, default=1.0,
                       help='Ratio of data to convert (for debugging)')

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

def clean_option_text(option_text):
    """Clean and normalize option text"""
    option_text = str(option_text).strip()

    # Remove any leading/trailing quotes
    option_text = option_text.strip("'\"")

    # Ensure there's a space after the dot if missing
    # Handle patterns like "A.Less than 3 seconds" -> "A. Less than 3 seconds"
    match = re.match(r'^([A-Z])\.(\S)', option_text)
    if match and match.group(2):
        # Add space after the dot
        option_text = f"{match.group(1)}. {match.group(2)}" + option_text[3:]

    return option_text

def parse_options_string(options_str):
    """Parse options field string to list"""
    try:
        if isinstance(options_str, list):
            return [clean_option_text(opt) for opt in options_str if str(opt).strip()]

        if not options_str or not isinstance(options_str, str):
            return []

        options_str = options_str.strip()

        # Debug: show what we're parsing
        if len(options_str) < 100:  # Only for short strings
            print(f"  Parsing options string: '{options_str}'")

        # Remove newlines and extra spaces
        options_str = options_str.replace('\n', ' ').replace('\r', ' ')
        options_str = re.sub(r'\s+', ' ', options_str)

        # Case 1: String is in list format like "['A. yes' 'B. no']" or "['A.Less than 3 seconds' 'B.3-5 seconds' 'C.5-10 seconds' 'D.More than 10 seconds']"
        if options_str.startswith('[') and options_str.endswith(']'):
            inner_str = options_str[1:-1].strip()

            # If empty list, return empty
            if not inner_str:
                return []

            # Try to parse as Python list with ast
            try:
                # Handle cases with missing commas
                inner_str = re.sub(r"'\s+'", "','", inner_str)  # Replace "' '" with "','"
                inner_str = re.sub(r'"\s+"', '","', inner_str)  # Replace '" "' with '","'

                # Try to parse as a Python list
                try:
                    options_list = ast.literal_eval(f'[{inner_str}]')
                except:
                    # If that fails, use regex to extract quoted strings
                    pattern = r"['\"]([^'\"]+)['\"]"
                    matches = re.findall(pattern, inner_str)
                    if matches:
                        options_list = matches
                    else:
                        # Try to split by spaces that are followed by [A-Z].
                        pattern = r'\s+(?=[A-Z]\.)'
                        parts = re.split(pattern, inner_str)
                        options_list = [part.strip() for part in parts if part.strip()]
            except Exception as e:
                print(f"Warning: Failed to parse list format: {e}")
                options_list = []

        # Case 2: String is not in list format, just concatenated options
        else:
            # Try to split by uppercase letter followed by dot
            pattern = r'(?=[A-Z]\.)'
            parts = re.split(pattern, options_str)
            options_list = [part.strip() for part in parts if part.strip() and len(part.strip()) > 2]

            # If we only got one part, it might be because there's no space between dot and text
            if len(options_list) == 1 and re.match(r'^[A-Z]\.[A-Z]', options_list[0]):
                # Handle cases like "A.Less than 3 secondsB.3-5 seconds"
                # Try a different pattern
                pattern2 = r'([A-Z]\.[^A-Z]+)'
                matches = re.findall(pattern2, options_str)
                if matches:
                    options_list = matches

        # Clean each option
        options_list = [clean_option_text(opt) for opt in options_list if str(opt).strip()]

        # Debug output
        if len(options_list) > 0 and len(options_str) < 100:
            print(f"  Parsed {len(options_list)} options: {options_list}")

        return options_list
    except Exception as e:
        print(f"Warning: Failed to parse options string: {e}")
        print(f"  Options string was: {options_str[:100] if options_str else 'None'}")
        return []

def is_single_letter_answer(answer):
    """Check if answer is a single letter (A, B, C, etc.)"""
    if not answer or not isinstance(answer, str):
        return False

    answer = answer.strip()
    # Check if answer is a single uppercase letter
    return len(answer) == 1 and answer.isalpha() and answer.isupper()

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

def format_options_for_prompt(options_list):
    """Format options list for inclusion in prompt"""
    if not options_list:
        return ""

    # Ensure all options are properly formatted
    formatted_options = []
    for i, option in enumerate(options_list):
        option = str(option).strip()

        # Add the letter if missing (A, B, C, etc.)
        if not re.match(r'^[A-Z]\.', option):
            # Try to extract the letter from the option text
            match = re.match(r'^([A-Z])\s*\.?\s*(.*)', option)
            if match:
                letter = match.group(1)
                text = match.group(2)
                option = f"{letter}. {text}"
            else:
                # Just add the letter based on position
                letter = chr(65 + i)  # 65 is 'A' in ASCII
                option = f"{letter}. {option}"
        else:
            # Ensure there's a space after the dot
            match = re.match(r'^([A-Z])\.(\S)', option)
            if match and match.group(2):
                option = f"{match.group(1)}. {match.group(2)}" + option[3:]

        formatted_options.append(option)

    # Join with semicolon and space
    options_text = "; ".join(formatted_options)
    return f"\nOptions: {options_text}\n"

def convert_to_training_format(record, system_prompt, image_root, record_idx):
    """Convert single record to training format"""
    # Parse images field
    images_dict = parse_images_string(record.get('images', {}))

    # Get image paths
    image_paths = get_image_paths(images_dict, record.get('source', ''), image_root)

    # Parse options
    options_str = record.get('options', '')
    options_list = parse_options_string(options_str)

    # Check if we have options and if answer is a single letter
    answer = record.get('answer', '').strip()
    has_options = len(options_list) > 0
    answer_is_single_letter = is_single_letter_answer(answer)

    # Debug output for first few records
    if record_idx < 5:
        print(f"Debug - Record {record_idx}:")
        print(f"  Original options string: {repr(options_str)}")
        print(f"  Parsed options list: {options_list}")
        print(f"  Answer: {answer}")
        print(f"  Has options: {has_options}")
        print(f"  Answer is single letter: {answer_is_single_letter}")

    # Build messages list
    messages = []

    # Add system prompt
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Build user message with image references
    user_content = record.get('question', '')

    # Add options to user content if available and answer is a single letter
    if has_options and answer_is_single_letter:
        options_text = format_options_for_prompt(options_list)
        if options_text:
            user_content += options_text

    # Add assistant message with formatted answer
    # Format answer as "Answer: the answer"
    original_answer = answer
    # if original_answer and not original_answer.startswith("Answer: "):
    #     assistant_content = f"Answer: {original_answer}"
    # else:
    assistant_content = original_answer

    # Create conversation
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_content})

    # Get record index/id
    record_id = record.get('idx', str(record_idx))

    # Build training sample
    training_sample = {
        "messages": messages,
        "id": record_id
    }

    # Add all original metadata
    for key, value in record.items():
        if key not in ['images', 'question', 'answer', 'options']:
            training_sample[key] = value

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
    output_path = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(output_path, exist_ok=True)

    # Read input data
    print(f"Reading input file: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract records
    if isinstance(data, dict) and 'records' in data:
        records = data['records']
    elif isinstance(data, list):
        records = data
    else:
        records = []

    print(f"Successfully read {len(records)} records")

    # Apply convert_ratio for debugging
    if args.convert_ratio < 1.0:
        convert_count = int(len(records) * args.convert_ratio)
        records = records[:convert_count]
        print(f"Converting {convert_count} records ({args.convert_ratio*100}% of total)")

    # Convert all records
    all_samples = []
    failed_records = 0
    options_added_count = 0

    for i, record in enumerate(records):
        try:
            sample = convert_to_training_format(record, args.system_prompt, args.image_root, i)
            all_samples.append(sample)

            # Check if options were added
            if 'options' in record and record['options']:
                options_list = parse_options_string(record['options'])
                answer = record.get('answer', '').strip()
                if options_list and is_single_letter_answer(answer):
                    options_added_count += 1

            # Print first 3 samples as examples
            if i < 3:
                print(f"\nSample {i+1}:")
                print(f"  Original question: {record.get('question', '')[:100]}...")
                print(f"  Original options: {repr(record.get('options', ''))}")
                parsed_options = parse_options_string(record.get('options', ''))
                print(f"  Parsed options: {parsed_options}")
                original_answer = record.get('answer', '')
                formatted_answer = f"Answer: {original_answer}" if original_answer and not original_answer.startswith("Answer: ") else original_answer
                print(f"  Original answer: {original_answer}")
                print(f"  Formatted answer: {formatted_answer}")
                if 'options' in record and record['options'] and parsed_options:
                    formatted = format_options_for_prompt(parsed_options)
                    print(f"  Formatted options: {formatted}")
                print("-" * 50)

        except Exception as e:
            print(f"Warning: Error processing record {i}: {e}")
            import traceback
            traceback.print_exc()
            failed_records += 1
            continue

    print(f"Successfully converted {len(all_samples)} records, failed: {failed_records}")
    print(f"Added options to {options_added_count} records with single-letter answers")

    # Test parsing for a few example options strings
    test_cases = [
        "['A. yes' 'B. no']",
        "A. yesB. no",
        "A.Less than 3 secondsB.3-5 secondsC.5-10 secondsD.More than 10 seconds",
        "['A. Less than 3 seconds', 'B. 3-5 seconds', 'C. 5-10 seconds', 'D. More than 10 seconds']",
        "['A.Less than 3 seconds' 'B.3-5 seconds' 'C.5-10 seconds'\n 'D.More than 10 seconds']"
    ]

    print("\nTesting options parsing:")
    for i, test_str in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"  Input: {repr(test_str)}")
        parsed = parse_options_string(test_str)
        print(f"  Parsed: {parsed}")
        formatted = format_options_for_prompt(parsed)
        print(f"  Formatted: {formatted}")

    # Split into train and validation sets
    train_samples, val_samples = split_dataset(all_samples, args.train_ratio, args.seed)

    # Save train set
    train_output = os.path.join(output_path, "ego3d_bench_train.jsonl")
    with open(train_output, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Save validation set
    val_output = os.path.join(output_path, "ego3d_bench_val.jsonl")
    with open(val_output, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Save full dataset
    full_output = os.path.join(output_path, "ego3d_bench_full.jsonl")
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
        "convert_ratio": args.convert_ratio,
        "categories": sorted(list(categories)),
        "sources": sorted(list(sources)),
        "samples_with_images": samples_with_images,
        "samples_with_options_added": options_added_count,
        "conversion_date": data.get('metadata', {}).get('conversion_date') if isinstance(data, dict) else None,
        "seed": args.seed
    }

    # Save statistics
    stats_output = os.path.join(output_path, "conversion_stats.json")
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nConversion completed!")
    print(f"Statistics:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Train set: {stats['train_samples']}")
    print(f"   Validation set: {stats['val_samples']}")
    print(f"   Train ratio: {stats['train_ratio']}")
    print(f"   Convert ratio: {stats['convert_ratio']}")
    print(f"   Categories: {', '.join(stats['categories'])}")
    print(f"   Data sources: {', '.join(stats['sources'])}")
    print(f"   Samples with images: {stats['samples_with_images']}")
    print(f"   Samples with options added: {stats['samples_with_options_added']}")
    print(f"\nOutput files:")
    print(f"   Train set: {train_output}")
    print(f"   Validation set: {val_output}")
    print(f"   Full dataset: {full_output}")
    print(f"   Statistics: {stats_output}")

if __name__ == "__main__":
    main()
