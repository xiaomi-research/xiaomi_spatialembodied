# UPDATE: Replace placeholder paths with your actual paths.
import json
import os
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Convert MapLMv2 dataset to MS-Swift format')

    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input JSON file (e.g., train_v2.json)')
    parser.add_argument('--output_dir', type=str,
                        default='data/preprocess_data/MapLMv2',
                        help='Output directory for converted data')
    parser.add_argument('--convert_ratio', type=float, default=1.0,
                        help='Ratio of data to convert (0.0-1.0)')
    parser.add_argument('--data_split', type=str, required=True,
                        choices=['train', 'val', 'test'],
                        help='Data split name (train/val/test)')
    parser.add_argument('--image_base_path', type=str,
                        default='data/maplm_v2/data/images/',
                        help='Base path for images')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')

    return parser.parse_args()

def get_view_description(img_paths: Dict[str, str], image_base_path: str) -> Tuple[str, List[str]]:
    """Generate view description and image list from image paths"""
    # Define view order and descriptions
    view_order = ['FRONT', 'BACK_LEFT', 'BACK_RIGHT', 'BEV']
    view_descriptions = {
        'FRONT': 'Front view',
        'BACK_LEFT': 'Left rear view',
        'BACK_RIGHT': 'Right rear view',
        'BEV': 'Bird\'s-eye view'
    }

    image_list = []
    view_descs = []
    # breakpoint()
    for view in view_order:
        if view in img_paths:
            # Build full image path
            img_path = os.path.join(image_base_path, img_paths[view])
            image_list.append(img_path)
            view_descs.append(view_descriptions[view])

    # Generate view description text
    if view_descs:
        if len(view_descs) == 1:
            view_text = f"{view_descs[0]}: <image>"
        else:
            view_text = ', '.join([f"{desc}: <image>" for desc in view_descs[:-1]])
            view_text += f", and {view_descs[-1]}: <image>"
    else:
        view_text = ""

    return view_text, image_list

def process_qa_item(question: str, answer, options: Optional[List] = None,
                   is_multichoice: bool = False, qa_type: str = None) -> Tuple[str, str]:
    """Process a single QA pair, return processed question and answer"""
    processed_question = question

    # Add options (if any)
    if options is not None and len(options) > 0:
        if is_multichoice:
            option_prompt = 'Please select the answer from the options below. You can choose multiple answers. Options: '
        else:
            option_prompt = 'Please select the answer from the options below. Options: '

        # Format options
        if qa_type == 'CAP':
            # CAP type is open-ended, no options needed
            pass
        else:
            option_str = '; '.join([f"{i+1}. {str(opt).replace('.', '')}" for i, opt in enumerate(options)])
            processed_question += ' ' + option_prompt + f'({option_str})'

    # Process answer
    if isinstance(answer, list):
        processed_answer = ', '.join([str(a).replace('.', '') for a in answer])
    else:
        processed_answer = str(answer).replace('.', '')

    return processed_question, processed_answer

def convert_to_ms_swift_format(args, original_data: Dict) -> List[Dict]:
    """Convert original data to MS-Swift format"""
    converted_samples = []
    # Define task types
    perception_keys = ['SCN', 'LAN', 'INT', 'CAP']
    behavior_keys = ['MOVE']

    # System prompt
    system_prompt = """You are monitoring an autonomous vehicle. Your input consists of multiple synchronized visual feeds providing different perspectives of the driving environment."""

    # Sampling
    sample_ids = list(original_data.keys())
    if args.convert_ratio < 1.0:
        random.seed(args.seed)
        sample_size = int(len(sample_ids) * args.convert_ratio)
        sample_ids = random.sample(sample_ids, sample_size)

    for sample_id in sample_ids:
        sample_data = original_data[sample_id]

        # breakpoint()
        # Get image paths and view description
        img_paths = sample_data.get('image_paths', {})
        view_text, image_list = get_view_description(img_paths, image_base_path=args.image_base_path)
        # breakpoint()
        if not view_text or not image_list:
            continue

        # Process perception QA
        perception_qa = sample_data.get('QA', {}).get('perception', {})
        for qa_key in perception_keys:
            if qa_key not in perception_qa:
                continue

            qa_item = perception_qa[qa_key]
            question = qa_item.get('question', '')
            answer = qa_item.get('answer', '')
            options = qa_item.get('option', None)

            if not question or not answer:
                continue

            # Process CAP type (special handling)
            if qa_key == 'CAP':
                question = 'Please provide a detailed description of the road in the current driving scenario.'

            # Process question and answer
            processed_question, processed_answer = process_qa_item(
                question, answer, options, is_multichoice=False, qa_type=qa_key
            )

            # Build complete user message
            user_content = f"{view_text}. {system_prompt} {processed_question}"
            # Build MS-Swift format
            swift_sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an autonomous driving assistant. Analyze the driving scenario from multiple perspectives and provide accurate answers."
                    },
                    {
                        "role": "user",
                        "content": user_content
                    },
                    {
                        "role": "assistant",
                        "content": processed_answer
                    }
                ],
                "images": image_list
                # "channel": "maplm_v2"
            }

            # breakpoint()
            if args.data_split != "test":
                swift_sample['channel'] = "maplm_v2"


            converted_samples.append(swift_sample)

        # Process behavior QA
        behavior_qa = sample_data.get('QA', {}).get('behavior', {})
        for qa_key in behavior_keys:
            if qa_key not in behavior_qa:
                continue

            qa_item = behavior_qa[qa_key]
            question = qa_item.get('question', '')
            answer = qa_item.get('answer', [])
            options = qa_item.get('option', None)

            if not question or not answer:
                continue

            # Process question and answer (MOVE is multiple choice)
            processed_question, processed_answer = process_qa_item(
                question, answer, options, is_multichoice=True, qa_type=qa_key
            )

            # Build complete user message
            user_content = f"{view_text}. {system_prompt} {processed_question}"

            # Build MS-Swift format
            swift_sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an autonomous driving assistant. Analyze the driving scenario from multiple perspectives and provide accurate answers."
                    },
                    {
                        "role": "user",
                        "content": user_content
                    },
                    {
                        "role": "assistant",
                        "content": processed_answer
                    }
                ],
                "images": image_list,
                "channel": "maplm_v2"
            }

            converted_samples.append(swift_sample)

    return converted_samples

def main():
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file path
    output_file = output_dir / f"maplm_v2_{args.data_split}_ms_swift.jsonl"

    # Read original data
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    print(f"Original data has {len(original_data)} samples")

    # Convert to MS-Swift format
    print("Converting to MS-Swift format...")
    converted_samples = convert_to_ms_swift_format(args, original_data)

    # Write output file
    print(f"Writing {len(converted_samples)} samples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Print statistics
    print(f"\nConversion completed successfully!")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {output_file}")
    print(f"Total samples converted: {len(converted_samples)}")
    print(f"Convert ratio: {args.convert_ratio}")
    print(f"Data split: {args.data_split}")

if __name__ == "__main__":
    main()
