# UPDATE: Replace placeholder paths with your actual paths.
import json
import os

def convert_lingoqa_val_to_jsonl(input_file, output_dir):
    """
    Convert LingoQA validation set data to target JSONL format

    Args:
        input_file: Input JSON file path
        output_dir: Output directory path
    """

    # Read input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set output file path
    output_file = os.path.join(output_dir, 'lingoqa_swift_val.jsonl')

    system_prompt = 'You are a driver. These are five continuous front view images. Answer the following question: '
    # Convert data and write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in data:
            # Build converted data format
            converted_item = {
                "messages": [
                    {
                        "role": "user",
                        # "content": "<image>" * len(item["images"]) + " " + item["question"]
                        "content": '<image>' * (len(item['images'])) + system_prompt + item['question']
                    },
                    {
                        "role": "assistant",
                        "content": item["answer"]
                    }
                ],
                "images": item["images"]
            }

            # Write to JSONL file (one JSON object per line)
            f_out.write(json.dumps(converted_item, ensure_ascii=False) + '\n')

    print(f"Data conversion complete! Processed {len(data)} records")
    print(f"Output file: {output_file}")

# Main execution
if __name__ == "__main__":
    # Input file path
    input_file = "data/lingoqa_val.json"

    dataset_name = 'lingoqa'
    # Output directory path
    output_dir = "data/preprocess_data/all_eval_data"
    output_dir = os.path.join(output_dir, dataset_name)

    # Execute conversion
    convert_lingoqa_val_to_jsonl(input_file, output_dir)
