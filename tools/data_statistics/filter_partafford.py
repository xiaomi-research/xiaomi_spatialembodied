# UPDATE: Replace placeholder paths with your actual paths.
import json
import os

def convert_bbox_floats_to_ints(input_file_path, output_file_path=None):
    """
    Convert floats in the original_bbox field of a JSONL file to integers

    Args:
        input_file_path: Input JSONL file path
        output_file_path: Output file path, if None then overwrite the original file
    """
    # If no output path specified, overwrite the original file
    if output_file_path is None:
        output_file_path = input_file_path

    processed_lines = []

    # Read and process each line
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse JSON line
                    data = json.loads(line.strip())

                    # Check and process original_bbox field
                    if 'original_bbox' in data['meta_data']:
                        # Convert each float in the list to integer
                        # data['original_bbox'] = [int(num) for num in data['original_bbox']]
                        data['meta_data'].pop('original_bbox')

                    # Convert processed data back to string
                    processed_lines.append(json.dumps(data, ensure_ascii=False))

                except json.JSONDecodeError as e:
                    print(f"JSON parse error at line {line_num}: {e}")
                    # Keep original line
                    processed_lines.append(line.strip())
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    # Keep original line
                    processed_lines.append(line.strip())

        # Write processed content
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines))

        print(f"Processing complete! File saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found {input_file_path}")
    except PermissionError:
        print(f"Error: No permission to access file {input_file_path}")
    except Exception as e:
        print(f"Error processing file: {e}")

# Usage example
if __name__ == "__main__":
    # Replace with your file path
    input_file = "data/c_rl_data_all/part_affordance_train_rel_qwen_v0_train_r1.jsonl"

    # Optional: specify output file path (recommended to output to a new file first to verify results)
    # output_file = input_file + ".processed.jsonl"
    # convert_bbox_floats_to_ints(input_file, output_file)

    # Overwrite the original file directly (make sure you have a backup)
    convert_bbox_floats_to_ints(input_file)
