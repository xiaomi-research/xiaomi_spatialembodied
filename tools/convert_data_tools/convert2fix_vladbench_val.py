# UPDATE: Replace placeholder paths with your actual paths.
import json
import sys
from pathlib import Path

def fix_vladbench_jsonl(input_file, output_file=None):
    """Fix VLADBench JSONL file format issues"""
    if output_file is None:
        output_file = str(input_file).replace('.jsonl', '_fixed.jsonl')

    fixed_images = 0
    fixed_content = 0
    line_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for i, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            line_count += 1

            try:
                # Parse JSON
                data = json.loads(line)

                # Fix images field
                if 'images' in data:
                    if isinstance(data['images'], str):
                        # Single string -> convert to array
                        data['images'] = [data['images']]
                        fixed_images += 1
                        print(f"Line {i}: Fixed images field (string -> array)")

                    # Ensure it's a list
                    elif isinstance(data['images'], list):
                        # Already an array, no fix needed
                        pass
                    else:
                        # Other type -> clear array
                        data['images'] = []
                        fixed_images += 1
                        print(f"Line {i}: Fixed images field (other type -> array)")

                system_prompt = "You are an excellent driver, please answer the question regarding driving scenes."
                # Insert the system prompt only if there is not one already: the previous
                # version inserted unconditionally, so re-running this script on its own
                # output stacked up duplicate system messages. The `if 'messages' in data`
                # guard below used to be dead too, because the insert had already run.
                if 'messages' in data and isinstance(data['messages'], list):
                    has_system = any(
                        isinstance(m, dict) and m.get('role') == 'system' for m in data['messages']
                    )
                    if not has_system:
                        data['messages'].insert(0, {"role": "system", "content": system_prompt})
                    # Fix messages/content field
                    for message in data['messages']:
                        if 'content' in message:
                            # Ensure content is a string
                            if not isinstance(message['content'], str):
                                # Try to convert to string
                                try:
                                    message['content'] = str(message['content'])
                                    fixed_content += 1
                                    print(f"Line {i}: Fixed content field (non-string -> string)")
                                except:
                                    # Conversion failed, set to empty string
                                    message['content'] = ""
                                    fixed_content += 1
                                    print(f"Line {i}: Fixed content field (set to empty string)")

                # Write fixed data
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

            except json.JSONDecodeError as e:
                print(f"Line {i}: JSON parse error - {e}")
                print(f"  Line content: {line[:100]}...")
                # Try cleaning trailing characters
                if line.endswith(','):
                    line = line[:-1]
                    try:
                        data = json.loads(line)
                        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                        print(f"  Fixed: removed trailing comma")
                    except:
                        print(f"  Cannot fix, skipping this line")
                else:
                    print(f"  Cannot fix, skipping this line")

    print(f"\nProcessing complete:")
    print(f"  Total lines: {line_count}")
    print(f"  Fixed images field lines: {fixed_images}")
    print(f"  Fixed content field lines: {fixed_content}")
    print(f"  Output file: {output_file}")

    return output_file

DEFAULT_INPUT = 'data/preprocess_data/vladbench_swift.jsonl'


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=DEFAULT_INPUT, help='Input JSONL file')
    parser.add_argument(
        '--output',
        default=None,
        help='Output JSONL file (default: <input>_fixed.jsonl)',
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise SystemExit(f"Input file not found: {args.input}")
    output_path = fix_vladbench_jsonl(args.input, args.output)
    print(f"Wrote {output_path}")


# This used to run at import time, so importing the module (or anything that imported it)
# rewrote a file as a side effect.
if __name__ == '__main__':
    main()
