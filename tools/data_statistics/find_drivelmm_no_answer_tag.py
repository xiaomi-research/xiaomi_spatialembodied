# UPDATE: Replace placeholder paths with your actual paths.
import json
import re

def has_answer_tag(text: str) -> bool:
    """Check if text contains <answer>...</answer> (case-insensitive, supports newlines)."""
    return bool(re.search(r'<answer>.*?</answer>', text, re.IGNORECASE | re.DOTALL))

input_path = 'data/c_rl_data_all/data/DriveLMMo1_swift_TRAIN_train_r1v2.jsonl'
output_path = 'data/c_rl_data_all/data/missing_answer_tag.jsonl'

missing_count = 0
total_count = 0

with open(input_path, 'r', encoding='utf-8') as fin, \
     open(output_path, 'w', encoding='utf-8') as fout:

    for line_num, line in enumerate(fin, 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(f"Invalid JSON at line {line_num}")
            continue

        total_count += 1
        assistant_content = None

        # Find assistant message
        for msg in record.get('messages', []):
            if msg.get('role') == 'assistant':
                assistant_content = msg.get('content', '')
                break

        if assistant_content is None:
            print(f"Warning: No assistant message in line {line_num}")
            continue

        if not has_answer_tag(assistant_content):
            missing_count += 1
            print(f"Missing <answer> tag at line {line_num}")
            # Optional: write to output file
            fout.write(line + '\n')

print(f"\nTotal records: {total_count}")
print(f"Records missing <answer> tag: {missing_count}")
print(f"Saved problematic records to: {output_path}")
