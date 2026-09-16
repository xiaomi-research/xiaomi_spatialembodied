# UPDATE: Replace placeholder paths with your actual paths.
#
# ⚠️ This script rewrites the input file IN PLACE (no backup). It used to also run at
# import time, so merely importing the module overwrote the dataset below. It now only
# runs when executed as a script, and backs the original up first.
import argparse
import json
import re
import shutil
from pathlib import Path


def update_option_format(text):
    """
    Update the multiple-choice option format from A) to A. in the text.
    :param text: Input text string
    :return: Updated text string
    """
    # Use regex to match and replace option format
    return re.sub(r'([A-Za-z])\)', r'\1.', text)

def process_jsonl(file_path):
    """
    Process a JSON Lines file, updating the multiple-choice option format within it.
    :param file_path: Path to the JSON Lines file
    """
    updated_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            for message in record.get('messages', []):
                if 'content' in message:
                    message['content'] = update_option_format(message['content'])
            updated_lines.append(json.dumps(record, ensure_ascii=False))

    # Overwrite the original file, keeping a one-off backup next to it.
    backup = Path(file_path).with_suffix(Path(file_path).suffix + '.bak')
    if not backup.exists():
        shutil.copy2(file_path, backup)
        print(f"[fix_drivelmm_option] backup written to {backup}")
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in updated_lines:
            file.write(line + '\n')


DEFAULT_FILE = 'data/c_rl_data_all/data/DriveLMMo1_swift_TRAIN_train_r1v2.jsonl'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', default=DEFAULT_FILE, help='JSONL file to rewrite in place')
    args = parser.parse_args()
    if not Path(args.file).exists():
        raise SystemExit(f"Input file not found: {args.file}")
    print(f"[fix_drivelmm_option] rewriting {args.file} in place")
    process_jsonl(args.file)
    print("[fix_drivelmm_option] done")


if __name__ == '__main__':
    main()
