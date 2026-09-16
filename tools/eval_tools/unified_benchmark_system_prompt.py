# UPDATE: Replace placeholder paths with your actual paths.
#
# ⚠️ DESTRUCTIVE BY DESIGN: this walks a directory tree and rewrites the `system` message
# of EVERY `.jsonl` file it finds, in place. It used to do that unconditionally on `main()`,
# with no dry-run and no backup. It now defaults to a dry-run; pass `--apply` to actually
# write, and the originals are copied into `--backup-dir` first.
#
# Note the default `BASE_DIR` overlaps the default output directory of the 15 converters in
# `tools/convert_data_tools/`, so running this with `--apply` rewrites their output too.
import argparse
import json
import os
import shutil
from pathlib import Path

# Base directory path
BASE_DIR = "data/preprocess_data/all_eval_data"

# New system prompt content (English)
NEW_SYSTEM_PROMPT = (
    "You are a professional multimodal AI assistant. Watch the video, image or multi-view images and answer questions. "
    "Enclose your final answer within <answer> tags, like this: <answer>your answer here</answer>."
)

def update_jsonl_file(file_path):
    """Return the rewritten lines for `file_path` without touching the file."""
    updated_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Find the message with role "system" in messages and replace content
            found_system = False
            for msg in data["messages"]:
                if msg["role"] == "system":
                    msg["content"] = NEW_SYSTEM_PROMPT
                    found_system = True
                    break
            if not found_system:
                # If no system message, insert one (though your data should have one)
                data["messages"].insert(0, {"role": "system", "content": NEW_SYSTEM_PROMPT})
            updated_lines.append(json.dumps(data, ensure_ascii=False))
    return updated_lines


def write_jsonl_file(file_path, updated_lines, *, backup_dir):
    """Back the file up, then overwrite it in place."""
    if backup_dir is not None:
        dest = Path(backup_dir) / Path(file_path).name
        if not dest.exists():
            shutil.copy2(file_path, dest)
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in updated_lines:
            f.write(line + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', default=BASE_DIR, help='Directory tree to walk')
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually rewrite the files. Without this flag the script only reports what it would change.',
    )
    parser.add_argument(
        '--backup-dir',
        default=None,
        help='Copy each file here before overwriting it (only used together with --apply).',
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(f"Directory not found: {args.dir}")

    targets = []
    for root, _dirs, files in os.walk(args.dir):
        targets.extend(
            os.path.join(root, f) for f in sorted(files) if f.endswith('.jsonl')
        )

    if not args.apply:
        print(f"[dry-run] would rewrite {len(targets)} .jsonl file(s) under {args.dir}:")
        for p in targets[:20]:
            print(f"    {p}")
        if len(targets) > 20:
            print(f"    ... and {len(targets) - 20} more")
        print("[dry-run] re-run with --apply to write. Add --backup-dir <path> to keep copies.")
        return

    for full_path in targets:
        print(f"Processing: {full_path}")
        updated_lines = update_jsonl_file(full_path)
        if updated_lines:
            write_jsonl_file(full_path, updated_lines, backup_dir=args.backup_dir)
    print(f"Updated {len(targets)} JSONL file(s).")


if __name__ == "__main__":
    main()
