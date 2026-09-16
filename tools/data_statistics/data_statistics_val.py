# UPDATE: Replace placeholder paths with your actual paths.
#
# Sample N% of several benchmark datasets into a training split.
#
# ⚠️ READ THIS BEFORE RUNNING ⚠️
# The inputs below live under `all_eval_data/` and end in `_test.jsonl` — they are the
# evaluation files for the benchmarks reported in the paper's Table 3. Drawing a training
# split from them means those very samples are seen during training *and* scored at
# evaluation time, which inflates the reported numbers.
#
# This script therefore now writes BOTH sides of the split:
#   * `<output_file>`  — the sampled records, to be added to training (unchanged behaviour)
#   * `<input>_heldout.jsonl` — the records that were NOT sampled, i.e. a de-contaminated
#     evaluation file. Point your evaluation at this file so the benchmark stays disjoint
#     from training.
#
# The original test file is never modified.
#
# vabench, embodied-r1, roboafford, where2place, Part-Affordance-2K need to sample 20% into training set
# LingoQA-Action	data/lingoqa_action_train_qwen_v0.json
# Surds	data/merged_surds_vqas_v0.jsonl
# Maplmv2	data/MapLM/filter_data/maplm_filter_06.jsonl
# Maplmv2-Multi	data/MapLM/filter_data/maplm_filter_06_multi.jsonl
# Drivingvqa	data/Drivevqa/filter_data/drivevqa_rel_filter_06.jsonl
# DriveLMM-o1	data/DriveLMMo1/filter_data/filtered_raw_data_drivelmm_06_v2.jsonl
# OmniDrive-vqa	data/Omnidrive/filter_data/omnidrive_vqa_filtered_06.jsonl
# OmniDrive-conv	data/Omnidrive/filter_data/omnidrive_vqa_filtered_multi_06.jsonl
# Refcoco	data/RefCOCO/raw_data/refcoco_train_qwen_v1.jsonl
# Flickr	data/Flickr/filter_data/flickr_filter_06.jsonl
# Vqav2	data/VQAv2/filter_data/vqa_filter_06.jsonl
# Gqa	data/Gqa/filter_data/gqa_filter_06.jsonl
# Bdd	data/bdd100k_labels_images_train_weather_qwen_v1.json
# roborefit	data/Roborefit/filter_data/roborefit_filter_06.jsonl
# Cosmosr1	data/Cosmosr1/filter_data/cosmosr1_filter_06.jsonl
# vladbench	data/vladbench_train_image_rel_qwen_v1.json
# robovqa	data/robovqa_understanding.jsonl


import argparse
import json
import os
import random
from pathlib import Path

DEFAULT_SEED = 42


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_subset_to_file(data, output_file_path):
    os.makedirs(os.path.dirname(output_file_path) or '.', exist_ok=True)
    is_jsonl = output_file_path.lower().endswith('.jsonl')
    with open(output_file_path, 'w', encoding='utf-8') as file:
        if is_jsonl:
            for item in data:
                file.write(json.dumps(item) + '\n')
        else:
            json.dump(data, file, ensure_ascii=False, indent=4)


def sample_data(input_file, output_file, sample_rate=0.2, *, seed=DEFAULT_SEED, heldout_suffix='_heldout'):
    """
    Split `input_file` into a sampled training subset and the held-out remainder.

    Writes the sampled records to `output_file` and the remainder to
    `<input stem><heldout_suffix><input suffix>` next to the input. The input file itself is
    left untouched.

    Returns `(n_sampled, n_heldout, heldout_path)`.
    """
    if input_file.lower().endswith('.json'):
        dataset = load_json_file(input_file)
    elif input_file.lower().endswith('.jsonl'):
        dataset = load_jsonl_file(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")

    # A fixed seed makes the split reproducible; the previous version used the global RNG,
    # so two runs produced different training sets.
    rng = random.Random(seed)
    n_sample = int(len(dataset) * sample_rate)
    sampled_idx = set(rng.sample(range(len(dataset)), n_sample))
    sampled_data = [d for i, d in enumerate(dataset) if i in sampled_idx]
    heldout_data = [d for i, d in enumerate(dataset) if i not in sampled_idx]

    save_subset_to_file(sampled_data, output_file)

    src = Path(input_file)
    heldout_path = src.with_name(f"{src.stem}{heldout_suffix}{src.suffix}")
    save_subset_to_file(heldout_data, str(heldout_path))

    return len(sampled_data), len(heldout_data), str(heldout_path)


# Example usage:
DATASETS = [
    # Modify paths according to your actual situation
    # ("/path/to/your/dataset.json", "/path/to/output/train_set.json"),
    ("data/preprocess_data/all_eval_data/embodied-r1/embodied-r1_ms_swift_test.jsonl",
     "data/preprocess_data/embodied-r1-train/embodied-r1_ms_swift_train.jsonl"),
    ("data/preprocess_data/all_eval_data/Part-Affordance-2K/part_affordance_ms_swift.jsonl",
     "data/preprocess_data/Part-Affordance-2K-train/part_affordance_ms_swift_train.jsonl"),
    ("data/preprocess_data/all_eval_data/RoboAfford/RoboAfford_ms_swift.jsonl",
     "data/preprocess_data/RoboAfford-train/RoboAfford_ms_swift_train.jsonl"),
    ("data/preprocess_data/all_eval_data/vabench-point-bbox/vabench_bbox_absolute_no_reasoning_ms_swift.jsonl",
     "data/preprocess_data/vabench-point-bbox-train/vabench_bbox_absolute_no_reasoning_ms_swift_train.jsonl"),
    ("data/preprocess_data/all_eval_data/where2place/where2place_ms_swift_absolute.jsonl",
     "data/preprocess_data/where2place-train/where2place_ms_swift_absolute_train.jsonl")
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-rate', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only report what would be written.',
    )
    args = parser.parse_args()

    print(
        "NOTE: the inputs are evaluation files. Each run writes a matching "
        "'*_heldout.<ext>' next to them; use those for evaluation so the benchmark stays "
        "disjoint from the training split.\n"
    )
    for dataset, output in DATASETS:
        if not os.path.isfile(dataset):
            print(f"[SKIP] input not found: {dataset}")
            continue
        if args.dry_run:
            print(f"[dry-run] {dataset}  ->  {output} (+ held-out sibling)")
            continue
        n_sampled, n_heldout, heldout_path = sample_data(
            dataset, output, args.sample_rate, seed=args.seed
        )
        print(f"{dataset}")
        print(f"    -> {output}  ({n_sampled} to train)")
        print(f"    -> {heldout_path}  ({n_heldout} held out — evaluate on this)")


if __name__ == '__main__':
    main()
