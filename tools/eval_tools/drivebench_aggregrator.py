# UPDATE: Replace placeholder paths with your actual paths.
import json
import os
import argparse
from pathlib import Path

# Try to import tabulate for pretty tables, fall back to simple text if not installed
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Define all Distortion Modes (corrected h256 -> h264, which is the common video compression distortion)
DISTORTION_MODES = [
    "biterror", "bright", "camcrash", "clean", "colorquant",
    "fog", "framelost", "h256",
    "lens", "lowlight", "motion", "rain", "saturate", "snow", "water", "zoom"
]

def load_json_file(filepath):
    """Safely load JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"Warning: File parsing failed (JSON format error): {filepath}")
        return None

def main():
    # ================= Parameter parsing =================
    parser = argparse.ArgumentParser(
        description="Aggregate model performance across multiple DriveBench distortion modes and generate a summary report."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name folder name (e.g., Qwen2.5VL-3B-Detection)"
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="results/eval_results/drivebench",
        help="Base directory path for DriveBench results"
    )

    parser.add_argument(
        "--output_suffix",
        type=str,
        default="aggregated_result_scores",
        help="Suffix name for the output file (default: aggregated_results)"
    )

    args = parser.parse_args()

    # Build model-specific results directory
    model_results_dir = os.path.join(args.base_dir, args.model_name)

    if not os.path.exists(model_results_dir):
        print(f"Error: Model results directory does not exist: {model_results_dir}")
        print("Please check --model_name or --base_dir parameters.")
        return

    results_data = {}
    scores = []
    missing_modes = []

    print(f"Scanning model: {args.model_name}")
    print(f"Directory path: {model_results_dir}")
    print("-" * 70)

    for mode in DISTORTION_MODES:
        # Build filename logic: {mode}_ms_swift_val_result.json
        filename = f"{mode}_ms_swift_val_result.json"
        filepath = os.path.join(model_results_dir, filename)

        data = load_json_file(filepath)

        if data is None:
            missing_modes.append(mode)
            continue

        # Extract key data
        try:
            overall_score = data['overall']['score']
            secondary_metrics = data.get('secondary_metrics', {})

            results_data[mode] = {
                "score": overall_score,
                "details": secondary_metrics
            }
            scores.append(overall_score)

        except KeyError as e:
            print(f"Error: File {filename} missing key field: {e}")
            continue

    if not scores:
        print("\nError: No valid result files found.")
        if missing_modes:
            print(f"Hint: The script tried to find the following modes but none were found: {missing_modes}")
            print("Please check if the filename format is '{mode}_ms_swift_val_result.json'")
        return

    # Calculate average
    avg_score = sum(scores) / len(scores)

    # ================= 1. Print table =================
    table_data = []
    for mode in DISTORTION_MODES:
        if mode in results_data:
            score = results_data[mode]['score']
            table_data.append([mode, f"{score:.4f}"])
        else:
            table_data.append([mode, "Missing"])

    # Add average row
    table_data.append(["-" * 10, "-" * 10])
    table_data.append(["AVERAGE", f"{avg_score:.4f}"])

    print("\n=== DriveBench Evaluation Results Summary ===")
    if HAS_TABULATE:
        print(tabulate(table_data, headers=["Mode", "Overall Score"], tablefmt="grid"))
    else:
        print(f"{'Mode':<15} | {'Score':<10}")
        print("-" * 28)
        for row in table_data:
            print(f"{row[0]:<15} | {row[1]:<10}")

    if missing_modes:
        print(f"\nNote: Files for the following Modes were not found ({len(missing_modes)} total): {missing_modes}")
        if "h256" in missing_modes and "h264" not in missing_modes:
             print("   (Hint: If the actual file is h264, the script has handled it automatically; if the file is indeed h256, please modify the DISTORTION_MODES list in the script)")

    # ================= 2. Generate summary JSON =================
    output_filename = f"{args.model_name}_{args.output_suffix}.json"
    output_path = os.path.join(model_results_dir, output_filename)

    output_json = {
        "overall": {
            "score": avg_score,
            "metric": "mixed-accuracy-language-aggregated",
            "total_modes_evaluated": len(scores),
            "missing_modes": missing_modes,
            "description": f"Aggregated average score across {len(scores)} distortion modes for {args.model_name}."
        },
        "secondary_metrics": {}
    }

    # Put details for each mode into secondary_metrics
    # Structure: secondary_metrics -> { "biterror": {...}, "bright": {...}, ... }
    for mode, content in results_data.items():
        output_json["secondary_metrics"][mode] = content["details"]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully generated summary file: {output_path}")
    print(f"Overall average score: {avg_score:.4f} (based on {len(scores)} valid modes)")

if __name__ == "__main__":
    main()
