# UPDATE: Replace placeholder paths with your actual paths.
import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

# ================= Configuration and Utility Functions =================

# [Fix] Optimized regex to handle both result/results and arbitrary characters in between
# Explanation: matches _result or _results, followed by any non-slash characters, ending with _scores
RESULT_SCORES_PATTERN = re.compile(r'_results?_[^/]*_scores\.json$')

def is_valid_result_file(filename: str) -> bool:
    """
    Determine if a file is a valid result file
    Compatible with:
    - xxx_result_scores.json
    - xxx_result_v1_scores.json
    - xxx_result_r1type_scores.json  <-- Fix focus: support extra characters in between
    - xxx_results_test_scores.json
    """
    # Strategy: as long as it contains 'result' (singular or plural) and ends with '_scores.json'
    if ('result' in filename.lower()) and (filename.endswith('_scores.json') or filename.endswith('_scores_r1.json')) :
        return True
    return False

def extract_overall_score(json_data: Dict) -> Optional[float]:
    """
    Extract overall score from JSON data
    Handles two formats:
    1. overall has a "score" key
    2. overall has a "metrics" dict
    """
    overall = json_data.get("overall", {})

    if not overall:
        return None

    # First format: has direct score
    if "score" in overall:
        return float(overall["score"])

    # Second format: metrics dict
    if "metrics" in overall:
        metrics = overall["metrics"]
        # If there is a total score, prefer total score
        if "score" in metrics:
            return float(metrics["score"])
        elif "accuracy" in metrics:
            return float(metrics["accuracy"])
        elif "total_score" in metrics:
            return float(metrics["total_score"])
        # If no explicit total score, return the first metric value
        elif metrics:
            first_val = list(metrics.values())[0]
            return float(first_val) if isinstance(first_val, (int, float)) else None

    return None

def get_metric_name(json_data: Dict) -> str:
    """
    Get the metric name
    """
    overall = json_data.get("overall", {})

    if "metric" in overall:
        return overall["metric"]
    elif "metrics" in overall and overall["metrics"]:
        # If there are multiple metrics, return the first metric name
        first_key = list(overall["metrics"].keys())[0]
        return first_key
    else:
        return "score"

def extract_model_name_from_filename(filename: str, dataset_name: str) -> str:
    """
    Optimize model name extraction: compatible with special characters (-/_/digits) in model names
    """
    # Remove .json suffix
    name_no_ext = filename.replace(".json", "")

    # Remove dataset prefix (handle dashed names like ego3d-bench)
    dataset_prefix = f"{dataset_name}_"
    if name_no_ext.startswith(dataset_prefix):
        name_no_ext = name_no_ext[len(dataset_prefix):]

    # Remove all result_xxx_scores related suffixes
    # Use regex to replace '_result...' to end part
    cleaned_name = re.sub(r'_results?_.*_scores$', '', name_no_ext)

    # Remove common extra suffixes
    non_model_suffixes = ["_full", "_val", "_test", "_train", "_mini", "_v1", "_v2"]
    for suffix in non_model_suffixes:
        if cleaned_name.endswith(suffix):
            cleaned_name = cleaned_name[:-len(suffix)]

    return cleaned_name.strip() if cleaned_name.strip() else "unknown_model"

def process_json_file(json_file: Path, dataset_name: str, model_name: str = None) -> Optional[Dict]:
    """
    Enhanced file processing: print detailed logs, ensure score extraction is correct
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        score = extract_overall_score(data)
        if score is not None:
            metric_name = get_metric_name(data)

            # Extract task name: remove dataset prefix and scores suffix from filename
            filename = json_file.stem
            task_name = re.sub(r'_results?_.*_scores$', '', filename)
            task_name = task_name.replace(f"{dataset_name}_", "").strip()
            if not task_name:
                task_name = "default_task"

            # Prefer directory name as model name (if exists), otherwise extract from filename
            final_model_name = model_name if model_name else extract_model_name_from_filename(json_file.name, dataset_name)

            result = {
                "dataset": dataset_name,
                "model": final_model_name,
                "task": task_name,
                "metric": metric_name,
                "score": score,
                "file_path": str(json_file)
            }

            return result
        else:
            print(f"  [WARN] No score extracted (format may not match): {json_file}")

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  [ERROR] File parse error {json_file}: {e}")
    except Exception as e:
        print(f"  [ERROR] File processing error {json_file}: {e}")

    return None

def collect_eval_results(eval_root: str = "eval_results") -> pd.DataFrame:
    """
    [Core Fix] Optimize directory traversal logic: ensure all subdirectories are covered, fix filename filtering issues
    """
    results = []

    eval_path = Path(eval_root)
    if not eval_path.exists():
        raise FileNotFoundError(f"Directory {eval_root} does not exist")

    print(f"\nStarting directory traversal: {eval_path.absolute()}")

    file_count = 0
    skip_count = 0

    # Recursively traverse all subdirectories
    for json_file in eval_path.rglob("*.json"):
        file_count += 1

        # [Fix] Use new lenient filtering logic, replacing original hardcoded string check
        if not is_valid_result_file(json_file.name):
            skip_count += 1
            continue

        # Parse dataset name and model name:
        # Expected path structure: eval_root/dataset_name/model_name/file.json
        try:
            relative_parts = json_file.relative_to(eval_path).parts
        except ValueError:
            continue

        if len(relative_parts) >= 2:
            dataset_name = relative_parts[0]  # First level is dataset name
            model_name = relative_parts[1]    # Second level is model name
        else:
            # If directory levels are insufficient, try to extract from filename (fallback)
            dataset_name = "unknown_dataset"
            model_name = None

        # Process file
        result = process_json_file(json_file, dataset_name, model_name)
        if result:
            results.append(result)

    # Print statistics
    print(f"Traversal complete: scanned {file_count} files, skipped {skip_count} non-target files, successfully parsed {len(results)} results")
    return pd.DataFrame(results)

# ================= Data Matrix Construction =================

def create_results_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create model x dataset results matrix"""
    if df.empty:
        return pd.DataFrame()

    # If there are duplicates, take average or first; here keep original logic and take first
    pivot_df = df.pivot_table(
        index='model',
        columns='dataset',
        values='score',
        aggfunc='first'
    )

    return pivot_df.reset_index()

def create_detailed_results_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create model x (dataset_task) detailed results matrix"""
    if df.empty:
        return pd.DataFrame()

    df['dataset_task'] = df['dataset'] + '_' + df['task']
    pivot_df = df.pivot_table(
        index='model',
        columns='dataset_task',
        values='score',
        aggfunc='first'
    )

    return pivot_df.reset_index()

# ================= Visualization Functions =================

def plot_heatmap(results_matrix: pd.DataFrame, save_path: str = "eval_results_heatmap.png"):
    """Generate heatmap"""
    if results_matrix.empty or len(results_matrix) <= 1 or len(results_matrix.columns) <= 1:
        print("[WARN] Insufficient data, cannot generate heatmap")
        return

    matrix_data = results_matrix.set_index('model')

    fig_width = max(10, len(matrix_data.columns) * 0.8)
    fig_height = max(6, len(matrix_data) * 0.5)
    plt.figure(figsize=(fig_width, fig_height))

    sns.heatmap(matrix_data,
                annot=True,
                fmt=".4f",
                cmap="YlOrRd",
                cbar_kws={'label': 'Score'},
                linewidths=0.5,
                linecolor='gray')

    plt.title("Model Performance Across Datasets", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Dataset", fontsize=12, fontweight='bold')
    plt.ylabel("Model", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Heatmap saved to: {save_path}")

def plot_grouped_bar_chart(df: pd.DataFrame, save_path: str = "eval_results_grouped_bar.png"):
    """Generate grouped bar chart"""
    if df.empty:
        print("[WARN] Data is empty, cannot generate bar chart")
        return

    pivot_df = df.pivot_table(
        index='model',
        columns='dataset',
        values='score',
        aggfunc='first'
    )

    if pivot_df.empty or len(pivot_df.columns) == 0:
        print("[WARN] Insufficient data, cannot generate bar chart")
        return

    plt.figure(figsize=(max(12, len(pivot_df.columns) * 0.8), 8))

    n_models = len(pivot_df)
    n_datasets = len(pivot_df.columns)

    x = np.arange(n_datasets)
    width = 0.8 / max(1, n_models)

    for i, (model, scores) in enumerate(pivot_df.iterrows()):
        positions = x + (i - n_models/2 + 0.5) * width
        plt.bar(positions, scores.values, width, label=model, alpha=0.8)

    plt.xlabel('Dataset', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison Across Datasets', fontsize=16, fontweight='bold', pad=20)

    if n_datasets > 0:
        plt.xticks(x, pivot_df.columns, rotation=45, ha='right')

    if n_models > 1:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Grouped bar chart saved to: {save_path}")

def plot_radar_chart(df: pd.DataFrame, save_path: str = "eval_results_radar.png"):
    """Generate radar chart"""
    models = df['model'].unique()
    datasets = df['dataset'].unique()

    if len(models) < 2 or len(datasets) < 3:
        print(f"[WARN] Number of models({len(models)}) less than 2 or number of datasets({len(datasets)}) less than 3, skipping radar chart")
        return

    data = []
    for model in models:
        model_scores = []
        for dataset in datasets:
            score = df[(df['model'] == model) & (df['dataset'] == dataset)]['score'].values
            model_scores.append(score[0] if len(score) > 0 else 0)
        data.append(model_scores)

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2*np.pi, len(datasets), endpoint=False).tolist()
    angles += angles[:1]

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    for i, (model, scores) in enumerate(zip(models, data)):
        scores_plus = scores + scores[:1]
        ax.plot(angles, scores_plus, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, scores_plus, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets, fontsize=10)

    all_scores = [score for scores in data for score in scores]
    if all_scores:
        ax.set_ylim(0, max(all_scores) * 1.2)
    ax.set_yticks([])

    plt.title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Radar chart saved to: {save_path}")

def plot_model_comparison(df: pd.DataFrame, save_dir: str = "."):
    """Generate model comparison chart for each dataset"""
    if df.empty:
        print("[WARN] Data is empty, cannot generate model comparison chart")
        return

    datasets = df['dataset'].unique()

    generated_count = 0
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        if dataset_df.empty or len(dataset_df['model'].unique()) < 1: # Draw even with only one model, for easy viewing
            continue

        plt.figure(figsize=(10, 6))

        sorted_df = dataset_df.sort_values('score', ascending=True)

        bars = plt.barh(sorted_df['model'], sorted_df['score'])

        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}',
                    ha='left', va='center', fontsize=10)

        plt.xlabel('Score')
        plt.title(f'{dataset} - Model Performance Comparison')
        plt.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        safe_dataset_name = dataset.replace('/', '_').replace('\\', '_').replace('-', '_')
        save_path = os.path.join(save_dir, f"{safe_dataset_name}_model_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Model comparison chart for dataset {dataset} saved to: {save_path}")
        generated_count += 1

    if generated_count == 0:
        print("[WARN] No model comparison charts generated (possibly insufficient models in datasets)")

# ================= Main Function =================

def main():
    parser = argparse.ArgumentParser(description='Evaluation result processing and visualization tool (fixed version)')

    parser.add_argument('--input', '-i', type=str,
                       default="results/eval_results20260301",
                       help='Input directory path containing evaluation result files')
    parser.add_argument('--output', '-o', type=str,
                       default="results/vis",
                       help='Output directory path for saving visualization results')
    parser.add_argument('--save-csv', action='store_true', default=True,
                       help='Whether to save raw data and results matrix in CSV format')
    parser.add_argument('--heatmap', action='store_true', default=True,
                       help='Whether to generate heatmap')
    parser.add_argument('--bar-chart', action='store_true', default=True,
                       help='Whether to generate grouped bar chart')
    parser.add_argument('--radar', action='store_true', default=True,
                       help='Whether to generate radar chart')
    parser.add_argument('--model-comparison', action='store_true', default=True,
                       help='Whether to generate model comparison chart for each dataset')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Show detailed processing information')
    # [New parameter] Specify model list to plot, multiple models separated by spaces
    parser.add_argument('--filter-models', '-m', nargs='+', type=str, default=[],
                       help='List of model names to visualize (multiple names separated by spaces); if not specified, all models are shown')

    args = parser.parse_args()

    BASE_ROOT = args.input
    BASE_SAVE_ROOT = args.output

    # Ensure save directory exists
    os.makedirs(BASE_SAVE_ROOT, exist_ok=True)

    print("="*60)
    print("Starting evaluation result processing (fixed version)")
    print(f"Input directory: {BASE_ROOT}")
    print(f"Output directory: {BASE_SAVE_ROOT}")
    print("="*60)

    # Collect all results
    try:
        df = collect_eval_results(BASE_ROOT)
    except Exception as e:
        print(f"[ERROR] Severe error during result collection: {e}")
        return

    if df.empty:
        print("[ERROR] No valid result files found! Please check input directory and filename format.")
        print("Hint: filenames must contain 'result' and end with '_scores.json'.")
        return

    # ================= [Core Addition] Model Filtering Logic =================
    # Filter specified models
    if args.filter_models:
        # Get all existing model names
        existing_models = set(df['model'].unique())
        # Check if specified models exist
        input_models = set(args.filter_models)
        invalid_models = input_models - existing_models
        valid_models = input_models & existing_models

        if invalid_models:
            print(f"[WARN] The following specified models do not exist, ignored: {sorted(list(invalid_models))}")
            print(f"   Available model list: {sorted(list(existing_models))}")

        if not valid_models:
            print("[ERROR] No valid model names, cannot continue visualization!")
            return

        # Filter data, keep only specified valid models
        df_filtered = df[df['model'].isin(valid_models)].copy()
        print(f"\n[OK] Data filtered, keeping only specified models: {sorted(list(valid_models))}")
        print(f"   Records before filtering: {len(df)}, records after filtering: {len(df_filtered)}")
    else:
        # No filter specified, use all data
        df_filtered = df.copy()
        print(f"\n[OK] No model filter, using all data ({len(df_filtered)} records total)")

    # Print basic info after filtering
    print(f"   Models involved: {sorted(df_filtered['model'].unique().tolist())}")
    print(f"   Datasets involved: {sorted(df_filtered['dataset'].unique().tolist())}")

    # Save CSV (save filtered data)
    if args.save_csv:
        # Save filtered raw data
        csv_save_path = os.path.join(BASE_SAVE_ROOT, "all_results_filtered.csv")
        df_filtered.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
        print(f"\n[OK] Filtered raw data saved to: {csv_save_path}")

        # Optional: save original all data (for comparison)
        csv_raw_path = os.path.join(BASE_SAVE_ROOT, "all_results_raw.csv")
        df.to_csv(csv_raw_path, index=False, encoding='utf-8-sig')
        print(f"[OK] Original all data saved to: {csv_raw_path}")

    # Create filtered results matrix
    matrix_df = create_results_matrix(df_filtered)
    if not matrix_df.empty:
        result_save_path = os.path.join(BASE_SAVE_ROOT, "results_matrix_filtered.csv")
        matrix_df.to_csv(result_save_path, index=False, encoding='utf-8-sig')
        print(f"[OK] Filtered results matrix saved to: {result_save_path}")
        print("\nFiltered results matrix preview:")
        print(matrix_df.to_string())
    else:
        print("[WARN] Filtered results matrix is empty")
        return

    # Set plotting fonts
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # Visualization (all using filtered data)
    if args.heatmap and not matrix_df.empty:
        heatmap_save_path = os.path.join(BASE_SAVE_ROOT, "eval_results_heatmap_filtered.png")
        plot_heatmap(matrix_df, heatmap_save_path)

    if args.bar_chart:
        bar_save_path = os.path.join(BASE_SAVE_ROOT, "eval_results_grouped_bar_filtered.png")
        plot_grouped_bar_chart(df_filtered, bar_save_path)

    if args.radar:
        radar_save_path = os.path.join(BASE_SAVE_ROOT, "eval_results_radar_filtered.png")
        plot_radar_chart(df_filtered, radar_save_path)

    if args.model_comparison:
        plot_model_comparison(df_filtered, BASE_SAVE_ROOT)

    # Print summary statistics
    print("\n" + "="*60)
    print("Processing complete! Generated file list:")
    for root, dirs, files in os.walk(BASE_SAVE_ROOT):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, BASE_SAVE_ROOT)
            print(f"   - {rel_path}")
    print("="*60)

if __name__ == "__main__":
    main()
