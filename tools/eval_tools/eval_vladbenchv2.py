# UPDATE: Replace placeholder paths with your actual paths.
import os
import json
import re
import pandas as pd
import argparse
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import sys
from pathlib import Path
sys.path.append("/path/to/eval/VLADBench-main")

try:
    from evaluate_utils_v1 import func_mapping, weighted_row_sum, weighted_total
except ImportError as e:
    # `evaluate_utils_v1` ships with the VLADBench repository; it is NOT part of this one.
    raise ImportError(
        "eval_vladbenchv2.py requires `evaluate_utils_v1`. It is not part of this "
        "repository.\n  Clone VLADBench and point the `sys.path.append('/path/to/eval/"
        "VLADBench-main')` line above at it."
    ) from e

current_dir = os.path.dirname(os.path.abspath(__file__))
base_model_dir = os.path.dirname(current_dir)  # since current_dir is .../<repo-root>/eval
sys.path.append(base_model_dir)
try:
    from eval.eval_utils.preprocessor import preprocess_model_output
except ImportError:
    # `eval.eval_utils` belongs to the upstream evaluation harness, not to this repository.
    # Fall back to the identity so the script still runs, but say so out loud: answer
    # normalisation is skipped, so the scores are not directly comparable to the official
    # benchmark numbers.
    def preprocess_model_output(text):
        return text

    print(
        "[WARN] eval.eval_utils.preprocessor not found; using an identity "
        "preprocess_model_output(). Scores may differ from the official benchmark."
    )

# breakpoint()
def extract_third_task_from_path(image_path: str) -> str:
    """Extract third-level task name from image path"""
    # breakpoint()
    if isinstance(image_path, list):
        first_path = image_path[0]
    else:
        first_path = image_path
    parts = first_path.split('/')
    try:
        vlad_idx = parts.index("VLADBench")
        return parts[vlad_idx + 3]
    except (ValueError, IndexError):
        raise ValueError(f"Cannot extract third_task from path {first_path}")

def extract_answer_from_tag(text):
    """Extract answer content from <answer> tags"""
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def parse_response_for_r1(response, r1_type=False):
    """Parse response based on whether R1-type is enabled"""
    if r1_type:
        response = preprocess_model_output(response)
        return extract_answer_from_tag(response)
    return response

def save_normalized_json_output(results: Dict[str, Any], output_path: str, model_name: str,
                               infer_file_path: str, aggregated_results: Dict) -> str:
    """Save normalized JSON output format"""
    # Get basename of input file
    infer_basename = os.path.basename(infer_file_path)
    json_filename = os.path.splitext(infer_basename)[0] + "_scores.json"

    # Build normalized output structure
    normalized_results = {
        "model_name": model_name,
        "evaluation_dataset": "VLADBench",
        "evaluation_time": pd.Timestamp.now().isoformat(),
        "infer_file": infer_basename,
        "overall": {
            "score": 0.0,
            "metric": "vladbench_score",
            "total_samples": 0,
            "evaluated_tasks": 0
        },
        "secondary_metrics": {},  # Changed to flattened key-value format
        "detailed_results": {}
    }

    # Extract second-level task scores (flattened key-value pairs)
    secondary_metrics = {}
    detailed_results = {}

    # Iterate through aggregated results, build detailed results
    for (first_task, second_task, third_task), task_data in aggregated_results.items():
        if model_name not in task_data['models']:
            continue

        # Ensure first-level task exists
        if first_task not in detailed_results:
            detailed_results[first_task] = {}
            normalized_results["detailed_results"][first_task] = {}

        # Ensure second-level task exists
        if second_task not in detailed_results[first_task]:
            detailed_results[first_task][second_task] = {}
            normalized_results["detailed_results"][first_task][second_task] = {}

        # Add third-level task detailed results
        model_result = task_data['models'][model_name]
        normalized_results["detailed_results"][first_task][second_task][third_task] = {
            "score": round(model_result['score'], 4),
            "accuracy": round(model_result['accuracy'], 4),
            "obey_rate": round(model_result['obey'], 4),
            "other_metric": round(model_result['others'], 4),
            "sample_count": task_data['samples']
        }

    # Calculate secondary metrics - changed to flattened key-value format
    for first_task in normalized_results["detailed_results"]:
        task_scores = []
        total_samples = 0

        for second_task in normalized_results["detailed_results"][first_task]:
            for third_task, metrics in normalized_results["detailed_results"][first_task][second_task].items():
                task_scores.append(metrics["score"] * metrics["sample_count"])
                total_samples += metrics["sample_count"]

        if total_samples > 0:
            weighted_score = sum(task_scores) / total_samples
        else:
            weighted_score = 0.0

        # Use first-level task name directly as secondary metric name
        secondary_metrics[f"{first_task}"] = round(weighted_score, 4)

    # Calculate overall metrics
    total_weighted_score = 0
    total_all_samples = 0
    total_tasks = 0

    for first_task, score in secondary_metrics.items():
        # Need to find sample count for each first-level task
        first_task_samples = 0
        for second_task in normalized_results["detailed_results"].get(first_task, {}):
            for third_task, metrics in normalized_results["detailed_results"][first_task][second_task].items():
                first_task_samples += metrics["sample_count"]

        total_weighted_score += score * first_task_samples
        total_all_samples += first_task_samples
        total_tasks += len(normalized_results["detailed_results"].get(first_task, {}).keys())

    if total_all_samples > 0:
        overall_score = total_weighted_score / total_all_samples
    else:
        overall_score = 0.0

    normalized_results["secondary_metrics"] = secondary_metrics
    normalized_results["overall"] = {
        "score": round(overall_score, 4),
        "total_samples": total_all_samples,
        "evaluated_tasks": total_tasks
    }

    # Save JSON file
    json_path = os.path.join(os.path.dirname(output_path), json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(normalized_results, f, ensure_ascii=False, indent=2)

    print(f"Normalized JSON results saved to: {json_path}")
    return json_path

def main():
    # ====== Argument parsing ======
    parser = argparse.ArgumentParser(description='VLADBench evaluation script')

    # Original arguments (maintain backward compatibility)
    parser.add_argument('--infer_root', type=str,
                       default="/path/to/data/infer_results/vladbench",
                       help='Inference result root directory (legacy parameter, for multi-model evaluation)')
    parser.add_argument('--eval_output_dir', type=str,
                       default="/path/to/data/eval_results/VLADBench",
                       help='Evaluation result output directory')
    parser.add_argument('--models', type=str, nargs='+',
                       default=["Qwen2_5_VL-3B"],
                       help='List of models to evaluate, separated by spaces (legacy parameter)')

    # New parameters (normalized interface)
    parser.add_argument('--infer_file_path', type=str, default=None,
                       help='Inference result JSONL file path (new parameter, single file evaluation)')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Name of the model being evaluated (new parameter)')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Original dataset metadata JSONL file path (new parameter, reserved for extension)')
    parser.add_argument('--output_dir', type=str,
                       default="/path/to/data/eval_results/VLADBench")

    # Common parameters
    parser.add_argument('--all_task_json', type=str,
                       default='/path/to/eval/VLADBench-main/all_task.json',
                       help='Task structure definition file path')
    parser.add_argument('--weights_config', type=str, default='',
                       help='Weight configuration JSON file path (optional)')
    parser.add_argument('--save_normalized_json', default=True, action='store_false',
                       help='Whether to save normalized JSON format output')

    # Add R1-type evaluation parameter
    parser.add_argument('--r1_type', action='store_true',
                       help='Whether this is R1-type evaluation (extract answers from <answer> tags)')

    args = parser.parse_args()

    # Detect run mode
    if args.infer_file_path and args.model_name:
        # New interface: single model evaluation mode
        print("Run mode: Single model evaluation (new interface)")
        models_to_evaluate = [args.model_name]
        infer_files = {args.model_name: args.infer_file_path}
    else:
        # Legacy interface: multi-model evaluation mode
        print("Run mode: Multi-model evaluation (legacy interface)")
        models_to_evaluate = args.models
        infer_files = {}
        for model in models_to_evaluate:
            infer_files[model] = os.path.join(args.infer_root, model, "val.jsonl")

    # Check required files
    for model, infer_file in infer_files.items():
        if not os.path.exists(infer_file):
            print(f"Error: Inference file does not exist: {infer_file}")
            if args.infer_file_path:  # If single model mode, exit directly
                sys.exit(1)
            else:  # Multi-model mode, skip this model
                print(f"Skipping model: {model}")
                models_to_evaluate.remove(model)

    if not models_to_evaluate:
        print("Error: No valid models to evaluate")
        sys.exit(1)

    eval_output_dir = args.output_dir
    all_task_json = args.all_task_json
    model_names = models_to_evaluate

    os.makedirs(eval_output_dir, exist_ok=True)

    # Read task definitions
    all_tasks = json.load(open(all_task_json, 'r', encoding='utf-8'))

    # Weight configuration
    if args.weights_config and os.path.exists(args.weights_config):
        with open(args.weights_config, 'r', encoding='utf-8') as f:
            weights = json.load(f)
    else:
        # Default weights
        weights = {
            'Vehicle_Recognition': [0.3, 0.5, 0.2],
            'VRU_Recognition': [0.3, 0.5, 0.2],
            'Obstruction_Recognition': [0.3, 0.5, 0.2],
            'Sign_Sign_Relation': [0.3, 0.5, 0.2],
            'Sign_Lane_Relation': [0.3, 0.5, 0.2],
            'Light_Lane_Relation': [0.3, 0.5, 0.2],
            'Lane_Speed_Relation': [0.3, 0.5, 0.2],
            'Lane_Change_Relation': [0.3, 0.5, 0.2],
            'VRU_Cutin': [0.7, 0.1, 0.2],
            'Vehicle_Cutin': [0.7, 0.1, 0.2],
            'VRU_Cross': [0.7, 0.1, 0.2],
            'Key_Obsturction_Detection': [0.8, 0, 0.2],
            'Risk_Prediction': [0.7, 0.1, 0.2],
            'Spatial_Temporal_Reasoning': [0.4, 0.4, 0.2]
        }


    # ======================== Helper functions ========================
    def create_model_output_dirs(model_name: str) -> str:
        """Create output directory for model and return path"""
        model_output_dir = os.path.join(eval_output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        return model_output_dir

    def save_model_results(model_name: str, all_results: List, aggregated_results: Dict):
        """Save evaluation results for a single model (maintain original format)"""
        # Create model output directory
        model_output_dir = create_model_output_dirs(model_name)

        # Save Excel and CSV results
        save_path = os.path.join(model_output_dir, f"{model_name}_results.xlsx")
        csv_path = os.path.join(model_output_dir, f"{model_name}_results.csv")

        # Filter results containing this model
        model_results = []
        for row in all_results:
            if len(row) >= 3:  # Ensure sufficient columns
                task_name = row[0]
                sample_count = row[1]
                # Find this model's position in results
                for i, col_name in enumerate(['Task', 'num'] + model_names):
                    if i < 2:
                        continue
                    if col_name == model_name and i < len(row):
                        score = row[i]
                        model_results.append([task_name, sample_count, score])
                        break

        # Create DataFrame and save
        df_model = pd.DataFrame(model_results, columns=['Task', 'num', 'Score'])
        df_model.to_excel(save_path, sheet_name='Results', index=False)
        df_model.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # Generate detailed report
        generate_detailed_report(model_name, aggregated_results, model_output_dir)

        # If normalized JSON output is enabled, save it
        if args.save_normalized_json:
            # Determine inference file path
            infer_file = infer_files.get(model_name)
            if infer_file:
                # Use new function to save normalized JSON
                json_filename = os.path.splitext(os.path.basename(infer_file))[0] + "_scores.json"
                json_path = os.path.join(model_output_dir, json_filename)
                # Call modified save_normalized_json_output function
                save_normalized_json_output({}, json_path, model_name, infer_file, aggregated_results)

        print(f"Evaluation results for model {model_name} saved to: {model_output_dir}")
        return save_path

    def generate_detailed_report(model_name: str, aggregated_results: Dict, output_dir: str):
        """Generate detailed evaluation report for a single model"""
        report_path = os.path.join(output_dir, f'{model_name}_detailed_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"VLADBench Detailed Evaluation Report - {model_name}\n")
            f.write("=" * 80 + "\n\n")

            # Add evaluation parameter info
            f.write(f"Evaluation parameters:\n")
            f.write(f"  R1-type evaluation: {'Enabled' if args.r1_type else 'Disabled'}\n")
            f.write(f"  Normalized JSON output: {'Enabled' if args.save_normalized_json else 'Disabled'}\n\n")

            model_stats = {
                'total_tasks': 0,
                'total_samples': 0,
                'total_score': 0.0,
                'weighted_score_sum': 0.0
            }

            for (first_task, second_task, third_task), task_data in aggregated_results.items():
                if model_name not in task_data['models']:
                    continue

                task_name = f"{first_task}/{second_task}/{third_task}"

                f.write(f"Task: {task_name}\n")
                f.write(f"Sample count: {task_data['samples']}\n")

                model_result = task_data['models'][model_name]
                f.write(f"  Score: {model_result['score']:.2f}\n")
                f.write(f"  Accuracy: {model_result['accuracy']:.2f}%\n")
                f.write(f"  Instruction compliance: {model_result['obey']:.2f}%\n")
                f.write(f"  Other metrics: {model_result['others']:.2f}%\n")

                # Update model statistics
                model_stats['total_tasks'] += 1
                model_stats['total_samples'] += task_data['samples']
                model_stats['total_score'] += model_result['score']
                model_stats['weighted_score_sum'] += task_data['samples'] * model_result['score']

                f.write("-" * 60 + "\n")

            # Calculate statistics
            if model_stats['total_tasks'] > 0:
                avg_score = model_stats['total_score'] / model_stats['total_tasks']
                weighted_avg = (model_stats['weighted_score_sum'] / model_stats['total_samples']
                              if model_stats['total_samples'] > 0 else 0)

                f.write("\n" + "=" * 80 + "\n")
                f.write("Model Overall Statistics\n")
                f.write("=" * 80 + "\n")
                f.write(f"Evaluated tasks: {model_stats['total_tasks']}\n")
                f.write(f"Total samples: {model_stats['total_samples']}\n")
                f.write(f"Average score: {avg_score:.2f}\n")
                f.write(f"Weighted average score: {weighted_avg:.2f}\n")
            else:
                f.write("No evaluation data for this model.\n")

        print(f"Detailed report saved to: {report_path}")

    def aggregate_first_level_metrics(aggregated_results: Dict, all_tasks: Dict) -> Dict:
        """Aggregate metrics for five top-level tasks"""
        first_level_metrics = {}

        for first_task in all_tasks:
            first_level_metrics[first_task] = {}

            for model in model_names:
                # Initialize statistics
                total_weighted_score = 0.0
                total_samples = 0

                # Iterate through all subtasks under this top-level task
                for second_task in all_tasks[first_task]:
                    if second_task == 'Ego_trajectory_Planning':
                        continue

                    for third_task in all_tasks[first_task][second_task]:
                        task_key = (first_task, second_task, third_task)

                        if task_key in aggregated_results and model in aggregated_results[task_key]['models']:
                            model_result = aggregated_results[task_key]['models'][model]
                            task_samples = aggregated_results[task_key]['samples']

                            # Accumulate weighted score
                            total_weighted_score += model_result['score'] * task_samples
                            total_samples += task_samples

                # Calculate weighted average score for this top-level task
                if total_samples > 0:
                    weighted_avg_score = total_weighted_score / total_samples
                else:
                    weighted_avg_score = 0.0

                first_level_metrics[first_task][model] = {
                    'score': round(weighted_avg_score, 2),
                    'total_samples': total_samples
                }

        return first_level_metrics

    def save_first_level_metrics(first_level_metrics: Dict, output_dir: str):
        """Save top-level task metrics to JSON file"""
        # Prepare data structure for saving
        metrics_data = {}

        for first_task, model_scores in first_level_metrics.items():
            metrics_data[first_task] = {}

            for model_name, scores in model_scores.items():
                metrics_data[first_task][model_name] = {
                    'score': scores['score'],
                    'total_samples': scores['total_samples']
                }

        # Save to JSON file
        output_path = os.path.join(output_dir, "first_level_metrics.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)

        print(f"Top-level task metrics saved to: {output_path}")

        # Also save as Excel format for easy viewing
        excel_path = os.path.join(output_dir, "first_level_metrics.xlsx")
        excel_data = []

        for first_task, model_scores in first_level_metrics.items():
            row = {'First_Level_Task': first_task}

            for model_name, scores in model_scores.items():
                row[f'{model_name}_score'] = scores['score']
                row[f'{model_name}_samples'] = scores['total_samples']

            excel_data.append(row)

        if excel_data:
            df = pd.DataFrame(excel_data)
            df.to_excel(excel_path, index=False)
            print(f"Top-level task metrics Excel saved to: {excel_path}")

        return output_path

    # ======================== Main logic ========================
    all_results = []
    total_results = []
    aggregated_results = {}  # For generating reports

    print(f"Starting VLADBench evaluation, model list: {model_names}")
    print(f"Evaluation results will be saved to: {eval_output_dir}")
    print(f"R1-type evaluation: {'Enabled' if args.r1_type else 'Disabled'}")

    for fir_ind, fir_task in enumerate(all_tasks):
        sec_tasks = all_tasks[fir_task]
        for sec_ind, sec_task in enumerate(sec_tasks):
            if sec_task == 'Ego_trajectory_Planning':
                continue
            third_tasks = sec_tasks[sec_task]
            third_rows = 0

            task_samples_by_model = {model: {} for model in model_names}

            for MODEL in model_names:
                jsonl_path = infer_files[MODEL]
                if not os.path.exists(jsonl_path):
                    print(f"WARNING: {jsonl_path} does not exist, skipping model {MODEL}")
                    continue

                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        item = json.loads(line.strip())

                        # If R1-type is enabled, extract answer from <answer> tag
                        if args.r1_type and "response" in item:
                            # Parse response, extract content from <answer> tag
                            pred_text = item["response"]
                            pred_text = preprocess_model_output(pred_text)
                            item["response"] = extract_answer_from_tag(pred_text)

                        img_path = item["images"][0]["path"]
                        try:
                            third_task = extract_third_task_from_path(img_path)
                        except Exception as e:
                            print(f"Skipping invalid sample: {e}")
                            continue

                        if third_task not in third_tasks:
                            continue

                        if third_task not in task_samples_by_model[MODEL]:
                            task_samples_by_model[MODEL][third_task] = []
                        task_samples_by_model[MODEL][third_task].append(item)

            for third_ind, third_task in enumerate(third_tasks):
                third_rows += 1
                model_scores = [third_task]
                total_num_for_ref = 0

                for MODEL in model_names:
                    samples = task_samples_by_model[MODEL].get(third_task, [])
                    if not samples:
                        print(f"Model {MODEL} has no data for {third_task}")
                        model_scores.append(0)
                        if total_num_for_ref == 0:
                            total_num_for_ref = 0
                        continue

                    data_for_eval = samples
                    ques_total_num, acc, obey, others_metric = func_mapping[third_task](data_for_eval, MODEL)

                    if total_num_for_ref == 0:
                        total_num_for_ref = ques_total_num

                    weight = weights.get(third_task, [0, 0.8, 0.2])
                    temp_score = 100 * others_metric * weight[0] + 100 * acc * weight[1] + 100 * obey * weight[2]

                    # === Record detailed metrics for report ===
                    task_key = (fir_task, sec_task, third_task)
                    if task_key not in aggregated_results:
                        aggregated_results[task_key] = {
                            'samples': total_num_for_ref,
                            'models': {}
                        }
                    aggregated_results[task_key]['models'][MODEL] = {
                        'score': temp_score,
                        'accuracy': acc * 100,
                        'obey': obey * 100,
                        'others': others_metric * 100
                    }
                    # ==============================

                    print(f'Model: {MODEL}, Third Task: {third_task}, Score: {temp_score:.2f}')
                    model_scores.append(temp_score)

                model_scores.insert(1, total_num_for_ref)
                all_results.append(model_scores.copy())
                total_results.append(model_scores.copy())

            all_results = weighted_row_sum(all_results, third_rows)

    # Calculate total score
    total_ = weighted_total(total_results)
    all_results.append(total_)

    # Save summary results for all models
    save_path = os.path.join(eval_output_dir, "all_results_metric.xlsx")
    df = pd.DataFrame(all_results, columns=['Task', 'num'] + model_names)
    df.to_excel(save_path, sheet_name='Sheet1', index=False)
    csv_path = save_path.replace('.xlsx', '.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Summary evaluation complete! Results saved to: {save_path}")

    # Save individual results for each model
    for model_name in model_names:
        try:
            save_model_results(model_name, all_results, aggregated_results)
        except Exception as e:
            print(f"Error saving results for model {model_name}: {e}")

    # Aggregate and save top-level task metrics
    try:
        first_level_metrics = aggregate_first_level_metrics(aggregated_results, all_tasks)
        save_first_level_metrics(first_level_metrics, eval_output_dir)
    except Exception as e:
        print(f"Error aggregating top-level task metrics: {e}")

    # If normalized JSON output is enabled, generate normalized output for each model
    if args.save_normalized_json:
        for model_name in model_names:
            try:
                # Determine inference file path
                infer_file = infer_files.get(model_name)
                if infer_file:
                    # Create model output directory
                    model_output_dir = create_model_output_dirs(model_name)

                    # Generate normalized JSON filename
                    json_filename = os.path.splitext(os.path.basename(infer_file))[0] + "_scores.json"
                    json_path = os.path.join(model_output_dir, json_filename)

                    # Save normalized JSON
                    save_normalized_json_output({}, json_path, model_name, infer_file, aggregated_results)
            except Exception as e:
                print(f"Error generating normalized JSON for model {model_name}: {e}")

if __name__ == "__main__":
    main()
