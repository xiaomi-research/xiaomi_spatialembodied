# UPDATE: Replace placeholder paths with your actual paths.
# Modified 20260302, adapted to add a dataset_path and infer_file_path file for data matching process, only evaluate matched items.
# Matching method: via image file name. This is the original data file {"messages": [{"role": "system", "content": "You are a professional multimodal AI assistant. Watch the video, image or multi-view images and answer questions. Enclose your final answer within <answer> tags, like this: <answer>your answer here</answer>."}, {"role": "user", "content": "<image>\nUnless explicitly stated otherwise, assume you are driving a car in France.\nList all relevant entities from the scene that are necessary to answer the following question, such as road signs, markings, signals, or other vehicles in the image, along with their bounding boxes.Then, select all correct answers to the following question from the available options. Detail your reasoning step by step based on these entities and relevant driving rules. Provide the letters corresponding to your answer in the format: '<answer>Answer(s): letters</answer>'. \nQuestion: The signage indicates that I cannot turn right\nOptions: (A) Yes (B) No."}, {"role": "assistant", "content": "The relevant entities for this problem are: no right turn sign [0.814, 0.055, 0.972, 0.297].\nReasoning: The no right turn sign applies specifically to trucks and vans. Therefore, I can turn right.\nAnswer(s): B."}], "images": "/path/to/DrivingVQA/images/0017.jpg"}
# This is the inference file {"response": "Answer(s): A", "labels": "The relevant entities for this problem are: directional sign [0.634, 0.256, 0.793, 0.465].\nReasoning: The directional sign indicates that it is possible to exit at the 2nd, 3rd, and 4th exits.\nAnswer(s): B, C, D.", "logprobs": null, "messages": [{"role": "user", "content": "<image>\nUnless explicitly stated otherwise, assume you are driving a car in France.\nList all relevant entities from the scene that are necessary to answer the following question, such as road signs, markings, signals, or other vehicles in the image, along with their bounding boxes.Then, select all correct answers to the following question from the available options. Detail your reasoning step by step based on these entities and relevant driving rules. Provide the letters corresponding to your answer in the format: 'Answer(s): <letters>'.\nQuestion: I could take\nOptions: (A) the first exit (B) the second exit (C) the third exit (D) the fourth exit.", "loss": null}, {"role": "assistant", "content": "Answer(s): A"}], "images": [{"bytes": null, "path": "/path/to/DrivingVQA/images/0000.jpg"}]}
import argparse
import os, sys
import json
import re
from datetime import datetime
from tqdm import tqdm
import glob
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
base_model_dir = os.path.dirname(current_dir)  # Since current_dir is .../<repo-root>/eval
sys.path.append(base_model_dir)
try:
    from eval.eval_utils.preprocessor import preprocess_model_output
except ImportError:
    # Fallback for when preprocessor is not available
    def preprocess_model_output(text):
        return text

def extract_answer_from_tag(text):
    """Extract answer content from <answer> tags - used for R1-type evaluation"""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def extract_result(text: str, r1_type: bool = False):
    """
    Extract answer from text, adapted to the extract_result function in your metrics.py

    Args:
        text: Text to extract answer from
        r1_type: Whether this is R1-type evaluation, if so extract from <answer> tags
    """
    if not text:
        return []

    # If R1-type evaluation, first try to extract from <answer> tags
    if r1_type:
        text = preprocess_model_output(text)
        extracted_text = extract_answer_from_tag(text)
        if extracted_text != text.strip():
            # If successfully extracted from tags, use the extracted text
            text = extracted_text

    # Regex pattern to match answer prefixes and extract answer characters
    answer_prefix_pattern = r"(Answer\(s\)?|Answers?)[:\s]*([A-D\d\(\), ]*)"
    answer_text = " ".join(
        match[1].strip().rstrip('.') for match in re.findall(answer_prefix_pattern, text, re.IGNORECASE))

    # Extract uppercase letters A-D or numbers in parentheses, deduplicate and sort
    results = sorted(set(re.findall(r"[A-D]|\(\d+\)", answer_text)))
    return results

def compute_scores_directly(preds, true_answers):
    """
    Compute evaluation scores directly, without going through files
    """
    exact_matches = 0
    true_positives, false_positives, false_negatives = 0, 0, 0

    for qid, pred in preds.items():
        true_set = set(true_answers[qid])
        pred_set = set(pred)

        # Only non-empty and exact match counts as correct
        if len(pred) != 0 and pred_set == true_set:
            exact_matches += 1

        # Empty predictions are not counted in precision/recall
        if pred_set or true_set:
            true_positives += len(true_set & pred_set)
            false_positives += len(pred_set - true_set)
            false_negatives += len(true_set - pred_set)

    total_non_empty = len([p for p in preds.values() if p])
    exam_score = 100 * exact_matches / len(preds) if preds else 0

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "exam_score": exam_score,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_score": f1_score * 100
    }

def extract_image_filename(image_data):
    """
    Extract image file name (compatible with two formats)
    Args:
        image_data: Image path data, supports two formats:
                    1. String format: "/path/0017.jpg"
                    2. List dict format: [{"path": "/path/0000.jpg"}]
    Returns:
        str: Image file name (e.g. "0017.jpg"), returns empty string if extraction fails
    """
    if not image_data:
        return ""

    # Handle string format (original dataset format)
    if isinstance(image_data, str):
        return os.path.basename(image_data.strip())

    # Handle list dict format (inference result format)
    elif isinstance(image_data, list) and len(image_data) > 0:
        for item in image_data:
            if isinstance(item, dict) and "path" in item:
                return os.path.basename(item["path"].strip())

    return ""

def load_dataset_data(dataset_path):
    """Load original dataset and extract image file name set"""
    dataset_image_names = set()

    if not dataset_path or not os.path.exists(dataset_path):
        print(f"Warning: Dataset path {dataset_path} not found, will use all inference data for evaluation")
        return dataset_image_names

    data = []
    if dataset_path.endswith('.json'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif dataset_path.endswith('.jsonl'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path} (only json/jsonl supported)")

    # Extract all image file names
    for item in data:
        if "images" in item:
            img_name = extract_image_filename(item["images"])
            if img_name:
                dataset_image_names.add(img_name)

    print(f"Loaded {len(dataset_image_names)} unique image file names from dataset")
    return dataset_image_names

def load_infer_data(infer_path, dataset_image_names=None):
    """
    Load data from infer file, supports multiple formats, and filters based on dataset image names
    Args:
        infer_path: Infer file path/directory
        dataset_image_names: Dataset image file name set for filtering
    Returns:
        tuple: (filtered inference data, original inference data count, matched count)
    """
    all_infer_data = []

    if os.path.isfile(infer_path):
        if infer_path.endswith('.json'):
            with open(infer_path, 'r', encoding='utf-8') as f:
                all_infer_data = json.load(f)
        elif infer_path.endswith('.jsonl'):
            with open(infer_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_infer_data.append(json.loads(line.strip()))
    elif os.path.isdir(infer_path):
        jsonl_files = glob.glob(os.path.join(infer_path, "*.jsonl"))
        jsonl_files.sort()
        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data_item = json.loads(line.strip())
                        all_infer_data.append(data_item)
    else:
        raise ValueError(f"Cannot find file or directory: {infer_path}")

    original_count = len(all_infer_data)

    # If dataset image name set is provided, filter
    filtered_data = []
    if dataset_image_names and len(dataset_image_names) > 0:
        for item in all_infer_data:
            # Extract image file name from inference data
            img_name = extract_image_filename(item.get("images", ""))
            if img_name and img_name in dataset_image_names:
                filtered_data.append(item)
    else:
        # No dataset matching criteria, use all data
        filtered_data = all_infer_data

    matched_count = len(filtered_data)
    print(f"Inference data total: {original_count}, matched count: {matched_count}")

    return filtered_data, original_count, matched_count

def eval_from_infer(infer_path, output_dir, model_name, dataset_path=None, r1_type=False):
    """Evaluate model output from infer file (with image file name matching logic added)

    Args:
        infer_path: Infer file path
        output_dir: Output directory
        model_name: Model name
        dataset_path: Original dataset path
        r1_type: Whether to use R1-type evaluation
    """

    # 1. Load dataset and extract image file name set
    dataset_image_names = load_dataset_data(dataset_path)

    # 2. Load and filter inference data (only keep matched items)
    print(f"Loading infer data: {infer_path}")
    infer_data, original_infer_count, matched_count = load_infer_data(infer_path, dataset_image_names)

    if len(infer_data) == 0:
        print("Error: No matching inference data loaded!")
        return None

    if r1_type:
        print(f"Using R1-type evaluation mode, will extract answers from <answer> tags")

    # Prepare storage for prediction results
    all_preds = {}
    all_true_answers = {}
    detailed_results = []

    # Statistics (updated to include matching-related statistics)
    stats = {
        "original_total": original_infer_count,
        "matched_total": len(infer_data),
        "no_response": 0,
        "no_labels": 0,
        "parsing_error": 0
    }

    # Process each matched data item
    for idx, item in enumerate(tqdm(infer_data, desc="Processing matched data")):
        # Generate ID (include image name for tracking)
        img_name = extract_image_filename(item.get("images", ""))
        item_id = f"{img_name}_{idx:06d}" if img_name else f"item_{idx:06d}"

        # Get model response and ground truth answer
        response = item.get("response", "")
        labels = item.get("labels", "")

        if not response:
            stats["no_response"] += 1
            continue

        if not labels:
            stats["no_labels"] += 1
            continue

        # Extract results from model response
        try:
            pred_answer = extract_result(response, r1_type)
        except Exception as e:
            print(f"Error extracting predicted answer (ID: {item_id}): {e}")
            stats["parsing_error"] += 1
            pred_answer = []

        # Extract results from labels
        try:
            true_answer = extract_result(labels, False)
        except Exception as e:
            print(f"Error extracting ground truth answer (ID: {item_id}): {e}")
            stats["parsing_error"] += 1
            true_answer = []

        # Store results
        all_preds[item_id] = pred_answer
        all_true_answers[item_id] = true_answer

        # Save detailed results
        detailed_results.append({
            "id": item_id,
            "image_name": img_name,
            "response": response,
            "labels": labels,
            "pred_answer": pred_answer,
            "true_answer": true_answer,
            "is_correct": set(pred_answer) == set(true_answer) if len(pred_answer)!=0 else 0
        })

    # Print statistics
    print("\n" + "="*50)
    print("Data statistics")
    print("="*50)
    print(f"Original inference data total: {stats['original_total']}")
    print(f"Matched data count: {stats['matched_total']}")
    print(f"Valid evaluation data count: {len(all_preds)}")
    print(f"No response field: {stats['no_response']}")
    print(f"No labels field: {stats['no_labels']}")
    print(f"Parsing errors: {stats['parsing_error']}")
    if r1_type:
        print(f"Evaluation mode: R1-type (extract answer from <answer> tags)")
    else:
        print(f"Evaluation mode: Standard mode")
    print("="*50)

    if len(all_preds) == 0:
        print("Error: No valid prediction data!")
        return None

    # Compute evaluation scores
    print("\nComputing evaluation scores...")
    scores = compute_scores_directly(all_preds, all_true_answers)

    # Add statistics to scores
    scores.update({
        "original_total_samples": stats['original_total'],
        "matched_total_samples": stats['matched_total'],
        "valid_samples": len(all_preds),
        "invalid_samples": stats['no_response'] + stats['no_labels'] + stats['parsing_error']
    })

    # Compute accuracy statistics
    correct_count = sum(1 for item in detailed_results if item["is_correct"])
    accuracy = correct_count / len(detailed_results) * 100 if detailed_results else 0

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create model name subdirectory
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Build unified format output results
    if os.path.isfile(infer_path):
        input_filename = Path(infer_path).stem
    else:
        input_filename = Path(infer_path).name

    # Build standardized output result structure
    standard_output = {
        "overall": {
            "metric": "accuracy",
            "score": scores.get("exam_score", 0),
        },
        "secondary_metrics": {
            "precision": scores.get("precision", 0),
            "recall": scores.get("recall", 0),
            "f1_score": scores.get("f1_score", 0),
            "accuracy": accuracy,
            "original_total_samples": scores.get("original_total_samples", 0),
            "matched_total_samples": scores.get("matched_total_samples", 0),
            "valid_samples": scores.get("valid_samples", 0),
            "invalid_samples": scores.get("invalid_samples", 0),
            "correct_count": correct_count
        },
        "evaluation_config": {
            "r1_type": r1_type,
            "dataset_path": dataset_path,
            "infer_path": infer_path,
            "evaluation_time": timestamp
        },
        "detailed_results": detailed_results
    }

    # Save as unified format JSON file
    unified_output_path = os.path.join(model_output_dir, f"{input_filename}_scores.json")
    with open(unified_output_path, 'w', encoding='utf-8') as f:
        json.dump(standard_output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved unified format results to: {unified_output_path}")

    # Save original format results (for backward compatibility)
    base_output_path = os.path.join(model_output_dir, f"eval_{timestamp}")

    def save_json(data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved original format results to: {filename}")

    # Save original format results
    predictions_path = f"{base_output_path}_predictions.json"
    true_answers_path = f"{base_output_path}_true_answers.json"
    detailed_path = f"{base_output_path}_detailed_results.json"
    scores_path = f"{base_output_path}_scores.json"

    save_json(all_preds, predictions_path)
    save_json(all_true_answers, true_answers_path)
    save_json(detailed_results, detailed_path)
    save_json(scores, scores_path)

    # Print evaluation results
    print("\n" + "="*50)
    print("Evaluation results summary")
    print("="*50)
    print(f"Data matching: original {stats['original_total']} items -> matched {stats['matched_total']} items")
    if r1_type:
        print(f"Evaluation mode: R1-type (extract answer from <answer> tags)")
    print(f"Exam Score (subset accuracy): {scores.get('exam_score', 0):.2f}%")
    print(f"Precision: {scores.get('precision', 0):.2f}%")
    print(f"Recall: {scores.get('recall', 0):.2f}%")
    print(f"F1 Score: {scores.get('f1_score', 0):.2f}%")
    print(f"Correct answer count: {correct_count} / {len(detailed_results)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Valid evaluation sample count: {scores.get('valid_samples', 0)}")
    print("="*50)

    # Create brief summary file
    summary_path = os.path.join(model_output_dir, "latest_eval_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write(f"Evaluation results summary - {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset path: {dataset_path}\n")
        f.write(f"Inference data source: {infer_path}\n")
        f.write(f"Evaluation mode: {'R1-type (extract answer from <answer> tags)' if r1_type else 'Standard mode'}\n")
        f.write(f"\nData matching statistics:\n")
        f.write(f"  Original inference data total: {stats['original_total']}\n")
        f.write(f"  Matched data count: {stats['matched_total']}\n")
        f.write(f"  Valid evaluation data count: {len(all_preds)}\n")
        f.write(f"  Invalid data count: {stats['no_response'] + stats['no_labels'] + stats['parsing_error']}\n")
        f.write(f"\nEvaluation metrics:\n")
        f.write(f"  Exam Score (subset accuracy): {scores.get('exam_score', 0):.2f}%\n")
        f.write(f"  Precision: {scores.get('precision', 0):.2f}%\n")
        f.write(f"  Recall: {scores.get('recall', 0):.2f}%\n")
        f.write(f"  F1 Score: {scores.get('f1_score', 0):.2f}%\n")
        f.write(f"  Correct answer count: {correct_count} / {len(detailed_results)}\n")
        f.write(f"  Accuracy: {accuracy:.2f}%\n")
        f.write(f"\nOutput files:\n")
        f.write(f"  Unified output file: {unified_output_path}\n")
        f.write(f"  Predictions: {predictions_path}\n")
        f.write(f"  True answers: {true_answers_path}\n")
        f.write(f"  Detailed results: {detailed_path}\n")
        f.write(f"  Evaluation scores: {scores_path}\n")
        f.write("="*50 + "\n")

    print(f"\nEvaluation complete! Results saved to: {model_output_dir}")
    print(f"Unified format results: {unified_output_path}")
    print(f"Detailed summary: {summary_path}")

    return standard_output

def main():
    parser = argparse.ArgumentParser(description="Evaluate model output from infer file (supports image file name matching)")
    parser.add_argument(
        "--infer_file_path",
        type=str,
        required=True,
        help="Infer file path or directory containing JSONL files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output results directory (default: ./eval_results)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Original dataset path (JSON/JSONL format) for image file name matching"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g. Qwen2.5-VL-3B)"
    )
    parser.add_argument(
        "--r1_type",
        action='store_true',
        help="Use R1-type evaluation mode, extract answers from <answer> tags"
    )

    args = parser.parse_args()

    # Run evaluation
    scores = eval_from_infer(
        args.infer_file_path,
        args.output_dir,
        args.model_name,
        args.dataset_path,
        args.r1_type
    )

    if scores:
        print("\n" + "="*50)
        print("Evaluation complete! Core results:")
        print("="*50)
        print(f"Matched data count: {scores['secondary_metrics']['matched_total_samples']}")
        print(f"Exam Score: {scores['overall']['score']:.2f}%")
        print(f"F1 Score: {scores['secondary_metrics']['f1_score']:.2f}%")
        print(f"Accuracy: {scores['secondary_metrics']['accuracy']:.2f}%")
        print(f"Results file: {os.path.join(args.output_dir, args.model_name, 'latest_eval_summary.txt')}")

if __name__ == "__main__":
    main()
