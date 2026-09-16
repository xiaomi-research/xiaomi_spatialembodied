# UPDATE: Replace placeholder paths with your actual paths.
import json
import torch
import pandas as pd
import click
import os
import re
from pathlib import Path
from datasets import Dataset
from functools import partial
from typing import Optional

import sys
sys.path.append("/path/to/LingoQA-main/benchmark")
current_dir = os.path.dirname(os.path.abspath(__file__))
base_model_dir = os.path.dirname(current_dir)  # Since current_dir is .../<repo-root>/eval
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

# Import from constants
try:
    from constants import Keys
except ImportError:
    class Keys:
        QUESTION = "question"
        RESPONSE = "response"
        LABELS = "labels"
        MESSAGES = "messages"

try:
    from judge import LingoJudge
except ImportError:
    # `judge.py` ships with the external LingoQA benchmark repo, not with this project.
    #
    # There used to be a fallback here that returned a constant 0.5 for every sample. That is
    # worse than having no scorer at all: scores > 0.0 are counted as correct a few lines
    # below, so the reported accuracy came out at exactly 100% while the output file still
    # looked well-formed. Fail loudly instead.
    class LingoJudge:
        def __init__(self):
            raise RuntimeError(
                "LingoJudge is unavailable, so LingoQA cannot be scored.\n"
                "  `judge.py` is not part of this repository. Clone the official LingoQA\n"
                "  benchmark (https://github.com/wayveai/LingoQA) and add its `benchmark`\n"
                "  directory to the `sys.path` line at the top of this file.\n"
                "  Refusing to emit scores, because the previous constant-0.5 fallback\n"
                "  silently reported 100% accuracy."
            )


def extract_answer_from_tag(text):
    """Extract answer content from <answer> tags"""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


@click.command()
@click.option('--infer_file_path', required=True, help='Path to inference jsonl file containing response and labels.')
@click.option('--output_dir', required=True, help='Output directory for evaluation results.')
@click.option('--dataset_path', help='Path to original dataset for meta information (optional).')
@click.option('--model_name', required=True, help='Name of the model being evaluated.')
@click.option('--batch_size', default=1, help='Batch size for evaluation.')
@click.option('--sample_size', default=0, help='Number of samples to evaluate (0 for all).')
@click.option('--r1_type', is_flag=True, default=False, help='Whether to use R1-type evaluation (extract answer from <answer> tags).')
def evaluate(infer_file_path: str, output_dir: str, dataset_path: Optional[str] = None,
             model_name: str = None, batch_size: int = 1, sample_size: int = 0,
             r1_type: bool = False) -> None:
    """
    Evaluate LingoQA inference results

    Args:
        infer_file_path: Path to the .jsonl file containing model predictions and ground truth
        output_dir: Output directory path
        dataset_path: Original dataset path (for meta information, optional)
        model_name: Model name
        batch_size: Evaluation batch size
        sample_size: Number of evaluation samples (0 means all)
        r1_type: Whether to use R1-type evaluation (extract answers from <answer> tags)
    """
    # Validate input file
    if not os.path.exists(infer_file_path):
        print(f"Error: Input file {infer_file_path} does not exist")
        return

    print(f"Using R1-type evaluation: {r1_type}")

    # Read and process jsonl data
    predictions = _process_jsonl_predictions(infer_file_path, sample_size, r1_type)

    if len(predictions) == 0:
        print("Error: No valid data loaded")
        return

    print(f"Loaded {len(predictions)} prediction samples")
    print(f"Model name: {model_name}")

    # Convert to Dataset
    dataset = Dataset.from_pandas(predictions)

    # Initialize evaluator
    try:
        if torch.cuda.is_available():
            judge = LingoJudge().eval().to("cuda:0")
            print("Using GPU for evaluation")
        else:
            judge = LingoJudge().eval()
            print("Using CPU for evaluation")
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        return

    # Execute evaluation
    try:
        dataset_evaluated = dataset.map(
            partial(_evaluate_batch, judge),
            batched=True,
            batch_size=batch_size
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return

    # Filter correct predictions
    dataset_filtered = dataset_evaluated.filter(_is_correct_prediction)

    # Calculate scores
    total_samples = dataset_evaluated.num_rows
    correct_samples = dataset_filtered.num_rows
    accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

    # Calculate secondary metrics
    secondary_metrics = _calculate_secondary_metrics(predictions, dataset_evaluated)

    # Generate output results
    output_data = _generate_output_format(
        accuracy=accuracy,
        total_samples=total_samples,
        correct_samples=correct_samples,
        secondary_metrics=secondary_metrics,
        detailed_results=predictions,
        dataset=dataset_evaluated,
        model_name=model_name,
        infer_file_path=infer_file_path,
        r1_type=r1_type
    )

    # Save results
    _save_results(output_data, output_dir, model_name, infer_file_path)

    # Print summary
    _print_evaluation_summary(total_samples, correct_samples, accuracy)


def _process_jsonl_predictions(jsonl_path: str, sample_size: int = 0, r1_type: bool = False) -> pd.DataFrame:
    """
    Process jsonl format prediction results

    Args:
        jsonl_path: jsonl file path
        sample_size: Sample size limit
        r1_type: Whether to use R1-type evaluation (extract answers from <answer> tags)

    Returns:
        DataFrame containing prediction data
    """
    data_records = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        # Limit sample size
        if sample_size > 0:
            lines = lines[:sample_size]

        for line_idx, line in enumerate(lines):
            try:
                item = json.loads(line.strip())

                # Extract question text
                question_text = _extract_question(item.get("messages", []))

                # Get raw prediction
                raw_prediction = item.get("response", "")

                # If using R1-type evaluation, extract answer from <answer> tags
                if r1_type:
                    raw_prediction = preprocess_model_output(raw_prediction)
                    extracted_answer = extract_answer_from_tag(raw_prediction)
                else:
                    extracted_answer = raw_prediction

                # Build data record
                record = {
                    "id": line_idx,  # Use line number as ID
                    "question": question_text,
                    "prediction": extracted_answer,
                    "reference": item.get("labels", ""),
                    "raw_response": raw_prediction,  # Save raw response
                }

                data_records.append(record)

            except json.JSONDecodeError as e:
                print(f"Warning: Error parsing line {line_idx+1}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_idx+1}: {e}")
                continue

    return pd.DataFrame(data_records)


def _extract_question(messages: list) -> str:
    """
    Extract question from message list

    Args:
        messages: Message list

    Returns:
        Extracted question text
    """
    if not messages:
        return ""

    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")

            # Remove image markers
            while "<image>" in content:
                content = content.replace("<image>", "")

            # Remove common system prompt prefixes
            prefixes_to_remove = [
                "You are a driver. These are five continuous front view images.",
                "You are a driver. These are five continuous front view images. Answer the following question: ",
                "Answer the following question: "
            ]

            for prefix in prefixes_to_remove:
                if content.startswith(prefix):
                    content = content[len(prefix):].lstrip()

            return content.strip()

    return ""


def _evaluate_batch(metric: LingoJudge, batch: dict) -> dict:
    """
    Batch evaluation function

    Args:
        metric: Evaluator
        batch: Batch data

    Returns:
        Batch data containing evaluation results
    """
    questions = batch.get("question", [])
    references = [[ref] for ref in batch.get("reference", [])]  # Convert to list of lists
    predictions = batch.get("prediction", [])

    if not questions or not references or not predictions:
        return batch

    # Calculate scores
    scores = metric.compute(questions, references, predictions)

    # Fix warning: use detach().clone() instead of torch.tensor()
    if isinstance(scores, torch.Tensor):
        # If scores is a tensor, convert to Python list
        scores_list = scores.detach().cpu().tolist()
        # Calculate probabilities
        probs = torch.sigmoid(scores).detach().cpu().tolist()
    else:
        # If scores is already a list
        scores_list = scores
        probs = [1.0 / (1.0 + torch.exp(-torch.tensor(score))) for score in scores_list]

    # Add evaluation results
    batch["score"] = scores_list
    batch["probability"] = probs
    batch["correct"] = [score > 0.0 for score in scores_list]

    return batch


def _is_correct_prediction(example: dict) -> bool:
    """
    Determine whether prediction is correct

    Args:
        example: Sample data

    Returns:
        Whether correct
    """
    return example.get("correct", False)


def _calculate_secondary_metrics(predictions_df: pd.DataFrame, dataset) -> dict:
    """
    Calculate secondary metrics

    Args:
        predictions_df: Prediction data DataFrame
        dataset: Evaluated dataset

    Returns:
        Secondary metrics dictionary
    """
    metrics = {}

    # Calculate score statistics
    if "score" in dataset.column_names:
        scores = dataset["score"]
        if scores:
            scores_numeric = [float(s) for s in scores if s is not None]
            if scores_numeric:
                metrics["score_mean"] = float(sum(scores_numeric) / len(scores_numeric))
                metrics["score_max"] = float(max(scores_numeric))
                metrics["score_min"] = float(min(scores_numeric))
                metrics["score_std"] = float(pd.Series(scores_numeric).std() if len(scores_numeric) > 1 else 0.0)

    # Calculate probability statistics
    if "probability" in dataset.column_names:
        probs = dataset["probability"]
        if probs:
            probs_numeric = [float(p) for p in probs if p is not None]
            if probs_numeric:
                metrics["probability_mean"] = float(sum(probs_numeric) / len(probs_numeric))
                metrics["probability_max"] = float(max(probs_numeric))
                metrics["probability_min"] = float(min(probs_numeric))

    # Calculate correct/incorrect distribution
    if "correct" in dataset.column_names:
        correct_list = dataset["correct"]
        if correct_list:
            correct_count = sum(correct_list)
            total_count = len(correct_list)
            metrics["correct_count"] = correct_count
            metrics["incorrect_count"] = total_count - correct_count

    return metrics


def _generate_output_format(accuracy: float, total_samples: int, correct_samples: int,
                           secondary_metrics: dict, detailed_results: pd.DataFrame,
                           dataset, model_name: str, infer_file_path: str,
                           r1_type: bool = False) -> dict:
    """
    Generate standard format output results

    Args:
        accuracy: Overall accuracy
        total_samples: Total sample count
        correct_samples: Correct sample count
        secondary_metrics: Secondary metrics
        detailed_results: Detailed results DataFrame
        dataset: Evaluated dataset
        model_name: Model name
        infer_file_path: Input file path
        r1_type: Whether to use R1-type evaluation

    Returns:
        Standard format output dictionary
    """
    # Prepare detailed results list
    detailed_list = []

    # Ensure columns exist
    score_col = dataset["score"] if "score" in dataset.column_names else [None] * len(detailed_results)
    prob_col = dataset["probability"] if "probability" in dataset.column_names else [None] * len(detailed_results)
    correct_col = dataset["correct"] if "correct" in dataset.column_names else [None] * len(detailed_results)

    for idx, row in detailed_results.iterrows():
        detailed_item = {
            "id": int(row["id"]),
            "question": str(row["question"]),
            "prediction": str(row["prediction"]),
            "reference": str(row["reference"]),
            "raw_response": str(row.get("raw_response", ""))
        }

        if idx < len(score_col) and score_col[idx] is not None:
            detailed_item["score"] = float(score_col[idx])

        if idx < len(prob_col) and prob_col[idx] is not None:
            detailed_item["probability"] = float(prob_col[idx])

        if idx < len(correct_col) and correct_col[idx] is not None:
            detailed_item["correct"] = bool(correct_col[idx])

        detailed_list.append(detailed_item)

    # Build standard three-layer output format
    output_data = {
        "overall": {
            "score": float(accuracy),
            "metric": "accuracy",
            "accuracy": float(accuracy),
            "total_samples": int(total_samples),
            "correct_samples": int(correct_samples),
            "incorrect_samples": int(total_samples - correct_samples)
        },
        "secondary_metrics": secondary_metrics,
        "detailed_results": detailed_list,
        "metadata": {
            "model_name": model_name,
            "infer_file": os.path.basename(infer_file_path),
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
            "r1_type_evaluation": r1_type
        }
    }

    return output_data


def _save_results(output_data: dict, output_dir: str, model_name: str, infer_file_path: str) -> None:
    """
    Save evaluation results

    Args:
        output_data: Output data
        output_dir: Output directory
        model_name: Model name
        infer_file_path: Input file path
    """
    # Create model name subdirectory
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Generate output filename (use basename of input JSONL file, add _scores.json suffix)
    infer_basename = os.path.splitext(os.path.basename(infer_file_path))[0]
    output_filename = f"{infer_basename}_scores.json"
    output_path = os.path.join(model_output_dir, output_filename)

    # Save JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation results saved to: {output_path}")

    # Also save a more readable CSV version
    csv_path = os.path.join(model_output_dir, f"{infer_basename}_detailed.csv")
    detailed_df = pd.DataFrame(output_data["detailed_results"])
    detailed_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Detailed CSV results saved to: {csv_path}")

    # Save statistics summary
    summary_path = os.path.join(model_output_dir, f"{infer_basename}_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("LINGOQA Evaluation Results Summary\n")
        f.write("="*60 + "\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"Evaluation file: {os.path.basename(infer_file_path)}\n")
        f.write(f"R1-type evaluation: {output_data['metadata']['r1_type_evaluation']}\n")
        f.write(f"Evaluation time: {output_data['metadata']['evaluation_timestamp']}\n")
        f.write("-"*60 + "\n")
        f.write("Overall metrics:\n")
        f.write(f"  Accuracy: {output_data['overall']['accuracy']*100:.2f}%\n")
        f.write(f"  Total samples: {output_data['overall']['total_samples']}\n")
        f.write(f"  Correct predictions: {output_data['overall']['correct_samples']}\n")
        f.write(f"  Incorrect predictions: {output_data['overall']['incorrect_samples']}\n")
        f.write("-"*60 + "\n")
        f.write("Secondary metrics:\n")
        for key, value in output_data['secondary_metrics'].items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("="*60 + "\n")

    print(f"Evaluation summary saved to: {summary_path}")
    print(f"All results saved in: {model_output_dir}")


def _print_evaluation_summary(total: int, correct: int, accuracy: float):
    """
    Print evaluation summary

    Args:
        total: Total sample count
        correct: Correct sample count
        accuracy: Evaluation score
    """
    print("\n" + "="*60)
    print("LINGOQA Evaluation Results Summary")
    print("="*60)
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    evaluate()
