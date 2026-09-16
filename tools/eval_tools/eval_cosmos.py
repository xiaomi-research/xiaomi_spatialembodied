#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific permission governing permissions and
# limitations under the License.

# /// script
# requires-python = ">=3.10"
# dependencies = []
# [tool.uv]
# exclude-newer = "2025-08-05T00:00:00Z"
# ///

import argparse
import json
import os
import sys
import re
from typing import Dict, List, Any
import warnings

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
# breakpoint()
def extract_answer_from_tag(text: str) -> str:
    """Extract answer content from <answer> tags"""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def parse_option_mapping(question_text: str) -> Dict[str, str]:
    """
    Parse the semantics corresponding to A and B from question text ("yes" or "no").
    Supports multiple format variants, such as "A: yes", "A. yes", "A - yes", etc.
    """
    if not question_text:
        warnings.warn("Empty question text, using default mapping A:yes, B:no")
        return {"A": "yes", "B": "no"}
    # breakpoint()
    # Match A/B lines (supports multiple delimiters: : . - space)
    a_match = re.search(r'^\s*[Aa][\s.:—-]+\s*(\w+)', question_text, re.MULTILINE | re.IGNORECASE)
    b_match = re.search(r'^\s*[Bb][\s.:—-]+\s*(\w+)', question_text, re.MULTILINE | re.IGNORECASE)

    a_val = a_match.group(1).strip().lower() if a_match else "yes"
    b_val = b_match.group(1).strip().lower() if b_match else "no"

    def normalize(val: str) -> str:
        val_lower = val.lower().strip('.,;:!?"\'')
        # Map common variants to standard semantics
        yes_variants = {"yes", "true", "correct", "success", "successful", "completed", "achieved", "y"}
        no_variants = {"no", "false", "incorrect", "failure", "unsuccessful", "incomplete", "failed", "n"}

        if val_lower in yes_variants:
            return "yes"
        elif val_lower in no_variants:
            return "no"
        else:
            # Try fuzzy matching when unrecognized
            if "yes" in val_lower or "succ" in val_lower or "compl" in val_lower:
                return "yes"
            elif "no" in val_lower or "fail" in val_lower or "inc" in val_lower:
                return "no"
            return val_lower  # Keep original value for later processing

    return {
        "A": normalize(a_val),
        "B": normalize(b_val)
    }

def map_letter_to_semantic(letter: str, option_map: Dict[str, str]) -> str:
    """Map option letter to semantics (yes/no)"""
    letter_clean = letter.strip().upper()
    if letter_clean in option_map:
        return option_map[letter_clean]
    return ""

def map_response_to_semantic(response: str, option_map: Dict[str, str], r1_type: bool = False) -> str:
    """
    Map model response to standardized semantic values ("yes" or "no").
    Handles multiple response formats:
      - Direct "yes"/"no" output (including case, punctuation)
      - "A"/"B" output (including punctuation, spaces)
      - R1 format: <answer>no</answer>
      - Mixed format: "The answer is B", etc.
    """
    if not isinstance(response, str) or not response.strip():
        return ""

    # R1 type: first extract <answer> tag content
    if r1_type:
        response = preprocess_model_output(response)
        response = extract_answer_from_tag(response)

    clean_resp = response.strip()

    # Case 1: Explicit yes/no (ignoring case, surrounding punctuation and spaces)
    if re.fullmatch(r'(?i)\s*(yes|no)[\s.!?,;:]*', clean_resp):
        return "yes" if clean_resp.lower().startswith('y') else "no"

    # Case 2: Single letter A/B (with possible punctuation)
    if re.fullmatch(r'(?i)\s*[AB][\s.!?,;:]*', clean_resp):
        letter = clean_resp.upper().strip(' .,;:!?"\'')
        return option_map.get(letter, "")

    # Case 3: A: / B: format
    if re.match(r'(?i)^\s*[AB]\s*[:.-]', clean_resp):
        letter = clean_resp[0].upper()
        return option_map.get(letter, "")

    # Case 4: Extract first valid letter (A/B) from text
    letter_match = re.search(r'(?i)\b([AB])\b', clean_resp)
    if letter_match:
        return option_map.get(letter_match.group(1).upper(), "")

    # Case 5: Try matching yes/no words (in sentences)
    if re.search(r'(?i)\b(yes|yeah|yup|correct|true)\b', clean_resp):
        return "yes"
    if re.search(r'(?i)\b(no|nope|incorrect|false)\b', clean_resp):
        return "no"

    return ""

def load_dataset_metadata(dataset_path: str) -> List[Dict[str, Any]]:
    """Load original dataset metadata."""
    metadata = []
    if dataset_path and os.path.exists(dataset_path):
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        metadata.append(json.loads(line))
        except Exception as e:
            print(f"Warning: Failed to load dataset metadata: {e}", file=sys.stderr)
    return metadata

def process_single_item(item: Dict[str, Any], idx: int, dataset_metadata: List[Dict[str, Any]], r1_type: bool = False) -> Dict[str, Any]:
    """Process a single data item, evaluating based on semantics (yes/no)."""
    response = item.get("response", "")
    label_letter = str(item.get("labels", "")).strip()

    # Get question text (for parsing option semantics)
    question_text = ""
    if idx < len(dataset_metadata) and dataset_metadata:
        meta = dataset_metadata[idx]
        messages = meta.get("messages", [])
        if messages and len(messages) > 0:
            # Usually the last message is the user question
            for conv in messages:
                if conv['role'] == 'user':
                    question_text = conv.get("content", "")

    # breakpoint()
    # Parse option semantic mapping
    option_map = parse_option_mapping(question_text)

    # Convert label (letter) to semantics
    true_semantic = map_letter_to_semantic(label_letter, option_map)

    # Convert prediction to semantics
    pred_semantic = map_response_to_semantic(response, option_map, r1_type)

    # Determine correctness (only compare when both are valid yes/no)
    is_correct = (pred_semantic == true_semantic and pred_semantic in ("yes", "no"))

    # Numerical: yes=1 (positive class), no=0 (negative class), invalid=-1
    label_numeric = 1 if true_semantic == "yes" else 0 if true_semantic == "no" else -1
    pred_numeric = 1 if pred_semantic == "yes" else 0 if pred_semantic == "no" else -1

    result = {
        "index": idx,
        "response": response,
        "label_letter": label_letter,
        "prediction_semantic": pred_semantic,
        "label_semantic": true_semantic,
        "is_correct": is_correct,
        "option_mapping": option_map,
        "question_snippet": question_text[:150] + "..." if len(question_text) > 150 else question_text,
        "label_numeric": label_numeric,
        "prediction_numeric": pred_numeric,
        "raw_prediction_letter": extract_prediction_from_response(response, r1_type=False)  # Keep original letter extraction
    }

    # Add metadata (for debugging and analysis)
    if idx < len(dataset_metadata) and dataset_metadata:
        meta = dataset_metadata[idx]
        result["metadata"] = {
            "video_path": meta.get("videos", [""])[0] if meta.get("videos") else "",
            "full_question": question_text
        }

    # Log parsing issues (for debugging)
    if true_semantic not in ("yes", "no"):
        result["warning"] = f"Invalid label semantic: '{true_semantic}' from label '{label_letter}'"
    if pred_semantic not in ("yes", "no") and response.strip():
        result["warning"] = result.get("warning", "") + f" | Invalid prediction semantic: '{pred_semantic}' from response '{response[:50]}'"

    return result

def extract_prediction_from_response(response: str, r1_type: bool = False) -> str:
    """Keep original letter extraction logic (for debugging and compatibility)"""
    if not response or not isinstance(response, str):
        return ""

    if r1_type:
        response = preprocess_model_output(response)
        response = extract_answer_from_tag(response)

    match = re.match(r'^\s*([A-Z])(?:\s*:|\s* $ )', response)
    if match:
        return match.group(1)

    response_upper = response.strip().upper()
    if response_upper.startswith(('A:', 'B:', 'C:', 'D:')):
        return response_upper[0]

    if response_upper in ['A', 'B', 'C', 'D']:
        return response_upper

    return ""

def calculate_overall_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate overall accuracy (only counting valid samples)"""
    valid_results = [r for r in results if r["label_semantic"] in ("yes", "no")]
    if not valid_results:
        return {"score": 0.0, "metric": "accuracy", "total_samples": 0.0, "correct_samples": 0.0, "valid_samples": 0.0}

    total = len(valid_results)
    correct = sum(1 for r in valid_results if r["is_correct"])
    accuracy = float(correct) / total if total > 0 else 0.0

    return {
        "score": accuracy,
        "metric": "accuracy",
        "total_samples": float(len(results)),
        "valid_samples": float(total),
        "correct_samples": float(correct)
    }

def calculate_secondary_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate secondary metrics (based on yes=1, no=0)"""
    # Only use valid samples (both label and prediction are yes/no)
    valid_results = [
        r for r in results
        if r.get("label_numeric") in (0, 1) and r.get("prediction_numeric") in (0, 1)
    ]

    if not valid_results:
        empty_metrics = {
            "accuracy": 0.0,
            "precision_yes": 0.0,
            "recall_yes": 0.0,
            "f1_yes": 0.0,
            "macro_f1": 0.0,
            "per_class_metrics": {"no": {}, "yes": {}},
            "confusion_matrix": {},
            "support_counts": {"no": 0, "yes": 0},
            "valid_samples": 0
        }
        return empty_metrics

    tp = tn = fp = fn = 0
    for r in valid_results:
        true_label = r["label_numeric"]
        pred_label = r["prediction_numeric"]
        if true_label == 1:  # yes (positive class)
            if pred_label == 1:
                tp += 1
            else:
                fn += 1
        else:  # no (negative class)
            if pred_label == 1:
                fp += 1
            else:
                tn += 1

    total_valid = len(valid_results)
    accuracy = (tp + tn) / total_valid if total_valid > 0 else 0.0

    # Positive class (yes) metrics
    precision_yes = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_yes = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_yes = 2 * precision_yes * recall_yes / (precision_yes + recall_yes) if (precision_yes + recall_yes) > 0 else 0.0

    # Negative class (no) metrics
    precision_no = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_no = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_no = 2 * precision_no * recall_no / (precision_no + recall_no) if (precision_no + recall_no) > 0 else 0.0

    macro_f1 = (f1_yes + f1_no) / 2

    return {
        "accuracy": accuracy,
        "precision_yes": precision_yes,
        "recall_yes": recall_yes,
        "f1_yes": f1_yes,
        "precision_no": precision_no,
        "recall_no": recall_no,
        "f1_no": f1_no,
        "macro_f1": macro_f1,
        "per_class_metrics": {
            "no": {
                "precision": precision_no,
                "recall": recall_no,
                "f1_score": f1_no,
                "support": tn + fp
            },
            "yes": {
                "precision": precision_yes,
                "recall": recall_yes,
                "f1_score": f1_yes,
                "support": tp + fn
            }
        },
        "confusion_matrix": {
            "TP (yes→yes)": tp,
            "FN (yes→no)": fn,
            "FP (no→yes)": fp,
            "TN (no→no)": tn,
            "total_valid_samples": total_valid
        },
        "support_counts": {
            "no": tn + fp,
            "yes": tp + fn
        },
        "valid_samples": total_valid
    }

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions on Cosmos-R1 dataset (semantic-aware evaluation)"
    )
    parser.add_argument(
        "--infer_file_path",
        type=str,
        required=True,
        help="Path to the inference JSONL file containing model responses and labels"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Path to the original dataset JSONL file (required for semantic mapping)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model being evaluated"
    )
    parser.add_argument(
        "--r1_type",
        action="store_true",
        help="Enable R1-type evaluation: extract answers from <answer> tags"
    )

    args = parser.parse_args()

    # Validate required parameters
    if not os.path.exists(args.infer_file_path):
        raise FileNotFoundError(f"Inference file not found: {args.infer_file_path}")

    if not args.dataset_path or not os.path.exists(args.dataset_path):
        raise ValueError(
            "dataset_path is required for semantic-aware evaluation. "
            "It contains the question text needed to map A/B options to yes/no semantics."
        )

    # --- Load Data ---
    print(f"Loading inference data from: {args.infer_file_path}")
    print(f"Loading dataset metadata from: {args.dataset_path}")
    print(f"R1-type evaluation: {'Enabled' if args.r1_type else 'Disabled'}")

    # Load inference data
    infer_data = []
    with open(args.infer_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                infer_data.append(json.loads(line))

    print(f"Loaded {len(infer_data)} samples from inference file")

    # Load metadata (required)
    dataset_metadata = load_dataset_metadata(args.dataset_path)
    print(f"Loaded {len(dataset_metadata)} samples from dataset metadata")

    if len(dataset_metadata) != len(infer_data):
        print(
            f"WARNING: Metadata count ({len(dataset_metadata)}) != Inference count ({len(infer_data)}). "
            "Evaluation may be inaccurate for mismatched samples.",
            file=sys.stderr
        )

    # --- Process Each Sample ---
    detailed_results = []
    invalid_mappings = 0
    for idx, item in enumerate(infer_data):
        try:
            result = process_single_item(item, idx, dataset_metadata, args.r1_type)
            detailed_results.append(result)

            # Count invalid mappings
            if result.get("warning"):
                invalid_mappings += 1
        except Exception as e:
            print(f"Error processing sample {idx}: {e}", file=sys.stderr)
            detailed_results.append({
                "index": idx,
                "error": str(e),
                "is_correct": False,
                "label_numeric": -1,
                "prediction_numeric": -1
            })

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(infer_data)} samples...")

    if invalid_mappings > 0:
        print(f"\nWarning: {invalid_mappings} samples had parsing issues (see detailed_results for details)")

    # --- Calculate Metrics ---
    overall_metrics = calculate_overall_metrics(detailed_results)
    secondary_metrics = calculate_secondary_metrics(detailed_results)

    # --- Prepare Output Structure ---
    output_data = {
        "evaluation_config": {
            "model_name": args.model_name,
            "infer_file": os.path.basename(args.infer_file_path),
            "dataset_file": os.path.basename(args.dataset_path),
            "r1_type": args.r1_type,
            "total_samples": len(infer_data)
        },
        "overall": overall_metrics,
        "secondary_metrics": secondary_metrics,
        "detailed_results": detailed_results
    }

    # --- Save Results ---
    model_output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    input_filename = os.path.basename(args.infer_file_path)
    base_name = os.path.splitext(input_filename)[0]
    if base_name.endswith('.jsonl'):
        base_name = base_name[:-6]

    output_filename = f"{base_name}_scores.json"
    output_path = os.path.join(model_output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # --- Print Summary ---
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS - {args.model_name}")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"\nOVERALL METRICS")
    print(f"Valid samples used: {int(overall_metrics['valid_samples'])}/{int(overall_metrics['total_samples'])}")
    print(f"Accuracy: {overall_metrics['score']:.4f} ({overall_metrics['score'] * 100:.2f}%)")
    print(f"Correct: {int(overall_metrics['correct_samples'])} / {int(overall_metrics['valid_samples'])}")

    print(f"\nSECONDARY METRICS (semantic: yes=success, no=failure)")
    print(f"Macro F1: {secondary_metrics['macro_f1']:.4f}")
    print(f"F1 (yes/success): {secondary_metrics['f1_yes']:.4f}")
    print(f"F1 (no/failure):  {secondary_metrics['f1_no']:.4f}")
    print(f"Precision (yes): {secondary_metrics['precision_yes']:.4f} | Recall (yes): {secondary_metrics['recall_yes']:.4f}")
    print(f"Precision (no):  {secondary_metrics['precision_no']:.4f} | Recall (no):  {secondary_metrics['recall_no']:.4f}")

    print(f"\nCLASS DISTRIBUTION")
    print(f"Class 'yes' (success): {secondary_metrics['support_counts']['yes']} samples")
    print(f"Class 'no'  (failure): {secondary_metrics['support_counts']['no']} samples")

    # --- Error Analysis ---
    if overall_metrics['score'] < 1.0:
        print(f"\nERROR ANALYSIS (first 10 errors with semantic context)")
        error_count = 0
        for result in detailed_results:
            if not result.get("is_correct", False) and result.get("label_semantic") in ("yes", "no"):
                q_snippet = result.get("question_snippet", "")[:80]
                mapping = result.get("option_mapping", {})
                print(f"\nSample {result['index']}:")
                print(f"  Question snippet: {q_snippet}")
                print(f"  Option mapping: A='{mapping.get('A', '?')}', B='{mapping.get('B', '?')}'")
                print(f"  Label: '{result.get('label_letter', '?')}' -> semantic '{result.get('label_semantic', '?')}'")
                print(f"  Prediction: '{result.get('response', '')[:50]}' -> semantic '{result.get('prediction_semantic', '?')}'")
                if result.get("warning"):
                    print(f"  Warning: {result['warning']}")
                error_count += 1
                if error_count >= 10:
                    remaining = sum(1 for r in detailed_results if not r.get("is_correct", False)) - 10
                    if remaining > 0:
                        print(f"\n... and {remaining} more errors (see detailed_results in output file)")
                    break

    print(f"\n{'='*60}")
    if args.r1_type:
        print("Evaluation mode: R1-type (answers extracted from <answer> tags)")
    print(f"Evaluation complete!")

if __name__ == "__main__":
    main()
