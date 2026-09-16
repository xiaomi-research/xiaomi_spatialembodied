# UPDATE: Replace placeholder paths with your actual paths.
import json
import jsonlines
import numpy as np
import os, sys
import argparse
sys.path.append("/path/to/eval/evalcap")

from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import re
import datetime
from pathlib import Path
import torch
import pandas as pd
import traceback

sys.path.append("/path/to/eval/LingoQA-main/benchmark")

current_dir = os.path.dirname(os.path.abspath(__file__))
base_model_dir = os.path.dirname(current_dir)  # since current_dir is .../<repo-root>/eval
sys.path.append(base_model_dir)
from eval.eval_utils.preprocessor import preprocess_model_output
# breakpoint()
# Import from constants
try:
    from constants import Keys
except ImportError:
    class Keys:
        QUESTION = "question"
        RESPONSE = "response"
        LABELS = "labels"
        MESSAGES = "messages"

# Try to import LingoJudge
try:
    from judge import LingoJudge
    LINGOJUDGE_AVAILABLE = True
except ImportError:
    LINGOJUDGE_AVAILABLE = False
    print("Warning: Cannot import LingoJudge, will use simple substitute")

def extract_answer_from_tag(text):
    """Extract answer content from <answer> tags"""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def evaluate_on_coco_caption(res_file, label_file, outfile='eval.json'):
    """
    Evaluate caption results using COCO evaluation tools

    Args:
        res_file: Prediction result file (COCO format)
        label_file: Label file (COCO format)
        outfile: Output evaluation result file path
    """
    try:
        coco = COCO(label_file)
        cocoRes = coco.loadRes(res_file)
        cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')

        # Evaluate all images
        cocoEval.params['image_id'] = cocoRes.getImgIds()

        # Evaluation results
        cocoEval.evaluate()
        result = cocoEval.eval

        if not outfile:
            print(result)
        else:
            with open(outfile, 'w') as fp:
                json.dump(result, fp, indent=4)

        return result
    except Exception as e:
        print(f"Error during COCO evaluation: {e}")
        return None


def load_original_dataset(dataset_path: str) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load the original dataset, build a mapping from sample_idx to sample,
    and maintain list order (for line-by-line correspondence)

    Args:
        dataset_path: Original dataset path

    Returns:
        (idx_to_sample, samples_list) - mapping dict and ordered list (in file line order)
    """
    idx_to_sample = {}
    samples_list = []  # List maintaining file original order

    try:
        with jsonlines.open(dataset_path) as reader:
            for idx, sample in enumerate(reader):
                # Get metadata
                metadata = sample.get('metadata', {})

                # Use sample_idx in metadata as unique identifier
                sample_idx = metadata.get('sample_idx', idx)

                # Get ground truth (assistant's content)
                messages = sample.get('messages', [])
                ground_truth = ""
                for msg in messages:
                    if msg.get('role') == 'assistant':
                        ground_truth = msg.get('content', '')
                        break

                # Get question
                question = ""
                for msg in messages:
                    if msg.get('role') == 'user':
                        question = msg.get('content', '')
                        break

                # Create sample info
                sample_info = {
                    'sample_idx': sample_idx,
                    'token': metadata.get('token', ''),
                    'ground_truth': ground_truth,
                    'question': question,
                    'question_type': metadata.get('question_type', 'unknown'),
                    'task_type': metadata.get('task_type', 'unknown'),
                    'data_split': metadata.get('data_split', 'unknown'),
                    'metadata': metadata
                }

                # Add to mapping and list (key change: also add to list to maintain order)
                idx_to_sample[str(sample_idx)] = sample_info
                samples_list.append(sample_info)

        print(f"Successfully loaded {len(idx_to_sample)} original samples")
        return idx_to_sample, samples_list

    except Exception as e:
        print(f"Failed to load original dataset {dataset_path}: {e}")
        return {}, []


def process_infer_results(infer_file: str, dataset_mapping: Dict[str, Dict[str, Any]],
                         dataset_list: List[Dict[str, Any]], r1_type: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """
    Process inference results, prioritize line-by-line correspondence matching,
    no longer rely on sample_idx field

    Args:
        infer_file: Inference result file path
        dataset_mapping: Original dataset sample_idx mapping (for fallback)
        dataset_list: Original dataset ordered list (for line-by-line correspondence)
        r1_type: Whether to use R1-type parsing (extract answers from <answer> tags)

    Returns:
        (predictions_list, ground_truth_list) - prediction and ground truth lists
    """
    predictions = []
    ground_truths = []
    matched_count = 0
    unmatched_count = 0
    line_matched_count = 0  # Count of direct line number matches

    try:
        with jsonlines.open(infer_file) as reader:
            for idx, sample in enumerate(reader):
                # Get prediction
                prediction = sample.get('response', '')

                # If R1-type parsing is enabled, extract answer from <answer> tag
                if r1_type:
                    prediction = preprocess_model_output(prediction)
                    prediction = extract_answer_from_tag(prediction)

                # ========== Key change: prioritize line-by-line correspondence ==========
                if idx < len(dataset_list):
                    # Line number matching mode (one-to-one correspondence)
                    dataset_sample = dataset_list[idx]
                    matched_gt = dataset_sample['ground_truth']

                    # If R1-type parsing is enabled, also extract from ground truth's <answer> tag
                    if r1_type:

                        matched_gt = extract_answer_from_tag(matched_gt)

                    sample_idx = dataset_sample['sample_idx']
                    match_info = f"line-by-line matched at index {idx}"
                    line_matched_count += 1
                    matched_count += 1
                    source = 'dataset_line_match'
                else:
                    # Fallback: line number exceeds range, use original sample_idx matching logic
                    print(f"Warning: line number {idx} exceeds dataset range, trying sample_idx matching...")

                    gt_from_infer = sample.get('labels', '')

                    # Try to get ground truth from messages
                    if not gt_from_infer:
                        messages = sample.get('messages', [])
                        for msg in messages:
                            if msg.get('role') == 'assistant':
                                gt_from_infer = msg.get('content', '')
                                break

                    # If R1-type parsing is enabled, extract from ground truth's <answer> tag
                    if r1_type and gt_from_infer:
                        gt_from_infer = extract_answer_from_tag(gt_from_infer)

                    # Try to extract sample_idx
                    sample_idx = None
                    messages = sample.get('messages', [])
                    for msg in messages:
                        if msg.get('role') == 'system' and 'sample_idx' in str(msg.get('content', '')):
                            content = msg.get('content', '')
                            match = re.search(r'sample_idx[:\s]*(\d+)', content)
                            if match:
                                sample_idx = match.group(1)
                                break

                    # Get sample_idx from metadata
                    if sample_idx is None:
                        metadata = sample.get('metadata', {})
                        sample_idx = str(metadata.get('sample_idx', idx))

                    # Check if in original dataset
                    if sample_idx in dataset_mapping:
                        dataset_sample = dataset_mapping[sample_idx]
                        if dataset_sample.get('ground_truth'):
                            matched_gt = dataset_sample['ground_truth']
                            if r1_type:
                                matched_gt = extract_answer_from_tag(matched_gt)
                        else:
                            matched_gt = gt_from_infer
                        match_info = f"sample_idx {sample_idx} matched"
                        matched_count += 1
                        source = 'dataset'
                    else:
                        matched_gt = gt_from_infer
                        match_info = f"not found in dataset"
                        unmatched_count += 1
                        source = 'infer'

                # Create unique ID for the sample
                sample_id = f"sample_{sample_idx}_{idx}"

                # Add to predictions list
                predictions.append({
                    'image_id': sample_id,
                    'caption': prediction,
                    'sample_idx': sample_idx,
                    'match_info': match_info,
                    'question': dataset_list[idx].get('question', '') if idx < len(dataset_list) else dataset_mapping.get(str(sample_idx), {}).get('question', '')
                })

                # Add to ground truth list
                ground_truths.append({
                    'image_id': sample_id,
                    'id': idx,
                    'caption': matched_gt,
                    'sample_idx': sample_idx,
                    'source': source if idx < len(dataset_list) else ('dataset' if sample_idx in dataset_mapping else 'infer'),
                    'question': dataset_list[idx].get('question', '') if idx < len(dataset_list) else dataset_mapping.get(str(sample_idx), {}).get('question', '')
                })

        print(f"Successfully processed {len(predictions)} samples")
        print(f"Line number direct matches: {line_matched_count}")
        print(f"Dataset matches: {matched_count}")
        print(f"Unmatched: {unmatched_count}")

        # Key check: warn if line counts don't match
        if len(predictions) != len(dataset_list):
            print(f"Warning: inference file line count ({len(predictions)}) does not match dataset line count ({len(dataset_list)})!")
            print(f"    Please check whether the two files truly correspond one-to-one.")

        return predictions, ground_truths

    except Exception as e:
        print(f"Failed to process inference results {infer_file}: {e}")
        traceback.print_exc()
        return [], []


def evaluate_with_lingojudge(predictions: List[Dict], ground_truths: List[Dict], batch_size: int = 1) -> Dict[str, Any]:
    """
    Evaluate Q&A results using LingoJudge

    Args:
        predictions: Prediction list
        ground_truths: Ground truth list
        batch_size: Batch processing size

    Returns:
        Evaluation result dictionary
    """
    if not LINGOJUDGE_AVAILABLE:
        print("Warning: LingoJudge unavailable, will use simple evaluation")
        return _simple_lingojudge_evaluation(predictions, ground_truths)

    try:
        # Initialize evaluator
        if torch.cuda.is_available():
            judge = LingoJudge().eval().to("cuda:0")
            print("Using GPU for LingoJudge evaluation")
        else:
            judge = LingoJudge().eval()
            print("Using CPU for LingoJudge evaluation")

        # Prepare data
        questions = []
        references = []
        model_predictions = []

        for pred, gt in zip(predictions, ground_truths):
            question = pred.get('question', '')
            reference = gt.get('caption', '')
            prediction = pred.get('caption', '')

            if question and reference and prediction:
                questions.append(question)
                references.append([reference])  # LingoJudge expects list of lists
                model_predictions.append(prediction)

        if not questions:
            print("Warning: No valid questions found for LingoJudge evaluation")
            return {
                'lingojudge_score': 0.0,
                'lingojudge_scores': [],
                'lingojudge_accuracy': 0.0,
                'lingojudge_correct_count': 0,
                'lingojudge_incorrect_count': len(predictions)
            }

        print(f"Evaluating {len(questions)} samples with LingoJudge")

        # Batch evaluation
        all_scores = []
        all_probs = []
        all_correct = []

        for i in range(0, len(questions), batch_size):
            batch_end = min(i + batch_size, len(questions))
            batch_questions = questions[i:batch_end]
            batch_references = references[i:batch_end]
            batch_predictions = model_predictions[i:batch_end]

            # Evaluate batch
            batch_scores = judge.compute(batch_questions, batch_references, batch_predictions)

            # Calculate probabilities
            if isinstance(batch_scores, torch.Tensor):
                batch_scores_list = batch_scores.detach().cpu().tolist()
                batch_probs = torch.sigmoid(batch_scores).detach().cpu().tolist()
            else:
                batch_scores_list = batch_scores
                batch_probs = [1.0 / (1.0 + torch.exp(-torch.tensor(score))) for score in batch_scores_list]

            # Determine correctness
            batch_correct = [score > 0.0 for score in batch_scores_list]

            all_scores.extend(batch_scores_list)
            all_probs.extend(batch_probs)
            all_correct.extend(batch_correct)

        # Calculate statistics
        if all_scores:
            accuracy = sum(all_correct) / len(all_correct) if all_correct else 0.0
            score_mean = float(np.mean(all_scores)) if all_scores else 0.0
            score_std = float(np.std(all_scores)) if len(all_scores) > 1 else 0.0
            prob_mean = float(np.mean(all_probs)) if all_probs else 0.0

            return {
                'lingojudge_score': score_mean,
                'lingojudge_accuracy': accuracy,
                'lingojudge_scores': all_scores,
                'lingojudge_probs': all_probs,
                'lingojudge_correct': all_correct,
                'lingojudge_correct_count': sum(all_correct),
                'lingojudge_incorrect_count': len(all_correct) - sum(all_correct),
                'lingojudge_score_mean': score_mean,
                'lingojudge_score_std': score_std,
                'lingojudge_prob_mean': prob_mean,
                'lingojudge_evaluation_count': len(questions)
            }
        else:
            return {
                'lingojudge_score': 0.0,
                'lingojudge_scores': [],
                'lingojudge_accuracy': 0.0,
                'lingojudge_correct_count': 0,
                'lingojudge_incorrect_count': len(predictions)
            }

    except Exception as e:
        print(f"LingoJudge evaluation error: {e}")
        traceback.print_exc()
        return _simple_lingojudge_evaluation(predictions, ground_truths)


def _simple_lingojudge_evaluation(predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, Any]:
    """
    Simple LingoJudge substitute evaluation (used when LingoJudge is unavailable)

    Args:
        predictions: Prediction list
        ground_truths: Ground truth list

    Returns:
        Evaluation result dictionary
    """
    print("Using simple LingoJudge substitute evaluation")

    scores = []
    correct_list = []

    for pred, gt in zip(predictions, ground_truths):
        prediction = pred.get('caption', '')
        reference = gt.get('caption', '')

        # Simple string matching evaluation
        if prediction.lower().strip() == reference.lower().strip():
            score = 1.0
            correct = True
        else:
            # Calculate simple similarity
            pred_words = set(prediction.lower().split())
            ref_words = set(reference.lower().split())
            if pred_words and ref_words:
                intersection = len(pred_words.intersection(ref_words))
                union = len(pred_words.union(ref_words))
                score = intersection / union if union > 0 else 0.0
            else:
                score = 0.0
            correct = score > 0.5

        scores.append(score)
        correct_list.append(correct)

    if scores:
        accuracy = sum(correct_list) / len(correct_list) if correct_list else 0.0
        score_mean = float(np.mean(scores)) if scores else 0.0
        score_std = float(np.std(scores)) if len(scores) > 1 else 0.0

        return {
            'lingojudge_score': score_mean,
            'lingojudge_accuracy': accuracy,
            'lingojudge_scores': scores,
            'lingojudge_correct': correct_list,
            'lingojudge_correct_count': sum(correct_list),
            'lingojudge_incorrect_count': len(correct_list) - sum(correct_list),
            'lingojudge_score_mean': score_mean,
            'lingojudge_score_std': score_std,
            'lingojudge_prob_mean': score_mean,  # In simple evaluation, probability equals score
            'lingojudge_evaluation_count': len(scores)
        }
    else:
        return {
            'lingojudge_score': 0.0,
            'lingojudge_scores': [],
            'lingojudge_accuracy': 0.0,
            'lingojudge_correct_count': 0,
            'lingojudge_incorrect_count': len(predictions)
        }


def create_coco_format_data(predictions: List[Dict], ground_truths: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Create COCO format data

    Args:
        predictions: Prediction list
        ground_truths: Ground truth list

    Returns:
        (coco_predictions, coco_ground_truths) - COCO format data
    """
    # Create COCO format predictions
    coco_predictions = []
    for pred in predictions:
        coco_predictions.append({
            'image_id': pred['image_id'],
            'caption': pred['caption']
        })

    # Create COCO format ground truths
    images_info = []
    annotations_info = []

    for i, gt in enumerate(ground_truths):
        image_id = gt['image_id']

        # Add image info
        images_info.append({
            'id': image_id,
            'file_name': f"{image_id}.jpg"  # Virtual filename
        })

        # Add annotation info
        annotations_info.append({
            'image_id': image_id,
            'id': i,
            'caption': gt['caption']
        })

    coco_ground_truths = {
        'images': images_info,
        'annotations': annotations_info,
        'type': 'captions',
        'info': {},
        'licenses': []
    }

    return coco_predictions, coco_ground_truths


def calculate_overall_metric(coco_metrics: Dict[str, Any], lingojudge_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate overall metrics
    Use LingoJudge score as the primary metric, COCO metrics as reference

    Args:
        coco_metrics: COCO evaluation metrics
        lingojudge_metrics: LingoJudge evaluation metrics

    Returns:
        Overall metrics dictionary
    """
    # Use LingoJudge score as the primary score
    overall_score = lingojudge_metrics.get('lingojudge_accuracy', 0.0)

    # Calculate weighted total score
    if coco_metrics:
        # If COCO metrics are available, calculate weighted average
        coco_weight = 0.7
        lingojudge_weight = 0.3

        cocer_score = coco_metrics.get("CIDEr", 0.0)
        spice_score = coco_metrics.get("SPICE", 0.0)
        meteor_score = coco_metrics.get("METEOR", 0.0)

        # Calculate COCO average score
        coco_scores = [cocer_score, spice_score, meteor_score]
        coco_avg = sum(coco_scores) / len(coco_scores) if coco_scores else 0.0

        # Weighted average
        weighted_score = (overall_score * lingojudge_weight + coco_avg * coco_weight)
    else:
        weighted_score = overall_score
        coco_scores = []

    return {
        "score": float(weighted_score),
        "metric": "lingojudge_score",
        "lingojudge_score": float(overall_score),
        "coco_metrics": {
            "CIDEr": float(coco_metrics.get("CIDEr", 0.0)) if coco_metrics else 0.0,
            "SPICE": float(coco_metrics.get("SPICE", 0.0)) if coco_metrics else 0.0,
            "METEOR": float(coco_metrics.get("METEOR", 0.0)) if coco_metrics else 0.0
        },
        "description": f"Overall score weighted by LingoJudge (70%) and COCO metrics (30%). LingoJudge accuracy: {lingojudge_metrics.get('lingojudge_accuracy', 0.0):.2%}"
    }


def get_secondary_metrics(coco_metrics: Dict[str, Any], lingojudge_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get secondary metrics

    Args:
        coco_metrics: COCO evaluation metrics
        lingojudge_metrics: LingoJudge evaluation metrics

    Returns:
        Secondary metrics dictionary
    """
    secondary_metrics = {}

    # Add COCO metrics
    if coco_metrics:
        secondary_metrics.update({
            "coco_Bleu_1": float(coco_metrics.get("Bleu_1", 0.0)),
            "coco_Bleu_2": float(coco_metrics.get("Bleu_2", 0.0)),
            "coco_Bleu_3": float(coco_metrics.get("Bleu_3", 0.0)),
            "coco_Bleu_4": float(coco_metrics.get("Bleu_4", 0.0)),
            "coco_METEOR": float(coco_metrics.get("METEOR", 0.0)),
            "coco_ROUGE_L": float(coco_metrics.get("ROUGE_L", 0.0)),
            "coco_CIDEr": float(coco_metrics.get("CIDEr", 0.0)),
            "coco_SPICE": float(coco_metrics.get("SPICE", 0.0))
        })

    # Add LingoJudge metrics
    secondary_metrics.update({
        "lingojudge_accuracy": float(lingojudge_metrics.get("lingojudge_accuracy", 0.0)),
        "lingojudge_correct_count": int(lingojudge_metrics.get("lingojudge_correct_count", 0)),
        "lingojudge_incorrect_count": int(lingojudge_metrics.get("lingojudge_incorrect_count", 0)),
        "lingojudge_score_mean": float(lingojudge_metrics.get("lingojudge_score_mean", 0.0)),
        "lingojudge_score_std": float(lingojudge_metrics.get("lingojudge_score_std", 0.0)),
        "lingojudge_prob_mean": float(lingojudge_metrics.get("lingojudge_prob_mean", 0.0)),
        "lingojudge_evaluation_count": int(lingojudge_metrics.get("lingojudge_evaluation_count", 0))
    })

    return secondary_metrics


def generate_detailed_results(predictions: List[Dict], ground_truths: List[Dict],
                            coco_metrics: Dict[str, Any], lingojudge_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate detailed results report

    Args:
        predictions: Prediction list
        ground_truths: Ground truth list
        coco_metrics: COCO evaluation metrics
        lingojudge_metrics: LingoJudge evaluation metrics

    Returns:
        Detailed results dictionary
    """
    # Create sample-level detailed information
    samples_detail = []

    # Ensure predictions and ground_truths are in the same order
    pred_dict = {p['image_id']: p for p in predictions}
    gt_dict = {gt['image_id']: gt for gt in ground_truths}

    # Use set intersection instead of list intersection
    all_image_ids = sorted(set(pred_dict.keys()).intersection(set(gt_dict.keys())))

    # Get LingoJudge scores
    lingojudge_scores = lingojudge_metrics.get('lingojudge_scores', [])
    lingojudge_probs = lingojudge_metrics.get('lingojudge_probs', [])
    lingojudge_correct = lingojudge_metrics.get('lingojudge_correct', [])

    for idx, img_id in enumerate(all_image_ids):
        pred_info = pred_dict[img_id]
        gt_info = gt_dict[img_id]

        pred = pred_info.get('caption', '')
        gt = gt_info.get('caption', '')
        question = pred_info.get('question', '')
        sample_idx = pred_info.get('sample_idx', 'unknown')

        # Calculate basic string similarity
        similarity = 0
        if pred and gt:
            # Simple word overlap similarity
            pred_words = set(pred.lower().split())
            gt_words = set(gt.lower().split())
            if pred_words and gt_words:
                intersection = pred_words.intersection(gt_words)
                union = pred_words.union(gt_words)
                similarity = len(intersection) / len(union) if union else 0

        # Calculate length statistics
        pred_len = len(pred.split())
        gt_len = len(gt.split())
        length_ratio = pred_len / gt_len if gt_len > 0 else 0

        # Calculate exact match metric
        exact_match = 1.0 if pred.strip() == gt.strip() else 0.0

        # Get LingoJudge scores
        lingojudge_score = lingojudge_scores[idx] if idx < len(lingojudge_scores) else 0.0
        lingojudge_prob = lingojudge_probs[idx] if idx < len(lingojudge_probs) else 0.0
        lingojudge_correct_flag = lingojudge_correct[idx] if idx < len(lingojudge_correct) else False

        # Build sample detail
        sample_detail = {
            'image_id': img_id,
            'sample_idx': sample_idx,
            'question': question,
            'prediction': pred,
            'ground_truth': gt,
            'metrics': {
                'similarity_score': float(similarity),
                'exact_match': float(exact_match),
                'prediction_length': int(pred_len),
                'ground_truth_length': int(gt_len),
                'length_ratio': float(length_ratio),
                'lingojudge_score': float(lingojudge_score),
                'lingojudge_probability': float(lingojudge_prob),
                'lingojudge_correct': bool(lingojudge_correct_flag)
            }
        }

        samples_detail.append(sample_detail)

    # Calculate statistics
    if samples_detail:
        similarity_scores = [s['metrics']['similarity_score'] for s in samples_detail]
        exact_matches = [s['metrics']['exact_match'] for s in samples_detail]
        pred_lengths = [s['metrics']['prediction_length'] for s in samples_detail]
        gt_lengths = [s['metrics']['ground_truth_length'] for s in samples_detail]
        length_ratios = [s['metrics']['length_ratio'] for s in samples_detail if s['metrics']['ground_truth_length'] > 0]
        lingojudge_scores_detail = [s['metrics']['lingojudge_score'] for s in samples_detail]
        lingojudge_probs_detail = [s['metrics']['lingojudge_probability'] for s in samples_detail]
        lingojudge_correct_detail = [s['metrics']['lingojudge_correct'] for s in samples_detail]

        detailed_summary = {
            'total_samples': len(samples_detail),
            'average_similarity': float(np.mean(similarity_scores)) if similarity_scores else 0,
            'exact_match_rate': float(np.mean(exact_matches)) if exact_matches else 0,
            'average_prediction_length': float(np.mean(pred_lengths)) if pred_lengths else 0,
            'average_ground_truth_length': float(np.mean(gt_lengths)) if gt_lengths else 0,
            'average_length_ratio': float(np.mean(length_ratios)) if length_ratios else 0,
            'std_prediction_length': float(np.std(pred_lengths)) if pred_lengths else 0,
            'std_ground_truth_length': float(np.std(gt_lengths)) if gt_lengths else 0,
            'average_lingojudge_score': float(np.mean(lingojudge_scores_detail)) if lingojudge_scores_detail else 0,
            'average_lingojudge_probability': float(np.mean(lingojudge_probs_detail)) if lingojudge_probs_detail else 0,
            'lingojudge_accuracy': float(np.mean(lingojudge_correct_detail)) if lingojudge_correct_detail else 0
        }

        sample_level_metrics = {
            'similarity_score': {
                'mean': float(np.mean(similarity_scores)) if similarity_scores else 0,
                'std': float(np.std(similarity_scores)) if similarity_scores else 0,
                'min': float(np.min(similarity_scores)) if similarity_scores else 0,
                'max': float(np.max(similarity_scores)) if similarity_scores else 0
            },
            'prediction_length': {
                'mean': float(np.mean(pred_lengths)) if pred_lengths else 0,
                'std': float(np.std(pred_lengths)) if pred_lengths else 0,
                'min': float(np.min(pred_lengths)) if pred_lengths else 0,
                'max': float(np.max(pred_lengths)) if pred_lengths else 0
            },
            'lingojudge_score': {
                'mean': float(np.mean(lingojudge_scores_detail)) if lingojudge_scores_detail else 0,
                'std': float(np.std(lingojudge_scores_detail)) if lingojudge_scores_detail else 0,
                'min': float(np.min(lingojudge_scores_detail)) if lingojudge_scores_detail else 0,
                'max': float(np.max(lingojudge_scores_detail)) if lingojudge_scores_detail else 0
            }
        }
    else:
        detailed_summary = {
            'total_samples': 0,
            'average_similarity': 0,
            'exact_match_rate': 0,
            'average_prediction_length': 0,
            'average_ground_truth_length': 0,
            'average_length_ratio': 0,
            'std_prediction_length': 0,
            'std_ground_truth_length': 0,
            'average_lingojudge_score': 0,
            'average_lingojudge_probability': 0,
            'lingojudge_accuracy': 0
        }
        sample_level_metrics = {
            'similarity_score': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'prediction_length': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'lingojudge_score': {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        }

    return {
        'summary': detailed_summary,
        'sample_level_metrics': sample_level_metrics,
        'samples': samples_detail
    }


def save_results_to_json(overall_metrics: Dict[str, Any], secondary_metrics: Dict[str, Any],
                         detailed_results: Dict[str, Any], output_path: str,
                         model_name: str, infer_file_path: str, r1_type: bool = False) -> Dict[str, Any]:
    """
    Save results in standard JSON format

    Args:
        overall_metrics: Overall metrics
        secondary_metrics: Secondary metrics
        detailed_results: Detailed results
        output_path: Output file path
        model_name: Model name
        infer_file_path: Inference file path
        r1_type: Whether to use R1-type parsing

    Returns:
        Result dictionary
    """
    # Build result dictionary
    result_dict = {
        "model_name": model_name,
        "infer_file": os.path.basename(infer_file_path),
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
        "evaluation_type": "r1_type" if r1_type else "standard",
        "overall": overall_metrics,
        "secondary_metrics": secondary_metrics,
        "detailed_results": detailed_results
    }

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    print(f"Evaluation results saved to: {output_path}")

    return result_dict


def main():
    parser = argparse.ArgumentParser(description='Evaluate Caption generation model')

    # Required arguments
    parser.add_argument('--infer_file_path', type=str, required=True,
                       help='Inference result file path (JSONL format)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output result save directory')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Original dataset file path (JSONL format)')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name')

    # Optional arguments
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Data split')
    parser.add_argument('--detailed_report', action='store_true',
                       help='Whether to generate detailed report file')
    parser.add_argument('--temp_dir', type=str, default=None,
                       help='Temporary file directory for saving intermediate results')

    # Additional arguments
    parser.add_argument('--r1_type', action='store_true',
                       help='Whether to use R1-type parsing (extract answers from <answer> tags)')
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='Batch size for LingoJudge evaluation')
    parser.add_argument('--sample_size', type=int, default=0,
                       help='Number of evaluation samples (0 means all)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine output file path
    input_basename = os.path.basename(args.infer_file_path)
    if input_basename.endswith('.jsonl'):
        if args.r1_type:
            # If R1-type, add _r1 suffix to filename
            output_filename = input_basename.replace('.jsonl', '_scores_r1.json')
        else:
            output_filename = input_basename.replace('.jsonl', '_scores.json')
    else:
        output_filename = f"{input_basename}_scores.json"

    # Create subdirectory for model name
    model_output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    output_json_path = os.path.join(model_output_dir, output_filename)

    # Create temporary directory
    if args.temp_dir:
        temp_dir = os.path.join(args.temp_dir, args.model_name, args.split)
    else:
        temp_dir = os.path.join(args.output_dir, 'temp', args.model_name, args.split)

    os.makedirs(temp_dir, exist_ok=True)

    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Inference file: {args.infer_file_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Data split: {args.split}")
    print(f"R1-type parsing: {'Enabled' if args.r1_type else 'Disabled'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sample count: {'All' if args.sample_size <= 0 else args.sample_size}")
    print(f"Output file: {output_json_path}")
    print(f"Temp directory: {temp_dir}")
    print("LingoJudge available: ", "Yes" if LINGOJUDGE_AVAILABLE else "No (will use simple evaluation)")
    print("=" * 60)

    # Step 1: Load original dataset (key change: get both dict and list)
    print("\n[Step 1] Loading original dataset...")
    dataset_mapping, dataset_list = load_original_dataset(args.dataset_path)

    if not dataset_mapping or not dataset_list:
        print("Error: Cannot load original dataset")
        return

    # Step 2: Process inference results (key change: pass dataset_list for line-by-line correspondence)
    print("\n[Step 2] Processing inference results...")
    predictions, ground_truths = process_infer_results(
        args.infer_file_path,
        dataset_mapping,
        dataset_list,  # Pass ordered list for one-to-one correspondence
        args.r1_type
    )

    if not predictions or not ground_truths:
        print("Error: No valid data processed")
        return

    print(f"Prediction sample count: {len(predictions)}")
    print(f"Ground truth sample count: {len(ground_truths)}")

    # Limit sample count
    if args.sample_size > 0 and args.sample_size < len(predictions):
        predictions = predictions[:args.sample_size]
        ground_truths = ground_truths[:args.sample_size]
        print(f"Sample count limited to: {args.sample_size}")

    # Step 3: Evaluate with LingoJudge
    print("\n[Step 3] Evaluating with LingoJudge...")
    lingojudge_metrics = evaluate_with_lingojudge(predictions, ground_truths, args.batch_size)

    print("\nLingoJudge evaluation results:")
    print("-" * 40)
    print(f"LingoJudge score: {lingojudge_metrics.get('lingojudge_score', 0.0):.4f}")
    print(f"LingoJudge accuracy: {lingojudge_metrics.get('lingojudge_accuracy', 0.0):.2%}")
    print(f"Correct predictions: {lingojudge_metrics.get('lingojudge_correct_count', 0)}")
    print(f"Incorrect predictions: {lingojudge_metrics.get('lingojudge_incorrect_count', 0)}")
    print(f"LingoJudge evaluation sample count: {lingojudge_metrics.get('lingojudge_evaluation_count', 0)}")
    print("-" * 40)

    # Step 4: Convert to COCO format and evaluate
    print("\n[Step 4] Converting to COCO format and evaluating...")
    coco_predictions, coco_ground_truths = create_coco_format_data(predictions, ground_truths)

    # Save intermediate files to temp directory
    pred_file = os.path.join(temp_dir, 'predictions_coco_format.json')
    gt_file = os.path.join(temp_dir, 'ground_truth_coco_format.json')

    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(coco_predictions, f, indent=4, ensure_ascii=False)

    with open(gt_file, 'w', encoding='utf-8') as f:
        json.dump(coco_ground_truths, f, indent=4, ensure_ascii=False)

    print(f"Prediction file saved: {pred_file}")
    print(f"Ground truth file saved: {gt_file}")

    # Run COCO evaluation
    results_file = os.path.join(temp_dir, 'evaluation_results.json')
    coco_metrics = evaluate_on_coco_caption(pred_file, gt_file, results_file)

    if coco_metrics:
        print("\nCOCO evaluation results:")
        print("-" * 40)
        for key, value in coco_metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
        print("-" * 40)
    else:
        print("Warning: COCO evaluation failed, will continue using LingoJudge results")
        coco_metrics = {}

    # Step 5: Calculate overall metrics
    print("\n[Step 5] Calculating overall metrics...")
    overall_metrics = calculate_overall_metric(coco_metrics, lingojudge_metrics)

    # Step 6: Get secondary metrics
    print("\n[Step 6] Getting secondary metrics...")
    secondary_metrics = get_secondary_metrics(coco_metrics, lingojudge_metrics)

    # Step 7: Generate detailed results
    print("\n[Step 7] Generating detailed results...")
    detailed_results = generate_detailed_results(predictions, ground_truths, coco_metrics, lingojudge_metrics)

    # Step 8: Save results in standard JSON format
    print("\n[Step 8] Saving results in standard JSON format...")
    result_dict = save_results_to_json(
        overall_metrics,
        secondary_metrics,
        detailed_results,
        output_json_path,
        args.model_name,
        args.infer_file_path,
        args.r1_type
    )

    # Step 9: Generate detailed report file (if requested)
    if args.detailed_report:
        print("\n[Step 9] Generating detailed report file...")
        detailed_report_file = os.path.join(temp_dir, 'detailed_report.json')
        with open(detailed_report_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=4, ensure_ascii=False)
        print(f"Detailed report saved: {detailed_report_file}")

    # Step 10: Save statistics
    print("\n[Step 10] Saving statistics...")
    stats = {
        'total_samples': len(predictions),
        'avg_prediction_length': detailed_results['summary']['average_prediction_length'],
        'avg_ground_truth_length': detailed_results['summary']['average_ground_truth_length'],
        'exact_match_rate': detailed_results['summary']['exact_match_rate'],
        'average_similarity': detailed_results['summary']['average_similarity'],
        'lingojudge_score': lingojudge_metrics.get('lingojudge_score', 0.0),
        'lingojudge_accuracy': lingojudge_metrics.get('lingojudge_accuracy', 0.0)
    }

    stats_file = os.path.join(temp_dir, 'statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    print(f"Statistics saved: {stats_file}")

    # Save detailed results in CSV format
    csv_path = os.path.join(model_output_dir, f"{input_basename.replace('.jsonl', '_detailed.csv')}")
    try:
        samples_df = pd.DataFrame(detailed_results['samples'])
        samples_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Detailed CSV results saved: {csv_path}")
    except Exception as e:
        print(f"Failed to save CSV results: {e}")

    # Print final result summary
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    print(f"Overall metrics:")
    print(f"  Total score: {result_dict['overall']['score']:.4f} (LingoJudge as primary)")
    print(f"  LingoJudge score: {result_dict['overall']['lingojudge_score']:.4f}")
    print(f"  LingoJudge accuracy: {lingojudge_metrics.get('lingojudge_accuracy', 0.0):.2%}")
    if coco_metrics:
        print(f"  COCO metrics:")
        print(f"    CIDEr: {result_dict['overall']['coco_metrics']['CIDEr']:.4f}")
        print(f"    SPICE: {result_dict['overall']['coco_metrics']['SPICE']:.4f}")
        print(f"    METEOR: {result_dict['overall']['coco_metrics']['METEOR']:.4f}")
    print(f"Secondary metrics:")
    for metric_name, metric_value in result_dict['secondary_metrics'].items():
        if isinstance(metric_value, float):
            print(f"  {metric_name}: {metric_value:.4f}")
        else:
            print(f"  {metric_name}: {metric_value}")
    print(f"Detailed results saved to: {output_json_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
