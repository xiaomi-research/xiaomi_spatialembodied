# UPDATE: Replace placeholder paths with your actual paths.
# Example: python eval_blink_v0.py
#   --pred_file /path/to/val.jsonl (required input)
#   --gt_file /path/to/converted_blink_data.jsonl (required input)
#   --output_file evaluation_results.json (optional output)
import json
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

def extract_prediction(response: str) -> str:
    """
    Extract prediction option (A or B) from model response.
    Prioritize matching standalone A/B; if not found, match the first A/B in text.
    """
    # Try matching standalone option (e.g., "A" or "A.")
    standalone_match = re.search(r'\b([AB])\b', response.strip())
    if standalone_match:
        return standalone_match.group(1)

    # If no standalone option, try matching A/B within text
    text_match = re.search(r'([AB])', response)
    if text_match:
        return text_match.group(1)

    return ""

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def evaluate_predictions(pred_data: List[Dict], gt_data: List[Dict]) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth
    """
    if len(pred_data) != len(gt_data):
        print(f"Warning: Prediction data ({len(pred_data)} lines) and ground truth data ({len(gt_data)} lines) have different lengths")
        # Use the minimum length
        min_len = min(len(pred_data), len(gt_data))
        pred_data = pred_data[:min_len]
        gt_data = gt_data[:min_len]

    total_samples = len(pred_data)
    correct = 0
    detailed_results = []

    # For secondary metrics
    option_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    for i, (pred_item, gt_item) in enumerate(zip(pred_data, gt_data)):
        # Get model prediction
        pred_response = pred_item.get('response', '')
        pred_label = pred_item.get('labels', '')

        # If response is non-empty, extract prediction from response first
        if pred_response:
            prediction = extract_prediction(pred_response)
        else:
            prediction = str(pred_label) if pred_label else ''

        # Get ground truth answer
        gt_answer = ''
        if 'messages' in gt_item and isinstance(gt_item['messages'], list):
            # Get assistant's message
            for msg in reversed(gt_item['messages']):
                if msg.get('role') == 'assistant':
                    gt_answer = str(msg.get('content', '')).strip()
                    break

        # Clean ground truth answer (remove spaces and periods)
        gt_answer_clean = gt_answer.replace('.', '').strip().upper()
        prediction_clean = prediction.upper()

        # Check if prediction is correct
        is_correct = prediction_clean == gt_answer_clean

        if is_correct:
            correct += 1

        # Compute per-option accuracy
        if gt_answer_clean in ['A', 'B']:
            option_stats[gt_answer_clean]['total'] += 1
            if is_correct:
                option_stats[gt_answer_clean]['correct'] += 1

        # Save detailed result
        sample_result = {
            'sample_id': i + 1,
            'prediction': prediction,
            'prediction_clean': prediction_clean,
            'true_answer': gt_answer,
            'true_answer_clean': gt_answer_clean,
            'is_correct': is_correct
        }

        # Add metadata if available
        if 'meta_data' in gt_item:
            sample_result['meta_data'] = gt_item['meta_data']
        elif 'images' in gt_item and isinstance(gt_item['images'], list) and len(gt_item['images']) > 0:
            # Extract info from image paths
            sample_result['image_paths'] = gt_item['images']

        detailed_results.append(sample_result)

    # Compute accuracy
    accuracy = (correct / total_samples * 100) if total_samples > 0 else 0

    # Compute secondary metrics
    secondary_metrics = {}
    # for option in ['A', 'B']:
    #     if option_stats[option]['total'] > 0:
    #         option_acc = (option_stats[option]['correct'] / option_stats[option]['total'] * 100)
    #         secondary_metrics[f'accuracy_{option.lower()}'] = round(option_acc, 2)
    #     else:
    #         secondary_metrics[f'accuracy_{option.lower()}'] = 0.0

    # Add summary statistics
    secondary_metrics['correct_samples'] = correct
    secondary_metrics['total_samples'] = total_samples
    secondary_metrics['error_samples'] = total_samples - correct

    # Build final result
    result = {
        'overall': {
            'score': round(accuracy, 2),
            'metric': 'Accuracy',
            'description': f'Ratio of correct predictions: {correct}/{total_samples}'
        },
        'secondary_metrics': secondary_metrics,
        # 'detailed_results': detailed_results
    }

    return result

def main():
    parser = argparse.ArgumentParser(description='Evaluate inference results against ground truth')
    parser.add_argument('--pred_file', type=str, required=True,
                       help='Inference result JSONL file path')
    parser.add_argument('--gt_file', type=str, required=True,
                       help='Ground truth JSONL file path')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                       help='Output result JSON file path')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed statistics')

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.pred_file).exists():
        print(f"Error: Prediction file does not exist: {args.pred_file}")
        sys.exit(1)

    if not Path(args.gt_file).exists():
        print(f"Error: Ground truth file does not exist: {args.gt_file}")
        sys.exit(1)

    print(f"Loading predictions: {args.pred_file}")
    pred_data = load_jsonl(args.pred_file)

    print(f"Loading ground truth: {args.gt_file}")
    gt_data = load_jsonl(args.gt_file)

    print(f"Starting evaluation...")
    print(f"Prediction samples: {len(pred_data)}")
    print(f"Ground truth samples: {len(gt_data)}")

    evaluation_results = evaluate_predictions(pred_data, gt_data)

    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    print(f"Evaluation complete! Results saved to: {args.output_file}")
    print(f"Accuracy: {evaluation_results['overall']['score']}%")
    print(f"Correct/Total: {evaluation_results['secondary_metrics']['correct_samples']}/{evaluation_results['secondary_metrics']['total_samples']}")

    # Print detailed statistics
    # if args.verbose:
    #     print("\nDetailed statistics:")
    #     print(f"  A-type accuracy: {evaluation_results['secondary_metrics'].get('accuracy_a', 0)}%")
    #     print(f"  B-type accuracy: {evaluation_results['secondary_metrics'].get('accuracy_b', 0)}%")

        # Print first few error samples
        # print("\nError sample details (first 5):")
        # error_count = 0
        # for i, result in enumerate(evaluation_results['detailed_results']):
        #     if not result['is_correct'] and error_count < 5:
        #         print(f"  Sample {result['sample_id']}: prediction={result['prediction']}, ground_truth={result['true_answer']}")
        #         error_count += 1

if __name__ == "__main__":
    main()
