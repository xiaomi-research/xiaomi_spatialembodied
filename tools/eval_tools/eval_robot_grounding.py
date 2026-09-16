# UPDATE: Replace placeholder paths with your actual paths.
import json, sys
import os
import re
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import jsonlines
from PIL import Image
import functools
import traceback

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


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Robot dataset Grounding task evaluation script')
    parser.add_argument('--infer_file_path', type=str, required=True,
                       help='Path to model inference output JSONL file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output result directory')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Original dataset file path')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for judging prediction correctness, default 0.5')
    parser.add_argument('--denormalize_models', type=str, default='qwen2-vl,internvl2.5,qwen3vl',
                       help='Comma-separated list of models requiring coordinate denormalization')
    parser.add_argument('--default_image_size', type=str, default='640,480',
                       help='Default image size, format: width,height, used when image cannot be read')
    parser.add_argument('--r1_type', action='store_true',
                       help='Enable R1-type evaluation, extract answer content from <answer> tags')
    return parser.parse_args()

def extract_answer_from_tag(text):
    """Extract answer content from <answer> tags"""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Ensure correct coordinate order
    x1_1, x2_1 = min(x1_1, x2_1), max(x1_1, x2_1)
    y1_1, y2_1 = min(y1_1, y2_1), max(y1_1, y2_1)
    x1_2, x2_2 = min(x1_2, x2_2), max(x1_2, x2_2)
    y1_2, y2_2 = min(y1_2, y2_2), max(y1_2, y2_2)

    # Calculate intersection region
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    # Calculate intersection area
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union

def parse_bbox_from_string(text: str, r1_type: bool = False) -> Optional[List[float]]:
    """Parse bounding box coordinates from string"""
    if not text or not isinstance(text, str):
        return None

    # If R1-type evaluation is enabled, first extract content from <answer> tags
    if r1_type:
        text = preprocess_model_output(text)
        text = extract_answer_from_tag(text)

    # Clean text
    text = text.strip()

    # Try parsing with multiple formats

    # Format 1: (x1,y1),(x2,y2) e.g. "(390,393),(524,531)"
    pattern1 = r'\((\d+),(\d+)\),\((\d+),(\d+)\)'
    match1 = re.search(pattern1, text)
    if match1:
        try:
            x1, y1, x2, y2 = map(int, match1.groups())
            return [float(x1), float(y1), float(x2), float(y2)]
        except:
            pass

    # Format 2: [x1, y1, x2, y2] e.g. "[257, 186, 340, 256]"
    pattern2 = r'\[([\d\s,\-\.]+)\]'
    match2 = re.search(pattern2, text)
    if match2:
        try:
            coords_str = match2.group(1)
            coords = [float(coord.strip()) for coord in re.split(r'[,\s]+', coords_str) if coord.strip()]
            if len(coords) == 4:
                return coords
        except:
            pass

    # Format 3: Contained in <answer> tags, e.g. "<answer>[x1, y1, x2, y2]</answer>"
    pattern3 = r'<answer>\[([\d\s,\-\.]+)\]</answer>'
    match3 = re.search(pattern3, text)
    if match3:
        try:
            coords_str = match3.group(1)
            coords = [float(coord.strip()) for coord in re.split(r'[,\s]+', coords_str) if coord.strip()]
            if len(coords) == 4:
                return coords
        except:
            pass

    # Format 4: JSON format containing bbox_2d
    pattern4 = r'bbox_2d["\s:\s]*\[([\d\s,\-\.]+)\]'
    match4 = re.search(pattern4, text)
    if match4:
        try:
            coords_str = match4.group(1)
            coords = [float(coord.strip()) for coord in re.split(r'[,\s]+', coords_str) if coord.strip()]
            if len(coords) == 4:
                return coords
        except:
            pass

    # Format 5: Special token format, e.g. Qwen2-VL series
    if '<|box_start|>' in text and '<|box_end|>' in text:
        pattern5 = r'<\|box_start\|>\(?(\d+),(\d+)\)?,?\(?(\d+),(\d+)\)?<\|box_end\|>'
        match5 = re.search(pattern5, text)
        if match5:
            try:
                x1, y1, x2, y2 = map(int, match5.groups())
                return [float(x1), float(y1), float(x2), float(y2)]
            except:
                pass

    # Format 6: Simple parenthesis format (x1, x2, y1, y2) e.g. "(0, 357, 468, 476)"
    # Note: This format may have two interpretations, handle according to context
    pattern6 = r'\(([\d\s,\-\.]+)\)'
    match6 = re.search(pattern6, text)
    if match6:
        try:
            coords_str = match6.group(1)
            coords = [float(coord.strip()) for coord in re.split(r'[,\s]+', coords_str) if coord.strip()]
            if len(coords) == 4:
                # Assume format is (x1, y1, x2, y2)
                return coords
        except:
            pass

    # Format 7: Simple bracket format [x1, x2, y1, y2] e.g. "[304, 305, 586, 379]"
    pattern7 = r'\[([\d\s,\-\.]+)\]'
    match7 = re.search(pattern7, text)
    if match7:
        try:
            coords_str = match7.group(1)
            coords = [float(coord.strip()) for coord in re.split(r'[,\s]+', coords_str) if coord.strip()]
            if len(coords) == 4:
                return coords
        except:
            pass

    # Format 8: In <answer> tags but using parentheses, e.g. "<answer>(0, 345, 644, 476)</answer>"
    pattern8 = r'<answer>\(([\d\s,\-\.]+)\)</answer>'
    match8 = re.search(pattern8, text)
    if match8:
        try:
            coords_str = match8.group(1)
            coords = [float(coord.strip()) for coord in re.split(r'[,\s]+', coords_str) if coord.strip()]
            if len(coords) == 4:
                return coords
        except:
            pass

    # Format 9: Try to directly match four-number pattern
    pattern9 = r'(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)'
    match9 = re.search(pattern9, text)
    if match9:
        try:
            x1, y1, x2, y2 = map(int, match9.groups())
            return [float(x1), float(y1), float(x2), float(y2)]
        except:
            pass

    # If all patterns fail, try extracting all numbers from string
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if len(numbers) >= 4:
        try:
            coords = [float(num) for num in numbers[:4]]
            return coords
        except:
            pass

    return None

def parse_ground_truth(labels: str) -> Optional[List[float]]:
    """Parse ground truth bounding box from labels"""
    if not labels or not isinstance(labels, str):
        return None

    # Standard format: <answer>[x1, y1, x2, y2]</answer>
    pattern = r'<answer>\[([\d\s,\-\.]+)\]</answer>'
    match = re.search(pattern, labels)
    if match:
        try:
            coords_str = match.group(1)
            coords = [float(coord.strip()) for coord in re.split(r'[,\s]+', coords_str) if coord.strip()]
            if len(coords) == 4:
                return coords
        except Exception as e:
            print(f"Failed to parse ground truth label: {labels}, error: {e}")
            return None

    # Try other possible formats
    return parse_bbox_from_string(labels, r1_type=False)

def denormalize_coordinates(bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
    """Convert per-mille normalized coordinates to absolute coordinates"""
    if len(bbox) != 4:
        return bbox

    width, height = image_size
    x1, y1, x2, y2 = bbox

    # If coordinates are in 0-1000 range, assume they are per-mille normalized coordinates
    if 0 <= x1 <= 1000 and 0 <= x2 <= 1000 and 0 <= y1 <= 1000 and 0 <= y2 <= 1000:
        x1 = x1 * width / 1000.0
        x2 = x2 * width / 1000.0
        y1 = y1 * height / 1000.0
        y2 = y2 * height / 1000.0

    return [x1, y1, x2, y2]

@functools.lru_cache(maxsize=1000)
def get_image_size(image_path: str) -> Optional[Tuple[int, int]]:
    """Read image file to get size, uses caching for performance"""
    try:
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                return img.size  # Returns (width, height)
        else:
            print(f"Warning: Image file does not exist: {image_path}")
            return None
    except Exception as e:
        print(f"Warning: Cannot read image size {image_path}: {e}")
        return None

def get_image_size_from_metadata(metadata: Dict, default_size: Tuple[int, int] = (640, 480)) -> Tuple[int, int]:
    """Extract image path from metadata and read image size"""
    # Get image path from metadata
    if 'images' in metadata and metadata['images']:
        image_info = metadata['images'][0]

        # Try different field names
        image_path = None
        if 'path' in image_info and image_info['path']:
            image_path = image_info['path']
        elif 'bytes' in image_info and image_info['bytes']:
            # If image byte data is available, could try to decode
            # Skipped for now, as we prefer file paths
            pass

        if image_path:
            # If path is relative, try to find in current directory
            if not os.path.isabs(image_path):
                # Try to find in parent directory of input file directory
                base_dir = os.path.dirname(args.infer_file_path)
                inferred_path = os.path.join(base_dir, '..', 'images', os.path.basename(image_path))
                if os.path.exists(inferred_path):
                    image_path = inferred_path
                else:
                    # Try other possible locations
                    possible_paths = [
                        image_path,
                        os.path.join('/path/to/data/preprocess_data/roborefit-benchmark-dataset/images/', os.path.basename(image_path)),
                        os.path.join('./images/', os.path.basename(image_path)),
                        os.path.join('../images/', os.path.basename(image_path)),
                        os.path.join('/path/to/data/public_robo_datasets/where2place/images/', os.path.basename(image_path)),
                    ]

                    for path in possible_paths:
                        if os.path.exists(path):
                            image_path = path
                            break

            # Read image size
            if image_path and os.path.exists(image_path):
                size = get_image_size(image_path)
                if size:
                    return size
            else:
                print(f"Warning: Image file does not exist: {image_path}")

    # If unable to get image size, return default value
    print(f"Warning: Cannot get image size, using default size {default_size}")
    return default_size

def evaluate_grounding_predictions(args):
    """Evaluate grounding prediction results"""

    # Read inference results
    predictions = []
    with jsonlines.open(args.infer_file_path, 'r') as f:
        for line in f:
            predictions.append(line)

    # Parse default image size
    try:
        default_width, default_height = map(int, args.default_image_size.split(','))
        default_size = (default_width, default_height)
    except:
        default_size = (640, 480)
        print(f"Warning: Cannot parse default image size '{args.default_image_size}', using {default_size}")

    # Check which models need denormalization
    denormalize_models = [m.strip().lower() for m in args.denormalize_models.split(',')]
    # need_denormalize = args.model_name.lower() in [m.lower() for m in denormalize_models]
    need_denormalize = False
    # if "qwen3-" in args.model_name.lower() or "qwen2-" in args.model_name.lower():
    if "qwen3" in args.model_name.lower() or ('qwen2' in args.model_name.lower() and 'qwen2.5' not in args.model_name.lower()):
        need_denormalize = True


    # Evaluate each prediction
    results = []
    ious = []
    correct_predictions = 0
    total_predictions = len(predictions)

    # Count valid samples
    valid_predictions = 0
    valid_ground_truths = 0
    valid_samples = 0

    for i, pred in enumerate(predictions):
        result_item = {
            'id': i,
            'prediction': pred.get('response', ''),
            'ground_truth': pred.get('labels', ''),
            'pred_bbox_valid': False,
            'gt_bbox_valid': False,
            'sample_valid': False
        }

        # Parse prediction bounding box
        pred_bbox = None
        try:
            # Choose parsing method based on whether R1-type evaluation is enabled
            if args.r1_type:
                pred_text = pred.get('response', '')
                pred_text = preprocess_model_output(pred_text)
                response_text = extract_answer_from_tag(pred_text)
                pred_bbox = parse_bbox_from_string(response_text, r1_type=False)  # r1_type=False here because already extracted
            else:
                pred_bbox = parse_bbox_from_string(pred.get('response', ''), r1_type=False)

            if pred_bbox is not None and len(pred_bbox) == 4:
                result_item['pred_bbox_valid'] = True
                result_item['pred_bbox'] = pred_bbox
                valid_predictions += 1
            else:
                result_item['pred_bbox'] = None
        except Exception as e:
            result_item['pred_bbox'] = None
            result_item['pred_error'] = str(e)
            print(f"Sample {i} failed to parse prediction box: {e}")

        # Parse ground truth bounding box
        gt_bbox = None
        try:
            gt_bbox = parse_ground_truth(pred.get('labels', ''))
            if gt_bbox is not None and len(gt_bbox) == 4:
                result_item['gt_bbox_valid'] = True
                result_item['gt_bbox'] = gt_bbox
                valid_ground_truths += 1
            else:
                result_item['gt_bbox'] = None
        except Exception as e:
            result_item['gt_bbox'] = None
            result_item['gt_error'] = str(e)
            print(f"Sample {i} failed to parse ground truth box: {e}")

        # Check if sample is valid
        if result_item['pred_bbox_valid'] and result_item['gt_bbox_valid']:
            result_item['sample_valid'] = True
            valid_samples += 1

            # If denormalization is needed, convert coordinates
            if need_denormalize and pred_bbox is not None:
                image_size = get_image_size_from_metadata(pred, default_size)
                result_item['image_size'] = image_size
                pred_bbox = denormalize_coordinates(pred_bbox, image_size)
                result_item['pred_bbox'] = pred_bbox
                result_item['pred_bbox_denormalized'] = True
            else:
                result_item['pred_bbox_denormalized'] = False

            # Calculate IoU
            iou = calculate_iou(pred_bbox, gt_bbox)
            is_correct = iou >= args.iou_threshold

            result_item.update({
                'iou': iou,
                'is_correct': is_correct
            })

            ious.append(iou)
            if is_correct:
                correct_predictions += 1
        else:
            result_item.update({
                'iou': 0.0,
                'is_correct': False
            })
            ious.append(0.0)

        results.append(result_item)

    # Calculate metrics

    # Divide by the number of samples actually scored, not by a hard-coded 2000: any run that
    # subsets the data (--eval_ratio, skipped samples) would otherwise be scaled down silently
    # while the output still claimed to be "based on valid samples".
    accuracy = correct_predictions / valid_samples if valid_samples > 0 else 0.0
    mean_iou = np.mean(ious) if ious else 0.0
    median_iou = np.median(ious) if ious else 0.0
    std_iou = np.std(ious) if ious else 0.0

    # Calculate valid rate
    prediction_valid_rate = valid_predictions / total_predictions if total_predictions > 0 else 0.0
    ground_truth_valid_rate = valid_ground_truths / total_predictions if total_predictions > 0 else 0.0
    sample_valid_rate = valid_samples / total_predictions if total_predictions > 0 else 0.0

    # Prepare output results - per required three-layer structure
    # Change: removed iou_distribution from secondary_metrics
    output_data = {
        'overall': {
            'score': accuracy,
            'metric': 'accuracy',
            'total_samples': total_predictions,
            'valid_samples': valid_samples,
            'sample_valid_rate': sample_valid_rate,
            'correct_predictions': correct_predictions,
            'iou_threshold': args.iou_threshold
        },
        'secondary_metrics': {
            'prediction_valid_rate': prediction_valid_rate,
            'ground_truth_valid_rate': ground_truth_valid_rate,
            'mean_iou': float(mean_iou),
            'median_iou': float(median_iou),
            'std_iou': float(std_iou),
            'iou_0.9': len([iou for iou in ious if iou >= 0.9]),
            'iou_0.7': len([iou for iou in ious if iou >= 0.7]),
            'iou_0.5': len([iou for iou in ious if iou >= 0.5]),
            'iou_0.3': len([iou for iou in ious if iou >= 0.3]),
            'iou_0.1': len([iou for iou in ious if iou >= 0.1]),
            # Removed iou_distribution section
        },
        'detailed_results': results
    }

    return output_data

def save_results(args, output_data):
    """Save evaluation results"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create model subdirectory
    model_output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Generate output filename - use basename of input JSONL file plus _scores.json
    input_filename = os.path.basename(args.infer_file_path)
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f"{base_name}_scores.json"
    output_path = os.path.join(model_output_dir, output_filename)

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Evaluation results saved to: {output_path}")
    return output_path

def main():
    # Parse arguments
    args = parse_arguments()

    print(f"Starting evaluation for model: {args.model_name}")
    print(f"Input file: {args.infer_file_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"R1-type evaluation enabled: {args.r1_type}")
    print(f"Denormalization needed: {args.model_name.lower() in [m.strip().lower() for m in args.denormalize_models.split(',')]}")
    print(f"Default image size: {args.default_image_size}")

    # Execute evaluation
    results = evaluate_grounding_predictions(args)

    # Save results
    output_path = save_results(args, results)

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Total samples: {results['overall']['total_samples']}")
    print(f"Valid samples: {results['overall']['valid_samples']} (valid rate: {results['overall']['sample_valid_rate']:.2%})")
    print(f"Prediction box parse success rate: {results['secondary_metrics']['prediction_valid_rate']:.2%}")
    print(f"Ground truth box parse success rate: {results['secondary_metrics']['ground_truth_valid_rate']:.2%}")
    print(f"Correct predictions: {results['overall']['correct_predictions']}")
    print(f"Accuracy: {results['overall']['score']:.4f} (based on valid samples)")
    print(f"Mean IoU: {results['secondary_metrics']['mean_iou']:.4f}")
    print(f"IoU distribution:")
    # Change: updated print logic, removed reference to iou_distribution
    thresholds = [0.9, 0.7, 0.5, 0.3, 0.1]
    for threshold in thresholds:
        count_key = f'iou_{threshold}'
        count = results['secondary_metrics'][count_key]
        percentage = count / results['overall']['valid_samples'] * 100 if results['overall']['valid_samples'] > 0 else 0
        print(f"  IoU>={threshold}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
