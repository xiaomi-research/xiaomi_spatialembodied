import json
import ast
import re
from pathlib import Path
import pandas as pd
import logging
from typing import List, Tuple, Set, Dict, Any
import argparse
import os
import math
import string
from collections import defaultdict
import sys

# Fix path import (keep original logic)
current_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ in locals() else os.getcwd()
base_model_dir = os.path.dirname(current_dir)
sys.path.append(base_model_dir)
# If preprocessor doesn't exist, add empty implementation to avoid errors
try:
    from eval.eval_utils.preprocessor import preprocess_model_output
except ImportError:
    def preprocess_model_output(text):
        return text  # Empty implementation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_centerness(point, bbox):
    """Compute centerness score of a point relative to a bbox"""
    x, y = point
    xmin, ymin, xmax, ymax = bbox

    if not (xmin <= x <= xmax and ymin <= y <= ymax):
        return 0.0

    left = x - xmin
    right = xmax - x
    top = y - ymin
    bottom = ymax - y

    left = max(left, 0.0)
    right = max(right, 0.0)
    top = max(top, 0.0)
    bottom = max(bottom, 0.0)

    lr_min = min(left, right)
    lr_max = max(left, right)
    tb_min = min(top, bottom)
    tb_max = max(top, bottom)

    lr_ratio = lr_min / lr_max if lr_max != 0 else 1.0
    tb_ratio = tb_min / tb_max if tb_max != 0 else 1.0

    return math.sqrt(lr_ratio * tb_ratio)

def check_in(gt, answer):
    """Fix: remove word boundary, use containment matching instead (more lenient)"""
    gt = normalize(gt)
    answer = normalize(answer)
    return gt in answer

_ARTICLE_RE = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
_PUNCT_TABLE = str.maketrans('', '', string.punctuation)

def normalize(text: str) -> str:
    """Normalize text: remove punctuation, articles, lowercase, extra spaces"""
    if not text:
        return ""
    text = text.strip().lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLE_RE.sub('', text)
    text = ' '.join(text.split())
    return text

def check_correct(answer, gt):
    """Check if answer is correct (match after normalization)"""
    return normalize(answer) == normalize(gt)

def extract_answer_from_tag(text):
    """Extract answer content from <answer> tags"""
    if not text:
        return ""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def load_dataset_json(dataset_json_path: Path) -> List[Dict[str, Any]]:
    """Load original dataset JSON file"""
    try:
        with open(dataset_json_path, 'r', encoding='utf-8') as f:
            dataset_data = json.load(f)
        logger.info(f"Loaded {len(dataset_data)} samples from {dataset_json_path}")
        return dataset_data
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_json_path}: {e}")
        return []

def process_jsonl_file(mllm_output_file: Path, dataset_path: Path = None) -> List[Dict[str, Any]]:
    """Process jsonl file, extract response and labels, and integrate dataset info"""
    mllm_output_data = []
    with open(mllm_output_file, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                mllm_output_data.append(json.loads(line))

    logger.info(f"Load {len(mllm_output_data)} mllm output from {mllm_output_file}.")

    # Load dataset (keep original logic)
    dataset_data = []
    if dataset_path:
        mllm_filename = mllm_output_file.stem
        base_name = mllm_filename.replace('_val_result', '')

        if 'yaw_vqas' in mllm_filename:
            task_name = 'yaw'
            base_name = '00_yaw_vqas'
        elif 'xy2d_vqas' in mllm_filename:
            task_name = 'xy2d'
            base_name = '01_xy2d_vqas'
        elif 'depth_vqas' in mllm_filename:
            task_name = 'depth'
            base_name = '02_depth_vqas'
        elif 'dis_vqas' in mllm_filename:
            task_name = 'dis'
            base_name = '03_dis_vqas'
        elif 'lr_vqas' in mllm_filename:
            task_name = 'lr'
            base_name = '04_lr_vqas'
        elif 'fb_vqas' in mllm_filename:
            task_name = 'fb'
            base_name = '05_fb_vqas'
        else:
            logger.warning(f"Unknown task type in file: {mllm_filename}")

        dataset_file = dataset_path / f"{base_name}.json"
        if dataset_file.exists():
            dataset_data = load_dataset_json(dataset_file)
            if len(dataset_data) != len(mllm_output_data):
                logger.warning(f"Dataset size ({len(dataset_data)}) doesn't match output size ({len(mllm_output_data)})")
        else:
            logger.warning(f"Dataset file not found: {dataset_file}")

    # Extract core information (keep original logic)
    match_mllm_output_list = []
    for idx, data in enumerate(mllm_output_data):
        try:
            pred_text = data.get('response', '')
            pred_text = preprocess_model_output(pred_text)
            response_content = extract_answer_from_tag(pred_text)
            gt_content = extract_answer_from_tag(data.get('labels', ''))

            prompt_content = data['messages'][0]['content'] if (data.get('messages') and len(data['messages'])>0) else ""
            if data['messages'][0]['role'] == 'system':

                prompt_content = data['messages'][1]['content'] if (data.get('messages') and len(data['messages'])>0) else ""

            output_dict = dict(
                vqa_idx=idx,
                output=response_content,
                gt_answer=gt_content,
                prompt=prompt_content
            )

            if dataset_data and idx < len(dataset_data):
                sample = dataset_data[idx]
                if 'obj_bbox' in sample:
                    output_dict['gt_bbox'] = sample['obj_bbox']
                if 'image_pixel' in sample:
                    output_dict['image_pixel'] = sample['image_pixel']
                if 'image_path' in sample:
                    output_dict['image_path'] = sample['image_path']
                if 'answer' in sample:
                    output_dict['dataset_answer'] = sample['answer']

            match_mllm_output_list.append(output_dict)

        except Exception as e:  # Fix: catch all exceptions to avoid losing samples
            logger.warning(f"Error processing line {idx}: {e}")
            continue

    logger.info(f"Processed {len(match_mllm_output_list)} valid outputs.")

    # Extract bbox from prompt (keep original logic)
    if not dataset_data and 'xy2d' in str(mllm_output_file):
        logger.info("Attempting to extract bbox from prompt...")
        for data in match_mllm_output_list:
            try:
                prompt = data.get('prompt', '')
                bbox_pattern = r'bbox\s*[:\s]*\[([\d,\s]+)\]'
                bbox_match = re.search(bbox_pattern, prompt, re.IGNORECASE)

                if bbox_match:
                    bbox_str = bbox_match.group(1)
                    bbox = [int(x.strip()) for x in bbox_str.split(',')]
                    if len(bbox) == 4:
                        data['gt_bbox'] = bbox
                else:
                    answer_pattern = r'<answer>\s*[\(（]?\s*(\d+)\s*,\s*(\d+)\s*[\)）]?\s*,\s*[\(（]?\s*(\d+)\s*,\s*(\d+)\s*[\)）]?\s*</answer>'
                    answer_match = re.search(answer_pattern, prompt, re.IGNORECASE)
                    if answer_match:
                        x1, y1, x2, y2 = map(int, answer_match.groups())
                        data['gt_bbox'] = [x1, y1, x2, y2]
            except Exception as e:
                logger.debug(f"Failed to extract bbox for sample {data['vqa_idx']}: {e}")
                continue

    return match_mllm_output_list

def evaluate_yaw_task(match_mllm_output_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate yaw task (fixed option matching and pairwise logic)"""
    valid_mllm_output_list = []
    for data in match_mllm_output_list:
        try:
            yaw_output = data['output']
            yaw_gt = data['gt_answer']
            prompt = data['prompt']

            # Fix: more lenient Options matching (supports arbitrary whitespace separators)
            yaw_options_match = re.search(r'Options:\s*\n(.*?)(?=\n\s*\n|\Z)', prompt, re.DOTALL)

            # breakpoint()
            if yaw_options_match:
                yaw_options = re.findall(r'-\s*(.*)', yaw_options_match.group(1))
                in_option = False
                for option in yaw_options:

                    if check_in(option, yaw_output):

                        in_option = True
                        break

                if in_option:
                    valid_mllm_output_list.append(dict(
                        vqa_idx=data['vqa_idx'],
                        output=yaw_output,
                        gt_answer=yaw_gt
                    ))
            # breakpoint()
        except Exception as e:  # Fix: catch all exceptions
            logger.warning(f"Error processing yaw sample {data.get('vqa_idx', 'unknown')}: {e}")
            continue
    logger.info(f"Get {len(valid_mllm_output_list)} valid vlm output for yaw task.")

    # Fix: pairwise evaluation logic (no longer relies on strictly consecutive indices)
    tmp_idx = 0
    correct_cnt = 0
    yaw_valid_num = 0
    yaw_qa_sum = len(match_mllm_output_list) // 2

    # Build vqa_idx to sample mapping for easy lookup
    valid_idx_map = {item['vqa_idx']: item for item in valid_mllm_output_list}

    for answer_idx in range(yaw_qa_sum):
        idx1 = 2 * answer_idx
        idx2 = 2 * answer_idx + 1

        # Check if both indices exist in valid samples
        if idx1 in valid_idx_map and idx2 in valid_idx_map:
            yaw_valid_num += 1
            sample1 = valid_idx_map[idx1]
            sample2 = valid_idx_map[idx2]

            if check_correct(sample1['output'], sample1['gt_answer']) and check_correct(sample2['output'], sample2['gt_answer']):
                correct_cnt += 1

    # Calculate score
    yaw_correct_rate = 0.0 if yaw_valid_num == 0 else correct_cnt / yaw_valid_num
    yaw_score = correct_cnt / yaw_qa_sum if yaw_qa_sum > 0 else 0.0
    yaw_valid_rate = yaw_valid_num / yaw_qa_sum if yaw_qa_sum > 0 else 0.0

    return {
        "yaw_valid_response_accuracy": yaw_correct_rate,
        "yaw_score": yaw_score,
        "yaw_valid_response_rate": yaw_valid_rate,
        "yaw_total_pairs": yaw_qa_sum,
        "yaw_valid_pairs": yaw_valid_num,
        "yaw_correct_pairs": correct_cnt
    }

def evaluate_xy2d_task(match_mllm_output_list: List[Dict[str, Any]], de_normalized: bool = False) -> Dict[str, Any]:
    """Evaluate xy2d task (keep original logic, this task works fine)"""
    valid_mllm_output_list = []

    for data in match_mllm_output_list:
        try:
            xy2d_output = data['output']

            pattern = r'\[\s*([-\d.,\s]+)\s*\]'
            match = re.search(pattern, xy2d_output)
            if not match:
                answer_pattern = r'\s*[\(（]?\s*(\d+)\s*,\s*(\d+)\s*[\)）]?\s*,\s*[\(（]?\s*(\d+)\s*,\s*(\d+)\s*[\)）]?\s*'
                answer_match = re.search(answer_pattern, xy2d_output, re.IGNORECASE)
                if answer_match:
                    x1, y1, x2, y2 = map(int, answer_match.groups())
                    xy2d_output_list = [x1, y1, x2, y2]
                else:
                    raise ValueError(f"No match found for pattern: {pattern} in output: {xy2d_output}")
            else:
                xy2d_output_list = ast.literal_eval(match.group(0))
                if isinstance(xy2d_output_list, tuple):
                    x1, y1 = xy2d_output_list[0]
                    x2, y2 = xy2d_output_list[1]
                    xy2d_output_list = [x1, y1, x2, y2]

            # Get image size
            if 'image_pixel' in data:
                image_size_str = data['image_pixel']
                if 'x' in image_size_str:
                    image_width, image_height = map(int, image_size_str.split('x'))
                else:
                    size_parts = re.findall(r'\d+', image_size_str)
                    if len(size_parts) >= 2:
                        image_width, image_height = int(size_parts[0]), int(size_parts[1])
                    else:
                        raise ValueError(f"Cannot parse image_pixel: {image_size_str}")
            else:
                image_match = re.search(r'image_pixel[:\s]*(\d+)x(\d+)', data['prompt'])
                if image_match:
                    image_width, image_height = int(image_match.group(1)), int(image_match.group(2))
                else:
                    image_width, image_height = 1600, 900

            if len(xy2d_output_list) == 2:
                x, y = xy2d_output_list
            elif len(xy2d_output_list) == 4:
                x1, y1, x2, y2 = xy2d_output_list
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
            else:
                raise ValueError(f"Invalid coordinate format: {xy2d_output_list}")

            x = x * image_width if x < 1 else x
            y = y * image_height if y < 1 else y
            if de_normalized:
                x = x * image_width / 1000
                y = y * image_height / 1000



            if 0 <= x < image_width and 0 <= y < image_height:
                if 'gt_bbox' in data:
                    gt_bbox = data['gt_bbox']
                else:
                    gt_bbox_match = re.search(r'\[\s*([-\d.,\s]+)\s*\]', data['gt_answer'])
                    if gt_bbox_match:
                        gt_bbox = ast.literal_eval(gt_bbox_match.group(0))
                    else:
                        logger.warning(f"No gt_bbox found for sample {data['vqa_idx']}")
                        continue

                if len(gt_bbox) == 4 or len(gt_bbox) == 2:
                    valid_mllm_output_list.append(dict(
                        vqa_idx=data['vqa_idx'],
                        output=[x, y],
                        gt_bbox=gt_bbox,
                        image_size=[image_width, image_height]
                    ))
                else:
                    raise ValueError(f"Invalid gt_bbox format: {gt_bbox}")
        except Exception as e:
            logger.warning(f"Error processing xy2d sample {data.get('vqa_idx', 'unknown')}: {e}")
            continue

    logger.info(f"Get {len(valid_mllm_output_list)} valid mllm output for xy2d task.")

    tmp_idx = 0
    score_cnt = 0
    xy2d_valid_num = 0
    xy2d_qa_sum = len(match_mllm_output_list)

    valid_idx_map = {item['vqa_idx']: item for item in valid_mllm_output_list}
    for answer_idx in range(xy2d_qa_sum):
        if answer_idx in valid_idx_map:
            xy2d_valid_num += 1
            obj_box = valid_idx_map[answer_idx]['gt_bbox']
            output_point = valid_idx_map[answer_idx]['output']
            centerness_score = compute_centerness(output_point, obj_box)
            score_cnt += centerness_score

    xy2d_correct_rate = 0.0 if xy2d_valid_num == 0 else score_cnt / xy2d_valid_num
    xy2d_score = score_cnt / xy2d_qa_sum if xy2d_qa_sum > 0 else 0.0
    xy2d_valid_rate = xy2d_valid_num / xy2d_qa_sum if xy2d_qa_sum > 0 else 0.0

    return {
        "xy2d_valid_response_accuracy": xy2d_correct_rate,
        "xy2d_score": xy2d_score,
        "xy2d_valid_response_rate": xy2d_valid_rate,
        "xy2d_total_questions": xy2d_qa_sum,
        "xy2d_valid_questions": xy2d_valid_num,
        "xy2d_total_centerness": score_cnt
    }

def evaluate_single_choice_task(match_mllm_output_list: List[Dict[str, Any]], task_name: str) -> Dict[str, Any]:
    """Evaluate single choice task (depth/dis/lr/fb) - fixed option matching and pairwise logic"""
    valid_mllm_output_list = []
    for data in match_mllm_output_list:
        try:
            output = data['output']
            gt = data['gt_answer']
            prompt = data['prompt']

            # Fix: more lenient Options matching
            options_match = re.search(r'Options:\s*\n(.*?)(?=\n\s*\n|\Z)', prompt, re.DOTALL)
            if options_match:
                options = re.findall(r'-\s*(.*)', options_match.group(1))
                in_option = False
                for option in options:
                    if check_in(option, output):
                        in_option = True
                        break
                if in_option:
                    valid_mllm_output_list.append(dict(
                        vqa_idx=data['vqa_idx'],
                        output=output,
                        gt_answer=gt
                    ))
        except Exception as e:
            logger.warning(f"Error processing {task_name} sample {data.get('vqa_idx', 'unknown')}: {e}")
            continue
    logger.info(f"Get {len(valid_mllm_output_list)} valid mllm output for {task_name}.")

    # Fix: pairwise/single evaluation logic (use index mapping, no longer rely on order)
    is_pair_task = task_name in ['dis', 'lr', 'fb']
    correct_cnt = 0
    valid_num = 0
    qa_sum = len(match_mllm_output_list) // 2 if is_pair_task else len(match_mllm_output_list)

    # Build index mapping for valid samples
    valid_idx_map = {item['vqa_idx']: item for item in valid_mllm_output_list}

    if is_pair_task:
        # Pairwise evaluation
        for answer_idx in range(qa_sum):
            idx1 = 2 * answer_idx
            idx2 = 2 * answer_idx + 1

            if idx1 in valid_idx_map and idx2 in valid_idx_map:
                valid_num += 1
                sample1 = valid_idx_map[idx1]
                sample2 = valid_idx_map[idx2]

                if check_correct(sample1['output'], sample1['gt_answer']) and check_correct(sample2['output'], sample2['gt_answer']):
                    correct_cnt += 1
    else:
        # Single evaluation (depth)
        for answer_idx in range(qa_sum):
            if answer_idx in valid_idx_map:
                valid_num += 1
                sample = valid_idx_map[answer_idx]
                if check_correct(sample['output'], sample['gt_answer']):
                    correct_cnt += 1

    # Calculate score
    correct_rate = 0.0 if valid_num == 0 else correct_cnt / valid_num
    score = correct_cnt / qa_sum if qa_sum > 0 else 0.0
    valid_rate = valid_num / qa_sum if qa_sum > 0 else 0.0

    prefix = f"{task_name}_"
    return {
        f"{prefix}valid_response_accuracy": correct_rate,
        f"{prefix}score": score,
        f"{prefix}valid_response_rate": valid_rate,
        f"{prefix}total_questions": qa_sum * 2 if is_pair_task else qa_sum,
        f"{prefix}valid_questions": valid_num * 2 if is_pair_task else valid_num,
        f"{prefix}correct_answers": correct_cnt * 2 if is_pair_task else correct_cnt,
        f"{prefix}is_pair_task": is_pair_task
    }

def main(args):
    """Main function (keep original logic)"""
    infer_file_path = Path(args.infer_file_path)
    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    output_dir = Path(args.output_dir)
    model_name = args.model_name

    # Backward compatibility with old parameters
    if not os.path.exists(infer_file_path) and hasattr(args, 'eval_root_dir'):
        eval_root_dir = Path(args.eval_root_dir)
        logger.warning(f"Using deprecated parameter 'eval_root_dir' as infer_file_path")
        infer_file_path = eval_root_dir

    if not dataset_path and hasattr(args, 'vqas_dir'):
        vqas_dir = Path(args.vqas_dir)
        logger.warning(f"Using deprecated parameter 'vqas_dir' as dataset_path")
        dataset_path = vqas_dir

    if not model_name and hasattr(args, 'eval_model_path'):
        model_name = args.eval_model_path
        logger.warning(f"Using deprecated parameter 'eval_model_path' as model_name")

    logger.info(f"Processing model: {model_name}")
    logger.info(f"Infer file path: {infer_file_path}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Output directory: {output_dir}")

    save_dir = output_dir / model_name
    save_dir.mkdir(exist_ok=True, parents=True)

    # Read files
    if infer_file_path.is_file():
        mllm_output_files = [infer_file_path]
    else:
        mllm_output_files = sorted(infer_file_path.glob("*.jsonl"))
        if not mllm_output_files:
            logger.warning(f"No jsonl files found in {infer_file_path}, trying json files...")
            mllm_output_files = sorted(infer_file_path.glob("*.json"))

    if not mllm_output_files:
        logger.error(f"No JSONL/JSON files found in {infer_file_path}")
        return

    # Initialize results
    all_results = {}
    task_results = {}
    csv_results = []

    for mllm_output_file in mllm_output_files:
        logger.info(f"Processing file: {mllm_output_file.name}")

        match_mllm_output_list = process_jsonl_file(mllm_output_file, dataset_path)

        mllm_output_file_name = mllm_output_file.name

        # Determine task type
        if 'yaw_vqas' in mllm_output_file_name:
            task_name = 'yaw'
            task_result = evaluate_yaw_task(match_mllm_output_list)
        elif 'xy2d_vqas' in mllm_output_file_name:
            if '-vggt-' in model_name:
                de_normalized = True
            else:
                de_normalized = False
            task_name = 'xy2d'
            task_result = evaluate_xy2d_task(match_mllm_output_list, de_normalized)
        elif 'depth_vqas' in mllm_output_file_name:
            task_name = 'depth'
            task_result = evaluate_single_choice_task(match_mllm_output_list, task_name)
        elif 'dis_vqas' in mllm_output_file_name:
            task_name = 'dis'
            task_result = evaluate_single_choice_task(match_mllm_output_list, task_name)
        elif 'lr_vqas' in mllm_output_file_name:
            task_name = 'lr'
            task_result = evaluate_single_choice_task(match_mllm_output_list, task_name)
        elif 'fb_vqas' in mllm_output_file_name:
            task_name = 'fb'
            task_result = evaluate_single_choice_task(match_mllm_output_list, task_name)
        else:
            logger.warning(f"Unknown task type in file: {mllm_output_file_name}")
            continue

        task_results[task_name] = task_result

        # Prepare CSV results
        csv_result = {}
        for key, value in task_result.items():
            if task_name in key:
                csv_result[key] = value
            else:
                csv_result[f"{task_name}_{key}"] = value
        csv_results.append(csv_result)

        logger.info(f"Task {task_name} result: {task_result}")

    if not task_results:
        logger.error("No valid task results found!")
        return

    # Calculate overall metrics
    total_score = 0
    total_valid_response_accuracy = 0
    total_valid_response_rate = 0
    task_count = len(task_results)

    for task_name, result in task_results.items():
        score_key = f"{task_name}_score" if f"{task_name}_score" in result else "score"
        valid_accuracy_key = f"{task_name}_valid_response_accuracy" if f"{task_name}_valid_response_accuracy" in result else "valid_response_accuracy"
        valid_rate_key = f"{task_name}_valid_response_rate" if f"{task_name}_valid_response_rate" in result else "valid_response_rate"

        total_score += result.get(score_key, 0)
        total_valid_response_accuracy += result.get(valid_accuracy_key, 0)
        total_valid_response_rate += result.get(valid_rate_key, 0)

    overall_result = {
        "score": total_score / task_count if task_count > 0 else 0,
        "metric": "spatial-score",
        "average_score": total_score / task_count if task_count > 0 else 0,
        "average_valid_response_accuracy": total_valid_response_accuracy / task_count if task_count > 0 else 0,
        "average_valid_response_rate": total_valid_response_rate / task_count if task_count > 0 else 0,
        "total_tasks": task_count
    }

    # Build secondary metrics
    secondary_metrics = {}
    for task_name, result in task_results.items():
        score_key = f"{task_name}_score" if f"{task_name}_score" in result else "score"
        secondary_metrics[task_name] = {
            "score": result.get(score_key, 0)
        }

    # Build detailed results
    detailed_results = {}
    for task_name, result in task_results.items():
        task_metrics = {}
        task_details = {}

        for key, value in result.items():
            if any(metric in key for metric in ["score", "valid_response_accuracy", "valid_response_rate"]):
                task_metrics[key] = value
            else:
                task_details[key] = value

        detailed_results[task_name] = {
            "metrics": task_metrics,
            "details": task_details
        }

    # Final results
    all_results = {
        "overall": overall_result,
        "secondary_metrics": secondary_metrics,
        "detailed_results": detailed_results,
        "metadata": {
            "model_name": model_name,
            "input_file": str(infer_file_path),
            "dataset_path": str(dataset_path) if dataset_path else None,
            "evaluation_time": pd.Timestamp.now().isoformat()
        }
    }

    # Save results
    if infer_file_path.is_file():
        base_name = infer_file_path.stem
    else:
        base_name = infer_file_path.name if infer_file_path.name else "surds_evaluation"

    output_filename = f"{base_name}_result_scores.json"
    if '_result' not in base_name:
        output_filename = f"{base_name}_result_scores.json"
    json_output_path = save_dir / output_filename

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(f"JSON evaluation results saved to: {json_output_path}")

    # Save CSV
    csv_output_path = save_dir / "eval_result.csv"
    if csv_results:
        df = pd.DataFrame(csv_results)
        df.to_csv(csv_output_path, index=False)
        logger.info(f"CSV evaluation results saved to: {csv_output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SURDS Dataset Evaluation Results")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Output directory: {save_dir}")
    print(f"Overall Score: {overall_result['average_score']:.4f}")
    print(f"Tasks evaluated: {', '.join(task_results.keys())}")
    print(f"Number of tasks: {task_count}")
    print("\nTask-wise Scores:")
    for task_name, metrics in secondary_metrics.items():
        print(f"  {task_name.upper()}: {metrics['score']:.4f}")
    print(f"\nFiles saved:")
    print(f"  JSON: {json_output_path}")
    if csv_results:
        print(f"  CSV: {csv_output_path}")
    print("="*60)

    # Print aggregated metrics
    if csv_results:
        valid_response_accuracy_sum = 0
        score_sum = 0
        valid_response_rate_sum = 0

        for result in csv_results:
            for k, v in result.items():
                if "valid_response_accuracy" in k:
                    valid_response_accuracy_sum += v
                if "score" in k and ("valid_response" not in k):
                    score_sum += v
                if "valid_response_rate" in k:
                    valid_response_rate_sum += v

        if len(csv_results) > 0:
            avg_valid_response_accuracy = valid_response_accuracy_sum / len(csv_results)
            avg_score = score_sum / len(csv_results)
            avg_valid_response_rate = valid_response_rate_sum / len(csv_results)

            print("\nAggregated Metrics (Original Format):")
            print(f"  Average Valid Response Accuracy: {avg_valid_response_accuracy:.4f}")
            print(f"  Average Score: {avg_score:.4f}")
            print(f"  Average Valid Response Rate: {avg_valid_response_rate:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SURDS dataset VLM output.")

    # Unified parameter interface
    parser.add_argument('--infer_file_path', type=str, default='',
                        help='Path to the JSONL file containing model responses. For SURDS, this can be a directory containing multiple task files.')
    parser.add_argument('--dataset_path', type=str, default='',
                        help='Path to the original dataset directory containing JSON files with GT annotations.')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Directory where evaluation results will be saved.')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Name of the model being evaluated.')

    # Backward-compatible old parameters
    parser.add_argument('--vqas_dir', type=str, default='',
                        help='(Deprecated) Specify the folder for the VQA to use for evaluation. Use --dataset_path instead.')
    parser.add_argument('--eval_root_dir', type=str, default='',
                        help='(Deprecated) Specify the root directory for VLM output files. Use --infer_file_path instead.')
    parser.add_argument('--eval_model_path', type=str, default='',
                        help='(Deprecated) Specify the path of the model to evaluate. Use --model_name instead.')
    parser.add_argument('--save_dir', type=str, default='',
                        help='(Deprecated) Specify the directory where evaluation results will be saved. Use --output_dir instead.')

    args = parser.parse_args()

    # Check required parameters
    if not args.infer_file_path and not args.eval_root_dir:
        logger.error("Either --infer_file_path or --eval_root_dir must be provided!")
        parser.print_help()
        sys.exit(1)

    if args.output_dir == './eval_results' and args.save_dir:
        args.output_dir = args.save_dir

    if not args.model_name and args.eval_model_path:
        args.model_name = args.eval_model_path

    main(args)
