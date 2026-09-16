# UPDATE: Replace placeholder paths with your actual paths.
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
try:
    import language_evaluation
except ImportError as e:
    raise ImportError(
        "eval_drivebench_withqwenv3.py requires the `language_evaluation` package.\n"
        "  Install it (e.g. `pip install language-evaluation`) or clone the repository it "
        "comes from and add it to `sys.path`."
    ) from e
import requests
import time
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, "/path/to/toolkit-main")
try:
    from evaluate.utils import preprocess_answer
    from evaluate.prompts import (PERCEPTION_MCQ_PROMPT, PERCEPTION_VQA_PROMPT,
                                  PREDICTION_MCQ_PROMPT, PREDICTION_VQA_PROMPT,
                                  PLANNING_VQA_PROMPT, BEHAVIOR_MCQ_PROMPT)
except ImportError as e:
    # These come from the external DriveBench "toolkit-main" repository (which also holds the
    # judge prompts). There is no sensible fallback: without the prompts there is nothing to
    # send the judge.
    raise ImportError(
        "eval_drivebench_withqwenv3.py requires the DriveBench toolkit (`evaluate.utils` and "
        "`evaluate.prompts`).\n"
        "  They are not part of this repository. Clone the DriveBench toolkit and point the "
        "`sys.path.insert(...)` line above at its root directory."
    ) from e

# Data class for storing sample information
@dataclass
class SampleData:
    question: str
    question_type: str
    tag: List[int]
    scene_token: str
    frame_token: str
    answer: str  # Ground truth
    pred: str  # Model prediction
    desc: Optional[str] = None
    score: Optional[float] = None  # Qwen score or accuracy
    is_correct: Optional[bool] = None  # Whether MCQ is correct
    language_metrics: Optional[Dict] = None  # VQA language metrics (BLEU/ROUGE/CIDEr)


class QwenEvaluator:
    """Use SGLang to call Qwen model for scoring"""

    def __init__(self,
                 server_url: str,
                 server_model_name: str = "Qwen3_VL_235B_A22B_Thinking",
                 timeout: int = 120,
                 max_retries: int = 3,
                 cache_file: Optional[str] = None):
        """Initialize Qwen evaluator

        Args:
            server_url: SGLang server URL
            server_model_name: Evaluator model name
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            cache_file: Cache file path to avoid repeated computation
        """
        self.server_url = server_url
        self.server_model_name = server_model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = {}
        self.cache_file = cache_file
        # Number of judge responses whose score could not be parsed; those samples are
        # excluded from the mean rather than defaulted to 0.5.
        self.unparsed_scores = 0

        if cache_file and os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.cache = json.load(f)

    def _get_cache_key(self, prompt: str, gt_answer: str, model_answer: str, desc: str = "") -> str:
        """Generate cache key"""
        content = f"{prompt[:100]}_{gt_answer[:50]}_{model_answer[:50]}_{desc[:50] if desc else ''}"
        return hashlib.md5(content.encode()).hexdigest()

    def _call_qwen_api(self, messages: List[Dict]) -> Optional[str]:
        """Call Qwen API"""
        for attempt in range(self.max_retries):
            payload = {
                "model": self.server_model_name,
                "messages": messages,
                "max_tokens": 100,
                "temperature": 0.0
            }

            try:
                response = requests.post(
                    self.server_url,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    print(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {response.status_code}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                print(f"API call exception (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                time.sleep(2 ** attempt)

        return None

    def _parse_score(self, response: str) -> Optional[float]:
        """Parse score from model response, supports 0-100 scale and multi-criteria scoring"""

        # First try to match total score pattern
        total_score_patterns = [
            r'Total Score:\s*(\d+(?:\.\d+)?)(?:/100)?',  # Total Score: 85 or Total Score: 85/100
            r'Total Score:\s*(\d+(?:\.\d+)?)(?:/100)?',   # Total Score: 85
            r'Total Score:\s*(\d+(?:\.\d+)?)(?:/100)?',   # Total Score: 85
            r'Final Score:\s*(\d+(?:\.\d+)?)(?:/100)?',   # Final Score: 85
        ]

        response_clean = response.strip()

        for pattern in total_score_patterns:
            match = re.search(pattern, response_clean, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # Ensure score is in 0-100 range
                    if 0 <= score <= 100:
                        return score / 100.0  # Convert to 0-1 range
                    elif score > 100:
                        return 1.0  # If over 100, treat as full score
                    else:
                        return 0.0
                except ValueError:
                    continue

        # If total score not found, try parsing individual section scores and sum them
        try:
            # Parse individual section scores
            criteria_scores = []

            # 1. Answer Correctness
            ac_match = re.search(r'Answer Correctness\s*\(?[0-9]+ points?\)?:\s*(\d+(?:\.\d+)?)',
                            response_clean, re.IGNORECASE)
            if not ac_match:
                ac_match = re.search(r'1\.\s*Answer Correctness.*?(\d+(?:\.\d+)?)',
                                response_clean, re.IGNORECASE | re.DOTALL)
            if ac_match:
                criteria_scores.append(float(ac_match.group(1)))

            # 2. Behavioral Understanding and details
            bu_match = re.search(r'Behavioral Understanding.*?(\d+(?:\.\d+)?)',
                            response_clean, re.IGNORECASE | re.DOTALL)
            if not bu_match:
                bu_match = re.search(r'2\.\s*Behavioral Understanding.*?(\d+(?:\.\d+)?)',
                                response_clean, re.IGNORECASE | re.DOTALL)
            if bu_match:
                criteria_scores.append(float(bu_match.group(1)))

            # 3. Reasoning and Justification
            rj_match = re.search(r'Reasoning and Justification.*?(\d+(?:\.\d+)?)',
                            response_clean, re.IGNORECASE | re.DOTALL)
            if not rj_match:
                rj_match = re.search(r'3\.\s*Reasoning and Justification.*?(\d+(?:\.\d+)?)',
                                response_clean, re.IGNORECASE | re.DOTALL)
            if rj_match:
                criteria_scores.append(float(rj_match.group(1)))

            # 4. Contextual Relevance
            cr_match = re.search(r'Contextual Relevance.*?(\d+(?:\.\d+)?)',
                            response_clean, re.IGNORECASE | re.DOTALL)
            if not cr_match:
                cr_match = re.search(r'4\.\s*Contextual Relevance.*?(\d+(?:\.\d+)?)',
                                response_clean, re.IGNORECASE | re.DOTALL)
            if cr_match:
                criteria_scores.append(float(cr_match.group(1)))

            # 5. Clarity and Grammar
            cg_match = re.search(r'Clarity and Grammar.*?(\d+(?:\.\d+)?)',
                            response_clean, re.IGNORECASE | re.DOTALL)
            if not cg_match:
                cg_match = re.search(r'5\.\s*Clarity and Grammar.*?(\d+(?:\.\d+)?)',
                                response_clean, re.IGNORECASE | re.DOTALL)
            if cg_match:
                criteria_scores.append(float(cg_match.group(1)))

            # If all section scores found, calculate total
            if len(criteria_scores) >= 3:  # At least 3 section scores found
                total = sum(criteria_scores)
                if 0 <= total <= 100:
                    return total / 100.0
        except (ValueError, IndexError) as e:
            print(f"Error parsing section scores: {e}")

        # If all above methods fail, try generic number matching patterns
        # First try matching 100-point scale scores
        patterns_100 = [
            r'(\d+(?:\.\d+)?)\s*points?(?:\s*total)?',  # 85 points
            r'Score[：:]\s*(\d+(?:\.\d+)?)(?:\s*/?\s*100)?',    # Score: 85
            r'Result[：:]\s*(\d+(?:\.\d+)?)(?:\s*/?\s*100)?',    # Result: 85
            r'Score[：:]\s*(\d+(?:\.\d+)?)(?:\s*/?\s*100)?',   # Score: 85
        ]

        for pattern in patterns_100:
            match = re.search(pattern, response_clean, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    if 0 <= score <= 100:
                        return score / 100.0
                    elif score > 100:
                        return 1.0
                except ValueError:
                    continue

        # Finally try simple number matching
        all_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response_clean)

        # Prefer numbers that look like scores
        candidate_scores = []
        for num_str in all_numbers:
            try:
                num = float(num_str)
                if 0 <= num <= 100:  # In the 0-100 range
                    candidate_scores.append(num)
            except ValueError:
                continue

        # Only trust an unlabelled number when it is the ONLY one in range. This used to take
        # `max(candidate_scores)`, but judge responses routinely restate the scoring rubric,
        # so the largest number is often just the top of the scale -- a response echoing
        # "score between 0 and 100" was read as a perfect 100.
        if len(candidate_scores) == 1:
            return candidate_scores[0] / 100.0

        # If even simple numbers cannot be found, try parsing "85/100" format
        fraction_pattern = r'(\d+(?:\.\d+)?)\s*/\s*100'
        fraction_match = re.search(fraction_pattern, response_clean)
        if fraction_match:
            try:
                score = float(fraction_match.group(1))
                if 0 <= score <= 100:
                    return score / 100.0
            except ValueError:
                pass

        # If all methods fail, try 5-point scale as fallback
        patterns_5 = [
            r'Rating[：:]?\s*(\d+(?:\.\d+)?)/5',
            r'(\d+(?:\.\d+)?)\s*/\s*5',
            r'Rating[：:]?\s*(\d+(?:\.\d+)?)\s*points?',
        ]

        for pattern in patterns_5:
            match = re.search(pattern, response_clean)
            if match:
                try:
                    score = float(match.group(1))
                    if score <= 5:
                        return score / 5.0
                except ValueError:
                    continue

        # Return None rather than a 0.5 placeholder: the caller's aggregation drops None
        # values (see `valid_scores = [s for s in scores if s is not None]`), so an
        # unparseable sample no longer silently drags the mean to the mid-point. The counter
        # makes the failure rate visible instead of hiding it.
        self.unparsed_scores += 1
        if self.unparsed_scores <= 5 or self.unparsed_scores % 50 == 0:
            print(
                f"[WARN] could not parse a score from the judge response "
                f"({self.unparsed_scores} so far); this sample is excluded from the mean. "
                f"Response: {response[:200]}..."
            )
        return None

    def evaluate_answer(self,
                        question: str,
                        gt_answer: str,
                        model_answer: str,
                        prompt_template: str,
                        desc: str = None,
                        question_type: str = None) -> Optional[float]:
        """Evaluate answer quality"""

        cache_key = self._get_cache_key(prompt_template, gt_answer, model_answer, desc or "")

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Create unified format dictionary
        format_dict = {
            "QUESTION": question,
            "GT": gt_answer,
            "PRED": model_answer,
            "DESC": desc or "",
            "question": question,  # Keep backward compatibility
            "gt_answer": gt_answer,
            "model_answer": model_answer,
            "correct_option": gt_answer  # Compatible with perception MCQ
        }

        # Format prompt using format_dict
        try:
            prompt = prompt_template.format(**format_dict)
        except KeyError as e:
            print(f"Warning: Placeholder not found when formatting prompt: {e}")
            # Fall back to old formatting method
            if question_type == "perception":
                if "VQA" in prompt_template:
                    prompt = prompt_template.format(
                        question=question,
                        gt_answer=gt_answer,
                        model_answer=model_answer
                    )
                else:
                    prompt = prompt_template.format(
                        question=question,
                        correct_option=gt_answer,
                        model_answer=model_answer
                    )
            else:
                prompt = prompt_template.format(
                    question=question,
                    gt_answer=gt_answer,
                    model_answer=model_answer
                )

        messages = [{
            "role": "user",
            "content": prompt
        }]

        response = self._call_qwen_api(messages)
        if not response:
            return None

        score = self._parse_score(response)

        if score is not None and self.cache_file:
            self.cache[cache_key] = score
            if len(self.cache) % 10 == 0:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f, ensure_ascii=False)

        return score

    def forward(self, data_item_tuple: Tuple[Dict, str]) -> Optional[float]:
        """Evaluate a single sample (for sequential processing)"""
        data_item, prompt_template = data_item_tuple
        score = self.evaluate_answer(
            question=data_item["question"],
            gt_answer=data_item["answer"],
            model_answer=data_item["pred"],
            desc=data_item.get("desc", ""),
            prompt_template=prompt_template,
            question_type=data_item.get("question_type")
        )
        return score


class EnhancedEvaluationSuit:
    def __init__(self,
                 sglang_url: str,
                 server_model_name: str = "Qwen3_VL_235B_A22B_Thinking",
                 desc_file: str = None,
                 output_dir: str = "./eval_results/DriveBench",
                 model_name: str = "unknown_model",
                 distortion_mode: str = "normal",
                 timeout: int = 120,
                 use_cache: bool = True,
                 use_sglang: bool = True):
        """Initialize enhanced evaluation suite

        Args:
            model_name: Name of the model being evaluated
            distortion_mode: Distortion mode name, e.g. biterror, rain, fog, etc.
            use_sglang: Whether to use SGLang for Qwen scoring evaluation (when False, only compute accuracy/language metrics)
        """

        self.sglang_url = sglang_url
        self.server_model_name = server_model_name
        self.model_name = model_name
        self.distortion_mode = distortion_mode
        self.output_dir = output_dir
        self.desc_file = desc_file
        self.use_sglang = use_sglang

        # Adjustment 1: Adjust output directory structure, remove distortion_mode subdirectory
        self.final_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(self.final_output_dir, exist_ok=True)

        # Only initialize QwenEvaluator when using SGLang
        self.qwen_evaluator = None
        if self.use_sglang:
            # Adjustment 2: Cache file path adapted to new directory
            cache_file = os.path.join(self.final_output_dir, f"{distortion_mode}_qwen_eval_cache.json") if use_cache else None
            self.qwen_evaluator = QwenEvaluator(
                server_url=sglang_url,
                server_model_name=server_model_name,
                timeout=timeout,
                cache_file=cache_file
            )
            print(f"Initialized QwenEvaluator (model: {server_model_name}, cache: {cache_file})")
        else:
            print("SGLang not used, only computing accuracy and language metrics (BLEU/ROUGE/CIDEr)")

        # Initialize language metrics evaluator (always enabled)
        self.language_eval = language_evaluation.CocoEvaluator(
            coco_types=["BLEU", "ROUGE_L", "CIDEr"]
        )

        # Load visual description file
        self.original_data = {}
        if desc_file and os.path.exists(desc_file):
            try:
                with open(desc_file, 'r') as f:
                    self.original_data = json.load(f)
                print(f"Loaded visual description file: {desc_file}")
            except Exception as e:
                print(f"Failed to load visual description file: {e}")

        # Initialize data storage
        self.all_samples = []
        self.results = {
            "perception": {"MCQ": {"qwen": [], "accuracy": []}, "VQA": {"qwen": [], "language": []}},
            "prediction": {"VQA": {"qwen": [], "language": []}},
            "planning": {"VQA": {"qwen": [], "language": []}},
            "behavior": {"MCQ": {"qwen": [], "accuracy": []}}
        }
        self.detailed_results = []
        self.stats = {
            "total_samples": 0,
            "valid_samples": 0,
            "correct_samples": 0,
            "by_question_type": defaultdict(lambda: {"count": 0, "correct": 0}),
            "by_task": defaultdict(lambda: defaultdict(lambda: {"count": 0, "scores": []}))
        }

    def extract_visual_description(self, scene_token, frame_token, question):
        """Extract visual description"""
        if not self.original_data:
            return None

        scene_data = self.original_data.get(scene_token, {})
        key_frames = scene_data.get("key_frames", {})
        frame_data = key_frames.get(frame_token, {})
        key_object = frame_data.get("key_object_infos", {})

        # Special questions don't need visual description
        if 'In this scenario, what are safe actions to take for the ego vehicle?' in question:
            return "No visual description needed for this question."

        # Extract object ID
        object_id_match = re.search(r'<(.*)>', question)
        if not object_id_match:
            return None

        object_id = object_id_match.group(1)
        object_id_prefix = ','.join(object_id.split(',')[:2])

        # Match corresponding visual description
        matching_key = next(
            (k for k in key_object if k.startswith(f"<{object_id_prefix}")),
            None
        )

        if matching_key:
            return key_object[matching_key].get('Visual_description', '').lower()
        return None

    def load_and_merge_data(self, infer_file_path: str, dataset_path: str, eval_ratio: float) -> List[Dict]:
        """Load and merge prediction file and metadata file"""
        print(f"\nLoading data files:")
        print(f"  - Prediction file: {infer_file_path}")
        print(f"  - Metadata file: {dataset_path}")

        # Load prediction results
        predictions = []
        try:
            with open(infer_file_path, 'r') as f:
                for line in tqdm(f, desc="Loading predictions"):
                    if line.strip():
                        predictions.append(json.loads(line.strip()))
        except Exception as e:
            print(f"Failed to load prediction file: {e}")
            return []

        # Load metadata
        metadata = []
        try:
            with open(dataset_path, 'r') as f:
                for line in tqdm(f, desc="Loading metadata"):
                    if line.strip():
                        metadata.append(json.loads(line.strip()))
        except Exception as e:
            print(f"Failed to load metadata file: {e}")
            return []

        # Sample by ratio
        if eval_ratio < 1.0:
            pred_len = len(predictions)
            meta_len = len(metadata)
            predictions = predictions[:round(pred_len * eval_ratio)]
            metadata = metadata[:round(meta_len * eval_ratio)]
            print(f"After sampling by ratio: predictions {len(predictions)}/{pred_len}, metadata {len(metadata)}/{meta_len}")

        # Align data length
        min_len = min(len(predictions), len(metadata))
        if len(predictions) != len(metadata):
            print(f"Prediction file and metadata file row count mismatch, truncated to minimum length: {min_len}")
            predictions = predictions[:min_len]
            metadata = metadata[:min_len]

        # Merge data
        merged_data = []
        for i, (pred, meta) in enumerate(zip(predictions, metadata)):
            # Extract question.
            # `convert_drivebench_val.py` emits messages as [system, user, assistant], so
            # `messages[0]` is the constant system prompt -- using it sent the judge the same
            # boilerplate for every sample and dropped the actual question, while the ground
            # truth was still read correctly, giving a plausible-looking but meaningless score.
            question = ""
            if "question" in meta and meta["question"]:
                question = meta["question"]
            elif "messages" in meta and meta["messages"]:
                question = next(
                    (
                        msg.get("content", "")
                        for msg in meta["messages"]
                        if isinstance(msg, dict) and msg.get("role") == "user"
                    ),
                    "",
                )
                if not question:
                    # Last resort: some metadata files carry a single-message record.
                    question = meta["messages"][0].get("content", "")
            if not question:
                question = f"Question {i}"

            # Extract ground truth answer
            answer = ""
            if "messages" in meta and len(meta["messages"]) > 1:
                for message in meta['messages']:
                    if message['role'] == 'assistant':
                        answer = message['content']

            elif "answer" in meta:
                answer = meta["answer"]

            # Extract model prediction
            model_answer = ""
            if "response" in pred:
                model_answer = pred["response"]
            elif "pred" in pred:
                model_answer = pred["pred"]

            # Build merged data item
            merged_item = {
                "question": question,
                "question_type": meta.get("question_type", "unknown"),
                "tag": [meta["tag"]] if isinstance(meta.get("tag"), int) else meta.get("tag", []),
                "scene_token": meta.get("scene_token", f"scene_{i}"),
                "frame_token": meta.get("frame_token", f"frame_{i}"),
                "answer": answer,
                "pred": model_answer
            }

            # Extract visual description
            merged_item["desc"] = self.extract_visual_description(
                merged_item["scene_token"],
                merged_item["frame_token"],
                merged_item["question"]
            )

            merged_data.append(merged_item)

        print(f"Successfully merged {len(merged_data)} samples")
        return merged_data

    def forward(self, data_item):
        """Route QA pair to corresponding evaluation bucket"""

        question_type = data_item.get("question_type", "")
        tag_list = data_item.get("tag", [])

        # Create sample data object
        sample_data = SampleData(
            question=data_item["question"],
            question_type=question_type,
            tag=tag_list,
            scene_token=data_item["scene_token"],
            frame_token=data_item["frame_token"],
            answer=data_item["answer"],
            pred=data_item["pred"],
            desc=data_item.get("desc")
        )
        self.all_samples.append(sample_data)

        # Update statistics
        self.stats["total_samples"] += 1
        self.stats["by_question_type"][question_type]["count"] += 1

        # Task routing mapping
        tag_mapping = {
            0: ("perception", "MCQ", PERCEPTION_MCQ_PROMPT),
            1: ("planning", "VQA", PLANNING_VQA_PROMPT),
            2: ("perception", "VQA", PERCEPTION_VQA_PROMPT),
            3: ("prediction", "VQA", PREDICTION_VQA_PROMPT),
            4: ("behavior", "MCQ", BEHAVIOR_MCQ_PROMPT),
            5: ("perception", "VQA", PERCEPTION_VQA_PROMPT),
        }

        # Route sample to corresponding evaluation bucket
        routed = False
        for tag in tag_list:
            if tag in tag_mapping:
                task, qtype, prompt = tag_mapping[tag]
                if self.use_sglang:  # Only add qwen evaluation data when using SGLang
                    self.results[task][qtype]["qwen"].append((data_item, prompt))
                if qtype == "MCQ":
                    self.results[task][qtype]["accuracy"].append(data_item)
                elif qtype == "VQA":
                    self.results[task][qtype]["language"].append(data_item)
                routed = True
                break

        if not routed:
            print(f"Unable to route sample (question_type={question_type}, tag={tag_list})")

    def eval_qwen_score(self, data):
        """Evaluate score using Qwen model (single-threaded)"""
        if not self.use_sglang or self.qwen_evaluator is None:
            print("Skipping Qwen scoring (SGLang not enabled)")
            return None

        print(f"\nEvaluating {len(data)} samples using Qwen...")
        scores = []
        for item in tqdm(data, desc="Qwen evaluation progress"):
            score = self.qwen_evaluator.forward(item)
            scores.append(score)

            # Update sample's Qwen score
            if score is not None:
                data_item = item[0] if isinstance(item, tuple) else item
                for sample in self.all_samples:
                    if sample.question == data_item["question"] and sample.answer == data_item["answer"]:
                        sample.score = score

        # Count valid scores
        valid_scores = [s for s in scores if s is not None]
        invalid_count = len(scores) - len(valid_scores)

        if invalid_count > 0:
            print(f"{invalid_count} samples failed Qwen evaluation")

        if not valid_scores:
            return None

        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"Qwen average score: {avg_score:.4f}")
        return avg_score

    def eval_accuracy(self, data):
        """Evaluate MCQ accuracy"""
        print(f"\nEvaluating MCQ accuracy (sample count: {len(data)})...")
        correct_count = 0
        scores = []

        for data_item in data:
            # Preprocess answers
            pred_answer = preprocess_answer(data_item["pred"])
            gt_answer = preprocess_answer(data_item["answer"])

            # Check if correct
            is_correct = pred_answer == gt_answer
            scores.append(is_correct)

            if is_correct:
                correct_count += 1

            # Update sample information
            for sample in self.all_samples:
                if sample.question == data_item["question"] and sample.answer == data_item["answer"]:
                    sample.is_correct = is_correct
                    sample.score = 1.0 if is_correct else 0.0

        # Calculate accuracy
        accuracy = correct_count / len(data) if data else 0.0

        # Update global statistics
        self.stats["valid_samples"] += len(data)
        self.stats["correct_samples"] += correct_count

        print(f"MCQ accuracy: {accuracy:.4f} ({correct_count}/{len(data)})")
        return accuracy

    def eval_language(self, data):
        """Evaluate VQA language metrics (BLEU, ROUGE_L, CIDEr)"""
        print(f"\nEvaluating VQA language metrics (sample count: {len(data)})...")
        if not data:
            return {}

        # Extract predictions and ground truth answers
        pred_list = [data_item['pred'] for data_item in data]
        gt_list = [[data_item['answer']] for data_item in data]  # CocoEvaluator requires 2D list

        # Compute language metrics
        results = self.language_eval.run_evaluation(pred_list, gt_list)

        # Format results
        language_metrics = {
            "BLEU": float(results.get('BLEU', 0.0)),
            "ROUGE_L": float(results.get('ROUGE_L', 0.0)),
            "CIDEr": float(results.get('CIDEr', 0.0))
        }

        # Update sample language metrics
        for i, data_item in enumerate(data):
            for sample in self.all_samples:
                if sample.question == data_item["question"] and sample.answer == data_item["answer"]:
                    sample.language_metrics = language_metrics
                    # VQA uses ROUGE_L as primary score
                    sample.score = language_metrics["ROUGE_L"]

        # Update global statistics
        self.stats["valid_samples"] += len(data)

        print(f"Language metrics: BLEU={language_metrics['BLEU']:.4f}, ROUGE_L={language_metrics['ROUGE_L']:.4f}, CIDEr={language_metrics['CIDEr']:.4f}")
        return language_metrics

    def save_detailed_results(self, filename: str = None):
        """Save detailed results to JSON (adapted to new directory structure)"""
        # Adjustment 3: Detailed results filename adapted to distortion mode
        filename = filename or f"{self.distortion_mode}_detailed_results.json"
        filepath = os.path.join(self.final_output_dir, filename)

        # Convert sample data to dictionary
        detailed_results = []
        for sample in self.all_samples:
            sample_dict = asdict(sample)
            # Handle None values
            for key, value in sample_dict.items():
                if value is None:
                    sample_dict[key] = ""
            detailed_results.append(sample_dict)

        # Save file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to: {filepath}")
        return filepath

    def save_scores_json(self, scores: Dict, infer_file_path: str, output_dir: str = None):
        """
        Final score calculation rules:
        1. Task-level weights: perception(0.5) > prediction(0.2) = planning(0.2) > behavior(0.1)
        2. Question type weights: MCQ(0.6) > VQA(0.4)
        3. First calculate MCQ+VQA weighted within each task, then calculate weighted across tasks
        """
        output_dir = output_dir or self.final_output_dir
        output_filename = f"{self.distortion_mode}_ms_swift_val_result.json"
        output_path = os.path.join(output_dir, output_filename)

        overall = {
            "score": 0.0,
            "metric": "",
            "total_samples": self.stats["total_samples"],
            "valid_samples": self.stats["valid_samples"],
            "correct_samples": self.stats["correct_samples"]
        }

        # ====================== Weight configuration ======================
        # 1. Task-level weights
        task_weights = {
            "perception": 0.7,
            "prediction": 0.1,
            "planning": 0.1,
            "behavior": 0.1
        }

        # 2. Question type weights: MCQ is larger
        subtype_weights = {
            "MCQ": 0.8,
            "VQA": 0.2
        }
        # ======================================================

        secondary_metrics = {}
        task_weighted_scores = {}  # Score for each task after internal MCQ/VQA weighting

        for task, qtypes in scores.items():
            secondary_metrics[task] = {}
            mcq_score = None
            vqa_score = None

            for qtype, metrics in qtypes.items():
                secondary_metrics[task][qtype] = {}

                # Copy metrics
                for metric_name, metric_value in metrics.items():
                    if metric_value is None:
                        secondary_metrics[task][qtype][metric_name] = None
                    elif isinstance(metric_value, dict):
                        secondary_metrics[task][qtype][metric_name] = {k: float(v) for k, v in metric_value.items()}
                    else:
                        secondary_metrics[task][qtype][metric_name] = float(metric_value)

                # Get core score for current question type
                if qtype == "MCQ":
                    # MCQ uses accuracy
                    if "accuracy" in metrics:
                        mcq_score = metrics["accuracy"]
                elif qtype == "VQA":
                    # VQA uses ROUGE_L
                    if "language_metrics" in metrics:
                        vqa_score = metrics["language_metrics"]["ROUGE_L"]

            # Calculate internal task weighting: MCQ(0.6) + VQA(0.4) weighted
            valid_sub_scores = []
            valid_sub_weights = []
            if mcq_score is not None:
                valid_sub_scores.append(mcq_score)
                valid_sub_weights.append(subtype_weights["MCQ"])
            if vqa_score is not None:
                valid_sub_scores.append(vqa_score)
                valid_sub_weights.append(subtype_weights["VQA"])

            if valid_sub_scores:
                # Normalize weights to avoid errors when only one question type exists
                sum_w = sum(valid_sub_weights)
                task_sub_score = sum(s * w for s, w in zip(valid_sub_scores, valid_sub_weights)) / sum_w
                task_weighted_scores[task] = task_sub_score
            else:
                task_weighted_scores[task] = 0.0

        # Finally: weighted average across tasks
        weighted_total = 0.0
        total_weight = 0.0
        for task, weight in task_weights.items():
            if task in task_weighted_scores:
                weighted_total += task_weighted_scores[task] * weight
                total_weight += weight

        if total_weight > 0:
            overall["score"] = float(weighted_total / total_weight)
        else:
            overall["score"] = 0.0

        # Label metric name as weighted-mixed
        overall["metric"] = "weighted-mixed-task-subtype"

        # A few example samples
        detailed_samples = []
        for idx, sample in enumerate(self.all_samples[:10]):
            sample_dict = {
                "id": idx + 1,
                "question": sample.question[:200],
                "question_type": sample.question_type,
                "tag": sample.tag,
                "ground_truth": sample.answer[:100],
                "model_prediction": sample.pred[:100],
                "score": float(sample.score) if sample.score is not None else 0.0,
                "is_correct": sample.is_correct
            }
            detailed_samples.append(sample_dict)

        # Output JSON with weights for traceability
        scores_json = {
            "overall": overall,
            "task_weights": task_weights,
            "subtype_weights": subtype_weights,
            "task_weighted_scores": {k: float(v) for k, v in task_weighted_scores.items()},
            "secondary_metrics": secondary_metrics,
            "detailed_samples": detailed_samples
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scores_json, f, indent=2, ensure_ascii=False)

        print(f"Weighted evaluation results saved to: {output_path}")
        return output_path

    def save_summary_report(self, scores: Dict, filename: str = None):
        """Save summary report (adapted to new directory structure)"""
        # Adjustment 5: Summary report filename adapted to distortion mode
        filename = filename or f"{self.distortion_mode}_summary_report.json"
        filepath = os.path.join(self.final_output_dir, filename)

        report = {
            "timestamp": datetime.now().isoformat(),
            "eval_model": self.model_name,
            "evaluator_model": self.server_model_name if self.use_sglang else None,
            "distortion_mode": self.distortion_mode,
            "use_sglang": self.use_sglang,
            "total_samples": self.stats["total_samples"],
            "valid_samples": self.stats["valid_samples"],
            "correct_samples": self.stats["correct_samples"],
            "scores": scores,
            "statistics": {}
        }

        # Calculate statistics for each task
        for task, qtypes in scores.items():
            report["statistics"][task] = {}
            for qtype, metrics in qtypes.items():
                if not metrics:
                    continue

                # Collect all scores for this task
                all_metrics = []
                for metric_name, metric_value in metrics.items():
                    if metric_value is None:
                        continue
                    if isinstance(metric_value, dict):
                        all_metrics.extend(list(metric_value.values()))
                    else:
                        all_metrics.append(metric_value)

                if all_metrics:
                    report["statistics"][task][qtype] = {
                        "count": len(self.results.get(task, {}).get(qtype, {}).get("accuracy", []) or
                                   self.results.get(task, {}).get(qtype, {}).get("language", []) or []),
                        "mean_score": float(np.mean(all_metrics)),
                        "std_score": float(np.std(all_metrics)) if len(all_metrics) > 1 else 0.0,
                        "min_score": float(np.min(all_metrics)),
                        "max_score": float(np.max(all_metrics))
                    }

        # Calculate accuracy by question type
        correct_by_type = {}
        for qtype, stats in self.stats["by_question_type"].items():
            if stats["count"] > 0:
                correct_samples = [s for s in self.all_samples
                                 if s.question_type == qtype and s.is_correct]
                correct_rate = len(correct_samples) / stats["count"] if stats["count"] > 0 else 0.0
                correct_by_type[qtype] = {
                    "count": stats["count"],
                    "correct_count": len(correct_samples),
                    "accuracy": float(correct_rate)
                }
        report["statistics"]["by_question_type"] = correct_by_type

        # Save report
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Summary report saved to: {filepath}")
        return filepath

    def save_experiment_log(self, scores: Dict, args: argparse.Namespace, filename: str = None):
        """Save experiment log (adapted to new directory structure)"""
        # Adjustment 6: Experiment log filename adapted to distortion mode
        filename = filename or f"{self.distortion_mode}_experiment_log.txt"
        filepath = os.path.join(self.final_output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DriveBench Evaluation Experiment Log\n")
            f.write("=" * 80 + "\n\n")

            # Basic information
            f.write("Experiment basic information:\n")
            f.write(f"  Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Evaluated model: {self.model_name}\n")
            f.write(f"  Distortion mode: {self.distortion_mode}\n")
            f.write(f"  Using SGLang: {'Yes' if self.use_sglang else 'No'}\n")
            if self.use_sglang:
                f.write(f"  Evaluator model: {self.server_model_name}\n")
                f.write(f"  SGLang server: {self.sglang_url}\n")
            f.write(f"  Prediction file: {args.infer_file_path}\n")
            f.write(f"  Metadata file: {args.dataset_path}\n")
            f.write(f"  Visual description file: {self.desc_file or 'Not used'}\n")
            # Adjustment 7: Output path updated in log
            f.write(f"  Output directory: {self.final_output_dir}\n")
            f.write(f"  Evaluation ratio: {args.eval_ratio}\n")
            f.write(f"  Task weights: perception=0.5, prediction=0.2, planning=0.2, behavior=0.1\n")
            f.write(f"  Question type weights: MCQ=0.6, VQA=0.4\n\n")

            # Evaluation results
            f.write("Evaluation results:\n")
            f.write("=" * 80 + "\n")
            for task, qtypes in scores.items():
                f.write(f"\n{task.upper()} task:\n")
                for qtype, metrics in qtypes.items():
                    if metrics:
                        f.write(f"  {qtype}:\n")
                        for metric, value in metrics.items():
                            if value is None:
                                f.write(f"    {metric}: Not evaluated\n")
                            elif isinstance(value, dict):
                                for sub_metric, sub_value in value.items():
                                    f.write(f"    {metric}.{sub_metric}: {sub_value:.4f}\n")
                            else:
                                f.write(f"    {metric}: {value:.4f}\n")

            # Statistics by question type
            f.write("\nStatistics by question type:\n")
            f.write("-" * 80 + "\n")
            for qtype, stats in self.stats["by_question_type"].items():
                if stats["count"] > 0:
                    correct_samples = [s for s in self.all_samples
                                     if s.question_type == qtype and s.is_correct]
                    correct_rate = len(correct_samples) / stats["count"] if stats["count"] > 0 else 0.0
                    f.write(f"  {qtype}: {stats['count']} samples, accuracy: {correct_rate:.4f}\n")

            # File list
            f.write("\nOutput file list:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Experiment log: {filepath}\n")
            f.write(f"  Detailed results: {os.path.join(self.final_output_dir, f'{self.distortion_mode}_detailed_results.json')}\n")
            f.write(f"  Summary report: {os.path.join(self.final_output_dir, f'{self.distortion_mode}_summary_report.json')}\n")
            f.write(f"  Standard evaluation results (JSON): {os.path.join(self.final_output_dir, f'{self.distortion_mode}_ms_swift_val_result.json')}\n")
            if self.use_sglang:
                f.write(f"  Evaluation cache: {os.path.join(self.final_output_dir, f'{self.distortion_mode}_qwen_eval_cache.json')}\n")

        print(f"Experiment log saved to: {filepath}")
        return filepath

    def evaluation(self, args: argparse.Namespace = None):
        """Run all evaluations and return score dictionary"""
        print("\nStarting evaluation process...")

        # Initialize score dictionary
        scores = {
            "perception": {"MCQ": {}, "VQA": {}},
            "prediction": {"VQA": {}},
            "planning": {"VQA": {}},
            "behavior": {"MCQ": {}}
        }

        # Define evaluation mapping (only includes qwen_score when use_sglang=True)
        evaluation_mapping = {
            "perception": {
                "MCQ": [("accuracy", self.eval_accuracy, "accuracy")],
                "VQA": [("language", self.eval_language, "language_metrics")]
            },
            "prediction": {
                "VQA": [("language", self.eval_language, "language_metrics")]
            },
            "planning": {
                "VQA": [("language", self.eval_language, "language_metrics")]
            },
            "behavior": {
                "MCQ": [("accuracy", self.eval_accuracy, "accuracy")]
            }
        }

        # If using SGLang, add Qwen scoring evaluation
        if self.use_sglang:
            evaluation_mapping["perception"]["MCQ"].insert(0, ("qwen", self.eval_qwen_score, "qwen_score"))
            evaluation_mapping["perception"]["VQA"].insert(0, ("qwen", self.eval_qwen_score, "qwen_score"))
            evaluation_mapping["prediction"]["VQA"].insert(0, ("qwen", self.eval_qwen_score, "qwen_score"))
            evaluation_mapping["planning"]["VQA"].insert(0, ("qwen", self.eval_qwen_score, "qwen_score"))
            evaluation_mapping["behavior"]["MCQ"].insert(0, ("qwen", self.eval_qwen_score, "qwen_score"))

        # Execute evaluation
        for task, qtypes in evaluation_mapping.items():
            for qtype, metrics in qtypes.items():
                for key, func, target in metrics:
                    data = self.results[task][qtype].get(key, [])
                    if not data:
                        print(f"{task}-{qtype}-{key} no data, skipping")
                        continue

                    # Execute evaluation
                    result = func(data)

                    # Save result
                    if result is not None:
                        scores[task][qtype][target] = result
                    else:
                        scores[task][qtype][target] = None if target == "qwen_score" else 0.0

        # Save various result files
        self.save_detailed_results()
        self.save_summary_report(scores)
        if args and hasattr(args, 'infer_file_path'):
            self.save_scores_json(scores, args.infer_file_path)
            self.save_experiment_log(scores, args)

        # Print final summary
        print("\n" + "=" * 80)
        print("Evaluation complete! Core statistics:")
        print(f"   Total samples: {self.stats['total_samples']}")
        print(f"   Valid samples: {self.stats['valid_samples']}")
        print(f"   Correct samples: {self.stats['correct_samples']}")
        if self.stats['valid_samples'] > 0:
            print(f"   Overall accuracy: {self.stats['correct_samples']/self.stats['valid_samples']:.4f}")
        print("=" * 80)

        return scores


# ========== Helper functions ==========
def extract_model_name_from_path(path: str) -> str:
    """Extract model name from file path"""
    dir_path = os.path.dirname(os.path.abspath(path))
    dir_name = os.path.basename(dir_path)

    # Model name matching patterns
    patterns = [
        r'(Qwen[^/]+)', r'(Llama[^/]+)', r'(GPT[^/]+)',
        r'(DeepSeek[^/]+)', r'([^/]+-?[0-9]*[BM])'
    ]

    for pattern in patterns:
        match = re.search(pattern, dir_name)
        if match:
            return match.group(1)

    return dir_name


def extract_distortion_mode_from_path(path: str) -> str:
    """Extract distortion mode name from file path"""
    filename = os.path.basename(path).lower()
    distortion_modes = [
        'normal', 'biterror', 'rain', 'fog', 'snow', 'motion_blur',
        'gaussian_noise', 'brightness', 'contrast', 'defocus_blur',
        'glass_blur', 'jpeg_compression', 'pixelate', 'shot_noise',
        'impulse_noise', 'elastic_transform', 'spatter', 'frost', 'zoom_blur'
    ]

    for mode in distortion_modes:
        if mode in filename:
            return mode

    return 'normal' if 'val' in filename or 'test' in filename else 'unknown'


def extract_eval_set_from_path(path: str) -> str:
    """Extract evaluation set (val/test) from file path"""
    path_lower = path.lower()
    if 'val' in path_lower:
        return 'val'
    elif 'test' in path_lower:
        return 'test'
    return 'unknown'


def get_eval_model_info(infer_file_path: str) -> Tuple[str, str, str]:
    """Extract evaluation information from prediction file path"""
    return (
        extract_model_name_from_path(infer_file_path),
        extract_distortion_mode_from_path(infer_file_path),
        extract_eval_set_from_path(infer_file_path)
    )


# ========== Main function ==========
def main():
    parser = argparse.ArgumentParser(description='DriveBench evaluation tool (supports skipping SGLang + dual-layer weighting)')

    # Core parameters
    parser.add_argument('--infer_file_path', type=str, required=True,
                       help='Model inference output file path (.jsonl format)')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Original dataset file path (.jsonl format)')

    # SGLang-related parameters
    parser.add_argument('--sglang_url', type=str,
                       default="http://<server-ip>:8000/v1/chat/completions",
                       help='SGLang server URL')
    parser.add_argument('--server_model_name', type=str,
                       default="Qwen3_VL_235B_A22B_Thinking",
                       help='Evaluator model name')
    parser.add_argument('--use_sglang', action='store_true', default=False,
                       help='Whether to use SGLang for Qwen scoring evaluation (default: No)')
    parser.add_argument('--no_sglang', action='store_false', dest='use_sglang',
                       help='Do not use SGLang (only compute accuracy/language metrics)')

    # Evaluation configuration parameters
    parser.add_argument('--model_name', type=str, default=None,
                       help='Name of the model being evaluated (automatically extracted from path)')
    parser.add_argument('--distortion_mode', type=str, default=None,
                       help='Distortion mode name (automatically extracted from path)')
    parser.add_argument('--eval_set', type=str, default=None,
                       help='Evaluation set name (automatically extracted from path)')
    parser.add_argument('--desc_file', type=str,
                       default="/path/to/data/visual_description.json",
                       help='Visual description file path')
    parser.add_argument('--output_dir', type=str,
                       default="/path/to/eval_results/DriveBench",
                       help='Output directory path')
    parser.add_argument('--timeout', type=int, default=120,
                       help='SGLang request timeout in seconds')
    parser.add_argument('--no_cache', action='store_true',
                       help='Disable Qwen evaluation cache')
    parser.add_argument('--eval_ratio', type=float, default=1.0,
                       help='Evaluation sample ratio (for debugging)')

    args = parser.parse_args()

    # Automatically extract evaluation information
    if args.model_name is None or args.distortion_mode is None or args.eval_set is None:
        model_name, distortion_mode, eval_set = get_eval_model_info(args.infer_file_path)
        args.model_name = args.model_name or model_name
        args.distortion_mode = args.distortion_mode or distortion_mode
        args.eval_set = args.eval_set or eval_set

    # Print configuration information
    print("=" * 80)
    print("DriveBench Evaluation System v2.2 (dual-layer weighted scoring)")
    print("=" * 80)
    print(f"Evaluated model: {args.model_name}")
    print(f"Distortion mode: {args.distortion_mode}")
    print(f"Evaluation set: {args.eval_set}")
    print(f"Using SGLang: {'Yes' if args.use_sglang else 'No'}")
    if args.use_sglang:
        print(f"Evaluator model: {args.server_model_name}")
        print(f"SGLang server: {args.sglang_url}")
    print(f"Inference file: {args.infer_file_path}")
    print(f"Dataset file: {args.dataset_path}")
    print(f"Visual description file: {args.desc_file if os.path.exists(args.desc_file) else 'Does not exist'}")
    print(f"Output directory: {os.path.join(args.output_dir, args.model_name)}")
    print(f"Evaluation ratio: {args.eval_ratio}")
    print("Task weights: perception=0.5, prediction=0.2, planning=0.2, behavior=0.1")
    print("Question type weights: MCQ=0.6, VQA=0.4")
    print("=" * 80)

    # Initialize evaluator
    evaluator = EnhancedEvaluationSuit(
        sglang_url=args.sglang_url,
        server_model_name=args.server_model_name,
        desc_file=args.desc_file if os.path.exists(args.desc_file) else None,
        output_dir=args.output_dir,
        model_name=args.model_name,
        distortion_mode=args.distortion_mode,
        timeout=args.timeout,
        use_cache=not args.no_cache and args.use_sglang,
        use_sglang=args.use_sglang
    )

    # Load and merge data
    merged_data = evaluator.load_and_merge_data(args.infer_file_path, args.dataset_path, args.eval_ratio)
    if not merged_data:
        print("No valid data loaded, exiting evaluation")
        return

    # Route data to evaluation buckets
    print("\nRouting data to evaluation buckets...")
    for data_item in tqdm(merged_data, desc="Processing data"):
        evaluator.forward(data_item)

    # Execute evaluation
    final_scores = evaluator.evaluation(args)

    # Print final results
    print("\nFinal evaluation results (dual-layer weighted):")
    for task, qtypes in final_scores.items():
        print(f"\n{task.upper()}:")
        for qtype, metrics in qtypes.items():
            if metrics:
                print(f"  {qtype}:")
                for metric, value in metrics.items():
                    if value is None:
                        print(f"    {metric}: Not evaluated")
                    elif isinstance(value, dict):
                        for sub_metric, sub_value in value.items():
                            print(f"    {metric}.{sub_metric}: {sub_value:.4f}")
                    else:
                        print(f"    {metric}: {value:.4f}")

if __name__ == '__main__':
    main()
