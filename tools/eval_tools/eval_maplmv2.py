# UPDATE: Replace placeholder paths with your actual paths.
import re, os
import argparse
import json
import random
import logging
from typing import Any, Dict, List
from datetime import datetime

try:
    from nltk.translate.bleu_score import sentence_bleu
except ImportError:
    print("Warning: nltk not installed. Open-ended questions will not be evaluated.")
    sentence_bleu = None

import sys
sys.path.append("/path/to/MAPLM-main/baseline/evaluation")

try:
    from utilsv2 import (
        acc_counter,
        compute_acc,
        ensure_dir,
        get_output_paths,
        load_jsonl_as_list,
        parse_user_content,
        parse_question_key,
        decide_task_type,
        label_to_choice_index,
        completion_to_answer,
        safe_json_dump,
        setup_logger,
        calculate_frame_accuracy,
        build_unified_scores_json,
        extract_lane_id,
        calculate_maplmv2_primary_metrics,
        maybe_extract_scalar_open_answer,   # NEW
    )
except ImportError as e:
    # `utilsv2` ships with the MAPLM baseline evaluation code; it is NOT part of this repo.
    raise ImportError(
        "eval_maplmv2.py requires the MAPLM `utilsv2` module. It is not part of this "
        "repository.\n  Clone MAPLM and point the `sys.path.append('/path/to/MAPLM-main/"
        "baseline/evaluation')` line above at it."
    ) from e

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


def extract_answer_from_tag(text: str) -> str:
    """Extract answer content from <answer> tags"""
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def _auto_bleu_weights(ref_tokens: List[str], hyp_tokens: List[str], default_4gram=(0.25, 0.25, 0.25, 0.25)):
    """
    NEW: BLEU weight adaptive order reduction to avoid 0 or very small values for short texts.
    n = min(4, len(ref), len(hyp))
    """
    n = min(4, len(ref_tokens), len(hyp_tokens))
    if n <= 0:
        return (1.0, 0.0, 0.0, 0.0)
    if n == 1:
        return (1.0, 0.0, 0.0, 0.0)
    if n == 2:
        return (0.5, 0.5, 0.0, 0.0)
    if n == 3:
        return (1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0)
    return default_4gram


def calculate_bleu_score(reference: str,
                         hypothesis: str,
                         weights=(0.25, 0.25, 0.25, 0.25),
                         logger: logging.Logger = None) -> float:
    """
    Open-ended still uses BLEU (no logic change)
    Only two enhancements:
      1) Use smoothing
      2) NEW: Automatically reduce weight order for short texts (BLEU-1/2/3), avoiding 0 or very small values for 2 vs 2
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    if not reference or not hypothesis:
        return 0.0

    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0

    # NEW: Auto order-reduced weights
    auto_weights = _auto_bleu_weights(ref_tokens, hyp_tokens, default_4gram=weights)

    try:
        smoothing_function = SmoothingFunction().method1
        bleu = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights=auto_weights,
            smoothing_function=smoothing_function
        )
        return float(bleu)
    except Exception as e:
        if logger:
            logger.debug(f"BLEU calculation failed: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="MapLMv2 evaluation script (unified output format + DES/FRM/QNS + open scalar parsing)"
    )

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Original meta jsonl (aligned line-by-line with infer jsonl)")
    parser.add_argument("--infer_file_path", type=str, required=True,
                        help="ms-swift inference output jsonl (contains response/labels)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Evaluation results save directory (will automatically add model_name subdirectory)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (subdirectory name)")

    parser.add_argument("--r1_type", action="store_true", default=False,
                        help="Enable R1-type evaluation, extract answers from <answer> tags")

    parser.add_argument("--exp_label", type=str,
                        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help="Experiment label")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--test_number", type=int, default=-1,
                        help="Number of evaluation samples (-1 for all)")
    parser.add_argument("--open_task_threshold", type=float, default=0.0,
                        help="Open-ended question threshold (for frame-level correct_bool statistics)")
    parser.add_argument("--bleu_weights", type=str, default="0.25,0.25,0.25,0.25",
                        help="BLEU weights (4 comma-separated values)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    random.seed(args.random_seed)

    logger = setup_logger(__name__, args.log_level)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        bleu_weights = tuple(float(w) for w in args.bleu_weights.split(","))
        if len(bleu_weights) != 4:
            raise ValueError
    except Exception:
        bleu_weights = (0.25, 0.25, 0.25, 0.25)
        logger.warning(f"BLEU weight parsing failed, using default: {bleu_weights}")

    logger.info("Starting MapLMv2 evaluation")
    logger.info(f"dataset_path: {args.dataset_path}")
    logger.info(f"infer_file_path: {args.infer_file_path}")
    logger.info(f"output_dir: {args.output_dir}")
    logger.info(f"model_name: {args.model_name}")
    logger.info(f"exp_label: {args.exp_label}")
    logger.info(f"r1_type: {args.r1_type}")

    meta_list = load_jsonl_as_list(args.dataset_path, max_items=args.test_number)
    infer_list = load_jsonl_as_list(args.infer_file_path, max_items=args.test_number)

    n = min(len(meta_list), len(infer_list))
    meta_list = meta_list[:n]
    infer_list = infer_list[:n]
    logger.info(f"Total samples: {n}")

    out_paths = get_output_paths(args)
    ensure_dir(out_paths["out_dir"])

    config_info = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": args.dataset_path,
        "infer_file_path": args.infer_file_path,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
        "exp_label": args.exp_label,
        "r1_type": args.r1_type,
        "random_seed": args.random_seed,
        "test_number": n if args.test_number == -1 else min(args.test_number, n),
        "open_task_threshold": args.open_task_threshold,
        "bleu_weights": bleu_weights,
        "open_scalar_parse": True,   # NEW: Record that scalar parsing is enabled
    }

    results: Dict[str, Dict[str, float]] = {
        "question_overall": acc_counter(),
        "frame_overall": acc_counter(),
        "closed_choice_overall": acc_counter(),
        "open_ended_overall": acc_counter(),
    }

    results_by_type: Dict[str, Dict[str, float]] = {}
    frame_results: Dict[str, List[Dict[str, Any]]] = {}
    detailed_results: List[Dict[str, Any]] = []

    closed_choice_count = 0
    open_ended_count = 0

    for idx in range(n):
        try:
            meta_item = meta_list[idx]
            infer_item = infer_list[idx]

            md = meta_item.get("metadata", {}) if isinstance(meta_item.get("metadata"), dict) else {}
            sample_id = md.get("sample_id")
            frame_key = str(sample_id) if sample_id is not None else f"sample_{idx}"

            messages = meta_item.get("messages") or infer_item.get("messages") or []
            user_content = parse_user_content(messages)
            question_key = parse_question_key(meta_item, user_content)

            completion = infer_item.get("response", "")
            completion = "" if completion is None else str(completion).strip()
            original_completion = completion
            if args.r1_type:
                completion = preprocess_model_output(completion)
                completion = extract_answer_from_tag(completion)

            true_label = infer_item.get("labels")

            task, choices = decide_task_type(meta_item, user_content)

            if question_key not in results:
                results[question_key] = acc_counter()

            qa_type = md.get("qa_type", "unknown")
            if qa_type not in results_by_type:
                results_by_type[qa_type] = acc_counter()

            lane_id = extract_lane_id(md, question_key, user_content)

            record = {
                "index": idx,
                "frame_key": frame_key,
                "sample_id": sample_id,
                "qa_type": qa_type,
                "question_key": question_key,
                "user_content": user_content,
                "lane_id": lane_id,
                "task": task,
                "choices": choices,
                "original_completion": original_completion,
                "completion": completion,
                "label": true_label,
                "metadata": {k: v for k, v in md.items() if k != "messages"},
            }

            if task == "closed choice":
                closed_choice_count += 1
                pred_answer = completion_to_answer(completion, choices)
                true_answer = label_to_choice_index(true_label, choices)
                if pred_answer is None or true_answer is None:
                    correct = False
                    score = 0.0
                else:
                    correct = (pred_answer == true_answer)
                    score = 1.0 if correct else 0.0

                results["closed_choice_overall"]["total"] += 1
                results["closed_choice_overall"]["correct"] += int(correct)

                results["question_overall"]["total"] += 1
                results["question_overall"]["correct"] += int(correct)

                results_by_type[qa_type]["total"] += 1
                results_by_type[qa_type]["correct"] += int(correct)

                results[question_key]["total"] += 1
                results[question_key]["correct"] += int(correct)

                frame_results.setdefault(frame_key, []).append({
                    "correct": correct,
                    "score": score,
                    "task_type": task
                })

                record.update({
                    "pred_answer": pred_answer,
                    "true_answer": true_answer,
                    "correct": correct,
                    "score": score,
                    "metric": "accuracy",
                    "correct_bool": correct,
                })

            else:
                open_ended_count += 1
                true_answer_str = "" if true_label is None else str(true_label).strip()

                # NEW: Scalar open question parsing (does not change BLEU, only changes input strings)
                true_eval, pred_eval, used_scalar = maybe_extract_scalar_open_answer(
                    qa_type=qa_type,
                    question_key=question_key,
                    true_answer=true_answer_str,
                    completion=completion,
                    user_content=user_content,
                )
                record["true_answer"] = true_answer_str
                record["true_answer_for_eval"] = true_eval
                record["completion_for_eval"] = pred_eval
                record["used_scalar_parse"] = bool(used_scalar)

                if sentence_bleu is None:
                    bleu = 0.0
                else:
                    bleu = calculate_bleu_score(true_eval, pred_eval, bleu_weights, logger=logger)

                correct_for_frame = bleu > args.open_task_threshold

                results["open_ended_overall"]["total"] += 1
                results["open_ended_overall"]["correct"] += float(bleu)

                results["question_overall"]["total"] += 1
                results["question_overall"]["correct"] += float(bleu)

                results_by_type[qa_type]["total"] += 1
                results_by_type[qa_type]["correct"] += float(bleu)

                results[question_key]["total"] += 1
                results[question_key]["correct"] += float(bleu)

                frame_results.setdefault(frame_key, []).append({
                    "correct": correct_for_frame,
                    "score": float(bleu),
                    "task_type": task
                })

                record.update({
                    "bleu": float(bleu),
                    "score": float(bleu),
                    "metric": "bleu",
                    "correct_bool": correct_for_frame,
                })

            detailed_results.append(record)

            if args.debug and idx < 10:
                logger.debug(
                    f"idx={idx} frame={frame_key} qa_type={qa_type} task={task} "
                    f"score={record.get('score')} used_scalar={record.get('used_scalar_parse', False)}"
                )

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            if args.debug:
                import traceback
                logger.error(traceback.format_exc())
            continue

    logger.info(f"Evaluation complete: closed_choice={closed_choice_count}, open_ended={open_ended_count}")

    frame_acc_results = calculate_frame_accuracy(frame_results)
    results["frame_overall"]["total"] = frame_acc_results["total"]
    results["frame_overall"]["correct"] = frame_acc_results["correct"]

    for qa_type, counter in results_by_type.items():
        results[f"type_{qa_type}"] = counter

    acc_dict = compute_acc(results)

    # Paper metrics (DES/FRM/QNS)
    maplm_primary = calculate_maplmv2_primary_metrics(detailed_results)

    # Original output
    safe_json_dump(acc_dict, out_paths["results_acc"])
    safe_json_dump(results, out_paths["results_raw"])
    safe_json_dump(frame_results, out_paths["frame_results"])
    safe_json_dump(config_info, out_paths["config_info"])

    with open(out_paths["detailed_report"], "w", encoding="utf-8") as f:
        for r in detailed_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    unified_scores = build_unified_scores_json(
        args=args,
        config_info=config_info,
        results=results,
        acc_dict=acc_dict,
        detailed_results=detailed_results,
        maplm_primary=maplm_primary,
    )
    safe_json_dump(unified_scores, out_paths["scores_json"])

    with open(out_paths["log_txt"], "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MapLMv2 Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Evaluation configuration:\n")
        f.write("-" * 40 + "\n")
        f.write(json.dumps(config_info, indent=2, ensure_ascii=False))
        f.write("\n\n")

        f.write("Key Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"DES={maplm_primary.get('DES', 0.0):.6f} ({maplm_primary.get('DES', 0.0)*100:.4f}%)\n")
        f.write(f"FRM={maplm_primary.get('FRM', 0.0):.6f} ({maplm_primary.get('FRM', 0.0)*100:.4f}%)\n")
        f.write(f"QNS={maplm_primary.get('QNS', 0.0):.6f} ({maplm_primary.get('QNS', 0.0)*100:.4f}%)\n")

    logger.info("===== Key Metrics =====")
    logger.info(f"DES={maplm_primary.get('DES', 0.0):.6f}  FRM={maplm_primary.get('FRM', 0.0):.6f}  QNS={maplm_primary.get('QNS', 0.0):.6f}")
    logger.info(f"Saved to: {out_paths['out_dir']}")
    logger.info(f"Unified scores: {out_paths['scores_json']}")


if __name__ == "__main__":
    main()
