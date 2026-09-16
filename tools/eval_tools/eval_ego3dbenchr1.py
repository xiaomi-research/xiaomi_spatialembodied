# UPDATE: Replace placeholder paths with your actual paths.
import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Your project utils (already implemented)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("/path/to/Ego3D-Bench-main")
try:
    from utils.eval import *  # noqa: F401,F403
    from utils.common import *  # noqa: F401,F403
except ImportError as e:
    # `utils` here is the Ego3D-Bench helper package, pulled in via the `sys.path.append`
    # above; it is NOT part of this repository.
    raise ImportError(
        "eval_ego3dbenchr1.py requires the Ego3D-Bench `utils` package. It is not part of "
        "this repository.\n  Clone Ego3D-Bench and point the `sys.path.append('/path/to/"
        "Ego3D-Bench-main')` line above at it."
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

NUMERIC_CATEGORIES = {
    "Ego_Centric_Absolute_Distance",
    "Object_Centric_Absolute_Distance",
}

YESNO_CATEGORIES = {
    "Ego_Centric_Relative_Distance",
    "Ego_Centric_Motion_Reasoning",
    "Object_Centric_Motion_Reasoning",
}

def extract_answer_from_tag(text):
    """Extract answer content from <answer> tags"""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline evaluation for Ego3D-Bench using precomputed inference JSONL."
    )
    parser.add_argument("--infer_file_path", type=str, required=True,
                        help="Path to inference results JSONL (contains response/labels).")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to Ego3D-Bench meta JSONL (contains id/category/source/messages/images).")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name used as the output subfolder.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory to store evaluation outputs.")
    parser.add_argument("--max_error_cases", type=int, default=50,
                        help="Max number of error cases stored per category in detailed_report.")
    parser.add_argument("--keep_per_category_logs", action="store_true",
                        help="Keep per-category JSONL logs (default: keep).")
    parser.add_argument("--no_cleanup", action="store_true",
                        help="Do not cleanup temporary files (if any).")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug logging.")
    # Add R1-type parameter
    parser.add_argument("--r1_type", action="store_true",
                        help="Enable R1-type evaluation (extract answer from <answer> tags).")
    return parser.parse_args()

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse JSON at line {ln} in {path}: {e}")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))

def extract_user_text_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    return ""

def extract_assistant_text_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return str(m.get("content", ""))
    return ""

def get_meta_id(meta: Dict[str, Any], fallback_index: int) -> str:
    if "id" in meta and meta["id"] is not None:
        return str(meta["id"])
    return f"unknown_{fallback_index}"

def get_infer_id(infer: Dict[str, Any], fallback_index: int) -> Optional[str]:
    if "id" in infer and infer["id"] is not None:
        return str(infer["id"])
    return None

def normalize_gt(meta: Dict[str, Any], infer: Dict[str, Any]) -> str:
    """
    Decide GT from available fields, priority:
      1) infer["labels"] if present
      2) meta["answer"] if present
      3) assistant content in meta["messages"] if present
    """
    if infer.get("labels") not in (None, ""):
        return str(infer.get("labels", "")).strip()
    if meta.get("answer") not in (None, ""):
        return str(meta.get("answer", "")).strip()
    if meta.get("messages") is not None:
        return extract_assistant_text_from_messages(meta.get("messages"))
    return ""

def extract_answer_tag(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return None

def parse_numeric_from_text(text: str) -> Optional[float]:
    if not isinstance(text, str):
        return None

    # 1. Prefer matching "Answer: 10" or "answer is 10.5" etc.
    m = re.search(r"(?:answer|distance|is|=|:)\s*([+-]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # 2. Fallback: match number in the last sentence (optional)
    sentences = text.strip().split('\n')
    for sent in reversed(sentences):
        sent = sent.strip()
        if not sent:
            continue
        m = re.search(r"\b([+-]?\d+(?:\.\d+)?)\b", sent)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    return None

def parse_yesno_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    t = text.strip().lower()
    m = re.search(r"\b(yes|no)\b", t)
    if m:
        return m.group(1)
    return None

def parse_choice_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None

    text = text.strip()

    # 1. Match "Answer: A" or "answer is A" or "correct answer is A"
    m = re.search(r"(?:answer|choice|option|correct)\s*(?:is|=|:)\s*([A-D])(?:\.|\)|\s|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 2. Match "A. 33 meters" or "(A)" or "A)" or "[A]"
    m = re.search(r"(?:^|\s|\(|\[)([A-D])(?:\)|\.|\]|\s|$)", text)
    if m:
        return m.group(1).upper()

    # 3. Match letter in the last sentence (fallback)
    sentences = text.split('\n')
    for sent in reversed(sentences):
        sent = sent.strip()
        if not sent:
            continue
        # Match "A. xxx" or "A) xxx"
        m = re.search(r"\b([A-D])(?:\.|\))\s+\w+", sent)
        if m:
            return m.group(1).upper()

    return None

def process_pred_for_category(raw_pred: str, category: str, r1_type: bool = False) -> str:
    """
    Produce Processed_Pred (plain answer only, no tags),
    consistent with the original Ego3D-Bench script behavior.

    Args:
        raw_pred: Raw prediction text
        category: Question category
        r1_type: Whether to use R1-type evaluation (extract answer from <answer> tags)
    """
    if not isinstance(raw_pred, str):
        raw_pred = str(raw_pred)

    # Modified: if R1-type evaluation is enabled, force extract from <answer> tags
    if r1_type:
        raw_pred = preprocess_model_output(raw_pred)
        # Use dedicated extract_answer_from_tag function
        raw_pred = extract_answer_from_tag(raw_pred)
    else:
        # Original logic: if model already returned <answer> tags, use them
        tagged = extract_answer_tag(raw_pred)
        if tagged is not None and tagged != "":
            raw_pred = tagged

    raw_pred = raw_pred.replace("\n", " ").strip()

    if category in NUMERIC_CATEGORIES:
        val = parse_numeric_from_text(raw_pred)
        return "" if val is None else str(val)

    # Fix 1: Correctly handle yes/no categories
    if category in YESNO_CATEGORIES:
        yn = parse_yesno_from_text(raw_pred)
        if yn == "yes":
            yn = 'A'
        else:
            yn = 'B'
        if yn is not None:
            return yn  # Return "yes" or "no", not raw text
        # If unable to parse yes/no, try multi-choice parsing (but should not happen)
        ch = parse_choice_from_text(raw_pred)
        if ch is not None:
            return ch.lower()
        return raw_pred.lower()  # Final fallback

    # Multi-choice category: only keep option letter
    ch = parse_choice_from_text(raw_pred)
    if ch is not None:
        return ch
    return raw_pred.strip()

def build_pairs(
    infer_list: List[Dict[str, Any]],
    meta_list: List[Dict[str, Any]],
    debug: bool = False,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    meta_by_id: Dict[str, Dict[str, Any]] = {}
    for i, m in enumerate(meta_list):
        mid = get_meta_id(m, i)
        meta_by_id[mid] = m

    infer_has_id = any(get_infer_id(x, i) is not None for i, x in enumerate(infer_list))
    if infer_has_id:
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        missing = 0
        for i, inf in enumerate(infer_list):
            iid = get_infer_id(inf, i)
            if iid is None:
                missing += 1
                continue
            meta = meta_by_id.get(iid)
            if meta is None:
                missing += 1
                continue
            pairs.append((inf, meta))
        if debug:
            print(f"[DEBUG] Matched by id: {len(pairs)} pairs. Missing/unmatched: {missing}.")
        if pairs:
            return pairs

    n = min(len(infer_list), len(meta_list))
    if debug:
        print(f"[DEBUG] Falling back to index alignment for {n} pairs.")
    return [(infer_list[i], meta_list[i]) for i in range(n)]

def write_per_category_logs(
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    out_dir: str,
    r1_type: bool = False,  # Add r1_type parameter
    debug: bool = False,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    cat_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for idx, (inf, meta) in enumerate(pairs):
        category = str(meta.get("category", "unknown"))
        question = ""
        if meta.get("messages") is not None:
            question = extract_user_text_from_messages(meta.get("messages"))
        if not question and inf.get("messages") is not None:
            question = extract_user_text_from_messages(inf.get("messages"))

        raw_pred = inf.get("response", "")
        gt = normalize_gt(meta, inf)

        # Modified: pass r1_type parameter to process_pred_for_category
        processed_pred = process_pred_for_category(str(raw_pred), category, r1_type)

        question_type = "exact_number" if category in NUMERIC_CATEGORIES else "multi_choice"

        row = {
            "Question": question,
            "Question_Type": question_type,
            "Pred": raw_pred,
            "Processed_Pred": processed_pred,
            "GT": gt,
        }

        if debug:
            row["_meta_id"] = meta.get("id", f"idx_{idx}")
            row["_source"] = meta.get("source", "unknown")
            row["_r1_type"] = r1_type  # Add r1_type marker

        cat_to_rows[category].append(row)

    cat_files: Dict[str, str] = {}
    for cat, rows in cat_to_rows.items():
        fp = os.path.join(out_dir, f"{cat}.jsonl")
        with open(fp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        cat_files[cat] = fp

    return cat_files

def safe_float(x: Any) -> Optional[float]:
    try:
        return float(str(x).strip())
    except Exception:
        return None

def compute_metrics_for_file(category: str, file_path: str) -> Dict[str, Any]:
    total = 0
    invalid_pred = 0

    if category in NUMERIC_CATEGORIES:
        y_true: List[float] = []
        y_pred: List[float] = []
        abs_errors: List[Tuple[float, int, Dict[str, Any]]] = []

        rows = load_jsonl(file_path)
        for i, r in enumerate(rows):
            total += 1
            pred = safe_float(r.get("Processed_Pred", ""))
            gt = safe_float(r.get("GT", ""))
            if pred is None or gt is None:
                invalid_pred += 1
                continue
            pred = min(pred, 100.0)
            y_true.append(gt)
            y_pred.append(pred)
            abs_errors.append((abs(pred - gt), i, r))

        if not y_true:
            return {
                "type": "exact_number",
                "total": total,
                "valid": 0,
                "invalid": invalid_pred,
                "rmse": None,
                "mae": None,
            }

        mae = sum(abs(p - g) for p, g in zip(y_pred, y_true)) / len(y_true)
        rmse = (sum((p - g) ** 2 for p, g in zip(y_pred, y_true)) / len(y_true)) ** 0.5

        abs_errors.sort(key=lambda x: x[0], reverse=True)
        worst_cases = []
        for err, _, r in abs_errors[:50]:
            worst_cases.append({
                "abs_error": err,
                "gt": r.get("GT", ""),
                "processed_pred": r.get("Processed_Pred", ""),
                "pred": r.get("Pred", ""),
                "question": r.get("Question", ""),
            })

        return {
            "type": "exact_number",
            "total": total,
            "valid": len(y_true),
            "invalid": invalid_pred,
            "rmse": rmse,
            "mae": mae,
            "worst_cases": worst_cases,
        }

    correct = 0
    wrong_cases: List[Dict[str, Any]] = []

    rows = load_jsonl(file_path)
    for i, r in enumerate(rows):
        total += 1
        pred = str(r.get("Processed_Pred", "")).strip().lower()
        gt = str(r.get("GT", "")).strip().lower()

        pred_norm = pred
        gt_norm = gt

        if category in YESNO_CATEGORIES:
            yn = parse_yesno_from_text(pred_norm)
            if yn is not None:
                pred_norm = yn
            yn_gt = parse_yesno_from_text(gt_norm)
            if yn_gt is not None:
                gt_norm = yn_gt
        else:
            ch = parse_choice_from_text(pred_norm)
            if ch is not None:
                pred_norm = ch.lower()
            ch_gt = parse_choice_from_text(gt_norm)
            if ch_gt is not None:
                gt_norm = ch_gt.lower()

        if pred_norm == gt_norm and gt_norm != "":
            correct += 1
        else:
            wrong_cases.append({
                "gt": r.get("GT", ""),
                "processed_pred": r.get("Processed_Pred", ""),
                "pred": r.get("Pred", ""),
                "question": r.get("Question", ""),
            })

    acc = correct / total if total > 0 else 0.0
    return {
        "type": "categorical",
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "wrong_cases": wrong_cases[:50],
    }

def evaluate_all_categories(cat_files: Dict[str, str], debug: bool = False) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for cat, fp in sorted(cat_files.items()):
        metrics = compute_metrics_for_file(cat, fp)
        results[cat] = metrics
        try:
            multi_choice = (cat not in NUMERIC_CATEGORIES)
            if debug:
                print(f"[DEBUG] Calling eval_logs for {cat} (multi_choice={multi_choice}) on {fp}")
            eval_logs(fp, multi_choice=multi_choice)
        except Exception as e:
            print(f"[WARN] eval_logs failed for {cat}: {e}")
    return results

def build_new_format_report(
    model_name: str,
    infer_file_path: str,
    dataset_path: str,
    cat_results: Dict[str, Any],
    max_error_cases: int,
    r1_type: bool = False,  # Add r1_type parameter
) -> Dict[str, Any]:
    # Fix 2: Correctly calculate total samples for categorical tasks
    total_categorical = 0
    for cat, r in cat_results.items():
        if r.get("type") == "categorical":
            total_categorical += int(r.get("total", 0))

    overall_correct = 0
    for cat, r in cat_results.items():
        if r.get("type") == "categorical":
            overall_correct += int(r.get("correct", 0))

    overall_accuracy = overall_correct / total_categorical if total_categorical > 0 else 0.0

    numeric_rmse_values = []
    numeric_mae_values = []

    for cat, r in cat_results.items():
        if r.get("type") == "exact_number":
            rmse = r.get("rmse")
            mae = r.get("mae")
            if rmse is not None:
                numeric_rmse_values.append(rmse)
            if mae is not None:
                numeric_mae_values.append(mae)

    avg_rmse = sum(numeric_rmse_values) / len(numeric_rmse_values) if numeric_rmse_values else None
    avg_mae = sum(numeric_mae_values) / len(numeric_mae_values) if numeric_mae_values else None

    # Change 1: Change secondary_metrics to a structure with metric names as keys and values as values
    secondary_metrics = {}
    for cat, r in cat_results.items():
        if r.get("type") == "categorical":
            # For categorical tasks, use accuracy as metric
            secondary_metrics[f"{cat}_accuracy"] = r.get("accuracy")
        elif r.get("type") == "exact_number":
            # For numeric tasks, record RMSE and MAE separately
            rmse_val = r.get("rmse")
            mae_val = r.get("mae")
            if rmse_val is not None:
                secondary_metrics[f"{cat}_rmse"] = rmse_val
            if mae_val is not None:
                secondary_metrics[f"{cat}_mae"] = mae_val

    # Keep detailed results structure unchanged
    detailed_results = {}
    for cat, r in cat_results.items():
        detailed = dict(r)
        if r.get("type") == "categorical" and "wrong_cases" in detailed:
            detailed["wrong_cases"] = detailed["wrong_cases"][:max_error_cases]
        elif r.get("type") == "exact_number" and "worst_cases" in detailed:
            detailed["worst_cases"] = detailed["worst_cases"][:max_error_cases]
        detailed_results[cat] = detailed

    return {
        "overall": {
            "score": overall_accuracy,
            "metrics":
                {
                    'accuracy': overall_accuracy,
                    "avg_numeric_rmse": avg_rmse,
                    "avg_numeric_mae": avg_mae,
                },
            "categorical_correct": overall_correct,
            "total_samples": total_categorical,  # Fix 3: Use categorical task sample count
            "note": "overall accuracy is for categorical tasks only; numeric tasks have separate RMSE/MAE metrics"
        },
        "secondary_metrics": secondary_metrics,  # Change 2: Use flattened secondary metrics
        "detailed_results": detailed_results
    }

def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("Ego3D-Bench Offline Evaluation")
    if args.r1_type:
        print("Mode: R1-type (extract answer from <answer> tags)")
    print("=" * 72)
    print(f"Model name: {args.model_name}")
    print(f"Infer results: {args.infer_file_path}")
    print(f"Dataset meta:  {args.dataset_path}")
    print(f"Output dir:    {args.output_dir}")
    print(f"R1-type mode:  {args.r1_type}")
    print("-" * 72)

    out_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(out_dir, exist_ok=True)

    per_category_dir = os.path.join(out_dir, "per_category_logs")
    os.makedirs(per_category_dir, exist_ok=True)

    infer_list = load_jsonl(args.infer_file_path)
    meta_list = load_jsonl(args.dataset_path)

    print(f"Loaded infer samples: {len(infer_list)}")
    print(f"Loaded meta samples:  {len(meta_list)}")

    if len(infer_list) != len(meta_list):
        print(f"[WARN] Length mismatch: infer={len(infer_list)} vs meta={len(meta_list)}. Matching may be partial.")

    pairs = build_pairs(infer_list, meta_list, debug=args.debug)
    print(f"Matched pairs: {len(pairs)}")

    # Modified: pass r1_type parameter to write_per_category_logs
    cat_files = write_per_category_logs(pairs, per_category_dir, r1_type=args.r1_type, debug=args.debug)
    print(f"Generated per-category logs: {len(cat_files)} categories.")
    if args.debug:
        for c, fp in cat_files.items():
            print(f"  - {c}: {fp}")

    print("-" * 72)
    print("Evaluating...")
    cat_results = evaluate_all_categories(cat_files, debug=args.debug)

    print("-" * 72)
    print("Generating reports...")

    new_report = build_new_format_report(
        model_name=args.model_name,
        infer_file_path=args.infer_file_path,
        dataset_path=args.dataset_path,
        cat_results=cat_results,
        max_error_cases=args.max_error_cases,
        r1_type=args.r1_type,  # Add r1_type parameter
    )

    infer_basename = os.path.basename(args.infer_file_path)
    if infer_basename.endswith('.jsonl'):
        base_name = infer_basename[:-6]
    else:
        base_name = infer_basename

    # Add R1-type marker in output filename
    if args.r1_type:
        output_filename = f"{base_name}_r1type_scores.json"
    else:
        output_filename = f"{base_name}_scores.json"
    output_path = os.path.join(out_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_report, f, ensure_ascii=False, indent=2)

    print(f"Saved new format report: {output_path}")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Fix 4: Old format report also uses categorical task sample count
    total_categorical = 0
    overall_correct = 0
    for cat, r in cat_results.items():
        if r.get("type") == "categorical":
            total_categorical += int(r.get("total", 0))
            overall_correct += int(r.get("correct", 0))

    overall_accuracy = overall_correct / total_categorical if total_categorical > 0 else 0.0

    trimmed_results: Dict[str, Any] = {}
    for cat, r in cat_results.items():
        rr = dict(r)
        if rr.get("type") == "categorical" and "wrong_cases" in rr:
            rr["wrong_cases"] = rr["wrong_cases"][:args.max_error_cases]
        if rr.get("type") == "exact_number" and "worst_cases" in rr:
            rr["worst_cases"] = rr["worst_cases"][:args.max_error_cases]
        trimmed_results[cat] = rr

    detailed_report = {
        "benchmark": "Ego3D-Bench",
        "model_name": args.model_name,
        "evaluation_time": now,
        "evaluation_mode": "r1-type" if args.r1_type else "standard",  # Add mode marker
        "inputs": {
            "infer_results": args.infer_file_path,
            "dataset_path": args.dataset_path,
        },
        "outputs": {
            "output_dir": out_dir,
            "per_category_logs_dir": per_category_dir,
        },
        "overall": {
            "total_samples": total_categorical,  # Fix 5: Use categorical task sample count
            "categorical_correct": overall_correct,
            "categorical_accuracy": overall_accuracy,
            "note": "overall accuracy counts categorical tasks only; numeric tasks are reported with MAE/RMSE per category.",
        },
        "per_category": trimmed_results,
    }

    detailed_path = os.path.join(out_dir, "detailed_report.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_report, f, ensure_ascii=False, indent=2)

    summary_path = os.path.join(out_dir, "summary.txt")
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("Ego3D-Bench Offline Evaluation Summary")
    if args.r1_type:
        lines.append(f"Evaluation mode: R1-type (extract answer from <answer> tags)")
    lines.append("=" * 72)
    lines.append(f"Model: {args.model_name}")
    lines.append(f"Time: {now}")
    lines.append(f"Infer JSONL: {args.infer_file_path}")
    lines.append(f"Meta JSONL:  {args.dataset_path}")
    lines.append(f"Output dir:  {out_dir}")
    lines.append(f"R1-type mode: {args.r1_type}")
    lines.append("-" * 72)
    lines.append(f"Overall (categorical only): {overall_accuracy:.4f}  ({overall_correct}/{total_categorical})")
    lines.append(f"New format report: {output_filename}")
    lines.append("-" * 72)

    for cat, r in sorted(trimmed_results.items()):
        if r.get("type") == "categorical":
            lines.append(f"[{cat}] accuracy={r.get('accuracy', 0.0):.4f}  ({r.get('correct', 0)}/{r.get('total', 0)})")
        else:
            lines.append(
                f"[{cat}] rmse={r.get('rmse', None)}  mae={r.get('mae', None)}  valid={r.get('valid', 0)}/{r.get('total', 0)}"
            )

    lines.append("-" * 72)
    lines.append(f"Detailed report: {os.path.basename(detailed_path)}")
    lines.append(f"New format report: {output_filename}")
    lines.append("=" * 72)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved summary:  {summary_path}")
    print(f"Saved detailed: {detailed_path}")

    if not args.keep_per_category_logs:
        print("[INFO] Removing per-category logs directory.")
        shutil.rmtree(per_category_dir, ignore_errors=True)

    print("=" * 72)
    print("Done.")
    print("=" * 72)

if __name__ == "__main__":
    main()
