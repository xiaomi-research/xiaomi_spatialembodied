import argparse
import json, os
import math
import re
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from tabulate import tabulate
from tqdm import tqdm

import sys
sys.path.append(
    "/path/to/STRIDE-QA-Bench/src"
)

from strideqa_bench import STRIDEQA_BENCH_ROOT
from strideqa_bench.utils.evaluate_utils import save_as_jsonl
from strideqa_bench.utils.metric_utils import judge_localization_success, judge_success
from strideqa_bench.utils.parser import parse_pred_value
from strideqa_bench.visualize.visualize import (
    create_lsr_analysis_plots,
    create_prediction_distribution_plots,
)

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

# -----------------------------
# New: Extract answer from <answer> tags
# -----------------------------
def extract_answer_from_tag(text: str) -> str:
    """Extract answer content from <answer> tags; if no tag found, return original text"""
    if not isinstance(text, str):
        text = str(text)

    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# -----------------------------
# Original mapping (keep)
# -----------------------------
TIME_BUCKET: dict[str, int] = {
    "ego_distance_data": 0,
    "ego_speed_data": 0,
    "target_speed_data": 0,
    "target_bearing_angle_data": 0,
    "ego_distance_data_4d_t1": 1,
    "ego_speed_data_4d_t1": 1,
    "target_speed_data_4d_t1": 1,
    "ego_distance_data_4d_t2": 2,
    "ego_speed_data_4d_t2": 2,
    "target_speed_data_4d_t2": 2,
    "ego_distance_data_4d_t3": 3,
    "ego_speed_data_4d_t3": 3,
    "target_speed_data_4d_t3": 3,
}

EGO_TARGET_DISTINCTION: dict[str, str] = {
    "ego_distance_data": "target",
    "ego_speed_data": "ego",
    "target_speed_data": "target",
    "target_bearing_angle_data": "target",
    "ego_distance_data_4d_t1": "target",
    "ego_speed_data_4d_t1": "ego",
    "target_speed_data_4d_t1": "target",
    "ego_distance_data_4d_t2": "target",
    "ego_speed_data_4d_t2": "ego",
    "target_speed_data_4d_t2": "target",
    "ego_distance_data_4d_t3": "target",
    "ego_speed_data_4d_t3": "ego",
    "target_speed_data_4d_t3": "target",
}

# -----------------------------
# Helpers
# -----------------------------
QID_RE = re.compile(r"(q\d+_g\d+)")

def extract_group_id(question_id: str) -> str:
    if "_g" in question_id:
        return "g" + question_id.split("_g")[1]
    return question_id

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def safe_first_number(text: str) -> Optional[float]:
    m = re.search(r"[-+]?\d+(\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def parse_value_fallback_only(qa_type: str, text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    IMPORTANT: This fallback NEVER calls parse_pred_value().
    It only extracts the first number and assigns default unit.
    """
    if text is None:
        return None, None
    if not isinstance(text, str):
        text = str(text)

    v = safe_first_number(text)
    if v is None:
        return None, None

    if qa_type == "distance":
        return v, "m"
    if qa_type == "velocity":
        return v, "m/s"
    if qa_type == "direction":
        return v, "deg"
    return v, None

def parse_value_with_fallback(qa_type: str, text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    For GT parsing, still prefer official parser first, then fallback.
    (GT text is usually short and stable.)
    """
    parsed = parse_pred_value(qa_type, text)
    if parsed is not None:
        v, u = parsed
        return v, u
    return parse_value_fallback_only(qa_type, text)

# -----------------------------
# timeout-safe pred parsing (fix hanging regex)
# -----------------------------
class _ParseTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _ParseTimeout()

def safe_parse_pred_value(
    qa_type: str,
    pred_text: str,
    *,
    timeout_s: float = 0.05,
    max_chars: int = 800,
    hard_skip_official_if_longer_than: int = 800,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Prefer official parse_pred_value, but avoid hanging/very slow regex on long texts.

    Key fixes vs your current version:
    - On timeout/exception, fallback DOES NOT call parse_pred_value again.
    - Optionally skip official parser when text is long (common for "reasoning" outputs),
      because that is exactly where parse_velocity can become extremely slow.
    """
    if pred_text is None:
        pred_text = ""
    if not isinstance(pred_text, str):
        pred_text = str(pred_text)

    # Truncate aggressively (default 800 chars is enough for numeric answers)
    if max_chars is not None and len(pred_text) > max_chars:
        pred_text = pred_text[:max_chars]

    # If still long after truncation, skip official parser altogether
    if hard_skip_official_if_longer_than is not None and len(pred_text) > hard_skip_official_if_longer_than:
        return parse_value_fallback_only(qa_type, pred_text)

    # If timeout disabled -> try official once, else fallback_only
    if timeout_s is None or timeout_s <= 0:
        try:
            parsed = parse_pred_value(qa_type, pred_text)
            if parsed is not None:
                return parsed
        except Exception:
            pass
        return parse_value_fallback_only(qa_type, pred_text)

    old_handler = None
    try:
        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)

        parsed = parse_pred_value(qa_type, pred_text)

        signal.setitimer(signal.ITIMER_REAL, 0)
        if parsed is not None:
            return parsed
        return (None, None)

    except _ParseTimeout:
        # CRITICAL: fallback_only, never call parse_pred_value again
        return parse_value_fallback_only(qa_type, pred_text)

    except Exception:
        return parse_value_fallback_only(qa_type, pred_text)

    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except Exception:
            pass
        try:
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass

def infer_question_id_from_infer_item(item: Dict[str, Any]) -> Optional[str]:
    images = item.get("images", [])
    for im in images:
        path = im.get("path") if isinstance(im, dict) else None
        if not path:
            continue
        m = QID_RE.search(path)
        if m:
            return m.group(1)

    messages = item.get("messages", [])
    for msg in messages:
        c = msg.get("content", "")
        m = QID_RE.search(c)
        if m:
            return m.group(1)
    return None

def qa_category_to_qa_type(qa_category: str) -> str:
    if "distance" in qa_category:
        return "distance"
    if "speed" in qa_category:
        return "velocity"
    if "bearing_angle" in qa_category:
        return "direction"
    raise ValueError(f"Unknown qa_category -> qa_type mapping: {qa_category}")

def build_meta_index_from_dataset_path(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    dataset_path supports:
    - Directory: contains strideqa_bench_t*_bench_image.jsonl files directly
    - File: a single meta jsonl (each line has metadata.id)
    """
    index: Dict[str, Dict[str, Any]] = {}

    if dataset_path.is_file():
        # If it's a file, load it directly
        for item in load_jsonl(dataset_path):
            md = item.get("metadata", {})
            qid = md.get("id")
            if qid:
                index[qid] = item
        return index

    # If it's a directory, find all matching jsonl files
    meta_files = list(dataset_path.glob("strideqa_bench_t*_bench_image.jsonl"))
    meta_files.extend(dataset_path.glob("strideqa_bench_t*_bench_video.jsonl"))

    if not meta_files:
        raise FileNotFoundError(
            f"Cannot find meta jsonl files in {dataset_path}. "
            f"Expected strideqa_bench_t*_bench_image.jsonl or strideqa_bench_t*_bench_video.jsonl"
        )

    print(f"Found {len(meta_files)} meta files:")
    for p in meta_files:
        print(f"  - {p.name}")

    for p in meta_files:
        for item in load_jsonl(p):
            md = item.get("metadata", {})
            qid = md.get("id")
            if qid:
                index[qid] = item
    return index

def load_annotation_data(annotation_csv_path: Path | str) -> dict[str, dict[str, str]]:
    df = pd.read_csv(annotation_csv_path)
    annotation_dict = {}
    for _, row in df.iterrows():
        group_id = row["group_id"]
        object_type = row["object_type"] if pd.notna(row["object_type"]) else None
        relation = row["relation"] if pd.notna(row["relation"]) else None
        annotation_dict[group_id] = {"object_type": object_type, "relation": relation}
    return annotation_dict

# -----------------------------
# Original evaluation core (unchanged except pred parsing call)
# -----------------------------
def run_evaluation(
    model_response_list: list[dict[str, Any]],
    config: dict[str, Any],
    annotation_dict: dict[str, dict[str, str]],
    *,
    parse_timeout_s: float,
    parse_max_chars: int,
    skip_official_if_longer_than: int,
) -> list[dict[str, Any]]:
    raw_results_list: list[dict[str, Any]] = []

    for model_response in tqdm(model_response_list):
        gt_value_dict = model_response.get("gt_value", {})

        time_subscript = TIME_BUCKET[model_response["qa_category"]]
        ego_or_target = EGO_TARGET_DISTINCTION[model_response["qa_category"]]

        question_id = model_response.get("question_id", "")
        group_id = extract_group_id(question_id)
        annotation_info = annotation_dict.get(group_id, {})
        object_type = annotation_info.get("object_type")
        relation = annotation_info.get("relation")

        for qa_type, gt_value in gt_value_dict.items():
            gt_unit = model_response["unit"][qa_type]
            tolerance = config["tolerance"][qa_type]

            pred: str = model_response.get("pred", "")

            pred_value, pred_unit = safe_parse_pred_value(
                qa_type,
                pred,
                timeout_s=parse_timeout_s,
                max_chars=parse_max_chars,
                hard_skip_official_if_longer_than=skip_official_if_longer_than,
            )

            details = judge_success(qa_type, gt_value, pred_value, tolerance)
            success = details["success"]
            details.update(tolerance)

            raw_results = {
                "question_id": model_response.get("question_id"),
                "group_id": group_id,
                "object_type": object_type,
                "relation": relation,
                "qa_category": model_response["qa_category"],
                "time": time_subscript,
                "ego_or_target": ego_or_target,
                "qa_type": qa_type,
                "gt": model_response.get("gt"),
                "pred": pred,
                "gt_value": gt_value,
                "pred_value": pred_value,
                "gt_unit": gt_unit,
                "pred_unit": pred_unit,
                "success": success,
                "details": details,
            }
            raw_results_list.append(raw_results)

    lsr_results_list = _calculate_localization_success(raw_results_list, config)
    raw_results_list.extend(lsr_results_list)
    return raw_results_list

def judge_localization_success_flexible(
    *,
    distance_gt,
    direction_gt,
    distance_pred,
    direction_pred,
    tolerance_config,
):
    """
    If one of distance/direction GT is missing, fall back to single-metric judge_success;
    if both are present, use official judge_localization_success for consistency.
    """
    # both missing -> should not happen if you guard earlier
    if distance_gt is None and direction_gt is None:
        return {"success": False, "mode": "none"}

    # both available -> use official (align with benchmark)
    if distance_gt is not None and direction_gt is not None:
        out = judge_localization_success(
            distance_gt=distance_gt,
            direction_gt=direction_gt,
            distance_pred=distance_pred,
            direction_pred=direction_pred,
            tolerance_config=tolerance_config,
        )
        out["mode"] = "distance+direction"
        return out

    # distance-only
    if distance_gt is not None and direction_gt is None:
        d = judge_success("distance", distance_gt, distance_pred, tolerance_config["distance"])
        return {
            "success": bool(d["success"]),
            "mode": "distance_only",
            "distance_success": bool(d["success"]),
            "direction_success": None,
            "distance_details": d,
        }

    # direction-only (theoretically this should not happen in your data)
    g = judge_success("direction", direction_gt, direction_pred, tolerance_config["direction"])
    return {
        "success": bool(g["success"]),
        "mode": "direction_only",
        "distance_success": None,
        "direction_success": bool(g["success"]),
        "direction_details": g,
    }


def _calculate_localization_success(
    raw_results_list: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    lsr_results: list[dict[str, Any]] = []
    grouped_data: dict[str, dict] = {}

    for result in raw_results_list:
        question_id = result["question_id"]
        image_group = extract_group_id(question_id)
        time = result["time"]
        ego_or_target = result["ego_or_target"]
        qa_type = result["qa_type"]
        group_key = f"{image_group}_{time}_{ego_or_target}"

        if group_key not in grouped_data:
            grouped_data[group_key] = {
                "image_group": image_group,
                "time": time,
                "ego_or_target": ego_or_target,
                "distance": None,
                "direction": None,
            }

        if qa_type == "distance":
            grouped_data[group_key]["distance"] = result
        elif qa_type == "direction":
            grouped_data[group_key]["direction"] = result

    for _, group_data in grouped_data.items():
        distance_result = group_data["distance"]
        direction_result = group_data["direction"]

        if distance_result is None and direction_result is None:
            continue

        primary_result = distance_result if distance_result is not None else direction_result

        distance_gt = distance_result["gt_value"] if distance_result is not None else None
        direction_gt = direction_result["gt_value"] if direction_result is not None else None
        distance_pred = distance_result["pred_value"] if distance_result is not None else None
        direction_pred = direction_result["pred_value"] if direction_result is not None else None

        # lsr_details = judge_localization_success(
        #     distance_gt=distance_gt,
        #     direction_gt=direction_gt,
        #     distance_pred=distance_pred,
        #     direction_pred=direction_pred,
        #     tolerance_config=config["tolerance"],
        # )
        lsr_details = judge_localization_success_flexible(
            distance_gt=distance_gt,
            direction_gt=direction_gt,
            distance_pred=distance_pred,
            direction_pred=direction_pred,
            tolerance_config=config["tolerance"],
        )


        lsr_result = {
            "question_id": primary_result["question_id"],
            "group_id": primary_result.get("group_id", ""),
            "object_type": primary_result.get("object_type"),
            "relation": primary_result.get("relation"),
            "qa_category": primary_result["qa_category"],
            "time": group_data["time"],
            "ego_or_target": group_data["ego_or_target"],
            "qa_type": "lsr",
            "gt_value": {"distance": distance_gt, "direction": direction_gt},
            "pred_value": {"distance": distance_pred, "direction": direction_pred},
            "gt_unit": {
                "distance": distance_result["gt_unit"] if distance_result is not None else None,
                "direction": direction_result["gt_unit"] if direction_result is not None else None,
            },
            "pred_unit": {
                "distance": distance_result["pred_unit"] if distance_result is not None else None,
                "direction": direction_result["pred_unit"] if direction_result is not None else None,
            },
            "success": lsr_details["success"],
            "details": lsr_details,
        }
        lsr_results.append(lsr_result)

    return lsr_results

def calculate_success_rate(raw_results_list: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    stats: dict[str, dict[int, list[int]]] = {}

    for raw_results in raw_results_list:
        qa_type = raw_results["qa_type"]
        subject = raw_results["ego_or_target"]
        time = raw_results["time"]
        success = raw_results["success"]

        if qa_type.lower() == "lsr":
            metric_name = "lsr"
        else:
            metric_name = f"{subject.lower()}_{qa_type.lower()}"

        stats.setdefault(metric_name, {})
        stats[metric_name].setdefault(time, [0, 0])
        stats[metric_name][time][0] += 1
        if success:
            stats[metric_name][time][1] += 1

    results: dict[str, list[dict[str, Any]]] = {}
    for qa_type, time_stats in stats.items():
        results[qa_type] = []
        total_counts = 0
        total_successes = 0

        for time, (counts, successes) in sorted(time_stats.items()):
            rate = successes / counts if counts > 0 else 0
            results[qa_type].append({"time": time, "success_rate": round(rate, 8), "sample_size": counts})
            total_counts += counts
            total_successes += successes

        if total_counts > 0:
            avg_rate = total_successes / total_counts
            results[qa_type].append({"time": "avg", "success_rate": round(avg_rate, 8), "sample_size": total_counts})

    return results

def compute_tlc_mlsr(df: pd.DataFrame, *, max_step: int = 3) -> pd.DataFrame:
    lsr = df[df.qa_type == "lsr"].copy()

    pivot = (
        lsr.pivot_table(index=["group_id", "relation"], columns="time", values="success", aggfunc="first")
        .fillna(False)
        .astype(bool)
    )

    frames = []
    for k in range(1, max_step + 1):
        seq_ok = pivot[0] & pivot.loc[:, list(range(1, k + 1))].all(axis=1)
        rel_df = (
            seq_ok.reset_index(name="ok")
            .groupby("relation")["ok"]
            .mean()
            .reset_index(name=f"TLC@{k}")
        )
        overall = pd.DataFrame([{"relation": "Overall", f"TLC@{k}": seq_ok.mean()}])
        frames.append(pd.concat([overall, rel_df], ignore_index=True))

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="relation", how="outer")

    strict_ok = pivot.loc[:, list(range(0, max_step + 1))].all(axis=1)
    strict_df = strict_ok.reset_index(name="ok").groupby("relation")["ok"].mean().reset_index(name="TLC")
    strict_overall = pd.DataFrame([{"relation": "Overall", "TLC": strict_ok.mean()}])
    strict_df = pd.concat([strict_overall, strict_df], ignore_index=True)
    merged = merged.merge(strict_df, on="relation", how="outer")

    mf_score = pivot.mean(axis=1)
    mf_df = mf_score.reset_index(name="score").groupby("relation")["score"].mean().reset_index(name="MLSR")
    mf_overall = pd.DataFrame([{"relation": "Overall", "MLSR": mf_score.mean()}])
    mf_df = pd.concat([mf_overall, mf_df], ignore_index=True)
    merged = merged.merge(mf_df, on="relation", how="outer")

    merged = pd.concat(
        [
            merged.loc[merged.relation == "Overall"],
            merged.loc[merged.relation != "Overall"].sort_values("TLC", ascending=False),
        ]
    ).reset_index(drop=True)

    return merged

def aggregate_results(raw_results_list: list[dict[str, Any]]) -> dict[str, Any]:
    jst = timezone(timedelta(hours=9))
    evaluated_at = datetime.now(jst).isoformat()
    result_dict: dict[str, Any] = {"evaluated_at": evaluated_at}

    success_rates = calculate_success_rate(raw_results_list)
    result_dict.update(success_rates)

    df_all = pd.DataFrame(raw_results_list)
    df_lsr = df_all[df_all.qa_type == "lsr"]
    df_tlc_mlsr = compute_tlc_mlsr(df_lsr)
    result_dict["temporal_consistency"] = df_tlc_mlsr.to_dict(orient="records")

    return result_dict

def round_half_up(value: float, ndigits: int = 0) -> float:
    multiplier = 10**ndigits
    return math.floor(value * multiplier + 0.5) / multiplier

def generate_metric_tables(data: dict[str, Any], filepath: str | Path) -> None:
    ordered_keys = ["target_distance", "target_direction", "ego_velocity", "target_velocity", "lsr", "temporal_consistency"]
    all_keys = set(data.keys())
    remaining_keys = sorted(all_keys - set(ordered_keys))
    reordered = {k: data.get(k) for k in ordered_keys if k in data}
    reordered |= {k: data[k] for k in remaining_keys}

    def fmt_val(v: Any) -> Any:
        if isinstance(v, float):
            return f"{round_half_up(v * 100, 1)}"
        return v

    with open(filepath, "w", encoding="utf-8") as f:
        for metric, records in reordered.items():
            if not isinstance(records, list) or not records:
                continue
            headers = list(records[0].keys())
            rows = [[fmt_val(rec.get(h)) for h in headers] for rec in records]
            f.write(f"### {metric.replace('_', ' ').title()}\n\n")
            f.write(tabulate(rows, headers, tablefmt="github"))
            f.write("\n\n")

# -----------------------------
# Data interface
# -----------------------------
def build_model_response_list_from_infer(
    infer_jsonl_files: List[Path],
    meta_index: Dict[str, Dict[str, Any]],
    r1_type: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build model response list, supporting R1 format parsing
    Args:
        infer_jsonl_files: List of model output jsonl files
        meta_index: Metadata index
        r1_type: Whether it's R1 format; if True, extract answer from <answer> tags
    """
    model_response_list: List[Dict[str, Any]] = []
    missing_meta = 0
    missing_qid = 0
    bad_category = 0
    bad_gt_parse = 0

    for f in infer_jsonl_files:
        for item in tqdm(load_jsonl(f), desc=f"Loading infer {f}"):
            qid = infer_question_id_from_infer_item(item)
            if not qid:
                missing_qid += 1
                continue

            meta = meta_index.get(qid)
            if meta is None:
                missing_meta += 1
                continue

            qa_category = meta.get("metadata", {}).get("qa_category") or meta.get("metadata", {}).get("task_type")
            if not qa_category:
                bad_category += 1
                continue

            try:
                qa_type = qa_category_to_qa_type(qa_category)
            except Exception:
                bad_category += 1
                continue

            gt_text = item.get("labels")
            if not gt_text:
                msgs = meta.get("messages", [])
                gt_text = msgs[-1]["content"] if msgs else None
            if not gt_text:
                bad_gt_parse += 1
                continue

            gt_value, gt_unit = parse_value_with_fallback(qa_type, gt_text)
            if gt_value is None:
                bad_gt_parse += 1
                continue

            if gt_unit is None:
                gt_unit = {"distance": "m", "velocity": "m/s", "direction": "deg"}.get(qa_type)

            # Modification 1: If R1 format is enabled, extract answer from <answer> tags
            pred_text = item.get("response", "")
            if r1_type:
                pred_text = preprocess_model_output(pred_text)
                pred_text = extract_answer_from_tag(pred_text)
                # Record original response for debugging
                model_response_list.append(
                    {
                        "question_id": qid,
                        "qa_category": qa_category,
                        "pred": pred_text,
                        "pred_raw": item.get("response", ""),  # Save original response
                        "gt": gt_text,
                        "gt_value": {qa_type: gt_value},
                        "unit": {qa_type: gt_unit},
                        "r1_type": True,  # Mark as R1 format
                    }
                )
            else:
                model_response_list.append(
                    {
                        "question_id": qid,
                        "qa_category": qa_category,
                        "pred": pred_text,
                        "gt": gt_text,
                        "gt_value": {qa_type: gt_value},
                        "unit": {qa_type: gt_unit},
                        "r1_type": False,  # Mark as non-R1 format
                    }
                )

    if missing_qid or missing_meta or bad_category or bad_gt_parse:
        print(
            "[Warn] build_model_response_list_from_infer stats:\n"
            f"  missing_qid: {missing_qid}\n"
            f"  missing_meta: {missing_meta}\n"
            f"  bad_category: {bad_category}\n"
            f"  bad_gt_parse: {bad_gt_parse}\n"
        )

    if not model_response_list:
        raise ValueError("No valid samples built from infer jsonl. Please check paths and formats.")

    print(f"Loaded {len(model_response_list)} model responses, R1_type: {r1_type}")
    return model_response_list

# -----------------------------
# detailed_report
# -----------------------------
def build_detailed_report(raw_results_list: List[Dict[str, Any]], topk: int = 100) -> Dict[str, Any]:
    df = pd.DataFrame(raw_results_list)

    num_df = df[df["qa_type"].isin(["distance", "velocity", "direction"])].copy()
    num_df["abs_err"] = (num_df["pred_value"] - num_df["gt_value"]).abs()
    num_df["missing_pred"] = num_df["pred_value"].isna()

    def agg_block(g: pd.DataFrame) -> Dict[str, Any]:
        valid = g[~g["missing_pred"]]
        out = {"sample_size": int(len(g)), "missing_pred": int(g["missing_pred"].sum())}
        if len(valid) > 0:
            out.update(
                {
                    "mae": float(valid["abs_err"].mean()),
                    "median_ae": float(valid["abs_err"].median()),
                    "p90_ae": float(valid["abs_err"].quantile(0.90)),
                }
            )
        return out

    by_metric_time = {}
    for (ego_tgt, qa_type, time), g in num_df.groupby(["ego_or_target", "qa_type", "time"]):
        by_metric_time[f"{ego_tgt}_{qa_type}_t{time}"] = agg_block(g)

    by_relation = (
        df.groupby(["relation", "qa_type"])["success"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "sample_size", "mean": "success_rate"})
    )
    by_relation_records = by_relation.to_dict(orient="records")

    worst = num_df[~num_df["missing_pred"]].sort_values("abs_err", ascending=False).head(topk)
    worst_cases = worst[
        ["question_id", "qa_category", "time", "ego_or_target", "qa_type", "gt_value", "pred_value", "abs_err", "gt", "pred"]
    ].to_dict(orient="records")

    return {
        "numeric_error_by_metric_time": by_metric_time,
        "success_by_relation": by_relation_records,
        "worst_cases_numeric": worst_cases,
    }

# -----------------------------
# Normalized json output
# -----------------------------
def _calc_overall_primary_success(raw_results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    primary_types = {"distance", "velocity", "direction"}
    total = 0
    succ = 0
    for r in raw_results_list:
        if r.get("qa_type") in primary_types:
            total += 1
            succ += 1 if r.get("success") else 0
    if total == 0:
        # No samples of the primary qa_types were found at all -- typically a mis-matched
        # inference file or a parsing problem. Reporting 0.0 made that look like a genuine
        # "the model got everything wrong" result; report null instead so it is obvious the
        # metric was not computed.
        print(
            "[WARN] overall_successful_rate: no samples with qa_type in "
            f"{sorted(primary_types)}; reporting null instead of 0.0. "
            "Check that the inference file matches the dataset."
        )
        return {"score": None, "metric": "overall_successful_rate", "sample_size": 0}
    return {"score": round(succ / total, 8), "metric": "overall_successful_rate", "sample_size": total}

def build_normalized_scores_json(
    *,
    aggregated: Dict[str, Any],
    raw_results_list: List[Dict[str, Any]],
    detailed_report: Dict[str, Any],
    model_name: str,
    infer_file_path: Path,
    dataset_path: Path,
    tolerance_yaml: Path,
    annotation_csv: Optional[Path],
    r1_type: bool = False,  # New: record whether R1 format is used
) -> Dict[str, Any]:
    secondary_success_rates = {k: v for k, v in aggregated.items() if k not in {"evaluated_at", "temporal_consistency"}}
    secondary_metrics = {
        "evaluated_at": aggregated.get("evaluated_at"),
        "success_rates": secondary_success_rates,
        "temporal_consistency": aggregated.get("temporal_consistency", []),
    }

    overall = {
        "model_name": model_name,
        "infer_file": str(infer_file_path),
        "dataset_path": str(dataset_path),
        "tolerance_yaml": str(tolerance_yaml),
        "annotation_csv": str(annotation_csv) if annotation_csv is not None else None,
        "evaluated_at": aggregated.get("evaluated_at"),
        "r1_type": r1_type,  # New: record whether R1 format is used
    }
    overall.update(_calc_overall_primary_success(raw_results_list))

    detailed_results = {
        "raw_results": raw_results_list,
        "detailed_report": detailed_report,
    }

    return {"overall": overall, "secondary_metrics": secondary_metrics, "detailed_results": detailed_results}

def infer_scores_filename(infer_file_path: Path, r1_type: bool = False) -> str:
    """Generate score filename; if R1 format is used, add _r1 suffix to the filename"""
    base = infer_file_path.stem  # Remove suffix
    if '_result' not in base:
        base = f'{base}_result'
    if r1_type:
        return f"{base}_r1_scores.json"
    return f"{base}_scores.json"

def output_results(
    *,
    out_dir: Path,
    infer_file_path: Path,
    normalized_scores: Dict[str, Any],
    aggregated: Dict[str, Any],
    raw_list: List[Dict[str, Any]],
    detailed_report: Dict[str, Any],
    do_plots: bool,
    r1_type: bool = False,  # New: control filename suffix
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Modification: use filename with R1 suffix
    scores_path = out_dir / infer_scores_filename(infer_file_path, r1_type)
    with scores_path.open("w", encoding="utf-8") as f:
        json.dump(normalized_scores, f, indent=2, ensure_ascii=False)
    print(f"Saved normalized scores to {scores_path}")

    with (out_dir / "metrics_report.json").open("w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=4, ensure_ascii=False)
    print(f"Saved scores to {out_dir / 'metrics_report.json'}")

    generate_metric_tables(aggregated, out_dir / "metrics_report.md")
    print(f"Saved metric report to {out_dir / 'metrics_report.md'}")

    save_path = out_dir / "raw_evaluation.json"
    save_as_jsonl(raw_list, save_path, lines=False, ensure_ascii=False)
    print(f"Saved raw results to {save_path}")

    detail_path = out_dir / "detailed_report.json"
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(detailed_report, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed report to {detail_path}")

    if do_plots:
        print("Generating visualization plots...")
        create_prediction_distribution_plots(raw_list, out_dir)
        create_lsr_analysis_plots(raw_list, out_dir)
        print(f"Saved visualization plots to {out_dir}")

# -----------------------------
# Args / main
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate STRIDE-QA (offline infer jsonl) with normalized score output.")

    parser.add_argument("--infer_file_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--annotation_csv", type=Path, default=None)
    parser.add_argument("--tolerance_yaml", type=Path, required=True)
    parser.add_argument("--no_plots", action="store_true")

    # New: R1 format parameter
    parser.add_argument("--r1_type", action="store_true",
                       help="If set, extract answer from <answer></answer> tags in model response")

    parser.add_argument("--parse_timeout_s", type=float, default=0.05)
    parser.add_argument("--parse_max_chars", type=int, default=800)
    parser.add_argument("--skip_official_if_longer_than", type=int, default=800)

    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    # config
    if not args.tolerance_yaml.exists():
        raise FileNotFoundError(f"tolerance_yaml not found: {args.tolerance_yaml}")
    with open(args.tolerance_yaml, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # annotation
    if args.annotation_csv is None or (not args.annotation_csv.exists()):
        print("Warning: annotation_sheet.csv not found, continue without annotation tags")
        annotation_dict = {}
        annotation_csv = None
    else:
        annotation_dict = load_annotation_data(args.annotation_csv)
        annotation_csv = args.annotation_csv
        print(f"Loaded annotation data for {len(annotation_dict)} groups from {annotation_csv}")

    # meta index
    meta_index = build_meta_index_from_dataset_path(args.dataset_path)
    print(f"Loaded meta index size: {len(meta_index)}")

    # infer files
    if not os.path.isfile(args.infer_file_path):
        raw_infer_files = os.listdir(args.infer_file_path)
        infer_files = [os.path.join(args.infer_file_path, file) for file in raw_infer_files]
    else:
        infer_files = [args.infer_file_path]

    print("Infer files:")
    for p in infer_files:
        print("  ", p)

    # Modification: pass r1_type parameter
    model_response_list = build_model_response_list_from_infer(
        infer_files,
        meta_index,
        r1_type=args.r1_type
    )

    # eval
    raw_results_list = run_evaluation(
        model_response_list,
        config,
        annotation_dict,
        parse_timeout_s=args.parse_timeout_s,
        parse_max_chars=args.parse_max_chars,
        skip_official_if_longer_than=args.skip_official_if_longer_than,
    )


    # debug
    df = pd.DataFrame(raw_results_list)
    print("non-lsr counts by qa_type,time:")
    print(df[df.qa_type != "lsr"].groupby(["qa_type","time"]).size())

    print("direction sample counts by time:")
    print(df[(df.qa_type == "direction")].groupby("time").size())

    print("lsr direction missing by time:")
    lsr = df[df.qa_type == "lsr"].copy()
    lsr["dir_gt_none"] = lsr["gt_value"].apply(lambda x: x.get("direction") is None)
    lsr["dir_pred_none"] = lsr["pred_value"].apply(lambda x: x.get("direction") is None)
    print(lsr.groupby("time")[["dir_gt_none","dir_pred_none"]].mean())


    # aggregate + detailed
    aggregated = aggregate_results(raw_results_list)
    detailed_report = build_detailed_report(raw_results_list, topk=200)

    # Modification: pass r1_type parameter
    normalized_scores = build_normalized_scores_json(
        aggregated=aggregated,
        raw_results_list=raw_results_list,
        detailed_report=detailed_report,
        model_name=args.model_name,
        infer_file_path=args.infer_file_path,
        dataset_path=args.dataset_path,
        tolerance_yaml=args.tolerance_yaml,
        annotation_csv=annotation_csv,
        r1_type=args.r1_type,
    )

    out_dir = args.output_dir / args.model_name
    # Modification: pass r1_type parameter
    output_results(
        out_dir=out_dir,
        infer_file_path=args.infer_file_path,
        normalized_scores=normalized_scores,
        aggregated=aggregated,
        raw_list=raw_results_list,
        detailed_report=detailed_report,
        do_plots=(not args.no_plots),
        r1_type=args.r1_type,
    )

if __name__ == "__main__":
    main(parse_args())
