import os
import logging
import re
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import requests
import time
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments for the evaluation system with unified interface."""
    parser = argparse.ArgumentParser(
        description="DriveLMM Evaluation System with SGLang - Unified Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments for unified interface
    parser.add_argument(
        "--infer_file_path",
        type=str,
        required=True,
        help="Path to the input JSONL file containing evaluation data (model responses and labels)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory for the output files (JSON, CSV, and report)"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model being evaluated (for organizing results)"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the original dataset file (for meta information)"
    )

    # R1 version support
    parser.add_argument(
        "--r1_type",
        action="store_true",
        help="If set, use R1 version answer extraction (from <answer> tag)"
    )

    parser.add_argument(
        "--only_mcq",
        action="store_true",
        help="If set, skip SGLang reasoning quality evaluation and only compute MCQ accuracy"
    )

    # Original evaluation arguments (renamed to avoid confusion)
    parser.add_argument(
        "--server_model_name",
        type=str,
        default="Qwen3_VL_235B_A22B_Thinking",
        help="Model name for evaluation requests (server-side model)"
    )

    parser.add_argument(
        "--server_url",
        type=str,
        required=True,
        help="SGLang server URL (e.g., http://host:port/v1/chat/completions)"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
        help="Maximum tokens for model evaluation responses"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature setting for model generation"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout for API requests in seconds"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API requests in seconds"
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed API calls"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples to process in parallel (set to 1 for sequential)"
    )

    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=1.0,
        help="Ratio of samples to evaluate (for debugging)"
    )

    return parser.parse_args()

def extract_answer_from_tag(text):
    """Extract answer content from <answer> tags (for R1 version)"""
    if not text:
        return ""

    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def extract_final_answer(text, r1_type=False):
    """Extract the final answer from LLM response text."""
    if not text:
        return ""

    # Fix point 1: Correctly distinguish R1 mode, only extract <answer> tags when r1_type is True
    # if r1_type:
    #     text = extract_answer_from_tag(text)
    #     # After R1 mode extracts the tag, return directly to avoid secondary splitting
    #     return text.strip()
    text = extract_answer_from_tag(text)
    # Non-R1 mode: extract final answer by markers
    markers = [
        "**Final Answer**:",
        "The final answer is:",
        "Final Answer:",
        "**Final Answer:**",
        "**Final Answer**",
        "**Final Decision**:",
        "Final Step:",
        "<CONCLUSION>"
    ]

    for marker in markers:
        if marker in text:
            # Extract content after the marker and clean it up
            answer_part = text.split(marker)[-1].strip()
            # Remove extra punctuation/newlines
            answer_part = re.sub(r'[\n\t]+', ' ', answer_part)
            answer_part = re.sub(r':', ' ', answer_part)
            return answer_part
    return text.strip()

def extract_options(question_text):
    """
    Extract multiple choice options from question text, return {letter: option text} mapping
    Supported formats: A) xxx, (A) xxx, A: xxx, A - xxx
    """
    if not question_text:
        return {}

    # Regex for matching options: capture letter (A-F) + delimiter + option text
    pattern = r'([A-Fa-f])[\)\:\-\s]\s*(.+?)(?=\s*[A-Fa-f][\)\:\-\s]|$)'
    matches = re.findall(pattern, question_text, re.DOTALL)

    options = {}
    for letter, text in matches:
        # Normalize letter to uppercase, clean option text (remove newlines, extra spaces)
        upper_letter = letter.upper()
        clean_text = re.sub(r'[\n\t]+', ' ', text).strip()
        # Remove trailing punctuation from option
        clean_text = clean_text.rstrip('.').rstrip(',').rstrip(';').strip()
        options[upper_letter] = clean_text

    # Handle "None of the options" special case
    if any("none of the options" in text.lower() for text in options.values()):
        pass
    elif "none of the options" in question_text.lower():
        options["F"] = "None of the options"

    return options

def extract_mcq_letter(answer_text, options_map):
    """
    Extract multiple choice letter (A-F) from answer text
    Priority: extract letter first, then match option text if letter not found
    """
    if not answer_text or not options_map:
        return ""

    # Step 1: Extract letter (A-F) at any position, ignoring case and delimiters
    letter_pattern = r'(?<!\w)[A-Fa-f](?=[\)\:\-\s]|$)'
    letter_matches = re.findall(letter_pattern, answer_text)
    if letter_matches:
        return letter_matches[0].upper()

    # Step 2: Match option text (fuzzy matching)
    clean_answer = re.sub(r'[^\w\s]', '', answer_text.lower()).strip()
    for letter, option_text in options_map.items():
        clean_option = re.sub(r'[^\w\s]', '', option_text.lower()).strip()
        # Exact or partial match (to handle model returning abbreviations)
        if clean_answer in clean_option or clean_option in clean_answer:
            return letter

    # Step 3: Handle "None of the options"
    if "none of the options" in clean_answer:
        return "F"

    return ""

def create_output_directory(args) -> str:
    """
    Create output directory structure based on unified interface.
    Format: {args.output_dir}/{args.model_name}/
    """
    model_output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    logger.info(f"Output directory: {model_output_dir}")
    return model_output_dir

def load_configuration(args):
    """Load system configuration."""
    system_prompt = """
    You are an autonomous driving reasoning evaluator. Your task is to assess the alignment, coherence, and quality of reasoning steps in text responses for safety-critical driving scenarios.

    You will evaluate the model-generated reasoning using the following metrics:

    1. Faithfulness-Step (1-10)
    Measures how well the model's reasoning steps align with the ground truth.
    9-10: All steps correctly match or closely reflect the reference.
    7-8: Most steps align, with minor deviations.
    5-6: Some steps align, but several are incorrect or missing.
    3-4: Few steps align; most are inaccurate or missing.
    1-2: Majority of steps are incorrect.

    2.  Informativeness-Step (1-10)
    Measures completeness of reasoning:
    9-10: Captures almost all critical information.
    7-8: Covers most key points, with minor omissions.
    5-6: Missing significant details.
    3-4: Only partial reasoning present.
    1-2: Poor extraction of relevant reasoning.

    3.  Risk Assessment Accuracy (1-10)
    Evaluates if the model correctly prioritizes high-risk objects or scenarios.
    9-10: Correctly identifies and prioritizes key dangers.
    7-8: Mostly accurate, with minor misprioritizations.
    5-6: Some important risks are overlooked.
    3-4: Significant misjudgments in risk prioritization.
    1-2: Misidentifies key risks or misses them entirely.

    4. Traffic Rule Adherence (1-10)
    Evaluates whether the response follows traffic laws and driving best practices.
    9-10: Fully compliant with legal and safe driving practices.
    7-8: Minor deviations, but mostly correct.
    5-6: Some inaccuracies in legal/safe driving recommendations.
    3-4: Several rule violations or unsafe suggestions.
    1-2: Promotes highly unsafe driving behavior.

    5.  Scene Awareness & Object Understanding (1-10)
    Measures how well the response interprets objects, their positions, and actions.
    9-10: Clearly understands all relevant objects and their relationships.
    7-8: Minor misinterpretations but mostly correct.
    5-6: Some key objects misunderstood or ignored.
    3-4: Many errors in object recognition and reasoning.
    1-2: Misidentifies or ignores key objects.

    6.  Repetition-Token (1-10)
    Identifies unnecessary repetition in reasoning.
    9-10: No redundancy, very concise.
    7-8: Minor repetition but still clear.
    5-6: Noticeable redundancy.
    3-4: Frequent repetition that disrupts reasoning.
    1-2: Excessive redundancy, making reasoning unclear.

    7.  Hallucination (1-10)
    Detects irrelevant or invented reasoning steps not aligned with ground truth.
    9-10: No hallucinations, all reasoning is grounded.
    7-8: One or two minor hallucinations.
    5-6: Some fabricated details.
    3-4: Frequent hallucinations.
    1-2: Majority of reasoning is hallucinated.

    8.  Semantic Coverage-Step (1-10)
    Checks if the response fully covers the critical reasoning elements.
    9-10: Nearly complete semantic coverage.
    7-8: Good coverage, some minor omissions.
    5-6: Partial coverage with key gaps.
    3-4: Major gaps in reasoning.
    1-2: Very poor semantic coverage.

    9. Commonsense Reasoning (1-10)
    Assesses the use of intuitive driving logic in reasoning.
    9-10: Displays strong commonsense understanding.
    7-8: Mostly correct, with minor gaps.
    5-6: Some commonsense errors.
    3-4: Frequent commonsense mistakes.
    1-2: Lacks basic driving commonsense.

    10. Missing Step (1-10)
    Evaluates if any necessary reasoning steps are missing.
    9-10: No critical steps missing.
    7-8: Minor missing steps, but answer is mostly intact.
    5-6: Some important steps missing.
    3-4: Many critical reasoning gaps.
    1-2: Response is highly incomplete.

    11. Relevance (1-10)
    Measures how well the response is specific to the given scenario and ground truth. This includes whether the answer is too general or vague instead of directly addressing the key aspects of the scenario.
    9-10: Highly specific and directly relevant to the driving scenario. Captures critical elements precisely, with no unnecessary generalization.
    7-8: Mostly relevant, but some minor parts may be overly generic or slightly off-focus.
    5-6: Somewhat relevant but lacks precision; response contains vague or general reasoning without clear scenario-based details.
    3-4: Mostly generic or off-topic reasoning, with significant irrelevant content.
    1-2: Largely irrelevant, missing key aspects of the scenario and failing to align with the ground truth.

    12. Missing Details (1-10)
    Evaluates the extent to which critical information is missing from the response, impacting the reasoning quality.
    9-10: No significant details are missing; response is comprehensive and complete.
    7-8: Covers most important details, with minor omissions that do not severely impact reasoning.
    5-6: Some essential details are missing, affecting the completeness of reasoning.
    3-4: Many critical reasoning steps or contextual details are absent, making the response incomplete.
    1-2: Response is highly lacking in necessary details, leaving major gaps in understanding.

    Final Evaluation:

    Compute the Overall Score as the average of all metric scores.

    Avoid subjective interpretation and adhere to the given thresholds.

    Always strictly follow these scoring guidelines.

    IMPORTANT: You MUST return ONLY a valid JSON object with the following structure, and nothing else. Do not include any additional text, explanations, or markdown formatting. The response must be parseable as JSON.

    Required JSON structure:
    {
        "Faithfulness-Step": <score>,
        "Informativeness-Step": <score>,
        "Risk Assessment Accuracy": <score>,
        "Traffic Rule Adherence": <score>,
        "Scene Awareness & Object Understanding": <score>,
        "Repetition-Token": <score>,
        "Hallucination": <score>,
        "Semantic Coverage-Step": <score>,
        "Commonsense": <score>,
        "Missing Step": <score>,
        "Relevance": <score>,
        "Missing Details": <score>,
        "Overall Score": <average_score>
    }
    """

    config = {
        "args": args,
        "system_prompt": system_prompt.strip()
    }

    return config

def call_qwen_api(prompt: str, config: Dict, retry_count: int = 0) -> Optional[str]:
    """Call Qwen model via SGLang API."""
    args = config["args"]

    payload = {
        "model": args.server_model_name,
        "messages": [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False
    }

    for attempt in range(args.max_retries + 1):
        try:
            response = requests.post(
                args.server_url,
                json=payload,
                timeout=args.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.warning(f"API request failed with status {response.status_code}: {response.text}")
                if attempt < args.max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"All retries failed for API call")
                    return None

        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout (attempt {attempt + 1}/{args.max_retries})")
            if attempt < args.max_retries:
                time.sleep(2 ** attempt)
                continue
            else:
                logger.error("Request timed out after all retries")
                return None

        except Exception as e:
            logger.error(f"API call error: {e}")
            if attempt < args.max_retries:
                time.sleep(2 ** attempt)
                continue
            else:
                return None

    return None

def extract_json_from_response(text: str) -> Optional[Dict]:
    """Extract JSON from model response, handling potential formatting issues."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                cleaned = match.strip().replace('\\"', '"')
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.warning(f"Could not extract valid JSON from response: {text[:200]}...")
                return None

    return None

def evaluate_steps(question: str, ground_truth: str, llm_response: str, config: Dict) -> Optional[Dict]:
    """Evaluate reasoning steps using Qwen model via SGLang."""
    prompt = f"""
Question: {question}

Ground Truth: {ground_truth}

LLM Response: {llm_response}

Evaluate the LLM response against the ground truth using the specified metrics. Return ONLY a valid JSON object with the scores.
"""

    response_text = call_qwen_api(prompt, config)
    if not response_text:
        logger.error("Failed to get evaluation response from model")
        return None

    result = extract_json_from_response(response_text)
    if not result:
        logger.error(f"Could not parse JSON from response: {response_text[:200]}...")
        return None

    return result

def process_sample(sample: Dict, config: Dict, sample_id: str = None) -> Optional[Dict]:
    """Process a single evaluation sample."""
    try:
        # Extract question
        question = ""
        for msg in sample.get("messages", []):
            if msg.get("role") == "user" and "content" in msg:
                content = msg["content"]
                if "Question:" in content:
                    question = content.split("Question:")[-1].strip()
                else:
                    question = content
                break

        if not question:
            logger.warning(f"No question found in sample {sample_id}")
            return None

        # Extract ground truth and model response
        ground_truth = sample.get("labels", "")
        llm_response = sample.get("response", "")
        if not ground_truth or not llm_response:
            logger.warning(f"Missing ground truth/LLM response for sample {sample_id}")
            return None

        # Extract final answer (Fix point 2: correctly call extract_final_answer)
        # breakpoint()
        gt_final_answer = extract_final_answer(ground_truth, r1_type=config['args'].r1_type)
        llm_final_answer = extract_final_answer(llm_response, r1_type=config['args'].r1_type)
        # breakpoint()
        # Calculate MCQ accuracy
        mcq = -1
        mcq_keywords = ["Choose from", "Select from", "A)", "B)", "C)", "D)", "E)", "F)", "(A)", "(B)"]
        is_mcq = any(keyword in question for keyword in mcq_keywords)

        if is_mcq:
            # Extract option mapping (letter -> text)
            options_map = extract_options(question)

            # Extract letters for ground truth and model answer
            gt_letter = extract_mcq_letter(gt_final_answer, options_map)
            llm_letter = extract_mcq_letter(llm_final_answer, options_map)

            # Only calculate accuracy when both letters are extracted
            if gt_letter and llm_letter:
                mcq = 1 if gt_letter == llm_letter else 0
                logger.debug(f"Sample {sample_id} - GT: {gt_letter}, Model: {llm_letter}, Correct: {mcq}")
            else:
                logger.warning(f"Sample {sample_id} - Failed to extract MCQ letters: GT={gt_final_answer[:50]}, Model={llm_final_answer[:50]}")

        # Skip SGLang evaluation (only_mcq mode)
        if config['args'].only_mcq:
            evaluation_result = {}
        else:
            evaluation_result = evaluate_steps(question, ground_truth, llm_response, config)
            if not evaluation_result:
                logger.warning(f"Evaluation failed for sample {sample_id}")
                return None

        # Build result
        result = {
            "id": sample_id or f"sample_{hash(str(sample))}",
            "question": question[:500],
            "ground_truth": ground_truth[:500],
            "llm_response": llm_response[:500],
            "llm_final_answer": llm_final_answer[:200],
            "mcq": mcq
        }
        result.update(evaluation_result)

        return result

    except Exception as e:
        logger.error(f"Error processing sample {sample_id}: {e}", exc_info=True)
        return None

def calculate_statistics(results: List[Dict], only_mcq: bool = False) -> Dict:
    """
    Calculate statistics from evaluation results.

    `only_mcq` must be passed in: this is a module-level function, so it has no access to the
    `config` dict built inside `main()` / `load_configuration()`. The previous version read
    `config['args'].only_mcq` and raised NameError for every non-MCQ run, which happened after
    all the (expensive) judge calls had already been made.
    """
    if not results:
        return {}

    df = pd.DataFrame(results)
    metrics = [
        "Faithfulness-Step", "Informativeness-Step", "Risk Assessment Accuracy",
        "Traffic Rule Adherence", "Scene Awareness & Object Understanding",
        "Repetition-Token", "Hallucination", "Semantic Coverage-Step",
        "Commonsense", "Missing Step", "Relevance", "Missing Details", "Overall Score"
    ]

    stats = {}

    # Calculate statistics for each metric (only in non-only_mcq mode)
    for metric in metrics:
        if metric in df.columns and not df[metric].isnull().all() and not only_mcq:
            values = pd.to_numeric(df[metric], errors='coerce').dropna()
            if len(values) > 0:
                stats[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "median": float(values.median()),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                    "count": int(len(values))
                }

    # Calculate MCQ statistics
    mcq_results = [r for r in results if r.get("mcq", -1) >= 0]
    if mcq_results:
        mcq_scores = [r["mcq"] for r in mcq_results]
        stats["mcq"] = {
            "accuracy": float(sum(mcq_scores) / len(mcq_scores)),
            "total": len(mcq_results),
            "correct": sum(mcq_scores),
            "incorrect": len(mcq_results) - sum(mcq_scores)
        }

    return stats

def generate_unified_scores_file(results: List[Dict], output_dir: str, input_file_path: str, args) -> str:
    """Generate unified scores file in JSON format."""
    basename = os.path.basename(input_file_path)
    if basename.endswith('.jsonl'):
        basename = basename[:-6]
    elif basename.endswith('.json'):
        basename = basename[:-5]

    output_filename = f"{basename}_scores.json"
    output_path = os.path.join(output_dir, output_filename)
    stats = calculate_statistics(results, only_mcq=args.only_mcq)

    # Build overall metrics
    overall_metrics = {}
    if args.only_mcq:
        overall_metrics["only_mcq_mode"] = True
    else:
        if "Overall Score" in stats:
            overall_score_stats = stats["Overall Score"]
            overall_metrics["average_overall_score"] = overall_score_stats["mean"]
            overall_metrics["normalized_overall_score"] = (overall_score_stats["mean"] / 10) * 100
            overall_metrics["score"] = (overall_score_stats["mean"] / 10) * 100
            overall_metrics["metric"] = "reasoning-score"

            if "Overall Score" in [r for r in results if "Overall Score" in r]:
                overall_scores = [r["Overall Score"] for r in results if "Overall Score" in r]
                overall_metrics["score_distribution"] = {
                    "0-3": len([s for s in overall_scores if s <= 3]) / len(overall_scores),
                    "4-6": len([s for s in overall_scores if 4 <= s <= 6]) / len(overall_scores),
                    "7-8": len([s for s in overall_scores if 7 <= s <= 8]) / len(overall_scores),
                    "9-10": len([s for s in overall_scores if s >= 9]) / len(overall_scores)
                }

    # MCQ accuracy (higher priority)
    if "mcq" in stats:
        overall_metrics["score"] = stats["mcq"]["accuracy"]
        overall_metrics["mcq_total"] = stats["mcq"]["total"]
        overall_metrics["mcq_correct"] = stats["mcq"]["correct"]
        overall_metrics['metric'] = 'accuracy'

    # Secondary metrics (only in non-only_mcq mode)
    secondary_metrics = {}
    metrics_to_include = [
        "Faithfulness-Step", "Informativeness-Step", "Risk Assessment Accuracy",
        "Traffic Rule Adherence", "Scene Awareness & Object Understanding",
        "Repetition-Token", "Hallucination", "Semantic Coverage-Step",
        "Commonsense", "Missing Step", "Relevance", "Missing Details"
    ]
    if not args.only_mcq:
        for metric in metrics_to_include:
            if metric in stats:
                secondary_metrics[metric] = stats[metric]["mean"]

    # Detailed results
    detailed_results = []
    for result in results:
        detailed_result = {
            "id": result.get("id", ""),
            "mcq_correct": result.get("mcq", -1) if result.get("mcq", -1) >= 0 else None,
            "llm_final_answer": result.get("llm_final_answer", "")
        }
        if not args.only_mcq:
            for metric in metrics_to_include + ["Overall Score"]:
                if metric in result:
                    detailed_result[metric] = result[metric]
        detailed_results.append(detailed_result)

    # Final score data
    scores_data = {
        "overall": overall_metrics,
        "secondary_metrics": secondary_metrics,
        "detailed_results": detailed_results,
        "metadata": {
            "model_name": args.model_name,
            "server_model_name": args.server_model_name,
            "input_file": input_file_path,
            "dataset_path": args.dataset_path,
            "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "r1_type": args.r1_type,
            "only_mcq": args.only_mcq,
            "total_samples": len(results),
            "failed_samples": len([r for r in results if r.get("Overall Score", 0) == 0 and not args.only_mcq])
        }
    }

    # Save file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Unified scores file saved to: {output_path}")

    # Save detailed results (backward compatible)
    detailed_file = os.path.join(output_dir, f"{basename}_detailed_results.json")
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    return output_path

def main():
    """Main function to execute the evaluation process."""
    args = parse_arguments()
    config = load_configuration(args)
    output_dir = create_output_directory(args)
    config["output_dir"] = output_dir

    logger.info(f"Starting DriveLMM evaluation with unified interface")
    logger.info(f"Input file: {args.infer_file_path}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Evaluated model: {args.model_name}")
    logger.info(f"Server model: {args.server_model_name}")
    logger.info(f"Server URL: {args.server_url}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"R1 version: {args.r1_type}")
    logger.info(f"Only MCQ evaluation mode: {args.only_mcq}")

    # Load data
    try:
        with open(args.infer_file_path, "r", encoding="utf-8") as f:
            samples = [json.loads(line.strip()) for line in f if line.strip()]
            if args.eval_ratio < 1.0:
                original_count = len(samples)
                samples = samples[:round(len(samples) * args.eval_ratio)]
                logger.info(f"Using {len(samples)} out of {original_count} samples (ratio: {args.eval_ratio})")
        logger.info(f"Loaded {len(samples)} samples")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return

    # Load dataset metadata
    dataset_meta = None
    if args.dataset_path and os.path.exists(args.dataset_path):
        try:
            with open(args.dataset_path, "r", encoding="utf-8") as f:
                dataset_meta = [json.loads(line.strip()) for line in f if line.strip()]
            logger.info(f"Loaded dataset metadata from {args.dataset_path}")
        except Exception as e:
            logger.warning(f"Failed to load dataset metadata: {e}")

    # Process samples
    results = []
    failed_samples = []
    for i, sample in enumerate(tqdm(samples, desc="Evaluating samples")):
        sample_id = f"sample_{i+1}"
        result = process_sample(sample, config, sample_id)
        if result:
            results.append(result)
        else:
            failed_samples.append(i)

        # Add delay only in non-only_mcq mode
        if not args.only_mcq and i < len(samples) - 1:
            time.sleep(args.delay)

    # Save results
    if results:
        scores_file = generate_unified_scores_file(results, output_dir, args.infer_file_path, args)

        stats = calculate_statistics(results, only_mcq=args.only_mcq)
        if not args.only_mcq and "Overall Score" in stats:
            overall_stats = stats["Overall Score"]
            normalized_score = (overall_stats["mean"] / 10) * 100
            logger.info(f"Average Overall Score: {overall_stats['mean']:.2f}/10")
            logger.info(f"Normalized Overall Score: {normalized_score:.2f}%")

        if "mcq" in stats:
            mcq_stats = stats["mcq"]
            logger.info(f"MCQ Accuracy: {mcq_stats['accuracy']*100:.2f}% ({mcq_stats['correct']}/{mcq_stats['total']})")

        logger.info(f"Evaluation completed successfully!")
        logger.info(f"Successfully evaluated: {len(results)}/{len(samples)} samples")
        logger.info(f"Unified scores saved to: {scores_file}")

        if failed_samples:
            logger.warning(f"Failed samples: {len(failed_samples)}")
            logger.warning(f"Failed sample indices: {failed_samples}")
    else:
        logger.error("No results were generated!")

    # Save configuration
    config_file = os.path.join(output_dir, "evaluation_config.json")
    config_info = {
        "infer_file_path": args.infer_file_path,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
        "server_model_name": args.server_model_name,
        "server_url": args.server_url,
        "r1_type": args.r1_type,
        "only_mcq": args.only_mcq,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "delay": args.delay,
        "max_retries": args.max_retries,
        "batch_size": args.batch_size,
        "eval_ratio": args.eval_ratio,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(samples),
        "successful_samples": len(results),
        "failed_samples": len(failed_samples)
    }

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    logger.info(f"Configuration saved to: {config_file}")

if __name__ == "__main__":
    main()
