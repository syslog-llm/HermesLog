import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger
from openai import OpenAI
from tqdm import tqdm


# ============================================================
# Task configuration
# ============================================================
LABEL2NAME = {
    1: "Processor CPU Caterr",
    2: "Memory Throttled | Uncorrectable Error Correcting Code",
    3: (
        "Hard Disk Drive Control Error | Computer System Bus Short Circuit | "
        "Programmable Gate Array Device Unknown"
    ),
}
LABEL2DETAILS = LABEL2NAME



# ============================================================
# Helpers
# ============================================================
def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"


def get_api_key() -> str:
    api_key = os.getenv("BYWLAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "API key not found. Set BYWLAI_API_KEY or OPENAI_API_KEY before running."
        )
    return api_key


def normalize_true_label(raw_label):
    """Normalize a ground-truth label when one is available."""
    if raw_label is None:
        return None

    if isinstance(raw_label, int):
        return LABEL2NAME.get(raw_label, str(raw_label))

    if isinstance(raw_label, str):
        stripped = raw_label.strip()
        if stripped.isdigit():
            return LABEL2NAME.get(int(stripped), stripped)
        return stripped

    return str(raw_label)


def usage_value(usage, field: str) -> int:
    if usage is None:
        return 0
    value = getattr(usage, field, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(field)
    return int(value or 0)


def percentile(values, q: int):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def mean_or_none(values):
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def sum_or_zero(values):
    return float(np.sum(np.asarray(values, dtype=float))) if values else 0.0


def extract_json_object(text: str):
    """
    Parse the first valid JSON object from a model response.

    Handles:
    1) plain JSON
    2) ```json ... ``` fences
    3) extra text before the JSON object
    """
    if not isinstance(text, str):
        return None

    cleaned = text.strip()

    # Remove a single Markdown code fence when present.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Fast path.
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # Robust fallback: try raw_decode from every opening brace.
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return None


# ============================================================
# Data
# ============================================================
def load_data(filename: str):
    """
    Load every case from a JSON file.

    Expected top-level format:
    [
        {
            "caseid": ...,
            "content": "...",
            "label": ...,   # optional
            "part": ...     # optional
        },
        ...
    ]

    Every record is returned in its original order.
    """
    input_path = Path(filename)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSON file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Input JSON must be a top-level array/list of case records."
        )

    invalid_records = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            invalid_records.append((index, "record is not a JSON object"))
            continue

        if "caseid" not in record:
            invalid_records.append((index, "missing 'caseid'"))

        if "content" not in record:
            invalid_records.append((index, "missing 'content'"))
        elif not isinstance(record["content"], str):
            invalid_records.append((index, "'content' is not a string"))

    if invalid_records:
        preview = "; ".join(
            f"index {index}: {reason}"
            for index, reason in invalid_records[:10]
        )
        if len(invalid_records) > 10:
            preview += f"; ... and {len(invalid_records) - 10} more"
        raise ValueError(f"Invalid input records: {preview}")

    logger.info(
        f"Loaded {len(data)} records from {input_path}; "
        "all records will be predicted."
    )
    return data


def analyze_label_distribution(data):
    labels = [
        normalize_true_label(record.get("label"))
        for record in data
        if record.get("label") is not None
    ]

    if labels:
        logger.info(f"Label distribution: {Counter(labels)}")
    else:
        logger.info(
            "No ground-truth labels found; prediction will run without "
            "accuracy/classification-report evaluation."
        )


def truncate_log(log: str, max_len: int = 1024):
    if len(log) > max_len:
        logger.debug(f"Log exceeds maximum length; truncate to {max_len} characters")
        return log[:max_len]
    return log


def add_evidence_ids(log: str) -> str:
    """
    The current dataset separates log events with '###'.
    Convert:
        event A###event B###event C
    into:
        #1 event A
        #2 event B
        #3 event C

    These IDs are later used by Reasoning V1 and CRC for evidence binding.
    """
    events = [item.strip() for item in log.split("###") if item.strip()]

    # Fallback for unexpected records without ###.
    if not events:
        events = [log.strip()]

    return "\n".join(
        f"#{idx} {event}"
        for idx, event in enumerate(events, start=1)
    )


def get_valid_evidence_ids(annotated_log: str):
    return set(re.findall(r"(?m)^#\d+\b", annotated_log))


# ============================================================
# LLM API + instrumentation
# ============================================================
def get_completion(client, messages, model, temperature, top_p):
    """
    One API call with client-side instrumentation.

    api_service_time_s is client-observed wall-clock service time of the
    configured API endpoint. It is not pure server-side inference time.
    """
    start = time.perf_counter()

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )

    end = time.perf_counter()

    usage = chat_completion.usage
    api_time = end - start

    metrics = {
        "api_service_time_s": api_time,
        "prompt_tokens": usage_value(usage, "prompt_tokens"),
        "completion_tokens": usage_value(usage, "completion_tokens"),
        "total_tokens": usage_value(usage, "total_tokens"),
    }

    if metrics["total_tokens"] == 0:
        metrics["total_tokens"] = (
            metrics["prompt_tokens"] + metrics["completion_tokens"]
        )

    return chat_completion.choices[0].message.content, metrics


# ============================================================
# Reasoning V1: round 1
# ============================================================
def build_reasoning_messages(annotated_log: str):
    system_prompt = """
You are an expert in intelligent operations and maintenance and server hardware
fault diagnosis.

Your task in this round is to analyze the provided operation fault log by
performing a structured four-stage reasoning process:

Observation -> Analysis -> Inference -> Aggregation

Important constraints:
1. Do NOT output or select the final fault category in this round.
2. Use only information that is present in the input log.
3. Do NOT fabricate severity levels, timestamps, events, hardware states, or
   causal relations that are not supported by the input.
4. Every important observation and every inferred fault clue must cite one or
   more evidence IDs from the input log.
5. Return valid JSON only. Do not use Markdown code fences.
""".strip()

    user_prompt = f"""
The operation fault log is:

{annotated_log}

Perform the following four stages.

1. Observation
Extract important events and explicit abnormal signals that are directly visible
in the log.

For every observation:
- describe the event concisely;
- cite its supporting evidence ID(s).

Do not infer the final fault category here.

2. Analysis
Analyze the observations and identify:
- affected_components: hardware or system components implicated by the evidence;
- abnormal_signals: explicit abnormal behaviors or states present in the log;
- relationships: evidence-supported relationships among the observed events;
- key_evidence: the most important evidence IDs for diagnosis.

Do not introduce information that is absent from the log.

3. Inference
Infer concise fault clues from the Observation and Analysis results.

For every fault clue:
- provide a short description;
- cite the supporting evidence ID(s).

Do not output the final predefined fault category.

4. Aggregation
Combine the fault clues into one concise evidence-grounded conclusion describing
what the log most strongly indicates.

Do NOT name or select any final category from the predefined taxonomy in this
round.

Return JSON only, using exactly this structure:

{{
  "observation": [
    {{
      "event": "...",
      "evidence": ["#1"]
    }}
  ],
  "analysis": {{
    "affected_components": ["..."],
    "abnormal_signals": ["..."],
    "relationships": ["..."],
    "key_evidence": ["#1"]
  }},
  "inference": [
    {{
      "description": "...",
      "evidence": ["#1"]
    }}
  ],
  "aggregation": {{
    "conclusion": "..."
  }}
}}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def validate_reasoning_v1(obj, valid_evidence_ids):
    """
    Hard structural checks for Reasoning V1.

    This script intentionally does NOT perform confidence filtering.
    """
    if not isinstance(obj, dict):
        return False, "reasoning output is not a JSON object"

    required = {"observation", "analysis", "inference", "aggregation"}
    missing = required - set(obj.keys())
    if missing:
        return False, f"missing top-level fields: {sorted(missing)}"

    if not isinstance(obj["observation"], list):
        return False, "observation must be a list"

    if not isinstance(obj["analysis"], dict):
        return False, "analysis must be an object"

    if not isinstance(obj["inference"], list):
        return False, "inference must be a list"

    if not isinstance(obj["aggregation"], dict):
        return False, "aggregation must be an object"

    analysis_required = {
        "affected_components",
        "abnormal_signals",
        "relationships",
        "key_evidence",
    }
    analysis_missing = analysis_required - set(obj["analysis"].keys())
    if analysis_missing:
        return False, f"analysis missing fields: {sorted(analysis_missing)}"

    if "conclusion" not in obj["aggregation"]:
        return False, "aggregation.conclusion is missing"

    # Validate evidence references in observation.
    for idx, item in enumerate(obj["observation"]):
        if not isinstance(item, dict):
            return False, f"observation[{idx}] must be an object"
        if "event" not in item or "evidence" not in item:
            return False, f"observation[{idx}] missing event/evidence"
        if not isinstance(item["evidence"], list):
            return False, f"observation[{idx}].evidence must be a list"

        invalid = set(item["evidence"]) - valid_evidence_ids
        if invalid:
            return False, (
                f"observation[{idx}] contains invalid evidence IDs: "
                f"{sorted(invalid)}"
            )

    # Validate evidence references in inference.
    for idx, item in enumerate(obj["inference"]):
        if not isinstance(item, dict):
            return False, f"inference[{idx}] must be an object"
        if "description" not in item or "evidence" not in item:
            return False, f"inference[{idx}] missing description/evidence"
        if not isinstance(item["evidence"], list):
            return False, f"inference[{idx}].evidence must be a list"

        invalid = set(item["evidence"]) - valid_evidence_ids
        if invalid:
            return False, (
                f"inference[{idx}] contains invalid evidence IDs: "
                f"{sorted(invalid)}"
            )

    # Validate analysis.key_evidence.
    key_evidence = obj["analysis"]["key_evidence"]
    if not isinstance(key_evidence, list):
        return False, "analysis.key_evidence must be a list"

    invalid = set(key_evidence) - valid_evidence_ids
    if invalid:
        return False, (
            f"analysis.key_evidence contains invalid evidence IDs: "
            f"{sorted(invalid)}"
        )

    return True, None


def analyze_log(
    client,
    annotated_log,
    model,
    temperature,
    top_p,
    max_retry,
):
    """
    Round 1:
    produce structured Reasoning V1.

    Retries are used only for malformed/invalid structured output.
    There is no confidence-based filtering in this script.
    """
    messages = build_reasoning_messages(annotated_log)
    valid_evidence_ids = get_valid_evidence_ids(annotated_log)

    total_metrics = {
        "api_service_time_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    last_raw = None
    last_error = None

    for attempt in range(max_retry + 1):
        raw, metrics = get_completion(
            client,
            messages,
            model,
            temperature,
            top_p,
        )

        last_raw = raw

        for key in total_metrics:
            total_metrics[key] += metrics[key]

        parsed = extract_json_object(raw)

        if parsed is None:
            last_error = "unable to parse JSON"
        else:
            is_valid, validation_error = validate_reasoning_v1(
                parsed,
                valid_evidence_ids,
            )
            if is_valid:
                return parsed, raw, total_metrics, attempt

            last_error = validation_error

        logger.warning(
            f"Reasoning V1 validation failed "
            f"(attempt {attempt + 1}/{max_retry + 1}): {last_error}"
        )

    return None, last_raw, total_metrics, max_retry


# ============================================================
# Reasoning-driven diagnosis: round 2
# ============================================================
def build_classification_messages(reasoning_v1):
    taxonomy = "\n".join(
        f"- {label_name}"
        for label_name in LABEL2DETAILS.values()
    )

    system_prompt = """
You are an expert in intelligent operations and maintenance and server hardware
fault diagnosis.

The cloud reasoning stage has already completed Observation, Analysis,
Inference, and Aggregation.

Your task now is only to map that completed reasoning to one final predefined
fault category.

Important constraints:
1. Base the diagnosis only on the supplied Reasoning V1.
2. Do not introduce new evidence or new analysis.
3. fault_type must exactly equal one of the provided category names.
4. Return valid JSON only. Do not use Markdown code fences.
""".strip()

    user_prompt = f"""
Reasoning V1:

{json.dumps(reasoning_v1, ensure_ascii=False, indent=2)}

Allowed fault categories:

{taxonomy}

Select the final category that is best supported by Reasoning V1.

Return JSON only:

{{
  "fault_type": "<exact category name>",
  "explanation": "<brief explanation based only on Reasoning V1>"
}}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_diagnosis(raw_text):
    obj = extract_json_object(raw_text)
    if not isinstance(obj, dict):
        return None, "unable to parse diagnosis JSON"

    fault_type = obj.get("fault_type")
    explanation = obj.get("explanation")

    if fault_type not in LABEL2NAME.values():
        return None, (
            "fault_type is not an exact predefined label: "
            f"{fault_type!r}"
        )

    if not isinstance(explanation, str):
        return None, "explanation must be a string"

    return {
        "fault_type": fault_type,
        "explanation": explanation,
    }, None


def classify_log(
    client,
    reasoning_v1,
    model,
    temperature,
    top_p,
    max_retry,
):
    """
    Round 2:
    classify based only on Reasoning V1.

    The original raw log is intentionally not included in this second API call.
    """
    messages = build_classification_messages(reasoning_v1)

    total_metrics = {
        "api_service_time_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    last_raw = None
    last_error = None

    for attempt in range(max_retry + 1):
        raw, metrics = get_completion(
            client,
            messages,
            model,
            temperature,
            top_p,
        )

        last_raw = raw

        for key in total_metrics:
            total_metrics[key] += metrics[key]

        diagnosis, error = parse_diagnosis(raw)

        if diagnosis is not None:
            return diagnosis, raw, total_metrics, attempt

        last_error = error
        logger.warning(
            f"Diagnosis validation failed "
            f"(attempt {attempt + 1}/{max_retry + 1}): {last_error}"
        )

    return None, last_raw, total_metrics, max_retry


# ============================================================
# Case-level processing
# ============================================================
def process_records(
    client,
    data,
    model,
    temperature,
    top_p,
    max_retry,
    max_log_len,
):
    results = []

    for record in tqdm(data, total=len(data), desc="Predicting all cases"):
        content = truncate_log(record["content"], max_len=max_log_len)
        annotated_content = add_evidence_ids(content)

        true_label = normalize_true_label(record.get("label"))

        # ---------- Stage 1: structured Reasoning V1 ----------
        reasoning_v1, reasoning_raw, reason_metrics, reason_retry_count = (
            analyze_log(
                client=client,
                annotated_log=annotated_content,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_retry=max_retry,
            )
        )

        if reasoning_v1 is None:
            logger.error(
                f'Case {record["caseid"]}: failed to obtain valid Reasoning V1'
            )

            result = {
                "caseid": record["caseid"],
                "part": record.get("part"),
                "label": record.get("label"),
                "true_label": true_label,
                "content": content,
                "annotated_content": annotated_content,
                "model": model,
                "reasoning_v1": None,
                "reasoning_v1_raw": reasoning_raw,
                "diagnosis": None,
                "diagnosis_raw": None,
                "matched_label": "__UNMATCHED__",
                "correct": False if true_label is not None else None,
                "reason_time_s": reason_metrics["api_service_time_s"],
                "classification_time_s": 0.0,
                "api_service_time_s": reason_metrics["api_service_time_s"],
                "prompt_tokens": reason_metrics["prompt_tokens"],
                "completion_tokens": reason_metrics["completion_tokens"],
                "total_tokens": reason_metrics["total_tokens"],
                "reason_retry_count": reason_retry_count,
                "classification_retry_count": 0,
                "api_call_count": 1 + reason_retry_count,
            }
            results.append(result)
            continue

        # ---------- Stage 2: diagnosis from Reasoning V1 only ----------
        diagnosis, diagnosis_raw, class_metrics, class_retry_count = (
            classify_log(
                client=client,
                reasoning_v1=reasoning_v1,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_retry=max_retry,
            )
        )

        if diagnosis is None:
            matched_label = "__UNMATCHED__"
            logger.error(
                f'Case {record["caseid"]}: failed to obtain valid diagnosis'
            )
        else:
            matched_label = diagnosis["fault_type"]

        reason_time_s = reason_metrics["api_service_time_s"]
        classification_time_s = class_metrics["api_service_time_s"]
        api_service_time_s = reason_time_s + classification_time_s

        prompt_tokens = (
            reason_metrics["prompt_tokens"] + class_metrics["prompt_tokens"]
        )
        completion_tokens = (
            reason_metrics["completion_tokens"] + class_metrics["completion_tokens"]
        )
        total_tokens = (
            reason_metrics["total_tokens"] + class_metrics["total_tokens"]
        )

        effective_tps = (
            completion_tokens / api_service_time_s
            if api_service_time_s > 0
            else 0.0
        )

        result = {
            "caseid": record["caseid"],
            "part": record.get("part"),
            "label": record.get("label"),
            "true_label": true_label,
            "content": content,
            "annotated_content": annotated_content,
            "model": model,

            # Core outputs for later CRC.
            "reasoning_v1": reasoning_v1,
            "diagnosis": diagnosis,

            # Raw model outputs retained for debugging.
            "reasoning_v1_raw": reasoning_raw,
            "diagnosis_raw": diagnosis_raw,

            # Evaluation / instrumentation.
            "matched_label": matched_label,
            "correct": (
                matched_label == true_label
                if true_label is not None
                else None
            ),
            "reason_time_s": reason_time_s,
            "classification_time_s": classification_time_s,
            "api_service_time_s": api_service_time_s,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "effective_token_throughput_tokens_s": effective_tps,
            "reason_retry_count": reason_retry_count,
            "classification_retry_count": class_retry_count,
            "api_call_count": (
                2 + reason_retry_count + class_retry_count
            ),
        }

        results.append(result)

        if true_label is None:
            logger.info(
                f'Case {record["caseid"]}: '
                f'pred={matched_label} | '
                f'API={api_service_time_s:.4f}s | '
                f'prompt={prompt_tokens} | completion={completion_tokens}'
            )
        elif result["correct"]:
            logger.success(
                f'Case {record["caseid"]}: correct | '
                f'pred={matched_label} | '
                f'API={api_service_time_s:.4f}s | '
                f'prompt={prompt_tokens} | completion={completion_tokens}'
            )
        else:
            logger.warning(
                f'Case {record["caseid"]}: true={true_label} | '
                f'pred={matched_label} | '
                f'API={api_service_time_s:.4f}s | '
                f'prompt={prompt_tokens} | completion={completion_tokens}'
            )

    return results


# ============================================================
# Aggregate statistics
# ============================================================
def analyze_performance(results):
    api_times = [r["api_service_time_s"] for r in results]
    reason_times = [r["reason_time_s"] for r in results]
    class_times = [r["classification_time_s"] for r in results]
    prompt_tokens = [r["prompt_tokens"] for r in results]
    completion_tokens = [r["completion_tokens"] for r in results]
    total_tokens = [r["total_tokens"] for r in results]

    total_completion = int(sum(completion_tokens))
    total_api_time = sum_or_zero(api_times)

    aggregate_effective_tps = (
        total_completion / total_api_time
        if total_api_time > 0
        else 0.0
    )

    valid_reasoning_count = sum(
        1 for r in results if r["reasoning_v1"] is not None
    )
    valid_diagnosis_count = sum(
        1 for r in results if r["diagnosis"] is not None
    )

    stats = {
        "num_cases": len(results),
        "valid_reasoning_v1_count": valid_reasoning_count,
        "valid_diagnosis_count": valid_diagnosis_count,

        "api_service_time_mean_s": mean_or_none(api_times),
        "api_service_time_p50_s": percentile(api_times, 50),
        "api_service_time_p95_s": percentile(api_times, 95),

        "reason_time_mean_s": mean_or_none(reason_times),
        "reason_time_p50_s": percentile(reason_times, 50),
        "reason_time_p95_s": percentile(reason_times, 95),

        "classification_time_mean_s": mean_or_none(class_times),
        "classification_time_p50_s": percentile(class_times, 50),
        "classification_time_p95_s": percentile(class_times, 95),

        "prompt_tokens_mean_per_case": mean_or_none(prompt_tokens),
        "completion_tokens_mean_per_case": mean_or_none(completion_tokens),
        "total_tokens_mean_per_case": mean_or_none(total_tokens),

        "prompt_tokens_total": int(sum(prompt_tokens)),
        "completion_tokens_total": total_completion,
        "total_tokens_total": int(sum(total_tokens)),

        "effective_token_throughput_tokens_s": float(
            aggregate_effective_tps
        ),
    }

    logger.info("===== LLM Performance Summary =====")
    logger.info(json.dumps(stats, ensure_ascii=False, indent=2))

    return stats


def analyze_correctness(results):
    if not results:
        logger.warning("No results; skip correctness analysis")
        return None

    labeled_results = [
        result
        for result in results
        if result.get("true_label") is not None
    ]

    if not labeled_results:
        logger.info(
            "No ground-truth labels are available; "
            "skip correctness analysis."
        )
        return None

    correct = sum(
        1
        for result in labeled_results
        if result["correct"] is True
    )
    accuracy = correct / len(labeled_results)

    logger.info(
        f"Accuracy: {accuracy * 100:.2f}% "
        f"({correct}/{len(labeled_results)} labeled cases)"
    )

    try:
        from sklearn.metrics import classification_report

        y_true = [result["true_label"] for result in labeled_results]
        y_pred = [result["matched_label"] for result in labeled_results]

        all_labels = list(LABEL2NAME.values())

        report = classification_report(
            y_true,
            y_pred,
            target_names=all_labels,
            labels=all_labels,
            digits=4,
            zero_division=0,
        )

        logger.info(f"Classification Report:\n{report}")

    except Exception as exc:
        logger.warning(
            f"Could not generate classification report: {exc}"
        )

    return accuracy


# ============================================================
# Output
# ============================================================
def save_json(obj, filename: Path):
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"Saved: {filename}")


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate structured HermesLog Reasoning V1 and "
            "reasoning-driven 3-class diagnoses."
        )
    )

    parser.add_argument(
        "--input",
        default="data/60-unlabeled.json",
        help="Input JSON file; every record in the top-level array is processed",
    )

    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "gpt-4o-all"),
        help="Model name accepted by the configured OpenAI-compatible endpoint",
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "LLM_BASE_URL",
            "https://api.bywlai.cn/v1",
        ),
        help="OpenAI-compatible API base URL",
    )

    # Kept identical to the original script by default.
    # For more deterministic JSON generation, you can run with:
    #   --temperature 0.2
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--max-retry",
        type=int,
        default=3,
        help=(
            "Maximum retries for malformed Reasoning V1 "
            "or diagnosis JSON"
        ),
    )

    parser.add_argument(
        "--max-log-len",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--output-dir",
        default="output-reasoning-v1",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = time.strftime("%Y%m%d%H%M%S")
    model_tag = safe_model_name(args.model)

    logger.add(
        output_dir / f"{model_tag}_{timestamp}.log"
    )

    logger.info(f"Model: {args.model}")
    logger.info(f"Base URL: {args.base_url}")
    logger.info(f"Input: {args.input}")
    logger.info(
        f"Decoding: temperature={args.temperature}, "
        f"top_p={args.top_p}, "
        f"max_retry={args.max_retry}"
    )

    client = OpenAI(
        api_key=get_api_key(),
        base_url=args.base_url,
    )

    data = load_data(args.input)

    if not data:
        raise RuntimeError(
            "Input JSON contains no records to process."
        )

    analyze_label_distribution(data)

    wall_start = time.perf_counter()

    results = process_records(
        client=client,
        data=data,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_retry=args.max_retry,
        max_log_len=args.max_log_len,
    )

    wall_total = (
        time.perf_counter() - wall_start
    )

    performance = analyze_performance(results)
    accuracy = analyze_correctness(results)

    performance["model"] = args.model
    performance["base_url"] = args.base_url
    performance["input_file"] = args.input
    performance["input_records_loaded"] = len(data)
    performance["all_input_records_processed"] = len(results) == len(data)
    performance["temperature"] = args.temperature
    performance["top_p"] = args.top_p
    performance["max_retry"] = args.max_retry
    performance["max_log_len"] = args.max_log_len
    performance["script_wall_clock_total_s"] = wall_total
    performance["accuracy"] = accuracy

    performance["note"] = (
        "This script intentionally does not perform confidence "
        "calculation or theta-based filtering. It only produces "
        "structured Reasoning V1 and final diagnosis for later CRC."
    )

    reasoning_output = (
        output_dir
        / f"{model_tag}_{timestamp}_reasoning_v1.json"
    )

    summary_output = (
        output_dir
        / f"{model_tag}_{timestamp}_summary.json"
    )

    save_json(results, reasoning_output)
    save_json(performance, summary_output)

    logger.info(
        f"Script wall-clock: {wall_total:.2f}s; "
        f"formal API-service mean: "
        f"{performance['api_service_time_mean_s']:.4f}s/case"
    )

    logger.info("Reasoning V1 generation complete.")


if __name__ == "__main__":
    main()
