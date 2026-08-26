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
# Dataset2 configuration
# ============================================================
LABEL2NAME = {
    1: "Power Supply Fault",
    2: "Fan Fault",
    3: "Optics Module Fault",
    4: "Port Failure",
    6: "CRC Error",
    7: "STP Fault",
    8: "BFD Down",
    9: "LACP Flapping",
    10: "OSPF Neighbor Flapping",
}


# ============================================================
# Defaults: aligned with the first script
# ============================================================
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-all")
DEFAULT_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://api.bywlai.cn/v1",
)
DEFAULT_TEMPERATURE = 1.2
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_RETRY = 3
DEFAULT_MAX_LOG_LEN = 1024


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



def normalize_true_label(raw_label, label2name):
    """Normalize a ground-truth label when one is available."""
    if raw_label is None:
        return None

    if isinstance(raw_label, int):
        return label2name.get(raw_label, str(raw_label))

    if isinstance(raw_label, str):
        stripped = raw_label.strip()
        if stripped.isdigit():
            return label2name.get(int(stripped), stripped)
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

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

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
    Load every case from the input JSON file.

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

    No case-ID filtering is performed.
    Records are processed in their original JSON order.
    """
    input_path = Path(filename)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input JSON file not found: {input_path}"
        )

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Input JSON must be a top-level array/list of case records."
        )

    invalid_records = []

    for index, record in enumerate(data):
        if not isinstance(record, dict):
            invalid_records.append(
                (index, "record is not a JSON object")
            )
            continue

        if "caseid" not in record:
            invalid_records.append(
                (index, "missing 'caseid'")
            )

        if "content" not in record:
            invalid_records.append(
                (index, "missing 'content'")
            )
        elif not isinstance(record["content"], str):
            invalid_records.append(
                (index, "'content' is not a string")
            )

    if invalid_records:
        preview = "; ".join(
            f"index {index}: {reason}"
            for index, reason in invalid_records[:10]
        )
        if len(invalid_records) > 10:
            preview += (
                f"; ... and {len(invalid_records) - 10} more"
            )
        raise ValueError(
            f"Invalid input records: {preview}"
        )

    logger.info(
        f"Loaded {len(data)} records from {input_path}; "
        "all records will be predicted."
    )

    return data


def analyze_label_distribution(data, label2name):
    labels = [
        normalize_true_label(record.get("label"), label2name)
        for record in data
        if record.get("label") is not None
    ]

    if labels:
        logger.info(
            f"Label distribution: {Counter(labels)}"
        )
    else:
        logger.info(
            "No ground-truth labels found; prediction will run "
            "without accuracy/classification-report evaluation."
        )


def truncate_log(log: str, max_len: int = DEFAULT_MAX_LOG_LEN):
    if len(log) > max_len:
        logger.debug(
            f"Log exceeds maximum length; "
            f"truncate to {max_len} characters"
        )
        return log[:max_len]
    return log


def add_evidence_ids(log: str) -> str:
    """
    Convert:
        event A###event B###event C

    into:
        #1 event A
        #2 event B
        #3 event C
    """
    events = [
        item.strip()
        for item in log.split("###")
        if item.strip()
    ]

    if not events:
        events = [log.strip()]

    return "\n".join(
        f"#{idx} {event}"
        for idx, event in enumerate(events, start=1)
    )


def get_valid_evidence_ids(annotated_log: str):
    return set(
        re.findall(r"(?m)^#\d+\b", annotated_log)
    )


# ============================================================
# LLM API + instrumentation
# ============================================================
def get_completion(
    client,
    messages,
    model,
    temperature,
    top_p,
):
    """
    One API call with client-side instrumentation.

    api_service_time_s is client-observed wall-clock service time
    of the configured API endpoint.
    """
    start = time.perf_counter()

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )

    end = time.perf_counter()

    usage = getattr(chat_completion, "usage", None)

    metrics = {
        "api_service_time_s": end - start,
        "prompt_tokens": usage_value(
            usage,
            "prompt_tokens",
        ),
        "completion_tokens": usage_value(
            usage,
            "completion_tokens",
        ),
        "total_tokens": usage_value(
            usage,
            "total_tokens",
        ),
    }

    if metrics["total_tokens"] == 0:
        metrics["total_tokens"] = (
            metrics["prompt_tokens"]
            + metrics["completion_tokens"]
        )

    return (
        chat_completion.choices[0].message.content,
        metrics,
    )


# ============================================================
# Round 1: common structured Reasoning V1
# ============================================================
def build_reasoning_messages(annotated_log: str):
    """
    Common to dataset2 and .

    The first round intentionally does NOT receive the fault taxonomy,
    so reasoning remains category-independent.
    """
    system_prompt = """
You are an expert in intelligent operations and maintenance (AIOps) and
fault diagnosis.

Your task in this round is to analyze the provided operation fault log by
performing a structured four-stage reasoning process:

Observation -> Analysis -> Inference -> Aggregation

Important constraints:
1. Do NOT output or select the final fault category in this round.
2. Use only information that is present in the input log.
3. Do NOT fabricate severity levels, timestamps, events, system states, or
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
- affected_components: components or system entities implicated by the evidence;
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
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def validate_reasoning_v1(
    obj,
    valid_evidence_ids,
):
    if not isinstance(obj, dict):
        return (
            False,
            "reasoning output is not a JSON object",
        )

    required = {
        "observation",
        "analysis",
        "inference",
        "aggregation",
    }

    missing = required - set(obj.keys())
    if missing:
        return (
            False,
            f"missing top-level fields: {sorted(missing)}",
        )

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

    analysis_missing = (
        analysis_required
        - set(obj["analysis"].keys())
    )

    if analysis_missing:
        return (
            False,
            f"analysis missing fields: "
            f"{sorted(analysis_missing)}",
        )

    if "conclusion" not in obj["aggregation"]:
        return (
            False,
            "aggregation.conclusion is missing",
        )

    for idx, item in enumerate(
        obj["observation"]
    ):
        if not isinstance(item, dict):
            return (
                False,
                f"observation[{idx}] must be an object",
            )

        if (
            "event" not in item
            or "evidence" not in item
        ):
            return (
                False,
                f"observation[{idx}] missing "
                "event/evidence",
            )

        if not isinstance(
            item["evidence"],
            list,
        ):
            return (
                False,
                f"observation[{idx}].evidence "
                "must be a list",
            )

        invalid = (
            set(item["evidence"])
            - valid_evidence_ids
        )

        if invalid:
            return (
                False,
                f"observation[{idx}] contains "
                f"invalid evidence IDs: "
                f"{sorted(invalid)}",
            )

    for idx, item in enumerate(
        obj["inference"]
    ):
        if not isinstance(item, dict):
            return (
                False,
                f"inference[{idx}] must be an object",
            )

        if (
            "description" not in item
            or "evidence" not in item
        ):
            return (
                False,
                f"inference[{idx}] missing "
                "description/evidence",
            )

        if not isinstance(
            item["evidence"],
            list,
        ):
            return (
                False,
                f"inference[{idx}].evidence "
                "must be a list",
            )

        invalid = (
            set(item["evidence"])
            - valid_evidence_ids
        )

        if invalid:
            return (
                False,
                f"inference[{idx}] contains "
                f"invalid evidence IDs: "
                f"{sorted(invalid)}",
            )

    key_evidence = (
        obj["analysis"]["key_evidence"]
    )

    if not isinstance(
        key_evidence,
        list,
    ):
        return (
            False,
            "analysis.key_evidence must be a list",
        )

    invalid = (
        set(key_evidence)
        - valid_evidence_ids
    )

    if invalid:
        return (
            False,
            "analysis.key_evidence contains "
            f"invalid evidence IDs: "
            f"{sorted(invalid)}",
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
    messages = build_reasoning_messages(
        annotated_log
    )

    valid_evidence_ids = (
        get_valid_evidence_ids(
            annotated_log
        )
    )

    total_metrics = {
        "api_service_time_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    last_raw = None
    last_error = None

    for attempt in range(
        max_retry + 1
    ):
        raw, metrics = get_completion(
            client=client,
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )

        last_raw = raw

        for key in total_metrics:
            total_metrics[key] += (
                metrics[key]
            )

        parsed = extract_json_object(
            raw
        )

        if parsed is None:
            last_error = (
                "unable to parse JSON"
            )
        else:
            (
                is_valid,
                validation_error,
            ) = validate_reasoning_v1(
                parsed,
                valid_evidence_ids,
            )

            if is_valid:
                return (
                    parsed,
                    raw,
                    total_metrics,
                    attempt,
                )

            last_error = (
                validation_error
            )

        logger.warning(
            "Reasoning V1 validation "
            f"failed (attempt "
            f"{attempt + 1}/"
            f"{max_retry + 1}): "
            f"{last_error}"
        )

    return (
        None,
        last_raw,
        total_metrics,
        max_retry,
    )


# ============================================================
# Round 2: common reasoning-driven diagnosis
# ============================================================
def build_classification_messages(
    reasoning_v1,
    label2name,
):
    taxonomy = "\n".join(
        f"- {label_name}"
        for label_name
        in label2name.values()
    )

    system_prompt = """
You are an expert in intelligent operations and maintenance (AIOps) and
fault diagnosis.

The reasoning stage has already completed Observation, Analysis,
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
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def parse_diagnosis(
    raw_text,
    label2name,
):
    """
    Strict diagnosis parsing, aligned with the first script.

    fault_type must be an exact member of label2name.values().
    """
    obj = extract_json_object(
        raw_text
    )

    if not isinstance(obj, dict):
        return (
            None,
            "unable to parse diagnosis JSON",
        )

    fault_type = obj.get(
        "fault_type"
    )
    explanation = obj.get(
        "explanation"
    )

    if fault_type not in label2name.values():
        return (
            None,
            "fault_type is not an exact "
            "predefined label: "
            f"{fault_type!r}",
        )

    if not isinstance(
        explanation,
        str,
    ):
        return (
            None,
            "explanation must be a string",
        )

    return {
        "fault_type": fault_type,
        "explanation": explanation,
    }, None


def classify_log(
    client,
    reasoning_v1,
    label2name,
    model,
    temperature,
    top_p,
    max_retry,
):
    """
    Round 2 uses Reasoning V1 only.
    The original raw log is intentionally not included.
    """
    messages = (
        build_classification_messages(
            reasoning_v1,
            label2name,
        )
    )

    total_metrics = {
        "api_service_time_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    last_raw = None
    last_error = None

    for attempt in range(
        max_retry + 1
    ):
        raw, metrics = get_completion(
            client=client,
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )

        last_raw = raw

        for key in total_metrics:
            total_metrics[key] += (
                metrics[key]
            )

        diagnosis, error = (
            parse_diagnosis(
                raw,
                label2name,
            )
        )

        if diagnosis is not None:
            return (
                diagnosis,
                raw,
                total_metrics,
                attempt,
            )

        last_error = error

        logger.warning(
            "Diagnosis validation failed "
            f"(attempt {attempt + 1}/"
            f"{max_retry + 1}): "
            f"{last_error}"
        )

    return (
        None,
        last_raw,
        total_metrics,
        max_retry,
    )


# ============================================================
# Case-level processing
# ============================================================
def process_records(
    client,
    data,
    label2name,
    model,
    temperature,
    top_p,
    max_retry,
    max_log_len,
):
    results = []

    for record in tqdm(
        data,
        total=len(data),
        desc="Predicting dataset2",
    ):
        case_start = time.perf_counter()

        content = truncate_log(
            record["content"],
            max_len=max_log_len,
        )

        annotated_content = (
            add_evidence_ids(content)
        )

        true_label = normalize_true_label(
            record.get("label"),
            label2name,
        )

        (
            reasoning_v1,
            reasoning_raw,
            reason_metrics,
            reason_retry_count,
        ) = analyze_log(
            client=client,
            annotated_log=annotated_content,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_retry=max_retry,
        )

        if reasoning_v1 is None:
            case_runtime_s = (
                time.perf_counter()
                - case_start
            )

            reason_time_s = (
                reason_metrics[
                    "api_service_time_s"
                ]
            )

            result = {
                "caseid": record["caseid"],
                "part": record.get("part"),
                "label": record.get("label"),
                "true_label": true_label,
                "content": content,
                "annotated_content": (
                    annotated_content
                ),
                "model": model,
                "reasoning_v1": None,
                "reasoning_v1_raw": (
                    reasoning_raw
                ),
                "diagnosis": None,
                "diagnosis_raw": None,
                "matched_label": "__UNMATCHED__",
                "correct": (
                    False
                    if true_label is not None
                    else None
                ),
                "case_runtime_s": (
                    case_runtime_s
                ),
                "reason_time_s": (
                    reason_time_s
                ),
                "classification_time_s": 0.0,
                "api_service_time_s": (
                    reason_time_s
                ),
                "client_overhead_s": max(
                    0.0,
                    case_runtime_s
                    - reason_time_s,
                ),
                "prompt_tokens": (
                    reason_metrics[
                        "prompt_tokens"
                    ]
                ),
                "completion_tokens": (
                    reason_metrics[
                        "completion_tokens"
                    ]
                ),
                "total_tokens": (
                    reason_metrics[
                        "total_tokens"
                    ]
                ),
                "effective_token_throughput_tokens_s": (
                    reason_metrics[
                        "completion_tokens"
                    ] / reason_time_s
                    if reason_time_s > 0
                    else 0.0
                ),
                "reason_retry_count": (
                    reason_retry_count
                ),
                "classification_retry_count": 0,
                "api_call_count": (
                    1 + reason_retry_count
                ),
            }

            results.append(result)

            logger.error(
                f'Case {record["caseid"]}: '
                "failed to obtain valid "
                "Reasoning V1"
            )
            continue

        (
            diagnosis,
            diagnosis_raw,
            class_metrics,
            class_retry_count,
        ) = classify_log(
            client=client,
            reasoning_v1=reasoning_v1,
            label2name=label2name,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_retry=max_retry,
        )

        if diagnosis is None:
            matched_label = "__UNMATCHED__"
            logger.error(
                f'Case {record["caseid"]}: '
                "failed to obtain valid "
                "diagnosis"
            )
        else:
            matched_label = (
                diagnosis["fault_type"]
            )

        reason_time_s = (
            reason_metrics[
                "api_service_time_s"
            ]
        )

        classification_time_s = (
            class_metrics[
                "api_service_time_s"
            ]
        )

        api_service_time_s = (
            reason_time_s
            + classification_time_s
        )

        prompt_tokens = (
            reason_metrics["prompt_tokens"]
            + class_metrics["prompt_tokens"]
        )

        completion_tokens = (
            reason_metrics[
                "completion_tokens"
            ]
            + class_metrics[
                "completion_tokens"
            ]
        )

        total_tokens = (
            reason_metrics["total_tokens"]
            + class_metrics["total_tokens"]
        )

        case_runtime_s = (
            time.perf_counter()
            - case_start
        )

        client_overhead_s = max(
            0.0,
            case_runtime_s
            - api_service_time_s,
        )

        effective_tps = (
            completion_tokens
            / api_service_time_s
            if api_service_time_s > 0
            else 0.0
        )

        result = {
            "caseid": record["caseid"],
            "part": record.get("part"),
            "label": record.get("label"),
            "true_label": true_label,
            "content": content,
            "annotated_content": (
                annotated_content
            ),
            "model": model,
            "reasoning_v1": reasoning_v1,
            "diagnosis": diagnosis,
            "reasoning_v1_raw": (
                reasoning_raw
            ),
            "diagnosis_raw": diagnosis_raw,
            "matched_label": matched_label,
            "correct": (
                matched_label == true_label
                if true_label is not None
                else None
            ),
            "case_runtime_s": (
                case_runtime_s
            ),
            "reason_time_s": reason_time_s,
            "classification_time_s": (
                classification_time_s
            ),
            "api_service_time_s": (
                api_service_time_s
            ),
            "client_overhead_s": (
                client_overhead_s
            ),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": (
                completion_tokens
            ),
            "total_tokens": total_tokens,
            "effective_token_throughput_tokens_s": (
                effective_tps
            ),
            "reason_retry_count": (
                reason_retry_count
            ),
            "classification_retry_count": (
                class_retry_count
            ),
            "api_call_count": (
                2
                + reason_retry_count
                + class_retry_count
            ),
        }

        results.append(result)

        if true_label is None:
            logger.info(
                f'Case {record["caseid"]}: '
                f'pred={matched_label} | '
                f'API={api_service_time_s:.4f}s | '
                f'prompt={prompt_tokens} | '
                f'completion={completion_tokens}'
            )
        elif result["correct"]:
            logger.success(
                f'Case {record["caseid"]}: '
                f'correct | '
                f'pred={matched_label} | '
                f'API={api_service_time_s:.4f}s | '
                f'prompt={prompt_tokens} | '
                f'completion={completion_tokens}'
            )
        else:
            logger.warning(
                f'Case {record["caseid"]}: '
                f'true={true_label} | '
                f'pred={matched_label} | '
                f'API={api_service_time_s:.4f}s | '
                f'prompt={prompt_tokens} | '
                f'completion={completion_tokens}'
            )

    return results


# ============================================================
# Aggregate statistics
# ============================================================
def analyze_performance(results):
    api_times = [
        r["api_service_time_s"]
        for r in results
    ]
    reason_times = [
        r["reason_time_s"]
        for r in results
    ]
    class_times = [
        r["classification_time_s"]
        for r in results
    ]
    case_runtimes = [
        r["case_runtime_s"]
        for r in results
    ]
    client_overheads = [
        r["client_overhead_s"]
        for r in results
    ]

    prompt_tokens = [
        r["prompt_tokens"]
        for r in results
    ]
    completion_tokens = [
        r["completion_tokens"]
        for r in results
    ]
    total_tokens = [
        r["total_tokens"]
        for r in results
    ]

    total_completion = int(
        sum(completion_tokens)
    )
    total_api_time = sum_or_zero(
        api_times
    )

    aggregate_effective_tps = (
        total_completion
        / total_api_time
        if total_api_time > 0
        else 0.0
    )

    valid_reasoning_count = sum(
        1
        for r in results
        if r["reasoning_v1"] is not None
    )

    valid_diagnosis_count = sum(
        1
        for r in results
        if r["diagnosis"] is not None
    )

    stats = {
        "num_cases": len(results),
        "valid_reasoning_v1_count": (
            valid_reasoning_count
        ),
        "valid_diagnosis_count": (
            valid_diagnosis_count
        ),

        "case_runtime_mean_s": (
            mean_or_none(case_runtimes)
        ),
        "case_runtime_p50_s": percentile(
            case_runtimes,
            50,
        ),
        "case_runtime_p95_s": percentile(
            case_runtimes,
            95,
        ),

        "api_service_time_mean_s": (
            mean_or_none(api_times)
        ),
        "api_service_time_p50_s": percentile(
            api_times,
            50,
        ),
        "api_service_time_p95_s": percentile(
            api_times,
            95,
        ),

        "reason_time_mean_s": (
            mean_or_none(reason_times)
        ),
        "reason_time_p50_s": percentile(
            reason_times,
            50,
        ),
        "reason_time_p95_s": percentile(
            reason_times,
            95,
        ),

        "classification_time_mean_s": (
            mean_or_none(class_times)
        ),
        "classification_time_p50_s": (
            percentile(
                class_times,
                50,
            )
        ),
        "classification_time_p95_s": (
            percentile(
                class_times,
                95,
            )
        ),

        "client_overhead_mean_s": (
            mean_or_none(client_overheads)
        ),
        "client_overhead_p50_s": percentile(
            client_overheads,
            50,
        ),
        "client_overhead_p95_s": percentile(
            client_overheads,
            95,
        ),

        "prompt_tokens_mean_per_case": (
            mean_or_none(prompt_tokens)
        ),
        "completion_tokens_mean_per_case": (
            mean_or_none(completion_tokens)
        ),
        "total_tokens_mean_per_case": (
            mean_or_none(total_tokens)
        ),

        "prompt_tokens_total": int(
            sum(prompt_tokens)
        ),
        "completion_tokens_total": (
            total_completion
        ),
        "total_tokens_total": int(
            sum(total_tokens)
        ),

        "effective_token_throughput_tokens_s": (
            float(
                aggregate_effective_tps
            )
        ),
    }

    logger.info(
        "===== LLM Performance Summary ====="
    )
    logger.info(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
    )

    return stats


def analyze_correctness(
    results,
    label2name,
):
    if not results:
        logger.warning(
            "No results; skip correctness analysis"
        )
        return None

    labeled_results = [
        result
        for result in results
        if result.get("true_label")
        is not None
    ]

    if not labeled_results:
        logger.info(
            "No ground-truth labels are "
            "available; skip correctness "
            "analysis."
        )
        return None

    correct = sum(
        1
        for result in labeled_results
        if result["correct"] is True
    )

    accuracy = (
        correct
        / len(labeled_results)
    )

    logger.info(
        f"Accuracy: {accuracy * 100:.2f}% "
        f"({correct}/"
        f"{len(labeled_results)} "
        "labeled cases)"
    )

    try:
        from sklearn.metrics import (
            classification_report,
        )

        y_true = [
            result["true_label"]
            for result in labeled_results
        ]
        y_pred = [
            result["matched_label"]
            for result in labeled_results
        ]

        all_labels = list(
            label2name.values()
        )

        report = classification_report(
            y_true,
            y_pred,
            target_names=all_labels,
            labels=all_labels,
            digits=4,
            zero_division=0,
        )

        logger.info(
            "Classification Report:\n"
            f"{report}"
        )

    except Exception as exc:
        logger.warning(
            "Could not generate "
            "classification report: "
            f"{exc}"
        )

    return accuracy


# ============================================================
# Dataset2 -> CRC-compatible input adapter
# ============================================================
def build_dataset2_crc_input(
    source_records,
    inference_results,
    label2name,
):
    """
    Build a CRC-compatible input JSON for dataset2.

    The CRC script requires each record to contain:
      - caseid
      - part
      - label
      - user_content
      - assistant_content

    Important:
    - user_content uses the FULL original source log, not the
      max_log_len-truncated cloud-inference content.
    - assistant_content contains structured Reasoning V1 plus the
      cloud diagnosis as a final `Label: ...` line.
    - If cloud reasoning/diagnosis failed, the record is still kept
      when possible so the CRC stage can run and expose the issue.
    """
    if len(source_records) != len(inference_results):
        raise RuntimeError(
            "Cannot build CRC input: source/result lengths differ: "
            f"{len(source_records)} != {len(inference_results)}"
        )

    taxonomy_text = " | ".join(
        label2name.values()
    )

    crc_records = []

    for source_record, result in zip(
        source_records,
        inference_results,
    ):
        if (
            source_record.get("caseid")
            != result.get("caseid")
            or source_record.get("part")
            != result.get("part")
        ):
            raise RuntimeError(
                "Cannot build CRC input: inference result order/key "
                "does not match the original source record."
            )

        # Keep the COMPLETE original log for CRC.
        raw_content = source_record["content"]

        # The current CRC extract_log() searches specifically for:
        #     content: ... categories:
        user_content = (
            "Input the contents of the operation fault log is:\n"
            f"content: {raw_content}\n"
            f"categories: {taxonomy_text}"
        )

        reasoning_v1 = result.get(
            "reasoning_v1"
        )

        if isinstance(reasoning_v1, dict):
            reasoning_text = json.dumps(
                reasoning_v1,
                ensure_ascii=False,
                indent=2,
            )
        else:
            reasoning_text = (
                result.get("reasoning_v1_raw")
                or ""
            ).strip()

        matched_label = result.get(
            "matched_label"
        )

        if matched_label in label2name.values():
            if reasoning_text:
                assistant_content = (
                    f"{reasoning_text}\n\n"
                    f"Label: {matched_label}"
                )
            else:
                assistant_content = (
                    f"Label: {matched_label}"
                )
        else:
            # The current CRC script tolerates a missing upstream Label:
            # it simply sets cloud_label=None and continues.
            assistant_content = reasoning_text

        crc_records.append(
            {
                "caseid": source_record.get(
                    "caseid"
                ),
                "part": source_record.get(
                    "part"
                ),
                "label": source_record.get(
                    "label"
                ),
                "user_content": user_content,
                "assistant_content": (
                    assistant_content
                ),
            }
        )

    return crc_records


# ============================================================
# Output
# ============================================================
def save_json(
    obj,
    filename: Path,
):
    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with filename.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        f"Saved: {filename}"
    )


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Dataset2 structured Reasoning V1 and "
            "reasoning-driven fault diagnosis."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input JSON file; every record "
            "in the top-level array is processed"
        ),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model name accepted by the "
            "OpenAI-compatible endpoint"
        ),
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=(
            "OpenAI-compatible API base URL"
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            "Sampling temperature. "
            f"Default: {DEFAULT_TEMPERATURE}"
        ),
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=(
            "Top-p sampling value. "
            f"Default: {DEFAULT_TOP_P}"
        ),
    )

    parser.add_argument(
        "--max-retry",
        type=int,
        default=DEFAULT_MAX_RETRY,
        help=(
            "Maximum retries for malformed "
            "Reasoning V1 or diagnosis JSON. "
            f"Default: {DEFAULT_MAX_RETRY}"
        ),
    )

    parser.add_argument(
        "--max-log-len",
        type=int,
        default=DEFAULT_MAX_LOG_LEN,
        help=(
            "Maximum input log length in "
            "characters. "
            f"Default: {DEFAULT_MAX_LOG_LEN}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="output-unified-reasoning-v1",
        help=(
            "Directory for logs and JSON outputs"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    label2name = LABEL2NAME

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = time.strftime(
        "%Y%m%d%H%M%S"
    )
    model_tag = safe_model_name(
        args.model
    )

    run_tag = (
        f"dataset2_"
        f"{model_tag}_"
        f"{timestamp}"
    )

    logger.add(
        output_dir
        / f"{run_tag}.log"
    )

    logger.info(
        "Dataset: dataset2"
    )
    logger.info(
        "Labels: "
        f"{list(label2name.values())}"
    )
    logger.info(
        f"Model: {args.model}"
    )
    logger.info(
        f"Base URL: {args.base_url}"
    )
    logger.info(
        f"Input: {args.input}"
    )
    logger.info(
        "Decoding: "
        f"temperature={args.temperature}, "
        f"top_p={args.top_p}, "
        f"max_retry={args.max_retry}, "
        f"max_log_len={args.max_log_len}"
    )

    client = OpenAI(
        api_key=get_api_key(),
        base_url=args.base_url,
    )

    data = load_data(
        args.input
    )

    if not data:
        raise RuntimeError(
            "Input JSON contains no "
            "records to process."
        )

    analyze_label_distribution(
        data,
        label2name,
    )

    wall_start = (
        time.perf_counter()
    )

    results = process_records(
        client=client,
        data=data,
        label2name=label2name,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_retry=args.max_retry,
        max_log_len=args.max_log_len,
    )

    wall_total = (
        time.perf_counter()
        - wall_start
    )

    performance = (
        analyze_performance(
            results
        )
    )

    accuracy = (
        analyze_correctness(
            results,
            label2name,
        )
    )

    performance["dataset"] = "dataset2"
    performance["label2name"] = (
        label2name
    )
    performance["num_classes"] = len(
        label2name
    )
    performance["model"] = (
        args.model
    )
    performance["base_url"] = (
        args.base_url
    )
    performance["input_file"] = (
        args.input
    )
    performance[
        "input_records_loaded"
    ] = len(data)
    performance[
        "all_input_records_processed"
    ] = (
        len(results) == len(data)
    )
    performance[
        "caseid_filtering"
    ] = False
    performance["temperature"] = (
        args.temperature
    )
    performance["top_p"] = (
        args.top_p
    )
    performance["max_retry"] = (
        args.max_retry
    )
    performance["max_log_len"] = (
        args.max_log_len
    )
    performance[
        "script_wall_clock_total_s"
    ] = wall_total
    performance[
        "script_wall_clock_mean_s_per_case"
    ] = (
        wall_total / len(data)
    )
    performance["accuracy"] = (
        accuracy
    )

    performance["note"] = (
        "Dataset2 structured Reasoning V1 with "
        "reasoning-only final classification, "
        "exact JSON diagnosis parsing, retry policy, "
        "API instrumentation, P95 latency statistics, "
        "and CRC-ready output."
    )

    reasoning_output = (
        output_dir
        / (
            f"{run_tag}_"
            "reasoning_v1.json"
        )
    )

    summary_output = (
        output_dir
        / (
            f"{run_tag}_"
            "summary.json"
        )
    )

    save_json(
        results,
        reasoning_output,
    )
    save_json(
        performance,
        summary_output,
    )

    # Additionally emit an input file that can be passed
    # directly to the current CRC script with --input.
    crc_input_records = (
        build_dataset2_crc_input(
            source_records=data,
            inference_results=results,
            label2name=label2name,
        )
    )

    crc_input_output = (
        output_dir
        / (
            f"{run_tag}_"
            "crc_input.json"
        )
    )

    save_json(
        crc_input_records,
        crc_input_output,
    )

    logger.info(
        "Dataset2 CRC-ready input: "
        f"{crc_input_output}"
    )

    logger.info(
        "Script wall-clock: "
        f"{wall_total:.2f}s; "
        "formal API-service mean: "
        f"{performance['api_service_time_mean_s']:.4f}s/case"
    )

    logger.info(
        "Dataset2 Reasoning V1 "
        "generation complete."
    )


if __name__ == "__main__":
    main()
