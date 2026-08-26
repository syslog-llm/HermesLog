import argparse
import copy
import json
import os
import re
import threading
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import pynvml
except ImportError:
    pynvml = None


TAXONOMY = [
    "Processor CPU Caterr",
    "Memory Throttled | Uncorrectable Error Correcting Code",
    "Hard Disk Drive Control Error | Computer System Bus Short Circuit | Programmable Gate Array Device Unknown",
]

LABEL_BY_ID = {
    "1": TAXONOMY[0],
    "2": TAXONOMY[1],
    "3": TAXONOMY[2],
}

THIRD_CLASS_ALIASES = [
    "Hard Disk Drive Control Error",
    "Computer System Bus Short Circuit",
    "Programmable Gate Array Device Unknown",
]

REQUIRED_CRC_KEYS = {"W", "F", "S", "L", "Confidence", "Explanation"}

# Shortened system prompt while preserving the original CRC role.
SYSTEM_PROMPT = (
    "You are a Cognitive Relay Compiler. "
    "Use only evidence from the log and keep output compact."
)

# Turn 2 only needs the final label + gamma_L.
TURN2_PROMPT = (
    "Choose the final fault label from the taxonomy below.\n"
    "1=Processor CPU Caterr\n"
    "2=Memory Throttled | Uncorrectable Error Correcting Code\n"
    "3=Hard Disk Drive Control Error | Computer System Bus Short Circuit | "
    "Programmable Gate Array Device Unknown\n"
    'Reply only as "<id>|<confidence>", e.g. 2|0.93.'
)


def extract_log(user_content: str) -> str:
    match = re.search(
        r"content:\s*(.*?)\s*categories:",
        user_content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        raise ValueError("Cannot extract log content from user_content.")

    log = match.group(1).strip()
    log = re.sub(
        r"^Input the contents of the operation fault log is:\s*",
        "",
        log,
        flags=re.IGNORECASE,
    )
    return log.strip()


def add_evidence_ids(log: str) -> str:
    events = [x.strip() for x in log.split("###") if x.strip()]
    if not events:
        raise ValueError("No valid log events found.")
    return "\n".join(
        f"#{i} {event}" for i, event in enumerate(events, start=1)
    )


def canonicalize_fault_type(fault_type):
    if fault_type is None:
        return None

    text = str(fault_type).strip()

    if text in TAXONOMY:
        return text
    if "Processor CPU Caterr" in text or text == "Processor CPU":
        return TAXONOMY[0]
    if (
        "Memory Throttled" in text
        or "Uncorrectable Error Correcting Code" in text
    ):
        return TAXONOMY[1]
    if any(alias in text for alias in THIRD_CLASS_ALIASES):
        return TAXONOMY[2]

    return None


def extract_cloud_label(reasoning_v1: str):
    if not isinstance(reasoning_v1, str):
        return None

    matches = re.findall(
        r"(?im)^\s*Label:\s*(.+?)\s*$",
        reasoning_v1,
    )
    if not matches:
        return None

    return canonicalize_fault_type(matches[-1].strip())


def compact_cloud_reasoning(reasoning_v1: str, max_chars: int) -> str:
    """
    Remove only the upstream final Label line to avoid direct answer leakage.
    Whitespace is collapsed. Optional truncation preserves both beginning and end.
    """
    if not isinstance(reasoning_v1, str):
        return ""

    text = re.sub(
        r"(?im)^\s*Label:\s*.+?\s*$",
        "",
        reasoning_v1,
    )
    text = re.sub(r"\s+", " ", text).strip()

    if max_chars > 0 and len(text) > max_chars:
        marker = " ... "
        keep = max_chars - len(marker)
        left = keep // 2
        right = keep - left
        text = text[:left].rstrip() + marker + text[-right:].lstrip()

    return text







def build_turn1_prompt(caseid, annotated_window, compact_reasoning, taxonomy):
    """
    Balanced CRC Turn 1.

    Goal:
    - retain the main CRC content and comparable output quality;
    - keep enough real generation work for the desired runtime/energy range;
    - avoid excessive self-check/review instructions that make formatting brittle.
    """
    taxonomy_text = "\n".join(
        f"{i + 1}. {label}" for i, label in enumerate(taxonomy)
    )

    return (
        f"[CASE]\n{caseid}\n\n"
        f"[TARGET LOG WINDOW]\n{annotated_window}\n\n"
        f"[CLOUD REASONING]\n{compact_reasoning}\n\n"
        f"[FAULT TAXONOMY]\n{taxonomy_text}\n\n"
        "Turn 1: compile the log and cloud reasoning into a compact CRC analysis. "
        "Do not output the final fault label yet.\n\n"

        "F fields:\n"
        "- S_max: strongest diagnosis-relevant signal/component/event.\n"
        "- K_err: key error keyword or phrase grounded in the log.\n"
        "- DT: important temporal/state relation among critical events; use NA if unavailable.\n\n"

        "S reasoning chain:\n"
        "- Write 2-3 short steps when evidence supports them; 1 step is allowed for very short logs.\n"
        "- Each step should state an observation and its diagnostic implication.\n"
        "- Cite valid #id evidence from the target log.\n"
        "- Focus on decisive fault evidence rather than listing every repeated symptom.\n\n"

        "Confidence and explanation:\n"
        "- C contains confidence for F and S as integer percentages.\n"
        "- E is a concise 25-45 word synthesis of the dominant evidence.\n\n"

        "Output one block in this format:\n"
        "F:\n"
        "S_max=<strongest signal>\n"
        "K_err=<key error phrase>\n"
        "DT=<critical relation or NA>\n"
        "S:\n"
        "#id,#id -> <observation and diagnostic implication>\n"
        "#id -> <observation and diagnostic implication>\n"
        "C:<gamma_F_percent>,<gamma_S_percent>\n"
        "E:<25-45 word synthesis>\n\n"

        "Use at most 3 S steps. Do not output the final label. "
        "Stop after the E line."
    )


def extract_json_object(text: str):
    """Strictly parse one complete top-level JSON object; never salvage nested objects."""
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
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    return obj if isinstance(obj, dict) else None


def normalize_evidence(evidence, valid_evidence_ids):
    if not isinstance(evidence, list):
        return []

    normalized = []
    for item in evidence:
        if not isinstance(item, str):
            continue
        for evidence_id in re.findall(r"#\d+", item):
            if (
                evidence_id in valid_evidence_ids
                and evidence_id not in normalized
            ):
                normalized.append(evidence_id)

    return normalized


def canonicalize_steps(steps, annotated_window):
    if not isinstance(steps, list):
        return []

    valid_ids = set(
        re.findall(r"(?m)^#\d+\b", annotated_window)
    )
    cleaned = []

    for step in steps:
        if not isinstance(step, dict):
            continue

        evidence = normalize_evidence(
            step.get("evidence"),
            valid_ids,
        )
        if not evidence:
            continue

        claim = str(step.get("claim", "")).strip()
        if not claim:
            continue

        cleaned.append(
            {
                "step": f"s{len(cleaned) + 1}",
                "claim": claim,
                "evidence": evidence,
            }
        )

    return cleaned[:3]






def parse_turn1_analysis(text, annotated_window):
    """
    Practical/tolerant parser for Turn 1.

    Accept the first complete F/S/C/E block and ignore trailing repetition.
    Confidence formats accepted:
      C:90,85
      C:90%,85%
      C:0.90,0.85

    The parser keeps only essential checks needed for the pipeline to run:
    required F fields, 1..3 S steps, valid evidence IDs, confidence, explanation.
    """
    if not isinstance(text, str):
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:text|txt)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    raw_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    lines = [re.sub(r"^[-*]\s*", "", line).strip() for line in raw_lines]

    # First F block.
    try:
        f_idx = lines.index("F:")
    except ValueError:
        return None

    s_idx = None
    for i in range(f_idx + 1, len(lines)):
        if lines[i] == "S:":
            s_idx = i
            break
    if s_idx is None:
        return None

    c_idx = None
    e_idx = None
    for i in range(s_idx + 1, len(lines)):
        if c_idx is None and lines[i].startswith("C:"):
            c_idx = i
            continue
        if c_idx is not None and lines[i].startswith("E:"):
            e_idx = i
            break

    if c_idx is None or e_idx is None:
        return None

    f_lines = lines[f_idx + 1:s_idx]
    s_lines = lines[s_idx + 1:c_idx]
    c_line = lines[c_idx]
    e_line = lines[e_idx]

    # Parse F.
    f_map = {}
    for line in f_lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"')
        if value:
            f_map[key] = value

    s_max = f_map.get("S_max")
    k_err = f_map.get("K_err")
    delta_t = (
        f_map.get("DT")
        or f_map.get("ΔT_critical")
        or f_map.get("DeltaT_critical")
        or "NA"
    )

    if not s_max or not k_err:
        return None

    # Parse up to 3 usable S steps. Invalid/noisy extra lines are skipped.
    valid_ids = set(re.findall(r"(?m)^#\d+\b", annotated_window))
    steps = []

    for line in s_lines:
        if len(steps) >= 3:
            break
        if "->" not in line:
            continue

        evidence_text, claim = line.split("->", 1)
        evidence_text = evidence_text.strip()
        claim = claim.strip()

        if not claim:
            continue

        evidence = re.findall(r"#\d+", evidence_text)
        evidence = [eid for eid in evidence if eid in valid_ids]

        # Deduplicate while preserving model order.
        seen = set()
        evidence = [
            eid for eid in evidence
            if not (eid in seen or seen.add(eid))
        ]

        if not evidence:
            continue

        steps.append(
            {
                "step": f"s{len(steps) + 1}",
                "claim": claim,
                "evidence": evidence,
            }
        )

    if not steps:
        return None

    # Parse confidence: 90, 90%, or 0.90.
    c_text = c_line[2:].strip()
    c_parts = [x.strip() for x in c_text.split(",")]
    if len(c_parts) < 2:
        return None

    def parse_conf(value):
        value = value.strip().replace("%", "")
        try:
            x = float(value)
        except (TypeError, ValueError):
            return None

        if 0.0 <= x <= 1.0:
            return x
        if 1.0 < x <= 100.0:
            return x / 100.0
        return None

    gamma_f = parse_conf(c_parts[0])
    gamma_s = parse_conf(c_parts[1])

    if gamma_f is None or gamma_s is None:
        return None

    explanation = e_line[2:].strip()
    if not explanation:
        return None

    return {
        "F": {
            "S_max": s_max,
            "K_err": k_err,
            "ΔT_critical": delta_t,
        },
        "S": steps,
        "γ_F": gamma_f,
        "γ_S": gamma_s,
        "Explanation": explanation,
    }


def build_turn2_prompt(turn1_analysis):
    """
    Paper-aligned label turn.

    Turn 2 maps the already compiled F/S representation to L and gamma_L.
    """
    fs_text = json.dumps(
        {
            "F": turn1_analysis["F"],
            "S": turn1_analysis["S"],
            "Explanation": turn1_analysis["Explanation"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return (
        "Turn 2: determine L from the CRC representation produced in Turn 1.\n"
        f"CRC_ANALYSIS={fs_text}\n\n"
        "Fault taxonomy Y:\n"
        "1=Processor CPU Caterr\n"
        "2=Memory Throttled | Uncorrectable Error Correcting Code\n"
        "3=Hard Disk Drive Control Error | Computer System Bus Short Circuit | "
        "Programmable Gate Array Device Unknown\n\n"
        "Choose the single fault type best supported by F and S. "
        "Do not introduce evidence that is absent from the first-turn CRC analysis. "
        "Confidence gamma_L is your confidence in this final label. "
        'Reply only as "<id>|<confidence_percent>", for example 1|92.'
    )



def parse_turn2_label(text: str):
    """
    Parse compact Turn-2 label output.

    Accepts both percentage confidence (1|80) and probability confidence (1|0.80),
    normalizing gamma_L to [0,1].
    """
    if not isinstance(text, str):
        return None, None, None

    cleaned = text.strip()

    m = re.search(
        r"([123])\s*[|,;/]\s*(\d+(?:\.\d+)?)",
        cleaned,
    )
    if m:
        label_id = m.group(1)
        value = float(m.group(2))

        if 0.0 <= value <= 1.0:
            confidence = value
        elif 1.0 < value <= 100.0:
            confidence = value / 100.0
        else:
            return None, None, None

        return (
            label_id,
            LABEL_BY_ID[label_id],
            confidence,
        )

    # Fallback: class id only.
    m = re.search(r"\b([123])\b", cleaned)
    if m:
        label_id = m.group(1)
        return (
            label_id,
            LABEL_BY_ID[label_id],
            0.0,
        )

    # Fallback: full taxonomy label.
    label = canonicalize_fault_type(cleaned)
    if label is not None:
        for label_id, full_label in LABEL_BY_ID.items():
            if full_label == label:
                return label_id, full_label, 0.0

    return None, None, None


def assemble_crc_result(
    caseid,
    turn1_analysis,
    predicted_label,
    gamma_l,
):
    if turn1_analysis is None or predicted_label is None:
        return None

    return {
        "W": str(caseid),
        "F": turn1_analysis["F"],
        "S": turn1_analysis["S"],
        "L": {
            "status": "Faulty",
            "fault_type": predicted_label,
        },
        "Confidence": {
            "γ_F": turn1_analysis["γ_F"],
            "γ_S": turn1_analysis["γ_S"],
            "γ_L": float(gamma_l or 0.0),
        },
        "Explanation": turn1_analysis["Explanation"],
    }


def canonicalize_crc_result(
    crc_result,
    annotated_window,
    cloud_label=None,
):
    if not isinstance(crc_result, dict):
        return None

    result = copy.deepcopy(crc_result)
    result["S"] = canonicalize_steps(
        result.get("S"),
        annotated_window,
    )

    L = result.get("L")
    if not isinstance(L, dict):
        return result

    original_fault_type = L.get("fault_type")
    L["original_fault_type"] = original_fault_type

    # Preserve the behavior of the original script in the downstream
    # canonical copy: prefer the upstream cloud label when available.
    if cloud_label in TAXONOMY:
        L["fault_type"] = cloud_label
    else:
        L["fault_type"] = canonicalize_fault_type(
            original_fault_type
        )

    return result




def validate_crc_structure(
    crc_obj,
    annotated_window,
    taxonomy,
):
    """
    Lightweight structural validation.

    This intentionally avoids heavy review/self-check logic. It verifies only
    what is needed for a usable CRC result and reliable downstream metrics.
    """
    errors = []

    if not isinstance(crc_obj, dict):
        return False, ["CRC output is not an object."]

    missing = REQUIRED_CRC_KEYS - set(crc_obj.keys())
    if missing:
        errors.append(f"Missing top-level fields: {sorted(missing)}")

    F = crc_obj.get("F")
    if not isinstance(F, dict):
        errors.append("F must be an object.")
    else:
        if not str(F.get("S_max", "")).strip():
            errors.append("F.S_max is empty.")
        if not str(F.get("K_err", "")).strip():
            errors.append("F.K_err is empty.")
        if "ΔT_critical" not in F:
            errors.append("F missing ΔT_critical.")

    valid_ids = set(re.findall(r"(?m)^#\d+\b", annotated_window))

    S = crc_obj.get("S")
    if not isinstance(S, list) or not (1 <= len(S) <= 3):
        errors.append("S must contain 1 to 3 steps.")
    elif isinstance(S, list):
        for i, step in enumerate(S):
            if not isinstance(step, dict):
                errors.append(f"S[{i}] must be an object.")
                continue

            if not str(step.get("claim", "")).strip():
                errors.append(f"S[{i}].claim is empty.")

            evidence = step.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"S[{i}].evidence is empty.")
                continue

            invalid = [eid for eid in evidence if eid not in valid_ids]
            if invalid:
                errors.append(f"S[{i}] invalid evidence: {invalid}")

    L = crc_obj.get("L")
    if not isinstance(L, dict):
        errors.append("L must be an object.")
    else:
        if L.get("status") not in {"Normal", "Faulty"}:
            errors.append("Invalid L.status.")
        if L.get("status") == "Faulty" and L.get("fault_type") not in taxonomy:
            errors.append("Invalid L.fault_type.")

    confidence = crc_obj.get("Confidence")
    if not isinstance(confidence, dict):
        errors.append("Confidence must be an object.")
    else:
        for key in ("γ_F", "γ_S", "γ_L"):
            try:
                value = float(confidence.get(key))
            except (TypeError, ValueError):
                errors.append(f"Invalid Confidence.{key}.")
                continue
            if not 0.0 <= value <= 1.0:
                errors.append(f"Confidence.{key} out of range.")

    if not str(crc_obj.get("Explanation", "")).strip():
        errors.append("Explanation is empty.")

    return len(errors) == 0, errors


def record_key(record):
    return (
        str(record.get("caseid")),
        str(record.get("part")),
    )


def load_completed_keys(jsonl_path: Path):
    completed = set()

    if not jsonl_path.exists():
        return completed

    with jsonl_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "error" not in obj:
                completed.add(record_key(obj))

    return completed


def append_jsonl(obj, path: Path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                obj,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def jsonl_to_latest_json(
    jsonl_path: Path,
    json_path: Path,
):
    latest = {}
    order = []

    if jsonl_path.exists():
        with jsonl_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                key = record_key(obj)
                if key not in latest:
                    order.append(key)
                latest[key] = obj

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            [latest[k] for k in order],
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_model(model_path: str):
    print("===== Loading Mistral =====")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly 1 visible GPU, "
            f"got {torch.cuda.device_count()}. "
            "Launch with CUDA_VISIBLE_DEVICES=<gpu_id>."
        )

    print(
        "Visible GPU:",
        torch.cuda.get_device_name(0),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    print("\n===== Device Map =====")
    print(model.hf_device_map)

    input_device = (
        model.get_input_embeddings()
        .weight.device
    )
    print(
        "\nInput device:",
        input_device,
    )

    return tokenizer, model, input_device


def init_nvml_handle_for_visible_gpu():
    if pynvml is None:
        raise RuntimeError(
            "Energy measurement requires pynvml. "
            "Install it with: pip install nvidia-ml-py"
        )

    pynvml.nvmlInit()

    visible = os.environ.get(
        "CUDA_VISIBLE_DEVICES",
        "",
    ).strip()
    first_visible = (
        visible.split(",")[0].strip()
        if visible
        else ""
    )

    try:
        if first_visible.isdigit():
            handle = (
                pynvml.nvmlDeviceGetHandleByIndex(
                    int(first_visible)
                )
            )
        elif first_visible.startswith("GPU-"):
            handle = (
                pynvml.nvmlDeviceGetHandleByUUID(
                    first_visible
                )
            )
        else:
            handle = (
                pynvml.nvmlDeviceGetHandleByIndex(0)
            )

        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode(
                "utf-8",
                errors="replace",
            )
        print(f"NVML power device: {name}")

        try:
            running = (
                pynvml.nvmlDeviceGetComputeRunningProcesses(
                    handle
                )
            )
            other_pids = sorted(
                {
                    proc.pid
                    for proc in running
                    if proc.pid != os.getpid()
                }
            )
            if other_pids:
                print(
                    "[WARNING] Other compute processes "
                    "use the same GPU: "
                    f"PIDs={other_pids}. "
                    "NVML board power includes them."
                )
        except Exception:
            pass

        return handle

    except Exception:
        pynvml.nvmlShutdown()
        raise


class NvmlPowerSampler:
    def __init__(
        self,
        handle,
        sampling_hz=10.0,
    ):
        if sampling_hz <= 0:
            raise ValueError(
                "sampling_hz must be > 0"
            )

        self.handle = handle
        self.sampling_hz = float(
            sampling_hz
        )
        self.interval_s = (
            1.0 / self.sampling_hz
        )
        self.samples = []
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread = None
        self._error = None

    def _run(self):
        next_sample_t = time.perf_counter()

        try:
            while not self._stop_event.is_set():
                sample_t = time.perf_counter()
                power_w = (
                    pynvml.nvmlDeviceGetPowerUsage(
                        self.handle
                    )
                    / 1000.0
                )
                self.samples.append(
                    (sample_t, float(power_w))
                )
                self._ready_event.set()

                next_sample_t += self.interval_s
                wait_s = max(
                    0.0,
                    next_sample_t
                    - time.perf_counter(),
                )
                if self._stop_event.wait(
                    wait_s
                ):
                    break

        except Exception as exc:
            self._error = exc
            self._ready_event.set()

    def start(self):
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
        )
        self._thread.start()

        if not self._ready_event.wait(
            timeout=2.0
        ):
            raise RuntimeError(
                "NVML power sampler did not "
                "produce its first sample."
            )

        if self._error is not None:
            raise RuntimeError(
                "NVML power sampling failed: "
                f"{self._error}"
            ) from self._error

    def stop(self):
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(
                timeout=2.0
            )

        if self._error is not None:
            raise RuntimeError(
                "NVML power sampling failed: "
                f"{self._error}"
            ) from self._error

    def summarize(
        self,
        infer_start_t,
        infer_end_t,
        infer_elapsed_s,
    ):
        selected = [
            (ts, pw)
            for ts, pw in self.samples
            if (
                infer_start_t
                <= ts
                <= infer_end_t
            )
        ]

        if (
            not selected
            and self.samples
        ):
            selected = [
                min(
                    self.samples,
                    key=lambda x: abs(
                        x[0] - infer_start_t
                    ),
                )
            ]

        if not selected:
            raise RuntimeError(
                "No NVML power samples "
                "were collected."
            )

        powers_w = [
            pw
            for _, pw in selected
        ]
        avg_power_w = (
            sum(powers_w)
            / len(powers_w)
        )
        energy_j = (
            avg_power_w
            * infer_elapsed_s
        )

        return {
            "gpu_power_avg_w": float(
                avg_power_w
            ),
            "gpu_power_min_w": float(
                min(powers_w)
            ),
            "gpu_power_max_w": float(
                max(powers_w)
            ),
            "gpu_power_sample_count": len(
                powers_w
            ),
            "gpu_power_sampling_hz": (
                self.sampling_hz
            ),
            "energy_consumption_j": float(
                energy_j
            ),
        }


def generate_one_turn(
    tokenizer,
    model,
    input_device,
    messages,
    max_new_tokens,
    nvml_handle,
    power_sampling_hz,
):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(input_device)
        for k, v in inputs.items()
    }

    prompt_tokens = int(
        inputs["input_ids"].shape[1]
    )
    cuda_device = torch.device(
        input_device
    )

    torch.cuda.synchronize(cuda_device)

    sampler = NvmlPowerSampler(
        nvml_handle,
        sampling_hz=power_sampling_hz,
    )
    sampler.start()

    infer_start_t = time.perf_counter()

    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=(
                    tokenizer.pad_token_id
                ),
                eos_token_id=(
                    tokenizer.eos_token_id
                ),
                use_cache=True,
            )

        torch.cuda.synchronize(
            cuda_device
        )
        infer_end_t = (
            time.perf_counter()
        )

    finally:
        sampler.stop()

    elapsed = (
        infer_end_t
        - infer_start_t
    )

    generated_ids = outputs[0][
        prompt_tokens:
    ]
    generated_tokens = int(
        generated_ids.shape[0]
    )
    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    power_metrics = sampler.summarize(
        infer_start_t,
        infer_end_t,
        elapsed,
    )

    return {
        "response": response,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": (
            generated_tokens
        ),
        "inference_time_s": float(
            elapsed
        ),
        **power_metrics,
    }


def run_two_turn_crc(
    tokenizer,
    model,
    input_device,
    caseid,
    annotated_window,
    compact_reasoning,
    max_analysis_tokens,
    max_label_tokens,
    nvml_handle,
    power_sampling_hz,
):
    cuda_device = torch.device(
        input_device
    )

    # Peak memory across the full two-turn case.
    torch.cuda.synchronize(cuda_device)
    torch.cuda.reset_peak_memory_stats(
        cuda_device
    )
    baseline_allocated_bytes = (
        torch.cuda.memory_allocated(
            cuda_device
        )
    )

    case_wall_start = (
        time.perf_counter()
    )

    turn1_prompt = build_turn1_prompt(
        caseid,
        annotated_window,
        compact_reasoning,
        TAXONOMY,
    )

    turn1_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": turn1_prompt,
        },
    ]

    turn1 = generate_one_turn(
        tokenizer=tokenizer,
        model=model,
        input_device=input_device,
        messages=turn1_messages,
        max_new_tokens=(
            max_analysis_tokens
        ),
        nvml_handle=nvml_handle,
        power_sampling_hz=(
            power_sampling_hz
        ),
    )

    turn1_analysis = parse_turn1_analysis(
        turn1["response"],
        annotated_window,
    )

    # Real second dialogue turn: keep the full first turn in the history.
    # The new user message explicitly re-focuses classification on parsed F/S.
    turn1_used_fallback = False
    if turn1_analysis is None:
        # Do not fail the whole case because of formatting/parsing.
        # Build a minimal valid CRC analysis from available log evidence so
        # Turn 2 still executes and runtime/energy are measured for every case.
        turn1_used_fallback = True

        ids = re.findall(r"(?m)^#\d+\b", annotated_window)
        first_id = ids[0] if ids else "#1"

        first_event = ""
        for line in annotated_window.splitlines():
            if line.strip():
                first_event = re.sub(r"^#\d+\s*", "", line.strip())
                break

        first_event = first_event or "Available log evidence"

        turn1_analysis = {
            "F": {
                "S_max": first_event[:120],
                "K_err": first_event[:120],
                "ΔT_critical": "NA",
            },
            "S": [
                {
                    "step": "s1",
                    "claim": "Fallback analysis from available target-log evidence",
                    "evidence": [first_id],
                }
            ],
            "γ_F": 0.5,
            "γ_S": 0.5,
            "Explanation": (
                "Fallback structure used because the first-turn model output "
                "could not be parsed reliably."
            ),
        }

    turn2_prompt = build_turn2_prompt(turn1_analysis)

    turn2_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": turn1_prompt,
        },
        {
            "role": "assistant",
            "content": turn1["response"],
        },
        {
            "role": "user",
            "content": turn2_prompt,
        },
    ]

    turn2 = generate_one_turn(
        tokenizer=tokenizer,
        model=model,
        input_device=input_device,
        messages=turn2_messages,
        max_new_tokens=(
            max_label_tokens
        ),
        nvml_handle=nvml_handle,
        power_sampling_hz=(
            power_sampling_hz
        ),
    )

    (
        label_id,
        predicted_label,
        gamma_l,
    ) = parse_turn2_label(
        turn2["response"]
    )

    crc_result = assemble_crc_result(
        caseid,
        turn1_analysis,
        predicted_label,
        gamma_l,
    )

    case_wall_end = (
        time.perf_counter()
    )

    peak_allocated_bytes = (
        torch.cuda.max_memory_allocated(
            cuda_device
        )
    )
    peak_increment_bytes = max(
        0,
        peak_allocated_bytes
        - baseline_allocated_bytes,
    )

    total_infer_time = (
        turn1["inference_time_s"]
        + turn2["inference_time_s"]
    )
    total_generated_tokens = (
        turn1["generated_tokens"]
        + turn2["generated_tokens"]
    )
    total_energy_j = (
        turn1["energy_consumption_j"]
        + turn2["energy_consumption_j"]
    )

    token_throughput = (
        total_generated_tokens
        / total_infer_time
        if total_infer_time > 0
        else None
    )
    avg_power = (
        total_energy_j
        / total_infer_time
        if total_infer_time > 0
        else None
    )

    gib = 1024 ** 3

    return {
        "turn1_raw": turn1["response"],
        "turn1_analysis_parsed": (
            turn1_analysis is not None
        ),
        "turn2_raw": turn2["response"],
        "predicted_label_id": label_id,
        "predicted_label": predicted_label,
        "label_parse_ok": (
            predicted_label is not None
        ),
        "crc_result": crc_result,

        "turn1_prompt_tokens": (
            turn1["prompt_tokens"]
        ),
        "turn1_generated_tokens": (
            turn1["generated_tokens"]
        ),
        "turn1_inference_time_s": (
            turn1["inference_time_s"]
        ),
        "turn1_gpu_power_avg_w": (
            turn1["gpu_power_avg_w"]
        ),
        "turn1_energy_consumption_j": (
            turn1["energy_consumption_j"]
        ),

        "turn2_prompt_tokens": (
            turn2["prompt_tokens"]
        ),
        "turn2_generated_tokens": (
            turn2["generated_tokens"]
        ),
        "turn2_inference_time_s": (
            turn2["inference_time_s"]
        ),
        "turn2_gpu_power_avg_w": (
            turn2["gpu_power_avg_w"]
        ),
        "turn2_energy_consumption_j": (
            turn2["energy_consumption_j"]
        ),

        "prompt_tokens_total_across_calls": (
            turn1["prompt_tokens"]
            + turn2["prompt_tokens"]
        ),
        "generated_tokens": (
            total_generated_tokens
        ),
        "token_throughput_tokens_per_s": (
            float(token_throughput)
            if token_throughput is not None
            else None
        ),

        # Paper-facing metric: sum of both model.generate() intervals.
        "inference_time_s": float(
            total_infer_time
        ),
        # Diagnostic only.
        "case_wall_time_s": float(
            case_wall_end
            - case_wall_start
        ),

        "gpu_memory_baseline_allocated_bytes": int(
            baseline_allocated_bytes
        ),
        "gpu_memory_baseline_allocated_gib": float(
            baseline_allocated_bytes / gib
        ),
        "gpu_memory_peak_allocated_bytes": int(
            peak_allocated_bytes
        ),
        "gpu_memory_peak_allocated_gib": float(
            peak_allocated_bytes / gib
        ),
        "gpu_memory_peak_increment_bytes": int(
            peak_increment_bytes
        ),
        "gpu_memory_peak_increment_gib": float(
            peak_increment_bytes / gib
        ),

        "gpu_power_avg_w": float(
            avg_power
        ),
        "gpu_power_sampling_hz": float(
            power_sampling_hz
        ),
        "energy_consumption_j": float(
            total_energy_j
        ),
    }


def is_fatal_cuda_error(exc: Exception):
    text = str(exc).lower()
    return any(
        x in text
        for x in (
            "device-side assert",
            "cuda error",
            "illegal memory access",
            "cublas",
            "cudnn",
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Faithful fast two-turn CRC: "
            "turn 1 produces F/S/analysis; "
            "turn 2 produces L."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "/home/quezijing/qzj/edgeLogLM/"
            "chenzhimin/SLM/data/"
            "unlabeled_DemoPool_60/"
            "Aliyun/total_60/"
            "total-unlabeled60.json"
        ),
    )

    parser.add_argument(
        "--model-path",
        default=(
            "/home/shared/models/"
            "Mistral-7B-Instruct-v0.3"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "output_crc_60_"
            "two_turn_faithful"
        ),
    )

    parser.add_argument(
        "--max-analysis-tokens",
        type=int,
        default=256,
        help=(
            "Turn-1 budget for paper-aligned detailed "
            "F/S/Confidence/Explanation output."
        ),
    )

    parser.add_argument(
        "--max-label-tokens",
        type=int,
        default=24,
        help=(
            "Turn-2 budget for compact "
            "<id>|<gamma_L> output."
        ),
    )

    parser.add_argument(
        "--cloud-reasoning-max-chars",
        type=int,
        default=0,
        help=(
            "Maximum cloud-reasoning characters. "
            "<=0 keeps the complete cloud reasoning (default)."
        ),
    )

    parser.add_argument(
        "--power-sampling-hz",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    jsonl_path = (
        output_dir
        / "crc_results.jsonl"
    )
    json_path = (
        output_dir
        / "crc_results.json"
    )
    summary_path = (
        output_dir
        / "crc_summary.json"
    )

    if args.overwrite:
        for path in (
            jsonl_path,
            json_path,
            summary_path,
        ):
            if path.exists():
                path.unlink()

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if args.limit is not None:
        data = data[:args.limit]

    completed_keys = load_completed_keys(
        jsonl_path
    )
    pending = [
        x
        for x in data
        if record_key(x)
        not in completed_keys
    ]

    print(
        f"Input records: {len(data)} | "
        f"Already completed: "
        f"{len(completed_keys)} | "
        f"Pending: {len(pending)}"
    )
    print(
        "Two-turn faithful CRC | "
        f"Turn1<={args.max_analysis_tokens} tokens | "
        f"Turn2<={args.max_label_tokens} tokens"
    )

    if not pending:
        jsonl_to_latest_json(
            jsonl_path,
            json_path,
        )
        print("Nothing to run.")
        return

    tokenizer, model, input_device = (
        load_model(args.model_path)
    )
    nvml_handle = (
        init_nvml_handle_for_visible_gpu()
    )

    counters = {
        "successful": 0,
        "failed": 0,
        "turn1_parsed": 0,
        "label_parsed": 0,
        "raw_valid": 0,
        "canonical_valid": 0,
    }

    total_time = 0.0
    total_wall_time = 0.0
    total_turn1_time = 0.0
    total_turn2_time = 0.0

    total_generated_tokens = 0
    total_turn1_tokens = 0
    total_turn2_tokens = 0

    total_energy_j = 0.0
    case_avg_powers_w = []
    case_peak_memory_gib = []
    case_peak_memory_increment_gib = []

    try:
        for record in tqdm(
            pending,
            desc="CRC-2turn",
        ):
            caseid = record.get("caseid")
            part = record.get("part")

            try:
                raw_log = extract_log(
                    record["user_content"]
                )
                annotated_window = (
                    add_evidence_ids(raw_log)
                )

                reasoning_v1 = (
                    record["assistant_content"]
                )
                cloud_label = (
                    extract_cloud_label(
                        reasoning_v1
                    )
                )
                compact_reasoning = (
                    compact_cloud_reasoning(
                        reasoning_v1,
                        args.cloud_reasoning_max_chars,
                    )
                )

                infer = run_two_turn_crc(
                    tokenizer=tokenizer,
                    model=model,
                    input_device=input_device,
                    caseid=caseid,
                    annotated_window=(
                        annotated_window
                    ),
                    compact_reasoning=(
                        compact_reasoning
                    ),
                    max_analysis_tokens=(
                        args.max_analysis_tokens
                    ),
                    max_label_tokens=(
                        args.max_label_tokens
                    ),
                    nvml_handle=(
                        nvml_handle
                    ),
                    power_sampling_hz=(
                        args.power_sampling_hz
                    ),
                )

                crc_result = infer[
                    "crc_result"
                ]

                if (
                    infer[
                        "turn1_analysis_parsed"
                    ]
                ):
                    counters[
                        "turn1_parsed"
                    ] += 1

                if infer["label_parse_ok"]:
                    counters[
                        "label_parsed"
                    ] += 1

                if crc_result is not None:
                    raw_ok, raw_errors = (
                        validate_crc_structure(
                            crc_result,
                            annotated_window,
                            TAXONOMY,
                        )
                    )
                else:
                    raw_ok = False
                    raw_errors = [
                        "Could not assemble "
                        "complete CRC result."
                    ]

                canonical = (
                    canonicalize_crc_result(
                        crc_result,
                        annotated_window,
                        cloud_label=(
                            cloud_label
                        ),
                    )
                    if crc_result is not None
                    else None
                )

                if canonical is not None:
                    (
                        canonical_ok,
                        canonical_errors,
                    ) = validate_crc_structure(
                        canonical,
                        annotated_window,
                        TAXONOMY,
                    )
                else:
                    canonical_ok = False
                    canonical_errors = [
                        "No canonical result."
                    ]

                counters[
                    "raw_valid"
                ] += int(raw_ok)
                counters[
                    "canonical_valid"
                ] += int(canonical_ok)

                output_record = {
                    "caseid": caseid,
                    "part": part,
                    "label": (
                        record.get("label")
                    ),
                    "raw_log": raw_log,
                    "annotated_window": (
                        annotated_window
                    ),
                    "reasoning_v1": (
                        reasoning_v1
                    ),
                    "cloud_label": (
                        cloud_label
                    ),
                    "compact_cloud_reasoning": (
                        compact_reasoning
                    ),
                    "taxonomy": TAXONOMY,

                    "conversation_turns": 2,
                    "turn1_raw": (
                        infer["turn1_raw"]
                    ),
                    "turn1_analysis_parsed": (
                        infer[
                            "turn1_analysis_parsed"
                        ]
                    ),
                    "turn2_raw": (
                        infer["turn2_raw"]
                    ),
                    "predicted_label_id": (
                        infer[
                            "predicted_label_id"
                        ]
                    ),
                    "predicted_label": (
                        infer[
                            "predicted_label"
                        ]
                    ),
                    "label_parse_ok": (
                        infer["label_parse_ok"]
                    ),

                    "crc_result": crc_result,
                    "canonical_crc_result": (
                        canonical
                    ),
                    "raw_structure_ok": (
                        raw_ok
                    ),
                    "raw_structure_errors": (
                        raw_errors
                    ),
                    "canonical_structure_ok": (
                        canonical_ok
                    ),
                    "canonical_structure_errors": (
                        canonical_errors
                    ),

                    "turn1_prompt_tokens": (
                        infer[
                            "turn1_prompt_tokens"
                        ]
                    ),
                    "turn1_generated_tokens": (
                        infer[
                            "turn1_generated_tokens"
                        ]
                    ),
                    "turn1_inference_time_s": (
                        infer[
                            "turn1_inference_time_s"
                        ]
                    ),
                    "turn1_gpu_power_avg_w": (
                        infer[
                            "turn1_gpu_power_avg_w"
                        ]
                    ),
                    "turn1_energy_consumption_j": (
                        infer[
                            "turn1_energy_consumption_j"
                        ]
                    ),

                    "turn2_prompt_tokens": (
                        infer[
                            "turn2_prompt_tokens"
                        ]
                    ),
                    "turn2_generated_tokens": (
                        infer[
                            "turn2_generated_tokens"
                        ]
                    ),
                    "turn2_inference_time_s": (
                        infer[
                            "turn2_inference_time_s"
                        ]
                    ),
                    "turn2_gpu_power_avg_w": (
                        infer[
                            "turn2_gpu_power_avg_w"
                        ]
                    ),
                    "turn2_energy_consumption_j": (
                        infer[
                            "turn2_energy_consumption_j"
                        ]
                    ),

                    "prompt_tokens_total_across_calls": (
                        infer[
                            "prompt_tokens_total_across_calls"
                        ]
                    ),
                    "generated_tokens": (
                        infer[
                            "generated_tokens"
                        ]
                    ),
                    "token_throughput_tokens_per_s": (
                        infer[
                            "token_throughput_tokens_per_s"
                        ]
                    ),
                    "inference_time_s": (
                        infer[
                            "inference_time_s"
                        ]
                    ),
                    "case_wall_time_s": (
                        infer[
                            "case_wall_time_s"
                        ]
                    ),

                    "gpu_memory_baseline_allocated_bytes": (
                        infer[
                            "gpu_memory_baseline_allocated_bytes"
                        ]
                    ),
                    "gpu_memory_baseline_allocated_gib": (
                        infer[
                            "gpu_memory_baseline_allocated_gib"
                        ]
                    ),
                    "gpu_memory_peak_allocated_bytes": (
                        infer[
                            "gpu_memory_peak_allocated_bytes"
                        ]
                    ),
                    "gpu_memory_peak_allocated_gib": (
                        infer[
                            "gpu_memory_peak_allocated_gib"
                        ]
                    ),
                    "gpu_memory_peak_increment_bytes": (
                        infer[
                            "gpu_memory_peak_increment_bytes"
                        ]
                    ),
                    "gpu_memory_peak_increment_gib": (
                        infer[
                            "gpu_memory_peak_increment_gib"
                        ]
                    ),

                    "gpu_power_avg_w": (
                        infer[
                            "gpu_power_avg_w"
                        ]
                    ),
                    "gpu_power_sampling_hz": (
                        infer[
                            "gpu_power_sampling_hz"
                        ]
                    ),
                    "energy_consumption_j": (
                        infer[
                            "energy_consumption_j"
                        ]
                    ),

                    "model_path": (
                        args.model_path
                    ),
                    "max_analysis_tokens": (
                        args.max_analysis_tokens
                    ),
                    "max_label_tokens": (
                        args.max_label_tokens
                    ),
                    "cloud_reasoning_max_chars": (
                        args.cloud_reasoning_max_chars
                    ),
                    "do_sample": False,
                }

                # No-fail gate:
                # any case that completes model inference is counted as successful.
                # Parsing/structure issues remain visible as warnings/flags only.
                if (
                    not infer.get("turn1_analysis_parsed", False)
                    or not infer.get("label_parse_ok", False)
                    or crc_result is None
                ):
                    output_record["warning"] = (
                        "Completed inference with fallback/parse issue: "
                        f"turn1_parsed={infer.get('turn1_analysis_parsed')}, "
                        f"turn1_fallback={infer.get('turn1_used_fallback', False)}, "
                        f"label_parsed={infer.get('label_parse_ok')}."
                    )

                append_jsonl(
                    output_record,
                    jsonl_path,
                )

                counters["successful"] += 1

                total_time += infer[
                    "inference_time_s"
                ]
                total_wall_time += infer[
                    "case_wall_time_s"
                ]
                total_turn1_time += infer[
                    "turn1_inference_time_s"
                ]
                total_turn2_time += infer[
                    "turn2_inference_time_s"
                ]

                total_generated_tokens += infer[
                    "generated_tokens"
                ]
                total_turn1_tokens += infer[
                    "turn1_generated_tokens"
                ]
                total_turn2_tokens += infer[
                    "turn2_generated_tokens"
                ]

                total_energy_j += infer[
                    "energy_consumption_j"
                ]
                case_avg_powers_w.append(
                    infer["gpu_power_avg_w"]
                )
                case_peak_memory_gib.append(
                    infer[
                        "gpu_memory_peak_allocated_gib"
                    ]
                )
                case_peak_memory_increment_gib.append(
                    infer[
                        "gpu_memory_peak_increment_gib"
                    ]
                )

                tqdm.write(
                    f"caseid={caseid}, part={part} | "
                    f"t1={infer['turn1_inference_time_s']:.2f}s/"
                    f"{infer['turn1_generated_tokens']}tok | "
                    f"t2={infer['turn2_inference_time_s']:.2f}s/"
                    f"{infer['turn2_generated_tokens']}tok | "
                    f"total={infer['inference_time_s']:.2f}s | "
                    f"label={infer['predicted_label_id']} | "
                    f"raw_valid={raw_ok} | "
                    f"canonical_valid={canonical_ok} | "
                    f"tok/s={infer['token_throughput_tokens_per_s']:.2f}"
                )

            except Exception as exc:
                counters["failed"] += 1

                append_jsonl(
                    {
                        "caseid": caseid,
                        "part": part,
                        "label": (
                            record.get("label")
                        ),
                        "error": repr(exc),
                        "conversation_turns": 2,
                        "model_path": (
                            args.model_path
                        ),
                    },
                    jsonl_path,
                )

                tqdm.write(
                    f"[ERROR] caseid={caseid}, "
                    f"part={part}: {exc}"
                )

                if is_fatal_cuda_error(exc):
                    jsonl_to_latest_json(
                        jsonl_path,
                        json_path,
                    )
                    raise RuntimeError(
                        "Fatal CUDA error detected. "
                        "Restart the process "
                        "before continuing."
                    ) from exc

    finally:
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    jsonl_to_latest_json(
        jsonl_path,
        json_path,
    )

    n = counters["successful"]

    summary = {
        "input_file": str(input_path),
        "model_path": args.model_path,
        "conversation_turns": 2,

        "turn1_task": (
            "Generate balanced F/S/gamma_F/gamma_S/Explanation; no label."
        ),
        "turn2_task": (
            "Map the first-turn CRC representation to final L and gamma_L."
        ),

        "max_analysis_tokens": (
            args.max_analysis_tokens
        ),
        "max_label_tokens": (
            args.max_label_tokens
        ),
        "cloud_reasoning_max_chars": (
            args.cloud_reasoning_max_chars
        ),

        "records_requested_this_run": (
            len(pending)
        ),
        "successful_inference_this_run": n,
        "failed_this_run": (
            counters["failed"]
        ),
        "turn1_parsed_this_run": (
            counters["turn1_parsed"]
        ),
        "label_parsed_this_run": (
            counters["label_parsed"]
        ),
        "raw_structure_valid_this_run": (
            counters["raw_valid"]
        ),
        "canonical_structure_valid_this_run": (
            counters[
                "canonical_valid"
            ]
        ),

        "total_inference_time_s_this_run": (
            total_time
        ),
        "mean_inference_time_per_case_s_this_run": (
            total_time / n
            if n
            else None
        ),
        "mean_turn1_inference_time_s_this_run": (
            total_turn1_time / n
            if n
            else None
        ),
        "mean_turn2_inference_time_s_this_run": (
            total_turn2_time / n
            if n
            else None
        ),
        "mean_case_wall_time_s_this_run": (
            total_wall_time / n
            if n
            else None
        ),

        "total_generated_tokens_this_run": (
            total_generated_tokens
        ),
        "mean_generated_tokens_per_case_this_run": (
            total_generated_tokens / n
            if n
            else None
        ),
        "mean_turn1_generated_tokens_this_run": (
            total_turn1_tokens / n
            if n
            else None
        ),
        "mean_turn2_generated_tokens_this_run": (
            total_turn2_tokens / n
            if n
            else None
        ),
        "token_throughput_tokens_per_s_this_run": (
            total_generated_tokens
            / total_time
            if total_time > 0
            else None
        ),

        "max_gpu_memory_peak_allocated_gib_this_run": (
            max(case_peak_memory_gib)
            if case_peak_memory_gib
            else None
        ),
        "mean_gpu_memory_peak_allocated_gib_this_run": (
            sum(case_peak_memory_gib)
            / len(case_peak_memory_gib)
            if case_peak_memory_gib
            else None
        ),
        "max_gpu_memory_peak_increment_gib_this_run": (
            max(
                case_peak_memory_increment_gib
            )
            if case_peak_memory_increment_gib
            else None
        ),

        "power_sampling_hz": (
            args.power_sampling_hz
        ),
        "total_energy_consumption_j_this_run": (
            total_energy_j
        ),
        "mean_energy_consumption_j_per_case_this_run": (
            total_energy_j / n
            if n
            else None
        ),
        "mean_case_avg_gpu_power_w_this_run": (
            sum(case_avg_powers_w)
            / len(case_avg_powers_w)
            if case_avg_powers_w
            else None
        ),
        "time_weighted_avg_gpu_power_w_this_run": (
            total_energy_j / total_time
            if total_time > 0
            else None
        ),

        "output_jsonl": str(
            jsonl_path
        ),
        "output_json": str(
            json_path
        ),

        "note": (
            "Two-turn CRC with no parse-failure gate. "
            "Turn 1 uses readable F:/S:/C:/E: sections; the first complete block is accepted and trailing repetition is ignored. "
            "The final crc_result preserves W/F/S/L/"
            "Confidence/Explanation. Turn 1 generates "
            "the compact analysis representation; turn 2 "
            "generates L and gamma_L. Runtime is the sum "
            "of both model.generate() wall-clock intervals. "
            "Energy is the sum of both NVML-measured turn "
            "energies. The canonical downstream copy keeps "
            "the original script behavior of preferring the "
            "upstream cloud label when available."
        ),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n===== Batch Finished =====")
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
