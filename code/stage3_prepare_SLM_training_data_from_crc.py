import argparse
import json
import re
from pathlib import Path


DATASET_CONFIGS = {
    "aliyun": {
        "display_name": "Aliyun",
        "taxonomy": [
            "Processor CPU Caterr",
            "Memory Throttled | Uncorrectable Error Correcting Code",
            (
                "Hard Disk Drive Control Error | "
                "Computer System Bus Short Circuit | "
                "Programmable Gate Array Device Unknown"
            ),
        ],
        "label_aliases": {
            "Processor CPU Caterr": "Processor CPU Caterr",
            "Processor CPU": "Processor CPU Caterr",
            "Memory Throttled": (
                "Memory Throttled | Uncorrectable Error Correcting Code"
            ),
            "Uncorrectable Error Correcting Code": (
                "Memory Throttled | Uncorrectable Error Correcting Code"
            ),
            "Hard Disk Drive Control Error": (
                "Hard Disk Drive Control Error | "
                "Computer System Bus Short Circuit | "
                "Programmable Gate Array Device Unknown"
            ),
            "Computer System Bus Short Circuit": (
                "Hard Disk Drive Control Error | "
                "Computer System Bus Short Circuit | "
                "Programmable Gate Array Device Unknown"
            ),
            "Programmable Gate Array Device Unknown": (
                "Hard Disk Drive Control Error | "
                "Computer System Bus Short Circuit | "
                "Programmable Gate Array Device Unknown"
            ),
        },
        "source_label_map": {
            1: "Processor CPU Caterr",
            2: "Memory Throttled | Uncorrectable Error Correcting Code",
            3: (
                "Hard Disk Drive Control Error | "
                "Computer System Bus Short Circuit | "
                "Programmable Gate Array Device Unknown"
            ),
            "1": "Processor CPU Caterr",
            "2": "Memory Throttled | Uncorrectable Error Correcting Code",
            "3": (
                "Hard Disk Drive Control Error | "
                "Computer System Bus Short Circuit | "
                "Programmable Gate Array Device Unknown"
            ),
        },
        "default_crc": "output_crc_60_final_v2/crc_results.json",
        "default_source": (
            "/home/quezijing/qzj/edgeLogLM/chenzhimin/SLM/data/"
            "unlabeled_DemoPool_60/Aliyun/total_60/"
            "total-unlabeled60.json"
        ),
        "default_output_dir": "CRC_DemoV2_Aliyun_all",
    },
    "zte": {
        "display_name": "ZTE",
        "taxonomy": [
            "Power Supply Fault",
            "Fan Fault",
            "Optics Module Fault",
            "Port Failure",
            "CRC Error (Cyclic Redundancy Check)",
            "STP Fault (Spanning Tree Protocol)",
            "BFD Down (Bidirectional Forwarding Detection)",
            "LACP Flapping (Link Aggregation Control Protocol)",
            "OSPF Neighbor Flapping (Open Shortest Path First)",
        ],
        "label_aliases": {
            "Power Supply Fault": "Power Supply Fault",
            "Fan Fault": "Fan Fault",
            "Optics Module Fault": "Optics Module Fault",
            "Port Failure": "Port Failure",
            "CRC Error": "CRC Error (Cyclic Redundancy Check)",
            "CRC Error (Cyclic Redundancy Check)": (
                "CRC Error (Cyclic Redundancy Check)"
            ),
            "STP Fault": "STP Fault (Spanning Tree Protocol)",
            "STP Fault (Spanning Tree Protocol)": (
                "STP Fault (Spanning Tree Protocol)"
            ),
            "BFD Down": "BFD Down (Bidirectional Forwarding Detection)",
            "BFD Down (Bidirectional Forwarding Detection)": (
                "BFD Down (Bidirectional Forwarding Detection)"
            ),
            "LACP Flapping": ("LACP Flapping (Link Aggregation Control Protocol)"),
            "LACP Flapping (Link Aggregation Control Protocol)": (
                "LACP Flapping (Link Aggregation Control Protocol)"
            ),
            "OSPF Neighbor Flapping": (
                "OSPF Neighbor Flapping (Open Shortest Path First)"
            ),
            "OSPF Neighbor Flapping (Open Shortest Path First)": (
                "OSPF Neighbor Flapping (Open Shortest Path First)"
            ),
        },
        # ZTE source JSON has no numeric label field.
        "source_label_map": {},
        "default_crc": "output_crc_ZTE/crc_results.json",
        "default_source": (
            "/home/quezijing/qzj/edgeLogLM/chenzhimin/SLM/data/"
            "unlabeled_DemoPool_60/ZTE/total_60/"
            "unlabeled60_reason.json"
        ),
        "default_output_dir": "CRC_DemoV2_ZTE_all",
    },
}


def get_config(dataset_name):
    key = dataset_name.lower()

    if key not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. " f"Choose Aliyun or ZTE."
        )

    return DATASET_CONFIGS[key]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def record_key(record):
    return (str(record.get("caseid")), str(record.get("part")))


def safe_text(value):
    """
    Convert arbitrary values to text and protect the exact separator
    used by the existing SLM code: split("Label: ").
    """
    if value is None:
        return "None"

    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        text = str(value)

    return text.replace("Label: ", "Label=")


def canonicalize_label(label, config):
    if label is None:
        return None

    text = str(label).strip()
    taxonomy = config["taxonomy"]

    if text in taxonomy:
        return text

    aliases = config["label_aliases"]

    if text in aliases:
        return aliases[text]

    for alias in sorted(aliases.keys(), key=len, reverse=True):
        if alias in text:
            return aliases[alias]

    return None


def extract_label_from_reasoning(text, config):
    """
    Extract final cloud Label: from source assistant_content.
    This is especially important for ZTE, which has no numeric label field.
    """
    if not isinstance(text, str):
        return None

    matches = re.findall(r"(?im)^\s*Label:\s*(.+?)\s*$", text)

    if not matches:
        return None

    return canonicalize_label(matches[-1].strip(), config)


def label_from_crc(crc_obj, config):
    if not isinstance(crc_obj, dict):
        return None

    L = crc_obj.get("L")

    if not isinstance(L, dict):
        return None

    return canonicalize_label(L.get("fault_type"), config)


def build_reason_from_crc(crc_obj):
    """
    Convert CRC <F,S,L> back to the old SLM-compatible Reason text.
    """
    if not isinstance(crc_obj, dict):
        return None

    F = crc_obj.get("F")
    S = crc_obj.get("S")

    lines = []

    if isinstance(F, dict):
        lines.append(
            "Feature: "
            f"S_max={safe_text(F.get('S_max'))}; "
            f"K_err={safe_text(F.get('K_err'))}; "
            f"ΔT_critical={safe_text(F.get('ΔT_critical'))}"
        )

    if isinstance(S, list):
        for idx, step in enumerate(S, start=1):
            if not isinstance(step, dict):
                continue

            step_name = safe_text(step.get("step") or f"s{idx}")
            claim = safe_text(step.get("claim"))

            evidence = step.get("evidence")

            if isinstance(evidence, list):
                evidence_text = ", ".join(safe_text(x) for x in evidence)
            else:
                evidence_text = safe_text(evidence)

            lines.append(f"{step_name}: {claim} " f"Evidence: {evidence_text}")

    if not lines:
        return None

    return "\n".join(lines)


def clean_crc_raw(crc_raw):
    """
    Final reason fallback for malformed/unparsed CRC cases.
    """
    if not isinstance(crc_raw, str):
        return None

    text = crc_raw.strip()

    if not text:
        return None

    return text.replace("Label: ", "Label=")


def resolve_training_label(crc_record, source_record, config):
    """
    Full-conversion label fallback order:

    1. canonical_crc_result.L.fault_type
    2. crc_result.L.fault_type
    3. crc_record.cloud_label
    4. Label: parsed directly from source assistant_content
    5. Aliyun source numeric label mapping

    ZTE normally resolves at 1/3/4 and does not require a numeric label.
    """
    canonical = crc_record.get("canonical_crc_result")

    raw_result = crc_record.get("crc_result")

    label = label_from_crc(canonical, config)

    if label is not None:
        return (label, "canonical_crc_result")

    label = label_from_crc(raw_result, config)

    if label is not None:
        return (label, "crc_result")

    label = canonicalize_label(crc_record.get("cloud_label"), config)

    if label is not None:
        return (label, "crc_cloud_label")

    label = extract_label_from_reasoning(source_record.get("assistant_content"), config)

    if label is not None:
        return (label, "source_assistant_content")

    source_label_map = config.get("source_label_map", {})

    label = source_label_map.get(source_record.get("label"))

    if label is not None:
        return (label, "source_numeric_label")

    return None, None


def resolve_reason(crc_record):
    """
    Full-conversion reason fallback order:

    1. canonical_crc_result
    2. crc_result
    3. crc_raw

    Therefore structure-invalid or malformed cases are still retained.
    """
    canonical = crc_record.get("canonical_crc_result")

    raw_result = crc_record.get("crc_result")

    reason = build_reason_from_crc(canonical)

    if reason is not None:
        return (reason, "canonical_crc_result")

    reason = build_reason_from_crc(raw_result)

    if reason is not None:
        return (reason, "crc_result")

    reason = clean_crc_raw(crc_record.get("crc_raw"))

    if reason is not None:
        return (reason, "crc_raw")

    return None, None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Unified full CRC -> DemoV2 converter "
            "for Aliyun and ZTE. "
            "All recoverable records are converted."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str.lower,
        choices=["aliyun", "zte"],
        required=True,
        help=("Aliyun or ZTE " "(case-insensitive)."),
    )

    parser.add_argument(
        "--crc",
        default=None,
        help=("CRC results JSON. " "If omitted, uses dataset-specific default."),
    )

    parser.add_argument(
        "--source",
        default=None,
        help=("Original source JSON. " "If omitted, uses dataset-specific default."),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=("Output directory. " "If omitted, uses dataset-specific default."),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = get_config(args.dataset)

    crc_path = Path(args.crc or config["default_crc"])

    source_path = Path(args.source or config["default_source"])

    output_dir = Path(args.output_dir or config["default_output_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    crc_records = load_json(crc_path)

    source_records = load_json(source_path)

    if not isinstance(crc_records, list):
        raise TypeError("CRC JSON must contain a list.")

    if not isinstance(source_records, list):
        raise TypeError("Source JSON must contain a list.")

    source_map = {record_key(record): record for record in source_records}

    parts = {1: [], 2: [], 3: []}

    audit_records = []

    stats = {
        "dataset": config["display_name"],
        "crc_total": len(crc_records),
        "source_total": len(source_records),
        "written": 0,
        "part1": 0,
        "part2": 0,
        "part3": 0,
        "reason_from_canonical": 0,
        "reason_from_raw_result": 0,
        "reason_from_crc_raw": 0,
        "label_from_canonical": 0,
        "label_from_raw_result": 0,
        "label_from_crc_cloud_label": 0,
        "label_from_source_assistant": 0,
        "label_from_source_numeric": 0,
        "skipped_bad_part": 0,
        "skipped_missing_source": 0,
        "skipped_missing_user_content": 0,
        "skipped_no_reason": 0,
        "skipped_no_label": 0,
    }

    for crc_record in crc_records:
        caseid = crc_record.get("caseid")

        try:
            part = int(crc_record.get("part"))
        except (TypeError, ValueError):
            stats["skipped_bad_part"] += 1
            continue

        if part not in parts:
            stats["skipped_bad_part"] += 1
            continue

        key = (str(caseid), str(part))

        source_record = source_map.get(key)

        if source_record is None:
            stats["skipped_missing_source"] += 1
            continue

        user_content = source_record.get("user_content")

        if not isinstance(user_content, str) or not user_content.strip():
            stats["skipped_missing_user_content"] += 1
            continue

        reason_text, reason_source = resolve_reason(crc_record)

        if reason_text is None:
            stats["skipped_no_reason"] += 1
            continue

        label, label_source = resolve_training_label(
            crc_record=crc_record, source_record=source_record, config=config
        )

        if label is None:
            stats["skipped_no_label"] += 1
            continue

        # Keep the exact delimiter expected by the existing SLM code:
        # assistant_content.split("Label: ")
        assistant_content = f"Reason: {reason_text}\n\n" f"Label: {label}"

        # IMPORTANT:
        # Training part files intentionally contain ONLY scalar columns.
        # This prevents PyArrow errors from mixed nested types such as
        # crc_feature.K_err being list/str/None.
        out_record = {
            "caseid": caseid,
            "part": part,
            # Aliyun keeps its original numeric source label.
            # ZTE has no such field, so this is null.
            "label": source_record.get("label"),
            "user_content": (user_content),
            "assistant_content": (assistant_content),
        }

        parts[part].append(out_record)

        stats["written"] += 1

        stats[f"part{part}"] += 1

        if reason_source == "canonical_crc_result":
            stats["reason_from_canonical"] += 1
        elif reason_source == "crc_result":
            stats["reason_from_raw_result"] += 1
        elif reason_source == "crc_raw":
            stats["reason_from_crc_raw"] += 1

        if label_source == "canonical_crc_result":
            stats["label_from_canonical"] += 1
        elif label_source == "crc_result":
            stats["label_from_raw_result"] += 1
        elif label_source == "crc_cloud_label":
            stats["label_from_crc_cloud_label"] += 1
        elif label_source == "source_assistant_content":
            stats["label_from_source_assistant"] += 1
        elif label_source == "source_numeric_label":
            stats["label_from_source_numeric"] += 1

        # Audit metadata is stored separately so it never enters
        # Hugging Face Dataset.from_pandas().
        audit_records.append(
            {
                "caseid": caseid,
                "part": part,
                "reason_source": (reason_source),
                "label_source": (label_source),
                "training_label": (label),
                "parsed_ok": (crc_record.get("parsed_ok", False)),
                "canonical_structure_ok": (
                    crc_record.get("canonical_structure_ok", False)
                ),
            }
        )

    # Write training files.
    for part in (1, 2, 3):
        path = output_dir / f"part{part}.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(parts[part], f, ensure_ascii=False, indent=2)

    summary_path = output_dir / "conversion_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    audit_path = output_dir / "conversion_audit.json"

    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit_records, f, ensure_ascii=False, indent=2)

    print("===== Unified CRC -> Demo V2 " "Conversion Finished =====")

    print(json.dumps(stats, ensure_ascii=False, indent=2))

    print("\nOutput files:")

    for part in (1, 2, 3):
        print(f"  {output_dir / f'part{part}.json'} " f"({len(parts[part])} samples)")

    print(f"  {summary_path}")

    print(f"  {audit_path}")

    if stats["written"] == len(crc_records):
        print(f"\nAll {stats['written']} CRC records " f"were converted successfully.")
    else:
        print(
            "\nWARNING: Not every CRC record "
            "was converted. Check conversion_summary.json."
        )


if __name__ == "__main__":
    main()
