#!/usr/bin/env python3
"""Export requested frozen cases and original images without rescoring."""

from __future__ import annotations

import argparse
import copy
import csv
import filecmp
import gzip
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

PROJECT = Path("/home/g203-4028/projects/knowledge-deficit-mitigation")
OUTPUT = PROJECT / "outputs/paper_20260929"
SLOTS = ("direct", "vcd", "ip_vcd", "copy_control")
RAW_FIELDS = (
    "text",
    "prompt",
    "reference_prompt",
    "neutral_prompt",
    "tokens",
    "config",
    "seed",
    "identity",
)
REFERENCE_FIELDS = (
    "uniform_reference",
    "accepted_reference",
    "gold_rank",
    "independent_correct_attempts",
    "independent_attempts",
)
AVAILABILITY_FIELDS = (
    "case_id",
    "group",
    "model",
    "sample_id",
    "component",
    "status",
    "detail",
    "source_file",
    "source_line",
)


def project_path(value: str, parent: Path) -> Path:
    """Resolve an existing project location and enforce the requested scope."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT / path
    path = path.resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError(f"path is outside {parent}: {value}")
    return path


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests",
        type=Path,
        default=OUTPUT / "work/case_requests.json",
    )
    args = parser.parse_args()
    request_path = project_path(str(args.requests), OUTPUT)
    request_data = json.loads(request_path.read_text(encoding="utf-8"))
    requested = request_data["cases"]
    if not isinstance(requested, list) or not requested:
        raise ValueError("case_requests.json must contain a nonempty cases list")
    case_ids = [case["case_id"] for case in requested]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case_id values must be unique")

    cases = copy.deepcopy(requested)
    counts = Counter()
    availability = []
    formal_scans = []
    independent_scans = []
    case_by_id = {case["case_id"]: case for case in cases}
    originals = {case["case_id"]: case for case in requested}
    formal_jobs = defaultdict(lambda: defaultdict(list))
    verified_raw = {}
    formal_outcomes = {}
    independent_outcomes = {}
    image_outcomes = {}

    def record(case, component, errors, source_file="", source_line=""):
        entry = {
            "case_id": case["case_id"],
            "group": case["group"],
            "model": case["model"],
            "sample_id": case["sample_id"],
            "component": component,
            "status": "available" if not errors else "unavailable_or_incomplete",
            "detail": "; ".join(errors),
            "source_file": source_file,
            "source_line": source_line,
        }
        availability.append(entry)
        return entry

    for case in cases:
        case["responses"] = {}
        for slot in SLOTS:
            request = originals[case["case_id"]].get("responses", {}).get(slot)
            counts["formal_response_slots_requested"] += 1
            if request is None:
                formal_outcomes[(case["case_id"], slot)] = record(
                    case, slot, ["response request is missing"]
                )
                continue
            case["responses"][slot] = copy.deepcopy(request)
            try:
                path = project_path(
                    request["source_file"], PROJECT / "data/responses/formal"
                )
                line = request["source_line"]
                if type(line) is not int or line < 1:
                    raise ValueError("source_line must be a positive 1-based integer")
            except (KeyError, TypeError, ValueError) as error:
                formal_outcomes[(case["case_id"], slot)] = record(
                    case,
                    slot,
                    [str(error)],
                    request.get("source_file", ""),
                    request.get("source_line", ""),
                )
                continue
            formal_jobs[path][line].append((case["case_id"], slot, request))

    for path, line_jobs in sorted(formal_jobs.items()):
        scan = {
            "source_file": relative(path),
            "open_count": 0,
            "requested_unique_lines": len(line_jobs),
            "matched_unique_lines": 0,
            "lines_read": 0,
            "errors": [],
        }
        unresolved = set(line_jobs)
        try:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                scan["open_count"] += 1
                for line_number, raw_line in enumerate(handle, 1):
                    scan["lines_read"] = line_number
                    if line_number not in line_jobs:
                        continue
                    unresolved.remove(line_number)
                    scan["matched_unique_lines"] += 1
                    try:
                        raw = json.loads(raw_line)
                    except (json.JSONDecodeError, TypeError) as error:
                        raw = None
                        parse_error = f"cannot parse requested JSON line: {error}"
                    for case_id, slot, request in line_jobs[line_number]:
                        case = case_by_id[case_id]
                        errors = []
                        if raw is None:
                            errors.append(parse_error)
                        else:
                            sample = raw.get("sample", {})
                            for label, actual, expected in (
                                ("response_key", raw.get("key"), request["response_key"]),
                                ("model", raw.get("model"), case["model"]),
                                ("sample_id", sample.get("id"), case["sample_id"]),
                                ("marker", raw.get("marker"), case["marker"]),
                                (
                                    "exact_question",
                                    sample.get("question"),
                                    case["exact_question"],
                                ),
                                ("target_class", sample.get("class"), case["target_class"]),
                            ):
                                counts[f"formal_{label}_checks"] += 1
                                if actual == expected:
                                    counts[f"formal_{label}_matched"] += 1
                                else:
                                    errors.append(f"{label} mismatch")
                            counts["formal_source_line_checks"] += 1
                            counts["formal_source_line_matched"] += 1
                            missing = [name for name in RAW_FIELDS if name not in raw]
                            if missing:
                                errors.append("missing original fields: " + ", ".join(missing))
                            if not isinstance(raw.get("text"), str):
                                errors.append("original text is not a string")
                            if not isinstance(sample.get("question"), str):
                                errors.append("original sample.question is not a string")
                            counts["formal_response_records_found"] += 1
                            if not errors:
                                exported = case["responses"][slot]
                                for name in RAW_FIELDS:
                                    exported[name] = copy.deepcopy(raw[name])
                                exported["current_source_file"] = relative(path)
                                exported["current_source_line"] = line_number
                                exported["raw_response"] = raw
                                exported["source_checks_passed"] = True
                                verified_raw[(case_id, slot)] = raw
                                counts["formal_response_records_verified"] += 1
                        formal_outcomes[(case_id, slot)] = record(
                            case, slot, errors, relative(path), line_number
                        )
                    if not unresolved:
                        break
        except (OSError, EOFError, UnicodeError) as error:
            scan["errors"].append(str(error))
        for line_number in sorted(unresolved):
            for case_id, slot, request in line_jobs[line_number]:
                errors = list(scan["errors"]) or ["requested source line was not found"]
                formal_outcomes[(case_id, slot)] = record(
                    case_by_id[case_id], slot, errors, relative(path), line_number
                )
        formal_scans.append(scan)

    model_samples = defaultdict(lambda: defaultdict(list))
    for case in cases:
        model_samples[case["model"]][case["sample_id"]].append(case["case_id"])
        raw_records = [
            verified_raw[(case["case_id"], slot)]
            for slot in SLOTS
            if (case["case_id"], slot) in verified_raw
            and formal_outcomes[(case["case_id"], slot)]["status"] == "available"
        ]
        if raw_records:
            case["exact_question"] = raw_records[0]["sample"]["question"]
            question_response = case["responses"][next(
                slot for slot in SLOTS
                if (case["case_id"], slot) in verified_raw
                and formal_outcomes[(case["case_id"], slot)]["status"] == "available"
            )]
            case["exact_question_source"] = {
                "source_file": question_response["current_source_file"],
                "source_line": question_response["current_source_line"],
                "field": "sample.question",
            }
        else:
            case["exact_question_source"] = {
                "source_file": relative(request_path),
                "field": "cases.exact_question",
                "raw_question_unavailable": True,
            }

    for model, samples in sorted(model_samples.items()):
        path = project_path(
            f"data/responses/independent/{model}.jsonl.gz",
            PROJECT / "data/responses/independent",
        )
        rows_by_sample = defaultdict(list)
        scan = {
            "source_file": relative(path),
            "open_count": 0,
            "requested_unique_samples": len(samples),
            "lines_read": 0,
            "records_selected": 0,
            "errors": [],
        }
        try:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                scan["open_count"] += 1
                for line_number, raw_line in enumerate(handle, 1):
                    scan["lines_read"] = line_number
                    raw = json.loads(raw_line)
                    if raw.get("sample_id") not in samples:
                        continue
                    exported = copy.deepcopy(raw)
                    exported["current_source_file"] = relative(path)
                    exported["current_source_line"] = line_number
                    rows_by_sample[raw["sample_id"]].append(exported)
                    scan["records_selected"] += 1
        except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as error:
            scan["errors"].append(str(error))
        for sample_id, selected_case_ids in samples.items():
            rows = rows_by_sample[sample_id]
            for case_id in selected_case_ids:
                case = case_by_id[case_id]
                case["independent_responses"] = rows
                errors = list(scan["errors"])
                counts["independent_case_sets_checked"] += 1
                counts["independent_records_exported"] += len(rows)
                counts["independent_expected_ten_checks"] += 1
                if len(rows) != 10:
                    errors.append(f"expected 10 independent responses, found {len(rows)}")
                else:
                    counts["independent_expected_ten_matched"] += 1
                replicates = [row.get("replicate") for row in rows]
                if len(rows) == 10 and set(replicates) == set(range(10)):
                    counts["independent_replicate_sets_matched"] += 1
                else:
                    errors.append(f"replicate sequence differs from 0..9: {replicates}")
                for row in rows:
                    for label, expected in (
                        ("model", case["model"]),
                        ("sample_id", sample_id),
                        ("question", case["exact_question"]),
                        ("target", case["target_class"]),
                        ("split", "eval"),
                    ):
                        counts[f"independent_{label}_checks"] += 1
                        if row.get(label) == expected:
                            counts[f"independent_{label}_matched"] += 1
                        else:
                            errors.append(
                                f"independent {label} mismatch at current line "
                                f"{row['current_source_line']}"
                            )
                    if not isinstance(row.get("answer"), str):
                        errors.append(
                            f"independent answer is not a string at current line "
                            f"{row['current_source_line']}"
                        )
                if not errors:
                    counts["independent_case_sets_verified"] += 1
                independent_outcomes[case_id] = record(
                    case, "independent_responses", errors, relative(path)
                )
        independent_scans.append(scan)

    images_dir = OUTPUT / "cases/images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_names = {}
    for case in cases:
        errors = []
        counts["images_requested"] += 1
        source_value = case.get("image_source")
        image_info = {"original_path": source_value}
        try:
            source = project_path(source_value, PROJECT / "data/images")
            image_info["source_file"] = relative(source)
            for slot in SLOTS:
                raw = verified_raw.get((case["case_id"], slot))
                if raw is None:
                    continue
                actual = project_path(
                    raw["sample"]["image_path"], PROJECT / "data/images"
                )
                counts["image_formal_source_checks"] += 1
                if actual != source:
                    errors.append(f"{slot} sample.image_path differs from requested image")
                else:
                    counts["image_formal_source_matched"] += 1
            destination = images_dir / source.name
            if source.name in image_names and image_names[source.name] != source:
                errors.append("different image paths share the same original filename")
            if not errors:
                with Image.open(source) as image:
                    image_info.update(
                        width=image.width,
                        height=image.height,
                        format=image.format,
                        original_filename=source.name,
                    )
                shutil.copy2(source, destination)
                counts["images_copied"] += 1
                image_names[source.name] = source
                image_info["copied_path"] = relative(destination)
                image_info["original_bytes"] = source.stat().st_size
                image_info["copied_bytes"] = destination.stat().st_size
                image_info["byte_equal"] = filecmp.cmp(
                    source, destination, shallow=False
                )
                counts["image_byte_equality_checks"] += 1
                if image_info["byte_equal"]:
                    counts["image_byte_equal"] += 1
                else:
                    errors.append("copied image bytes differ from original")
                counts["image_dimensions_recorded"] += 1
        except (OSError, KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
        case["image"] = image_info
        image_outcomes[case["case_id"]] = record(
            case, "image", errors, source_value or ""
        )

    for case in cases:
        case_id = case["case_id"]
        for field in REFERENCE_FIELDS:
            counts["frozen_reference_value_checks"] += 1
            if case.get(field) != originals[case_id].get(field):
                raise AssertionError(f"{case_id}: changed frozen field {field}")
            counts["frozen_reference_values_unchanged"] += 1
        entries = [
            formal_outcomes[(case_id, slot)] for slot in SLOTS
        ] + [independent_outcomes[case_id], image_outcomes[case_id]]
        case["availability_issues"] = [
            {"component": entry["component"], "detail": entry["detail"]}
            for entry in entries
            if entry["status"] != "available"
        ]
        case["availability_status"] = (
            "complete" if not case["availability_issues"] else "partial"
        )

    cases_path = OUTPUT / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    # Validate the actual exported strings and complete records after JSON roundtrip.
    with cases_path.open(encoding="utf-8") as handle:
        exported_cases = [json.loads(line) for line in handle]
    for before, after in zip(cases, exported_cases, strict=True):
        counts["case_roundtrip_checks"] += 1
        if before != after:
            raise AssertionError(f"{before['case_id']}: export changed JSON values")
        counts["case_roundtrip_matched"] += 1
        for slot in SLOTS:
            record_value = after["responses"].get(slot, {})
            raw = record_value.get("raw_response")
            if raw is None:
                continue
            for field in RAW_FIELDS:
                if field in raw:
                    counts["original_formal_field_roundtrip_checks"] += 1
                    if record_value[field] != raw[field]:
                        raise AssertionError(
                            f"{before['case_id']}/{slot}: original {field} changed"
                        )
                    counts["original_formal_field_roundtrip_matched"] += 1
            counts["original_formal_text_roundtrip_matched"] += 1
        for original_row, exported_row in zip(
            before["independent_responses"],
            after["independent_responses"],
            strict=True,
        ):
            counts["original_independent_record_roundtrip_checks"] += 1
            if original_row != exported_row:
                raise AssertionError("independent source record changed on export")
            counts["original_independent_record_roundtrip_matched"] += 1

    availability_path = OUTPUT / "case_availability.csv"
    with availability_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AVAILABILITY_FIELDS)
        writer.writeheader()
        writer.writerows(availability)
    complete = sum(case["availability_status"] == "complete" for case in cases)
    checks = {
        "status": "complete" if complete == len(cases) else "partial",
        "request_file": relative(request_path),
        "script": "scripts/paper_20260929/export_cases.py",
        "cases_requested": len(requested),
        "cases_exported": len(cases),
        "unique_samples": len({case["sample_id"] for case in cases}),
        "unique_model_samples": len({
            (case["model"], case["sample_id"]) for case in cases
        }),
        "cases_complete": complete,
        "cases_partial": len(cases) - complete,
        "counts": dict(sorted(counts.items())),
        "formal_source_scans": formal_scans,
        "independent_source_scans": independent_scans,
        "case_results": [
            {
                "case_id": case["case_id"],
                "availability_status": case["availability_status"],
                "issues": case["availability_issues"],
            }
            for case in cases
        ],
        "frozen_reference_fields_preserved": list(REFERENCE_FIELDS),
        "scoring": "existing request values copied; no scoring performed",
        "hashing": "no new hashes computed",
        "image_validation": "original file copied and direct byte equality checked",
        "outputs": [
            relative(cases_path),
            relative(availability_path),
            "outputs/paper_20260929/case_checks.json",
            "outputs/paper_20260929/cases/images/",
        ],
    }
    (OUTPUT / "case_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": checks["status"],
        "cases_exported": len(cases),
        "cases_complete": complete,
        "formal_source_files": len(formal_scans),
        "independent_source_files": len(independent_scans),
        "independent_responses": counts["independent_records_exported"],
        "images_copied": counts["images_copied"],
        "availability_rows": len(availability),
        "case_checks": "outputs/paper_20260929/case_checks.json",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

