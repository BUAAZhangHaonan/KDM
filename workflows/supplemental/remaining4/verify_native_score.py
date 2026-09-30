#!/usr/bin/env python3
"""Verify sealed native partial-score keys, source bindings and unresolved counts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, within
from kdm.pipeline import task_id
from workflows.main_results.score import qah
from workflows.supplemental.remaining11.score import rows
from workflows.supplemental.remaining4.native import METHODS, MODELS, STAGE
from workflows.supplemental.remaining4.score_native import COND, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    folder = within(ROOT, args.score_dir)
    receipt = json.loads((folder / "receipt.json").read_text())
    source_proof = json.loads((folder / "received_source_verification.json").read_text())
    records = [record for _, record, _ in rows(folder / "score_rows.jsonl.gz")]
    conditions = [record for _, record, _ in rows(folder / "condition_counts.jsonl")]
    packet = [record for _, record, _ in rows(folder / "boundary_queue.jsonl")]
    if (file_hash(folder / "score_rows.jsonl.gz") != receipt["score_rows_sha256"]
            or len(records) != receipt["rows"] or source_proof["rows"] != len(records)
            or source_proof["passed"] is not True or len(source_proof["parts"]) != receipt["sealed_parts"]
            or sum(part["cpu_validation"]["rows"] for part in source_proof["parts"]) != len(records)
            or len({record["key"] for record in records}) != len(records)
            or len(conditions) != 8 or len({(r["model"], r["method"]) for r in conditions}) != 8):
        raise ValueError("Sealed row, part, condition or source denominator differs")
    samples = {row["id"]: row for _, row, _ in rows(ROOT / "data/current/all.jsonl")
        if row["dataset"] == "food101" and row["split"] == "eval"}
    for record in records:
        if record["model"] not in MODELS or record["method"] not in METHODS or record["stage"] != STAGE:
            raise ValueError("Native score is outside the selected registered stage")
        task = {field: record[field] for field in COND if field not in {"model", "dataset", "split"}}
        task["sample"] = samples[record["sample_id"]]
        if (record["key"] != task_id(record["model"], task)
                or record["qa_key"] != qah(record["question"], record["answer"])
                or record["target_class"] != task["sample"]["class"]
                or record["canonical_name_in_primary_score"] not in {0, 1, None}
                or record["literal_extracted_name_score"] not in {0, 1, None}
                or type(record["abstain"]) not in {bool, type(None)}
                or record["reference_G"] is not None or record["reference_complete"] is not False):
            raise ValueError("Native registered task, complete QA, score domain or pending reference differs")
    for condition in conditions:
        selected = [row for row in records if row["model"] == condition["model"] and row["method"] == condition["method"]]
        if condition["received_n"] != len(selected) or condition["received_n"] + condition["unreceived_n"] != 2424:
            raise ValueError("Partial row count was used as a complete registered denominator")
        expected = {
            "canonical_correct": sum(r["canonical_name_in_primary_score"] == 1 for r in selected),
            "canonical_incorrect": sum(r["canonical_name_in_primary_score"] == 0 for r in selected),
            "canonical_pending": sum(r["canonical_name_in_primary_score"] is None for r in selected),
            "literal_correct": sum(r["literal_extracted_name_score"] == 1 for r in selected),
            "literal_pending": sum(r["literal_extracted_name_score"] is None for r in selected),
            "abstain_true": sum(r["abstain"] is True for r in selected),
            "abstain_false": sum(r["abstain"] is False for r in selected),
            "abstain_pending": sum(r["abstain"] is None for r in selected),
        }
        if any(condition[key] != value for key, value in expected.items()):
            raise ValueError("Native condition decision or unresolved count differs")
        if Counter(r["target_class"] for r in selected) != condition["received_category_counts"]:
            raise ValueError("Partial category coverage differs")
        if condition["full_condition_effect_calculated"] or condition["reference_connected_n"] != 0:
            raise ValueError("This finite partial checkpoint falsely claims complete effects or reference joins")
    unknown = {r["key"] for r in records if r["canonical_name_in_primary_score"] is None or r["abstain"] is None}
    memberships = [member for item in packet for member in item["memberships"]]
    if (len(packet) != receipt["boundary_QA"] or len({item["qa_key"] for item in packet}) != len(packet)
            or {member["key"] for member in memberships} != unknown or len(memberships) != len(unknown)):
        raise ValueError("Minimal boundary packet differs from actual unknown source members")
    by_key = {record["key"]: record for record in records}
    for item in packet:
        if item["qa_key"] != qah(item["question"], item["answer"]):
            raise ValueError("Boundary complete question and answer differ")
        for member in item["memberships"]:
            source = by_key[member["key"]]
            if (source["question"], source["answer"]) != (item["question"], item["answer"]):
                raise ValueError("Boundary source answer membership differs")
            for field in ("source_path", "source_line", "source_identity", "raw_line_sha256"):
                if member[field] != source[field]:
                    raise ValueError("Boundary source-line provenance differs")
    result = {"schema": "kdm_remaining4_native_partial_score_verification_v1", "passed": True,
        "rows": len(records), "parts": len(source_proof["parts"]), "unique_keys": len(records),
        "condition_rows": len(conditions), "boundary_QA": len(packet), "boundary_memberships": len(memberships),
        "canonical_pending": receipt["canonical_pending"], "abstain_pending": receipt["abstain_pending"],
        "unknowns_assigned_as_resolved": 0, "old_raw_scanned": False,
        "source_verification_sha256": file_hash(folder / "received_source_verification.json"),
        "score_rows_sha256": file_hash(folder / "score_rows.jsonl.gz"),
        "boundary_queue_sha256": file_hash(folder / "boundary_queue.jsonl"),
        "GPU_initialized": False, "new_API_calls": 0, "verifier_sha256": file_hash(Path(__file__))}
    save_json(within(ROOT, args.output), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
