#!/usr/bin/env python3
"""Verify source score objects and fixed coverage in a finite native checkpoint."""
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
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.score import rows
from workflows.supplemental.remaining4.native import METHODS, MODELS
from workflows.supplemental.remaining4.score_native import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    folder = within(ROOT, args.checkpoint)
    receipt_path = folder / "coverage_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["schema"] != "kdm_remaining4_native_score_checkpoint_v1" or receipt["passed"] is not True:
        raise ValueError("A successful finite native checkpoint receipt is required")
    for name, sha in receipt["outputs"].items():
        if file_hash(folder / name) != sha:
            raise ValueError("Checkpoint output differs from its immutable receipt")
    merged = [record for _line, record, _sha in rows(folder / "score_rows.jsonl.gz")]
    by_key = {record["key"]: record for record in merged}
    source_keys, sources = set(), []
    for source in receipt["sources"]:
        path = within(ROOT, source["path"])
        source_folder = path.parent
        source_receipt = json.loads((source_folder / "receipt.json").read_text())
        source_verify = json.loads((source_folder / "verification.json").read_text())
        if (file_hash(path) != source["sha256"]
                or file_hash(source_folder / "receipt.json") != source["receipt_sha256"]
                or file_hash(source_folder / "verification.json") != source["verification_sha256"]
                or source_verify["passed"] is not True
                or source_receipt["canonical_scorer_sha256"] != file_hash(Path(frozen.__file__))):
            raise ValueError("Previously accepted finite score source binding differs")
        count = 0
        for _line, record, _sha in rows(path):
            key = record["key"]
            if key in source_keys or key not in by_key or record != by_key[key]:
                raise ValueError("A source score object changed or repeated during checkpoint merge")
            source_keys.add(key)
            count += 1
        if count != source["rows"] or count != source_receipt["rows"]:
            raise ValueError("Finite source score row denominator differs")
        sources.append({"path": source["path"], "sha256": source["sha256"],
                        "rows_checked": count, "all_objects_identical_by_key": True})
    if (len(merged) != receipt["rows"] or len(by_key) != len(merged)
            or source_keys != set(by_key) or len(by_key) != receipt["unique_keys"]):
        raise ValueError("Checkpoint unique keys or exact source union differs")
    conditions = [item for _line, item, _sha in rows(folder / "condition_counts.jsonl")]
    if len(conditions) != 8 or {(item["model"], item["method"]) for item in conditions} != {(m, d) for m in MODELS for d in METHODS}:
        raise ValueError("Checkpoint does not contain exactly the registered eight native cells")
    for condition in conditions:
        selected = [r for r in merged if (r["model"], r["method"]) == (condition["model"], condition["method"])]
        quota = Counter(r["target_class"] for r in selected)
        full = len(selected) == 2424 and len(quota) == 101 and set(quota.values()) == {24}
        if (condition["received_n"] != len(selected) or condition["received_n"] + condition["missing_n"] != 2424
                or quota != condition["received_category_counts"] or condition["complete_input"] != full
                or condition["canonical_pending"] != sum(r["canonical_name_in_primary_score"] is None for r in selected)
                or condition["literal_pending"] != sum(r["literal_extracted_name_score"] is None for r in selected)
                or condition["abstain_pending"] != sum(r["abstain"] is None for r in selected)
                or condition["reference_connected_n"] != 0 or condition["reference_metrics_computed"]):
            raise ValueError("Fixed category quota, actual pending decisions or explicit reference scope differs")
    missing = [record for _line, record, _sha in rows(folder / "missing_keys.jsonl")]
    if len(missing) != receipt["missing_rows"] or len(merged) + len(missing) != 19392:
        raise ValueError("Native full-plan coverage differs")
    result = {"schema": "kdm_remaining4_native_checkpoint_verification_v1", "passed": True,
        "rows": len(merged), "unique_keys": len(by_key), "missing_rows": len(missing), "duplicate_keys": 0,
        "source_score_objects_checked": len(source_keys), "all_source_objects_identical_by_key": True,
        "condition_rows": 8, "complete_input_conditions": sum(item["complete_input"] for item in conditions),
        "canonical_pending": sum(item["canonical_pending"] for item in conditions),
        "literal_pending": sum(item["literal_pending"] for item in conditions),
        "abstain_pending": sum(item["abstain_pending"] for item in conditions),
        "per_complete_condition_quota": "101 classes x 24 unique eval samples", "sources": sources,
        "checkpoint_receipt_sha256": file_hash(receipt_path),
        "score_rows_sha256": file_hash(folder / "score_rows.jsonl.gz"),
        "verifier_sha256": file_hash(Path(__file__)), "raw_reopened": False,
        "reference_metrics_computed": False, "GPU_initialized": False, "new_API_calls": 0}
    save_json(within(ROOT, args.output), result)
    print(json.dumps({key: value for key, value in result.items() if key != "sources"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
