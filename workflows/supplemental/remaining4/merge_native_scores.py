#!/usr/bin/env python3
"""Merge verified sealed native score checkpoints without reopening raw answers."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, stable_hash, within
from kdm.pipeline import task_id
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.score import rows
from workflows.supplemental.remaining4.native import METHODS, MODELS, STAGE, native_tasks
from workflows.supplemental.remaining4.score_native import COND, save_json, save_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = within(ROOT, args.output)
    output.mkdir(parents=True, exist_ok=False)
    samples = [row for _line, row, _sha in rows(ROOT / "data/current/all.jsonl")
               if row["dataset"] == "food101" and row["split"] == "eval"]
    if len(samples) != 2424 or len({row["id"] for row in samples}) != 2424:
        raise ValueError("Registered Food-101 eval input differs")
    source_rows, source_metadata, keys = [], [], set()
    for name in args.score_dir:
        folder = within(ROOT, name)
        receipt = json.loads((folder / "receipt.json").read_text())
        verification = json.loads((folder / "verification.json").read_text())
        path = folder / "score_rows.jsonl.gz"
        actual_sha = file_hash(path)
        if (receipt["schema"] != "kdm_remaining4_native_partial_score_v1"
                or verification["passed"] is not True
                or receipt["score_rows_sha256"] != actual_sha
                or verification["score_rows_sha256"] != actual_sha
                or receipt["canonical_scorer_sha256"] != file_hash(Path(frozen.__file__))):
            raise ValueError("Source native score lacks verified canonical-score provenance")
        received = [record for _line, record, _sha in rows(path)]
        if len(received) != receipt["rows"]:
            raise ValueError("Verified source native-score row count differs")
        for record in received:
            if record["key"] in keys:
                raise ValueError("Verified source native score repeats a task key")
            if record["qa_key"] != frozen.qah(record["question"], record["answer"]):
                raise ValueError("Native exact-QA identity differs")
            keys.add(record["key"])
        source_rows.extend(received)
        source_metadata.append({"path": str(path.relative_to(ROOT)), "sha256": actual_sha,
                                "receipt_path": str((folder / "receipt.json").relative_to(ROOT)),
                                "receipt_sha256": file_hash(folder / "receipt.json"),
                                "verification_sha256": file_hash(folder / "verification.json"),
                                "rows": len(received), "raw_reopened": False})
    inventory_path = ROOT / "configs/kdm/models.json"
    inventory = {record["key"]: record for record in json.loads(inventory_path.read_text())}
    counts, pending, missing = [], {}, []
    for model in MODELS:
        for method in METHODS:
            tasks = {task_id(model, task): task for task in native_tasks(samples, method)}
            selected = [record for record in source_rows if record["model"] == model and record["method"] == method]
            received_keys = {record["key"] for record in selected}
            if len(tasks) != 2424 or len(received_keys) != len(selected) or not received_keys <= set(tasks):
                raise ValueError("Native merged keys differ from the registered unique condition plan")
            for record in selected:
                task = tasks[record["key"]]
                if (record["sample_id"] != task["sample"]["id"] or record["target_class"] != task["sample"]["class"]
                        or record["dataset"] != "food101" or record["split"] != "eval" or record["kind"] != STAGE
                        or record["marker"] != "NONE" or record["reference_marker"] != "NONE"
                        or record["guided"] or record["reference_guided"] or record["replicate"] != 0):
                    raise ValueError("Native merged condition or registered sample identity differs")
                primary_pending = record["canonical_name_in_primary_score"] is None
                behavior_pending = record["abstain"] is None
                if primary_pending or behavior_pending:
                    packet = pending.setdefault(record["qa_key"], {"qa_key": record["qa_key"], "question": record["question"],
                        "answer": record["answer"], "memberships": []})
                    packet["memberships"].append({field: record[field] for field in
                        ("model", "method", "sample_id", "target_class", "key", "source_path", "source_line",
                         "source_identity", "raw_line_sha256", "source_claim", "source_part")})
                    packet["memberships"][-1].update(canonical_pending=primary_pending, abstain_pending=behavior_pending)
            quotas = Counter(record["target_class"] for record in selected)
            if any(count > 24 for count in quotas.values()):
                raise ValueError("Merged class quota exceeds the fixed 101 by 24 input")
            condition = {"model": model, "dataset": "food101", "split": "eval", "method": method,
                         "kind": STAGE, "marker": "NONE", "reference_marker": "NONE", "guided": False,
                         "reference_guided": False, "replicate": 0}
            p = sum(record["canonical_name_in_primary_score"] is None for record in selected)
            a = sum(record["abstain"] is None for record in selected)
            l = sum(record["literal_extracted_name_score"] is None for record in selected)
            full = len(selected) == 2424 and len(quotas) == 101 and set(quotas.values()) == {24}
            counts.append({**condition, "hf_model_id": inventory[model]["hf_model_id"],
                "condition_key": "supplemental_native:" + stable_hash(condition),
                "expected_n": 2424, "received_n": len(selected), "missing_n": 2424 - len(selected),
                "canonical_correct": sum(record["canonical_name_in_primary_score"] == 1 for record in selected),
                "canonical_incorrect": sum(record["canonical_name_in_primary_score"] == 0 for record in selected),
                "canonical_pending": p, "abstain_pending": a, "literal_pending": l,
                "abstain_true": sum(record["abstain"] is True for record in selected),
                "complete_input": full, "complete_primary_and_behavior": full and not p and not a,
                "reference_connected_n": 0, "reference_GT_status": "not_connected_in_score_checkpoint",
                "reference_metrics_computed": False, "received_category_counts": dict(quotas)})
            missing.extend({"key": key, "model": model, "stage": STAGE, "method": method,
                            "sample_id": task["sample"]["id"], "split": "eval", "sample": task["sample"],
                            **{field: task[field] for field in COND if field in task}}
                           for key, task in tasks.items() if key not in received_keys)
    if len(source_rows) != sum(row["received_n"] for row in counts):
        raise ValueError("Merged sources contain a condition outside the registered eight native cells")
    save_rows(output / "score_rows.jsonl.gz", source_rows)
    save_rows(output / "condition_counts.jsonl", counts)
    save_rows(output / "boundary_queue.jsonl", [pending[key] for key in sorted(pending)])
    save_rows(output / "missing_keys.jsonl", missing)
    save_json(output / "coverage_receipt.json", {"schema": "kdm_remaining4_native_score_checkpoint_v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "passed": True, "expected_rows": 8 * 2424,
        "rows": len(source_rows), "unique_keys": len(keys), "missing_rows": len(missing), "condition_rows": len(counts),
        "complete_input_conditions": sum(record["complete_input"] for record in counts),
        "complete_primary_and_behavior_conditions": sum(record["complete_primary_and_behavior"] for record in counts),
        "canonical_pending": sum(record["canonical_pending"] for record in counts),
        "abstain_pending": sum(record["abstain_pending"] for record in counts),
        "literal_pending": sum(record["literal_pending"] for record in counts), "boundary_QA": len(pending),
        "sources": source_metadata, "raw_reopened": False, "GPU_initialized": False,
        "new_API_calls": 0, "new_generations": 0, "scientific_parameters_changed": False,
        "reference_metrics_computed": False, "missing_plan_only": True,
        "merger_sha256": file_hash(Path(__file__)), "model_inventory_sha256": file_hash(inventory_path),
        "outputs": {path.name: file_hash(path) for path in output.iterdir() if path.is_file()}})
    print(json.dumps({"rows": len(source_rows), "missing_rows": len(missing), "boundary_QA": len(pending),
                      "conditions": [{field: row[field] for field in ("model", "method", "received_n", "missing_n",
                                       "canonical_pending", "abstain_pending", "literal_pending")} for row in counts]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
