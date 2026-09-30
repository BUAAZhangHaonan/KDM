#!/usr/bin/env python3
"""Validate the core CPU score/table outputs against frozen full denominators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, within
from workflows.paper_core.score_native import COND, load_references, metrics, rows, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    score = within(ROOT, args.score_dir)
    receipt = json.loads((score / "scoring_receipt.json").read_text())
    native = pd.DataFrame([row for _, row, _ in rows(score / "score_rows.jsonl.gz")])
    tables = pd.read_csv(score / "all_core_conditions_same_eval.csv")
    if (len(native) != receipt["native_rows"] or native.duplicated(["model", "sample_id"]).any()
            or tables.duplicated(list(COND)).any() or not tables.n.eq(2424).all()
            or len(tables) != 352 + 5 + native.model.nunique()):
        raise ValueError("Core native score or full condition table denominator differs")
    refs, gt = load_references()
    checks = {}
    for key, group in native.groupby(list(COND), observed=True):
        condition = dict(zip(COND, key))
        quota = group.groupby("target_class").size()
        if len(group) != 2424 or len(quota) != 101 or not quota.eq(24).all():
            raise ValueError("Core native condition lacks its complete 101 by 24 quotas")
        if not all(row.uniform_reference == gt[(row.model, row.sample_id)] for row in group.itertuples()):
            raise ValueError("Core native uniform reference join differs")
        actual = metrics(group, condition)
        selected = tables
        for field, value in condition.items():
            selected = selected[selected[field].eq(value)]
        if len(selected) != 1:
            raise ValueError("Native exact condition key is absent or duplicated in the comparison table")
        saved = selected.iloc[0]
        for field in ("n", "canonical_correct", "canonical_pending", "literal_pending", "literal_correct", "literal_incorrect", "abstain_pending",
                      "abstentions", "reference_positive", "tp", "fp", "fn", "tn", "abstention_known_n"):
            if int(saved[field]) != actual[field]:
                raise ValueError("Native count field differs from actual resolved decisions: " + field)
        if saved.abstention_known_n + saved.abstain_pending != 2424:
            raise ValueError("Unknown abstention was assigned a resolved confusion-matrix state")
        if not saved.primary_complete and pd.notna(saved.accuracy):
            raise ValueError("Incomplete native correctness was published as full-denominator accuracy")
        if saved.literal_correct + saved.literal_incorrect + saved.literal_pending != 2424:
            raise ValueError("Literal count fields do not cover the registered full denominator")
        if not saved.literal_complete and pd.notna(saved.literal_accuracy):
            raise ValueError("Incomplete literal sensitivity was published as complete accuracy")
        if not saved.behavior_complete and any(pd.notna(saved[field]) for field in ("precision", "recall", "abstention_f1")):
            raise ValueError("Incomplete behavior was published as a complete abstention metric")
    frozen = tables[tables.source_namespace.eq("frozen_paper_20260929")].set_index("source_condition_id").sort_index()
    original = pd.read_csv(ROOT / "outputs/paper_core_20260930/run_20260930_core_p0/frozen/all_conditions_352.csv").set_index("condition_id").sort_index()
    if len(frozen) != 352 or frozen.index.tolist() != original.index.tolist():
        raise ValueError("Frozen condition identities changed in the unified table")
    count_fields = {"n": "n", "canonical_correct": "correct", "literal_correct": "literal_correct", "abstentions": "abstentions",
                    "reference_positive": "reference_positive", "tp": "tp", "fp": "fp", "fn": "fn"}
    for output_field, source_field in count_fields.items():
        mismatch = int(np.count_nonzero(frozen[output_field].to_numpy() != original[source_field].to_numpy()))
        checks[output_field + "_frozen_mismatches"] = mismatch
    for field in ("accuracy", "precision", "recall", "abstention_f1"):
        a, b = frozen[field].to_numpy(dtype=float), original[field].to_numpy(dtype=float) / 100
        mismatch = int(np.count_nonzero(~np.isclose(a, b, atol=1e-12, rtol=1e-12, equal_nan=True)))
        checks[field + "_frozen_mismatches"] = mismatch
    if any(checks.values()):
        raise ValueError("Frozen results differ from the independently recomputed P0 reference table")
    boundary = [row for _, row, _ in rows(score / "boundary_queue.jsonl")]
    expected = set(native[native.canonical_name_in_primary_score.isna() | native.abstain.isna()].qa_key)
    if {row["qa_key"] for row in boundary} != expected or len(boundary) != len(expected):
        raise ValueError("Native boundary queue differs from actual unresolved exact QA")
    result = {"schema": "kdm_core_native_cpu_output_verification_v1", "passed": True,
              "native_rows": len(native), "native_conditions": native.model.nunique(), "condition_table_rows": len(tables),
              "frozen_conditions_verified": 352, "frozen_counts_and_metrics_mismatches": checks,
              "reference_eval_rows": len(refs), "reference_join_missing": 0, "native_class_quota": "101x24",
              "boundary_unique_QA": len(boundary), "unknowns_assigned_as_resolved": 0,
              "score_rows_sha256": file_hash(score / "score_rows.jsonl.gz"),
              "table_sha256": file_hash(score / "all_core_conditions_same_eval.csv"),
              "verifier_sha256": file_hash(Path(__file__)), "GPU_initialized": False, "new_API_calls": 0}
    save_json(within(ROOT, args.output), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
