#!/usr/bin/env python3
"""Export a source-bound compact projection of a genuinely closed main panel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from kdm.io import file_hash, within
from workflows.main_results.score import qah
from workflows.paper_core.assemble_nine_main_food import output_json
from workflows.paper_core.recover_frozen_qa import load_recovered

COND = ["model", "dataset", "split", "method", "kind", "marker",
        "reference_marker", "guided", "reference_guided", "replicate"]
SCIENCE = ["correct_canonical", "correct_literal", "abstain", "uniform_reference",
           "reference_complete", "original_source_preserved", "accepted_complete_decision"]
TABLES = ["condition_coverage.csv", "metrics_all_complete.csv",
          "best_observed_joint_operating_points.csv", "best_observed_independent_endpoints.csv",
          "dev_selected_operating_points.csv", "paired_main_comparisons.csv", "transitions.csv",
          "sources.json", "accepted_corrections_applied.json", "source_conflicts.json"]


def require(test, reason):
    if not test:
        raise ValueError(reason)


def checked_parquet(frame, path):
    frame.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), restored)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-panel", required=True)
    parser.add_argument("--frozen-qa-recovery", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    panel, output = (within(ROOT, p) for p in (args.main_panel, args.output))
    output.relative_to(ROOT / "outputs/paper_core_20261002_dev_viz")
    receipt_path = panel / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt["passed"] is True and receipt["complete_main_panel"] is True
            and receipt["complete_conditions"] == 123 and receipt["missing_or_pending_rows"] == 0
            and receipt["source_conflict_count"] == 0, "All 123 main conditions must actually be closed")
    path = panel / "main_score_rows.parquet"
    require(file_hash(path) == receipt["outputs"][path.name], "The accepted main source changed")
    frame = pd.read_parquet(path)
    require(len(frame) == 298152 and not frame.duplicated(["condition_id", "sample_id"]).any(),
            "The accepted main panel loses or repeats a condition/sample key")
    require(frame[SCIENCE].notna().all().all(), "Scientific or acceptance fields are unresolved")
    require(frame.correct_canonical.isin([0, 1]).all() and frame.correct_literal.isin([0, 1]).all()
            and not (frame.correct_canonical.eq(1) & frame.abstain).any(),
            "The closed canonical/literal/abstention values differ")
    require(frame.groupby(["model", "sample_id"]).uniform_reference.nunique().eq(1).all(),
            "A model/input reference changes between methods")
    quotas = frame.groupby(["condition_id", "target_class"]).size()
    require(len(quotas) == 123 * 101 and quotas.eq(24).all(), "The Food class quotas differ")

    manifest = json.loads(within(ROOT, receipt["source_manifest"]).read_text())
    frozen_path = within(ROOT, manifest["frozen_scores"])
    require(file_hash(frozen_path) == manifest["frozen_scores_sha256"], "The frozen score source changed")
    frozen = pd.read_parquet(frozen_path, columns=["qa_id", "correct_canonical", "correct_literal",
                                                  "abstain", "uniform_reference"])
    qa_path = frozen_path.with_name("qa.parquet")
    recovered_directory = within(ROOT, args.frozen_qa_recovery)
    recovered, recovery_receipt = load_recovered(args.frozen_qa_recovery)
    old_qa = recovered.set_index("qa_id")
    mask = frame.source_score_path.eq(manifest["frozen_scores"])
    indices = frame.loc[mask, "source_score_line"].astype("int64").to_numpy() - 1
    original = frozen.iloc[indices].copy()
    original.index = frame.index[mask]
    for field in ("correct_canonical", "correct_literal", "abstain", "uniform_reference"):
        require(frame.loc[mask, field].eq(original[field]).all(), "An accepted frozen scientific value changed")
    for field in ("qa_key", "question", "answer"):
        values = original.qa_id.map(old_qa[field])
        require(values.notna().all(), "A frozen QA pointer is not present in the original dictionary")
        frame.loc[mask, field] = values
    require(frame.question.notna().all() and frame.answer.notna().all(),
            "A complete original question/answer remains unavailable")
    computed = [qah(q, a) for q, a in zip(frame.question, frame.answer)]
    require(all(stored is None or pd.isna(stored) or stored == actual
                for stored, actual in zip(frame.qa_key, computed)), "An actual QA key differs from its full text")
    frame["qa_key"] = computed
    qa = frame[["qa_key", "question", "answer"]].drop_duplicates().sort_values("qa_key").reset_index(drop=True)
    require(not qa.qa_key.duplicated().any(), "A QA hash has multiple complete contents")
    qa.insert(0, "qa_id", pd.Series(range(len(qa)), dtype="int32"))
    conditions = frame[["condition_id", *COND]].drop_duplicates().sort_values("condition_id").reset_index(drop=True)
    require(len(conditions) == 123, "An actual condition identity has conflicting fields")
    conditions.insert(0, "compact_condition_id", pd.Series(range(123), dtype="int16"))
    source_fields = ["source_id", "source_score_path", "source_score_sha256"]
    sources = frame[source_fields].drop_duplicates().sort_values(source_fields).reset_index(drop=True)
    sources.insert(0, "compact_source_id", pd.Series(range(len(sources)), dtype="int16"))
    compact = frame.merge(conditions[["condition_id", "compact_condition_id"]], on="condition_id", validate="many_to_one")
    compact = compact.merge(qa[["qa_key", "qa_id"]], on="qa_key", validate="many_to_one")
    compact = compact.merge(sources, on=source_fields, validate="many_to_one")
    scalar_fields = ["compact_condition_id", "sample_id", "target_class", *SCIENCE,
                     "compact_source_id", "source_score_line", "qa_id"]
    compact = compact[scalar_fields]
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("main_scores.parquet", compact), ("condition_dictionary.parquet", conditions),
                       ("qa_dictionary.parquet", qa), ("source_dictionary.parquet", sources)):
        checked_parquet(data, output / name)
    restored = compact.merge(conditions, on="compact_condition_id", validate="many_to_one")
    restored = restored.merge(qa, on="qa_id", validate="many_to_one")
    restored = restored.merge(sources, on="compact_source_id", validate="many_to_one")
    before = frame.set_index(["condition_id", "sample_id"]).sort_index()
    after = restored.set_index(["condition_id", "sample_id"]).sort_index()
    compare = [*SCIENCE, "target_class", "question", "answer", "qa_key", "source_score_path", "source_score_line"]
    pd.testing.assert_frame_equal(before[compare], after[compare], check_dtype=False)
    for name in TABLES:
        source = panel / name
        require(file_hash(source) == receipt["outputs"][name], "An accepted comparison table changed: " + name)
        shutil.copyfile(source, output / name)
    shutil.copyfile(recovered_directory / "receipt.json", output / "frozen_QA_recovery_receipt.json")
    shutil.copyfile(recovered_directory / "recovered_QA_source_bindings.parquet",
                    output / "frozen_QA_raw_source_bindings.parquet")
    shutil.copyfile(receipt_path, output / "source_panel_receipt.json")
    output_json(output / "receipt.json", {
        "schema": "kdm_closed_main_compact_projection_v1", "passed": True,
        "rows": len(compact), "conditions": len(conditions), "unique_QA": len(qa),
        "actual_complete_QA_recovered_from_frozen_dictionary": int(mask.sum()),
        "scientific_fields_and_full_QA_exact_roundtrip": True, "new_scoring_or_annotation": 0,
        "source_parquet": str(path.relative_to(ROOT)), "source_sha256": file_hash(path),
        "source_receipt_sha256": file_hash(receipt_path), "frozen_QA_source_sha256": file_hash(qa_path),
        "actual_full_QA_recovery_source": str(recovered_directory.relative_to(ROOT)),
        "actual_full_QA_recovery_receipt_sha256": file_hash(recovered_directory / "receipt.json"),
        "original_dictionary_QA_recovered_by_exact_raw_pointers": recovery_receipt["previously_missing_full_QA"],
        "source_line_semantics": "one-based original JSONL line or original parquet row; source file SHA retained",
        "source_only_fields": [c for c in frame if c not in {*scalar_fields, *COND, *source_fields,
                                                              "condition_id", "qa_key", "question", "answer"}],
        "full_original_metadata_available_at_source_parquet": True, "GPU_initialized": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "actual_command": [sys.executable, *sys.argv],
        "runner_sha256": file_hash(Path(__file__)),
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"passed": True, "rows": len(compact), "conditions": 123,
                      "compact_bytes": sum(p.stat().st_size for p in output.iterdir() if p.is_file())}))


if __name__ == "__main__":
    main()
