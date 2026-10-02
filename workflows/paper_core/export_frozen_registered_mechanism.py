#!/usr/bin/env python3
"""Project only the registered VCD/M3ID matrices and ref-off from frozen data."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kdm.io import file_hash, within
from workflows.paper_core.assemble_nine_main_food import FIVE, output_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = within(ROOT, args.output)
    output.relative_to(ROOT / "outputs/paper_core_20261002_dev_viz")
    source = ROOT / "outputs/paper_20260929"
    conditions = pd.read_csv(source / "conditions.csv")
    scores = pd.read_parquet(source / "scores.parquet")
    if len(conditions) != 352 or len(scores) != 853248:
        raise ValueError("The frozen five-model source changed")
    selected = conditions[conditions.method.isin(["vcd", "m3id"])
                          & conditions.kind.isin(["main", "reference_instruction_removed"])].copy()
    groups = selected.groupby(["model", "method", "kind"]).size()
    expected = {(model, method, kind): size for model in FIVE for method in ("vcd", "m3id")
                for kind, size in (("main", 16), ("reference_instruction_removed", 4))}
    if groups.to_dict() != expected or not selected.guided.all():
        raise ValueError("The registered five-by-forty mechanism inventory differs")
    if not selected.loc[selected.kind.eq("main"), "reference_guided"].all():
        raise ValueError("A matrix source lacks its original reference guidance")
    if selected.loc[selected.kind.eq("reference_instruction_removed"), "reference_guided"].any():
        raise ValueError("A ref-off source retains reference guidance")
    subset = scores[scores.condition_id.isin(selected.condition_id)].copy()
    if len(subset) != 484800 or subset.duplicated(["condition_id", "sample_id"]).any():
        raise ValueError("The frozen mechanism keys lose or repeat records")
    quota = subset.groupby(["condition_id", "target_class"]).size()
    if len(quota) != 200 * 101 or not quota.eq(24).all():
        raise ValueError("The frozen mechanism conditions lack 101-by-24 quotas")
    fields = ["condition_id", "sample_id", "target_class", "correct_canonical", "correct_literal",
              "abstain", "uniform_reference", "accepted_reference", "source_file_id", "source_line",
              "qa_id", "seed", "detail_id", "binding_id", "score_source_line"]
    compact = subset[fields].copy()
    qa = pd.read_parquet(source / "qa.parquet", columns=["qa_id", "question", "answer"])
    qa = qa[qa.qa_id.isin(compact.qa_id)].copy()
    if compact.qa_id.isna().any() or not set(compact.qa_id).issubset(set(qa.qa_id)):
        raise ValueError("A mechanism source lacks its complete original QA")
    metadata = ("sources.csv", "prompts_and_configs.json", "bindings.json", "runtime_checks.json")
    for name in (*metadata, "runtime_records.csv"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("mechanism_scores.parquet", compact), ("qa_dictionary.parquet", qa)):
        data.to_parquet(output / name, index=False)
        pd.testing.assert_frame_equal(data.reset_index(drop=True), pd.read_parquet(output / name))
    selected.to_csv(output / "conditions.csv", index=False)
    for name in metadata:
        shutil.copyfile(source / name, output / name)
    with (source / "runtime_records.csv").open("rb") as raw, (output / "runtime_records.csv.gz").open("wb") as target:
        with gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as compressed:
            shutil.copyfileobj(raw, compressed)
    with gzip.open(output / "runtime_records.csv.gz", "rb") as compressed:
        if compressed.read() != (source / "runtime_records.csv").read_bytes():
            raise ValueError("The original runtime records changed during lossless compression")
    output_json(output / "receipt.json", {
        "schema": "kdm_registered_frozen_five_mechanism_projection_v1", "passed": True,
        "conditions": 200, "rows": 484800, "scope": "VCD/M3ID 4x4 matrices and four ref-off conditions/model",
        "exploratory_SID_or_guided_DoLa_DeCo_included": False,
        "scientific_values_and_QA_exact_roundtrip": True, "original_source_objects_modified": 0,
        "new_scoring_or_annotation": 0, "GPU_initialized": False,
        "source_scores": str((source / "scores.parquet").relative_to(ROOT)),
        "source_scores_sha256": file_hash(source / "scores.parquet"),
        "source_conditions_sha256": file_hash(source / "conditions.csv"),
        "source_QA_sha256": file_hash(source / "qa.parquet"),
        "source_runtime_records_sha256": file_hash(source / "runtime_records.csv"),
        "runtime_records_transformation": "gzip compression with exact byte roundtrip; all original metadata retained",
        "source_only_original_fields": [name for name in subset if name not in fields],
        "source_parquet_row_recoverable_by_condition_id_and_sample_id": True,
        "runner_sha256": file_hash(Path(__file__)), "actual_command": [sys.executable, *sys.argv],
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"passed": True, "conditions": 200, "rows": len(compact),
                      "bytes": sum(p.stat().st_size for p in output.iterdir() if p.is_file())}))


if __name__ == "__main__":
    main()
