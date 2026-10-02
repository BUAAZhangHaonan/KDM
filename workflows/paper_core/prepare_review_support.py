#!/usr/bin/env python3
"""Copy finite completed supporting evidence for the nine-model review package."""
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
from workflows.paper_core.assemble_nine_main_food import output_json

BASE = ROOT / "outputs/paper_core_20261002_dev_viz"
OLD = BASE / "review_package_staging_20261002_1700/package_v6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = within(ROOT, args.output)
    output.relative_to(BASE)
    output.mkdir(parents=True, exist_ok=False)
    sources = []

    def copy(source, destination):
        source = Path(source)
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        digest = file_hash(source)
        shutil.copyfile(source, target)
        if file_hash(target) != digest:
            raise ValueError("A copied supporting artifact differs from its original")
        sources.append({"output": destination, "source": str(source.relative_to(ROOT)),
                        "sha256": digest, "transformation": "exact byte copy"})

    def table(data, destination, source, operation):
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".parquet":
            data.to_parquet(target, index=False)
            pd.testing.assert_frame_equal(data.reset_index(drop=True), pd.read_parquet(target))
        else:
            data.to_csv(target, index=False)
        sources.append({"output": destination, "source": str(source.relative_to(ROOT)),
                        "source_sha256": file_hash(source), "output_sha256": file_hash(target),
                        "rows": len(data), "transformation": operation})

    copy(ROOT / "outputs/paper_20260929/references.parquet", "references/core5_references.parquet")
    copy(ROOT / "outputs/paper_20260929/sources.csv", "references/core5_original_source_dictionary.csv")
    ref_path = OLD / "compact/reference4.parquet"
    refs = pd.read_parquet(ref_path)
    current_ref = ROOT / "outputs/supplemental/remaining4/reference_full_20261001_1600/checkpoints/v2_reviewed_role_reference/reference_G.jsonl"
    current = pd.read_json(current_ref, lines=True)
    fields = ["model", "sample_id", "split", "target_class", "gold_rank", "attempt_count", "correct_count", "reference_G", "reference_complete"]
    keys = ["model", "sample_id"]
    if len(refs) != 19392 or refs.duplicated(keys).any() or not refs.reference_complete.all():
        raise ValueError("The extra four-model Food reference panel is incomplete")
    pd.testing.assert_frame_equal(refs[fields].sort_values(keys).reset_index(drop=True),
                                  current[fields].sort_values(keys).reset_index(drop=True), check_dtype=False)
    attempts = pd.read_parquet(OLD / "compact/reference4_attempts.parquet")
    if len(attempts) != 193920 or attempts.duplicated(keys + ["replicate"]).any():
        raise ValueError("The four-model independent attempts lose or repeat keys")
    counts = attempts.groupby(keys).agg(attempts=("canonical", "size"), correct=("canonical", "sum"))
    linked = refs.set_index(keys).join(counts, validate="one_to_one")
    if not linked.attempts.eq(10).all() or not linked.correct.eq(linked.correct_count).all():
        raise ValueError("The compact attempts do not reproduce the frozen reference counts")
    for name in ("reference4.parquet", "reference4_attempts.parquet", "reference_source_dictionary.parquet", "reference4_receipt.json"):
        copy(OLD / "compact" / name, "references/" + name)
    sources.append({"output": "references/reference4.parquet", "current_reference_authority": str(current_ref.relative_to(ROOT)),
                    "current_reference_sha256": file_hash(current_ref), "all_current_reference_values_equal": True})

    dev = BASE / "dev_selection/all5_preserved_20261003_0010"
    decisions = pd.read_csv(dev / "dev_decisions.csv")
    if len(decisions) != 32320 or not decisions.split.eq("dev").all():
        raise ValueError("The completed registered dev selection inventory changed")
    if decisions.duplicated(["model", "method", "marker", "sample_id"]).any():
        raise ValueError("The completed development panel repeats keys")
    table(decisions, "dev_selection/registered_dev_decisions.parquet", dev / "dev_decisions.csv",
          "exact values from completed dev-only candidate records; historical candidates retained for provenance")
    selected = json.loads((dev / "selected_configs.json").read_text())
    ip_selected = [row for row in selected if row["method"] == "instruction_vcd"]
    if len(ip_selected) != 5 or any(row["selection"] != "dev_selected" or row["n"] != 404 for row in ip_selected):
        raise ValueError("The actual five-model IP-VCD selection is incomplete")
    output_json(output / "dev_selection/IP_VCD_selected_configs5.json", ip_selected)
    copy(dev / "analysis_receipt.json", "dev_selection/original_registered_selection_receipt.json")

    viz = BASE / "merged_viz31_all_five_closed_20261003_0410"
    receipt = json.loads((viz / "receipt.json").read_text())
    if not receipt["passed"] or (receipt["rows"], receipt["conditions"], receipt["pending_QA"],
                                  receipt["quality_pending_rows"], receipt["abstain_pending_rows"], receipt["reference_join_missing"]) != (15872, 31, 0, 0, 0, 0):
        raise ValueError("The completed core five-model VizWiz panel is unresolved")
    for name in ("new_scores.parquet", "sources.csv", "score_source_receipts.csv", "receipt.json"):
        copy(viz / name, "vizwiz/core5_512/" + name)
    curated = BASE / "viz_curated_main_mechanism_20261003_0415"
    for name in ("core5_viz512_main_native_direct_cda_ip.csv", "core5_viz512_guided_mechanism_only.csv", "receipt.json"):
        copy(curated / name, "vizwiz/core5_512/" + ("curated_selection_receipt.json" if name == "receipt.json" else name))

    cda = BASE / "cda_audit_20261002_1700/results"
    trace = pd.read_parquet(cda / "cda_trace_audit.parquet")
    trace = trace[trace.marker.eq("UNKNOWN")].copy()
    if len(trace) != 12120 or trace.groupby("model").size().ne(2424).any():
        raise ValueError("The main CDA UNKNOWN configuration is incomplete")
    table(trace, "cda/CDA_UNKNOWN_answer_trace.parquet", cda / "cda_trace_audit.parquet", "filter the five original UNKNOWN configurations; no new scoring")
    summary = pd.read_csv(cda / "cda_trace_summary.csv")
    table(summary[summary.marker.eq("UNKNOWN")].copy(), "cda/CDA_UNKNOWN_weight_state_reference_summary.csv",
          cda / "cda_trace_summary.csv", "retain only exact UNKNOWN marker groups")
    configs = pd.read_csv(cda / "cda_config_identity.csv")
    table(configs[configs.marker.eq("UNKNOWN")].copy(), "cda/CDA_UNKNOWN_config_sources.csv", cda / "cda_config_identity.csv", "retain only exact UNKNOWN marker source identities")
    for name in ("sources.csv", "cda_execution_differences.csv", "validation.json", "execution_receipt.json"):
        copy(cda / name, "cda/original_audit/" + name)

    for source_dir, target_dir in ((OLD / "evidence/core_mechanism_existing", "mechanism/core5"),
                                    (OLD / "evidence/extended_four_view_full101", "mechanism/extra4_representative101"),
                                    (OLD / "evidence/core_full_case_paths12", "cases/full_paths12"),
                                    (OLD / "local_sources/core_cases12/data/images", "cases/images12"),
                                    (OLD / "local_sources/native_analysis_input/cases", "cases/beef_carpaccio_eval039"),
                                    (OLD / "registered_current", "registered_protocol"),
                                    (BASE / "replay_audit_20261002_1700/final", "replay_source_audit")):
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and path.suffix != ".pyc":
                copy(path, target_dir + "/" + path.relative_to(source_dir).as_posix())
    for directory in ("core_natural_receipts", "extended_reference_receipts", "extended_label_authority_receipts"):
        source_dir = OLD / "evidence" / directory
        for path in sorted(source_dir.glob("*")):
            if path.is_file():
                copy(path, "evidence/" + directory + "/" + path.name)
    for name in ("extended_natural4.parquet", "core_natural5.parquet"):
        copy(OLD / "compact/ledgers" / name, "mechanism/natural_reference/" + name)
    for name in ("source_dictionary.parquet", "identity_dictionary.parquet", "qa_dictionary.parquet", "label_dictionary.parquet"):
        copy(OLD / "compact" / name, "mechanism/natural_reference/dictionaries/" + name)
    for path in (OLD / "local_sources/registered/configs").glob("*"):
        if path.is_file():
            copy(path, "registered_lists/" + path.name)
    output_json(output / "sources.json", sources)
    output_json(output / "receipt.json", {
        "schema": "kdm_finite_review_support_projection_v1", "passed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "GPU_initialized": False,
        "frozen_source_objects_modified": 0, "new_scoring_or_semantic_judgments": 0,
        "core_food_references": 24240, "extra_food_references": 19392,
        "extra_independent_attempts": 193920, "registered_dev_decisions": 32320,
        "actual_IP_VCD_dev_selected_models": 5, "IP_M3ID_or_extra4_dev_selection_done": False,
        "core_VizWiz_rows": 15872, "core_VizWiz_conditions": 31, "extra4_VizWiz_method_panel_complete": False,
        "CDA_UNKNOWN_answers": 12120, "CDA_equation4_full_vector_check_available": False,
        "extra4_full_diagnostic_and_case_paths_complete": False,
        "all_copied_files_equal_their_source_bytes": True,
        "runner_sha256": file_hash(Path(__file__)), "actual_command": [sys.executable, *sys.argv],
        "outputs": {str(p.relative_to(output)): file_hash(p) for p in output.rglob("*") if p.is_file()}})
    print(json.dumps({"passed": True, "files": sum(p.is_file() for p in output.rglob("*")),
                      "bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file())}))


if __name__ == "__main__":
    main()
