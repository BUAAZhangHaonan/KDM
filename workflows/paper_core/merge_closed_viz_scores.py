"""Merge accepted Viz score cohorts, preserving their original scientific rows."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import file_hash, within


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-receipt", action="append", required=True)
    parser.add_argument("--expected-generated-rows", type=int, required=True)
    parser.add_argument("--expected-direct-rows", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--executor-agent", required=True)
    parser.add_argument("--executor-model", required=True)
    parser.add_argument("--executor-effort", required=True)
    args = parser.parse_args()
    require(args.expected_generated_rows > 0 and args.expected_generated_rows % 512 == 0,
            "Generated conditions must each contain the frozen 512 inputs")
    require(args.expected_direct_rows >= 0 and args.expected_direct_rows % 512 == 0,
            "Direct conditions must each contain the frozen 512 inputs")
    paths = [within(ROOT, value) for value in args.score_receipt]
    require(len(paths) == len(set(paths)), "A source scoring receipt was supplied twice")
    output = within(ROOT, args.out)
    output.resolve().relative_to(ROOT / "outputs/paper_core_20261002_dev_viz")
    output.mkdir(parents=False, exist_ok=False)
    tables, metrics, parts, source_receipts = [], [], [], []
    keys, qa_keys, condition_samples = set(), set(), {}
    identity_ids, behavior_counts = set(), Counter()
    references_hash = None
    protocol = None
    count = 0
    with (output / "score_rows.jsonl.gz").open("xb") as target:
        with gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as stream:
            for receipt_path in paths:
                receipt = load(receipt_path)
                require(receipt.get("schema") == "kdm_core_dev_viz_scoring_v1"
                        and receipt.get("passed") is True and receipt.get("stage") == "viz512",
                        "Every input must be a real accepted core Viz scorer receipt")
                require(all(receipt.get(field) == 0 for field in (
                    "pending_QA", "pending_memberships", "quality_pending_rows",
                    "abstain_pending_rows", "reference_join_missing", "pilot_partial_conditions")),
                    "An input cohort still contains a scoring or coverage gap")
                require(receipt["conditions"] == receipt["raw_complete_conditions"]
                        == receipt["primary_complete_conditions"] == len(receipt["source_parts"]),
                        "An input condition is not fully accepted")
                current = {field: receipt[field] for field in (
                    "reference_sha256", "canonical_scorer_sha256", "reused_inference_source_sha256",
                    "official_vqa_normalizer_sha256", "official_vqa_scoring_sha256", "runner_sha256")}
                require(protocol is None or current == protocol, "Input scoring protocols differ")
                protocol = current
                for name in ("score_rows.jsonl.gz", "new_scores.parquet", "metrics_all.csv",
                             "sources.csv", "frozen_references_used.csv"):
                    require(file_hash(receipt_path.parent / name) == receipt["outputs"][name],
                            "An accepted score artifact changed bytes")
                reference_file = receipt_path.parent / "frozen_references_used.csv"
                reference_hash = file_hash(reference_file)
                require(references_hash is None or reference_hash == references_hash,
                        "The finite official reference roster differs between input cohorts")
                if references_hash is None:
                    shutil.copyfile(reference_file, output / reference_file.name)
                    references_hash = reference_hash
                table = pq.read_table(receipt_path.parent / "new_scores.parquet")
                require(not tables or table.schema.equals(tables[0].schema, check_metadata=True),
                        "The original analysis schemas or types differ")
                compact_rows = table.to_pylist()
                require(len(compact_rows) == receipt["rows"], "Source score count differs from its receipt")
                source_count = 0
                with gzip.open(receipt_path.parent / "score_rows.jsonl.gz", "rb") as source:
                    for line in source:
                        require(line.endswith(b"\n"), "A scientific score line is incomplete")
                        row = json.loads(line)
                        require(source_count < len(compact_rows), "Full and compact score counts differ")
                        compact_row = compact_rows[source_count]
                        require(all(row[field] == compact_row[field]
                                    and type(row[field]) is type(compact_row[field])
                                    for field in table.column_names),
                                "Full and compact score row values or types differ")
                        require(row["key"] not in keys, "A task key occurs in two input cohorts")
                        require(row["dataset"] == "vizwiz" and row["split"] == "eval"
                                and row["abstain"] is not None and row["quality_score"] is not None
                                and row["annotated_answerable"] is not None,
                                "An input scientific decision or dataset identity is incomplete")
                        key = (row["model"], row["condition_id"])
                        samples = condition_samples.setdefault(key, set())
                        require(row["sample_id"] not in samples, "A condition repeats a sample")
                        samples.add(row["sample_id"])
                        keys.add(row["key"])
                        qa_keys.add(row["qa_key"])
                        identity_ids.add(row["source_identity"])
                        behavior_counts[row["behavior_source"]] += 1
                        stream.write(line)
                        source_count += 1
                require(source_count == table.num_rows == receipt["rows"], "Input score row count is incomplete")
                count += source_count
                tables.append(table)
                metric = pd.read_csv(receipt_path.parent / "metrics_all.csv")
                require(len(metric) == receipt["conditions"] and bool(metric.primary_complete.all()),
                        "An original metric condition is not accepted")
                metrics.append(metric)
                for part in receipt["source_parts"]:
                    require(part["rows"] == part["expected_n"] == 512 and part["raw_complete"] is True,
                            "A source part is not a completed 512-input condition")
                    identity_path = within(ROOT, part["identity_path"])
                    require(file_hash(identity_path) == part["identity_sha256"], "An original identity changed bytes")
                    require(load(identity_path)["identity"] == part["ledger_identity"],
                            "A source part identity differs from its original sidecar")
                    parts.append(part)
                source_receipts.append({"path": str(receipt_path), "sha256": file_hash(receipt_path),
                    "original_schema": receipt["schema"], "rows": receipt["rows"], "conditions": receipt["conditions"],
                    "outputs": receipt["outputs"], "actual_command": receipt["actual_command"],
                    "original_rule_executor": receipt["rule_executor"], "decision_files": receipt["decision_files"],
                    "viz_authority_files": receipt["viz_authority_files"],
                    "accepted_native_authority_receipts": receipt["accepted_native_authority_receipts"],
                    "viz_final_census_source": receipt["viz_final_census_source"]})
    generated = [part for part in parts if part["method"] != "direct"]
    direct = [part for part in parts if part["method"] == "direct"]
    require(sum(part["rows"] for part in generated) == args.expected_generated_rows
            and sum(part["rows"] for part in direct) == args.expected_direct_rows,
            "Actual generated/Direct rows differ from the explicitly requested scope")
    require(count == args.expected_generated_rows + args.expected_direct_rows
            and len(condition_samples) == len(parts) and all(len(samples) == 512 for samples in condition_samples.values()),
            "The merged condition denominators are incomplete")
    require(len({frozenset(samples) for samples in condition_samples.values()}) == 1,
            "Conditions do not use the identical finite input set")
    require(identity_ids <= {part["ledger_identity"] for part in parts},
            "A scientific row has no original bound identity sidecar")
    table = pa.concat_tables(tables)
    pq.write_table(table, output / "new_scores.parquet")
    table.to_pandas().to_csv(output / "new_scores.csv", index=False)
    pd.concat(metrics, ignore_index=True).to_csv(output / "metrics_all.csv", index=False)
    pd.DataFrame(parts).to_csv(output / "sources.csv", index=False)
    pd.DataFrame([{key: value for key, value in source.items() if key not in {
        "outputs", "actual_command", "original_rule_executor", "decision_files",
        "viz_authority_files", "accepted_native_authority_receipts", "viz_final_census_source"}}
        for source in source_receipts]).to_csv(output / "score_source_receipts.csv", index=False)
    for name in ("pending_qa.jsonl", "pending_memberships.jsonl", "pending_luna_payload.jsonl"):
        (output / name).touch(exist_ok=False)
    receipt = {"schema": "kdm_source_bound_core_viz_score_cohort_merge_v1", "passed": True, "stage": "viz512",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "rows": count, "conditions": len(parts),
        "raw_complete_conditions": len(parts), "primary_complete_conditions": len(parts), "pilot_partial_conditions": 0,
        "generated_rows": args.expected_generated_rows, "generated_conditions": len(generated),
        "frozen_Direct_rows": args.expected_direct_rows, "frozen_Direct_conditions": len(direct),
        "canonical_pending_rows": None, "literal_pending_rows": None, "quality_pending_rows": 0,
        "abstain_pending_rows": 0, "pending_QA": 0, "pending_memberships": 0, "reference_join_missing": 0,
        "unique_QA": len(qa_keys), "behavior_source_counts": dict(behavior_counts), "source_parts": parts,
        "source_score_receipts": source_receipts, "original_scoring_protocol": protocol,
        "same_full_JSON_and_analysis_row_order_values_and_types_verified": True,
        "original_source_score_artifacts_changed": False, "new_semantic_judgments": 0,
        "new_generations": 0, "new_API_calls": 0, "GPU_initialized": False, "active_raw_opened": 0,
        "actual_command": [sys.executable, *sys.argv], "merger_source_sha256": file_hash(Path(__file__)),
        "actual_executor": {"agent": args.executor_agent, "model": args.executor_model, "effort": args.executor_effort,
                            "call_id": ""},
        "outputs": {path.name: file_hash(path) for path in output.iterdir()}}
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({field: receipt[field] for field in ("passed", "stage", "rows", "conditions",
        "generated_rows", "generated_conditions", "frozen_Direct_rows", "quality_pending_rows", "pending_QA")}), flush=True)


if __name__ == "__main__":
    main()
