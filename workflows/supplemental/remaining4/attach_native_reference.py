#!/usr/bin/env python3
"""Join immutable selected-model uniform references to complete native scores."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, within
from workflows.supplemental.remaining11.analysis import condition_metrics, key_of, write_csv
from workflows.supplemental.remaining11.score import rows
from workflows.supplemental.remaining4.native import METHODS, MODELS
from workflows.supplemental.remaining4.score_native import save_json, save_rows

READY = {"onevision", "qwen3vl"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-score-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = within(ROOT, args.checkpoint)
    reference_folder = within(ROOT, args.reference_score_dir)
    output = within(ROOT, args.output)
    score_path, reference_path = checkpoint / "score_rows.jsonl.gz", reference_folder / "reference_G.jsonl"
    proof = json.loads((checkpoint / "verification.json").read_text())
    native_receipt = json.loads((checkpoint / "coverage_receipt.json").read_text())
    reference_receipt = json.loads((reference_folder / "execution_receipt.json").read_text())
    if (proof["passed"] is not True or proof["rows"] != 19392 or proof["missing_rows"]
            or proof["canonical_pending"] or proof["literal_pending"] or proof["abstain_pending"]
            or file_hash(score_path) != proof["score_rows_sha256"]
            or file_hash(checkpoint / "coverage_receipt.json") != proof["checkpoint_receipt_sha256"]
            or file_hash(reference_path) != reference_receipt["reference_G_sha256"]
            or file_hash(reference_folder / "summary.json") != reference_receipt["summary_sha256"]):
        raise ValueError("Previously verified native scores or reference source binding differs")
    samples = {sample["id"]: sample for _line, sample, _sha in rows(ROOT / "data/current/all.jsonl")
               if sample["dataset"] == "food101" and sample["split"] == "eval"}
    if len(samples) != 2424 or Counter(sample["class"] for sample in samples.values()) != Counter({c: 24 for c in {s["class"] for s in samples.values()}}):
        raise ValueError("Registered eval sample or category quota differs")
    references, reference_records = {}, []
    for line, row, line_sha in rows(reference_path):
        if row["split"] != "eval":
            continue
        key = row["model"], row["sample_id"]
        sample = samples.get(row["sample_id"])
        if (row["model"] not in MODELS or sample is None or key in references
                or row["dataset"] != "food101" or row["target_class"] != sample["class"]):
            raise ValueError("Uniform reference model/sample/target identity differs")
        if row["reference_complete"]:
            attempts = row["attempts"]
            if (row["attempt_count"] != 10 or len(attempts) != 10
                    or {a["replicate"] for a in attempts} != set(range(10))
                    or any(a["canonical_name_in_primary_score"] not in (0, 1) for a in attempts)
                    or row["correct_count"] != sum(a["canonical_name_in_primary_score"] == 1 for a in attempts)
                    or not isinstance(row["gold_rank"], int) or not 1 <= row["gold_rank"] <= 101
                    or type(row["reference_G"]) is not bool
                    or row["reference_G"] != (row["gold_rank"] > 1 and row["correct_count"] == 0)):
                raise ValueError("Existing complete uniform reference fails its registered source validation")
        elif row["reference_G"] is not None:
            raise ValueError("An incomplete reference publishes a Boolean GT")
        references[key] = row
        reference_records.append({"model": row["model"], "sample_id": row["sample_id"], "target_class": row["target_class"],
            "dataset": "food101", "split": "eval", "reference_G": row["reference_G"],
            "reference_complete": row["reference_complete"], "gold_rank": row["gold_rank"],
            "attempt_count": row["attempt_count"], "correct_count": row["correct_count"],
            "missing_replicates": row["missing_replicates"], "reference_source_path": str(reference_path.relative_to(ROOT)),
            "reference_source_line": line, "reference_source_line_sha256": line_sha,
            "rank_source": row["rank_source"],
            "attempt_source_identities": sorted({attempt["source_identity"] for attempt in row["attempts"]}),
            "attempt_source_paths": sorted({attempt["source_path"] for attempt in row["attempts"]}),
            "reference_GT_recomputed": False})
    if set(references) != {(model, sid) for model in MODELS for sid in samples}:
        raise ValueError("Reference input must contain the exact selected four-model eval keys")
    coverage = []
    for model in MODELS:
        selected = [r for (m, _sid), r in references.items() if m == model]
        full = all(r["reference_complete"] for r in selected)
        if (model in READY) != full:
            raise ValueError("Existing reference-ready models differ from the explicitly authorized OneVision/Qwen3 join")
        coverage.append({"model": model, "eval_n": 2424,
            "reference_complete_n": sum(r["reference_complete"] for r in selected),
            "reference_pending_n": sum(not r["reference_complete"] for r in selected),
            "reference_positive_complete_n": sum(r["reference_G"] is True for r in selected),
            "reference_source_sha256": reference_receipt["reference_G_sha256"]})
    ref_provenance = {(r["model"], r["sample_id"]): r for r in reference_records}
    joined, groups = [], defaultdict(dict)
    counts = {(r["model"], r["method"]): r for _line, r, _sha in rows(checkpoint / "condition_counts.jsonl")}
    originals = list(rows(score_path))
    if len(originals) != 19392 or len({r["key"] for _line, r, _sha in originals}) != 19392:
        raise ValueError("Native source unique key coverage differs")
    for line, original, line_sha in originals:
        reference = references[original["model"], original["sample_id"]]
        source = ref_provenance[original["model"], original["sample_id"]]
        if original["target_class"] != reference["target_class"]:
            raise ValueError("Uniform GT join target identity differs")
        row = {**original, "reference_G": reference["reference_G"], "reference_complete": reference["reference_complete"],
            "reference_status": "source_complete" if reference["reference_complete"] else "source_pending",
            "reference_source_path": source["reference_source_path"], "reference_source_line": source["reference_source_line"],
            "reference_source_line_sha256": source["reference_source_line_sha256"],
            "reference_source_sha256": reference_receipt["reference_G_sha256"],
            "native_source_score_path": str(score_path.relative_to(ROOT)), "native_source_score_line": line,
            "native_source_score_line_sha256": line_sha,
            "native_source_score_sha256": native_receipt["outputs"]["score_rows.jsonl.gz"],
            "source_cohort": "kdm_remaining4_native_generation_v1", "reference_GT_recomputed": False}
        for name, value in original.items():
            if name not in {"reference_G", "reference_complete", "reference_status"} and row[name] != value:
                raise ValueError("Reference attachment changed a native score object field")
        key = key_of(row)
        if row["sample_id"] in groups[key]:
            raise ValueError("Reference join repeats a native condition/sample key")
        groups[key][row["sample_id"]] = row
        joined.append(row)
    metrics = []
    for key, condition in groups.items():
        if set(condition) != set(samples) or Counter(r["target_class"] for r in condition.values()) != Counter({c: 24 for c in {s["class"] for s in samples.values()}}):
            raise ValueError("Native reference metrics lack the exact complete eval set and quotas")
        result = condition_metrics(key, condition, references)
        complete = result["reference_condition_complete"]
        correct, tp, fp, fn = (result[name] for name in ("canonical_correct", "abstention_tp_known_n", "abstention_fp_known_n", "abstention_fn_known_n"))
        result.update(condition_key=counts[result["model"], result["method"]]["condition_key"],
            precision_numerator=tp if complete else None, precision_denominator=tp + fp if complete else None,
            precision_null_reason="pending_uniform_reference" if not complete else ("no_abstentions" if tp + fp == 0 else None),
            recall_numerator=tp if complete else None, recall_denominator=tp + fn if complete else None,
            recall_null_reason="pending_uniform_reference" if not complete else ("no_reference_positives" if tp + fn == 0 else None),
            joint_J_numerator=correct + tp if complete else None, joint_J_denominator=2424 if complete else None,
            joint_J=(correct + tp) / 2424 if complete else None,
            joint_J_null_reason=None if complete else "pending_uniform_reference",
            reference_GT_recomputed=False, paired_method_effects_computed=False)
        metrics.append(result)
    if len(metrics) != 8 or {(r["model"], r["method"]) for r in metrics} != {(m, d) for m in MODELS for d in METHODS}:
        raise ValueError("Joined metrics do not retain all eight registered native conditions")
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "native_scores_reference_attached.jsonl.gz", joined)
    save_rows(output / "references_eval.jsonl", reference_records)
    save_rows(output / "condition_metrics.jsonl", metrics)
    write_csv(output / "condition_metrics.csv", metrics)
    save_rows(output / "reference_coverage.jsonl", coverage)
    source_hashes = {"native_score": file_hash(score_path), "native_verification": file_hash(checkpoint / "verification.json"),
        "reference_G": file_hash(reference_path), "reference_execution_receipt": file_hash(reference_folder / "execution_receipt.json")}
    if source_hashes["native_score"] != proof["score_rows_sha256"] or source_hashes["reference_G"] != reference_receipt["reference_G_sha256"]:
        raise ValueError("Immutable inputs changed during attachment")
    save_json(output / "receipt.json", {"schema": "kdm_remaining4_native_reference_attachment_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "native_rows": 19392, "native_conditions": 8,
        "reference_ready_conditions": 4, "reference_pending_conditions": 4, "ready_models": sorted(READY),
        "reference_eval_unique_keys": len(references), "reference_eval_complete_unique_keys": 4848,
        "reference_GT_recomputed": False, "original_native_scoring_fields_unchanged": True,
        "paired_method_effects_computed": False, "source_hashes": source_hashes,
        "reference_coverage": coverage, "attachment_sha256": file_hash(Path(__file__)),
        "metric_function_source_sha256": file_hash(ROOT / "workflows/supplemental/remaining11/analysis.py"),
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
        "raw_reopened": False, "new_generations": 0, "GPU_initialized": False, "new_API_calls": 0})
    print(json.dumps([{name: row[name] for name in ("model", "method", "n", "canonical_correct", "abstain_n",
        "reference_complete_n", "precision_numerator", "precision_denominator", "abstention_precision",
        "recall_numerator", "recall_denominator", "abstention_recall", "joint_J_numerator", "joint_J_denominator", "joint_J")}
        for row in metrics], indent=2), flush=True)


if __name__ == "__main__":
    main()
