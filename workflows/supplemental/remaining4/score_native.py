#!/usr/bin/env python3
"""Validate and score only explicitly received immutable native four-model parts."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, stable_hash, within
from kdm.pipeline import task_id
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.generate import verify_output
from workflows.supplemental.remaining11.score import infer_qa, load_authority, rows, score_target
from workflows.supplemental.remaining4.native import METHODS, MODELS, STAGE, native_tasks, source_inputs

COND = ("model", "dataset", "split", "method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")
DEFAULT_AUTHORITY = "outputs/supplemental/remaining11/run_20260930_140337/assets/asset_manifest.json"
DEFAULT_AUTHORITY_SCORE = "outputs/supplemental/remaining11/run_20260930_140337/scores/selected4_v10_root46"
CORE = "outputs/paper_core_20260930/run_20260930_core_p1_cpu"


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_rows(path, values):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "xt", encoding="utf-8", newline="\n") as stream:
        for record in values:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def validate_parts(args):
    manifest_path = within(ROOT, args.received_manifest)
    received = json.loads(manifest_path.read_text())
    if received["schema"] != "kdm_remaining4_native_received_manifest_v1" or not received["parts"]:
        raise ValueError("Explicit received sealed-part manifest is required")
    plans, proven_parts, records, seen = {}, [], [], set()
    for item in received["parts"]:
        model, method = item["model"], item["method"]
        if model not in MODELS or method not in METHODS:
            raise ValueError("Received part is outside the four native models and two registered methods")
        files = item["files"]
        for entry in files.values():
            path = within(ROOT, entry["received_path"])
            if file_hash(path) != entry["sha256"]:
                raise ValueError("Received immutable source bytes differ: " + str(path))
        raw_path = within(ROOT, files["raw"]["received_path"])
        meta = json.loads(within(ROOT, files["identity"]["received_path"]).read_text())
        complete = json.loads(within(ROOT, files["complete"]["received_path"]).read_text())
        admission = json.loads(within(ROOT, files["admission"]["received_path"]).read_text())
        owner = json.loads(within(ROOT, files["owner"]["received_path"]).read_text())
        owner_keys = [r["key"] for _, r, _ in rows(within(ROOT, files["keys"]["received_path"]))]
        definition = meta["definition"]
        if (meta["identity"] != stable_hash(definition)
                or definition["schema"] != "kdm_remaining4_native_generation_v1"
                or definition["model"] != model or definition["method"] != method
                or definition["claim_id"] != item["claim_id"] or definition["part"] != item["part"]
                or definition["registered_backend"]["key"] != model
                or not complete["generation_complete"] or complete["raw_sha256"] != files["raw"]["sha256"]
                or complete["identity_sha256"] != files["identity"]["sha256"]
                or complete["rows"] != item["rows"] or complete["part"] != item["part"]):
            raise ValueError("Sealed producer receipt, source identity or part declaration differs")
        if (definition["runtime_admission"] != admission["runtime_admission"]
                or definition["backend"] != definition["runtime_admission"]["runtime_spec"]
                or definition["source_provenance"] != admission["source_provenance"]
                or owner["claim_id"] != item["claim_id"]
                or owner["keys_sha256"] != files["keys"]["sha256"]
                or owner["plan"] != definition["task_plan"]
                or len(owner_keys) != len(set(owner_keys))
                or len(owner_keys) != definition["task_plan"]["expected_generation_rows"]):
            raise ValueError("Original admitted runtime or mutually exclusive claim proof differs")
        if (model, method) not in plans:
            samples, spec, config, provenance = source_inputs(model, method)
            plans[model, method] = {"model": model, "stage": STAGE, "cfg": config,
                "tasks": {task_id(model, task): task for task in native_tasks(samples, method)},
                "spec": spec, "provenance": provenance}
        plan = plans[model, method]
        if definition["source_provenance"] != plan["provenance"] or definition["registered_backend"] != plan["spec"]:
            raise ValueError("Frozen input, algorithms, parameters or original runtime registration differs")
        actual = list(rows(raw_path))
        keys = [row["key"] for _, row, _ in actual]
        if (len(keys) != len(set(keys)) or seen.intersection(keys) or not set(keys) <= set(owner_keys)
                or not set(owner_keys) <= set(plan["tasks"]) or len(keys) != definition["expected_part_rows"]):
            raise ValueError("Received native part keys overlap, are unregistered, or differ from claim ownership")
        part_identity = {key: value for key, value in definition.items() if key not in {"shard", "n_shards", "base_config"}}
        coverage = verify_output(raw_path, plan, [plan["tasks"][key] for key in keys], part_identity,
            definition["shard"], definition["n_shards"])
        if any(complete[field] != value for field, value in coverage.items()):
            raise ValueError("Source complete receipt differs from actual CPU row validation")
        audit = json.loads(within(ROOT, files["operator_audit"]["received_path"]).read_text())
        if not audit.get("passed") and audit.get("status") != "passed":
            raise ValueError("Claim lacks a successful actual operator audit")
        seen.update(keys)
        proven_parts.append({**item, "cpu_validation": coverage, "actual_source_identity": meta["identity"],
            "frozen_parameters_unchanged": True, "native_prompt_and_algorithm_verified": True})
        records.extend((item, line, row, line_sha) for line, row, line_sha in actual)
    return manifest_path, plans, proven_parts, records


def authority_inputs(args):
    manifest_path = within(ROOT, args.authority_manifest)
    original = json.loads(manifest_path.read_text())
    selected = {"census_final_labels": original["census_final_labels"], "historical_labels": []}
    current = within(ROOT, args.authority_score_dir)
    summary = json.loads((current / "summary.json").read_text())
    receipt = json.loads((current / "execution_receipt.json").read_text())
    if file_hash(current / "summary.json") != receipt["summary_sha256"]:
        raise ValueError("Previously closed four-model decision authority receipt differs")
    paths = [within(ROOT, path) for path in summary["decision_files"]]
    paths += [ROOT / CORE / "annotation/core_p1_native_luna_medium_20260930/root_reviewed.jsonl",
        ROOT / CORE / "annotation/gemma_native_luna_medium_20260930/root_reviewed.jsonl",
        ROOT / CORE / "annotation/root_literal5_20260930/root_reviewed.jsonl",
        ROOT / CORE / "scores/v6_all5_literal_root_closed/root_direct_authority_mapping.jsonl"]
    paths += [within(ROOT, name) for name in args.decision_file]
    reviews, behavior, provenance, _historical, decisions = load_authority(current, selected, paths)
    return reviews, behavior, provenance, decisions, {
        "source_manifest": str(manifest_path.relative_to(ROOT)), "source_manifest_sha256": file_hash(manifest_path),
        "frozen_exact_QA_source": "outputs/annotations/main_results/reviewed_answers.jsonl",
        "final_census_behavior_source": selected["census_final_labels"],
        "decision_files": [{"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)} for path in paths],
        "old_raw_scanned": False, "old_score_rows_recomputed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--received-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authority-manifest", default=DEFAULT_AUTHORITY)
    parser.add_argument("--authority-score-dir", default=DEFAULT_AUTHORITY_SCORE)
    parser.add_argument("--decision-file", action="append", default=[])
    parser.add_argument("--prior-score-dir", action="append", default=[],
                        help="Verified previous score rows used only to exclude task keys; decisions remain unchanged")
    args = parser.parse_args()
    manifest_path, plans, proven_parts, inputs = validate_parts(args)
    previous_keys, previous_sources = set(), []
    for name in args.prior_score_dir:
        previous = within(ROOT, name)
        prior_receipt = json.loads((previous / "receipt.json").read_text())
        prior_verification = json.loads((previous / "verification.json").read_text())
        prior_path = previous / "score_rows.jsonl.gz"
        prior_hash = file_hash(prior_path)
        if (prior_verification["passed"] is not True or prior_receipt["score_rows_sha256"] != prior_hash
                or prior_verification["score_rows_sha256"] != prior_hash
                or prior_verification["source_verification_sha256"] != file_hash(previous / "received_source_verification.json")
                or prior_verification["rows"] != prior_receipt["rows"]
                or prior_verification["canonical_pending"] != prior_receipt["canonical_pending"]
                or prior_verification["abstain_pending"] != prior_receipt["abstain_pending"]
                or prior_receipt["canonical_scorer_sha256"] != file_hash(Path(frozen.__file__))):
            raise ValueError("Previous verified native-score source binding differs")
        prior_keys = {row["key"] for _, row, _ in rows(prior_path)}
        if len(prior_keys) != prior_receipt["rows"] or previous_keys.intersection(prior_keys):
            raise ValueError("Previous native score keys are repeated or incomplete")
        previous_keys.update(prior_keys)
        previous_sources.append({"path": str(prior_path.relative_to(ROOT)), "sha256": prior_hash,
                                 "rows": len(prior_keys), "raw_rescanned": False,
                                 "canonical_pending": prior_receipt["canonical_pending"],
                                 "abstain_pending": prior_receipt["abstain_pending"],
                                 "usage": "task_key_exclusion_only", "decisions_reused": False})
    if any(raw["key"] in previous_keys for _source, _line, raw, _line_sha in inputs):
        raise ValueError("New sealed native part repeats a previously scored task key")
    output = within(ROOT, args.output)
    output.mkdir(parents=True, exist_ok=False)
    reviews, behavior, behavior_sources, decisions, authorities = authority_inputs(args)
    samples = next(iter(plans.values()))["tasks"].values()
    classes = sorted({task["sample"]["class"] for task in samples})
    if len(classes) != 101:
        raise ValueError("Frozen canonical class set differs")
    patterns, cache, scored, pending = frozen.compile_classes(classes), {}, [], {}
    for source, line, raw, line_sha in inputs:
        sample, answer = raw["sample"], raw["text"]
        qkey = frozen.qah(sample["question"], answer)
        if qkey not in cache:
            cache[qkey] = infer_qa(sample["question"], answer, patterns, reviews, behavior, decisions)
        inferred = cache[qkey]
        canonical, literal, reason = score_target(answer, sample["class"], inferred, patterns)
        names = sorted({name for variant in inferred["variants"] for name in frozen.extract(variant, patterns)["literal"]})
        if not names and inferred["parsed"]["primary_name"]:
            names = [inferred["parsed"]["primary_name"]]
        record = {**{field: raw[field] for field in COND if field not in {"dataset", "split"}},
            "dataset": "food101", "split": "eval", "main_marker": raw["marker"], "stage": STAGE,
            "sample_id": sample["id"], "target_class": sample["class"], "question": sample["question"], "answer": answer,
            "key": raw["key"], "original_key": raw["key"], "qa_key": qkey,
            "canonical_name_in_primary_score": canonical, "literal_extracted_name_score": literal,
            "literal_extracted_names": names, "abstain": inferred["abstain"], "score_reason": reason,
            "behavior_source": inferred["behavior_source"], "primary_extraction": inferred["parsed"],
            "behavior_history_sources": behavior_sources.get(qkey, []),
            "name_history_source": inferred["review"] if inferred["review"] else None,
            "decision_source": inferred["decision"], "annotation_model": inferred["annotation_model"],
            "annotation_effort": inferred["annotation_effort"], "annotation_call_id": inferred["annotation_call_id"],
            "source_path": source["files"]["raw"]["received_path"], "source_line": line,
            "source_remote_path": source["files"]["raw"]["remote_path"], "source_remote_host": source["source_host"],
            "source_identity": raw["identity"], "raw_line_sha256": line_sha,
            "raw_source_sha256": source["files"]["raw"]["sha256"], "source_claim": source["claim_id"], "source_part": source["part"],
            "seed": raw["seed"], "config": raw["config"], "terminated": raw["terminated"],
            "prompt": raw["prompt"], "reference_prompt": raw["reference_prompt"],
            "reference_G": None, "reference_complete": False,
            "reference_status": "not_connected_in_this_finite_sealed_partial_score"}
        scored.append(record)
        if canonical is None or inferred["abstain"] is None:
            packet = pending.setdefault(qkey, {"qa_key": qkey, "question": sample["question"], "answer": answer,
                "reasons": [], "primary_extraction": inferred["parsed"], "memberships": []})
            packet["reasons"] = sorted(set(packet["reasons"] + [reason, inferred["behavior_source"]]))
            packet["memberships"].append({key: record[key] for key in ("model", "method", "sample_id", "target_class", "key",
                "source_path", "source_line", "source_identity", "raw_line_sha256", "source_claim", "source_part")})
            packet["memberships"][-1].update(canonical_pending=canonical is None, abstain_pending=inferred["abstain"] is None)
    save_rows(output / "score_rows.jsonl.gz", scored)
    save_rows(output / "boundary_queue.jsonl", [pending[key] for key in sorted(pending)])
    counts = []
    for model in MODELS:
        for method in METHODS:
            selected = [row for row in scored if row["model"] == model and row["method"] == method]
            if len(selected) > 2424 or len({row["key"] for row in selected}) != len(selected):
                raise ValueError("Native condition received more than the registered unique eval tasks")
            quota = Counter(row["target_class"] for row in selected)
            if any(count > 24 for count in quota.values()):
                raise ValueError("Native partial category count exceeds the registered eval quota")
            counts.append({"model": model, "method": method, "dataset": "food101", "split": "eval",
                "kind": STAGE, "main_marker": "NONE", "reference_marker": "NONE", "guided": False, "reference_guided": False,
                "replicate": 0, "expected_n": 2424, "received_n": len(selected), "unreceived_n": 2424 - len(selected),
                "canonical_correct": sum(row["canonical_name_in_primary_score"] == 1 for row in selected),
                "canonical_incorrect": sum(row["canonical_name_in_primary_score"] == 0 for row in selected),
                "canonical_pending": sum(row["canonical_name_in_primary_score"] is None for row in selected),
                "literal_correct": sum(row["literal_extracted_name_score"] == 1 for row in selected),
                "literal_pending": sum(row["literal_extracted_name_score"] is None for row in selected),
                "abstain_true": sum(row["abstain"] is True for row in selected),
                "abstain_false": sum(row["abstain"] is False for row in selected),
                "abstain_pending": sum(row["abstain"] is None for row in selected),
                "reference_connected_n": 0, "full_condition_effect_calculated": False,
                "received_category_counts": dict(quota)})
    save_rows(output / "condition_counts.jsonl", counts)
    save_json(output / "received_source_verification.json", {"schema": "kdm_remaining4_native_source_verification_v1",
        "passed": True, "parts": proven_parts, "rows": len(scored), "duplicate_keys": 0,
        "parameters_changed": False, "frozen_algorithm_changed": False, "old_outputs_changed": False,
        "GPU_admission_newly_claimed": False, "GPU_initialized": False})
    receipt = {"schema": "kdm_remaining4_native_partial_score_v1", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_parts": len(proven_parts), "rows": len(scored), "unique_QA": len(cache), "boundary_QA": len(pending),
        "canonical_pending": sum(r["canonical_name_in_primary_score"] is None for r in scored),
        "abstain_pending": sum(r["abstain"] is None for r in scored),
        "literal_pending": sum(r["literal_extracted_name_score"] is None for r in scored),
        "planned_native_conditions": 8, "full_condition_rows": 2424, "condition_counts": counts,
        "reference_connected_n": 0, "full_condition_effect_calculated": False,
        "behavior_source_counts": dict(Counter(r["behavior_source"] for r in scored)), "authority_inputs": authorities,
        "received_manifest": str(manifest_path.relative_to(ROOT)), "received_manifest_sha256": file_hash(manifest_path),
        "previous_verified_score_keys_checked": len(previous_keys), "previous_score_sources": previous_sources,
        "score_scope": "only_new_received_sealed_parts", "previous_raw_rescanned": False,
        "canonical_scorer_sha256": file_hash(Path(frozen.__file__)),
        "inference_function_source_sha256": file_hash(ROOT / "workflows/supplemental/remaining11/score.py"),
        "scorer_sha256": file_hash(Path(__file__)), "score_rows_sha256": file_hash(output / "score_rows.jsonl.gz"),
        "new_generations": 0, "new_API_calls": 0, "GPU_initialized": False}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key not in {"authority_inputs", "condition_counts"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
