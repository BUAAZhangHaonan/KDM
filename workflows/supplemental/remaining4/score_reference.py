#!/usr/bin/env python3
"""Validate and score only new sealed InternVL attempts with the existing scorer."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, stable_hash, within
from kdm.pipeline import task_id
from kdm.prompts import task_prompt
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.generate import ordered_tasks, validate_inputs, verify_output
from workflows.supplemental.remaining11.score import rows, score
from workflows.supplemental.remaining4.score_native import CORE, DEFAULT_AUTHORITY, DEFAULT_AUTHORITY_SCORE, save_json, save_rows

MODEL = "internvl35_8b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--received-manifest", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--decision-file", action="append", default=[])
    parser.add_argument("--prior-score-dir", action="append", default=[])
    args = parser.parse_args()
    received_path = within(ROOT, args.received_manifest)
    received = json.loads(received_path.read_text())
    if received["schema"] != "kdm_remaining4_reference_received_manifest_v1" or not received["source_files_sha_verified"]:
        raise ValueError("A byte-verified finite reference received manifest is required")
    samples, methods, spec, config, _names, provenance = validate_inputs(ROOT, MODEL, "food101", "independent")
    tasks = {task_id(MODEL, task): task for task in ordered_tasks(samples, methods, "independent", "food101")}
    if len(tasks) != 48480:
        raise ValueError("Registered full ten-attempt plan differs")
    authority_folder = within(ROOT, DEFAULT_AUTHORITY_SCORE)
    authority_receipt = json.loads((authority_folder / "execution_receipt.json").read_text())
    authority_summary_path = authority_folder / "summary.json"
    authority_reference = authority_folder / "reference_G.jsonl"
    if (file_hash(authority_summary_path) != authority_receipt["summary_sha256"]
            or file_hash(authority_reference) != authority_receipt["reference_G_sha256"]):
        raise ValueError("Previously accepted reference-key authority binding differs")
    old_keys = {attempt["key"] for _line, ref, _sha in rows(authority_reference) if ref["model"] == MODEL for attempt in ref["attempts"]}
    for name in args.prior_score_dir:
        folder = within(ROOT, name)
        prior_receipt = json.loads((folder / "finite_execution_receipt.json").read_text())
        prior_path = folder / "independent_score_rows_enriched.jsonl.gz"
        if not prior_receipt["passed"] or file_hash(prior_path) != prior_receipt["outputs"][prior_path.name]:
            raise ValueError("Previous finite attempt score source differs")
        prior_keys = {row["key"] for _line, row, _sha in rows(prior_path)}
        if len(prior_keys) != prior_receipt["rows"] or old_keys.intersection(prior_keys):
            raise ValueError("Previous attempt keys overlap or differ from accepted coverage")
        old_keys.update(prior_keys)
    raw_by_key, proven, sources, owner_key_sets = {}, [], [], {}
    plan = {"model": MODEL, "stage": "independent", "cfg": config}
    for item in received["parts"]:
        if (item["model"], item["dataset"], item["stage"], item["source_host"]) != (MODEL, "food101", "independent", "d4030"):
            raise ValueError("Received part is outside the explicitly authorized independent source")
        files = item["files"]
        raw_path = within(ROOT, files["raw"]["received_path"])
        meta = json.loads(within(ROOT, files["identity"]["received_path"]).read_text())
        complete = json.loads(within(ROOT, files["complete"]["received_path"]).read_text())
        admission = json.loads(within(ROOT, files["admission"]["received_path"]).read_text())
        definition = meta["definition"]
        for entry in files.values():
            if file_hash(within(ROOT, entry["received_path"])) != entry["sha256"]:
                raise ValueError("Received immutable reference proof member bytes differ")
        claim = item["claim_id"]
        if claim not in owner_key_sets:
            owner = json.loads(within(ROOT, files["owner"]["received_path"]).read_text())
            owner_keys = [row["key"] for _line, row, _sha in rows(within(ROOT, files["keys"]["received_path"]))]
            ordered_sha = hashlib.sha256("".join(key + "\n" for key in owner_keys).encode()).hexdigest()
            if (owner["claim_id"] != claim or owner["plan"] != definition["task_plan"]
                    or owner["keys_sha256"] != files["keys"]["sha256"]
                    or len(owner_keys) != len(set(owner_keys))
                    or len(owner_keys) != definition["task_plan"]["expected_generation_rows"]
                    or ordered_sha != definition["task_plan"]["ordered_selected_keys_sha256"]
                    or not set(owner_keys) <= set(tasks)):
                raise ValueError("Independent claim ownership or registered full task keys differ")
            owner_key_sets[claim] = set(owner_keys)
        if (meta["identity"] != stable_hash(definition) or meta["identity"] != item["source_identity"]
                or definition["schema"] != "kdm_remaining11_generation_v1"
                or definition["model"] != MODEL or definition["dataset"] != "food101"
                or definition["stage"] != "independent" or definition["claim_id"] != claim or definition["part"] != item["part"]
                or definition["registered_backend"] != spec or definition["source_provenance"] != provenance
                or definition["base_config"] != asdict(config)
                or definition["backend"] != definition["runtime_admission"]["runtime_spec"]
                or definition["runtime_admission"] != admission["runtime_admission"]
                or definition["source_provenance"] != admission["source_provenance"]
                or definition["task_plan"] != admission["task_plan"]
                or complete["rows"] != item["rows"] or not complete["generation_complete"]
                or complete["raw_sha256"] != files["raw"]["sha256"]
                or complete["identity_sha256"] != files["identity"]["sha256"]):
            raise ValueError("Original independent runtime, parameters, input/algorithm provenance or source identity differs")
        actual = list(rows(raw_path))
        actual_keys = [row["key"] for _line, row, _sha in actual]
        if (len(actual_keys) != len(set(actual_keys)) or not set(actual_keys) <= owner_key_sets[claim]
                or set(actual_keys).intersection(old_keys) or set(actual_keys).intersection(raw_by_key)):
            raise ValueError("New sealed reference repeats previously scored keys or leaves original ownership")
        identity = {name: value for name, value in definition.items() if name not in {"shard", "n_shards", "base_config"}}
        validation = verify_output(raw_path, plan, [tasks[key] for key in actual_keys], identity, definition["shard"], definition["n_shards"])
        if any(complete[name] != value for name, value in validation.items()):
            raise ValueError("Actual CPU independent row verification differs from its sealed producer receipt")
        for line, row, line_sha in actual:
            if row["prompt"] != task_prompt(row["sample"]["question"], row["marker"], row["guided"], attempt=True) or row["reference_prompt"] is not None:
                raise ValueError("Registered independent prompt or reference prompt differs")
            raw_by_key[row["key"]] = (item, line, row, line_sha)
        proven.append({**item, "actual_source_identity": meta["identity"], "cpu_validation": validation,
                       "frozen_parameters_unchanged": True, "registered_attempt_prompt_verified": True})
        sources.append({"model": MODEL, "stage": "independent", "received_raw_path": files["raw"]["received_path"],
                        "received_receipt_path": files["complete"]["received_path"], "cohort": "registered_native_independent"})
    if len(raw_by_key) != received["received_rows"]:
        raise ValueError("Finite received independent source row coverage differs")
    run = within(ROOT, args.run)
    assets = run / "assets"
    assets.mkdir(parents=True, exist_ok=False)
    origin_manifest_path = within(ROOT, DEFAULT_AUTHORITY)
    origin_manifest = json.loads(origin_manifest_path.read_text())
    manifest = {"schema": "kdm_remaining4_finite_reference_score_asset_manifest_v1", "sources": [],
        "census_final_labels": origin_manifest["census_final_labels"], "historical_labels": [],
        "original_authority_manifest": str(origin_manifest_path.relative_to(ROOT)),
        "original_authority_manifest_sha256": file_hash(origin_manifest_path), "old_raw_scanned": False}
    save_json(assets / "asset_manifest.json", manifest)
    source_manifest_path = assets / "received_for_existing_scorer.json"
    save_json(source_manifest_path, {"parts": sources, "source_received_manifest": str(received_path.relative_to(ROOT)),
        "source_received_manifest_sha256": file_hash(received_path), "only_new_verified_independent_parts": True})
    proof_path = assets / "received_source_verification.json"
    save_json(proof_path, {"schema": "kdm_remaining4_independent_source_verification_v1", "passed": True,
        "rows": len(raw_by_key), "parts": proven, "duplicate_keys": 0, "previous_scored_keys_excluded": len(old_keys),
        "old_raw_scanned": False, "parameters_changed": False, "GPU_initialized": False})
    authority_summary = json.loads(authority_summary_path.read_text())
    decision_files = [within(ROOT, name) for name in authority_summary["decision_files"]]
    decision_files += [ROOT / CORE / "annotation/core_p1_native_luna_medium_20260930/root_reviewed.jsonl",
        ROOT / CORE / "annotation/gemma_native_luna_medium_20260930/root_reviewed.jsonl",
        ROOT / CORE / "annotation/root_literal5_20260930/root_reviewed.jsonl",
        ROOT / CORE / "scores/v6_all5_literal_root_closed/root_direct_authority_mapping.jsonl"]
    decision_files += [within(ROOT, name) for name in args.decision_file]
    output = run / "scores" / args.output_name
    with (assets / "existing_scorer_stdout.log").open("x", encoding="utf-8") as log, redirect_stdout(log):
        summary = score(run, output, decision_files, (MODEL,), (source_manifest_path,))
    enriched, pending, repeats = [], 0, set()
    for _line, record, _sha in rows(output / "score_rows.jsonl.gz"):
        item, line, raw, line_sha = raw_by_key[record["key"]]
        if (record["key"] in repeats or record["stage"] != "independent" or record["replicate"] != raw["replicate"]
                or record["source_identity"] != raw["identity"] or record["raw_line_sha256"] != line_sha
                or record["seed"] != raw["seed"] or record["config_sha256"] != stable_hash(raw["config"])):
            raise ValueError("Existing scorer changed independent identity, attempt, seed, configuration or source membership")
        repeats.add(record["key"])
        enriched.append({**record, "attempt": True, "attempt_ordinal": raw["replicate"] + 1, "config": raw["config"],
            "prompt": raw["prompt"], "reference_prompt": raw["reference_prompt"],
            "source_remote_host": item["source_host"], "source_remote_path": item["files"]["raw"]["remote_path"],
            "source_claim": item["claim_id"], "source_part": item["part"], "raw_source_sha256": item["files"]["raw"]["sha256"],
            "source_CPU_verification_path": str(proof_path.relative_to(ROOT)), "source_CPU_verification_sha256": file_hash(proof_path)})
        pending += record["canonical_name_in_primary_score"] is None
    if repeats != set(raw_by_key):
        raise ValueError("Existing scorer did not cover the exact new sealed attempt keys")
    save_rows(output / "independent_score_rows_enriched.jsonl.gz", enriched)
    references = [record for _line, record, _sha in rows(output / "reference_G.jsonl")]
    if (len(references) != 4848 or len({r["sample_id"] for r in references}) != 4848
            or sum(r["attempt_count"] for r in references) != len(enriched)
            or any(r["reference_G"] is not None or r["reference_complete"] for r in references)):
        raise ValueError("Incomplete InternVL rank coverage was converted into a Boolean uniform GT")
    coverage = [{"split": split, "registered_samples": 2424, "registered_attempts": 24240,
        "received_attempts": sum(r["split"] == split for r in enriched),
        "samples_with_any_attempt": sum(r["split"] == split and r["attempt_count"] > 0 for r in references),
        "samples_with_all10": sum(r["split"] == split and r["attempt_count"] == 10 for r in references),
        "samples_with_all10_correctness_decided": sum(r["split"] == split and r["correct_count"] is not None for r in references),
        "rank_missing_samples": sum(r["split"] == split and r["gold_rank"] is None for r in references),
        "uniform_GT_complete_samples": 0} for split in ("dev", "eval")]
    save_rows(output / "attempt_coverage.jsonl", coverage)
    receipt = {"schema": "kdm_remaining4_finite_independent_score_execution_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "rows": len(enriched), "parts": len(proven),
        "registered_expected_attempts": 48480, "unreceived_registered_attempts": 48480 - len(old_keys) - len(enriched),
        "previous_scored_keys_excluded": len(old_keys), "canonical_pending": pending,
        "abstain_pending": sum(r["abstain"] is None for r in enriched),
        "literal_pending": sum(r["literal_extracted_name_score"] is None for r in enriched),
        "boundary_QA": summary["pending_boundary_questions"], "unique_QA": summary["unique_scored_QA"],
        "attempt_ordinal_policy": "replicate0..9 preserved; explicit ordinal=replicate+1", "attempt_coverage": coverage,
        "uniform_GT_complete": 0, "uniform_GT_pending": 4848,
        "reference_pending_reason": "registered Food candidate rank missing; less than ten or undecided attempts remain pending",
        "existing_scorer_source_sha256": file_hash(ROOT / "workflows/supplemental/remaining11/score.py"),
        "canonical_scorer_source_sha256": file_hash(Path(frozen.__file__)), "adapter_sha256": file_hash(Path(__file__)),
        "received_manifest_sha256": file_hash(received_path), "source_CPU_verification_sha256": file_hash(proof_path),
        "decision_files": [{"path": str(p.relative_to(ROOT)), "sha256": file_hash(p)} for p in decision_files],
        "old_raw_scanned": False, "old_score_rows_recomputed": False, "whether_should_abstain_cross_validation": False,
        "GPU_initialized": False, "new_generations": 0, "new_API_calls": 0,
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / "finite_execution_receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"decision_files", "outputs"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
