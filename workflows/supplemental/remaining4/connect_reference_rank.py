#!/usr/bin/env python3
"""Receive the explicit sealed rank snapshot and connect accepted finite attempts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
ROOT_6403 = "/home/team/zhanghaonan/TAFFC/knowledge-deficit-mitigation"
ROOT_D4030 = "/home/d4030/projects/knowledge-deficit-mitigation/supplemental/remaining4/runtime_20260930"
MODEL = "internvl35_8b"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def scp(host, paths, target):
    subprocess.run(["scp", "-o", "BatchMode=yes", *[host + ":" + str(path) for path in paths], str(target)], check=True)


def receive(snapshot_path, directory, central_relative):
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if (snapshot["schema"] != "kdm_compact_intern_rank_snapshot_v1" or snapshot["model"] != MODEL
            or snapshot["stage"] != "candidate" or snapshot["expected"] != 4848
            or snapshot["immutable_rows"] != 408 or snapshot["active_raw_files_hashed"] != 0
            or snapshot["original_d4030_first8"]["rows"] != 8):
        raise ValueError("This receipt is limited to the specifically authorized old 408-rank snapshot")
    central = PurePosixPath(central_relative)
    if central.is_absolute() or ".." in central.parts:
        raise ValueError("Received directory must stay project-relative")
    directory.mkdir(parents=True, exist_ok=False)
    first_folder = directory / "original_first8_progress"
    first_folder.mkdir()
    first_pointer = snapshot["original_d4030_first8"]
    scp("6403", [first_pointer["complete_path"], first_pointer["keys_path"]], first_folder)
    first_progress = json.loads((first_folder / "complete.json").read_text(encoding="utf-8"))
    if (first_progress["status"] != "generation_complete" or first_progress["completed"] != 8
            or first_progress["expected"] != 8 or not first_progress["generation_complete"]
            or first_progress["model"] != MODEL or first_progress["stage"] != "candidate"
            or len(first_progress["completed_parts"]) != 1):
        raise ValueError("Original d4030 first-eight production is not proven complete")
    first_part = first_progress["completed_parts"][0]
    if first_part["part"] != 0 or first_part["rows"] != 8:
        raise ValueError("Original rank timing coverage differs")
    raw_relative = PurePosixPath(first_part["raw_path"])
    record_relative = PurePosixPath(str(raw_relative.parent).replace("/raw/", "/records/", 1))
    claim_relative = PurePosixPath(first_progress["claim_path"])
    raw_name = raw_relative.stem
    first = {"model": MODEL, "dataset": "food101", "stage": "candidate", "host": "d4030",
        "claim_id": first_progress["claim_id"], "part": 0, "rows": 8,
        "raw_path": str(PurePosixPath(ROOT_D4030) / raw_relative),
        "identity_path": str(PurePosixPath(ROOT_D4030) / raw_relative.with_suffix(".identity.json")),
        "complete_path": str(PurePosixPath(ROOT_D4030) / record_relative / (raw_name + ".complete.json")),
        "owner_path": str(PurePosixPath(ROOT_D4030) / claim_relative / "owner.json"),
        "keys_path": str(PurePosixPath(ROOT_D4030) / claim_relative / "keys.jsonl"),
        "admission_path": str(PurePosixPath(ROOT_D4030) / record_relative / "admission.json"),
        "admission_sha256": first_progress["admission_sha256"],
        "original_progress_source": first_pointer, "original_progress_sha256": digest(first_folder / "complete.json")}
    selected = [first, *snapshot["immutable_parts"]]
    if (sum(item["rows"] for item in selected) != 408
            or len({(item["host"], item["claim_id"], item["part"]) for item in selected}) != len(selected)):
        raise ValueError("400 sealed 6403 ranks plus the original eight do not match 408 unique declared rows")

    def transport(item):
        host = item["host"]
        source_root = ROOT_6403 if host == "6403" else ROOT_D4030 if host == "d4030" else None
        if (not source_root or (item["model"], item["dataset"], item["stage"]) != (MODEL, "food101", "candidate")):
            raise ValueError("A rank member leaves the authorized source scope")
        paths = {kind: item[kind + "_path"] for kind in ("raw", "identity", "complete", "owner", "keys", "admission")}
        for path in paths.values():
            remote = PurePosixPath(path)
            remote.relative_to(PurePosixPath(source_root))
            if ".." in remote.parts:
                raise ValueError("Explicit rank proof member leaves its source root")
        target = directory / item["claim_id"]
        target.mkdir(exist_ok=False)
        scp(host, [path for kind, path in paths.items() if kind != "raw"], target)
        complete = json.loads((target / PurePosixPath(paths["complete"]).name).read_text(encoding="utf-8"))
        if (not complete["generation_complete"] or complete["rows"] != item["rows"]
                or complete["claim_id"] != item["claim_id"] or complete["part"] != item["part"]):
            raise ValueError("Rank part lacks its immutable actual producer receipt")
        pinned = {"raw": complete["raw_sha256"], "identity": complete["identity_sha256"]}
        for kind in ("raw", "identity", "complete", "admission"):
            declared = item.get(kind + "_sha256")
            if declared and kind in pinned and pinned[kind] != declared:
                raise ValueError("Immutable rank snapshot and original producer source digest differ")
            if declared:
                pinned[kind] = declared
        for kind, path in paths.items():
            if kind != "raw" and kind in pinned and digest(target / PurePosixPath(path).name) != pinned[kind]:
                raise ValueError("Received original rank metadata differs from its pinned digest")
        scp(host, [paths["raw"]], target)
        if digest(target / PurePosixPath(paths["raw"]).name) != pinned["raw"]:
            raise ValueError("Received sealed rank raw bytes differ from original producer receipt")
        files = {kind: {"remote_path": path, "received_path": str(central / item["claim_id"] / PurePosixPath(path).name),
                       "sha256": digest(target / PurePosixPath(path).name),
                       "source_declared_sha256": pinned.get(kind),
                       "digest_provenance": "original_source_receipt_or_snapshot" if kind in pinned else "actual_received_small_metadata"}
                 for kind, path in paths.items()}
        meta = json.loads((target / PurePosixPath(paths["identity"]).name).read_text(encoding="utf-8"))
        return {**{name: item[name] for name in ("model", "dataset", "stage", "claim_id", "part", "rows")},
            "source_host": host, "source_root": source_root, "files": files, "source_identity": meta["identity"],
            "original_snapshot_row": item, "source_finished_utc": complete["finished_utc"]}

    with ThreadPoolExecutor(max_workers=2) as executor:
        parts = list(executor.map(transport, selected))
    receipt = {"schema": "kdm_remaining4_rank_received_manifest_v1", "passed": True, "received_rows": 408,
        "parts": parts, "snapshot_path": str(snapshot_path.resolve()), "snapshot_sha256": digest(snapshot_path),
        "snapshot_captured_utc": snapshot["captured_utc"], "first8_progress_sha256": digest(first_folder / "complete.json"),
        "receiver_sha256": digest(Path(__file__)), "source_files_sha_verified": True, "active_raw_files_read": False,
        "remote_raw_files_hashed": False, "GPU_queries": 0, "CPU_row_validation": "pending_connect_stage"}
    write_json(directory / "received_manifest.json", receipt)
    print(json.dumps({"rows": 408, "parts": len(parts), "6403_rows": 400, "d4030_rows": 8}))


def connect(received_path, score_dir, output):
    from dataclasses import asdict
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from kdm.io import stable_hash, stable_seed, within
    from kdm.prompts import closed_prompt
    from workflows.supplemental.remaining11.generate import generation_key, ordered_tasks, validate_inputs, verify_output
    from workflows.supplemental.remaining11.score import rows
    from workflows.supplemental.remaining4.score_native import DEFAULT_AUTHORITY_SCORE, save_rows
    received_path = within(ROOT, str(received_path))
    score_dir = within(ROOT, str(score_dir))
    output = within(ROOT, str(output))
    received = json.loads(received_path.read_text())
    if received["schema"] != "kdm_remaining4_rank_received_manifest_v1" or not received["source_files_sha_verified"]:
        raise ValueError("Actual byte-verified finite rank reception is required")
    samples, methods, spec, config, names, provenance = validate_inputs(ROOT, MODEL, "food101", "candidate")
    tasks = {generation_key(MODEL, "candidate", task): task for task in ordered_tasks(samples, methods, "candidate", "food101")}
    ranks, proven = {}, []
    for item in received["parts"]:
        files = item["files"]
        for entry in files.values():
            if digest(within(ROOT, entry["received_path"])) != entry["sha256"]:
                raise ValueError("Finite received rank proof bytes differ")
        meta = json.loads(within(ROOT, files["identity"]["received_path"]).read_text())
        definition = meta["definition"]
        owner = json.loads(within(ROOT, files["owner"]["received_path"]).read_text())
        admission = json.loads(within(ROOT, files["admission"]["received_path"]).read_text())
        source_provenance = definition["source_provenance"]
        optimization = source_provenance.get("compute_optimization")
        source_base = {k: v for k, v in source_provenance.items() if k != "compute_optimization"}
        if (meta["identity"] != stable_hash(definition) or meta["identity"] != item["source_identity"]
                or definition["registered_backend"] != spec or source_base != provenance
                or definition["backend"] != definition["runtime_admission"]["runtime_spec"]
                or definition["runtime_admission"] != admission["runtime_admission"]
                or definition["source_provenance"] != admission["source_provenance"]
                or definition["task_plan"] != admission["task_plan"] or owner["plan"] != definition["task_plan"]
                or definition["task_plan"]["base_config"] != asdict(config)
                or definition["task_plan"]["source_provenance"] != source_provenance
                or definition["claim_id"] != item["claim_id"] or definition["part"] != item["part"]
                or definition["names"] != names or definition["ranking_rule"] != "mean_log_probability"):
            raise ValueError("Original ranking identity, registered backend, input/algorithm, configuration or claim differs")
        gate_path = None
        if optimization:
            mode = {"identical_clean_prompt_prefill_reuse": "clone",
                    "identical_clean_prompt_readonly_prefill_reuse": "readonly"}.get(optimization["name"])
            if (mode is None or optimization["scientific_parameters_changed"] is not False
                    or optimization["original_candidate_token_forwards"] is not True
                    or mode == "clone" and optimization.get("independent_KV_clones") is not True
                    or mode == "readonly" and (optimization.get("independent_cache_objects") is not True
                                                or optimization.get("readonly_shared_prefill_tensors") is not True)):
                raise ValueError("Rank source leaves its actual admitted prefill implementation or scientific parameters")
            gate_path = received_path.parent / "compute_gate" / mode / "success.json"
            gate = json.loads(gate_path.read_text())
            if (digest(gate_path) != optimization["gate_sha256"] or gate["status"] != "pass"
                    or gate["model"] != MODEL or gate["inputs"] != 8 or gate["candidate_score_rows"] != 808
                    or gate["max_absolute_score_difference"] != 0 or gate["rank_differences"] != 0
                    or gate["cache_sha256"] != optimization["cache_sha256"]
                    or gate["scientific_parameters_changed"] is not False
                    or gate["original_algorithm_sha256"] != digest(ROOT / "src/kdm/pipeline.py")
                    or mode == "readonly" and (gate["all_base_tensor_equality_checks_passed"] is not True
                                               or gate["tensor_integrity_gate_enabled"] is not True)):
                raise ValueError("Original source cache comparison gate or frozen candidate algorithm bytes differ")
        keys = [r["key"] for _line, r, _sha in rows(within(ROOT, files["keys"]["received_path"]))]
        if (len(keys) != len(set(keys)) or len(keys) != definition["task_plan"]["expected_generation_rows"]
                or owner["keys_sha256"] != files["keys"]["sha256"] or not set(keys) <= set(tasks)
                or hashlib.sha256("".join(key + "\n" for key in keys).encode()).hexdigest() != definition["task_plan"]["ordered_selected_keys_sha256"]):
            raise ValueError("Rank original key ownership differs")
        raw_path = within(ROOT, files["raw"]["received_path"])
        raw = list(rows(raw_path))
        actual_keys = [row["key"] for _line, row, _sha in raw]
        if len(actual_keys) != len(set(actual_keys)) or not set(actual_keys) <= set(keys):
            raise ValueError("Rank received row keys repeat or leave source ownership")
        identity = {k: v for k, v in definition.items() if k not in {"names", "ranking_rule"}}
        validation = verify_output(raw_path, {"model": MODEL, "stage": "candidate", "names": names, "cfg": config},
                                   [tasks[key] for key in actual_keys], identity, 0, 1)
        complete = json.loads(within(ROOT, files["complete"]["received_path"]).read_text())
        if any(complete[name] != value for name, value in validation.items()) or validation["rows"] != item["rows"]:
            raise ValueError("Actual all-101 finite probabilities and mean-rank validation differs from original producer receipt")
        for line, row, line_sha in raw:
            sample_id = row["sample"]["id"]
            if (sample_id in ranks or row["prompt"] != closed_prompt(row["sample"]["question"], names)
                    or row["seed"] != stable_seed(sample_id, MODEL, 0)):
                raise ValueError("Rank exact closed prompt, stable seed or sample keys differ")
            ranks[sample_id] = {"gold_rank": row["gold_rank"], "model": MODEL, "sample_id": sample_id,
                "split": row["sample"]["split"], "target_class": row["sample"]["class"], "key": row["key"],
                "source_path": str(raw_path.relative_to(ROOT)), "source_line": line, "source_identity": row["identity"],
                "raw_line_sha256": line_sha, "raw_source_sha256": files["raw"]["sha256"],
                "source_host": item["source_host"], "source_remote_path": files["raw"]["remote_path"],
                "source_claim": item["claim_id"], "source_part": item["part"], "ranking_rule": row["ranking_rule"],
                "seed": row["seed"], "full_candidate_scores": row["candidate_scores"],
                "source_compute_optimization": optimization, "actual_original_identity_definition": definition}
        proven.append({**item, "actual_source_identity": meta["identity"], "cpu_validation": validation,
            "frozen_scientific_parameters_unchanged": True, "source_compute_optimization": optimization,
            "source_compute_gate_received_path": str(gate_path) if gate_path else None,
            "source_compute_gate_sha256": digest(gate_path) if gate_path else None})
    if len(ranks) != 408:
        raise ValueError("Actual finite rank keys do not cover the authorized 408 rows")
    score_receipt = json.loads((score_dir / "finite_execution_receipt.json").read_text())
    score_path = score_dir / "independent_score_rows_enriched.jsonl.gz"
    previous_refs = score_dir / "reference_G.jsonl"
    score_sha = digest(score_path)
    if (not score_receipt["passed"] or score_receipt["canonical_pending"] or score_receipt["abstain_pending"]
            or score_receipt["literal_pending"] or score_sha != score_receipt["outputs"][score_path.name]
            or digest(previous_refs) != score_receipt["outputs"][previous_refs.name]):
        raise ValueError("Independent attempt source is not actually accepted, closed and byte-bound")
    scored = {r["key"]: r for _line, r, _sha in rows(score_path)}
    authority = within(ROOT, DEFAULT_AUTHORITY_SCORE)
    authority_receipt = json.loads((authority / "execution_receipt.json").read_text())
    historical_ref_path = authority / "reference_G.jsonl"
    historical_ref_sha = digest(historical_ref_path)
    if historical_ref_sha != authority_receipt["reference_G_sha256"]:
        raise ValueError("Already accepted historical reference metadata bytes differ")
    historical = {r["sample_id"]: r["historical_accepted_reference"] for _line, r, _sha in rows(historical_ref_path) if r["model"] == MODEL}
    references, attempt_keys = [], set()
    for _line, ref, _sha in rows(previous_refs):
        if ref["model"] != MODEL or ref["gold_rank"] is not None or ref["reference_G"] is not None:
            raise ValueError("The old finite attempt table unexpectedly already includes ranks or Boolean GT")
        for attempt in ref["attempts"]:
            row = scored[attempt["key"]]
            fields = ("replicate", "canonical_name_in_primary_score", "abstain", "seed", "source_identity", "raw_line_sha256")
            if any(attempt[name] != row[name] for name in fields) or attempt["key"] in attempt_keys or row["sample_id"] != ref["sample_id"]:
                raise ValueError("Reference attempt members do not retain actual score, ordinal, seed or identity")
            attempt_keys.add(attempt["key"])
        all_ten = {a["replicate"] for a in ref["attempts"]} == set(range(10))
        resolved = all_ten and all(a["canonical_name_in_primary_score"] is not None for a in ref["attempts"])
        correct = sum(a["canonical_name_in_primary_score"] == 1 for a in ref["attempts"]) if resolved else None
        if correct != ref["correct_count"]:
            raise ValueError("Actual current-canonical ten-attempt correct count differs")
        rank = ranks.get(ref["sample_id"])
        complete = rank is not None and resolved
        references.append({**ref, "gold_rank": rank["gold_rank"] if rank else None, "rank_source": rank,
            "reference_G": rank["gold_rank"] > 1 and correct == 0 if complete else None, "reference_complete": complete,
            "historical_accepted_reference": historical.get(ref["sample_id"]),
            "historical_accepted_reference_source_path": str(historical_ref_path.relative_to(ROOT)),
            "historical_accepted_reference_source_sha256": historical_ref_sha,
            "current_attempt_score_source_path": str(score_path.relative_to(ROOT)), "current_attempt_score_source_sha256": score_sha,
            "reference_missing_reason": None if complete else "rank_missing" if rank is None and resolved else "rank_and_attempts_missing" if rank is None else "fewer_than_ten_or_unresolved_attempts"})
    if len(references) != 4848 or len({r["sample_id"] for r in references}) != 4848 or attempt_keys != set(scored):
        raise ValueError("Finite rank/reference join changed registered sample or attempt-key coverage")
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "rank_rows_source_bound.jsonl.gz", list(ranks.values()))
    save_rows(output / "reference_G.jsonl", references)
    write_json(output / "received_source_verification.json", {"schema": "kdm_remaining4_finite_rank_source_validation_v1", "passed": True,
        "rows": len(ranks), "parts": proven, "source_received_manifest_sha256": digest(received_path), "duplicate_keys": 0,
        "registered_full_rank_rows": 4848, "missing_rank_rows": 4440, "GPU_initialized": False})
    coverage = [{"split": split, "registered_samples": 2424, "rank_received": sum(r["split"] == split for r in ranks.values()),
        "all_ten_correctness_closed": sum(r["split"] == split and r["correct_count"] is not None for r in references),
        "reference_complete": sum(r["split"] == split and r["reference_complete"] for r in references),
        "reference_positive_complete": sum(r["split"] == split and r["reference_G"] is True for r in references),
        "reference_pending": sum(r["split"] == split and not r["reference_complete"] for r in references)} for split in ("dev", "eval")]
    write_json(output / "receipt.json", {"schema": "kdm_remaining4_finite_rank_reference_join_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "rank_rows": 408, "attempt_rows": len(scored),
        "reference_rows": 4848, "coverage": coverage, "reference_complete": sum(r["reference_complete"] for r in references),
        "reference_pending": sum(not r["reference_complete"] for r in references),
        "source_attempt_receipt_sha256": digest(score_dir / "finite_execution_receipt.json"), "source_received_manifest_sha256": digest(received_path),
        "reference_rule": "registered gold_rank > 1 and all ten current canonical correct attempts = 0",
        "historical_reference_nonnull": sum(r["historical_accepted_reference"] is not None for r in references),
        "source_scorer_frozen": True, "old_raw_reopened": False, "GPU_initialized": False, "new_generations": 0, "new_API_calls": 0,
        "connector_sha256": digest(Path(__file__)), "outputs": {p.name: digest(p) for p in output.iterdir()}})
    print(json.dumps({"rank_rows": 408, "attempt_rows": len(scored), "coverage": coverage}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("receive", "connect"))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--central-relative")
    parser.add_argument("--received-manifest", type=Path)
    parser.add_argument("--score-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "receive":
        if not all((args.snapshot, args.directory, args.central_relative)):
            parser.error("Explicit old sealed snapshot, exclusive local directory and central-relative path are required")
        receive(args.snapshot, args.directory, args.central_relative)
    else:
        if not all((args.received_manifest, args.score_dir, args.output)):
            parser.error("Actual received rank manifest, accepted finite attempt scores and exclusive output are required")
        connect(args.received_manifest, args.score_dir, args.output)


if __name__ == "__main__":
    main()
