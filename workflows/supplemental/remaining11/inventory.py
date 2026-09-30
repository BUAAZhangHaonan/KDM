"""Recover indexed historical assets and audit their actual keys and sources."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


MODELS = (
    "gemma3_12b", "glm46v", "internvl35_8b", "llava15_7b", "onevision",
    "minicpm45", "phi35", "qwen3vl", "qwen35_9b", "llava15_13b",
    "llava16_vicuna",
)
CENSUS_PREFIX = (
    "outputs/annotations/luna_census_remaining_v1/"
    "remaining108_plus2_20260924/census_merged_v1/"
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def iter_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                yield line_number, json.loads(line)


def source_model(path: str) -> str | None:
    for model in sorted(MODELS, key=len, reverse=True):
        if model in path:
            return model
    aliases = {"llava15_": "llava15_7b", "vicuna_": "llava16_vicuna"}
    for name, model in aliases.items():
        if name in path:
            return model
    return None


def recover(args: argparse.Namespace) -> None:
    run = args.project / args.run
    recovered = run / "recovered"
    recovered.mkdir(parents=True, exist_ok=False)
    assets = run / "assets"
    with args.index.open(encoding="utf-8-sig", newline="") as source:
        indexed = list(csv.DictReader(source))
    selected = {
        row["member_path"]: row
        for row in indexed
        if "remaining11_sync" not in row["member_path"]
        and not row["member_path"].endswith((".tar", ".tar.gz"))
    }
    for name in ("labels.jsonl", "identity.json", "merge_receipt.json", "metrics.json",
                 "labels_delta.jsonl", "unresolved.jsonl"):
        selected[CENSUS_PREFIX + name] = {"member_path": CENSUS_PREFIX + name}
    rows = []
    with tarfile.open(args.archive, "r:") as archive:
        for name, provenance in selected.items():
            parts = PurePosixPath(name).parts
            if PurePosixPath(name).is_absolute() or ".." in parts:
                raise ValueError(f"Unsafe archive member: {name}")
            member = archive.getmember(name)
            if not member.isfile():
                raise ValueError(f"Archive member is not a regular file: {name}")
            if "bytes" in provenance and member.size != int(provenance["bytes"]):
                raise ValueError(f"Archive member size conflicts with index: {name}")
            destination = recovered.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable archive member: {name}")
            with stream, destination.open("xb") as target:
                shutil.copyfileobj(stream, target, 1024 * 1024)
            rows.append({
                "source_container": str(args.archive), "source_member": name,
                "recovered_path": str(destination), "bytes": member.size,
                "existing_recorded_sha256": provenance.get("existing_recorded_sha256", ""),
            })
            if destination.suffix == ".jsonl" and name.startswith("outputs/raw/"):
                first = next(iter_rows(destination), None)
                if first:
                    row = first[1]
                    print(json.dumps({"event": "raw_recovered", "member": name,
                                      "fields": list(row), "sample": row},
                                     ensure_ascii=False)[:4000], flush=True)
    write_json(assets / "recovery_manifest.json", {"files": rows, "files_recovered": len(rows)})
    print(json.dumps({"event": "recovery_complete", "files": len(rows),
                      "bytes": sum(row["bytes"] for row in rows)}), flush=True)


def recover_evidence(args: argparse.Namespace) -> None:
    run = args.project / args.run
    archive_manifest = read_json(args.archive.with_name("historical_outputs_manifest.json"))
    selected = []
    for row in archive_manifest["files"]:
        name = row["path"]
        destination = run / "recovered" / name
        if destination.exists() or name.endswith((".tar", ".tar.gz")):
            continue
        if (name.startswith("outputs/records/remaining11_glm_independent_v1/")
                or ("remaining11_sync" in name and name.endswith(".json"))):
            selected.append(row)
    copied = []
    with tarfile.open(args.archive, "r:") as archive:
        for row in selected:
            name = row["path"]
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise ValueError(f"Unsafe archive member: {name}")
            member = archive.getmember(name)
            if not member.isfile():
                raise ValueError(f"Not a file: {name}")
            destination = run / "recovered" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable member: {name}")
            with stream, destination.open("xb") as target:
                shutil.copyfileobj(stream, target, 1024 * 1024)
            copied.append({"source_container": str(args.archive), "source_member": name,
                           "recovered_path": str(destination), "bytes": member.size,
                           "existing_recorded_sha256": row.get("sha256", "")})
    write_json(run / "assets" / "evidence_recovery_manifest.json", {"files": copied})
    raw_paths = [row for row in archive_manifest["files"]
                 if row["path"].startswith("outputs/raw/")]
    write_json(run / "assets" / "historical_raw_index.json", {"files": raw_paths})
    print(json.dumps({"event": "evidence_recovery_complete", "files": len(copied),
                      "historical_raw_paths": len(raw_paths)}), flush=True)


def independent_task(sample: dict[str, Any], replicate: int) -> dict[str, Any]:
    return {"sample": sample, "method": "direct", "marker": "UNKNOWN",
            "reference_marker": "UNKNOWN", "guided": False,
            "reference_guided": False, "attempt": True, "replicate": replicate,
            "kind": "independent_attempt"}


def audit(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.project / "src"))
    from kdm.io import stable_hash, stable_seed
    from kdm.decoding import DecodeConfig
    from kdm.pipeline import experiment_tasks, task_id
    from kdm.prompts import closed_prompt, task_prompt

    run = args.project / args.run
    assets = run / "assets"
    recovered = run / "recovered"
    samples = [row for _, row in iter_rows(args.project / "data/current/all.jsonl")]
    sample_map = {row["id"]: row for row in samples}
    if len(sample_map) != 9167:
        raise ValueError("Original sample inventory is not 9167 unique records")
    food = [row for row in samples if row["dataset"] == "food101"]
    names = sorted({row["class"] for row in food})
    quotas = Counter((row["split"], row["class"]) for row in food)
    if len(food) != 4848 or len(names) != 101 or set(quotas.values()) != {24}:
        raise ValueError("Food-101 source split or category quotas differ")
    method_plan = read_json(args.project / "configs/kdm/method_plan.json")
    recovery = read_json(assets / "recovery_manifest.json")
    recovery_map = {row["recovered_path"]: row for row in recovery["files"]}
    sources = []
    source_keys = {}
    source_key_sample = {}
    source_key_payload = {}
    raw_paths = sorted((recovered / "outputs/raw").rglob("*.jsonl"))
    for path in raw_paths:
        if path.name not in {"independent.jsonl", "closed.jsonl"}:
            continue
        source_id = stable_hash(str(path.relative_to(recovered)))[:16]
        stage = "candidate" if path.name == "closed.jsonl" else "independent"
        sidecar_path = path.with_suffix(".identity.json")
        sidecar = read_json(sidecar_path)
        definition = sidecar["definition"]
        identity = sidecar["identity"]
        if stable_hash(definition) != identity:
            raise ValueError(f"Invalid source sidecar identity: {path}")
        model = definition["model"]
        if model not in MODELS:
            continue
        keys = set()
        key_sample = {}
        key_payload = {}
        counts = Counter()
        splits = Counter()
        sample_attempts = defaultdict(set)
        errors = []
        configs = Counter()
        first_row = None
        for line_number, row in iter_rows(path):
            first_row = first_row or row
            counts["rows"] += 1
            sample = row["sample"]
            sid = sample["id"]
            if sid not in sample_map or any(sample[field] != sample_map[sid][field]
                for field in ("dataset", "question", "gold", "split", "cluster")):
                errors.append({"line": line_number, "error": "sample_binding_mismatch"})
                continue
            if sample["dataset"] != "food101":
                counts["outside_food101"] += 1
                continue
            if row.get("model") != model or row.get("identity") != identity:
                errors.append({"line": line_number, "error": "identity_model_mismatch"})
            if row.get("status", "ok") != "ok":
                errors.append({"line": line_number, "error": "non_success_raw"})
            if stage == "candidate":
                expected_key = stable_hash([model, sid, "closed"])
                candidate_scores = row["candidate_scores"]
                labels = [score["label"] for score in candidate_scores]
                if len(labels) != 101 or set(labels) != set(names):
                    errors.append({"line": line_number, "error": "candidate_label_coverage"})
                finite = all(isinstance(score["mean_logp"], (int, float))
                    and math.isfinite(score["mean_logp"])
                    and math.isfinite(score["sum_logp"])
                    and score["n_tokens"] > 0 for score in candidate_scores)
                if not finite:
                    errors.append({"line": line_number, "error": "nonfinite_candidate_probability"})
                target = row["target"]
                if target != sample["gold"][0]:
                    errors.append({"line": line_number, "error": "candidate_target_mismatch"})
                target_score = next(score["mean_logp"] for score in candidate_scores
                                    if score["label"] == target)
                rank = 1 + sum(score["mean_logp"] > target_score for score in candidate_scores)
                if row["gold_rank"] != rank:
                    errors.append({"line": line_number, "error": "strict_greater_rank_mismatch"})
                if row["prompt"] != closed_prompt(sample["question"], labels):
                    errors.append({"line": line_number, "error": "closed_prompt_mismatch"})
                counts["finite_candidate_rows"] += int(finite)
                counts["rank1"] += int(rank == 1)
                payload = stable_hash(candidate_scores)
            else:
                replicate = row["replicate"]
                task = independent_task(sample, replicate)
                expected_key = task_id(model, task)
                if row.get("seed") != stable_seed(sid, model, replicate):
                    errors.append({"line": line_number, "error": "stable_seed_mismatch"})
                if row["prompt"] != task_prompt(sample["question"], attempt=True):
                    errors.append({"line": line_number, "error": "independent_prompt_mismatch"})
                if not 0 <= replicate < 10:
                    errors.append({"line": line_number, "error": "attempt_replicate_range"})
                sample_attempts[sid].add(replicate)
                config = row["config"]
                if (config["max_tokens"], config["temperature"], config["top_p"]) != (32, 1.0, 1.0):
                    errors.append({"line": line_number, "error": "independent_registered_config_mismatch"})
                configs[stable_hash(config)] += 1
                probabilities = row.get("selected_log_probabilities", [])
                if any(not math.isfinite(value) for value in probabilities):
                    errors.append({"line": line_number, "error": "nonfinite_selected_probability"})
                probability = row.get("first_probability")
                if probability is not None and (not math.isfinite(probability) or not 0 <= probability <= 1):
                    errors.append({"line": line_number, "error": "invalid_first_probability"})
                counts["terminated"] += int(row.get("terminated") is True)
                counts["token_budget_exhausted"] += int(row.get("terminated") is False)
                counts["empty_text"] += int(not row.get("text", "").strip())
                payload = stable_hash({"text": row["text"], "seed": row["seed"], "config": config})
            if row["key"] != expected_key:
                errors.append({"line": line_number, "error": "original_task_key_mismatch"})
            if row["key"] in keys:
                counts["duplicate_keys"] += 1
            keys.add(row["key"])
            key_sample[row["key"]] = sid
            key_payload[row["key"]] = payload
            splits[sample["split"]] += 1
        with path.open("rb") as source:
            source.seek(-1, 2)
            trailing_newline = source.read(1) == b"\n"
        metadata = {"source_id": source_id, "model": model, "stage": stage,
                    "path": str(path), "identity_path": str(sidecar_path), "identity": identity,
                    "source_definition": definition,
                    "source_container": str(args.archive),
                    "source_member": str(path.relative_to(recovered)),
                    "existing_recorded_sha256": recovery_map[str(path)].get("existing_recorded_sha256", ""),
                    "rows": counts["rows"], "food_rows": len(keys), "unique_keys": len(keys),
                    "coverage_by_split": dict(splits), "counts": dict(counts),
                    "trailing_newline": trailing_newline,
                    "raw_fields": list(first_row) if first_row else [],
                    "raw_config_variants": dict(configs),
                    "attempt_count_distribution": dict(Counter(map(len, sample_attempts.values()))),
                    "cohort": first_row.get("sampling_backend", definition["schema"]) if first_row else definition["schema"],
                    "validation_error_count": len(errors), "validation_errors": errors[:20],
                    "status": "source_validated" if not errors and not counts["duplicate_keys"] else "hold_source_validation"}
        write_json(assets / "source_audits" / f"{model}_{stage}_{source_id}.json", metadata)
        sources.append(metadata)
        source_keys[source_id] = keys
        source_key_sample[source_id] = key_sample
        source_key_payload[source_id] = key_payload
        print(json.dumps({"event": "source_audited", "model": model, "stage": stage,
                          "rows": len(keys), "errors": len(errors), "path": str(path)},
                         ensure_ascii=False), flush=True)

    census_labels_path = recovered / CENSUS_PREFIX / "labels.jsonl"
    census = {}
    census_counts = Counter()
    for _, row in iter_rows(census_labels_path):
        if row["key"] in census:
            raise ValueError("Duplicate final census label key")
        census[row["key"]] = (row["model"], row["sample_id"], row["text"],
                               row["raw_identity"], row["label"])
        census_counts[row["label"]] += 1
    if len(census) != 293344 or "unresolved" in census_counts:
        raise ValueError("Final census label closure is incomplete")
    inclusion = []
    census_sources = []
    for model in MODELS:
        path = args.project / "data/responses/census" / f"{model}.jsonl.gz"
        sidecar = read_json(path.with_name(f"{model}.identity.json"))
        if stable_hash(sidecar["definition"]) != sidecar["identity"]:
            raise ValueError(f"Census sidecar identity mismatch: {model}")
        runtime = read_json(args.project / "configs/runtime" / f"{model}.json")
        backend = sidecar["definition"]["backend"]
        runtime_fields = ("factory", "dtype", "hf_model_id", "thinking_mode")
        runtime_differences = {field: {"source": backend.get(field), "current": runtime.get(field)}
                               for field in runtime_fields if backend.get(field) != runtime.get(field)}
        for field in ("dtype", "model_path"):
            if backend.get("kwargs", {}).get(field) != runtime.get("kwargs", {}).get(field):
                runtime_differences[f"kwargs.{field}"] = {
                    "source": backend.get("kwargs", {}).get(field),
                    "current": runtime.get("kwargs", {}).get(field)}
        weights = lambda value: {(item["filename"], item.get("hub_recorded_sha256"))
                                 for item in value.get("weights", [])}
        if weights(backend) != weights(runtime):
            runtime_differences["weights"] = {"source": sorted(weights(backend)), "current": sorted(weights(runtime))}
        if backend.get("processor", {}).get("files") != runtime.get("processor", {}).get("files"):
            runtime_differences["processor_files"] = {
                "source": backend.get("processor", {}).get("files"),
                "current": runtime.get("processor", {}).get("files")}
        derived_definition = {"schema": "kdm_census_to_formal_direct_source_mapping_v1",
                              "model": model, "backend": backend,
                              "source_path": str(path), "source_identity": sidecar["identity"],
                              "source_sidecar_path": str(path.with_name(f"{model}.identity.json")),
                              "source_kind": "census", "derived_kind": "main",
                              "rule": "exact guided UNKNOWN Food-101 eval response; only task kind and task key change",
                              "runtime_differences": runtime_differences}
        derived_identity = stable_hash(derived_definition)
        derived_path = assets / "derived_formal" / f"{model}_direct_unknown.jsonl"
        derived_path.parent.mkdir(parents=True, exist_ok=True)
        derived_keys = set()
        derived_key_sample = {}
        derived_key_payload = {}
        derived_errors = []
        rows = Counter()
        abstains = Counter()
        keys = set()
        identities = set()
        with gzip.open(path, "rb") as source, derived_path.open("x", encoding="utf-8") as target:
            for line_number, raw_line in enumerate(source, 1):
                row = json.loads(raw_line)
                key = row["key"]
                if key in keys:
                    raise ValueError(f"Duplicate census source key: {model}")
                keys.add(key)
                sid = row["sample"]["id"]
                actual = (model, sid, row["text"], row["identity"])
                label = census[key]
                if actual != label[:4]:
                    raise ValueError(f"Final census source binding mismatch: {model} {key}")
                identities.add(row["identity"])
                if row["guided"]:
                    dataset = row["sample"]["dataset"]
                    rows[dataset] += 1
                    abstains[dataset] += int(label[4] == "abstain")
                if not (row["guided"] and row["sample"]["dataset"] == "food101"
                        and row["sample"]["split"] == "eval"):
                    continue
                sample = row["sample"]
                task = {"sample": sample, "method": "direct", "marker": "UNKNOWN",
                        "reference_marker": "UNKNOWN", "guided": True,
                        "reference_guided": True, "replicate": 0, "kind": "main"}
                original_task = {**task, "kind": "census"}
                if row["key"] != task_id(model, original_task):
                    derived_errors.append({"line": line_number, "error": "census_original_key_mismatch"})
                if sample != sample_map[sid]:
                    derived_errors.append({"line": line_number, "error": "census_full_sample_binding_mismatch"})
                if row["identity"] != sidecar["identity"]:
                    derived_errors.append({"line": line_number, "error": "census_raw_identity_mismatch"})
                if row["config"] != asdict(DecodeConfig()):
                    derived_errors.append({"line": line_number, "error": "census_full_decode_config_mismatch"})
                if row["prompt"] != task_prompt(sample["question"], "UNKNOWN", True):
                    derived_errors.append({"line": line_number, "error": "census_prompt_mismatch"})
                if row["seed"] != stable_seed(sid, model, 0):
                    derived_errors.append({"line": line_number, "error": "census_seed_mismatch"})
                if (not isinstance(row.get("terminated"), bool)
                        or len(row["tokens"]) > 32
                        or len(row["tokens"]) != len(row["selected_log_probabilities"])
                        or any(not math.isfinite(value) for value in row["selected_log_probabilities"])):
                    derived_errors.append({"line": line_number, "error": "census_termination_or_probability_mismatch"})
                if runtime_differences:
                    derived_errors.append({"line": line_number, "error": "census_runtime_identity_difference"})
                formal_key = task_id(model, task)
                derived = {**row, "kind": "main", "key": formal_key,
                           "identity": derived_identity, "original_key": key,
                           "source_identity": row["identity"], "source_kind": "census",
                           "source_path": str(path), "source_line": line_number,
                           "source_row_sha256": hashlib.sha256(raw_line).hexdigest(),
                           "source_row_sha256_encoding": "exact decompressed UTF-8 line including original newline",
                           "derivation": "census_guided_unknown_eval_to_formal_direct"}
                target.write(json.dumps(derived, ensure_ascii=False, allow_nan=False) + "\n")
                derived_keys.add(formal_key)
                derived_key_sample[formal_key] = sid
                derived_key_payload[formal_key] = stable_hash({"text": row["text"], "seed": row["seed"], "config": row["config"]})
        if len(keys) != 18334:
            raise ValueError(f"Census source row count mismatch: {model}")
        if len(derived_keys) != 2424:
            raise ValueError(f"Census formal mapping coverage mismatch: {model}")
        derived_sidecar_path = derived_path.with_suffix(".identity.json")
        write_json(derived_sidecar_path, {"identity": derived_identity, "definition": derived_definition})
        source_id = stable_hash(str(derived_path))[:16]
        sources.append({"source_id": source_id, "model": model, "stage": "formal",
                        "path": str(derived_path), "identity_path": str(derived_sidecar_path),
                        "identity": derived_identity, "source_definition": derived_definition,
                        "source_path": str(path), "source_identity": sidecar["identity"],
                        "rows": len(derived_keys), "unique_keys": len(derived_keys),
                        "coverage_by_split": {"eval": len(derived_keys)},
                        "validation_error_count": len(derived_errors),
                        "validation_errors": derived_errors[:20], "cohort": backend["factory"],
                        "runtime_differences": runtime_differences,
                        "status": "source_validated" if not derived_errors else "hold_source_validation"})
        source_keys[source_id] = derived_keys
        source_key_sample[source_id] = derived_key_sample
        source_key_payload[source_id] = derived_key_payload
        census_sources.append({"model": model, "path": str(path), "rows": len(keys),
                               "identity_path": str(path.with_name(f"{model}.identity.json")),
                               "raw_identities": sorted(identities), "sidecar": sidecar})
        for dataset, n in rows.items():
            inclusion.append({"model": model, "dataset": dataset, "guided_census_rows": n,
                              "confirmed_abstentions": abstains[dataset],
                              "selected": abstains[dataset] > 0})
    write_json(assets / "registered_inclusion.json", {"rows": inclusion})

    historical_labels = []
    for path in sorted((recovered / "outputs/annotations/remaining11_v1").glob("*/labels.jsonl")):
        count = Counter()
        model = None
        identities = set()
        keys = set()
        for _, row in iter_rows(path):
            model = row["model"]
            if row["key"] in keys:
                raise ValueError(f"Duplicate preliminary annotation key: {path}")
            keys.add(row["key"])
            count[row["screening_label"]] += 1
            identities.add(row["source_identity"])
        historical_labels.append({"model": model, "stage": "independent", "path": str(path),
                                  "rows": len(keys), "screening_counts": dict(count),
                                  "source_identities": sorted(identities),
                                  "status": "preliminary_retained_current_scoring_required"})

    model_stages = {}
    gaps = []
    for model in MODELS:
        model_stages[model] = {}
        for stage in ("candidate", "independent", "formal"):
            selected = set()
            stage_sources = [source for source in sources
                             if source["model"] == model and source["stage"] == stage
                             and source["status"] == "source_validated"]
            stage_sources.sort(key=lambda source: (source["unique_keys"], source["path"]), reverse=True)
            selected_source_ids = []
            overlapping_payload_conflicts = 0
            chosen_payload = {}
            split_counts = Counter()
            for source in stage_sources:
                source_id = source["source_id"]
                own = source_keys[source_id] - selected
                for key in source_keys[source_id] & selected:
                    if chosen_payload[key] != source_key_payload[source_id][key]:
                        overlapping_payload_conflicts += 1
                source["selected_rows"] = len(own)
                source["unselected_overlap_rows"] = len(source_keys[source_id] & selected)
                selected.update(own)
                chosen_payload.update({key: source_key_payload[source_id][key] for key in own})
                if own:
                    selected_source_ids.append(source_id)
                    keys_path = assets / "selected_keys" / f"{model}_{stage}_{source_id}.jsonl"
                    keys_path.parent.mkdir(parents=True, exist_ok=True)
                    with keys_path.open("x", encoding="utf-8") as target:
                        for key in sorted(own):
                            sid = source_key_sample[source_id][key]
                            split = sample_map[sid]["split"]
                            split_counts[split] += 1
                            target.write(json.dumps({"key": key, "sample_id": sid, "split": split}) + "\n")
                    source["usable_keys_file"] = str(keys_path)
                    source["status"] = "selected_complete" if len(own) == (4848 if stage == "candidate" else 48480) else "selected_partial"
                else:
                    source["status"] = "superseded_overlap_source"
            remaining_path = assets / "remaining_keys" / f"{model}_{stage}.jsonl"
            remaining_path.parent.mkdir(parents=True, exist_ok=True)
            expected = 0
            remaining = 0
            with remaining_path.open("x", encoding="utf-8") as target:
                tasks = (experiment_tasks(food, methods=method_plan[model]["food101"])
                         if stage == "formal" else
                         (independent_task(sample, replicate) for sample in food for replicate in range(10))
                         if stage == "independent" else ({"sample": sample} for sample in food))
                for task in tasks:
                    sample = task["sample"]
                    key = (stable_hash([model, sample["id"], "closed"])
                           if stage == "candidate" else task_id(model, task))
                    expected += 1
                    if key not in selected:
                        remaining += 1
                        row = {"key": key, "model": model, "stage": stage,
                               "sample_id": sample["id"], "split": sample["split"],
                               **{field: value for field, value in task.items() if field != "sample"}}
                        target.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
            state = {"expected_rows": expected, "reusable_rows": len(selected),
                     "coverage_by_split": dict(split_counts), "remaining_rows": remaining,
                     "sources": selected_source_ids, "remaining_keys_file": str(remaining_path),
                     "overlapping_payload_conflicts_preserved": overlapping_payload_conflicts,
                     "owner": "unclaimed_at_inventory", "current_primary_scoring_complete": False,
                     "status": "raw_complete_scoring_pending" if remaining == 0 else "raw_partial" if selected else "raw_missing"}
            if expected != len(selected) + remaining:
                raise ValueError(f"Task coverage mismatch: {model} {stage}")
            model_stages[model][stage] = state
            gaps.append({"model": model, "dataset": "food101", "stage": stage,
                         "expected_rows": expected, "reusable_rows": len(selected),
                         "remaining_rows": remaining, "status": state["status"],
                         "source_ids": "|".join(selected_source_ids),
                         "remaining_keys_file": str(remaining_path),
                         "current_primary_scoring_complete": False})
            print(json.dumps({"event": "stage_gap_written", "model": model, "stage": stage,
                              "reusable": len(selected), "remaining": remaining}), flush=True)
    failure_evidence = []
    for path in sorted((recovered / "outputs/records").rglob("*.json")):
        if path.name in {"error.json", "entry_error.json", "root_actual_review_hold_v1.json"}:
            failure_evidence.append({"path": str(path), "model": source_model(str(path)),
                                     "evidence": read_json(path)})
    manifest = {"schema": "kdm_remaining11_asset_manifest_v1", "project": str(args.project),
                "run": str(args.run), "recovered_root": str(recovered), "models": model_stages,
                "sources": sources, "historical_labels": historical_labels,
                "census_final_labels": {"path": str(census_labels_path), "rows": len(census),
                                        "counts": dict(census_counts), "unresolved": 0},
                "census_sources": census_sources, "registered_inclusion": inclusion,
                "failure_evidence": failure_evidence,
                "source_selection_rule": "largest complete source first; preserve superseded source and cohort; one source per original task key",
                "audit_scope": "raw row keys, identities, sample bindings, candidate counts/ranks, finite probabilities, splits, attempts and source coverage; inherited weight identities reused",
                "formal_reuse": "explicit source-bound census-to-formal Direct UNKNOWN mappings, original key and identity retained; only validated source/backend records count as reusable",
                "vizwiz_queue": [{**row, "stage": "independent_registered_queue",
                                  "existing_probe_sources_audited": False} for row in inclusion if row["dataset"] == "vizwiz"]}
    write_json(assets / "asset_manifest.json", manifest)
    with (assets / "asset_gaps.csv").open("x", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(gaps[0]))
        writer.writeheader()
        writer.writerows(gaps)
    print(json.dumps({"event": "asset_audit_complete", "raw_sources": len(sources),
                      "rows_reusable": sum(row["reusable_rows"] for row in gaps),
                      "remaining_food_rows": sum(row["remaining_rows"] for row in gaps),
                      "registered_inclusion": inclusion}, ensure_ascii=False), flush=True)


def verify_receipts(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.project / "src"))
    from kdm.io import stable_hash

    assets = args.project / args.run / "assets"
    manifest = read_json(assets / "asset_manifest.json")
    by_identity = {source["identity"]: source for source in manifest["sources"]}
    label_checks = []
    for label_source in manifest["historical_labels"]:
        labels = {row["key"]: row for _, row in iter_rows(Path(label_source["path"]))}
        source_identity = label_source["source_identities"]
        if len(source_identity) != 1 or source_identity[0] not in by_identity:
            raise ValueError(f"Preliminary label source identity is not in audited raw: {label_source['path']}")
        raw_source = by_identity[source_identity[0]]
        count = 0
        hashes = Counter()
        with Path(raw_source["path"]).open("rb") as source:
            for line_number, raw_line in enumerate(source, 1):
                row = json.loads(raw_line)
                label = labels[row["key"]]
                sample = row["sample"]
                if any(label[field] != value for field, value in {
                    "model": row["model"], "sample_id": sample["id"],
                    "split": sample["split"], "replicate": row["replicate"],
                    "seed": row["seed"], "text": row["text"],
                    "source_identity": row["identity"], "source_line": line_number,
                }.items()):
                    raise ValueError(f"Preliminary label raw binding mismatch: {label_source['path']}:{line_number}")
                expected_sha = label["source_row_sha256"]
                methods = {
                    "exact_line_including_newline": hashlib.sha256(raw_line).hexdigest(),
                    "exact_line_without_newline": hashlib.sha256(raw_line.rstrip(b"\r\n")).hexdigest(),
                    "stable_hash_json_object": stable_hash(row),
                }
                matched = [name for name, value in methods.items() if value == expected_sha]
                if not matched:
                    raise ValueError(f"Preliminary label recorded row SHA is not bound to raw: {label_source['path']}:{line_number}")
                hashes["|".join(matched)] += 1
                count += 1
        if count != len(labels):
            raise ValueError(f"Preliminary label extra or missing keys: {label_source['path']}")
        label_checks.append({"model": label_source["model"], "labels": count,
                             "source_identity": source_identity[0], "source_path": raw_source["path"],
                             "recorded_row_sha_algorithms": dict(hashes),
                             "source_bound": True, "current_primary_score_reuse": False})
        print(json.dumps({"event": "historical_labels_source_verified", **label_checks[-1]}), flush=True)
    census_checks = []
    for source in manifest["census_sources"]:
        backend = source["sidecar"]["definition"]["backend"]
        runtime = read_json(args.project / "configs/runtime" / f"{source['model']}.json")
        differences = {field: {"source": backend.get(field), "current": runtime.get(field)}
                       for field in ("versions", "thinking_mode", "input_limits")
                       if backend.get(field) != runtime.get(field)}
        if differences:
            raise ValueError(f"Census runtime fields differ: {source['model']} {differences}")
        census_checks.append({"model": source["model"], "runtime_versions_equal": True,
                              "thinking_mode_equal": True, "input_limits_equal": True})
    recovered = Path(manifest["recovered_root"])
    refs = (
        "outputs/records/remaining11_onevision_v1/independent_native16/reference.json",
        "outputs/records/remaining11_minicpm45_v2/independent_native16/reference.json",
        "outputs/records/independent_k100_qwen35_9b_v1/native16/reference.json",
        "outputs/records/independent_k100_qwen35_9b_v1/engine16/check.json",
        "outputs/records/independent_k100_onevision_v1/engine16/check.json",
        "outputs/records/remaining11_closed_v1/llava16_vicuna_v2/production_identity.json",
    )
    backend_evidence = []
    for member in refs:
        path = recovered / member
        record = read_json(path)
        definition = record.get("definition", record)
        evidence = {field: definition[field] for field in (
            "schema", "model", "backend", "production_spec", "backend_spec",
            "versions", "checkpoint_path", "actual_processor", "execution",
            "native_reference_sha256", "eligible", "completed", "expected",
        ) if field in definition}
        backend_evidence.append({"source_path": str(path), "evidence": evidence})
    write_json(assets / "source_binding_verification.json", {
        "schema": "kdm_remaining11_existing_source_binding_verification_v1",
        "historical_label_checks": label_checks, "census_runtime_checks": census_checks,
        "backend_reference_evidence": backend_evidence,
        "K100_current_targeted_cpu_observation": {
            "checked_date": "2026-09-30", "host": "RTX_Pro_6000",
            "environment": "/home/k100/projects/knowledge-deficit-mitigation/.environments/vicuna_native_tf553",
            "python": "3.11.14", "transformers_package_metadata": "5.5.3",
            "pth": "readonly_vllm_torch_runtime.pth",
            "pth_target": "/home/k100/projects/knowledge-deficit-mitigation/.environments/vllm024/lib/python3.12/site-packages",
            "pth_target_exists": False, "torch_metadata_error": "PackageNotFoundError",
            "probe_scope": "read-only filesystem and importlib.metadata; no GPU/model loading",
        },
    })
    print(json.dumps({"event": "source_binding_verification_complete",
                      "historical_labels": sum(row["labels"] for row in label_checks),
                      "census_runtime_models": len(census_checks)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("recover", "recover-evidence", "audit", "verify-receipts"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path(
        "/home/g203-4028/projects/knowledge-deficit-mitigation-archive/20260929/"
        "historical_outputs.tar"))
    args = parser.parse_args()
    {"recover": recover, "recover-evidence": recover_evidence, "audit": audit,
     "verify-receipts": verify_receipts}[args.action](args)


if __name__ == "__main__":
    main()
