#!/usr/bin/env python3
"""Generate explicit remaining-model gaps with the registered algorithms.

CPU --check-plan reads the original input/proof identities and expands tasks.
GPU --execute creates an exclusive claim and immutable completed raw parts.
Use each model's registered Python and the supplemental execution worker.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.decoding import DecodeConfig
from kdm.frozen import canonical_runtime_spec, load_contract
from kdm.io import Ledger, atomic_json, file_hash, read_jsonl, stable_hash, stable_seed, within
from kdm.pipeline import closed_rank, experiment_tasks, make_backend, run_tasks, task_id
from kdm.prompts import MARKERS
from kdm.protocol import validate_method_runtime, validate_native_runtime_files, validate_resource_runtime

MODELS = (
    "gemma3_12b", "glm46v", "internvl35_8b", "llava15_7b", "onevision",
    "minicpm45", "phi35", "qwen3vl", "qwen35_9b", "llava15_13b",
    "llava16_vicuna",
)
STAGES = ("formal", "independent", "candidate")
FORMAL_ORDER = ("unknown_main", "unknown_controls", "prompt_matrix")
TASK_FIELDS = (
    "method", "marker", "reference_marker", "guided", "reference_guided",
    "replicate", "kind",
)
GENERATION_CONFIG = "workflows/supplemental/remaining11/generation.json"
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def formal_stage(task: dict) -> str:
    if task["marker"] == "UNKNOWN" and task["reference_marker"] == "UNKNOWN":
        return "unknown_main" if task["kind"] == "main" else "unknown_controls"
    return "prompt_matrix"


def owns(sample: dict, shard: int, n_shards: int) -> bool:
    return int(stable_hash(sample["id"])[:8], 16) % n_shards == shard


def ordered_tasks(samples: list[dict], methods: list[str], stage: str, dataset: str,
                  attempt_count: int = 10):
    if stage == "formal":
        for block in FORMAL_ORDER:
            for sample in samples:
                if sample["split"] != "eval":
                    continue
                for task in experiment_tasks((sample,), methods=methods):
                    if formal_stage(task) == block:
                        yield task
        return
    if stage == "independent":
        for sample in samples:
            if dataset == "vizwiz" and sample["split"] != "eval":
                continue
            for replicate in range(attempt_count):
                yield {
                    "sample": sample, "method": "direct", "marker": "UNKNOWN",
                    "reference_marker": "UNKNOWN", "guided": False,
                    "reference_guided": False, "attempt": True,
                    "replicate": replicate, "kind": "independent_attempt",
                }
        return
    if stage == "candidate":
        if dataset != "food101":
            raise ValueError("Candidate ranking is registered for Food-101 only")
        for sample in samples:
            yield {"sample": sample, "kind": "candidate_rank"}
        return
    raise ValueError("Unknown generation stage")


def generation_key(model: str, stage: str, task: dict) -> str:
    if stage == "candidate":
        return stable_hash([model, task["sample"]["id"], "closed"])
    return task_id(model, task)


def condition_of(model: str, dataset: str, stage: str, task: dict) -> dict:
    condition = {"model": model, "dataset": dataset, "split": task["sample"]["split"]}
    if stage == "candidate":
        return {**condition, "method": "closed_rank", "kind": "candidate_rank", "replicate": 0}
    return {
        **condition, "method": task["method"], "kind": task["kind"],
        "main_marker": task["marker"], "reference_marker": task["reference_marker"],
        "guided": task["guided"], "reference_guided": task["reference_guided"],
        "replicate": task["replicate"],
    }


def read_missing_keys(path: Path, model: str, stage: str, dataset: str) -> dict:
    missing = {}
    for ordinal, row in enumerate(read_jsonl(path)):
        key = row.get("key")
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError(f"Invalid missing key at row {ordinal + 1}")
        if key in missing:
            raise ValueError(f"Duplicate missing key at row {ordinal + 1}")
        for field, expected in (("model", model), ("stage", stage), ("dataset", dataset)):
            if field in row and row[field] != expected:
                raise ValueError(f"Missing manifest {field} differs at row {ordinal + 1}")
        missing[key] = (ordinal, row)
    if not missing:
        raise ValueError("Missing-key manifest is empty")
    return missing


def validate_missing_binding(row: dict, task: dict) -> None:
    sample = task["sample"]
    for field in ("sample_id", "split"):
        expected = sample["id"] if field == "sample_id" else sample["split"]
        if field in row and row[field] != expected:
            raise ValueError("Missing key sample metadata differs: " + field)
    for field in (*TASK_FIELDS, "attempt"):
        if field in row and row[field] != task.get(field):
            raise ValueError("Missing key task metadata differs: " + field)


def validate_proofs(root: Path, spec: dict, model: str, methods: list[str],
                    stage: str, manifest: dict, freeze: dict) -> dict:
    """Use the existing proof checks with real canonical file-path resolution."""
    import kdm.io

    entries = {entry["original_path"]: entry for entry in manifest["entries"]}
    canonical_entries = {entry["canonical_blob_path"]: entry for entry in manifest["entries"]}
    original_within = kdm.io.within
    verified = {}

    def proof_path(check_root, relative):
        name = str(relative)
        entry = entries.get(name) or canonical_entries.get(name)
        if entry is None:
            return original_within(check_root, relative)
        original = entry["original_path"]
        if freeze["files"].get(original) != entry["sha256"]:
            raise ValueError("Canonical proof is not bound to the frozen contract: " + original)
        path = original_within(check_root, entry["canonical_blob_path"])
        if original not in verified:
            if file_hash(path) != entry["sha256"]:
                raise ValueError("Canonical proof bytes changed: " + original)
            verified[original] = entry["sha256"]
        return path

    kdm.io.within = proof_path
    try:
        validate_native_runtime_files(root, spec, model)
        if stage == "formal":
            validate_method_runtime(root, spec, methods)
            registry = json.loads((root / "configs/runtime/hosts.json").read_text(encoding="utf-8"))
            if registry["model_hosts"].get(model) == "6403":
                validate_resource_runtime(root, canonical_runtime_spec(root, spec))
    finally:
        kdm.io.within = original_within
    return verified


def validate_inputs(root: Path, model: str, dataset: str, stage: str) -> tuple:
    declaration = json.loads((root / GENERATION_CONFIG).read_text(encoding="utf-8"))
    if (declaration.get("schema") != "kdm_remaining11_generation_v1"
            or tuple(declaration.get("models", ())) != MODELS
            or tuple(declaration.get("formal_order", ())) != FORMAL_ORDER
            or declaration.get("output_root") != "outputs/supplemental/remaining11"
            or declaration.get("food_reference_splits") != ["dev", "eval"]
            or declaration.get("vizwiz_reference_splits") != ["eval"]
            or declaration.get("automatic_retry") is not False
            or declaration.get("automatic_parameter_changes") is not False):
        raise ValueError("Supplemental generation declaration changed")
    manifest, freeze = load_contract(root)
    required = (
        "configs/kdm/study.json", "configs/kdm/method_plan.json",
        f"configs/runtime/{model}.json", "data/current/all.jsonl",
        "data/current/interface16.jsonl",
    )
    hashes = {}
    for relative in required:
        expected = freeze["files"].get(relative)
        actual = file_hash(root / relative)
        if expected is None or actual != expected:
            raise ValueError("Registered input bytes changed: " + relative)
        hashes[relative] = actual
    study = json.loads((root / "configs/kdm/study.json").read_text(encoding="utf-8"))
    fixed = {
        "max_new_tokens": 32, "greedy": True, "alpha": 1.0, "beta": 0.1,
        "m3id_lambda": 0.02, "m3id_threshold": 0.3, "attempt_count": 10,
        "attempt_temperature": 1.0, "attempt_top_p": 1.0,
    }
    if any(study.get(field) != expected for field, expected in fixed.items()):
        raise ValueError("Registered decoding or attempt parameters differ")
    if tuple(study["markers"]) != MARKERS:
        raise ValueError("Registered main/reference marker matrix changed")
    spec = json.loads((root / f"configs/runtime/{model}.json").read_text(encoding="utf-8"))
    if spec.get("key") != model or spec.get("availability") != "resolved":
        raise ValueError("Unresolved or mismatched registered runtime")
    methods = json.loads((root / "configs/kdm/method_plan.json").read_text(encoding="utf-8"))[model][dataset]
    if len(methods) != len(set(methods)) or not {"vcd", "m3id", "dola", "deco"} <= set(methods):
        raise ValueError("Incomplete registered method matrix")
    all_samples = list(read_jsonl(root / "data/current/all.jsonl"))
    if len(all_samples) != 9167 or len({sample["id"] for sample in all_samples}) != 9167:
        raise ValueError("Original full manifest coverage changed")
    samples = [sample for sample in all_samples if sample["dataset"] == dataset]
    expected_splits = {"dev": 2424, "eval": 2424} if dataset == "food101" else {"dev": 818, "eval": 3501}
    if Counter(sample["split"] for sample in samples) != expected_splits:
        raise ValueError("Full registered dataset splits are required")
    names = sorted({sample["class"] for sample in all_samples if sample["dataset"] == "food101"})
    if len(names) != 101:
        raise ValueError("Canonical Food-101 class inventory differs")
    food_counts = Counter((sample["split"], sample["class"]) for sample in all_samples
                          if sample["dataset"] == "food101")
    if len(food_counts) != 202 or set(food_counts.values()) != {24}:
        raise ValueError("Food-101 split/class quotas differ")
    proof_hashes = validate_proofs(root, spec, model, methods, stage, manifest, freeze)
    algorithm_paths = {
        "src/kdm/pipeline.py", "src/kdm/decoding.py", "src/kdm/prompts.py",
        "src/kdm/probability.py", "src/kdm/cda.py",
    }
    recorded_blobs = {}
    for entry in freeze["source_blobs"]:
        metadata, relative = entry.split("\t", 1)
        recorded_blobs[relative] = metadata.split(" ")[1]
    algorithm_blobs = {}
    for relative in sorted(algorithm_paths):
        actual = subprocess.check_output(["git", "hash-object", str(root / relative)],
                                         cwd=root, text=True).strip()
        if recorded_blobs.get(relative) != actual:
            raise ValueError("Registered generation algorithm changed: " + relative)
        algorithm_blobs[relative] = actual
    cfg = DecodeConfig()
    if stage == "independent":
        cfg = DecodeConfig(temperature=study["attempt_temperature"], top_p=study["attempt_top_p"])
    provenance = {
        "frozen_contract_sha256": manifest["original_contract_sha256"],
        "registered_input_sha256": hashes,
        "algorithm_git_blobs": algorithm_blobs,
        "verified_model_proof_sha256": proof_hashes,
        "generation_config_sha256": file_hash(root / GENERATION_CONFIG),
    }
    return samples, methods, spec, cfg, names, provenance


def selected_tasks(plan: dict):
    for task in ordered_tasks(plan["samples"], plan["methods"], plan["stage"], plan["dataset"]):
        key = generation_key(plan["model"], plan["stage"], task)
        if key in plan["selected_keys"]:
            yield task


def load_plan(args: argparse.Namespace, root: Path = ROOT) -> dict:
    if args.model not in MODELS or args.stage not in STAGES:
        raise ValueError("Model or stage lies outside the supplemental task")
    if args.dataset not in {"food101", "vizwiz"}:
        raise ValueError("Unknown registered dataset")
    if args.n_shards < 1 or not 0 <= args.shard < args.n_shards:
        raise ValueError("Invalid sample shard")
    if args.key_start < 0 or (args.key_stop is not None and args.key_stop <= args.key_start):
        raise ValueError("Invalid missing-key ordinal interval")
    samples, methods, spec, cfg, names, provenance = validate_inputs(root, args.model, args.dataset, args.stage)
    path = within(root, args.missing_keys)
    missing = read_missing_keys(path, args.model, args.stage, args.dataset)
    stop = len(missing) if args.key_stop is None else args.key_stop
    if stop > len(missing) or args.key_start >= len(missing):
        raise ValueError("Missing-key ordinal interval exceeds the manifest")
    found = set()
    selected = set()
    counts, full_counts, sample_ids = Counter(), Counter(), set()
    condition_counts, full_condition_counts, conditions = Counter(), Counter(), {}
    digest = hashlib.sha256()
    full_digest = hashlib.sha256()
    for task in ordered_tasks(samples, methods, args.stage, args.dataset):
        key = generation_key(args.model, args.stage, task)
        full_digest.update((key + "\n").encode())
        condition = formal_stage(task) if args.stage == "formal" else args.stage
        full_counts[condition] += 1
        fields = condition_of(args.model, args.dataset, args.stage, task)
        condition_key = stable_hash(fields)
        conditions[condition_key] = fields
        full_condition_counts[condition_key] += 1
        if key not in missing:
            continue
        ordinal, row = missing[key]
        if key in found:
            raise ValueError("Duplicate key in registered task expansion")
        found.add(key)
        validate_missing_binding(row, task)
        if not args.key_start <= ordinal < stop or not owns(task["sample"], args.shard, args.n_shards):
            continue
        selected.add(key)
        sample_ids.add(task["sample"]["id"])
        counts[condition] += 1
        condition_counts[condition_key] += 1
        digest.update((key + "\n").encode())
    if found != set(missing):
        unknown = sorted(set(missing) - found)
        raise ValueError(f"Missing manifest contains {len(unknown)} unregistered keys: {unknown[:3]}")
    if not selected:
        raise ValueError("Selected explicit missing-key shard is empty")
    summary = {
        "schema": "kdm_remaining11_generation_plan_v1", "model": args.model,
        "dataset": args.dataset, "stage": args.stage, "shard": args.shard,
        "n_shards": args.n_shards, "missing_keys_path": str(path.relative_to(root)),
        "missing_keys_sha256": file_hash(path), "missing_manifest_rows": len(missing),
        "key_start": args.key_start, "key_stop": stop,
        "expected_generation_rows": len(selected), "selected_samples": len(sample_ids),
        "stage_expected": dict(counts), "registered_full_stage_rows": sum(full_counts.values()),
        "registered_full_stage_counts": dict(full_counts),
        "ordered_selected_keys_sha256": digest.hexdigest(),
        "ordered_registered_keys_sha256": full_digest.hexdigest(),
        "methods": methods, "base_config": asdict(cfg), "source_provenance": provenance,
        "conditions": [
            {"condition_key": key, **conditions[key], "registered_rows": full_condition_counts[key],
             "selected_missing_rows": condition_counts[key]}
            for key in sorted(conditions)
        ],
        "checks": {
            "original_input_bytes": True, "original_algorithm_blobs": True,
            "native_runtime_proofs": True, "method_runtime_proofs": args.stage == "formal",
            "explicit_missing_keys_registered": True, "unique_missing_keys": True,
            "dataset_split_and_class_quotas": True, "no_gpu_initialized": True,
        },
        "limitations": "CPU provenance and task coverage; no GPU execution or label completion",
    }
    return {
        "model": args.model, "dataset": args.dataset, "stage": args.stage,
        "samples": samples, "methods": methods, "spec": spec, "cfg": cfg,
        "names": names, "selected_keys": selected, "summary": summary,
    }


def claim_directory(args: argparse.Namespace, plan: dict, run: Path) -> Path:
    import fcntl

    if not NAME_PATTERN.fullmatch(args.claim_id) or not NAME_PATTERN.fullmatch(args.owner):
        raise ValueError("Invalid claim identifier or owner")
    folder = run / "claims" / args.model / args.dataset / args.stage
    folder.mkdir(parents=True, exist_ok=True)
    claim = folder / args.claim_id
    recovery_sources = []
    continuation = None
    if getattr(args, 'continuation_receipt', None):
        from workflows.supplemental.remaining11.retirement import validate_retirement
        continuation = validate_retirement(ROOT, args.continuation_receipt, args.claim_id,
            plan['summary'], plan['selected_keys'],
            os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','), args.owner)
    with (folder / "ownership.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        for other in folder.iterdir():
            if not other.is_dir():
                continue
            if continuation is not None and other.name == continuation['source_claim_id']:
                markers = [json.loads(path.read_text()) for path in other.glob('retired*.json')]
                markers = [item for item in markers if item['retirement_sha256'] == continuation['retirement_sha256']]
                if len(markers) != 1:
                    raise ValueError('Original claim lacks a unique matching appended retirement')
                retired = markers[0]
                if (retired['retirement_sha256'] != continuation['retirement_sha256']
                        or retired['source_owner_sha256'] != file_hash(other / 'owner.json')
                        or retired['source_keys_sha256'] != file_hash(other / 'keys.jsonl')
                        or args.claim_id not in retired['replacement_claim_ids']):
                    raise ValueError('Original administrative retirement does not bind to this replacement')
                continue
            release_path = other / 'released.json'
            if release_path.is_file():
                release = json.loads(release_path.read_text())
                if release.get('replacement_claim_id') == args.claim_id:
                    from workflows.supplemental.remaining11.dispatch import validate_import_release

                    recovery_sources.append(validate_import_release(
                        other, args.claim_id, plan['summary'],
                        os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','), args.owner))
                    continue
            key_path = other / "keys.jsonl"
            if not key_path.is_file():
                raise ValueError("Existing shard claim lacks a complete key ownership list")
            for row in read_jsonl(key_path):
                if row["key"] in plan["selected_keys"]:
                    raise ValueError("Explicit missing key is already claimed: " + row["key"])
        claim.mkdir(exist_ok=False)
        with (claim / "keys.jsonl").open("x", encoding="utf-8", newline="\n") as target:
            for task in selected_tasks(plan):
                target.write(json.dumps({"key": generation_key(args.model, args.stage, task),
                                         "sample_id": task["sample"]["id"]}) + "\n")
            target.flush()
            os.fsync(target.fileno())
        atomic_json(claim / "owner.json", {
            "claim_id": args.claim_id, "owner": args.owner, "pid": os.getpid(),
            "created_utc": now(), "model": args.model, "dataset": args.dataset,
            "stage": args.stage, "plan": plan["summary"],
            "keys_sha256": file_hash(claim / "keys.jsonl"),
            "input_recovery_sources": recovery_sources,
            "administrative_continuation": continuation,
        })
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return claim


def verify_output(path: Path, plan: dict, tasks: list[dict], identity: dict,
                  shard: int, n_shards: int) -> dict:
    meta_path = path.with_suffix(".identity.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if plan["stage"] == "candidate":
        definition = {**identity, "names": plan["names"], "ranking_rule": "mean_log_probability"}
    else:
        definition = {**identity, "model": plan["model"], "shard": shard,
                      "n_shards": n_shards, "base_config": asdict(plan["cfg"])}
    if meta["definition"] != definition or meta["identity"] != stable_hash(definition):
        raise ValueError("Raw-part identity sidecar differs from admitted configuration")
    expected = {generation_key(plan["model"], plan["stage"], task): task for task in tasks}
    seen, truncated, conditions = set(), 0, Counter()
    for row in read_jsonl(path):
        key = row["key"]
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate raw-part key")
        task = expected[key]
        if (row["identity"] != meta["identity"] or row["model"] != plan["model"]
                or row["sample"] != task["sample"] or row["status"] != "ok"):
            raise ValueError("Raw-part identity, sample or status differs")
        if plan["stage"] == "candidate":
            scores = row["candidate_scores"]
            if (len(scores) != 101 or {entry["label"] for entry in scores} != set(plan["names"])
                    or row["target"] != task["sample"]["class"]
                    or row["ranking_rule"] != "mean_log_probability"):
                raise ValueError("Candidate ranking labels or rule differ")
            if any(not math.isfinite(entry[field]) or entry["n_tokens"] < 1
                   for entry in scores for field in ("sum_logp", "mean_logp")):
                raise ValueError("Candidate ranking has invalid log probabilities")
            target = next(entry for entry in scores if entry["label"] == row["target"])
            if row["gold_rank"] != 1 + sum(entry["mean_logp"] > target["mean_logp"] for entry in scores):
                raise ValueError("Candidate gold rank differs from original comparison rule")
        else:
            actual_task = {field: row[field] for field in TASK_FIELDS}
            actual_task["sample"] = row["sample"]
            if "attempt" in row:
                actual_task["attempt"] = row["attempt"]
            if actual_task != task or task_id(plan["model"], actual_task) != key:
                raise ValueError("Raw response task fields differ")
            tokens = row["tokens"]
            if (not isinstance(row["terminated"], bool) or not 1 <= len(tokens) <= 32
                    or len(row["selected_log_probabilities"]) != len(tokens)
                    or not all(math.isfinite(value) for value in row["selected_log_probabilities"])):
                raise ValueError("Response token budget or probability evidence is invalid")
            config = row["config"]
            allowed = asdict(plan["cfg"])
            allowed["method"] = task["method"]
            if task["method"] in {"m3id", "instruction_m3id"}:
                if config["m3id_offset"] != len(row["offset_prompt_tokens"]):
                    raise ValueError("M3ID offset differs from the registered prompt tokens")
                allowed["m3id_offset"] = config["m3id_offset"]
            if config != allowed:
                raise ValueError("Raw decoding parameters differ")
            if row["seed"] != stable_seed(task["sample"]["id"], plan["model"], task["replicate"]):
                raise ValueError("Raw stable sample/model/replicate seed differs")
            truncated += int(not row["terminated"])
        seen.add(key)
        condition = formal_stage(task) if plan["stage"] == "formal" else plan["stage"]
        conditions[condition] += 1
    if seen != set(expected):
        raise ValueError("Raw part is missing admitted task keys")
    return {
        "rows": len(seen), "truncated_rows": truncated, "stage_counts": dict(conditions),
        "raw_sha256": file_hash(path), "identity_sha256": file_hash(meta_path),
        "generation_complete": True, "labeling_complete": False, "research_complete": False,
    }


def run_candidate(backend, plan: dict, tasks: list[dict], path: Path, identity: dict) -> int:
    from PIL import Image
    from kdm.execution import resolve_image_path

    ledger = Ledger(path, {**identity, "names": plan["names"], "ranking_rule": "mean_log_probability"})
    for task in tasks:
        sample = task["sample"]
        started = time.perf_counter()
        with Image.open(resolve_image_path(sample["image_path"], ROOT)) as image:
            result = closed_rank(backend, image.convert("RGB"), sample["question"],
                                 plan["names"], sample["class"])
        ledger.add(generation_key(plan["model"], "candidate", task), {
            "status": "ok", "model": plan["model"], "sample": sample,
            "kind": "candidate_rank", "seed": stable_seed(sample["id"], plan["model"], 0),
            "wall_s": time.perf_counter() - started, **result,
        })
    return len(ledger.keys)


def execute(args: argparse.Namespace) -> None:
    from workflows.supplemental.remaining11.execution import validate_supplemental_runtime

    if not NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("Invalid exclusive supplemental run name")
    if args.chunk_rows < 1:
        raise ValueError("Raw part size must be positive")
    plan = load_plan(args)
    cards = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    admission = validate_supplemental_runtime(ROOT, plan["spec"], args.model, cards,
                                             args.stage, args.claim_id, args.owner)
    if not isinstance(admission.get("runtime_spec"), dict) or not isinstance(admission.get("execution"), dict):
        raise ValueError("Supplemental runtime admission lacks its actual spec or execution receipt")
    run_root = getattr(args, "run_root", None)
    if run_root is not None and not run_root.startswith("outputs/supplemental/remaining4/"):
        raise ValueError("Explicit selected-four output must remain in its supplemental directory")
    run = within(ROOT, run_root or ("outputs/supplemental/remaining11/" + args.run_name))
    if run_root is not None:
        run.relative_to(ROOT / "outputs/supplemental/remaining4")
    claim = claim_directory(args, plan, run)
    output = run / "raw" / args.model / args.dataset / args.stage / args.claim_id
    record = run / "records" / args.model / args.dataset / args.stage / args.claim_id
    output.mkdir(parents=True, exist_ok=False)
    record.mkdir(parents=True, exist_ok=False)
    identity = {
        "schema": "kdm_remaining11_generation_v1", "model": args.model,
        "dataset": args.dataset, "stage": args.stage, "claim_id": args.claim_id,
        "owner": args.owner, "backend": admission["runtime_spec"],
        "registered_backend": plan["spec"], "runtime_admission": admission,
        "source_provenance": plan["summary"]["source_provenance"],
        "task_plan": plan["summary"], "runner_sha256": file_hash(Path(__file__)),
        "main_marker_field": "marker", "no_automatic_retry": True,
        "human_review_claimed": False, "paid_api_calls": 0,
    }
    recovered = json.loads((claim / 'owner.json').read_text())['input_recovery_sources']
    if recovered:
        identity['input_recovery_sources'] = recovered
    continued = json.loads((claim / 'owner.json').read_text()).get('administrative_continuation')
    if continued is not None:
        identity['administrative_continuation'] = continued
    retained_ownership = getattr(args, 'retained_source_ownership', None)
    if retained_ownership is not None:
        identity['retained_source_ownership'] = retained_ownership
    atomic_json(record / "admission.json", identity)
    progress = {
        "model": args.model, "dataset": args.dataset, "stage": args.stage,
        "claim_id": args.claim_id, "owner": args.owner, "pid": os.getpid(),
        "started_utc": now(), "status": "loading", "completed": 0,
        "expected": plan["summary"]["expected_generation_rows"], "completed_parts": [],
        "claim_path": str(claim.relative_to(ROOT)), "current_task_key": None,
    }
    atomic_json(record / "progress.json", progress)
    current_tasks = []
    current_output = None
    last_write = 0.0
    try:
        backend = make_backend(admission["runtime_spec"], "cuda:0")
        progress["status"] = "running"
        atomic_json(record / "progress.json", progress)
        stream = selected_tasks(plan)
        part = 0
        while True:
            current_tasks = list(islice(stream, args.chunk_rows))
            if not current_tasks:
                break
            basename = f"{args.stage}_{args.model}_part_{part:05d}"
            current_output = output / (basename + ".jsonl")
            part_identity = {**identity, "part": part, "expected_part_rows": len(current_tasks)}
            part_start = progress["completed"]

            def tracked_tasks():
                nonlocal last_write
                for ordinal, task in enumerate(current_tasks):
                    progress["current_task_key"] = generation_key(args.model, args.stage, task)
                    yield task
                    progress["completed"] = part_start + ordinal + 1
                    if time.monotonic() - last_write >= 15:
                        progress["updated_utc"] = now()
                        atomic_json(record / "progress.json", progress)
                        last_write = time.monotonic()

            if args.stage == "candidate":
                count = run_candidate(backend, plan, tracked_tasks(), current_output, part_identity)
                progress["completed"] = part_start + count
            else:
                count = run_tasks(backend, args.model, tracked_tasks(), current_output,
                                  part_identity, plan["cfg"], args.shard, args.n_shards)
            if count != len(current_tasks):
                raise ValueError("Registered runner did not complete the explicit raw part")
            coverage = verify_output(current_output, plan, current_tasks, part_identity,
                                     args.shard, args.n_shards)
            complete = {
                **coverage, "finished_utc": now(), "model": args.model,
                "dataset": args.dataset, "stage": args.stage, "claim_id": args.claim_id,
                "owner": args.owner, "part": part, "raw_path": str(current_output.relative_to(ROOT)),
            }
            atomic_json(record / (basename + ".complete.json"), complete)
            progress["completed_parts"].append({"part": part, "rows": count,
                                                 "raw_path": complete["raw_path"]})
            progress.update(updated_utc=now(), current_task_key=None)
            atomic_json(record / "progress.json", progress)
            print(json.dumps({"event": "raw_part_complete", **complete}), flush=True)
            part += 1
        if progress["completed"] != progress["expected"]:
            raise ValueError("Claim did not complete all explicitly admitted missing keys")
        progress.update(status="generation_complete", updated_utc=now())
        atomic_json(record / "progress.json", progress)
        atomic_json(record / "complete.json", {
            **progress, "admission_sha256": file_hash(record / "admission.json"),
            "generation_complete": True, "labeling_complete": False,
            "gt_complete": False, "research_complete": False,
        })
        atomic_json(claim / "complete.json", {
            "claim_id": args.claim_id, "owner": args.owner,
            "rows": progress["completed"], "finished_utc": now(),
            "record_complete_path": str((record / "complete.json").relative_to(ROOT)),
        })
    except BaseException as exc:
        failure = {
            "model": args.model, "dataset": args.dataset, "stage": args.stage,
            "claim_id": args.claim_id, "owner": args.owner,
            "error": type(exc).__name__ + ": " + str(exc),
            "traceback": traceback.format_exc(), "when": now(),
            "current_task_key": progress["current_task_key"],
            "current_raw_path": str(current_output.relative_to(ROOT)) if current_output else None,
            "completed_rows": progress["completed"],
            "current_part_expected_keys": [generation_key(args.model, args.stage, task)
                                           for task in current_tasks],
            "no_retry_performed": True,
        }
        atomic_json(record / "failure.json", failure)
        progress.update(status="failed", error=failure["error"], updated_utc=now())
        atomic_json(record / "progress.json", progress)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--dataset", default="food101", choices=("food101", "vizwiz"))
    parser.add_argument("--missing-keys", required=True)
    parser.add_argument("--key-start", type=int, default=0)
    parser.add_argument("--key-stop", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--run-name")
    parser.add_argument("--claim-id")
    parser.add_argument("--owner")
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--continuation-receipt")
    parser.add_argument("--plan-output")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        if not all((args.run_name, args.claim_id, args.owner)):
            parser.error("--execute requires --run-name, --claim-id and --owner")
        execute(args)
    else:
        summary = load_plan(args)["summary"]
        if args.plan_output:
            destination = within(ROOT, args.plan_output)
            if destination.exists():
                raise ValueError("Plan output already exists")
            atomic_json(destination, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
