#!/usr/bin/env python3
"""Execute explicit Food-101 native VCD/M3ID gaps for the selected four models."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.decoding import DecodeConfig
from kdm.frozen import load_contract
from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash, stable_seed, within
from kdm.pipeline import make_backend, run_tasks, task_id
from kdm.prompts import task_prompt
from workflows.supplemental.remaining11.generate import (
    NAME_PATTERN, owns, read_missing_keys, validate_missing_binding, validate_proofs, verify_output,
)

MODELS = ("internvl35_8b", "onevision", "phi35", "qwen3vl")
METHODS = ("vcd", "m3id")
STAGE = "native_unguided"
CONFIG = "configs/supplemental/remaining4/native_baselines.json"
ALGORITHMS = ("src/kdm/pipeline.py", "src/kdm/decoding.py", "src/kdm/prompts.py",
              "src/kdm/probability.py", "src/kdm/models/hf.py", "src/kdm/models/backbone.py",
              "src/kdm/models/internvl_dual.py", "src/kdm/models/internvl_preprocessing.py",
              "src/kdm/models/remote.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new_json(path, value):
    if path.exists():
        raise ValueError("Output already exists: " + str(path))
    atomic_json(path, value)


def source_inputs(model, method):
    declaration = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    fixed = {"schema": "kdm_remaining4_native_baselines_v1", "models": list(MODELS),
             "dataset": "food101", "split": "eval", "methods": list(METHODS),
             "kind": STAGE, "marker": "NONE", "reference_marker": "NONE", "guided": False,
             "reference_guided": False, "replicate": 0, "rows_per_condition": 2424,
             "condition_count": 8, "planned_rows": 19392,
             "parameters_source": "configs/kdm/study.json", "methods_source": "configs/kdm/method_plan.json",
             "algorithm_source": "src/kdm/pipeline.py", "direct_source": "existing_exact_unguided_census",
             "output_root": "outputs/supplemental/remaining4", "automatic_retry": False,
             "automatic_parameter_changes": False}
    if declaration != fixed or model not in MODELS or method not in METHODS:
        raise ValueError("Native baseline declaration or requested scope differs")
    manifest, freeze = load_contract(ROOT)
    paths = ("configs/kdm/study.json", "configs/kdm/method_plan.json", f"configs/runtime/{model}.json",
             "data/current/all.jsonl", "data/current/interface16.jsonl")
    hashes = {path: file_hash(ROOT / path) for path in paths}
    if any(freeze["files"].get(path) != digest for path, digest in hashes.items()):
        raise ValueError("Native baseline frozen input bytes differ")
    study = json.loads((ROOT / paths[0]).read_text(encoding="utf-8"))
    cfg = DecodeConfig(method=method, alpha=study["alpha"], beta=study["beta"],
                       m3id_lambda=study["m3id_lambda"], m3id_threshold=study["m3id_threshold"],
                       max_tokens=study["max_new_tokens"])
    if not study["greedy"] or cfg != DecodeConfig(method=method):
        raise ValueError("Registered native decoding parameters differ from the frozen implementation")
    registered = json.loads((ROOT / paths[1]).read_text(encoding="utf-8"))[model]["food101"]
    if not set(METHODS) <= set(registered):
        raise ValueError("Requested native methods are absent from the original method registration")
    spec = json.loads((ROOT / paths[2]).read_text(encoding="utf-8"))
    if spec["key"] != model or spec["availability"] != "resolved" or spec.get("api"):
        raise ValueError("Native baseline runtime identity is unresolved or is an API")
    samples = [sample for sample in read_jsonl(ROOT / paths[3])
               if sample["dataset"] == "food101" and sample["split"] == "eval"]
    counts = Counter(sample["class"] for sample in samples)
    if len(samples) != 2424 or len({sample["id"] for sample in samples}) != 2424 or len(counts) != 101 or set(counts.values()) != {24}:
        raise ValueError("Full Food-101 eval keys and 101 by 24 quotas are required")
    recorded = {line.split("\t", 1)[1]: line.split(" ", 2)[1] for line in freeze["source_blobs"]}
    blobs = {}
    for relative in ALGORITHMS:
        data = (ROOT / relative).read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest != recorded.get(relative):
            raise ValueError("Frozen native algorithm blob changed: " + relative)
        blobs[relative] = digest
    proofs = validate_proofs(ROOT, spec, model, list(METHODS), "formal", manifest, freeze)
    provenance = {"frozen_contract_sha256": manifest["original_contract_sha256"],
                  "registered_input_sha256": hashes, "algorithm_git_blobs": blobs,
                  "verified_model_proof_sha256": proofs, "native_config_sha256": file_hash(ROOT / CONFIG)}
    return samples, spec, cfg, provenance


def native_tasks(samples, method):
    for sample in samples:
        yield {"sample": sample, "method": method, "marker": "NONE", "reference_marker": "NONE",
               "guided": False, "reference_guided": False, "replicate": 0, "kind": STAGE}


def key_row(model, task):
    return {"key": task_id(model, task), "model": model, "stage": STAGE, "dataset": "food101",
            "sample_id": task["sample"]["id"], "split": "eval",
            **{key: value for key, value in task.items() if key != "sample"}}


def prepare_manifest(args):
    destination = within(ROOT, args.manifest_dir)
    destination.mkdir(parents=True, exist_ok=False)
    sources, completed = [], set()
    for value in args.completed_raw:
        path = within(ROOT, value)
        meta = json.loads(path.with_suffix(".identity.json").read_text(encoding="utf-8"))
        if (meta["identity"] != stable_hash(meta["definition"])
                or meta["definition"].get("schema") != "kdm_remaining4_native_generation_v1"):
            raise ValueError("Completed raw source is outside the native baseline identity")
        rows = list(read_jsonl(path))
        model = meta["definition"]["model"]
        method = meta["definition"]["task_plan"]["method"]
        samples, spec, cfg, provenance = source_inputs(model, method)
        expected = {task_id(model, task): task for task in native_tasks(samples, method)}
        if meta["definition"]["source_provenance"] != provenance:
            raise ValueError("Completed source frozen inputs or algorithms differ")
        keys = [row["key"] for row in rows]
        if len(keys) != len(set(keys)) or not set(keys) <= set(expected) or completed.intersection(keys):
            raise ValueError("Completed source contains unregistered or repeated keys")
        plan = {"model": model, "stage": STAGE, "cfg": cfg}
        identity = {key: value for key, value in meta["definition"].items()
                    if key not in {"shard", "n_shards", "base_config"}}
        coverage = verify_output(path, plan, [expected[key] for key in keys], identity,
                                 meta["definition"]["shard"], meta["definition"]["n_shards"])
        completed.update(keys)
        sources.append({"path": str(path.relative_to(ROOT)), "identity": meta["identity"], **coverage})
    conditions, total = [], 0
    for model in MODELS:
        for method in METHODS:
            samples, spec, cfg, provenance = source_inputs(model, method)
            path = destination / f"{model}_{method}_missing_keys.jsonl"
            missing = [key_row(model, task) for task in native_tasks(samples, method)
                       if task_id(model, task) not in completed]
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                for row in missing:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(missing)
            conditions.append({"model": model, "dataset": "food101", "split": "eval", "method": method,
                               "kind": STAGE, "main_marker": "NONE", "reference_marker": "NONE",
                               "guided": False, "reference_guided": False, "replicate": 0,
                               "expected": 2424, "reused": 2424 - len(missing), "missing": len(missing),
                               "missing_keys_path": str(path.relative_to(ROOT)),
                               "missing_keys_sha256": file_hash(path), "base_config": asdict(cfg),
                               "source_provenance": provenance,
                               "registered_runtime": {key: spec[key] for key in ("key", "factory", "dtype", "gpu_count", "kwargs", "versions")}})
    result = {"schema": "kdm_remaining4_native_gap_manifest_v1", "created_utc": now(),
              "conditions": conditions, "planned_rows": 19392, "reused_rows": len(completed),
              "missing_rows": total, "completed_raw_sources": sources,
              "runner_sha256": file_hash(Path(__file__)), "gpu_admission": "not_run",
              "checks": {"unique_task_keys": True, "frozen_inputs_and_proofs": True,
                         "frozen_algorithm_blobs": True, "eval_101_by_24": True}}
    write_new_json(destination / "manifest.json", result)
    print(json.dumps({"manifest": str((destination / "manifest.json").relative_to(ROOT)),
                      "conditions": len(conditions), "planned_rows": 19392, "reused_rows": len(completed),
                      "missing_rows": total, "gpu_admission": "not_run"}), flush=True)


def load_plan(args):
    if args.n_shards < 1 or not 0 <= args.shard < args.n_shards:
        raise ValueError("Invalid native sample shard")
    samples, spec, cfg, provenance = source_inputs(args.model, args.method)
    path = within(ROOT, args.missing_keys)
    missing = read_missing_keys(path, args.model, STAGE, "food101")
    stop = len(missing) if args.key_stop is None else args.key_stop
    if not 0 <= args.key_start < stop <= len(missing):
        raise ValueError("Native missing-key interval is outside the explicit manifest")
    found, tasks = set(), []
    for task in native_tasks(samples, args.method):
        key = task_id(args.model, task)
        if key not in missing:
            continue
        ordinal, row = missing[key]
        validate_missing_binding(row, task)
        found.add(key)
        if args.key_start <= ordinal < stop and owns(task["sample"], args.shard, args.n_shards):
            tasks.append(task)
    if found != set(missing) or not tasks:
        raise ValueError("Native missing manifest contains unregistered keys or the selected shard is empty")
    ordered_keys = [task_id(args.model, task) for task in tasks]
    summary = {"schema": "kdm_remaining4_native_generation_plan_v1", "model": args.model,
               "dataset": "food101", "split": "eval", "stage": STAGE, "method": args.method,
               "kind": STAGE, "main_marker": "NONE", "reference_marker": "NONE", "guided": False,
               "reference_guided": False, "replicate": 0, "registered_condition_rows": 2424,
               "missing_manifest_rows": len(missing), "missing_keys_path": str(path.relative_to(ROOT)),
               "missing_keys_sha256": file_hash(path), "key_start": args.key_start, "key_stop": stop,
               "shard": args.shard, "n_shards": args.n_shards, "expected_generation_rows": len(tasks),
               "ordered_selected_keys_sha256": hashlib.sha256(("\n".join(ordered_keys) + "\n").encode()).hexdigest(),
               "base_config": asdict(cfg), "source_provenance": provenance,
               "class_counts": dict(Counter(task["sample"]["class"] for task in tasks)),
               "operator_audit_first_rows": min(8, len(tasks)), "gpu_admission": "not_run",
               "checks": {"original_input_bytes": True, "original_algorithm_blobs": True,
                          "native_and_method_proofs": True, "unique_registered_missing_keys": True,
                          "registered_eval_class_quotas": True}}
    return {"model": args.model, "dataset": "food101", "stage": STAGE, "spec": spec,
            "cfg": cfg, "tasks": tasks, "selected_keys": set(ordered_keys), "summary": summary}


def claim_directory(args, plan, run):
    import fcntl

    folder = run / "claims" / args.model / args.method
    folder.mkdir(parents=True, exist_ok=True)
    claim = folder / args.claim_id
    with (folder / "ownership.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for other in folder.iterdir():
            if not other.is_dir():
                continue
            key_path = other / "keys.jsonl"
            if not key_path.is_file():
                raise ValueError("Existing native claim lacks its complete key ownership list")
            if any(row["key"] in plan["selected_keys"] for row in read_jsonl(key_path)):
                raise ValueError("Native task keys overlap an existing claim: " + str(other))
        claim.mkdir(exist_ok=False)
        with (claim / "keys.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for task in plan["tasks"]:
                stream.write(json.dumps(key_row(args.model, task)) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        write_new_json(claim / "owner.json", {"claim_id": args.claim_id, "owner": args.owner,
                                              "pid": os.getpid(), "created_utc": now(),
                                              "plan": plan["summary"], "keys_sha256": file_hash(claim / "keys.jsonl")})
    return claim


def execute(args):
    import workflows.supplemental.remaining11.execution as execution
    from workflows.supplemental.remaining4.native_audit import NativeAuditBackend

    plan = load_plan(args)
    execution.REGISTRY = str(within(ROOT, args.host_registry).relative_to(ROOT))
    cards = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    os.environ["KDM_SUPPLEMENTAL_DATASET"] = "food101"
    admission = execution.validate_supplemental_runtime(ROOT, plan["spec"], args.model, cards,
                                                       STAGE, args.claim_id, args.owner)
    run = within(ROOT, "outputs/supplemental/remaining4/" + args.run_name)
    claim = claim_directory(args, plan, run)
    output = run / "raw" / args.model / args.method / args.claim_id
    record = run / "records" / args.model / args.method / args.claim_id
    output.mkdir(parents=True, exist_ok=False)
    record.mkdir(parents=True, exist_ok=False)
    identity = {"schema": "kdm_remaining4_native_generation_v1", "model": args.model, "method": args.method,
                "dataset": "food101", "stage": STAGE, "claim_id": args.claim_id, "owner": args.owner,
                "backend": admission["runtime_spec"], "registered_backend": plan["spec"],
                "runtime_admission": admission, "source_provenance": plan["summary"]["source_provenance"],
                "task_plan": plan["summary"], "runner_sha256": file_hash(Path(__file__)),
                "operator_observer_sha256": file_hash(ROOT / "workflows/supplemental/remaining4/native_audit.py"),
                "shared_input_observer_sha256": file_hash(ROOT / "workflows/paper_core/native_audit.py"),
                "no_automatic_retry": True, "paid_api_calls": 0, "human_review_claimed": False}
    write_new_json(record / "admission.json", identity)
    progress = {"model": args.model, "method": args.method, "stage": STAGE, "claim_id": args.claim_id,
                "owner": args.owner, "pid": os.getpid(), "started_utc": now(), "status": "loading",
                "completed": 0, "expected": len(plan["tasks"]), "completed_parts": [], "current_task_key": None}
    atomic_json(record / "progress.json", progress)
    current_tasks, raw_path = [], None
    try:
        backend = make_backend(admission["runtime_spec"], "cuda:0")
        stream, part = iter(plan["tasks"]), 0
        while True:
            current_tasks = list(islice(stream, min(8, len(plan["tasks"])) if part == 0 else args.chunk_rows))
            if not current_tasks:
                break
            name = f"native_{args.model}_{args.method}_part_{part:05d}"
            raw_path = output / (name + ".jsonl")
            part_identity = {**identity, "part": part, "expected_part_rows": len(current_tasks)}
            observer = NativeAuditBackend(backend, args.model, current_tasks, plan["cfg"]) if part == 0 else backend

            def tracked_tasks():
                for task in current_tasks:
                    progress.update(status="running", current_task_key=task_id(args.model, task), updated_utc=now())
                    atomic_json(record / "progress.json", progress)
                    yield task

            count = run_tasks(observer, args.model, tracked_tasks(), raw_path, part_identity,
                              plan["cfg"], args.shard, args.n_shards)
            coverage = verify_output(raw_path, plan, current_tasks, part_identity, args.shard, args.n_shards)
            if count != len(current_tasks):
                raise ValueError("Native runner did not complete all explicitly claimed part keys")
            if part == 0:
                audit = observer.proof(list(read_jsonl(raw_path)))
                write_new_json(record / "operator_audit.json", {**audit, "raw_sha256": file_hash(raw_path),
                                                               "identity": stable_hash(part_identity),
                                                               "runtime_admission": admission})
                del observer
            complete = {**coverage, "model": args.model, "method": args.method, "stage": STAGE,
                        "claim_id": args.claim_id, "owner": args.owner, "part": part, "finished_utc": now(),
                        "raw_path": str(raw_path.relative_to(ROOT))}
            write_new_json(record / (name + ".complete.json"), complete)
            progress["completed"] += count
            progress["completed_parts"].append({"part": part, "rows": count, "raw_path": complete["raw_path"]})
            progress.update(updated_utc=now(), current_task_key=None)
            atomic_json(record / "progress.json", progress)
            print(json.dumps({"event": "raw_part_complete", **complete}), flush=True)
            part += 1
        if progress["completed"] != progress["expected"]:
            raise ValueError("Native claim has incomplete key coverage")
        progress.update(status="generation_complete", updated_utc=now())
        atomic_json(record / "progress.json", progress)
        write_new_json(record / "complete.json", {**progress, "generation_complete": True,
                                                 "labeling_complete": False, "research_complete": False,
                                                 "admission_sha256": file_hash(record / "admission.json")})
        write_new_json(claim / "complete.json", {"claim_id": args.claim_id, "owner": args.owner,
                                                "rows": progress["completed"], "finished_utc": now(),
                                                "record_complete_path": str((record / "complete.json").relative_to(ROOT))})
    except BaseException as exc:
        failure = {"error": type(exc).__name__ + ": " + str(exc), "traceback": traceback.format_exc(),
                   "when": now(), "current_task_key": progress["current_task_key"],
                   "current_raw_path": str(raw_path.relative_to(ROOT)) if raw_path else None,
                   "current_part_expected_keys": [task_id(args.model, task) for task in current_tasks],
                   "completed_rows": progress["completed"], "no_retry_performed": True}
        write_new_json(record / "failure.json", failure)
        progress.update(status="failed", error=failure["error"], updated_utc=now())
        atomic_json(record / "progress.json", progress)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-manifest", action="store_true")
    mode.add_argument("--check-plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--completed-raw", action="append", default=[])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--missing-keys")
    parser.add_argument("--key-start", type=int, default=0)
    parser.add_argument("--key-stop", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--plan-output")
    parser.add_argument("--run-name")
    parser.add_argument("--claim-id")
    parser.add_argument("--owner")
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--host-registry", default="workflows/supplemental/remaining11/host_registry.json")
    args = parser.parse_args()
    if args.prepare_manifest:
        if not args.manifest_dir:
            parser.error("--prepare-manifest requires --manifest-dir")
        prepare_manifest(args)
        return
    if not all((args.model, args.method, args.missing_keys)):
        parser.error("--check-plan/--execute require --model, --method and --missing-keys")
    if args.execute:
        if args.chunk_rows < 1 or not all(NAME_PATTERN.fullmatch(value or "") for value in (args.run_name, args.claim_id, args.owner)):
            parser.error("--execute requires valid --run-name, --claim-id, --owner and positive --chunk-rows")
        execute(args)
    else:
        summary = load_plan(args)["summary"]
        if args.plan_output:
            write_new_json(within(ROOT, args.plan_output), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
