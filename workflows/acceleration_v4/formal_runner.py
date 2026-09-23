#!/usr/bin/env python3
"""Frozen formal generation for the acceleration panel; no GPU work on --check-plan.

Run only through scripts/worker.sh with the original model spec's Python.
This runner changes task order and admission provenance, not decoding methods.
It never calls a paid judge, creates a human-review receipt, or resumes outputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kdm.decoding import DecodeConfig
from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash, within
from kdm.pipeline import experiment_tasks, make_backend, run_tasks, task_id
from kdm.protocol import validate_freeze, validate_method_runtime, validate_native_runtime_files, validate_runtime

PANEL = "workflows/acceleration_v4/panel.json"
AMENDMENT = "docs/current/PROTOCOL_AMENDMENT_20260923_ACCELERATION.md"
ANNOTATIONS = "outputs/annotations/deepseek_v2/census/merged_after_retry151_v1"
CONFIG = DecodeConfig()
STAGES = ("unknown_main", "unknown_controls", "prompt_matrix")
FIELDS = ("method", "marker", "reference_marker", "guided", "reference_guided", "replicate", "kind")
FIVE = {"qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b"}


def now():
    return datetime.now(timezone.utc).isoformat()


def stage_of(task):
    if task["marker"] == "UNKNOWN" and task["reference_marker"] == "UNKNOWN":
        return "unknown_main" if task["kind"] == "main" else "unknown_controls"
    return "prompt_matrix"


def owns(sample, shard, n_shards):
    return int(stable_hash(sample["id"])[:8], 16) % n_shards == shard


def ordered_tasks(samples, methods, shard=0, n_shards=1):
    if n_shards < 1 or not 0 <= shard < n_shards:
        raise ValueError("Invalid shard")
    # Only one sample's small method expansion is retained at a time.
    for stage in STAGES:
        for sample in samples:
            if sample["split"] != "eval" or not owns(sample, shard, n_shards):
                continue
            for task in experiment_tasks((sample,), methods=methods):
                if stage_of(task) == stage:
                    yield task


def annotation_admission(root, model, samples, panel_entry):
    folder = root / ANNOTATIONS
    identity_path = folder / "identity.json"
    meta = json.loads(identity_path.read_text())
    if stable_hash(meta["definition"]) != meta["identity"]:
        raise ValueError("Invalid merged annotation identity")
    metrics_path = folder / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    labels_path = folder / "labels.jsonl"
    if metrics["annotation_identity"] != meta["identity"] or file_hash(labels_path) != metrics["labels_sha256"]:
        raise ValueError("Merged labels differ from recorded metrics source")
    sources_path = folder / "queue.sources.json"
    if file_hash(sources_path) != metrics["queue_sources_sha256"] or file_hash(sources_path) != meta["definition"]["queue_sources_sha256"]:
        raise ValueError("Annotation census sources changed")
    sources = json.loads(sources_path.read_text())
    relative = f"outputs/raw/current/{model}/census.jsonl"
    matches = [x for x in sources["sources"] if x["path"] == relative]
    if len(matches) != 1:
        raise ValueError("Model census source is not uniquely registered")
    source = matches[0]
    raw_path, side_path = within(root, source["path"]), within(root, source["identity_path"])
    if file_hash(raw_path) != source["sha256"] or file_hash(side_path) != source["identity_sha256"]:
        raise ValueError("Original census or identity has changed")
    side = json.loads(side_path.read_text())
    if stable_hash(side["definition"]) != side["identity"] or side["identity"] != source["identity"]:
        raise ValueError("Invalid census sidecar")
    sample_map = {s["id"]: s for s in samples}
    raw = {}
    total = 0
    for row in read_jsonl(raw_path):
        total += 1
        if row["model"] != model or row["identity"] != side["identity"]:
            raise ValueError("Census model/identity mismatch")
        if row["sample"]["dataset"] != "food101":
            continue
        if row["sample"] != sample_map.get(row["sample"]["id"]) or row["key"] in raw:
            raise ValueError("Census Food sample mismatch or duplicate")
        expected_task = {k: row[k] for k in FIELDS}
        expected_task["sample"] = row["sample"]
        if row["key"] != task_id(model, expected_task) or row["status"] != "ok":
            raise ValueError("Invalid original census key/status")
        raw[row["key"]] = row
    if total != source["rows"] or len(raw) != 9696:
        raise ValueError("Original census Food coverage incomplete")
    expected_pairs = {(s["id"], guided) for s in samples for guided in (False, True)}
    if {(x["sample"]["id"], x["guided"]) for x in raw.values()} != expected_pairs:
        raise ValueError("Original Food guided/unguided pair coverage differs")
    seen = set()
    counts = Counter()
    for label in read_jsonl(labels_path):
        if label["identity"] != meta["identity"]:
            raise ValueError("Mixed merged annotation identity")
        if label.get("model") != model or label.get("dataset") != "food101":
            continue
        key = label["key"]
        if key not in raw or key in seen:
            raise ValueError("Unknown or duplicate selected Food label")
        row = raw[key]
        if (label["raw_record_sha256"] != stable_hash(row) or label["raw_identity"] != row["identity"]
                or label["text"] != row["text"] or label["sample_id"] != row["sample"]["id"]
                or label["split"] != row["sample"]["split"] or label["guided"] != row["guided"]
                or label["label"] not in {"abstain", "invalid", "answer_assertive", "answer_uncertain"}):
            raise ValueError("Automatic label does not bind to census content")
        seen.add(key)
        if row["guided"]:
            split = row["sample"]["split"]
            counts[split + "_valid"] += 1
            counts[split + "_abstain"] += int(label["label"] == "abstain")
    baseline = {x["split"]: x for x in panel_entry["guided_food_baseline"]}
    for split in ("dev", "eval"):
        if counts[split + "_valid"] != baseline[split]["annotated"] or counts[split + "_abstain"] != baseline[split]["abstain"]:
            raise ValueError("Baseline admission counts differ from fixed panel")
    if counts["dev_abstain"] + counts["eval_abstain"] <= 0:
        raise ValueError("No observed baseline abstention in selected condition")
    return {
        "source": "existing automatic DeepSeek labels bound to original census",
        "human_review_claimed": False, "new_api_calls": 0,
        "annotation_identity": meta["identity"],
        "labels_sha256": metrics["labels_sha256"], "metrics_sha256": file_hash(metrics_path),
        "identity_sha256": file_hash(identity_path), "sources_sha256": file_hash(sources_path),
        "census_source": source, "food_census_rows": len(raw), "valid_food_labels": len(seen),
        "unresolved_food_keys": sorted(set(raw) - seen), "guided_counts": dict(counts),
        "selection": "Fixed resource-limited panel; positive observed original abstention; no intervention outcomes used",
    }


def load_plan(model, shard=0, n_shards=1, root=ROOT):
    if model not in FIVE:
        raise ValueError("Model is outside fixed first panel")
    if n_shards < 1 or not 0 <= shard < n_shards:
        raise ValueError("Invalid shard")
    panel = json.loads((root / PANEL).read_text())
    if {x["key"] for x in panel["models"]} != FIVE or len(panel["models"]) != 5:
        raise ValueError("Fixed panel inventory changed")
    entry = next(x for x in panel["models"] if x["key"] == model)
    # The panel records content identities, not mutable current-path promises.
    for relative, digest in panel["sources"].items():
        if file_hash(within(root, relative)) != digest:
            raise ValueError("Panel input source changed: " + relative)
    freeze = validate_freeze(root)
    spec_path = root / f"configs/runtime/{model}.json"
    if file_hash(spec_path) != freeze["files"][str(spec_path.relative_to(root))]:
        raise ValueError("Frozen model spec changed")
    spec = json.loads(spec_path.read_text())
    validate_native_runtime_files(root, spec, model)
    methods = json.loads((root / "configs/kdm/method_plan.json").read_text())[model]["food101"]
    if methods != entry["method_plan"]:
        raise ValueError("Panel method matrix changed")
    validate_method_runtime(root, spec, methods)
    all_samples = list(read_jsonl(root / "data/current/all.jsonl"))
    if len(all_samples) != 9167 or len({x["id"] for x in all_samples}) != 9167:
        raise ValueError("Original manifest coverage changed")
    samples = [s for s in all_samples if s["dataset"] == "food101"]
    if Counter(s["split"] for s in samples) != {"dev": 2424, "eval": 2424}:
        raise ValueError("Full Food dev/eval coverage required")
    annotation = annotation_admission(root, model, samples, entry)
    stage_counts, matrix = Counter(), Counter()
    task_digest = hashlib.sha256()
    keys = set()
    for task in ordered_tasks(samples, methods, shard, n_shards):
        key = task_id(model, task)
        if key in keys:
            raise ValueError("Duplicate planned task")
        keys.add(key)
        stage_counts[stage_of(task)] += 1
        matrix[task["kind"] + "/" + task["method"]] += 1
        task_digest.update((key + "\n").encode())
    if not keys:
        raise ValueError("Empty shard")
    if n_shards == 1 and len(keys) != entry["stage_five_eval_generation_tasks"]:
        raise ValueError("Original generation-task total changed")
    summary = {
        "model": model, "shard": shard, "n_shards": n_shards,
        "full_eval_questions": 2424,
        "shard_eval_questions": sum(s["split"] == "eval" and owns(s, shard, n_shards) for s in samples),
        "expected_generation_rows": len(keys), "stage_expected": dict(stage_counts),
        "method_expected": dict(matrix), "ordered_task_keys_sha256": task_digest.hexdigest(),
        "full_model_expected": entry["stage_five_eval_generation_tasks"],
        "annotation_admission": annotation,
        "checks": {"frozen_sources": True, "native_and_method_proofs": True,
                   "panel_sources": True, "automatic_label_raw_binding": True,
                   "unique_task_keys": True, "no_gpu_initialized": True},
        "limitations": "CPU admission/coverage checks only; no model execution or research completion",
    }
    return samples, methods, spec, freeze, summary


def verify_output(path, model, samples, methods, identity, shard, n_shards):
    meta = json.loads(path.with_suffix(".identity.json").read_text())
    definition = {**identity, "model": model, "shard": shard,
                  "n_shards": n_shards, "base_config": asdict(CONFIG)}
    if meta["definition"] != definition or meta["identity"] != stable_hash(definition):
        raise ValueError("Output ledger identity mismatch")
    expected = {task_id(model, t) for t in ordered_tasks(samples, methods, shard, n_shards)}
    by_id = {s["id"]: s for s in samples}
    seen, stages = set(), Counter()
    for row in read_jsonl(path):
        key = row["key"]
        task = {k: row[k] for k in FIELDS}
        task["sample"] = row["sample"]
        if key not in expected or key in seen or key != task_id(model, task):
            raise ValueError("Unexpected, duplicate or content-mismatched output task")
        if row["identity"] != meta["identity"] or row["model"] != model or row["sample"] != by_id.get(row["sample"]["id"]):
            raise ValueError("Output identity/sample mismatch")
        if row["status"] != "ok" or not isinstance(row["terminated"], bool) or not 1 <= len(row["tokens"]) <= CONFIG.max_tokens:
            raise ValueError("Missing actual successful token/termination output")
        if len(row["selected_log_probabilities"]) != len(row["tokens"]) or not all(math.isfinite(v) for v in row["selected_log_probabilities"]):
            raise ValueError("Missing or nonfinite token probabilities")
        if row["config"]["method"] != row["method"] or row["config"]["max_tokens"] != 32 or row["config"]["temperature"] != 0:
            raise ValueError("Frozen decode configuration mismatch")
        seen.add(key)
        stages[stage_of(task)] += 1
    if seen != expected:
        raise ValueError("Missing formal output tasks")
    return {"generation_complete": True, "rows": len(seen), "stage_counts": dict(stages),
            "raw_sha256": file_hash(path), "sidecar_sha256": file_hash(path.with_suffix(".identity.json")),
            "labeling_complete": False, "gt_complete": False, "research_complete": False}


def execute(args):
    samples, methods, spec, freeze, plan = load_plan(args.model, args.shard, args.n_shards)
    cards = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    # This verifies original Python/versions, registered host/cards and inherited FD20+ locks.
    execution = validate_runtime(ROOT, spec, args.model, cards)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_name):
        raise ValueError("Invalid fresh run name")
    suffix = f"shard_{args.shard:03d}_of_{args.n_shards:03d}"
    record = ROOT / "outputs/records/acceleration_v4" / args.run_name / args.model / suffix
    output = ROOT / "outputs/raw/acceleration_v4" / args.run_name / args.model / suffix
    if record.exists() or output.exists():
        raise ValueError("Fresh shard directories required; no retry or automatic resume")
    record.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    identity = {
        "schema": "kdm_acceleration_v4_frozen_formal_generation", "model": args.model,
        "backend": spec, "backend_spec_sha256": file_hash(ROOT / f"configs/runtime/{args.model}.json"),
        "freeze_receipt_sha256": file_hash(ROOT / "outputs/records/preregistration_freeze.json"),
        "source_blobs": freeze["source_blobs"], "execution": execution,
        "panel_path": PANEL, "panel_sha256": file_hash(ROOT / PANEL),
        "amendment_path": AMENDMENT, "amendment_sha256": file_hash(ROOT / AMENDMENT),
        "manifest_sha256": file_hash(ROOT / "data/current/all.jsonl"),
        "method_plan_sha256": file_hash(ROOT / "configs/kdm/method_plan.json"),
        "runner_sha256": file_hash(Path(__file__)), "base_config": asdict(CONFIG),
        "task_order": list(STAGES), "task_plan": plan, "methods": methods,
        "no_automatic_retry": True, "human_review_claimed": False,
        "optimization": "No decoder/cache modification; unchanged frozen run_tasks",
    }
    atomic_json(record / "admission.json", identity)
    raw = output / "formal.jsonl"
    progress = {"model": args.model, "pid": os.getpid(), "started_utc": now(), "status": "loading",
                "shard": args.shard, "n_shards": args.n_shards, "output": str(raw.relative_to(ROOT)),
                "completed": 0, "expected": plan["expected_generation_rows"], "stage_counts": {},
                "stage_expected": plan["stage_expected"], "execution": execution,
                "labeling_complete": False, "gt_complete": False}
    atomic_json(record / "progress.json", progress)
    stage_counts = Counter()
    last_write = 0.0
    current_task = None

    def tracked_tasks():
        nonlocal last_write, current_task
        for task in ordered_tasks(samples, methods, args.shard, args.n_shards):
            current_task = task_id(args.model, task)
            stage = stage_of(task)
            if progress.get("stage") != stage:
                progress.update(stage=stage, updated_utc=now())
                atomic_json(record / "progress.json", progress)
            yield task
            # The frozen runner has durably added this record before requesting the next item.
            progress["completed"] += 1
            stage_counts[stage] += 1
            progress["stage_counts"] = dict(stage_counts)
            if time.monotonic() - last_write >= 10 or stage_counts[stage] == plan["stage_expected"][stage]:
                progress["updated_utc"] = now()
                atomic_json(record / "progress.json", progress)
                last_write = time.monotonic()

    try:
        backend = make_backend(spec, "cuda:0")
        progress["status"] = "running"
        atomic_json(record / "progress.json", progress)
        count = run_tasks(backend, args.model, tracked_tasks(), raw, identity,
                          CONFIG, shard=args.shard, n_shards=args.n_shards)
        if count != plan["expected_generation_rows"]:
            raise ValueError("Frozen run_tasks returned incomplete coverage")
        coverage = verify_output(raw, args.model, samples, methods, identity, args.shard, args.n_shards)
        atomic_json(record / "complete.json", {**coverage, "finished_utc": now(),
                    "model": args.model, "shard": args.shard, "n_shards": args.n_shards,
                    "admission_sha256": file_hash(record / "admission.json")})
        progress.update(status="generation_complete", updated_utc=now(), completed=count)
        atomic_json(record / "progress.json", progress)
    except BaseException as exc:
        progress.update(status="failed", error=type(exc).__name__ + ": " + str(exc), updated_utc=now())
        atomic_json(record / "progress.json", progress)
        atomic_json(record / "failure.json", {"error": progress["error"], "task_key": current_task,
                    "traceback": traceback.format_exc(), "when": now(),
                    "completed": progress["completed"], "no_retry_performed": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(FIVE))
    parser.add_argument("--run-name", help="Fresh versioned output name; required for GPU execution")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--check-plan", action="store_true", help="Read-only CPU provenance and coverage checks; no model loading")
    args = parser.parse_args()
    if args.check_plan:
        *_, summary = load_plan(args.model, args.shard, args.n_shards)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        if not args.run_name:
            parser.error("--run-name required unless --check-plan")
        execute(args)


if __name__ == "__main__":
    main()

