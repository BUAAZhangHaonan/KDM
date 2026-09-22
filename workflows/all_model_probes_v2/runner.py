#!/usr/bin/env python3
"""Versioned all-split independent attempts and Food-101 closed scoring."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kdm.decoding import DecodeConfig, generate
from kdm.execution import resolve_image_path
from kdm.io import Ledger, atomic_json, file_hash, read_jsonl, stable_hash, stable_seed
from kdm.pipeline import make_backend, closed_rank, task_id
from kdm.prompts import task_prompt
from kdm.protocol import validate_freeze, validate_runtime

AMENDMENT = "docs/current/PROTOCOL_AMENDMENT_20260922.md"
WORKFLOW = "workflows/all_model_probes_v2"
CONFIG = DecodeConfig(temperature=1.0, top_p=1.0)


def now():
    return datetime.now(timezone.utc).isoformat()


def tasks_for_sample(sample):
    for replicate in range(10):
        yield {"sample": sample, "method": "direct", "marker": "UNKNOWN",
               "reference_marker": "UNKNOWN", "guided": False,
               "reference_guided": False, "attempt": True,
               "replicate": replicate, "kind": "independent_attempt"}


def load_inputs(root=ROOT):
    samples = list(read_jsonl(root / "data/current/all.jsonl"))
    names = sorted(json.loads((root / "configs/kdm/food_aliases.json").read_text()))
    if len(samples) != 9167 or len({s["id"] for s in samples}) != 9167:
        raise ValueError("All 9167 original samples required")
    from collections import Counter
    counts = Counter((s["dataset"], s["split"]) for s in samples)
    expected = {("food101", "dev"): 2424, ("food101", "eval"): 2424,
                ("vizwiz", "dev"): 818, ("vizwiz", "eval"): 3501}
    if counts != expected or len(names) != 101:
        raise ValueError("Original all-split dataset or 101 classes changed")
    return samples, names


class PrefixMemo:
    """Cache only deterministic CPU Step values, never mutable GPU KV states.

    Underlying frozen session.next fully reconstructs any non-extension prefix.
    Reusing a previously measured prefix cannot alter its distribution or RNG.
    """
    def __init__(self, session):
        self.session = session
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def next(self, prefix):
        key = tuple(prefix)
        if key not in self.cache:
            self.cache[key] = self.session.next(key)
            self.misses += 1
        else:
            self.hits += 1
        return self.cache[key]


class MemoBackend:
    def __init__(self, backend):
        self.backend = backend
        self.last_session = None

    def encode(self, text):
        return self.backend.encode(text)

    def session(self, *args, **kwargs):
        self.last_session = PrefixMemo(self.backend.session(*args, **kwargs))
        return self.last_session


def compare_attempt(reference, observed):
    if reference["tokens"] != observed["tokens"] or reference["text"] != observed["text"]:
        raise ValueError("Memoization changed independent generation")
    a = np.array(reference["selected_log_probabilities"])
    b = np.array(observed["selected_log_probabilities"])
    error = float(np.max(np.abs(a - b)))
    if error > 1e-6:
        raise ValueError("Memoization changed token log probabilities")
    return error


def compare_closed(reference, observed):
    if reference["gold_rank"] != observed["gold_rank"]:
        raise ValueError("Memoization changed closed rank")
    errors = []
    for a, b in zip(reference["candidate_scores"], observed["candidate_scores"]):
        if a["label"] != b["label"] or a["n_tokens"] != b["n_tokens"]:
            raise ValueError("Memoization changed candidate tokenization")
        errors += [abs(a["sum_logp"] - b["sum_logp"]), abs(a["mean_logp"] - b["mean_logp"])]
    error = max(errors)
    if error > 1e-6 or len(reference["candidate_scores"]) != 101:
        raise ValueError("Memoization changed 101-class scores")
    return error


def verify_coverage(output, model, samples, names):
    expected = {task_id(model, task) for sample in samples for task in tasks_for_sample(sample)}
    original = {sample["id"]: sample for sample in samples}
    seen = set()
    for row in read_jsonl(output / "independent.jsonl"):
        if row["key"] in seen or row["key"] not in expected or row["model"] != model:
            raise ValueError("Invalid independent coverage")
        if row["config"] != asdict(CONFIG) or not row.get("tokens") or original.get(row["sample"]["id"]) != row["sample"]:
            raise ValueError("Invalid independent response")
        seen.add(row["key"])
    if seen != expected:
        raise ValueError("Missing independent attempts")
    food = {s["id"]: s for s in samples if s["dataset"] == "food101"}
    seen = set()
    for row in read_jsonl(output / "closed.jsonl"):
        sid = row["sample"]["id"]
        if sid in seen or food.get(sid) != row["sample"] or row["model"] != model:
            raise ValueError("Invalid closed coverage")
        if [x["label"] for x in row["candidate_scores"]] != names:
            raise ValueError("Closed scoring omitted a Food-101 class")
        if any(not np.isfinite(x["sum_logp"]) or not np.isfinite(x["mean_logp"]) for x in row["candidate_scores"]):
            raise ValueError("Nonfinite closed score")
        seen.add(sid)
    if seen != set(food):
        raise ValueError("Missing closed scores")
    return {"complete": True, "independent": len(expected), "closed": len(food),
            "all_splits": True, "closed_classes": 101,
            "output_sha256": {p.name: file_hash(p) for p in output.glob("*.jsonl")}}


def execute(model, run_dir):
    run = ROOT / run_dir
    if not run.resolve().is_relative_to(ROOT / "outputs/records/probes_v2"):
        raise ValueError("Run record root must remain in probes_v2")
    samples, names = load_inputs()
    spec_path = ROOT / f"configs/runtime/{model}.json"
    spec = json.loads(spec_path.read_text())
    freeze = validate_freeze(ROOT)
    if file_hash(spec_path) != freeze["files"][f"configs/runtime/{model}.json"]:
        raise ValueError("Frozen runtime spec changed")
    amendment = ROOT / AMENDMENT
    if not amendment.is_file():
        raise ValueError("Recorded user scope amendment required")
    execution = validate_runtime(ROOT, spec, model, os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    output = ROOT / "outputs/raw/probes_v2" / run.name / model
    if output.exists():
        raise ValueError("Fresh output required; no automatic retry or resume")
    output.mkdir(parents=True)
    identity = {"schema": "kdm_all_model_probes_v2", "model": model,
                "stage": "all_splits_independent_and_closed",
                "backend": spec, "backend_spec_sha256": file_hash(spec_path),
                "manifest_sha256": file_hash(ROOT / "data/current/all.jsonl"),
                "freeze_receipt_sha256": file_hash(ROOT / "outputs/records/preregistration_freeze.json"),
                "source_blobs": freeze["source_blobs"], "execution": execution,
                "amendment_path": AMENDMENT, "amendment_sha256": file_hash(amendment),
                "workflow_sha256": {p.name: file_hash(p) for p in sorted((ROOT / WORKFLOW).glob("*.py"))},
                "repeats": 10, "splits": ["dev", "eval"], "base_config": asdict(CONFIG),
                "candidate_names": names, "sampling_seed_rule": "stable_seed(sample_id, model, replicate)",
                "optimization": "deterministic CPU prefix-Step memoization; unchanged frozen session, sampling and closed_rank",
                "no_automatic_retry": True}
    attempts = Ledger(output / "independent.jsonl", {**identity, "record_kind": "independent"})
    closed = Ledger(output / "closed.jsonl", {**identity, "record_kind": "closed"})
    progress_path = run / f"{model}_progress.json"
    progress = {"model": model, "pid": os.getpid(), "started_utc": now(),
                "output": str(output.relative_to(ROOT)), "status": "loading",
                "independent": 0, "closed": 0, "expected_independent": 91670, "expected_closed": 4848,
                "execution": execution}
    atomic_json(progress_path, progress)
    current = {"phase": "load"}
    try:
        backend = make_backend(spec, "cuda:0")
        proof = {"passed": False, "model": model, "sample_id": samples[0]["id"],
                 "attempts": [], "scope": "actual model exact-prefix memoization versus unchanged frozen functions",
                 "workflow_sha256": identity["workflow_sha256"]}
        progress["status"] = "running"
        atomic_json(progress_path, progress)
        for index, sample in enumerate(samples):
            current = {"phase": "independent", "sample_id": sample["id"]}
            with Image.open(resolve_image_path(sample["image_path"], ROOT)) as im:
                image = im.convert("RGB")
            prompt = task_prompt(sample["question"], guided=False, attempt=True)
            main = PrefixMemo(backend.session(image, prompt))
            for task in tasks_for_sample(sample):
                current = {"phase": "independent", "sample_id": sample["id"], "replicate": task["replicate"]}
                seed = stable_seed(sample["id"], model, task["replicate"])
                start = time.perf_counter()
                result = generate(main, None, CONFIG, backend.eos, backend.decode, seed)
                if index == 0 and task["replicate"] < 2:
                    reference_session = backend.session(image, prompt)
                    reference = generate(reference_session, None, CONFIG, backend.eos, backend.decode, seed)
                    del reference_session
                    proof["attempts"].append({"replicate": task["replicate"],
                        "tokens_equal": True, "max_log_probability_error": compare_attempt(reference, result)})
                key = task_id(model, task)
                attempts.add(key, {"key": key, **{k: v for k, v in task.items() if k != "sample"},
                    "model": model, "sample": sample, "prompt": prompt, "reference_prompt": None,
                    "config": asdict(CONFIG), "seed": seed, "wall_s": time.perf_counter() - start, **result})
                progress["independent"] += 1
            del main
            if sample["dataset"] == "food101":
                current = {"phase": "closed", "sample_id": sample["id"]}
                start = time.perf_counter()
                memo_backend = MemoBackend(backend)
                result = closed_rank(memo_backend, image, sample["question"], names, sample["class"])
                memo_stats = {"hits": memo_backend.last_session.hits, "misses": memo_backend.last_session.misses}
                del memo_backend
                if index == 0:
                    reference = closed_rank(backend, image, sample["question"], names, sample["class"])
                    proof["closed_max_score_error"] = compare_closed(reference, result)
                    proof["closed_gold_rank_equal"] = True
                    proof["passed"] = True
                    atomic_json(run / f"{model}_memoization_verification.json", proof)
                key = stable_hash([model, sample["id"], "closed"])
                closed.add(key, {"status": "ok", "model": model, "sample": sample,
                    "wall_s": time.perf_counter() - start, "prefix_memo": memo_stats, **result})
                progress["closed"] += 1
            progress.update(last_sample_id=sample["id"], last_update_utc=now())
            atomic_json(progress_path, progress)
        current = {"phase": "verification"}
        receipt = verify_coverage(output, model, samples, names)
        atomic_json(run / f"{model}_complete.json", {**receipt, "finished_utc": now(),
            "identity": identity, "semantic_annotation_complete": False})
        progress.update(status="complete", finished_utc=now())
        atomic_json(progress_path, progress)
    except BaseException as exc:
        failure = {**current, "model": model, "at_utc": now(),
                   "error": type(exc).__name__ + ": " + str(exc), "traceback": traceback.format_exc()}
        with (run / f"{model}.errors.jsonl").open("a") as f:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        progress.update(status="failed", error=failure["error"], failed_task=current, finished_utc=now())
        atomic_json(progress_path, progress)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    execute(args.model, args.run_dir)
