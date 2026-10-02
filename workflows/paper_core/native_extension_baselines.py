#!/usr/bin/env python3
"""Independent native DoLa/DeCo/SID parts for the four registered extensions.

Reuse the actual-input audit and original pipeline; admission remains the
remaining11 frozen-proof and host/runtime chain, including its device maps.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.decoding import DecodeConfig
from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash, stable_seed
from kdm.pipeline import make_backend, run_tasks, task_id
from native_baselines import AuditBackend, now, task
from workflows.supplemental.remaining11.generate import validate_inputs
from workflows.supplemental.remaining11 import execution

MODELS = ("internvl35_8b", "onevision", "phi35", "qwen3vl")
METHODS = ("dola", "deco", "sid")
OUTPUT = "outputs/paper_core_20261002_dev_viz/native_extension_baselines"


def inputs(model, method):
    samples, methods, spec, cfg, _names, provenance = validate_inputs(
        ROOT, model, "food101", "formal")
    if method not in methods:
        raise ValueError("Requested native extension method is not registered for Food")
    samples = [sample for sample in samples if sample["split"] == "eval"]
    if len(samples) != 2424 or len({s["id"] for s in samples}) != 2424:
        raise ValueError("Native extension eval input coverage differs")
    if asdict(cfg) != asdict(DecodeConfig()):
        raise ValueError("Native extension frozen decoder defaults differ")
    return samples, spec, provenance


def admit(spec, args):
    execution.REGISTRY = args.registry
    cards = args.physical_gpus.split(",")
    if (not cards or any(not card.isdigit() for card in cards)
            or len(cards) != len(set(cards))):
        raise ValueError("Invalid native extension physical GPU list")
    os.environ["KDM_SUPPLEMENTAL_DATASET"] = "food101"
    os.environ["KDM_SUPPLEMENTAL_METHOD"] = args.method
    result = execution.validate_supplemental_runtime(
        ROOT, spec, args.model, cards, "formal", args.run_id, args.owner)
    receipt = result["execution"]
    # Some old single-worker registries have no explicit capacity threshold.
    # Reject an already occupied card that cannot even hold the checkpoint;
    # existing, higher admission budgets still apply unchanged above.
    weight_mib = math.ceil(sum(w["size_bytes"] for w in spec["weights"]) / 1048576)
    if any(item["free_mib"] < weight_mib
           for item in receipt["gpu_observation_before_loading"].values()):
        raise ValueError("Native extension card cannot hold its unchanged checkpoint weight bytes")
    receipt["native_weight_only_lower_bound_mib_per_card"] = weight_mib
    fixed = {name: receipt[name] for name in (
        "host", "hostname", "project_root", "physical_gpus", "registry_path",
        "registry_sha256", "admission_source_sha256", "model", "stage", "claim_id",
        "owner", "environment", "checkpoint", "runtime_path_evidence")}
    return result["runtime_spec"], receipt, fixed


def execute(args):
    samples, frozen_spec, provenance = inputs(args.model, args.method)
    chosen = samples[args.start:args.stop]
    if len(chosen) < 8:
        raise ValueError("A native extension part needs its eight retained pilot inputs")
    cfg = DecodeConfig(method=args.method)
    if (cfg.max_tokens, cfg.temperature, cfg.top_p, cfg.alpha, cfg.beta,
            cfg.deco_alpha, cfg.deco_topk, cfg.deco_topp) != (32, 0., 1., 1., .1, .6, 20, .9):
        raise ValueError("Registered native extension decoder defaults differ")
    if args.phase == "plan":
        return {"schema": "kdm_native_extension_plan_v1", "model": args.model,
                "method": args.method, "start": args.start, "stop": args.stop,
                "expected": len(chosen), "pilot_in_full": 8, "config": asdict(cfg),
                "provenance": provenance, "actual_gpu_generation": False,
                "checked_at_utc": now()}
    actual, admitted, fixed = admit(frozen_spec, args)
    directory = ROOT / OUTPUT / args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{args.method}_{args.start:04d}_{args.stop:04d}"
    raw = directory / "raw" / f"{stem}.jsonl"
    gate = directory / f"{stem}.pilot.json"
    claim = directory / f"{stem}.claim.json"
    identity = {
        "schema": "kdm_native_extension_part_identity_v1",
        "protocol": "native_extension_baselines_20261003", "model": args.model,
        "dataset": "food101", "split": "eval", "method": args.method,
        "condition": {k: v for k, v in task(chosen[0], args.method).items() if k != "sample"},
        "frozen_spec_sha256": stable_hash(frozen_spec), "runtime_spec": actual,
        "admission_fixed": fixed, "provenance": provenance, "config": asdict(cfg),
        "start": args.start, "stop": args.stop,
        "sample_ids_sha256": stable_hash([s["id"] for s in chosen]),
        "runner_sha256": file_hash(Path(__file__)),
        "shared_native_audit_source_sha256": file_hash(Path(__file__).with_name("native_baselines.py")),
        "shared_input_description_source_sha256": file_hash(Path(__file__).with_name("native_audit.py")),
        "author": {"agent": "/root/native_baselines", "model": "gpt-6.1-sol", "effort": "max", "call_id": ""},
    }
    if args.phase == "pilot":
        if raw.exists() or gate.exists() or claim.exists():
            raise FileExistsError("Preserve existing native extension pilot/claim; no regeneration")
        with claim.open("x") as stream:
            json.dump({"identity": stable_hash(identity), "owner": args.owner,
                       "expected": len(chosen), "sample_ids": [s["id"] for s in chosen],
                       "created_at_utc": now()}, stream)
    else:
        proof = json.loads(gate.read_text())
        owned = json.loads(claim.read_text())
        if (not proof["passed"] or proof["completed"] != 8
                or proof["identity"] != stable_hash(identity)
                or owned["identity"] != stable_hash(identity) or owned["owner"] != args.owner):
            raise ValueError("Full native extension run requires this unchanged actual pilot and owned part")
    atomic_json(directory / f"{stem}.admission_{args.phase}.json", admitted)
    started_load = time.perf_counter()
    backend = make_backend(actual, device="cuda:0")
    load_wall = time.perf_counter() - started_load
    selected = chosen[:8] if args.phase == "pilot" else chosen
    if args.phase == "pilot":
        backend = AuditBackend(backend, args.model, args.method, selected)
    started = time.perf_counter()
    count = run_tasks(backend, args.model, [task(s, args.method) for s in selected],
                      raw, identity, cfg=cfg)
    rows = list(read_jsonl(raw))
    expected = {task_id(args.model, task(s, args.method)) for s in selected}
    if len(rows) != len(expected) or {r["key"] for r in rows} != expected or count != len(expected):
        raise ValueError("Native extension part key coverage or duplicate audit failed")
    for row in rows:
        if (row["config"] != asdict(cfg) or row["status"] != "ok" or not row["tokens"]
                or len(row["tokens"]) > 32 or row["terminated"] != (row["tokens"][-1] in backend.eos)
                or row["seed"] != stable_seed(row["sample"]["id"], args.model, 0)
                or len(row["selected_log_probabilities"]) != len(row["tokens"])
                or not all(math.isfinite(v) for v in row["selected_log_probabilities"])
                or not math.isfinite(row["first_probability"])):
            raise ValueError("Native extension identity, probability, token or termination audit failed")
    receipt = {
        "schema": "kdm_native_extension_part_receipt_v1", "phase": args.phase,
        "model": args.model, "method": args.method, "identity": stable_hash(identity),
        "identity_scope": "runner_definition_before_run_tasks_ledger_envelope",
        "ledger_identity": json.loads(raw.with_suffix(".identity.json").read_text())["identity"],
        "raw_path": str(raw), "raw_sha256": file_hash(raw),
        "identity_path": str(raw.with_suffix(".identity.json")),
        "identity_sha256": file_hash(raw.with_suffix(".identity.json")),
        "completed": len(rows), "expected_full_part": len(chosen),
        "start": args.start, "stop": args.stop, "model_load_wall_s": load_wall,
        "execution_wall_s": time.perf_counter() - started,
        "raw_generation_wall_s": sum(r["wall_s"] for r in rows),
        "completed_at_utc": now(), "passed": True, "condition": identity["condition"],
        "registry": args.registry, "pilot_in_full": True, "actual_gpu_generation": True,
    }
    if args.phase == "pilot":
        receipt.update(backend.complete(rows))
        atomic_json(gate, receipt)
    else:
        receipt["pilot_source_sha256"] = file_hash(gate)
        atomic_json(directory / f"{stem}.complete.json", receipt)
    return {k: v for k, v in receipt.items() if k != "records"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=("plan", "pilot", "full"), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=2424)
    parser.add_argument("--physical-gpus", default="")
    parser.add_argument("--registry", default="workflows/supplemental/remaining11/host_registry.json")
    parser.add_argument("--owner", default="/root/native_baselines")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_id):
        parser.error("run-id must be a simple exclusive directory name")
    if not 0 <= args.start < args.stop <= 2424:
        parser.error("part must be an explicit nonempty range of the frozen 2424 eval inputs")
    path = (ROOT / args.registry).resolve()
    path.relative_to(ROOT)
    print(json.dumps(execute(args), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
