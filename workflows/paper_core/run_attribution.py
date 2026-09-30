#!/usr/bin/env python3
"""Run bounded representative first-position attribution with registered backends."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from PIL import Image
from kdm.execution import resolve_image_path
from kdm.io import Ledger, atomic_json, file_hash, read_jsonl, stable_hash, stable_seed
from kdm.pipeline import make_backend
from attribution import measure_attribution
import execution
from native import MODELS


def _now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _budget_scope(path, budget, gpu_count):
    started = time.perf_counter()
    try:
        yield started
    except BaseException as error:
        budget["last_error"] = {"type": type(error).__name__, "message": str(error), "at_utc": _now()}
        raise
    finally:
        used = (time.perf_counter()-started)*gpu_count
        budget["used_gpu_seconds"] += used
        budget["last_call_gpu_seconds"] = used
        budget["updated_at_utc"] = _now()
        atomic_json(path, budget)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpus", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--representative-list", type=Path, required=True)
    parser.add_argument("--prior-budget", type=Path)
    parser.add_argument("--limit", type=int, choices=(8, 101), required=True)
    parser.add_argument("--registry", choices=("workflows/paper_core/host_registry.json",
                        "workflows/paper_core/host_registry_a100.json"),
                        default="workflows/paper_core/host_registry.json")
    args = parser.parse_args()
    execution.REGISTRY = args.registry
    cards = args.physical_gpus.split(",")
    with args.representative_list.open(encoding="utf-8") as handle:
        representative = list(csv.DictReader(handle))
    if len(representative) != 101 or len({row["sample_id"] for row in representative}) != 101:
        raise ValueError("The original representative list must contain exactly 101 unique inputs")
    representative_ids = {row["sample_id"] for row in representative}
    source_rows = [row for row in read_jsonl(args.inputs) if row["model"] == args.model
                   and row["sample_id"] in representative_ids]
    groups = {}
    for row in source_rows:
        groups.setdefault(row["sample_id"], {})[row["condition"]] = row
    expected_conditions = {"direct_guided", "direct_unguided", "vcd_unknown", "instruction_vcd_unknown"}
    if len(groups) != 101 or any(set(group) != expected_conditions for group in groups.values()):
        raise ValueError("Expected 101 representative inputs with all four frozen source replies")
    if len({group["direct_unguided"]["target_class"] for group in groups.values()}) != 101:
        raise ValueError("Representative class coverage differs")
    samples = {row["id"]: row for row in read_jsonl(ROOT / "data/current/all.jsonl")}
    frozen_spec = json.loads((ROOT / f"configs/runtime/{args.model}.json").read_text())
    admission = execution.admit(ROOT, frozen_spec, args.model, cards)
    spec = execution.runtime_spec(ROOT, frozen_spec, args.model)
    run = ROOT / "outputs/paper_core_20260930" / args.run_id
    run.mkdir(parents=True, exist_ok=True)
    identity = {"schema": "kdm_representative_first_position_v1", "model": args.model,
        "protocol": "P1_ATTRIBUTION", "marker": "UNKNOWN", "alpha": 1.0, "beta": 0.1,
        "noise_step": 500, "fixed_admission": admission["fixed_identity"],
        "source_inputs_sha256": file_hash(args.inputs),
        "representative_list_sha256": file_hash(args.representative_list),
        "source_manifest_sha256": file_hash(ROOT / "data/current/all.jsonl"),
        "runner_sha256": file_hash(Path(__file__)),
        "measurement_sha256": file_hash(Path(__file__).with_name("attribution.py")),
        "math_sha256": file_hash(Path(__file__).with_name("four_view_math.py")),
        "no_free_guided_generations": True,
        "prior_budget_source": str(args.prior_budget) if args.prior_budget else None,
        "prior_budget_sha256": file_hash(args.prior_budget) if args.prior_budget else None,
        "budget_seconds_per_model_all_attribution_phases": 2880}
    ledger = Ledger(run / f"{args.model}.representative_events.jsonl", identity)
    budget_path = run / f"{args.model}.budget.json"
    prior_used = 0.0
    if args.prior_budget:
        prior = json.loads(args.prior_budget.read_text())
        if prior["model"] != args.model or prior["gpu_count"] != len(cards):
            raise ValueError("Prior measured budget belongs to a different model or GPU count")
        prior_used = prior["used_gpu_seconds"]
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else {
        "model": args.model, "gpu_count": len(cards), "used_gpu_seconds": prior_used,
        "prior_budget_source": str(args.prior_budget) if args.prior_budget else None,
        "cap_gpu_seconds": 2880.0, "includes": "all attribution model loading and forwards; native generation is separate"}
    if budget["used_gpu_seconds"] >= budget["cap_gpu_seconds"]:
        raise RuntimeError("The model's share of the four-GPU-hour attribution budget is exhausted")
    selected = [(row["sample_id"], groups[row["sample_id"]]) for row in representative[:args.limit]]
    completed_before = len(ledger.keys)
    with _budget_scope(budget_path, budget, len(cards)) as started:
        backend = make_backend(spec, "cuda:0")
        _measure_selected(args, selected, ledger, samples, backend, started, budget, cards)
        budget["representative_completed"] = len(ledger.keys)
    used = budget["last_call_gpu_seconds"]
    result = {"model": args.model, "new_completed": len(ledger.keys)-completed_before,
        "completed": len(ledger.keys), "expected": 101,
        "phase": "representative_first_position", "limit": args.limit,
        "actual_gpu_seconds_this_call": used, "cumulative_budget": budget,
        "first_eight_gate_passed": len(ledger.keys) >= 8,
        "natural_reference_status": "not_run", "diagnostic_status": "not_run",
        "cases_status": "not_run", "identity": stable_hash(identity), "completed_at_utc": _now()}
    atomic_json(run / f"{args.model}.representative_{args.limit}.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _measure_selected(args, selected, ledger, samples, backend, started, budget, cards):
    for sample_id, group in selected:
        key = stable_hash([args.model, sample_id, "representative_first_position", "UNKNOWN"])
        if key in ledger.keys:
            continue
        elapsed_gpu = (time.perf_counter()-started)*len(cards)
        if budget["used_gpu_seconds"] + elapsed_gpu >= budget["cap_gpu_seconds"]:
            break
        sample = samples[sample_id]
        if sample["split"] != "eval" or sample["dataset"] != "food101":
            raise ValueError("Attribution input is not registered Food eval")
        seed = stable_seed(sample_id, args.model, 0)
        if any(row["seed"] != seed or row["question"] != sample["question"] for row in group.values()):
            raise ValueError("Frozen attribution source question or perturbation seed differs")
        image_path = resolve_image_path(sample["image_path"], ROOT)
        branches = {condition: {"tokens": row["tokens"], "text": row["text"],
                    "source_key": row["source_key"], "source_identity": row["source_identity"],
                    "source_path": row["source_path"], "source_line": row["source_line"]}
                    for condition, row in group.items()}
        first_tokens = sorted({row["tokens"][0] for row in group.values() if row["tokens"]})
        marker_tokens = list(backend.encode("UNKNOWN"))
        candidates = sorted(set(first_tokens) | set(marker_tokens[:1]))
        pairs = [(candidates[0], token) for token in candidates[1:]]
        with Image.open(image_path) as original:
            image = original.convert("RGB")
            measured = measure_attribution(backend, image, sample["question"], [()], seed,
                candidate_token_ids=candidates, candidate_pairs=pairs,
                source_meta={"model": args.model, "sample_id": sample_id,
                    "panel": "representative101", "target_class": sample["class"],
                    "image_source": sample["image_path"], "resolved_image": str(image_path),
                    "frozen_response_branches": branches,
                    "candidate_pair_semantics": "observed frozen first-branch tokens and literal marker-token proxies; no semantic abstention inference"})
        ledger.add(key, {"model": args.model, "sample_id": sample_id,
            "dataset": "food101", "split": "eval", "target_class": sample["class"],
            "panel": "representative101", "method": "four_view_same_prefix",
            "kind": "first_position", "main_marker": "UNKNOWN", "reference_marker": "UNKNOWN",
            "guided_views": ["g", "h"], "unguided_views": ["c", "r"], "replicate": 0,
            "seed": seed, "measured": measured, "completed_at_utc": _now(), "status": "ok"})


if __name__ == "__main__":
    raise SystemExit(main())
