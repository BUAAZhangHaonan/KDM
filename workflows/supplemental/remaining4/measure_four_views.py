#!/usr/bin/env python3
"""Run a finite four-view representative panel with the original algorithms."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workflows/paper_core"))
from PIL import Image
from kdm.decoding import DecodeConfig, generate
from kdm.execution import resolve_image_path
from kdm.io import Ledger, atomic_json, file_hash, read_jsonl, stable_hash, stable_seed, within
from kdm.pipeline import make_backend
from kdm.prompts import task_prompt
from workflows.paper_core.attribution import measure_attribution
from workflows.paper_core.run_attribution_details import _BudgetBackend
from workflows.supplemental.remaining4.native import MODELS, source_inputs
from workflows.supplemental.remaining11 import execution as supplemental_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physical-gpus", required=True)
    parser.add_argument("--host-registry", required=True)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--limit", type=int, choices=(8, 101), required=True)
    parser.add_argument("--prior-budget", type=Path)
    parser.add_argument("--key-start", type=int, choices=(0, 8), default=0)
    args = parser.parse_args()
    run = within(ROOT, args.out)
    run.mkdir(parents=True, exist_ok=False)
    rows = [row for row in read_jsonl(args.inputs) if row["model"] == args.model]
    if (len(rows) != 101 or len({row["sample"]["id"] for row in rows}) != 101
            or len(Counter(row["sample"]["class"] for row in rows)) != 101):
        raise ValueError("Exactly 101 class-balanced frozen representative sources required")
    _, frozen_spec, _, provenance = source_inputs(args.model, "vcd")
    cards = args.physical_gpus.split(",")
    supplemental_execution.REGISTRY = str(within(ROOT, args.host_registry).relative_to(ROOT))
    os.environ["KDM_SUPPLEMENTAL_DATASET"] = "food101"
    admitted = supplemental_execution.validate_supplemental_runtime(
        ROOT, frozen_spec, args.model, cards, "attribution", args.claim_id, args.owner)
    prior = json.loads(args.prior_budget.read_text()) if args.prior_budget else {
        "model": args.model, "gpu_count": len(cards), "used_gpu_seconds": 0.0}
    if (prior["model"] != args.model or prior["gpu_count"] != len(cards)
            or not 0 <= prior["used_gpu_seconds"] < 2880):
        raise ValueError("Invalid or exhausted original per-model attribution budget")
    prior_complete = None
    if args.key_start:
        if args.limit != 101 or not args.prior_budget:
            raise ValueError("Continuation requires the successful eight-input budget source")
        prior_complete_path = args.prior_budget.parent / "complete.json"
        prior_complete = json.loads(prior_complete_path.read_text())
        if (prior_complete["model"] != args.model or prior_complete["limit"] != 8
                or not prior_complete["first_eight_gate_passed"]
                or prior_complete["outputs"] != {"representative": 8, "natural_reference": 8}):
            raise ValueError("First-eight source gate is incomplete")
        for phase in ("representative.events.jsonl", "natural_reference.events.jsonl"):
            original_rows = list(read_jsonl(args.prior_budget.parent / phase))
            if len(original_rows) != 8 or {row["sample_id"] for row in original_rows} != {row["sample"]["id"] for row in rows[:8]}:
                raise ValueError("First-eight completed source keys differ")
    identity = {"schema": "kdm_remaining4_four_view_representative_v1", "model": args.model,
                "claim_id": args.claim_id, "owner": args.owner, "marker": "UNKNOWN",
                "source_inputs_sha256": file_hash(args.inputs), "provenance": provenance,
                "admission": admitted["execution"], "runner_sha256": file_hash(Path(__file__)),
                "measurement_sha256": file_hash(ROOT / "workflows/paper_core/attribution.py"),
                "formula_sha256": file_hash(ROOT / "workflows/paper_core/four_view_math.py"),
                "cap_gpu_seconds": 2880, "new_guided_vcd_generation": False,
                "key_start": args.key_start, "prior_eight_directory": str(args.prior_budget.parent) if args.key_start else None}
    atomic_json(run / "admission.json", identity)
    representative = Ledger(run / "representative.events.jsonl", {**identity, "phase": "representative_first_position"})
    natural = Ledger(run / "natural_reference.events.jsonl", {**identity, "phase": "natural_reference"})
    started = time.perf_counter()
    outputs = {"representative": 0, "natural_reference": 0}
    try:
        remaining = 2880 - prior["used_gpu_seconds"]
        backend = _BudgetBackend(make_backend(admitted["runtime_spec"], "cuda:0"), started, remaining, cards)
        cfg = DecodeConfig(method="direct", max_tokens=32, temperature=0.0, top_p=1.0)
        for row in rows[args.key_start:args.limit]:
            sample = row["sample"]
            seed = stable_seed(sample["id"], args.model, 0)
            if row["seed"] != seed or sample["split"] != "eval" or sample["dataset"] != "food101":
                raise ValueError("Representative sample/seed differs")
            if not row["native_vcd"]["tokens"]:
                raise ValueError("Native generated token path is empty")
            answer_token = row["native_vcd"]["tokens"][0]
            marker_tokens = list(backend.encode("UNKNOWN"))
            pair = [(answer_token, marker_tokens[0])] if answer_token != marker_tokens[0] else []
            with Image.open(resolve_image_path(sample["image_path"], ROOT)) as source:
                image = source.convert("RGB")
                measurement = measure_attribution(
                    backend, image, sample["question"], [()], seed,
                    candidate_token_ids=[answer_token, *marker_tokens[:1]], candidate_pairs=pair,
                    source_meta={"model": args.model, "sample_id": sample["id"],
                                 "panel": "representative101", "source": row,
                                 "candidate_pair_semantics": "native answer first token versus literal marker proxy; semantic abstention scored separately"})
                representative.add(stable_hash([args.model, sample["id"], "representative101"]),
                                   {"model": args.model, "sample_id": sample["id"], "dataset": "food101",
                                    "split": "eval", "seed": seed, "panel": "representative101",
                                    "source": row, "measured": measurement, "status": "ok"})
                prompt = task_prompt(sample["question"], guided=False)
                session = backend.session(image, prompt, reference="noise", seed=seed)
                result = generate(session, None, cfg, backend.eos, backend.decode, seed)
                natural.add(stable_hash([args.model, sample["id"], "natural101"]),
                            {"model": args.model, "sample_id": sample["id"], "sample": sample,
                             "dataset": "food101", "split": "eval", "seed": seed, "panel": "natural101",
                             "prompt": prompt, "config": asdict(cfg), "noise_generated_r": result,
                             "source": row, "status": "ok"})
            outputs = {"representative": len(representative.keys), "natural_reference": len(natural.keys)}
    finally:
        elapsed = (time.perf_counter() - started) * len(cards)
        budget = {"model": args.model, "gpu_count": len(cards),
                  "used_gpu_seconds": prior["used_gpu_seconds"] + elapsed,
                  "last_call_gpu_seconds": elapsed, "cap_gpu_seconds": 2880,
                  "includes": "model loading, all four forwards, natural reference generation and failures",
                  "updated_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(run / "budget.json", budget)
    expected_this_call = args.limit - args.key_start
    atomic_json(run / "complete.json", {"model": args.model, "outputs": outputs, "limit": args.limit,
                "prior_completed_inputs": args.key_start,
                "total_completed": {phase: value + args.key_start for phase, value in outputs.items()},
                "prior_eight_source": identity["prior_eight_directory"],
                "expected_full_panel": 101, "first_eight_gate_passed": all(value >= 8 for value in outputs.values()),
                "gpu_phase_complete_for_limit": all(value == expected_this_call for value in outputs.values()),
                "diagnostic_and_cases": "separate finite actual-source tasks", "budget": budget,
                "labels_complete": False, "finished_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"run": str(run), "outputs": outputs, "budget": budget}))


if __name__ == "__main__":
    main()
