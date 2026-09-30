#!/usr/bin/env python3
"""Thin P1_NATIVE runner.

The entry point owns planning, immutable run identity, and the disk gate.  Model
execution remains delegated to the registered KDM backend and pipeline.  It
does not change ``src/kdm`` or any frozen configuration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.decoding import DecodeConfig
from kdm.frozen import load_contract
from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash
from kdm.pipeline import make_backend, run_tasks, task_id
from native_audit import NativeAuditBackend

MODELS = (
    "qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b",
)
DATASET = "food101"
NATIVE_LIMITS = (8, 2424)
IDENTITY_SCHEMA = "kdm_p1_native_vcd_identity_v1"


def _execution_api():
    """Resolve the host admission implementation supplied by the core runner."""
    from execution import admit, runtime_spec
    return runtime_spec, admit


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cards(value: str) -> list[str]:
    cards = [part.strip() for part in value.split(",") if part.strip()]
    if not cards:
        raise ValueError("--physical-gpus must contain at least one card")
    if len(cards) != len(set(cards)):
        raise ValueError("--physical-gpus contains a duplicate card")
    return cards


def _samples() -> list[dict]:
    path = ROOT / "data/current/all.jsonl"
    rows = [row for row in read_jsonl(path)
            if row.get("dataset") == DATASET and row.get("split") == "eval"]
    if len(rows) != 2424:
        raise ValueError(f"P1_NATIVE Food-101 eval manifest must contain 2424 rows, got {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("P1_NATIVE eval manifest contains duplicate ids")
    return rows


def _task(sample: dict) -> dict:
    # NONE is the registered unmodified marker.  Keeping guided false on both
    # sides is part of the native condition and is included in the task key.
    return {
        "sample": sample,
        "method": "vcd",
        "marker": "NONE",
        "reference_marker": "NONE",
        "guided": False,
        "reference_guided": False,
        "replicate": 0,
        "kind": "native_unguided",
    }


def _fixed_admission(admission: dict) -> dict:
    return admission["fixed_identity"]


def _identity(model: str, frozen_spec: dict, spec: dict, admission: dict) -> dict:
    cfg = DecodeConfig(method="vcd", alpha=1.0, beta=0.1, temperature=0.0,
                       top_p=1.0, max_tokens=32)
    return {
        "schema": IDENTITY_SCHEMA,
        "protocol": "P1_NATIVE",
        "dataset": DATASET,
        "model": model,
        "frozen_spec": frozen_spec,
        "frozen_spec_sha256": stable_hash(frozen_spec),
        "model_spec": spec,
        "model_spec_sha256": stable_hash(spec),
        "admission": _fixed_admission(admission),
        "decode_config": asdict(cfg),
        "condition": {
            "guided": False, "marker": "NONE", "reference_marker": "NONE",
            "noise_step": 500, "reference": "noise",
        },
        "source": {
            "eval_manifest": "data/current/all.jsonl",
            "eval_manifest_sha256": file_hash(ROOT / "data/current/all.jsonl"),
            "core_registry": _fixed_admission(admission)["registry_sha256"],
            "core_source_sha256": _fixed_admission(admission)["admission_source_sha256"],
            "native_source_sha256": file_hash(Path(__file__)),
            "native_audit_source_sha256": file_hash(Path(__file__).with_name("native_audit.py")),
        },
        "code_authors": [{"agent": "/root/scoring", "model": "gpt-5.6-luna",
                          "effort": "medium", "call_id": ""},
                         {"agent": "/root", "model": "gpt-6.1-sol",
                          "effort": "max", "call_id": ""}],
        # run_id and limit deliberately do not occur in the identity.
    }


def _gate_path(run_dir: Path, model: str) -> Path:
    return run_dir / f"{model}.gate_8.json"


def check_plan(model: str, run_id: str, cards: list[str]) -> dict:
    manifest, frozen = load_contract(ROOT)
    runtime_spec, admit = _execution_api()
    frozen_spec = json.loads((ROOT / f"configs/runtime/{model}.json").read_text())
    spec = runtime_spec(ROOT, frozen_spec, model)
    admission = admit(ROOT, frozen_spec, model, cards)
    identity = _identity(model, frozen_spec, spec, admission)
    return {
        "schema": "kdm_p1_native_check_plan_v1", "run_id": run_id,
        "model": model, "limit_choices": list(NATIVE_LIMITS),
        "eval_rows": 2424, "identity": identity,
        "manifest_entries": len(manifest.get("entries", [])),
        "admission": admission, "checked_at": _now(),
    }


def execute(model: str, run_id: str, limit: int, cards: list[str]) -> dict:
    manifest, frozen = load_contract(ROOT)
    runtime_spec, admit = _execution_api()
    frozen_spec = json.loads((ROOT / f"configs/runtime/{model}.json").read_text())
    spec = runtime_spec(ROOT, frozen_spec, model)
    admission = admit(ROOT, frozen_spec, model, cards)
    identity = _identity(model, frozen_spec, spec, admission)
    run_dir = ROOT / "outputs/paper_core_20260930" / run_id
    raw_path = run_dir / "raw" / f"{model}_native_vcd.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    samples = _samples()
    tasks = [_task(sample) for sample in samples]
    gate = _gate_path(run_dir, model)
    if limit == 2424:
        proof = json.loads(gate.read_text(encoding="utf-8"))
        if not proof.get("passed") or proof["identity"] != stable_hash(identity):
            raise RuntimeError("P1_NATIVE full run requires eight real gates with the same identity")
    if limit == 8 and raw_path.exists():
        raise FileExistsError("Preserve existing eight-input gate; no automatic regeneration")
    atomic_json(run_dir / f"{model}.admission_{limit}.json", admission)
    atomic_json(run_dir / f"{model}.identity.json", identity)
    load_started = time.perf_counter()
    backend = make_backend(spec, device="cuda:0")
    load_wall = time.perf_counter() - load_started
    if limit == 8:
        backend = NativeAuditBackend(backend, model, samples[:8])
    cfg = DecodeConfig(method="vcd", alpha=1.0, beta=0.1, temperature=0.0,
                       top_p=1.0, max_tokens=32)
    started = time.perf_counter()
    selected = tasks if limit == 2424 else tasks[:8]
    count = run_tasks(backend, model, selected, raw_path, identity, cfg=cfg)
    rows = list(read_jsonl(raw_path))
    if len(rows) != limit or len({row["key"] for row in rows}) != limit:
        raise ValueError("Native generation coverage or duplicate audit failed")
    if {row["key"] for row in rows} != {task_id(model, task) for task in selected}:
        raise ValueError("Native generation keys differ from the frozen eval input list")
    if limit == 8:
        proof = {"model": model, "identity": stable_hash(identity), "required": 8,
                 **backend.proof(rows, backend.eos), "model_load_wall_s": load_wall,
                 "written_at_utc": _now()}
        atomic_json(gate, proof)
    else:
        atomic_json(run_dir / f"{model}.complete.json", {
            "model": model, "rows": len(rows), "expected": 2424,
            "raw_sha256": file_hash(raw_path), "identity": stable_hash(identity),
            "gate_identity": proof["identity"], "completed_at_utc": _now()})
    return {"schema": "kdm_p1_native_execute_v1", "model": model,
            "run_id": run_id, "limit": limit, "rows": count,
            "raw": str(raw_path), "gate": {"path": str(gate), "passed": proof["passed"], "completed": proof["completed"]},
            "elapsed_s": time.perf_counter() - started}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpus", required=True)
    parser.add_argument("--limit", type=int, choices=NATIVE_LIMITS, default=8)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check-plan", action="store_true")
    args = parser.parse_args(argv)
    cards = _cards(args.physical_gpus)
    if args.execute == args.check_plan:
        parser.error("choose exactly one of --execute or --check-plan")
    result = (execute(args.model, args.run_id, args.limit, cards)
              if args.execute else check_plan(args.model, args.run_id, cards))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
