#!/usr/bin/env python3
"""Generate core4 native M3ID VizWiz conditions on their original 4028 route.

Reuse the completed Food runner's native operator, prompt-offset clock and
retained eight-row gate. Model algorithms and frozen runtimes are unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
OUT = ROOT / "outputs/supplemental/native_m3id_viz_20261010"
WORKFLOW = ROOT / "workflows/supplemental/native_m3id_viz_20261010"
MODELS = ("qwen25vl", "qwen35_4b", "minicpm26", "gemma3_4b")
MODEL_CARDS = {"qwen25vl": "0", "qwen35_4b": "1", "minicpm26": "4", "gemma3_4b": "5"}
ROWS_PER_MODEL = 512
AUTHOR = {"agent": "/root/guided_std_audit", "role": "Codex implementation agent",
          "model_identifier": None, "reasoning_effort": None}
VIZ_IMAGE_PREFIX = "/home/g203-4028/projects/knowledge-deficit-mitigation/cache/assets/vizwiz/extracted/val"
VIZ_IMAGE_DESTINATION = "cache/assets/vizwiz/extracted/val"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    from kdm.io import atomic_json
    atomic_json(Path(path), value)


def native_task(sample):
    return dict(sample=sample, method="m3id", kind="native_unguided",
                marker="NONE", reference_marker="NONE", guided=False,
                reference_guided=False, replicate=0)


def select_samples(all_samples, roster):
    ids = roster.get("sample_ids") if isinstance(roster, dict) else roster
    if (not isinstance(ids, list) or len(ids) != ROWS_PER_MODEL
            or any(not isinstance(sid, str) or not sid for sid in ids)
            or len(set(ids)) != ROWS_PER_MODEL):
        raise ValueError("Exactly 512 explicit unique sample IDs are required")
    selected = {}
    for sample in all_samples:
        if sample.get("id") not in set(ids):
            continue
        if (sample.get("dataset") != "vizwiz" or sample.get("split") != "eval"
                or sample["id"] in selected):
            raise ValueError("Fixed sample identity is not unique current VizWiz eval")
        selected[sample["id"]] = sample
    if set(selected) != set(ids):
        raise ValueError("Some fixed VizWiz IDs are absent from current all.jsonl")
    samples = [selected[sid] for sid in ids]
    references = roster.get("samples_reference") if isinstance(roster, dict) else None
    if references is not None:
        if not isinstance(references, list) or len(references) != ROWS_PER_MODEL:
            raise ValueError("Fixed sample reference length differs")
        by_id = {sample["id"]: sample for sample in references}
        if len(by_id) != ROWS_PER_MODEL or set(by_id) != set(ids):
            raise ValueError("Fixed sample reference identities differ")
        for sample in samples:
            reference = by_id[sample["id"]]
            if any(reference.get(key) != sample.get(key)
                   for key in ("question", "gold", "annotated_answerable")):
                raise ValueError("Fixed sample question/gold/answerability differs from current all.jsonl")
    return samples


def prepare(sample_ids_path=None):
    from prepare_4028 import prepare as register
    register(sample_ids_path)


def worker(model, cards):
    import torch
    from kdm.decoding import DecodeConfig
    from kdm.frozen import load_contract
    from kdm.io import file_hash, read_jsonl, stable_hash, stable_seed
    from kdm.pipeline import make_backend, run_tasks, task_id
    from kdm.prompts import task_prompt
    from workflows.supplemental.remaining11.generate import validate_proofs
    import native_4028_admission as execution
    from workflows.supplemental.remaining4.native_audit import NativeAuditBackend
    reg = read(OUT / "SOURCE.json")
    if model not in reg["models"] or len(cards) != reg["frozen_specs"][model]["gpu_count"]:
        raise ValueError("Worker model or physical GPU count differs from registration")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(cards):
        raise ValueError("Visible devices differ from the explicitly locked physical cards")
    directory = OUT / "models" / model
    directory.mkdir(exist_ok=True)
    with (directory / "claim.json").open("x", encoding="utf-8") as f:
        json.dump(dict(model=model, cards=cards, pid=os.getpid(), argv=sys.argv,
                       owner=AUTHOR, started_utc=now()), f)
    started = time.perf_counter()
    try:
        for name, expected in {**reg["source_hashes"], **reg["input_hashes"]}.items():
            if file_hash(ROOT / name) != expected:
                raise ValueError("Registered source or input changed: " + name)
        if file_hash(OUT / "host_registry.json") != reg["admission_registry_sha256"]:
            raise ValueError("Native admission registry changed")
        manifest, frozen = load_contract(ROOT)
        spec = reg["frozen_specs"][model]
        proofs = validate_proofs(ROOT, spec, model, ["m3id"], "formal", manifest, frozen)
        execution.REGISTRY = str((OUT / "host_registry.json").relative_to(ROOT))
        os.environ["KDM_SUPPLEMENTAL_DATASET"] = "vizwiz"
        os.environ["KDM_SUPPLEMENTAL_METHOD"] = "m3id"
        admitted = execution.validate_supplemental_runtime(ROOT, spec, model, cards,
                                                         "formal", "native_m3id_viz_20261010", "/root")
        write(directory / "admission.json", dict(admission=admitted, inherited_proofs=proofs, observed_utc=now()))
        actual = admitted["runtime_spec"]
        cfg = DecodeConfig(method="m3id")
        identity = dict(registration_sha256=file_hash(OUT / "SOURCE.json"), model=model,
                        task=reg["task"], config=asdict(cfg), runtime_spec=actual,
                        clock=reg["clock"], authors=[AUTHOR])
        write(directory / "identity.json", identity)
        raw = directory / "raw.jsonl"
        tasks = [native_task(s) for s in reg["samples"]]
        write(directory / "CURRENT_STATE.json", dict(status="loading", updated_utc=now()))
        load_started = time.perf_counter()
        backend = make_backend(actual, "cuda:0")
        for device in range(spec["gpu_count"]):
            torch.cuda.synchronize(device)
        load_s = time.perf_counter() - load_started
        devices = Counter(str(p.device) for p in backend.model.parameters())
        dtypes = Counter(str(p.dtype) for p in backend.model.parameters() if p.is_floating_point())
        if (set(dtypes) != {"torch.bfloat16"}
                or set(devices) != {f"cuda:{i}" for i in range(spec["gpu_count"])}):
            raise ValueError("Loaded native precision or placement differs")
        write(directory / "loaded.json", dict(model_load_seconds=load_s, parameter_devices=dict(devices),
                                               parameter_dtypes=dict(dtypes), loaded_utc=now()))
        audit = NativeAuditBackend(backend, model, tasks[:8], cfg)
        run_tasks(audit, model, tasks[:8], raw, identity, cfg=cfg)
        pilot = list(read_jsonl(raw))
        proof = audit.proof(pilot)
        proof.update(registration_sha256=file_hash(OUT / "SOURCE.json"), identity_sha256=stable_hash(identity),
                     pilot_retained_in_full=True, recorded_utc=now())
        write(directory / "operator_gate8.json", proof)
        del audit
        write(directory / "CURRENT_STATE.json", dict(status="generating", completed=8, expected=ROWS_PER_MODEL,
                                                      updated_utc=now()))
        print(json.dumps(dict(event="gate_passed", model=model, pilot_rows=8,
                              pilot_seconds=sum(r["wall_s"] for r in pilot), model_load_seconds=load_s)), flush=True)
        run_tasks(backend, model, tasks, raw, identity, cfg=cfg)
        rows = list(read_jsonl(raw))
        if (len(rows) != ROWS_PER_MODEL or len({r["key"] for r in rows}) != ROWS_PER_MODEL
                or {r["key"] for r in rows} != {task_id(model, t) for t in tasks}):
            raise ValueError("Native complete input/key coverage differs")
        for row in rows:
            prompt = task_prompt(row["sample"]["question"], guided=False)
            expected_config = asdict(cfg)
            expected_config["m3id_offset"] = len(row["offset_prompt_tokens"])
            if (row["prompt"] != prompt or row["reference_prompt"] != prompt
                    or row["guided"] or row["reference_guided"] or row["replicate"] != 0
                    or row["config"] != expected_config or row["status"] != "ok"
                    or row["seed"] != stable_seed(row["sample"]["id"], model, 0)
                    or not 0 < len(row["tokens"]) <= 32
                    or row["terminated"] != (row["tokens"][-1] in backend.eos)):
                raise ValueError("Native output prompt, configuration or termination differs")
        receipt = dict(passed=True, model=model, completed=ROWS_PER_MODEL, expected=ROWS_PER_MODEL,
                       raw_sha256=file_hash(raw), source_sha256=file_hash(OUT / "SOURCE.json"),
                       actual_operator_gate_sha256=file_hash(directory / "operator_gate8.json"),
                       generation_seconds=sum(r["wall_s"] for r in rows), model_load_seconds=load_s,
                       elapsed_seconds=time.perf_counter()-started, eos_count=sum(r["terminated"] for r in rows),
                       max_token_count=sum(not r["terminated"] for r in rows), completed_utc=now())
        write(directory / "complete.json", receipt)
        write(directory / "CURRENT_STATE.json", dict(status="complete", **receipt))
        print(json.dumps(receipt), flush=True)
    except Exception as exc:
        failure = dict(status="failed", model=model, error=type(exc).__name__+": "+str(exc),
                       traceback=traceback.format_exc(), failed_utc=now(), automatic_retry=False,
                       automatic_parameter_changes=False)
        write(directory / "failure.json", failure)
        write(directory / "CURRENT_STATE.json", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker"))
    parser.add_argument("--sample-ids")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--cards")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.sample_ids)
    else:
        expected = (MODEL_CARDS.get(args.model),)
        if not args.model or args.cards not in expected:
            parser.error("Each core4 model requires its fixed original physical card: Q25=0, Q35=1, Mini=4, Gemma=5")
        worker(args.model, args.cards.split(","))


if __name__ == "__main__":
    main()
