#!/usr/bin/env python3
"""Run the registered dev404 and VizWiz512 gaps through the frozen KDM chain."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.decoding import DecodeConfig
from kdm.frozen import load_contract
from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash, stable_seed, within
from kdm.pipeline import make_backend, run_tasks, task_id
from kdm.prompts import MARKERS, task_prompt
from workflows.paper_core.native_audit import input_description
from workflows.supplemental.remaining11.generate import validate_proofs
import workflows.paper_core.execution as execution

MODELS = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b")
DEV_METHODS = ("vcd", "instruction_vcd", "cda_visual", "m3id")
ROSTERS = {"dev404": "phrase_selection_dev404.csv", "viz512": "vizwiz_eval512.csv"}


def now():
    return datetime.now(timezone.utc).isoformat()


def roster(stage):
    path = ROOT / "workflows/paper_core" / ROSTERS[stage]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_n, dataset, split = (404, "food101", "dev") if stage == "dev404" else (512, "vizwiz", "eval")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != expected_n or len(set(ids)) != expected_n:
        raise ValueError("Fixed roster count or unique ID check failed")
    samples = {row["id"]: row for row in read_jsonl(ROOT / "data/current/all.jsonl")}
    selected = [samples[sid] for sid in ids]
    if any(sample["dataset"] != dataset or sample["split"] != split for sample in selected):
        raise ValueError("Fixed roster differs from the registered split")
    if stage == "dev404":
        if set(Counter(sample["class"] for sample in selected).values()) != {4}:
            raise ValueError("Food dev roster must contain four inputs in each of 101 classes")
        if len({sample["class"] for sample in selected}) != 101:
            raise ValueError("Food dev class count differs")
    else:
        if Counter(int(row["annotated_answerable"]) for row in rows) != {0: 166, 1: 346}:
            raise ValueError("VizWiz fixed official-answerability counts differ")
        for row, sample in zip(rows, selected):
            if "annotated_answerable" not in sample or int(sample["annotated_answerable"]) != int(row["annotated_answerable"]):
                raise ValueError("VizWiz answerability must match the registered official field")
    return path, selected


def task(sample, method, marker):
    native = method == "vcd_native"
    actual_method = "vcd" if native else method
    return dict(sample=sample, method=actual_method, marker="NONE" if native else marker,
                reference_marker="NONE" if native else marker, guided=not native,
                reference_guided=not native and actual_method in {"vcd", "m3id"},
                replicate=0, kind="native_unguided" if native else (
                    "instruction_preserving" if actual_method in {"instruction_vcd", "cda_visual"} else "main"))


def planned_tasks(model, stage, selected_path=None):
    _, samples = roster(stage)
    if stage == "dev404":
        return [task(sample, method, marker) for method in DEV_METHODS for marker in MARKERS for sample in samples]
    selected = json.loads(within(ROOT, selected_path).read_text(encoding="utf8"))
    chosen = {row["method"]: row["marker"] for row in selected if row["model"] == model}
    if set(chosen) != set(DEV_METHODS):
        raise ValueError("VizWiz requires all four complete Food-dev-selected configurations")
    configurations = [("vcd_native", "NONE"), *[(method, chosen[method]) for method in DEV_METHODS]]
    if chosen["vcd"] != chosen["instruction_vcd"]:
        configurations.append(("vcd", chosen["instruction_vcd"]))
    return [task(sample, method, marker) for method, marker in configurations for sample in samples]


def read_missing(path, model, tasks):
    expected = {task_id(model, row): row for row in tasks}
    missing = list(read_jsonl(within(ROOT, path)))
    keys = [row["key"] for row in missing]
    if not keys or len(keys) != len(set(keys)) or not set(keys) <= expected.keys():
        raise ValueError("Missing task list must be nonempty, unique, and within the registered panel")
    for row in missing:
        original = expected[row["key"]]
        if row.get("model", model) != model or row.get("sample_id", original["sample"]["id"]) != original["sample"]["id"]:
            raise ValueError("Missing task source identity differs")
    return [expected[key] for key in keys]


def condition(row):
    return stable_hash({key: value for key, value in row.items() if key != "sample"})[:16]


def audit_rows(model, tasks, raw):
    expected = {task_id(model, row): row for row in tasks}
    rows = list(read_jsonl(raw))
    if len(rows) != len(expected) or {row["key"] for row in rows} != expected.keys():
        raise ValueError("Completed condition has duplicate or missing keys")
    for row in rows:
        item = expected[row["key"]]
        if row["status"] != "ok" or not row["tokens"] or len(row["tokens"]) > 32:
            raise ValueError("Generation token or status evidence failed")
        if row["sample"] != item["sample"] or row["seed"] != stable_seed(item["sample"]["id"], model, 0):
            raise ValueError("Generated sample or perturbation seed differs")
        for field in ("method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate"):
            if row[field] != item[field]:
                raise ValueError("Generated condition identity differs: " + field)
        if row["prompt"] != task_prompt(item["sample"]["question"], item["marker"], item["guided"]):
            raise ValueError("Generated prompt differs")
        cfg = asdict(DecodeConfig(method=item["method"]))
        if item["method"] == "m3id":
            cfg["m3id_offset"] = len(row["offset_prompt_tokens"])
        if row["config"] != cfg or not isinstance(row["terminated"], bool):
            raise ValueError("Decode settings or termination evidence differs")
        if not all(math.isfinite(value) for value in row["selected_log_probabilities"]):
            raise ValueError("Generated selected probabilities are not finite")
        if len(row["selected_log_probabilities"]) != len(row["tokens"]):
            raise ValueError("Generated token/probability count differs")
    return rows


class PilotInputs:
    def __init__(self, backend, path):
        self.backend, self.path, self.current = backend, path, None

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def session(self, image, prompt, reference="clean", seed=0, need_layers=False):
        session = self.backend.session(image, prompt, reference=reference, seed=seed, need_layers=need_layers)
        with self.path.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(dict(sample_id=self.current["sample"]["id"],
                                        condition=condition(self.current), prompt=prompt, reference=reference,
                                        seed=seed, inputs=input_description(session.inputs)), allow_nan=False) + "\n")
        return session


def execute(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_id):
        raise ValueError("Invalid exclusive run name")
    if args.registry == "a100":
        execution.REGISTRY = "workflows/paper_core/host_registry_a100.json"
    cards = args.physical_gpus.split(",")
    frozen_spec = json.loads((ROOT / f"configs/runtime/{args.model}.json").read_text())
    manifest, freeze = load_contract(ROOT)
    proofs = validate_proofs(ROOT, frozen_spec, args.model, list(DEV_METHODS), "formal", manifest, freeze)
    admission = execution.admit(ROOT, frozen_spec, args.model, cards)
    spec = execution.runtime_spec(ROOT, frozen_spec, args.model)
    tasks = read_missing(args.missing_keys, args.model, planned_tasks(args.model, args.stage, args.selected_configs))
    roster_path, samples = roster(args.stage)
    base = ROOT / "outputs/paper_core_20261002_dev_viz" / args.run_id / args.model
    base.mkdir(parents=True, exist_ok=True)
    writer_lock = (base / "writer.lock").open("a")
    fcntl.flock(writer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(schema="kdm_core_dev_viz_identity_v1", stage=args.stage, model=args.model,
                    frozen_spec=frozen_spec, runtime_spec=spec, admission=admission["fixed_identity"], proofs=proofs,
                    roster_sha256=file_hash(roster_path), missing_keys_sha256=file_hash(within(ROOT, args.missing_keys)),
                    selected_configs_sha256=file_hash(within(ROOT, args.selected_configs)) if args.selected_configs else None,
                    decode=asdict(DecodeConfig()), noise_step=500, source_sha256=file_hash(Path(__file__)),
                    author={"agent": "/root", "model": "gpt-6.1-sol", "effort": "max", "call_id": ""})
    gate_path = base / "pilot_receipt.json"
    if args.mode == "full":
        gate = json.loads(gate_path.read_text())
        release = json.loads(within(ROOT, args.budget_release).read_text())
        if not gate["passed"] or gate["identity"] != stable_hash(identity) or not release["released"]:
            raise ValueError("Full execution requires the unchanged real pilot and released budget")
        if release["stage"] != args.stage or release["run_id"] != args.run_id:
            raise ValueError("Budget release belongs to another registered task")
        if args.stage == "dev404" and release["estimated_gpu_hours"] > 24 and not release.get("additional_user_authorization"):
            raise ValueError("An estimated dev budget above 24 GPU hours requires user authorization")
    elif gate_path.exists():
        raise FileExistsError("Existing completed pilot must be reused")
    atomic_json(base / "identity.json", identity)
    atomic_json(base / f"admission_{args.mode}.json", admission)
    groups = defaultdict(list)
    pilot_ids = {sample["id"] for sample in samples[:8]}
    for row in tasks:
        groups[condition(row)].append(row)
    started = time.perf_counter()
    backend = make_backend(spec, device="cuda:0")
    load_wall = time.perf_counter() - started
    if args.mode == "pilot":
        backend = PilotInputs(backend, base / "pilot_inputs.jsonl")
    cfg = DecodeConfig()
    timing = []
    for cid, all_tasks in groups.items():
        selected = [row for row in all_tasks if row["sample"]["id"] in pilot_ids] if args.mode == "pilot" else all_tasks
        if not selected:
            continue
        first = selected[0]
        label = f"{first['method']}_{MARKERS.index(first['marker']) if first['marker'] in MARKERS else 'NONE'}_{cid}"
        raw = base / "raw" / f"{label}.jsonl"
        before = time.perf_counter()
        if args.mode == "pilot":
            if raw.exists():
                raise FileExistsError("Existing partial pilot needs explicit evidence-based recovery")
            for row in selected:
                backend.current = row
                run_tasks(backend, args.model, [row], raw, identity, cfg=cfg)
        else:
            run_tasks(backend, args.model, selected, raw, identity, cfg=cfg)
        rows = audit_rows(args.model, selected, raw)
        detail = dict(condition=cid, method=first["method"], marker=first["marker"], rows=len(rows),
                      required=len(all_tasks), generation_wall_s=sum(row["wall_s"] for row in rows),
                      execution_wall_s=time.perf_counter() - before, raw=str(raw.relative_to(ROOT)), raw_sha256=file_hash(raw))
        timing.append(detail)
        if args.mode == "full":
            atomic_json(raw.with_suffix(".complete.json"), dict(passed=True, identity=stable_hash(identity), **detail, completed_at_utc=now()))
        else:
            sealed = base / "sealed_pilot" / raw.name
            sealed.parent.mkdir(parents=True, exist_ok=True)
            if sealed.exists():
                raise FileExistsError("An immutable completed pilot snapshot already exists")
            shutil.copy2(raw, sealed)
            shutil.copy2(raw.with_suffix(".identity.json"), sealed.with_suffix(".identity.json"))
            atomic_json(sealed.with_suffix(".complete.json"), dict(passed=True, identity=stable_hash(identity), **{
                **detail, "raw": str(sealed.relative_to(ROOT))}, completed_at_utc=now(), scope="pilot_only"))
        atomic_json(base / "CURRENT_STATE.json", dict(stage=args.stage, model=args.model, mode=args.mode,
                                                      conditions_completed=timing, required_missing=len(tasks), updated_utc=now()))
    receipt = dict(passed=True, identity=stable_hash(identity), model=args.model, stage=args.stage,
                   actual_rows=sum(row["rows"] for row in timing), required_missing=len(tasks), conditions=timing,
                   model_load_wall_s=load_wall, elapsed_s=time.perf_counter() - started, completed_at_utc=now())
    atomic_json(gate_path if args.mode == "pilot" else base / "complete_receipt.json", receipt)
    writer_lock.close()
    print(json.dumps(receipt, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=ROSTERS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpus")
    parser.add_argument("--registry", choices=("core", "a100"), default="core")
    parser.add_argument("--missing-keys")
    parser.add_argument("--selected-configs")
    parser.add_argument("--mode", choices=("plan", "pilot", "full"), default="pilot")
    parser.add_argument("--budget-release")
    args = parser.parse_args()
    if args.mode == "plan":
        path, samples = roster(args.stage)
        tasks = planned_tasks(args.model, args.stage, args.selected_configs) if args.stage == "dev404" or args.selected_configs else []
        keys = [task_id(args.model, row) for row in tasks]
        if len(keys) != len(set(keys)):
            raise ValueError("Planned tasks contain duplicate keys")
        print(json.dumps(dict(stage=args.stage, model=args.model, inputs=len(samples), planned_tasks=len(tasks),
                              roster_sha256=file_hash(path), GPU_mutations=0, generations_completed=0)))
        return
    if not args.physical_gpus or not args.missing_keys:
        parser.error("Execution requires allocated physical GPUs and an explicit missing-key manifest")
    if args.stage == "viz512" and not args.selected_configs:
        parser.error("VizWiz must use frozen Food-dev-selected configurations")
    if args.mode == "full" and not args.budget_release:
        parser.error("Full execution requires a measured budget release")
    execute(args)


if __name__ == "__main__":
    main()
