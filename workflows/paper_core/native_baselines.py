#!/usr/bin/env python3
"""Run registered native DoLa/DeCo/SID on explicit immutable Food eval parts.

Only scheduling and actual-input observations are added. The frozen task,
model adapters, layer projections, SID attention and decoder remain unchanged.
An eight-input pilot is retained in the same part and counted in its full run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.decoding import DecodeConfig
from kdm.frozen import load_contract
from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash, stable_seed
from kdm.pipeline import make_backend, run_tasks, task_id
from kdm.probability import log_normalize
from kdm.prompts import task_prompt
from native_audit import input_description
from workflows.supplemental.remaining11.generate import validate_proofs

MODELS = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b")
METHODS = ("dola", "deco", "sid")
OUTPUT = "outputs/paper_core_20261002_dev_viz/native_baselines"
ORIGINAL_REGISTRY = "configs/runtime/hosts.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def task(sample, method):
    return {"sample": sample, "method": method, "marker": "NONE",
            "reference_marker": "NONE", "guided": False,
            "reference_guided": False, "replicate": 0, "kind": "native_unguided"}


def inputs(model, method):
    manifest, frozen = load_contract(ROOT)
    for name in ("data/current/all.jsonl", "configs/kdm/study.json",
                 "configs/kdm/method_plan.json", f"configs/runtime/{model}.json"):
        if frozen["files"].get(name) != file_hash(ROOT / name):
            raise ValueError("Frozen registered input changed: " + name)
    spec = json.loads((ROOT / f"configs/runtime/{model}.json").read_text())
    methods = json.loads((ROOT / "configs/kdm/method_plan.json").read_text())[model]["food101"]
    if method not in methods or (method == "sid" and model not in ("qwen25vl", "llava16_mistral")):
        raise ValueError("Native method is outside the registered five-model support")
    proof = validate_proofs(ROOT, spec, model, [method], "formal", manifest, frozen)
    samples = [s for s in read_jsonl(ROOT / "data/current/all.jsonl")
               if s["dataset"] == "food101" and s["split"] == "eval"]
    if (len(samples) != 2424 or len({s["id"] for s in samples}) != 2424
            or len(Counter(s["class"] for s in samples)) != 101
            or set(Counter(s["class"] for s in samples).values()) != {24}):
        raise ValueError("Food eval input ids or 101-class quotas differ")
    recorded = {}
    for item in frozen["source_blobs"]:
        metadata, name = item.split("\t", 1)
        recorded[name] = metadata.split(" ")[1]
    algorithms = {}
    for name in ("src/kdm/pipeline.py", "src/kdm/decoding.py", "src/kdm/prompts.py",
                 "src/kdm/probability.py"):
        blob = subprocess.check_output(["git", "hash-object", str(ROOT / name)], text=True).strip()
        if recorded.get(name) != blob:
            raise ValueError("Frozen native algorithm changed: " + name)
        algorithms[name] = blob
    return samples, spec, {"freeze_contract_sha256": manifest["original_contract_sha256"],
                           "verified_method_proof_sha256": proof,
                           "algorithm_git_blobs": algorithms}


def admission(spec, model, cards, registry):
    if registry == ORIGINAL_REGISTRY:
        from kdm.protocol import validate_execution_runtime
        from workflows.supplemental.remaining11.execution import checkpoint_identity
        value = validate_execution_runtime(ROOT, spec, model, cards)
        identity = checkpoint_identity(spec)
        return spec, {"fixed_identity": {"registry_sha256": file_hash(ROOT / registry),
                       "environment": {"environment_python": spec["environment_python"],
                                       "versions": spec["versions"]},
                       "checkpoint": identity}, "execution": value,
                       "observed_at_utc": now()}
    import execution
    execution.REGISTRY = registry
    actual = execution.runtime_spec(ROOT, spec, model)
    value = execution.admit(ROOT, spec, model, cards)
    # The location adapter also checks VCD's native interface. Target methods
    # are independently admitted by inputs() and observed by this pilot.
    return actual, value


class AuditBackend:
    def __init__(self, backend, model, method, samples):
        self.backend, self.model, self.method = backend, model, method
        self.samples = samples
        self.records = []
        self.clean_logits = {}

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def session(self, image, prompt, reference="clean", seed=0, need_layers=False):
        session = self.backend.session(image, prompt, reference=reference, seed=seed, need_layers=need_layers)
        if reference == "clean":
            index = len(self.records)
            sample = self.samples[index]
            if prompt != task_prompt(sample["question"], guided=False):
                raise ValueError("Native main prompt differs from registered unguided text")
            if need_layers != (self.method in ("dola", "deco")):
                raise ValueError("Native layer output request differs")
            self.records.append({"sample_id": sample["id"], "main_prompt": prompt,
                                 "main_inputs": input_description(session.inputs),
                                 "prompt_token_ids": session.inputs["input_ids"].detach().cpu().tolist(),
                                 "step_checks": [], "sid_attention_checks": []})
        elif reference == "sid" and self.method == "sid":
            index = len(self.records) - 1
            record = self.records[index]
            if (prompt != record["main_prompt"] or need_layers
                    or seed != stable_seed(record["sample_id"], self.model, 0)):
                raise ValueError("Native SID prompt, seed or layer request differs")
            if input_description(session.inputs) != record["main_inputs"]:
                raise ValueError("SID clean/reference processor inputs differ")
            record.update(reference_prompt=prompt, reference_seed=seed,
                          reference_inputs_equal=True, sid_layer_path=session.control.layer_path,
                          sid_decoder_layers=len(session.control.layers))
            session.control.audit = lambda event: self.sid_event(index, event)
        else:
            raise ValueError("Unexpected native method reference")
        return AuditSession(self, session, index, reference)

    def sid_event(self, index, event):
        from kdm.models.sid import keep_visual_indices, restricted_attention_mask
        torch = self.backend.torch
        attention = event["attention"]
        if not bool(torch.isfinite(attention).all()):
            raise ValueError("Actual SID attention is not finite")
        expected = keep_visual_indices(attention, event["start"], event["length"])
        if not torch.equal(expected, event["selected"]) or len(expected) != 100:
            raise ValueError("Actual SID rank-100 attention selection differs")
        # Reconstruct the exact original-mask restrictions using its actual
        # query length, KV span, dtype and device; no alternate attention path.
        mask = event["mask"]
        hidden = torch.empty((1, event["query_length"], 1), dtype=mask.dtype, device=mask.device)
        expected_mask = restricted_attention_mask(torch, event["original_mask"], hidden,
                                                  event["kv_length"], event["start"],
                                                  event["length"], expected)
        if not torch.equal(mask, expected_mask):
            raise ValueError("Actual SID causal mask differs")
        record = self.records[index]
        prefix = list(self.current_prefix)
        events = record["sid_attention_checks"]
        if not events or events[-1]["prefix_token_ids"] != prefix:
            events.append({"prefix_token_ids": prefix, "attention_shape": list(attention.shape),
                           "visual_start": event["start"], "visual_length": event["length"],
                           "query_length": event["query_length"], "kv_length": event["kv_length"],
                           "selected_visual_indices": expected.detach().cpu().tolist(),
                           "downstream_layers_checked": [], "rank100_equal": True,
                           "causal_mask_equal": True, "attention_finite": True})
        events[-1]["downstream_layers_checked"].append(event["layer"])

    def complete(self, rows):
        if len(self.records) != 8 or len(rows) != 8:
            raise ValueError("Native pilot must include eight actual inputs")
        by_id = {row["sample"]["id"]: row for row in rows}
        for record in self.records:
            row = by_id[record["sample_id"]]
            checks = record["step_checks"]
            if not checks or len(checks) != len(row["tokens"]):
                raise ValueError("Native pilot lacks actual token-step checks")
            if [c["argmax"] for c in checks] != row["tokens"]:
                raise ValueError("Native generated tokens differ from the registered operator")
            if self.method == "sid":
                events = record["sid_attention_checks"]
                if len(events) != len(row["tokens"]):
                    raise ValueError("Native SID pilot lacks real attention at each generated token")
                expected_layers = list(range(2, record["sid_decoder_layers"]))
                if any(e["downstream_layers_checked"] != expected_layers for e in events):
                    raise ValueError("SID did not check all downstream decoder layers")
            record.update(tokens=row["tokens"], text=row["text"], config=row["config"], seed=row["seed"],
                          terminated=row["terminated"], truncated=not row["terminated"],
                          eos_token_ids=sorted(self.backend.eos), wall_s=row["wall_s"])
        if self.clean_logits:
            raise ValueError("Native SID has unmatched clean/reference states")
        return {"passed": True, "completed": 8, "records": self.records,
                "generation_wall_s": sum(row["wall_s"] for row in rows)}


class AuditSession:
    def __init__(self, owner, session, index, reference):
        self.owner, self.session, self.index, self.reference = owner, session, index, reference

    def next(self, prefix):
        self.owner.current_prefix = tuple(prefix)
        step = self.session.next(prefix)
        logits = np.asarray(step.logits, dtype=np.float64)
        if not np.isfinite(logits).all():
            raise ValueError("Actual native full-vocabulary logits are not finite")
        method = self.owner.method
        p = log_normalize(logits)
        record = self.owner.records[self.index]
        if method == "sid" and self.reference == "clean":
            self.owner.clean_logits[(self.index, tuple(prefix))] = p
            return step
        selected_layer, weight = None, 1.0
        if method == "sid":
            clean = self.owner.clean_logits.pop((self.index, tuple(prefix)))
            keep = clean >= clean.max() + np.log(.1)
            score = 2 * clean - p
        elif method == "dola":
            if not step.early_raw or any(not np.isfinite(z).all() for z in step.early_raw.values()):
                raise ValueError("Native DoLa lacks finite real raw layer projections")
            layers = {i: log_normalize(np.asarray(z, dtype=np.float64)) for i, z in step.early_raw.items()}
            def js(q):
                a, b = np.exp(p), np.exp(q)
                mid = (a + b) / 2
                ma, mb = a > 0, b > 0
                return float(.5 * (np.sum(a[ma] * (p[ma] - np.log(mid[ma])))
                                    + np.sum(b[mb] * (q[mb] - np.log(mid[mb])))))
            selected_layer = max(sorted(layers), key=lambda i: js(layers[i]))
            keep = p >= p.max() + np.log(.1)
            score = p - layers[selected_layer]
        else:
            if not step.early_normalized or any(not np.isfinite(z).all() for z in step.early_normalized.values()):
                raise ValueError("Native DeCo lacks finite real normalized layer projections")
            indices = np.argsort(-p, kind="stable")[:20]
            cutoff = min(len(indices), int(np.searchsorted(np.cumsum(np.exp(p[indices])), .9)) + 1)
            candidates = indices[:cutoff]
            value, selected_layer = -np.inf, None
            for layer, z in sorted(step.early_normalized.items()):
                proposal = float(np.exp(log_normalize(z)[candidates]).max())
                if proposal > value:
                    value, selected_layer = proposal, layer
            weight = .6 * value
            score = logits + weight * np.asarray(step.early_normalized[selected_layer], dtype=np.float64)
            keep = np.zeros(len(p), dtype=bool)
            keep[candidates] = True
        output = log_normalize(np.where(keep, score, -np.inf))
        if not np.isfinite(output[keep]).all() or not math.isclose(float(np.exp(output[keep]).sum()), 1., abs_tol=1e-8):
            raise ValueError("Native supported probability normalization differs")
        record["step_checks"].append({"prefix_token_ids": list(prefix), "argmax": int(np.argmax(output)),
                                      "selected_layer": selected_layer, "weight": float(weight),
                                      "support_size": int(keep.sum()), "vocabulary_size": len(p),
                                      "raw_layer_indices": sorted(step.early_raw),
                                      "normalized_layer_indices": sorted(step.early_normalized),
                                      "finite_logits": True, "supported_probability_sum": float(np.exp(output[keep]).sum())})
        return step


def execute(args):
    samples, frozen_spec, provenance = inputs(args.model, args.method)
    chosen = samples[args.start:args.stop]
    if len(chosen) < 8:
        raise ValueError("Each exclusive native part needs at least eight inputs for its retained pilot")
    directory = ROOT / OUTPUT / args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{args.method}_{args.start:04d}_{args.stop:04d}"
    raw = directory / "raw" / f"{stem}.jsonl"
    gate = directory / f"{stem}.pilot.json"
    claim = directory / f"{stem}.claim.json"
    cfg = DecodeConfig(method=args.method)
    if (cfg.max_tokens, cfg.temperature, cfg.top_p, cfg.alpha, cfg.beta,
            cfg.deco_alpha, cfg.deco_topk, cfg.deco_topp) != (32, 0., 1., 1., .1, .6, 20, .9):
        raise ValueError("Registered native decoder defaults differ")
    if args.phase == "plan":
        return {"schema": "kdm_native_baseline_plan_v1", "model": args.model, "method": args.method,
                "start": args.start, "stop": args.stop, "expected": len(chosen),
                "pilot_in_full": 8, "config": asdict(cfg), "provenance": provenance,
                "actual_gpu_generation": False, "checked_at_utc": now()}
    cards = args.physical_gpus.split(",")
    if len(set(cards)) != len(cards) or any(not card.isdigit() for card in cards):
        raise ValueError("Invalid physical GPU list")
    actual, admitted = admission(frozen_spec, args.model, cards, args.registry)
    identity = {"schema": "kdm_native_baseline_part_identity_v1", "protocol": "native_baselines_20261003",
                "model": args.model, "dataset": "food101", "split": "eval", "method": args.method,
                "condition": {k: v for k, v in task(chosen[0], args.method).items() if k != "sample"},
                "frozen_spec_sha256": stable_hash(frozen_spec), "runtime_spec": actual,
                "admission_fixed": admitted["fixed_identity"], "provenance": provenance,
                "config": asdict(cfg), "start": args.start, "stop": args.stop,
                "sample_ids_sha256": stable_hash([s["id"] for s in chosen]),
                "runner_sha256": file_hash(Path(__file__)),
                "author": {"agent": "/root/native_baselines", "model": "gpt-6.1-sol", "effort": "max", "call_id": ""}}
    if args.phase == "pilot":
        if raw.exists() or gate.exists() or claim.exists():
            raise FileExistsError("Preserve existing native pilot/claim; no automatic regeneration")
        with claim.open("x") as stream:
            json.dump({"identity": stable_hash(identity), "owner": args.owner, "expected": len(chosen),
                       "sample_ids": [s["id"] for s in chosen], "created_at_utc": now()}, stream)
    else:
        proof = json.loads(gate.read_text())
        owned = json.loads(claim.read_text())
        if (not proof["passed"] or proof["completed"] != 8 or proof["identity"] != stable_hash(identity)
                or owned["identity"] != stable_hash(identity) or owned["owner"] != args.owner):
            raise ValueError("Native full run requires this unchanged actual pilot and owned part")
    atomic_json(directory / f"{stem}.admission_{args.phase}.json", admitted)
    load_start = time.perf_counter()
    backend = make_backend(actual, device="cuda:0")
    load_wall = time.perf_counter() - load_start
    selected = chosen[:8] if args.phase == "pilot" else chosen
    if args.phase == "pilot":
        backend = AuditBackend(backend, args.model, args.method, selected)
    start = time.perf_counter()
    count = run_tasks(backend, args.model, [task(s, args.method) for s in selected], raw, identity, cfg=cfg)
    rows = list(read_jsonl(raw))
    expected = {task_id(args.model, task(s, args.method)) for s in selected}
    if len(rows) != len(expected) or {r["key"] for r in rows} != expected or count != len(expected):
        raise ValueError("Native part key coverage or duplicate audit failed")
    for row in rows:
        if (row["config"] != asdict(cfg) or row["status"] != "ok" or not row["tokens"]
                or len(row["tokens"]) > 32 or row["terminated"] != (row["tokens"][-1] in backend.eos)
                or row["seed"] != stable_seed(row["sample"]["id"], args.model, 0)
                or len(row["selected_log_probabilities"]) != len(row["tokens"])
                or not all(math.isfinite(v) for v in row["selected_log_probabilities"])
                or not math.isfinite(row["first_probability"])):
            raise ValueError("Native raw identity, probability, token or termination audit failed")
    receipt = {"schema": "kdm_native_baseline_part_receipt_v1", "phase": args.phase,
               "model": args.model, "method": args.method, "identity": stable_hash(identity),
               "raw_path": str(raw), "raw_sha256": file_hash(raw),
               "identity_path": str(raw.with_suffix(".identity.json")),
               "identity_sha256": file_hash(raw.with_suffix(".identity.json")),
               "completed": len(rows), "expected_full_part": len(chosen), "start": args.start, "stop": args.stop,
               "model_load_wall_s": load_wall, "execution_wall_s": time.perf_counter() - start,
               "raw_generation_wall_s": sum(r["wall_s"] for r in rows), "completed_at_utc": now(),
               "passed": True, "condition": identity["condition"], "registry": args.registry,
               "pilot_in_full": True, "actual_gpu_generation": True}
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
    parser.add_argument("--registry", default=ORIGINAL_REGISTRY)
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
