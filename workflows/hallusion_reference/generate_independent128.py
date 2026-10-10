#!/usr/bin/env python3
"""Frozen Hallusion whole951 independent native-HF attempts, ten seeds per input.

This entry is separate from the original Hallusion panel. It never changes its
sources or outputs, never reuses an answer trajectory, and has no retry, model,
precision, parameter, batch-size, or execution-engine fallback.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from kdm.decoding import DecodeConfig, generate
from kdm.io import file_hash, read_jsonl, stable_hash, stable_seed
from kdm.pipeline import json_safe, make_backend, task_id
from kdm.prompts import task_prompt
from kdm.protocol import validate_native_runtime_files
from workflows.general_vqa_direct.generate import admit, now, process_identity, relative, write_once
from workflows.hallusion_blind.generate128 import input_evidence

MODELS = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b",
          "internvl35_8b", "onevision", "phi35", "qwen3vl")
BASE = "outputs/hallusion_independent128_20261008"
PROTOCOL = BASE + "/registration/protocol.json"
MANIFEST = "data/general_vqa_direct_20261004/frozen/manifest_hallusionbench.jsonl"
MANIFEST_SHA256 = "a7d2c81813474478e9045e3164ce400a8ad8ab59e86a5f71c674f296d0a06e51"
CONDITIONS = "outputs/hallusion_blind128_20261006/registration/conditions.json"
CONDITIONS_SHA256 = "67d24d4b92b4f5e4ad4af1f466f80c0b121a2178f9cec52eac87971a03d5f2a3"
SOURCES = (
    "workflows/hallusion_reference/generate_independent128.py",
    "workflows/hallusion_reference/test_independent128.py",
    "workflows/hallusion_blind/generate128.py",
    "workflows/general_vqa_direct/generate.py",
    "src/kdm/io.py", "src/kdm/pipeline.py", "src/kdm/decoding.py",
    "src/kdm/probability.py", "src/kdm/prompts.py", "src/kdm/protocol.py",
    "src/kdm/frozen.py", "src/kdm/execution.py",
    "src/kdm/models/hf.py", "src/kdm/models/backbone.py", "src/kdm/models/remote.py",
    "src/kdm/models/sid.py", "src/kdm/models/internvl_dual.py",
    "src/kdm/models/internvl_preprocessing.py",
    "workflows/supplemental/remaining11/execution.py",
    "workflows/supplemental/remaining4/k100_intern_registered_matrix.py",
    "workflows/supplemental/remaining4/internvl_k100_single.py",
)
PROMPT_POLICY = "question_strip_plus_task_prompt_attempt_true"
SEED_RULE = "stable_seed(sample.id,model,replicate)"
ATTEMPT_SUFFIX = "\nGive your best estimate as a specific short answer."
EXPECTED_PROTOCOL = {
    "schema": "kdm_hallusion_independent128_protocol_v1",
    "models": list(MODELS), "methods": ["direct"], "replicates": list(range(10)),
    "temperature": 1.0, "top_p": 1.0, "output_token_cap": 128,
    "termination_policy": "EOS_or_128_generated_tokens", "prompt_policy": PROMPT_POLICY,
    "seed_rule": SEED_RULE, "attempt": True, "guided": False, "reference_guided": False,
    "marker": "NONE", "reference_marker": "NONE", "kind": "independent_attempt",
    "batch_size": 1, "expected_answers": 85590,
    "conditions_source": CONDITIONS, "conditions_sha256": CONDITIONS_SHA256,
}


def independent_prompt(sample):
    question = sample["question"]
    prompt = task_prompt(question, marker="NONE", guided=False, attempt=True)
    if prompt != question.strip() + ATTEMPT_SUFFIX:
        raise ValueError("The original attempt=True prompt implementation changed")
    return prompt


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _same(left, right):
    """Content equality also distinguishes JSON booleans from numeric values."""
    return stable_hash(left) == stable_hash(right)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_plan(args, root=ROOT):
    root = Path(root).resolve()
    if args.batch_size != 1 or args.methods != ["direct"]:
        raise ValueError("Only native Direct batch_size=1 is registered; no automatic lowering")
    if args.model not in MODELS:
        raise ValueError("Model lies outside the original nine-checkpoint panel")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.claim_id) or not args.owner.strip():
        raise ValueError("Invalid claim identifier or owner")
    if args.protocol != PROTOCOL or args.output != BASE + "/runs":
        raise ValueError("The independent protocol and output directories are fixed")
    protocol_path = relative(root, args.protocol)
    protocol_sha = file_hash(protocol_path)
    if not _sha(args.protocol_sha256) or protocol_sha != args.protocol_sha256:
        raise ValueError("Independent protocol bytes differ from the explicit frozen SHA256")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for field, expected in EXPECTED_PROTOCOL.items():
        if not _same(protocol.get(field), expected):
            raise ValueError("Independent protocol field differs: " + field)
    entry = protocol["dataset_entry"]
    expected_entry = {"manifest": MANIFEST, "manifest_sha256": MANIFEST_SHA256, "expected_rows": 951,
                      "source_split": "official_main_all_visual_inputs_1_or_2", "new_split": "blind_test"}
    if not _same(entry, expected_entry):
        raise ValueError("The exact original Hallusion whole951 dataset entry is required")
    source_sha = {name: file_hash(root / name) for name in SOURCES}
    runtime_sha = protocol.get("runtime_sha256")
    if (not isinstance(runtime_sha, dict) or set(runtime_sha) != set(MODELS)
            or not all(_sha(value) for value in runtime_sha.values())):
        raise ValueError("All nine original runtime-spec SHA256 values must be registered")
    registry_path = relative(root, args.registry)
    registry_sha = file_hash(registry_path)
    if protocol.get("registry_sha256") != registry_sha:
        raise ValueError("Independent host registry bytes changed")
    if protocol.get("registry") != args.registry:
        raise ValueError("Independent host registry path changed")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    matches = [host for host, details in registry["hosts"].items()
               if details["hostname"] == socket.gethostname() and Path(details["root"]) == root]
    if len(matches) != 1:
        raise ValueError("Source hostname/project root is not uniquely registered")
    source_host = matches[0]
    sources_by_host = protocol.get("source_sha256_by_host")
    if (not isinstance(sources_by_host, dict) or set(sources_by_host) != set(registry["hosts"])
            or any(not isinstance(value, dict) or set(value) != set(SOURCES)
                   or not all(_sha(sha) for sha in value.values()) for value in sources_by_host.values())
            or not _same(protocol.get("source_sha256"), sources_by_host.get("4028"))
            or not _same(source_sha, sources_by_host[source_host])):
        raise ValueError("Registered host-specific independent source bytes changed")
    condition_path = relative(root, CONDITIONS)
    if file_hash(condition_path) != CONDITIONS_SHA256:
        raise ValueError("Original main-panel condition bytes changed")
    frozen_conditions = json.loads(condition_path.read_text(encoding="utf-8"))
    if set(row["model"] for row in frozen_conditions) != set(MODELS):
        raise ValueError("Original condition panel no longer identifies exactly nine checkpoints")
    direct = [row for row in frozen_conditions if row["model"] == args.model and row["method"] == "direct"]
    if len(direct) != 1:
        raise ValueError("The original model must have exactly one registered Direct checkpoint")
    source_condition = direct[0]
    native_cfg = DecodeConfig(**source_condition["config"])
    if native_cfg != DecodeConfig(method="direct", max_tokens=128, temperature=0.0, top_p=1.0):
        raise ValueError("Original Direct configuration differs from the frozen native parameters")
    cfg = replace(native_cfg, max_tokens=128, temperature=1.0, top_p=1.0)
    spec_path = root / f"configs/runtime/{args.model}.json"
    spec_sha = file_hash(spec_path)
    if spec_sha != runtime_sha[args.model]:
        raise ValueError("Original runtime specification changed")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    dtype = "float16" if args.model == "onevision" else "bfloat16"
    if (spec["key"] != args.model or spec["availability"] != "resolved" or spec["dtype"] != dtype
            or spec["kwargs"].get("dtype", "bfloat16") != dtype or spec.get("api")
            or spec.get("hf_model_id") != source_condition["checkpoint"]):
        raise ValueError("The original checkpoint, native backend or precision differs")
    # This checks existing original proof files only. It does not initialize CUDA,
    # query a GPU, import a model factory, or manufacture runtime proof.
    validate_native_runtime_files(root, spec, args.model)
    manifest = relative(root, MANIFEST)
    if file_hash(manifest) != MANIFEST_SHA256:
        raise ValueError("Frozen Hallusion manifest bytes changed")
    samples = list(read_jsonl(manifest))
    if len(samples) != 951 or len({sample["id"] for sample in samples}) != 951:
        raise ValueError("All original 951 distinct inputs are required")
    required = {"id", "dataset", "split", "source_split", "question", "gold", "prompt",
                "image_paths", "image_sha256", "source_record", "question_source_sha256"}
    for sample in samples:
        if (not required <= set(sample) or sample["dataset"] != "hallusionbench" or sample["split"] != "eval"
                or sample["source_split"] != expected_entry["source_split"]
                or not isinstance(sample["id"], str) or not sample["id"]
                or not isinstance(sample["question"], str) or not sample["question"].strip()
                or not isinstance(sample["image_paths"], list) or len(sample["image_paths"]) != 1
                or not isinstance(sample["image_sha256"], list) or len(sample["image_sha256"]) != 1
                or not _sha(sample["image_sha256"][0]) or not _sha(sample["question_source_sha256"])):
            raise ValueError("Invalid frozen Hallusion single-image input")
        relative(root, sample["image_paths"][0])
        independent_prompt(sample)
    source_condition_identity = stable_hash(source_condition)
    conditions, cfgs = {}, {}
    for replicate in range(10):
        condition = {"model": args.model, "method": "direct", "marker": "NONE", "reference_marker": "NONE",
                     "guided": False, "reference_guided": False, "attempt": True,
                     "replicate": replicate, "kind": "independent_attempt", "config": asdict(cfg),
                     "checkpoint": source_condition["checkpoint"],
                     "source_condition_identity": source_condition_identity}
        condition_identity = stable_hash(condition)
        if condition_identity in conditions:
            raise ValueError("Duplicate independent condition")
        conditions[condition_identity] = condition
        cfgs[condition_identity] = cfg
    by_replicate = {row["replicate"]: identity for identity, row in conditions.items()}
    tasks, first8 = {}, []
    for sample_index, sample in enumerate(samples):
        seeds = [stable_seed(sample["id"], args.model, replicate) for replicate in range(10)]
        if len(set(seeds)) != 10:
            raise ValueError("The ten independent seeds collide for a frozen input")
        for replicate in range(10):
            identity = by_replicate[replicate]
            condition = conditions[identity]
            item = {"sample": sample, **{field: condition[field] for field in (
                "method", "marker", "reference_marker", "guided", "reference_guided", "attempt", "replicate",
                "kind", "source_condition_identity")}, "condition_identity": identity}
            key = task_id(args.model, item)
            if key in tasks:
                raise ValueError("Duplicate independent task key")
            tasks[key] = item
            if sample_index < 8 and replicate == 0:
                first8.append(key)
    if len(tasks) != 9510 or len(first8) != 8:
        raise ValueError("The model must cover exactly 951 inputs times ten attempts")
    timing_set = set(first8)
    selected = first8 + [key for key in tasks if key not in timing_set]
    assignment_sha = None
    if args.task_keys:
        assignment = relative(root, args.task_keys)
        assigned = json.loads(assignment.read_text(encoding="utf-8"))["keys"]
        if (not isinstance(assigned, list) or not assigned or not all(isinstance(key, str) for key in assigned)
                or len(set(assigned)) != len(assigned) or not set(assigned) <= set(tasks)):
            raise ValueError("Foreign, empty or duplicate explicit task assignment")
        selected = [key for key in selected if key in set(assigned)]
        assignment_sha = file_hash(assignment)
    definition = {"schema": "kdm_hallusion_independent128_model_v1", "model": args.model,
                  "dtype": dtype, "runtime_sha256": spec_sha,
                  "protocol": args.protocol, "protocol_sha256": protocol_sha,
                  "registry": args.registry, "registry_sha256": registry_sha,
                  "source_host": source_host, "source_sha256": source_sha, "dataset_entry": entry,
                  "conditions_source": CONDITIONS, "conditions_sha256": CONDITIONS_SHA256,
                  "source_condition": source_condition, "conditions": list(conditions.values()),
                  "engine": "registered_hf_session", "batch_size": 1,
                  "prompt_policy": PROMPT_POLICY, "seed_rule": SEED_RULE,
                  "session_policy": "new_native_session_per_trial", "output_token_cap": 128,
                  "termination": "EOS_or_128_generated_tokens", "expected_model_rows": 9510,
                  "expected_total_rows": 85590, "first8_keys": first8,
                  "first8_scope": "eight_distinct_source_order_samples_replicate0_counted_in_formal_trials"}
    return {"definition": definition, "identity": stable_hash(definition), "spec": spec,
            "conditions": conditions, "cfgs": cfgs,
            "datasets": {"hallusionbench": {"entry": entry, "identity": stable_hash(entry)}},
            "tasks": tasks, "selected": selected, "output": relative(root, args.output) / args.model,
            "assignment_sha256": assignment_sha, "root": root, "first8_keys": first8}


def run_one(backend, item, cfg, model, root=ROOT):
    from PIL import Image
    sample = item["sample"]
    if cfg != DecodeConfig(method="direct", max_tokens=128, temperature=1.0, top_p=1.0):
        raise ValueError("Independent native Direct sampling configuration differs")
    if getattr(backend, "generate_text", None) or not set(backend.eos):
        raise ValueError("A native token-session backend and actual EOS tokens are required")
    if len(sample["image_paths"]) != 1 or len(sample["image_sha256"]) != 1:
        raise ValueError("Original Hallusion single-image source required")
    path = relative(root, sample["image_paths"][0])
    if file_hash(path) != sample["image_sha256"][0]:
        raise ValueError("Frozen image bytes changed")
    prompt = independent_prompt(sample)
    seed = stable_seed(sample["id"], model, item["replicate"])
    session = None
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        # A fresh native session owns its own autoregressive state on every call.
        session = backend.session(rgb, prompt)
        evidence = {**input_evidence(session, prompt),
                    "source_image_count": 1, "source_image_paths": sample["image_paths"],
                    "source_image_sha256": sample["image_sha256"],
                    "exact_question_sha256": stable_hash(sample["question"]),
                    "attempt_prompt_sha256": stable_hash(prompt)}
        result = generate(session, None, cfg, backend.eos, backend.decode, seed)
        if not result["terminated"] and len(result["tokens"]) != 128:
            raise ValueError("The decoder stopped before an actual EOS or 128 generated tokens")
        return {**result, "generation_source": "independent_hallusion128",
                "prompt": prompt, "reference_prompt": None, "neutral_prompt": None,
                "config": asdict(cfg), "seed": seed, "offset_prompt_tokens": None, "noise": None,
                "input_evidence": evidence, "branch_inputs": {"main": evidence},
                "eos_token_ids": sorted(set(backend.eos)),
                "finish_reason": "eos" if result["terminated"] else "length",
                "truncated": not result["terminated"]}
    finally:
        session = None


def validate_rows(path, plan, keys, claim_identity, eos):
    if not eos or any(type(token) is not int or token < 0 for token in eos):
        raise ValueError("Actual nonempty EOS token identities are required")
    rows = list(read_jsonl(path))
    if [row.get("key") for row in rows] != keys or len(set(keys)) != len(keys):
        raise ValueError("Chunk key order, coverage or uniqueness differs")
    for row, key in zip(rows, keys):
        item = plan["tasks"][key]
        sample = item["sample"]
        cfg = plan["cfgs"][item["condition_identity"]]
        prompt = independent_prompt(sample)
        if (not _same({field: row.get(field) for field in item}, item)
                or row.get("identity") != plan["identity"] or row.get("claim_identity") != claim_identity
                or row.get("model") != plan["definition"]["model"]
                or row.get("dataset_identity") != plan["datasets"]["hallusionbench"]["identity"]
                or row.get("status") != "ok" or not isinstance(row.get("text"), str)
                or not _same(row.get("config"), asdict(cfg)) or row.get("prompt") != prompt
                or type(row.get("seed")) is not int
                or row["seed"] != stable_seed(sample["id"], row["model"], item["replicate"])
                or row.get("generation_source") != "independent_hallusion128"
                or any(row.get(field) is not None for field in ("reference_prompt", "neutral_prompt", "offset_prompt_tokens", "noise"))):
            raise ValueError("Frozen sample/model/condition/prompt/source/config/seed binding differs")
        tokens = row.get("tokens")
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= 128
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("Generated tokens lie outside the registered budget")
        ended = tokens[-1] in eos
        if (row.get("terminated") is not ended or row.get("truncated") is not (not ended)
                or row.get("finish_reason") != ("eos" if ended else "length")
                or row.get("eos_token_ids") != sorted(eos) or any(token in eos for token in tokens[:-1])
                or (not ended and len(tokens) != 128)):
            raise ValueError("Actual EOS and 128-token length termination must be recorded separately")
        if not _finite(row.get("wall_s")) or row["wall_s"] < 0:
            raise ValueError("Invalid actual measured wall time")
        probabilities = row.get("selected_log_probabilities")
        if (not isinstance(probabilities, list) or len(probabilities) != len(tokens)
                or any(not _finite(value) or value > 1e-12 for value in probabilities)):
            raise ValueError("Invalid selected-token log probabilities")
        trace = row.get("trace")
        if not isinstance(trace, list) or len(trace) != len(tokens):
            raise ValueError("Token trace is incomplete")
        for token, logp, step in zip(tokens, probabilities, trace):
            if (not isinstance(step, dict) or type(step.get("token")) is not int or step["token"] != token
                    or not _finite(step.get("log_probability"))
                    or not _finite(step.get("sampling_log_probability"))
                    or step["sampling_log_probability"] > 1e-12
                    or not math.isclose(step["log_probability"], logp, rel_tol=1e-12, abs_tol=1e-12)
                    or not math.isclose(step["sampling_log_probability"], logp, rel_tol=1e-12, abs_tol=1e-12)
                    or step.get("weight") != 0.0 or step.get("layer") is not None or step.get("active") is not False):
                raise ValueError("Direct sampling token/probability trace differs")
        sequence = row.get("sequence_log_probability")
        first = row.get("first_probability")
        if (not _finite(sequence) or not math.isclose(sequence, sum(probabilities), rel_tol=1e-12, abs_tol=1e-12)
                or not _finite(first) or not 0 <= first <= 1
                or not math.isclose(first, math.exp(probabilities[0]), rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("Sequence or first-token probability evidence differs")
        branches = row.get("branch_inputs")
        evidence = row.get("input_evidence")
        if (not isinstance(branches, dict) or set(branches) != {"main"}
                or not isinstance(evidence, dict) or not _same(branches["main"], evidence)
                or evidence.get("prompt") != prompt or not _sha(evidence.get("input_ids_sha256"))
                or type(evidence.get("input_token_count")) is not int or evidence["input_token_count"] <= 0
                or evidence.get("source_image_count") != 1
                or evidence.get("source_image_paths") != sample["image_paths"]
                or evidence.get("source_image_sha256") != sample["image_sha256"]
                or evidence.get("exact_question_sha256") != stable_hash(sample["question"])
                or evidence.get("attempt_prompt_sha256") != stable_hash(prompt)):
            raise ValueError("Actual sole native branch does not retain the frozen image and attempt prompt")
    return {"rows": len(rows), "truncated_rows": sum(row["truncated"] for row in rows),
            "dataset_counts": {"hallusionbench": len(rows)}}


def completed_keys(plan):
    folder = plan["output"]
    ledger = folder / "completed_keys.jsonl"
    if not ledger.exists():
        return set()
    definition = {"identity": plan["identity"], "definition": plan["definition"]}
    if not (folder / "identity.json").exists() or not _same(json.loads((folder / "identity.json").read_text()), definition):
        raise ValueError("Completed ledger lacks its exact frozen model/source identity")
    grouped, seen = {}, set()
    for row in read_jsonl(ledger):
        if (row.get("key") in seen or row.get("identity") != plan["identity"]
                or row.get("dataset") != "hallusionbench" or row.get("key") not in plan["tasks"]):
            raise ValueError("Duplicate, foreign or out-of-manifest completed-key ledger entry")
        seen.add(row["key"])
        grouped.setdefault(row["receipt"], []).append(row)
    complete = set()
    for name, entries in grouped.items():
        receipt_path = relative(folder, name)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        raw = relative(folder, receipt["raw_path"])
        owner_path = relative(folder, receipt["owner_path"])
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        keys = [entry["key"] for entry in entries]
        if (receipt["status"] != "complete" or receipt["identity"] != plan["identity"]
                or receipt["keys"] != keys or file_hash(raw) != receipt["raw_sha256"]
                or file_hash(owner_path) != receipt["owner_sha256"]
                or stable_hash(owner) != receipt["claim_identity"] or owner["identity"] != plan["identity"]
                or owner.get("admission", {}).get("host") != plan["definition"]["source_host"]
                or not set(keys) <= set(owner["keys"])
                or not _same(owner["source_sha256"], plan["definition"]["source_sha256"])
                or any(entry["receipt_sha256"] != file_hash(receipt_path) for entry in entries)):
            raise ValueError("Completed receipt/owner/raw/source binding differs")
        checked = validate_rows(raw, plan, keys, receipt["claim_identity"], set(receipt["eos_token_ids"]))
        if not _same(checked, receipt["validation"]):
            raise ValueError("Completed chunk no longer passes full row validation")
        complete.update(keys)
    return complete


def claim(args, plan, admission):
    import fcntl
    folder = plan["output"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "ownership.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        definition = {"identity": plan["identity"], "definition": plan["definition"]}
        identity_path = folder / "identity.json"
        if identity_path.exists():
            if not _same(json.loads(identity_path.read_text()), definition):
                raise ValueError("Output directory belongs to another frozen source/model/protocol identity")
        else:
            write_once(identity_path, definition)
        complete = completed_keys(plan)
        pending = set(plan["selected"]) - complete
        claims = folder / "claims"
        claims.mkdir(exist_ok=True)
        for previous_run in claims.iterdir():
            if not previous_run.is_dir():
                continue
            previous = json.loads((previous_run / "owner.json").read_text())
            released = set()
            if (previous_run / "released.json").exists():
                release = json.loads((previous_run / "released.json").read_text())
                if release["claim_identity"] != stable_hash(previous) or release["reason"] != "stop_after_completed_chunk":
                    raise ValueError("Unbound ownership release")
                released = set(release["keys"])
                if not released <= set(previous["keys"]):
                    raise ValueError("Ownership release contains foreign keys")
            if pending & (set(previous["keys"]) - complete - released):
                raise ValueError("A pending key already has an owner; no retry or automatic takeover")
        if not pending:
            return None, [], None
        run = claims / args.claim_id
        run.mkdir(exist_ok=False)
        keys = [key for key in plan["selected"] if key in pending]
        owner = {"claim_id": args.claim_id, "owner": args.owner, "started_utc": now(),
                 **process_identity(), "keys": keys, "identity": plan["identity"],
                 "dataset_entries": {name: row["entry"] for name, row in plan["datasets"].items()},
                 "admission": admission, "batch_size": 1, "chunk_rows": args.chunk_rows,
                 "assignment_path": args.task_keys, "assignment_sha256": plan["assignment_sha256"],
                 "source_sha256": plan["definition"]["source_sha256"], "argv": sys.argv,
                 "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=plan["root"], text=True).strip()}
        write_once(run / "owner.json", owner)
        return run, keys, owner


def seal(plan, run, owner, number, pending_path, keys, eos):
    import fcntl
    checked = validate_rows(pending_path, plan, keys, stable_hash(owner), eos)
    raw = run / f"chunk_{number:05d}.jsonl"
    if raw.exists():
        raise ValueError("Refusing to replace a sealed chunk")
    pending_path.rename(raw)
    raw.chmod(0o444)
    receipt_path = run / f"chunk_{number:05d}.complete.json"
    receipt = {"status": "complete", "completed_utc": now(), "identity": plan["identity"],
               "claim_identity": stable_hash(owner), "keys": keys, "eos_token_ids": sorted(eos),
               "raw_path": str(raw.relative_to(plan["output"])), "raw_sha256": file_hash(raw),
               "owner_path": str((run / "owner.json").relative_to(plan["output"])),
               "owner_sha256": file_hash(run / "owner.json"), "validation": checked}
    write_once(receipt_path, receipt)
    with (plan["output"] / "ownership.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = plan["output"] / "completed_keys.jsonl"
        recorded = {row["key"] for row in read_jsonl(ledger)} if ledger.exists() else set()
        if recorded & set(keys):
            raise ValueError("Completed ledger already contains a chunk key")
        with ledger.open("a", encoding="utf-8") as stream:
            for key in keys:
                stream.write(json.dumps({"key": key, "dataset": "hallusionbench", "identity": plan["identity"],
                    "receipt": str(receipt_path.relative_to(plan["output"])),
                    "receipt_sha256": file_hash(receipt_path)}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"event": "sealed_chunk", "chunk": number, **checked}), flush=True)


def execute(args, plan):
    actual, admission = admit(args, plan, root=plan["root"])
    if admission["host"] != plan["definition"]["source_host"]:
        raise ValueError("Admission host differs from the frozen source host")
    run, keys, owner = claim(args, plan, admission)
    if run is None:
        return {"status": "already_complete", "selected_rows": len(plan["selected"])}
    started, active_key = time.perf_counter(), None
    try:
        backend = make_backend(actual, "cuda:0")
        loaded = time.perf_counter()
        eos = set(backend.eos)
        if not eos or getattr(backend, "generate_text", None):
            raise ValueError("Registered native token-session backend and actual EOS tokens required")
        number, chunk_keys, timing = 0, [], {}
        for index, key in enumerate(keys):
            active_key = key
            item = plan["tasks"][key]
            cfg = plan["cfgs"][item["condition_identity"]]
            before = time.perf_counter()
            result = run_one(backend, item, cfg, args.model, root=plan["root"])
            wall = time.perf_counter() - before
            row = {"key": key, **item, "model": args.model, "identity": plan["identity"],
                   "dataset_identity": plan["datasets"]["hallusionbench"]["identity"],
                   "claim_identity": stable_hash(owner), "wall_s": wall, **result}
            pending = run / f"chunk_{number:05d}.pending.jsonl"
            with pending.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            chunk_keys.append(key)
            if key in set(plan["first8_keys"]):
                timing[key] = {"key": key, "sample_id": item["sample"]["id"], "replicate": item["replicate"],
                               "wall_s": wall, "n_tokens": len(row["tokens"])}
                if len(timing) == 8:
                    measured = [timing[key] for key in plan["first8_keys"]]
                    measurement = {"status": "actual_first8", "method": "direct", "model": args.model,
                        "identity": plan["identity"], "claim_identity": stable_hash(owner),
                        "rows": measured, "completed_inputs": 8, "completed_trials": 8,
                        "mean_input_wall_s": sum(row["wall_s"] for row in measured) / 8,
                        "mean_output_tokens": sum(row["n_tokens"] for row in measured) / 8,
                        "model_load_wall_s": loaded - started, "measured_utc": now(), "config": asdict(cfg),
                        "scope": "eight distinct source-order inputs at replicate0; eight trials counted in the formal9510; completed-key progress requires sealed chunks"}
                    write_once(run / "first8_direct.json", measurement)
                    print(json.dumps({"event": "actual_first8", **measurement}), flush=True)
            if len(chunk_keys) >= args.chunk_rows or index + 1 == len(keys):
                seal(plan, run, owner, number, pending, chunk_keys, eos)
                number += 1
                chunk_keys = []
                if (run / "STOP_AFTER_CHUNK").exists() and index + 1 < len(keys):
                    write_once(run / "released.json", {"claim_identity": stable_hash(owner),
                        "reason": "stop_after_completed_chunk", "released_utc": now(), "keys": keys[index + 1:]})
                    return {"status": "stopped_after_completed_chunk", "completed_rows": index + 1,
                            "released_rows": len(keys) - index - 1}
        complete = completed_keys(plan)
        if not set(keys) <= complete:
            raise ValueError("Claim completion lacks fully verified sealed response chunks")
        receipt = {"status": "complete", "identity": plan["identity"], "claim_identity": stable_hash(owner),
                   "completed_rows": len(keys), "selected_rows": len(plan["selected"]),
                   "completed_selected_rows": len(set(plan["selected"]) & complete),
                   "completed_utc": now(), "wall_s": time.perf_counter() - started}
        write_once(run / "complete.json", receipt)
        return receipt
    except BaseException as exc:
        write_once(run / "failed.json", {"status": "failed", "failed_utc": now(), "key": active_key,
            "claim_identity": stable_hash(owner), "error": type(exc).__name__ + ": " + str(exc),
            "traceback": traceback.format_exc(), "automatic_retry": False,
            "automatic_parameter_change": False, "automatic_precision_change": False,
            "automatic_engine_fallback": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--methods", nargs="+", default=["direct"])
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", default=BASE + "/runs")
    parser.add_argument("--registry", default="workflows/hallusion_blind/host_registry.json")
    parser.add_argument("--cards", default="")
    parser.add_argument("--claim-id", default="cpu_plan_check")
    parser.add_argument("--owner", default="/root")
    parser.add_argument("--task-keys", help="Project-relative JSON object containing an exclusive keys list")
    parser.add_argument("--chunk-rows", type=int, choices=(1, 8, 16), default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--k100-intern", action="store_true")
    args = parser.parse_args(argv)
    plan = load_plan(args)
    if args.check_plan:
        result = {"status": "CPU_plan_checked", "model": args.model, "model_identity": plan["identity"],
                  "protocol_sha256": plan["definition"]["protocol_sha256"],
                  "runtime_sha256": plan["definition"]["runtime_sha256"],
                  "manifest_sha256": MANIFEST_SHA256, "expected_model_rows": len(plan["tasks"]),
                  "selected_rows": len(plan["selected"]), "replicates": list(range(10)),
                  "replicate_counts": dict(Counter(item["replicate"] for item in plan["tasks"].values())),
                  "first8_keys": plan["first8_keys"], "max_tokens": 128,
                  "temperature": 1.0, "top_p": 1.0, "batch_size": 1, "gpu_initialized": False}
    elif args.verify:
        complete = completed_keys(plan)
        expected = set(plan["selected"])
        result = {"status": "complete" if expected <= complete else "incomplete", "model_identity": plan["identity"],
                  "expected_rows": len(expected), "completed_rows": len(expected & complete)}
    else:
        if not args.cards or args.claim_id == "cpu_plan_check":
            parser.error("--execute requires explicit --cards and a distinct --claim-id")
        result = execute(args, plan)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), flush=True)
    return 2 if args.verify and result["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
