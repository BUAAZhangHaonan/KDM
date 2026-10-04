#!/usr/bin/env python3
"""Native Direct generation with exact frozen prompts and per-dataset budgets.

The model remains loaded across datasets and sealed chunks. Dataset entries are
bound separately, so adding another dataset to the protocol does not change an
existing dataset's identity. No retry, batch lowering, or engine fallback exists.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
from datetime import datetime, timezone
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
from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash, stable_seed, within
from kdm.pipeline import make_backend, task_id
from kdm.protocol import validate_environment, validate_native_runtime_files
from workflows.supplemental.remaining11.execution import checkpoint_identity

MODELS = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b",
          "internvl35_8b", "onevision", "phi35", "qwen3vl")
SOURCES = ("workflows/general_vqa_direct/generate.py", "workflows/general_vqa_direct/inputs.py",
           "src/kdm/pipeline.py", "src/kdm/decoding.py", "src/kdm/probability.py",
           "src/kdm/models/hf.py", "src/kdm/models/backbone.py", "src/kdm/models/remote.py",
           "src/kdm/models/internvl_dual.py", "src/kdm/models/internvl_preprocessing.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def relative(root, value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected a project-relative path: " + str(value))
    return within(root, path)


def write_once(path, value):
    if path.exists():
        raise ValueError("Refusing to replace evidence: " + str(path))
    atomic_json(path, value)
    path.chmod(0o444)


def process_identity():
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": os.getpid(), "start_tick": int(fields[19]), "hostname": socket.gethostname(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def direct_task(sample):
    return {"sample": sample, "method": "direct", "marker": "NONE", "reference_marker": "NONE",
            "guided": False, "reference_guided": False, "replicate": 0, "kind": "general_vqa_direct"}


def ids_from_file(path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            raise ValueError("Blank sample-ID line")
        item = json.loads(line) if line.startswith(("{", '"')) else line
        rows.append(item["id"] if isinstance(item, dict) else item)
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("Sample-ID assignment is empty or duplicated")
    return rows


def load_plan(args, root=ROOT):
    if args.batch_size != 1:
        raise ValueError("The registered HF session interface requires batch_size=1; no automatic lowering")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.claim_id) or not args.owner.strip():
        raise ValueError("Invalid claim identifier or owner")
    protocol_path = relative(root, args.protocol)
    protocol = json.loads(protocol_path.read_text())
    if (protocol["schema"] != "kdm_general_vqa_direct_protocol_v1"
            or protocol["temperature"] != 0 or protocol["top_p"] != 1
            or args.model not in protocol["models"]):
        raise ValueError("Protocol does not admit the requested greedy Direct model")
    datasets, tasks = {}, {}
    for name in args.datasets:
        entry = protocol["datasets"][name]
        if type(entry["max_tokens"]) is not int or entry["max_tokens"] <= 0:
            raise ValueError("Each dataset must explicitly freeze its positive token budget")
        manifest = relative(root, entry["manifest"])
        if file_hash(manifest) != entry["manifest_sha256"]:
            raise ValueError("Frozen manifest bytes changed: " + name)
        samples = list(read_jsonl(manifest))
        if len(samples) != entry["expected_rows"]:
            raise ValueError("Frozen manifest row count differs: " + name)
        cfg = DecodeConfig(method="direct", max_tokens=entry["max_tokens"], temperature=0.0, top_p=1.0)
        datasets[name] = {"entry": entry, "identity": stable_hash(entry), "cfg": cfg}
        for sample in samples:
            required = {"id", "dataset", "split", "source_split", "question", "gold",
                        "prompt", "image_paths", "image_sha256"}
            if (not required <= sample.keys() or sample["dataset"] != name or sample["split"] != "eval"
                    or not isinstance(sample["id"], str) or not sample["id"]
                    or not isinstance(sample["prompt"], str) or not sample["prompt"].strip()
                    or not isinstance(sample["image_paths"], list) or not sample["image_paths"]
                    or not isinstance(sample["image_sha256"], list)
                    or len(sample["image_paths"]) != len(sample["image_sha256"])):
                raise ValueError("Frozen manifest fields are invalid: " + str(sample.get("id")))
            item = direct_task(sample)
            key = task_id(args.model, item)
            if key in tasks:
                raise ValueError("Duplicate sample/task key")
            for image_path, sha in zip(sample["image_paths"], sample["image_sha256"]):
                if not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{64}", sha):
                    raise ValueError("Every image needs its frozen SHA256")
                relative(root, image_path)
            tasks[key] = item
    selected = list(tasks)
    assignment_sha = None
    if args.sample_ids:
        assignment_path = relative(root, args.sample_ids)
        ids = ids_from_file(assignment_path)
        by_id = {item["sample"]["id"]: key for key, item in tasks.items()}
        if not set(ids) <= set(by_id):
            raise ValueError("Assignment contains IDs outside the selected frozen datasets")
        selected = [by_id[sample_id] for sample_id in ids]
        assignment_sha = file_hash(assignment_path)
    # CPU plan does not require all image bytes to have arrived. Execution opens
    # and verifies every selected image before invoking its unchanged processor.
    spec_path = root / f"configs/runtime/{args.model}.json"
    spec = json.loads(spec_path.read_text())
    dtype = "float16" if args.model == "onevision" else "bfloat16"
    if (spec["key"] != args.model or spec["availability"] != "resolved" or spec["dtype"] != dtype
            or spec["kwargs"].get("dtype", "bfloat16") != dtype or spec.get("api")):
        raise ValueError("The model or native precision differs from registration")
    validate_native_runtime_files(root, spec, args.model)
    definition = {"schema": "kdm_general_vqa_direct_model_v1", "model": args.model,
                  "runtime_sha256": file_hash(spec_path), "dtype": dtype,
                  "source_sha256": {name: file_hash(root / name) for name in SOURCES},
                  "engine": "registered_hf_session", "batch_size": 1,
                  "prompt_policy": "exact_manifest_prompt_no_added_suffix",
                  "seed_rule": "stable_seed(sample_id,model,replicate=0)"}
    output = relative(root, args.output)
    output.relative_to(root / "outputs/general_vqa_direct")
    return {"definition": definition, "identity": stable_hash(definition), "spec": spec,
            "datasets": datasets, "tasks": tasks, "selected": selected,
            "output": output / args.model, "assignment_sha256": assignment_sha}


def admit(args, plan, root=ROOT):
    registry_path = relative(root, args.registry)
    registry = json.loads(registry_path.read_text())
    matches = [(name, row) for name, row in registry["hosts"].items()
               if row["hostname"] == socket.gethostname() and Path(row["root"]) == root]
    if len(matches) != 1:
        raise ValueError("Execution hostname/project root is not uniquely registered")
    host, details = matches[0]
    if "d4030" in (host + details["hostname"]).lower():
        raise ValueError("d4030 is not authorized for this experiment")
    actual = copy.deepcopy(plan["spec"])
    gate_receipt = None
    if args.k100_intern:
        if args.model != "internvl35_8b" or host != "k100":
            raise ValueError("The existing InternVL single map is registered for K100 only")
        from workflows.supplemental.remaining4.k100_intern_registered_matrix import single_spec
        declaration = registry["internvl_single_gate"]
        gate_path = relative(root, declaration["path"])
        gate = json.loads(gate_path.read_text())
        if (file_hash(gate_path) != declaration["sha256"] or not gate["passed"]
                or not gate["production_allowed"] or gate["completed"] != 8
                or file_hash(root / "workflows/supplemental/remaining4/internvl_k100_single.py") != gate["factory_sha256"]
                or file_hash(relative(root, gate["operator_audit_path"])) != gate["operator_audit_sha256"]):
            raise ValueError("Existing InternVL single-map admission identity differs")
        actual = single_spec(actual)
        gate_receipt = {**declaration, "scope": "existing hardware/checkpoint/single-device map"}
    override = registry.get("runtime_overrides", {}).get(host, {}).get(args.model)
    if override is None:
        if registry.get("model_hosts", {}).get(args.model) != host:
            raise ValueError("No model runtime is registered on this host")
    else:
        if set(override) != {"environment_python", "model_path", "source_evidence"}:
            raise ValueError("Only registered checkpoint/environment paths may move")
        actual["environment_python"] = override["environment_python"]
        actual["kwargs"]["model_path"] = override["model_path"]
        actual["processor"]["path"] = override["model_path"]
    cards = args.cards.split(",")
    if (len(cards) != actual["gpu_count"] or len(cards) != len(set(cards))
            or not all(card.isdigit() for card in cards)
            or not set(cards) <= {str(card) for card in details["allowed_gpus"]}
            or os.environ.get("CUDA_VISIBLE_DEVICES") != args.cards
            or details.get("max_workers_per_gpu", 1) != 1 or os.environ.get("KDM_GPU_SLOTS", "")):
        raise ValueError("Physical GPU count, authorization or exclusive ownership differs")
    for index, card in enumerate(sorted(cards, key=int)):
        if Path(os.readlink(f"/proc/self/fd/{20 + index}")) != root / f"outputs/locks/gpu_{card}.lock":
            raise ValueError("The existing registered worker GPU lock is missing")
    environment, checkpoint = validate_environment(actual), checkpoint_identity(actual)
    observed = {}
    for line in subprocess.check_output(["nvidia-smi", "-i", args.cards,
            "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"], text=True).splitlines():
        card, uuid, free = [part.strip() for part in line.split(",")]
        observed[card] = {"uuid": uuid, "free_mib": int(free)}
    minimum = details.get("minimum_free_mib_by_model", {}).get(args.model,
                  registry.get("minimum_free_mib", {}).get(args.model))
    if set(observed) != set(cards) or minimum is None:
        raise ValueError("Missing GPU observations or registered memory admission")
    if any(observed[card]["uuid"] != details["gpu_uuids"][card]
           or observed[card]["free_mib"] < minimum for card in cards):
        raise ValueError("GPU UUID or free-memory admission failed")
    return actual, {"host": host, "project_root": str(root), "physical_gpus": cards,
            "observed_gpus": observed, "minimum_free_mib": minimum, "runtime_spec": actual,
            "environment": environment, "checkpoint": checkpoint,
            "registry": str(registry_path.relative_to(root)), "registry_sha256": file_hash(registry_path),
            "runtime_path_evidence": override, "internvl_single_gate": gate_receipt}


def validate_rows(path, plan, keys, claim_identity, eos):
    rows = list(read_jsonl(path))
    if [row.get("key") for row in rows] != keys or len(set(keys)) != len(keys):
        raise ValueError("Completed chunk key order/coverage/uniqueness differs")
    for row, key in zip(rows, keys):
        item = plan["tasks"][key]
        sample = item["sample"]
        ds = plan["datasets"][sample["dataset"]]
        if (any(row.get(field) != value for field, value in item.items())
                or row.get("identity") != plan["identity"] or row.get("claim_identity") != claim_identity
                or row.get("dataset_identity") != ds["identity"] or row.get("status") != "ok"
                or row.get("model") != plan["definition"]["model"] or not isinstance(row.get("text"), str)
                or row.get("prompt") != sample["prompt"] or row.get("config") != asdict(ds["cfg"])
                or row.get("seed") != stable_seed(sample["id"], row["model"], 0)):
            raise ValueError("Completed row does not match the frozen dataset/model/Direct condition")
        evidence = row.get("input_evidence", {})
        order = evidence.get("source_image_indices_in_input", [])
        if (evidence.get("source_image_count") != len(sample["image_paths"])
                or set(order) != set(range(len(sample["image_paths"])))
                or evidence.get("image_occurrences") != len(order)
                or evidence.get("exact_manifest_prompt_sha256") != stable_hash(sample["prompt"])
                or type(evidence.get("input_token_count")) is not int or evidence["input_token_count"] <= 0
                or not re.fullmatch("[0-9a-f]{64}", evidence.get("input_ids_sha256", ""))):
            raise ValueError("Native input evidence does not retain the exact prompt and every image")
        tokens = row.get("tokens")
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= ds["cfg"].max_tokens
                or any(type(token) is not int or token < 0 for token in tokens)
                or type(row.get("terminated")) is not bool or row["terminated"] != (tokens[-1] in eos)
                or any(token in eos for token in tokens[:-1])
                or (not row["terminated"] and len(tokens) != ds["cfg"].max_tokens)
                or not isinstance(row.get("wall_s"), (int, float))
                or not math.isfinite(row["wall_s"]) or row["wall_s"] < 0):
            raise ValueError("Completed row has invalid native token/EOS/budget/time evidence")
    return {"rows": len(rows), "truncated_rows": sum(not row["terminated"] for row in rows),
            "dataset_counts": dict(Counter(row["sample"]["dataset"] for row in rows))}


def completed_keys(plan):
    ledger = plan["output"] / "completed_keys.jsonl"
    if not ledger.exists():
        return set()
    grouped, seen = {}, set()
    for row in read_jsonl(ledger):
        if row["key"] in seen or row["identity"] != plan["identity"]:
            raise ValueError("Duplicate/foreign completed-key ledger entry")
        seen.add(row["key"])
        if row["dataset"] in plan["datasets"]:
            if row["key"] not in plan["tasks"]:
                raise ValueError("Completed key lies outside its frozen manifest")
            grouped.setdefault(row["receipt"], []).append(row)
    complete = set()
    for name, entries in grouped.items():
        receipt_path = relative(plan["output"], name)
        receipt = json.loads(receipt_path.read_text())
        raw = relative(plan["output"], receipt["raw_path"])
        owner_path = relative(plan["output"], receipt["owner_path"])
        owner = json.loads(owner_path.read_text())
        keys = [entry["key"] for entry in entries]
        if (receipt["status"] != "complete" or receipt["identity"] != plan["identity"]
                or receipt["keys"] != keys or file_hash(raw) != receipt["raw_sha256"]
                or file_hash(owner_path) != receipt["owner_sha256"]
                or stable_hash(owner) != receipt["claim_identity"]
                or any(entry["receipt_sha256"] != file_hash(receipt_path) for entry in entries)):
            raise ValueError("Completed ledger receipt/source binding differs")
        checked = validate_rows(raw, plan, keys, receipt["claim_identity"], set(receipt["eos_token_ids"]))
        if checked != receipt["validation"]:
            raise ValueError("Completed chunk no longer passes its validation")
        complete.update(keys)
    return complete


def claim(args, plan, admission):
    import fcntl
    folder = plan["output"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "ownership.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        definition = {"identity": plan["identity"], "definition": plan["definition"]}
        path = folder / "identity.json"
        if path.exists():
            if json.loads(path.read_text()) != definition:
                raise ValueError("Output directory belongs to another model/source identity")
        else:
            write_once(path, definition)
        complete = completed_keys(plan)
        pending = set(plan["selected"]) - complete
        claims = folder / "claims"
        claims.mkdir(exist_ok=True)
        for other in claims.iterdir():
            if not other.is_dir():
                continue
            previous = json.loads((other / "owner.json").read_text())
            released = set()
            if (other / "released.json").exists():
                release = json.loads((other / "released.json").read_text())
                if release["claim_identity"] != stable_hash(previous) or release["reason"] != "stop_after_completed_chunk":
                    raise ValueError("Unbound ownership release")
                released = set(release["keys"])
                if not released <= set(previous["keys"]):
                    raise ValueError("Ownership release contains keys outside its claim")
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
                 "assignment_path": args.sample_ids, "assignment_sha256": plan["assignment_sha256"],
                 "source_sha256": plan["definition"]["source_sha256"], "argv": sys.argv,
                 "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        write_once(run / "owner.json", owner)
        return run, keys, owner


def seal(plan, run, owner, number, pending_path, keys, eos):
    import fcntl
    checked = validate_rows(pending_path, plan, keys, stable_hash(owner), eos)
    raw = run / f"chunk_{number:05d}.jsonl"
    if raw.exists():
        raise ValueError("Refusing to replace a completed chunk")
    pending_path.rename(raw)
    raw.chmod(0o444)
    path = run / f"chunk_{number:05d}.complete.json"
    receipt = {"status": "complete", "completed_utc": now(), "identity": plan["identity"],
               "claim_identity": stable_hash(owner), "keys": keys, "eos_token_ids": sorted(eos),
               "raw_path": str(raw.relative_to(plan["output"])), "raw_sha256": file_hash(raw),
               "owner_path": str((run / "owner.json").relative_to(plan["output"])),
               "owner_sha256": file_hash(run / "owner.json"), "validation": checked}
    write_once(path, receipt)
    with (plan["output"] / "ownership.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = plan["output"] / "completed_keys.jsonl"
        recorded = {row["key"] for row in read_jsonl(ledger)} if ledger.exists() else set()
        if recorded & set(keys):
            raise ValueError("Completed ledger already contains a chunk key")
        with ledger.open("a", encoding="utf-8") as stream:
            for key in keys:
                stream.write(json.dumps({"key": key, "dataset": plan["tasks"][key]["sample"]["dataset"],
                    "identity": plan["identity"], "receipt": str(path.relative_to(plan["output"])),
                    "receipt_sha256": file_hash(path)}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"event": "sealed_chunk", "chunk": number, **checked}), flush=True)


def execute(args, plan):
    from PIL import Image
    from workflows.general_vqa_direct.inputs import direct_session
    actual, admission = admit(args, plan)
    run, keys, owner = claim(args, plan, admission)
    if run is None:
        return {"status": "already_complete", "selected_rows": len(plan["selected"])}
    started, active_key = time.perf_counter(), None
    try:
        backend = make_backend(actual, "cuda:0")
        loaded = time.perf_counter()
        eos, claim_identity = set(backend.eos), stable_hash(owner)
        if not eos or getattr(backend, "generate_text", None):
            raise ValueError("Native token-session backend/EOS required")
        first8, chunk_keys, number = {}, [], 0
        for index, key in enumerate(keys):
            active_key = key
            item = plan["tasks"][key]
            sample, name = item["sample"], item["sample"]["dataset"]
            cfg = plan["datasets"][name]["cfg"]
            pending_path = run / f"chunk_{number:05d}.pending.jsonl"
            row_started, images = time.perf_counter(), []
            for filename, sha in zip(sample["image_paths"], sample["image_sha256"]):
                path = relative(ROOT, filename)
                if file_hash(path) != sha:
                    raise ValueError("Image bytes differ from the frozen manifest: " + filename)
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            seed = stable_seed(sample["id"], args.model, 0)
            session = direct_session(backend, images, sample)
            input_evidence = session.general_vqa_input_evidence
            result = generate(session, None, cfg, eos, backend.decode, seed)
            del session, images
            wall = time.perf_counter() - row_started
            row = {"key": key, **item, "model": args.model, "identity": plan["identity"],
                   "dataset_identity": plan["datasets"][name]["identity"], "claim_identity": claim_identity,
                   "prompt": sample["prompt"], "reference_prompt": None, "neutral_prompt": None,
                   "config": asdict(cfg), "seed": seed, "wall_s": wall,
                   "input_evidence": input_evidence, **result}
            with pending_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            chunk_keys.append(key)
            timings = first8.setdefault(name, [])
            if len(timings) < 8:
                timings.append({"key": key, "wall_s": wall})
                if len(timings) == 8:
                    total = sum(entry["wall_s"] for entry in timings)
                    write_once(run / f"first8_{name}.json", {"status": "actual_first8", "dataset": name,
                        "identity": plan["identity"], "dataset_identity": plan["datasets"][name]["identity"],
                        "claim_identity": claim_identity, "completed_inputs": 8, "rows": timings,
                        "mean_input_wall_s": total / 8, "model_load_wall_s": loaded - started,
                        "measured_utc": now(), "config": asdict(cfg),
                        "scope": "actual outputs; completed-key accounting requires sealed chunks"})
                    print(json.dumps({"event": "actual_first8", "dataset": name, "mean_input_wall_s": total / 8}), flush=True)
            next_dataset = plan["tasks"][keys[index + 1]]["sample"]["dataset"] if index + 1 < len(keys) else None
            if len(chunk_keys) == args.chunk_rows or next_dataset != name:
                seal(plan, run, owner, number, pending_path, chunk_keys, eos)
                number += 1
                chunk_keys = []
                if (run / "STOP_AFTER_CHUNK").exists() and index + 1 < len(keys):
                    write_once(run / "released.json", {"claim_identity": claim_identity,
                        "reason": "stop_after_completed_chunk", "released_utc": now(), "keys": keys[index + 1:]})
                    return {"status": "stopped_after_completed_chunk", "completed_rows": index + 1,
                            "released_rows": len(keys) - index - 1}
        complete = completed_keys(plan)
        if not set(keys) <= complete:
            raise ValueError("Claim completion lacks verified completed-key ledger rows")
        receipt = {"status": "complete", "identity": plan["identity"], "claim_identity": claim_identity,
                   "completed_rows": len(keys), "selected_rows": len(plan["selected"]),
                   "completed_selected_rows": len(set(plan["selected"]) & complete),
                   "dataset_counts": dict(Counter(plan["tasks"][key]["sample"]["dataset"] for key in keys)),
                   "completed_utc": now(), "wall_s": time.perf_counter() - started}
        write_once(run / "complete.json", receipt)
        return receipt
    except BaseException as exc:
        write_once(run / "failed.json", {"status": "failed", "failed_utc": now(), "key": active_key,
            "claim_identity": stable_hash(owner), "error": type(exc).__name__ + ": " + str(exc),
            "traceback": traceback.format_exc(), "automatic_retry": False, "automatic_parameter_change": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--sample-ids", help="Project-relative JSONL {id:...} or one ID per line; order retained")
    parser.add_argument("--output", required=True, help="Directory under outputs/general_vqa_direct")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--cards", required=True)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--chunk-rows", type=int, choices=(64, 128), default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--k100-intern", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args)
    if args.check_plan:
        result = {"status": "CPU_plan_checked", "model_identity": plan["identity"],
                  "datasets": {name: {"identity": row["identity"], "entry": row["entry"]}
                               for name, row in plan["datasets"].items()},
                  "selected_rows": len(plan["selected"]), "batch_size": 1, "gpu_initialized": False}
    elif args.verify:
        complete = completed_keys(plan)
        expected = set(plan["selected"])
        result = {"status": "complete" if expected <= complete else "incomplete",
                  "expected_rows": len(expected), "completed_rows": len(expected & complete),
                  "dataset_counts": dict(Counter(plan["tasks"][key]["sample"]["dataset"] for key in expected & complete))}
    else:
        result = execute(args, plan)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 2 if args.verify and result["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
