#!/usr/bin/env python3
"""One-shot, CPU-only collection of immutable General VQA Direct chunks.

Run --collect from Windows using its existing SSH aliases. --inventory is sent
over SSH stdin and never writes to a producer. --finalize runs on the central
host, reuses generate.validate_rows, and publishes raw_paths only after every
selected chunk passes. No polling, pending transfer, GPU or ledger mutation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import socket
import subprocess
import sys
import tarfile
from types import SimpleNamespace

HOSTS = {
    "4028": {"ssh": "4028-root", "root": "/home/g203-4028/projects/knowledge-deficit-mitigation"},
    "4029": {"ssh": "4029", "root": "/home/hdd3/zhanghaonan/projects/knowledge-deficit-mitigation/supplemental/remaining11/runtime_20260930"},
    "6403": {"ssh": "6403", "root": "/home/team/zhanghaonan/TAFFC/knowledge-deficit-mitigation"},
    "k100": {"ssh": "RTX_Pro_6000", "root": "/home/k100/projects/knowledge-deficit-mitigation"},
}
MODELS = {"qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b",
          "internvl35_8b", "onevision", "phi35", "qwen3vl"}
RUN = "outputs/general_vqa_direct/run_20261004"
NATIVE_RUN = "outputs/general_vqa_direct/native_fast_20261004"
NATIVE_ENGINE = "registered_hf_native_generate_greedy_v1"
NATIVE_SOURCE = "workflows/general_vqa_direct/native_fast.py"
NATIVE_SOURCE_SHA256 = "be61ce15138e848df0f7412ff31882110a8c87e08ba484c28d4c5a3d5f3c6cf3"
COLLECTED = "outputs/general_vqa_direct/collected"
PROTOCOL = "data/general_vqa_direct_20261004/frozen/protocol.json"
SCRIPT = "workflows/general_vqa_direct/collect_sealed.py"
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]


def now():
    return datetime.now(timezone.utc).isoformat()


def collected_root(run):
    if run not in (RUN, NATIVE_RUN):
        raise ValueError("Unregistered collection run")
    return COLLECTED if run == RUN else COLLECTED + "/native_fast_20261004"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stable_hash(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode())


def strict_json(data):
    def no_constant(value):
        raise ValueError("Non-finite JSON value: " + value)
    def unique_pairs(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON object key: " + key)
            obj[key] = value
        return obj
    return json.loads(data, parse_constant=no_constant, object_pairs_hook=unique_pairs)


def safe_relative(value):
    p = PurePosixPath(value)
    if (not isinstance(value, str) or not value or p.is_absolute() or ".." in p.parts
            or "\\" in value or p.as_posix() != value or any(ord(c) < 32 for c in value)):
        raise ValueError("Unsafe relative path: " + str(value))
    return p


def inside(root, relative):
    p = root.joinpath(*safe_relative(relative).parts)
    if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source/destination escapes its named root: " + str(p))
    return p


def immutable_bytes(path):
    before = path.stat()
    if not path.is_file() or path.is_symlink() or before.st_mode & 0o222:
        raise ValueError("Expected an immutable regular file: " + str(path))
    value = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Immutable file changed during read: " + str(path))
    return value


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def inventory(root, host, run_name=RUN):
    if host not in HOSTS or str(root) != HOSTS[host]["root"]:
        raise ValueError("Source host/root is not in this four-host collection")
    collected_root(run_name)
    run = root / run_name
    report = {"schema": "general_vqa_sealed_inventory_v1", "host": host, "root": str(root),
              "hostname": socket.gethostname(), "started_utc": now(), "run": run_name,
              "files": {}, "chunks": [], "deferred": [], "ledger_snapshots": {}}
    if not run.is_dir():
        if run_name == NATIVE_RUN:
            report.update(finished_utc=now(), missing_run=True)
            return report
        raise ValueError("Missing producer output: " + str(run))
    def remember(relative):
        path = inside(run, relative)
        data = immutable_bytes(path)
        value = {"sha256": digest(data), "bytes": len(data)}
        if relative in report["files"] and report["files"][relative] != value:
            raise ValueError("Source changed between chunks: " + relative)
        report["files"][relative] = value
        return data
    for model in sorted(MODELS):
        model_dir = run / model
        if not model_dir.is_dir():
            continue
        # Enumerate receipts first. The later ledger read may contain newer
        # chunks, but those are outside this finite inventory.
        receipts = sorted(model_dir.glob("claims/*/chunk_*.complete.json"))
        if not receipts:
            continue
        ledger_path = model_dir / "completed_keys.jsonl"
        ledger_data = ledger_path.read_bytes() if ledger_path.exists() else b""
        prefix = ledger_data if not ledger_data or ledger_data.endswith(b"\n") else ledger_data[:ledger_data.rfind(b"\n") + 1]
        grouped, seen = {}, set()
        for line in prefix.splitlines():
            entry = strict_json(line)
            if entry["key"] in seen:
                raise ValueError("Duplicate source ledger key: " + model)
            seen.add(entry["key"])
            safe_relative(entry["receipt"])
            grouped.setdefault(entry["receipt"], []).append(entry)
        report["ledger_snapshots"][model] = {"complete_prefix_sha256": digest(prefix),
            "complete_prefix_bytes": len(prefix), "rows": len(seen),
            "ignored_incomplete_tail_bytes": len(ledger_data) - len(prefix)}
        for path in receipts:
            if not re.fullmatch(r"chunk_\d{5}\.complete\.json", path.name):
                raise ValueError("Unexpected receipt filename")
            receipt_relative = path.relative_to(model_dir).as_posix()
            receipt_data = immutable_bytes(path)
            receipt = strict_json(receipt_data)
            entries = grouped.get(receipt_relative, [])
            keys = receipt["keys"]
            if not keys or len(keys) != len(set(keys)):
                raise ValueError("Empty/duplicate receipt keys")
            if [v["key"] for v in entries] != keys:
                if [v["key"] for v in entries] == keys[:len(entries)] and len(entries) < len(keys):
                    report["deferred"].append({"model": model, "receipt": receipt_relative,
                        "reason": "receipt_not_fully_bound_in_ledger_snapshot"})
                    continue
                raise ValueError("Source ledger/receipt key order differs")
            if any(v["receipt_sha256"] != digest(receipt_data) or v["identity"] != receipt["identity"] for v in entries):
                raise ValueError("Source ledger receipt SHA/identity mismatch")
            roles = {"receipt": model + "/" + receipt_relative,
                     "raw": model + "/" + receipt["raw_path"],
                     "owner": model + "/" + receipt["owner_path"],
                     "identity": model + "/identity.json"}
            expected_raw = receipt_relative.replace(".complete.json", ".jsonl")
            expected_owner = str(PurePosixPath(receipt_relative).parent / "owner.json")
            if receipt["raw_path"] != expected_raw or receipt["owner_path"] != expected_owner:
                raise ValueError("Receipt points outside its exact claim/chunk")
            for relative in roles.values():
                remember(relative)
            identity = strict_json(immutable_bytes(inside(run, roles["identity"])))
            engine = identity["definition"].get("engine", "registered_hf_session")
            if (run_name == NATIVE_RUN) != (engine == NATIVE_ENGINE):
                raise ValueError("Producer engine does not belong to the selected run")
            if engine == NATIVE_ENGINE:
                owner = strict_json(immutable_bytes(inside(run, roles["owner"])))
                gate = owner["admission"]["native_fast_gate"]
                relative = model + "/gates/" + owner["claim_id"] + ".json"
                if gate["path"] != run_name + "/" + relative:
                    raise ValueError("Native gate path differs from its exact model/claim")
                roles["gate"] = relative
                if digest(remember(relative)) != gate["sha256"]:
                    raise ValueError("Native owner gate SHA differs")
            if report["files"][roles["receipt"]]["sha256"] != digest(receipt_data):
                raise ValueError("Receipt changed during inventory")
            report["chunks"].append({"model": model, "roles": roles, "ledger_entries": entries})
    report["finished_utc"] = now()
    return report


def add_unique_keys(seen, model, keys):
    values = [(model, key) for key in keys]
    if len(values) != len(set(values)) or any(value in seen for value in values):
        raise ValueError("Duplicate (model,key) across collected chunks")
    seen.update(values)


def audit_binding(inv, chunk, loader):
    """Authenticate the four files and source-ledger binding without imports."""
    values = {}
    for role, relative in chunk["roles"].items():
        safe_relative(relative)
        data = loader(relative)
        expected = inv["files"][relative]
        if len(data) != expected["bytes"] or digest(data) != expected["sha256"]:
            raise ValueError("Transferred " + role + " SHA/size differs")
        values[role] = data
    receipt, owner, identity = [strict_json(values[k]) for k in ("receipt", "owner", "identity")]
    if not values["raw"].endswith(b"\n"):
        raise ValueError("Sealed JSONL is missing its final newline")
    rows = [strict_json(line) for line in values["raw"].splitlines()]
    keys = receipt["keys"]
    if (receipt["status"] != "complete" or digest(values["raw"]) != receipt["raw_sha256"]
            or digest(values["owner"]) != receipt["owner_sha256"]
            or stable_hash(owner) != receipt["claim_identity"]
            or identity["identity"] != stable_hash(identity["definition"])
            or identity["definition"]["model"] != chunk["model"]
            or owner["identity"] != identity["identity"] or receipt["identity"] != identity["identity"]
            or owner["source_sha256"] != identity["definition"]["source_sha256"]
            or owner["admission"]["host"] != inv["host"]
            or owner["admission"]["project_root"] != inv["root"]
            or owner["hostname"] != inv["hostname"]
            or owner["admission"]["runtime_spec"]["key"] != chunk["model"]
            or owner["admission"]["runtime_spec"]["dtype"] != identity["definition"]["dtype"]
            or owner["batch_size"] != identity["definition"]["batch_size"]):
        raise ValueError("Raw/owner/receipt/model/host identity binding differs")
    owner_keys = owner["keys"]
    if len(owner_keys) != len(set(owner_keys)) or not keys or len(keys) != len(set(keys)):
        raise ValueError("Owner/receipt keys are empty or duplicated")
    start = owner_keys.index(keys[0])
    if owner_keys[start:start + len(keys)] != keys or [row["key"] for row in rows] != keys:
        raise ValueError("Raw/receipt/owner key order differs")
    roles, model = chunk["roles"], chunk["model"]
    if (roles["raw"] != model + "/" + receipt["raw_path"]
            or roles["owner"] != model + "/" + receipt["owner_path"]
            or roles["identity"] != model + "/identity.json"
            or roles["receipt"] != roles["raw"].replace(".jsonl", ".complete.json")
            or PurePosixPath(roles["owner"]).parent.name != owner["claim_id"]):
        raise ValueError("Receipt/claim paths differ")
    if [v["key"] for v in chunk["ledger_entries"]] != keys:
        raise ValueError("Frozen ledger key order differs")
    for row, entry in zip(rows, chunk["ledger_entries"]):
        if (entry["receipt_sha256"] != digest(values["receipt"])
                or entry["identity"] != identity["identity"]
                or model + "/" + entry["receipt"] != roles["receipt"]
                or entry["dataset"] != row["sample"]["dataset"]):
            raise ValueError("Frozen ledger SHA/dataset binding differs")
    eos = receipt["eos_token_ids"]
    if not eos or len(eos) != len(set(eos)) or any(type(v) is not int or v < 0 for v in eos):
        raise ValueError("Invalid declared native EOS IDs")
    engine = identity["definition"].get("engine", "registered_hf_session")
    if engine == NATIVE_ENGINE:
        declaration = owner["admission"]["native_fast_gate"]
        gate = strict_json(values["gate"])
        if (inv.get("run", RUN) != NATIVE_RUN
                or roles["gate"] != model + "/gates/" + owner["claim_id"] + ".json"
                or declaration["path"] != NATIVE_RUN + "/" + roles["gate"]
                or declaration["sha256"] != digest(values["gate"])
                or declaration.get("passed") is not True or declaration.get("completed") != 8
                or gate.get("schema") != "kdm_native_direct_exact8_gate_v1"
                or gate.get("status") != "passed" or gate.get("passed") is not True
                or gate.get("completed") != 8 or len(gate.get("rows", [])) != 8
                or gate.get("executed_model_generation") is not True
                or gate.get("engine") != NATIVE_ENGINE or gate.get("model") != model
                or gate.get("identity") != identity["identity"]
                or gate.get("source_sha256") != identity["definition"]["source_sha256"]
                or gate["source_sha256"].get(NATIVE_SOURCE) != NATIVE_SOURCE_SHA256
                or type(gate.get("pid")) is not int or gate["pid"] <= 0
                or type(gate.get("start_tick")) is not int or gate["start_tick"] < 0
                or not isinstance(gate.get("boot_id"), str) or not gate["boot_id"]
                or any(gate.get(field) != owner.get(field) for field in ("pid", "start_tick", "boot_id", "hostname"))
                or declaration.get("reference_identity") != gate["reference"]["source"]["identity"]
                or any(gate["admission"].get(field) != owner["admission"].get(field) for field in (
                    "host", "project_root", "physical_gpus", "observed_gpus", "runtime_spec",
                    "environment", "checkpoint", "registry", "registry_sha256"))
                or any(row.get("native_fast_gate") != declaration for row in rows)):
            raise ValueError("Native gate/owner/source/runtime binding differs or gate was not actually passed")
    elif engine != "registered_hf_session" or "gate" in roles:
        raise ValueError("Unknown producer engine or unexpected gate")
    return receipt, owner, identity, rows


def audit_numbers_and_images(rows, parts_for_sample, engine="registered_hf_session"):
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite numeric evidence")
        if isinstance(value, dict):
            for item in value.values(): finite(item)
        if isinstance(value, list):
            for item in value: finite(item)
    for row in rows:
        finite(row)
        if engine == NATIVE_ENGINE:
            if (row.get("engine") != NATIVE_ENGINE
                    or row.get("probability_recording") != "not_requested_not_measured"
                    or any(field in row for field in ("selected_log_probabilities", "trace",
                        "sequence_log_probability", "first_probability"))):
                raise ValueError("Native fast row must explicitly omit unmeasured probabilities")
        else:
            if engine != "registered_hf_session" or row.get("engine", engine) != engine:
                raise ValueError("Unexpected row engine")
            tokens, lp, trace = row["tokens"], row["selected_log_probabilities"], row["trace"]
            if (len(lp) != len(tokens) or len(trace) != len(tokens)
                    or any(type(v) not in (int, float) or v > 1e-8 for v in lp)
                    or not math.isclose(row["sequence_log_probability"], sum(lp), rel_tol=1e-9, abs_tol=1e-9)
                    or not math.isclose(row["first_probability"], math.exp(lp[0]), rel_tol=1e-9, abs_tol=1e-12)):
                raise ValueError("Invalid selected/sequence/first probability evidence")
            for token, probability, item in zip(tokens, lp, trace):
                # At temperature zero sampling_distribution is a point mass.
                if item["token"] != token or item["log_probability"] != probability or item["sampling_log_probability"] != 0.0:
                    raise ValueError("Token trace differs from greedy Direct probability evidence")
        sample = row["sample"]
        expected = [value for kind, value in parts_for_sample(sample, len(sample["image_paths"])) if kind == "image"] if sample["dataset"] == "mmmu" else [0]
        if row["input_evidence"]["source_image_indices_in_input"] != expected:
            raise ValueError("Native full-image input order differs from frozen prompt")
        if row.get("reference_prompt") is not None or row.get("neutral_prompt") is not None:
            raise ValueError("Direct row contains an extra reference/neutral prompt")


def audit_native_gate(gate, plan, eos, parts_for_sample, stable_seed):
    rows = gate["rows"]
    keys = [row["key"] for row in rows]
    references = gate["reference"]["sealed_references"]
    if (len(keys) != 8 or len(set(keys)) != 8 or [row["key"] for row in references] != keys
            or gate["reference"]["source"]["identity"] != plan["identity"]):
        raise ValueError("Gate lacks exactly eight distinct bound original references")
    for ref in references:
        for field in ("receipt", "raw_path"):
            safe_relative(ref[field])
        if any(not re.fullmatch(r"[0-9a-f]{64}", ref[field]) for field in ("receipt_sha256", "raw_sha256")):
            raise ValueError("Gate reference source SHA is invalid")
    for row in rows:
        sample = plan["tasks"][row["key"]]["sample"]
        ds = plan["datasets"][sample["dataset"]]
        ids, evidence, tokens = row["input_ids"], row["input_evidence"], row["tokens"]
        if (not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], list) or not ids[0]
                or any(type(value) is not int for value in ids[0])
                or row["sample_id"] != sample["id"] or row["dataset"] != sample["dataset"]
                or row["dataset_identity"] != ds["identity"] or row["config"] != asdict(ds["cfg"])
                or row["seed"] != stable_seed(sample["id"], plan["definition"]["model"], 0)
                or evidence["input_ids_sha256"] != stable_hash(ids)
                or evidence["input_token_count"] != len(ids[0])
                or evidence["exact_manifest_prompt_sha256"] != stable_hash(sample["prompt"])
                or evidence["source_image_count"] != len(sample["image_paths"])):
            raise ValueError("Native gate exact input IDs/prompt/dataset binding differs")
        order = [value for kind, value in parts_for_sample(sample, len(sample["image_paths"])) if kind == "image"] if sample["dataset"] == "mmmu" else [0]
        if evidence["source_image_indices_in_input"] != order or evidence["image_occurrences"] != len(order):
            raise ValueError("Native gate image input order differs")
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= ds["cfg"].max_tokens
                or any(type(token) is not int or token < 0 for token in tokens)
                or not isinstance(row["reference_tokens"], list)
                or any(type(token) is not int or token < 0 for token in row["reference_tokens"])
                or tokens != row["reference_tokens"] or row.get("tokens_equal") is not True
                or type(row["terminated"]) is not bool or type(row["reference_terminated"]) is not bool
                or row["terminated"] != row["reference_terminated"]
                or row.get("termination_equal") is not True or row["terminated"] != (tokens[-1] in eos)
                or any(token in eos for token in tokens[:-1])
                or (not row["terminated"] and len(tokens) != ds["cfg"].max_tokens)
                or any(type(row[field]) not in (int, float) or not math.isfinite(row[field]) or row[field] < 0
                       for field in ("native_wall_s", "reference_wall_s"))):
            raise ValueError("Native gate full token/EOS/budget equivalence differs")


def install_immutable(path, data):
    if path.exists():
        if immutable_bytes(path) != data:
            raise ValueError("Existing collected file differs; refusing replacement: " + str(path))
        return False
    write_new(path, data)
    return True


def finalize(root, snapshot, run_name=RUN):
    if str(root) != HOSTS["4028"]["root"]:
        raise ValueError("Finalize is confined to the central project")
    collected = collected_root(run_name)
    dest = root / collected / "snapshots" / snapshot
    transport = strict_json((dest / "transport.json").read_bytes())
    if transport["collector_sha256"] != digest(Path(__file__).read_bytes()):
        raise ValueError("Collector source differs from transferred snapshot")
    if transport.get("run", RUN) != run_name:
        raise ValueError("Transport run differs from selected collection")
    sys.path[:0] = [str(root), str(root / "src")]
    from workflows.general_vqa_direct import generate as producer
    from workflows.general_vqa_direct.inputs import parts_for_sample
    protocol = strict_json((root / PROTOCOL).read_bytes())
    seen, raw_paths, chunks, plans = set(), [], [], {}
    counts, truncations, copied, reused = Counter(), Counter(), 0, 0
    inventories = []
    for host in HOSTS:
        inv_bytes = (dest / (host + ".inventory.json")).read_bytes()
        meta = transport["hosts"][host]
        if digest(inv_bytes) != meta["inventory_sha256"]:
            raise ValueError("Inventory transfer SHA differs")
        inv = strict_json(inv_bytes)
        if inv["host"] != host or inv["root"] != HOSTS[host]["root"] or inv["run"] != run_name:
            raise ValueError("Inventory host/root/run differs")
        inventories.append(inv)
        archive = None
        members = {}
        if host != "4028":
            archive_path = dest / "incoming" / (host + ".tar")
            if digest(archive_path.read_bytes()) != meta["archive_sha256"]:
                raise ValueError("Transport archive SHA differs")
            archive = tarfile.open(archive_path, "r:")
            for member in archive.getmembers():
                if not member.isfile() or member.name in members or member.name not in inv["files"]:
                    raise ValueError("Archive contains duplicate/non-regular/unselected members")
                safe_relative(member.name)
                members[member.name] = member
            if set(members) != set(inv["files"]):
                raise ValueError("Archive member coverage differs")
        def source_bytes(relative):
            if archive is None:
                return immutable_bytes(inside(root / run_name, relative))
            return archive.extractfile(members[relative]).read()
        try:
            for chunk in inv["chunks"]:
                receipt, owner, identity, rows = audit_binding(inv, chunk, source_bytes)
                model = chunk["model"]
                for name, entry in owner["dataset_entries"].items():
                    if protocol["datasets"].get(name) != entry:
                        raise ValueError("Owner dataset differs from canonical frozen protocol")
                plan_key = (model, tuple(sorted(owner["dataset_entries"])))
                if plan_key not in plans:
                    args = SimpleNamespace(batch_size=1, claim_id="collector", owner="collector", protocol=PROTOCOL,
                        model=model, datasets=list(plan_key[1]), sample_ids=None, output=run_name)
                    plans[plan_key] = producer.load_plan(args, root=root)
                    if run_name == NATIVE_RUN:
                        native_sha = digest((root / NATIVE_SOURCE).read_bytes())
                        if native_sha != NATIVE_SOURCE_SHA256:
                            raise ValueError("Native producer revision differs from fixed admitted SHA")
                        definition = plans[plan_key]["definition"]
                        definition["engine"] = NATIVE_ENGINE
                        definition["source_sha256"][NATIVE_SOURCE] = native_sha
                        plans[plan_key]["identity"] = stable_hash(definition)
                plan = plans[plan_key]
                if identity != {"identity": plan["identity"], "definition": plan["definition"]}:
                    raise ValueError("Model/source identity differs from central frozen files")
                if not set(owner["keys"]) <= set(plan["tasks"]):
                    raise ValueError("Owner contains keys outside the frozen dataset plan")
                engine = identity["definition"].get("engine", "registered_hf_session")
                audit_numbers_and_images(rows, parts_for_sample, engine)
                roles = chunk["roles"]
                if engine == NATIVE_ENGINE:
                    gate = strict_json(source_bytes(roles["gate"]))
                    gate_key = (model, tuple(sorted({row["dataset"] for row in gate["rows"]})), "original_gate")
                    if gate_key not in plans:
                        gate_args = SimpleNamespace(batch_size=1, claim_id="collector", owner="collector", protocol=PROTOCOL,
                            model=model, datasets=list(gate_key[1]), sample_ids=None, output=RUN)
                        plans[gate_key] = producer.load_plan(gate_args, root=root)
                    audit_native_gate(gate, plans[gate_key], set(receipt["eos_token_ids"]), parts_for_sample, producer.stable_seed)
                base = root / run_name if host == "4028" else root / collected / host
                # Files are authenticated before installation. Only the separate
                # collected tree is writable; original central chunks are indexed.
                if host != "4028":
                    for relative in roles.values():
                        fresh = install_immutable(inside(base, relative), source_bytes(relative))
                        copied += int(fresh)
                        reused += int(not fresh)
                raw = inside(base, roles["raw"])
                checked = producer.validate_rows(raw, plan, receipt["keys"], receipt["claim_identity"], set(receipt["eos_token_ids"]))
                if checked != receipt["validation"]:
                    raise ValueError("Recomputed row validation differs from sealed receipt")
                add_unique_keys(seen, model, receipt["keys"])
                if str(raw) in raw_paths:
                    raise ValueError("Duplicate raw path")
                raw_paths.append(str(raw))
                for dataset, number in checked["dataset_counts"].items():
                    counts[host + "/" + model + "/" + dataset] += number
                truncations[model] += checked["truncated_rows"]
                chunks.append({"host": host, "model": model, "raw_path": str(raw),
                    "raw_sha256": receipt["raw_sha256"], "receipt_path": str(inside(base, roles["receipt"])),
                    "receipt_sha256": inv["files"][roles["receipt"]]["sha256"],
                    "owner_path": str(inside(base, roles["owner"])), "owner_sha256": receipt["owner_sha256"],
                    "identity_path": str(inside(base, roles["identity"])),
                    "identity_sha256": inv["files"][roles["identity"]]["sha256"],
                    "identity": receipt["identity"], "claim_identity": receipt["claim_identity"],
                    "engine": engine, **({"native_fast_gate_path": str(inside(base, roles["gate"])),
                        "native_fast_gate_sha256": inv["files"][roles["gate"]]["sha256"]} if engine == NATIVE_ENGINE else {}), **checked})
        finally:
            if archive is not None: archive.close()
    report = {"schema": "general_vqa_sealed_collection_v1", "status": "passed", "finished_utc": now(),
        "snapshot": snapshot, "run": run_name, "collected_root": collected,
        "central_root": str(root), "collector_sha256": transport["collector_sha256"],
        "protocol_sha256": digest((root / PROTOCOL).read_bytes()),
        "validation_source_sha256": digest((root / "workflows/general_vqa_direct/generate.py").read_bytes()),
        "rows": len(seen), "chunks": len(chunks), "unique_model_key_pairs": len(seen),
        "counts_by_host_model_dataset": dict(sorted(counts.items())), "truncated_rows_by_model": dict(truncations),
        "copied_file_operations": copied, "reused_file_operations": reused,
        "deferred": [{"host": inv["host"], **item} for inv in inventories for item in inv["deferred"]],
        "source_inventory_intervals": {inv["host"]: [inv["started_utc"], inv["finished_utc"]] for inv in inventories},
        "gpu_initialized": False, "active_outputs_modified": False, "raw_files": chunks}
    paths_bytes = ("\n".join(raw_paths) + ("\n" if raw_paths else "")).encode()
    report["raw_paths_sha256"] = digest(paths_bytes)
    write_new(dest / "raw_paths.txt", paths_bytes)
    write_new(dest / "manifest.json", json_bytes(report))
    summary = "# Sealed Direct collection\n\n" + f"Status: passed. {len(seen)} unique (model,key) rows in {len(chunks)} immutable chunks.\n\n"
    summary += "| Host / model / dataset | Rows |\n|---|---:|\n"
    summary += "".join(f"| {key} | {value} |\n" for key, value in sorted(counts.items()))
    summary += f"\nDeferred receipts: {len(report['deferred'])}. No pending files, active model directories, producer ledgers or GPUs were modified.\n"
    write_new(dest / "summary.md", summary.encode())
    return {key: report[key] for key in ("status", "rows", "chunks", "counts_by_host_model_dataset", "deferred")} | {
        "raw_paths": str(dest / "raw_paths.txt"), "manifest": str(dest / "manifest.json")}


def ssh(host, command, **kwargs):
    return subprocess.run(["ssh", *SSH_OPTIONS, HOSTS[host]["ssh"], command], check=True, **kwargs)


def collect(local_stage, snapshot, run_name=RUN):
    collected = collected_root(run_name)
    local = local_stage / snapshot
    local.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    inventories = {}
    def get_inventory(host):
        command = "python3 - --inventory --host " + shlex.quote(host) + " --root " + shlex.quote(HOSTS[host]["root"]) + " --run " + shlex.quote(run_name)
        result = ssh(host, command, input=source, stdout=subprocess.PIPE)
        inv = strict_json(result.stdout)
        write_new(local / (host + ".inventory.json"), result.stdout)
        return host, inv
    with ThreadPoolExecutor(max_workers=4) as pool:
        for host, inv in pool.map(get_inventory, HOSTS):
            inventories[host] = inv
            print(json.dumps({"event": "inventoried", "host": host, "chunks": len(inv["chunks"]), "deferred": len(inv["deferred"])}), flush=True)
    transport = {"collector_sha256": digest(source), "created_utc": now(), "run": run_name, "hosts": {}}
    for host, inv in inventories.items():
        transport["hosts"][host] = {"inventory_sha256": digest((local / (host + ".inventory.json")).read_bytes())}
        if host != "4028":
            archive_path = local / (host + ".tar")
            names = b"".join((name + "\0").encode() for name in sorted(inv["files"]))
            command = "tar -cf -" + (" -C " + shlex.quote(HOSTS[host]["root"] + "/" + run_name) if names else "") + " --null -T -"
            with archive_path.open("xb") as stream:
                ssh(host, command, input=names, stdout=stream)
            transport["hosts"][host]["archive_sha256"] = digest(archive_path.read_bytes())
    write_new(local / "transport.json", json_bytes(transport))
    central = HOSTS["4028"]["root"]
    remote = central + "/" + collected + "/snapshots/" + snapshot
    setup = "from pathlib import Path; p=Path(" + repr(remote) + "); p.mkdir(parents=True,exist_ok=False); (p/'incoming').mkdir()"
    ssh("4028", "python3 -c " + shlex.quote(setup))
    for path in sorted(local.iterdir()):
        target = remote + ("/incoming/" if path.suffix == ".tar" else "/") + path.name
        subprocess.run(["scp", *SSH_OPTIONS, str(path), HOSTS["4028"]["ssh"] + ":" + target], check=True)
    command = "PYTHONDONTWRITEBYTECODE=1 " + shlex.quote(central + "/venv/bin/python") + " " + shlex.quote(central + "/" + SCRIPT) + " --finalize --root " + shlex.quote(central) + " --snapshot " + shlex.quote(snapshot) + " --run " + shlex.quote(run_name)
    result = ssh("4028", command, stdout=subprocess.PIPE)
    final = strict_json(result.stdout)
    for name in ["manifest.json", "raw_paths.txt", "summary.md"]:
        subprocess.run(["scp", *SSH_OPTIONS, HOSTS["4028"]["ssh"] + ":" + remote + "/" + name, str(local / name)], check=True)
    print(json.dumps({**final, "local_snapshot": str(local)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--inventory", action="store_true")
    modes.add_argument("--collect", action="store_true")
    modes.add_argument("--finalize", action="store_true")
    parser.add_argument("--host", choices=tuple(HOSTS))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--local-stage", type=Path)
    parser.add_argument("--run", choices=(RUN, NATIVE_RUN), default=RUN)
    parser.add_argument("--snapshot", default=datetime.now(timezone.utc).strftime("sealed_%Y%m%dT%H%M%S_%fZ"))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.snapshot):
        raise ValueError("Invalid snapshot name")
    if args.inventory:
        print(json.dumps(inventory(args.root, args.host, args.run), ensure_ascii=False, allow_nan=False))
    elif args.finalize:
        print(json.dumps(finalize(args.root, args.snapshot, args.run), ensure_ascii=False, allow_nan=False))
    else:
        if args.local_stage is None:
            parser.error("--collect requires an explicit workspace --local-stage")
        collect(args.local_stage.resolve(), args.snapshot, args.run)


if __name__ == "__main__":
    main()
