#!/usr/bin/env python3
"""Seal one owned registered Food matrix worker after its next persisted response."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import signal
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from kdm.io import file_hash, read_jsonl, stable_hash, within
from workflows.supplemental.remaining11 import generate


def process(pid):
    path = Path("/proc") / str(pid)
    if not path.exists() or (path / "stat").read_text().split(") ", 1)[1].split()[0] == "Z":
        return None
    return (path / "cmdline").read_bytes().split(b"\0")[:-1]


def wait_exit(pid):
    for _ in range(1000):
        if process(pid) is None:
            return
        time.sleep(.01)
    raise RuntimeError("Owned producer did not exit; do not dispatch replacement keys")


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def preflight(args):
    run = within(ROOT, args.run_root)
    run.relative_to(ROOT / "outputs/supplemental/remaining4")
    claim = run / "claims" / args.model / "food101/formal" / args.claim_id
    record = run / "records" / args.model / "food101/formal" / args.claim_id
    raw = run / "raw" / args.model / "food101/formal" / args.claim_id
    owner = json.loads((claim / "owner.json").read_text())
    admitted = json.loads((record / "admission.json").read_text())
    if ((record / "complete.json").exists() or (record / "failure.json").exists()
            or (record / "sealed_partial_receipt.json").exists()
            or owner["owner"] != args.owner or owner["claim_id"] != args.claim_id
            or owner["pid"] != args.producer_pid or admitted["task_plan"] != owner["plan"]
            or owner["keys_sha256"] != file_hash(claim / "keys.jsonl")):
        raise ValueError("Exact unfinished owned registered source is absent")
    summary = owner["plan"]
    request = SimpleNamespace(model=args.model, dataset="food101", stage="formal",
        missing_keys=summary["missing_keys_path"], key_start=summary["key_start"],
        key_stop=summary["key_stop"], shard=summary["shard"], n_shards=summary["n_shards"])
    plan = generate.load_plan(request)
    if plan["summary"] != summary or admitted["registered_backend"] != plan["spec"]:
        raise ValueError("Frozen source plan, checkpoint, or provenance changed")
    tasks = list(generate.selected_tasks(plan))
    keys = [generate.generation_key(args.model, "formal", task) for task in tasks]
    if [row["key"] for row in read_jsonl(claim / "keys.jsonl")] != keys:
        raise ValueError("Original claim is not its exact ordered registered key list")
    command = process(args.producer_pid)
    entries = {
        "registered": os.fsencode(ROOT / "workflows/supplemental/remaining4/registered_matrix.py"),
        "k100_single": os.fsencode(ROOT / "workflows/supplemental/remaining4/k100_intern_registered_matrix.py"),
    }
    actual_entries = [name for name, path in entries.items() if command and command.count(path) == 1]
    if command is None or len(actual_entries) != 1:
        raise ValueError("Producer PID is not the exact registered VCD/M3ID matrix entry")
    text = [part.decode() for part in command]
    for flag, value in (("--run-root", args.run_root),
            ("--claim-id", args.claim_id), ("--owner", args.owner)):
        if text.count(flag) != 1 or text[text.index(flag) + 1] != value:
            raise ValueError("Producer command differs from its owned source: " + flag)
    if actual_entries[0] == "registered":
        if text.count("--model") != 1 or text[text.index("--model") + 1] != args.model:
            raise ValueError("Registered producer model differs from its owned source")
    else:
        backend = admitted["backend"]
        if (ROOT != Path("/home/k100/projects/knowledge-deficit-mitigation")
                or args.model != "internvl35_8b" or "--model" in text
                or text.count("--phase") != 1 or text[text.index("--phase") + 1] != "production"
                or text.count("--condition-gate") != 1 or text.count("--host-registry") != 1
                or backend["factory"] != "workflows.supplemental.remaining4.internvl_k100_single:InternVLK100SingleBackend"
                or backend["gpu_count"] != 1 or backend["dtype"] != "bfloat16"
                or set(backend["kwargs"]["device_map"].values()) != {0}
                or backend["kwargs"]["max_memory"] != {"0": "44GiB"}
                or admitted["registered_backend"]["factory"] != "kdm.models.internvl_dual:InternVLDualBackend"):
            raise ValueError("Source is not the explicitly admitted K100 single-card production route")
        gate_path = within(ROOT, text[text.index("--condition-gate") + 1])
        gate = json.loads(gate_path.read_text())
        registry_path = within(ROOT, text[text.index("--host-registry") + 1])
        if (gate.get("passed") is not True or gate.get("production_allowed") is not True
                or gate.get("completed") != 8 or gate.get("scientific_unique_increment") != 0
                or gate["runner_sha256"] != file_hash(Path(os.fsdecode(entries["k100_single"])))
                or gate["factory_sha256"] != file_hash(ROOT / "workflows/supplemental/remaining4/internvl_k100_single.py")
                or gate["registry_sha256"] != file_hash(registry_path)):
            raise ValueError("Exact immutable single-card production gate or source identity changed")
    cards = admitted["runtime_admission"]["execution"]["physical_gpus"]
    if actual_entries[0] == "k100_single" and [int(card) for card in cards] != [0]:
        raise ValueError("K100 single-card source has an unexpected physical GPU")
    locks = [os.readlink(f"/proc/{args.producer_pid}/fd/{20 + offset}")
             for offset in range(len(cards))]
    if locks != [str(ROOT / "outputs/locks" / f"gpu_{card}.lock")
                 for card in sorted(cards, key=int)]:
        raise ValueError("Inherited physical GPU locks belong to another source")
    for flag, value in (("--missing-keys", summary["missing_keys_path"]),
            ("--key-start", str(summary["key_start"])),
            ("--key-stop", str(summary["key_stop"]))):
        if text.count(flag) != 1 or text[text.index(flag) + 1] != value:
            raise ValueError("Matrix producer differs from exact owned input range: " + flag)
    if text.count("--execute") != 1:
        raise ValueError("Matrix worker is not an executing single claim")
    return run, claim, record, raw, plan, tasks, keys, command, locks


def inventory(raw, plan, tasks):
    bindings, parts, count = [], [], 0
    for path in sorted(raw.glob("formal_*.jsonl")):
        content = path.read_bytes()
        if not content.endswith(b"\n"):
            raise ValueError("Preserve an incomplete raw write; no source handoff is declared")
        rows = [json.loads(line) for line in content.splitlines()]
        if not rows:
            raise ValueError("Source has an empty raw part")
        sidecar = path.with_suffix(".identity.json")
        definition = json.loads(sidecar.read_text())["definition"]
        identity = {key: value for key, value in definition.items()
                    if key not in ("shard", "n_shards", "base_config")}
        selected = tasks[count:count + len(rows)]
        expected = [generate.generation_key(plan["model"], "formal", task) for task in selected]
        if [row["key"] for row in rows] != expected:
            raise ValueError("Actual raw rows are not the exact admitted ordered prefix")
        proof = generate.verify_output(path, plan, selected, identity,
            plan["summary"]["shard"], plan["summary"]["n_shards"])
        parts.append({"path": str(path.relative_to(ROOT)), "rows": len(rows),
                      "raw_sha256": proof["raw_sha256"],
                      "identity_sha256": proof["identity_sha256"],
                      "completed_prefix_validation": proof})
        bindings.extend({"key": row["key"], "sample_id": row["sample"]["id"],
                         "raw_path": str(path.relative_to(ROOT)), "raw_line": index + 1}
                        for index, row in enumerate(rows))
        count += len(rows)
    return count, parts, bindings


def execute(args):
    _run, claim, record, raw, plan, tasks, keys, command, locks = preflight(args)
    initial = sum(len(path.read_bytes().splitlines()) for path in raw.glob("formal_*.jsonl"))
    if not 1 <= initial < len(keys) - 8:
        raise ValueError("Move requires actual generated rows and at least eight remaining keys")
    if args.check_source:
        return {"source_checked": True, "actual_rows_at_check": initial,
                "owned_rows": len(keys), "producer_active": True,
                "source_modified": False, "gpu_initialized": False}
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if fd < 0 or libc.inotify_add_watch(fd, os.fsencode(raw), 0x00000008) < 0:
        if fd >= 0:
            os.close(fd)
        raise OSError(ctypes.get_errno(), "Cannot observe the actual ledger close event")
    stopped = False
    try:
        if process(args.producer_pid) != command:
            raise RuntimeError("Owned matrix worker changed before complete-input seal")
        deadline = time.monotonic() + args.wait_max_s
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                raise TimeoutError("No next persisted input; original owned matrix worker continues")
            os.read(fd, 65536)
            count = sum(len(path.read_bytes().splitlines()) for path in raw.glob("formal_*.jsonl"))
            if count > initial:
                break
        os.kill(args.producer_pid, signal.SIGSTOP)
        stopped = True
        count, parts, bindings = inventory(raw, plan, tasks)
        if not initial < count < len(keys) or len(keys) - count < 8:
            raise ValueError("Actual persisted boundary does not leave the planned continuation")
        frozen = {item["path"]: item["raw_sha256"] for item in parts}
        os.kill(args.producer_pid, signal.SIGTERM)
        os.kill(args.producer_pid, signal.SIGCONT)
        stopped = False
        wait_exit(args.producer_pid)
        if any(file_hash(ROOT / path) != digest for path, digest in frozen.items()):
            raise ValueError("Source changed after its actual complete-input boundary stop")
        for name, rows in (("sealed_completed_keys.jsonl", bindings),
                           ("sealed_remaining_keys.jsonl", [{"key": key} for key in keys[count:]])):
            with (record / name).open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        receipt = {"schema": "kdm_selected4_registered_sealed_prefix_v1",
            "model": args.model, "dataset": "food101", "stage": "formal",
            "claim_id": args.claim_id, "owner": args.owner,
            "completed": count, "expected_original_claim": len(keys),
            "remaining": len(keys) - count, "parts": parts,
            "original_owner_path": str((claim / "owner.json").relative_to(ROOT)),
            "original_owner_sha256": file_hash(claim / "owner.json"),
            "original_keys_path": str((claim / "keys.jsonl").relative_to(ROOT)),
            "original_keys_sha256": file_hash(claim / "keys.jsonl"),
            "admission_path": str((record / "admission.json").relative_to(ROOT)),
            "admission_sha256": file_hash(record / "admission.json"),
            "completed_keys_path": str((record / "sealed_completed_keys.jsonl").relative_to(ROOT)),
            "completed_keys_sha256": file_hash(record / "sealed_completed_keys.jsonl"),
            "remaining_keys_path": str((record / "sealed_remaining_keys.jsonl").relative_to(ROOT)),
            "remaining_keys_sha256": file_hash(record / "sealed_remaining_keys.jsonl"),
            "ordered_completed_keys_sha256": stable_hash(keys[:count]),
            "producer_pid": args.producer_pid, "producer_command": [part.decode() for part in command],
            "cancelled_supervisor_pid": None, "single_matrix_claim": True,
            "no_external_future_queue_cancelled": True, "original_locks": locks,
            "stop_trigger": "next actual raw close; verified complete newline and admitted ordered prefix",
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "helper_sha256": file_hash(Path(__file__)),
            "whole_claim_complete": False, "scientific_failure": False,
            "source_originals_modified": False, "scientific_parameters_changed": False}
        write_new(record / "sealed_partial_receipt.json", receipt)
        return {key: value for key, value in receipt.items() if key != "parts"}
    finally:
        os.close(fd)
        if stopped and process(args.producer_pid) == command:
            os.kill(args.producer_pid, signal.SIGCONT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("internvl35_8b", "onevision", "phi35", "qwen3vl"), required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--producer-pid", type=int, required=True)
    parser.add_argument("--wait-max-s", type=int, default=60)
    parser.add_argument("--check-source", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.wait_max_s <= 60:
        parser.error("Use one close-event wait of at most sixty seconds")
    print(json.dumps(execute(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
