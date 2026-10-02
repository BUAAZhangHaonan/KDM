#!/usr/bin/env python3
"""Seal an owned native producer at its next persisted input for a host move."""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import select
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).parent)]

from kdm.decoding import DecodeConfig
from kdm.io import file_hash, stable_hash, stable_seed
from kdm.pipeline import task_id
from native_baselines import MODELS, OUTPUT, inputs, now, task


def process(pid):
    path = Path("/proc") / str(pid)
    if not path.exists():
        return None
    state = (path / "stat").read_text().split(") ", 1)[1].split()[0]
    if state == "Z":
        return None
    return (path / "cmdline").read_bytes().replace(b"\0", b" ").decode()


def wait_exit(pid):
    for _ in range(1000):
        if process(pid) is None:
            return
        time.sleep(.01)
    raise RuntimeError("Owned producer has not exited; preserve source and stop this handoff")


def complete_lines(path):
    content = path.read_bytes()
    end = content.rfind(b"\n") + 1
    return content, [json.loads(line) for line in content[:end].splitlines()]


def execute(args):
    samples, _spec, provenance = inputs(args.model, args.method)
    directory = ROOT / OUTPUT / args.run_id
    stem = f"{args.model}_{args.method}_0000_2424"
    raw = directory / "raw" / (stem + ".jsonl")
    pilot = directory / (stem + ".pilot.json")
    claim = directory / (stem + ".claim.json")
    identity_path = raw.with_suffix(".identity.json")
    receipt_path = directory / (stem + ".sealed_partial.json")
    if receipt_path.exists() or (directory / (stem + ".complete.json")).exists():
        raise FileExistsError("Preserve existing sealed or complete source")
    proof = json.loads(pilot.read_text())
    owned = json.loads(claim.read_text())
    ledger = json.loads(identity_path.read_text())
    if (not proof["passed"] or proof["completed"] != 8
            or owned["owner"] != args.owner or owned["identity"] != proof["identity"]
            or ledger["identity"] != proof["ledger_identity"]):
        raise ValueError("The exact owned source pilot/claim/ledger is not present")
    command = process(args.producer_pid)
    if (command is None or "workflows/paper_core/native_baselines.py " not in command
            or f"--model {args.model} " not in command
            or f"--method {args.method} " not in command
            or f"--run-id {args.run_id} " not in command or "--phase full " not in command):
        raise ValueError("Producer PID does not run the exact owned native full part")
    original_lock = os.readlink(f"/proc/{args.producer_pid}/fd/20")
    if Path(original_lock).parent != ROOT / "outputs/locks":
        raise ValueError("Owned producer lock belongs to another workspace")
    initial_bytes, initial_rows = complete_lines(raw)
    if not 8 <= len(initial_rows) < 2416:
        raise ValueError("Move requires an actual source prefix and at least eight remaining inputs")
    supervisor_command = process(args.supervisor_pid)
    if supervisor_command != "bash " + str(directory / "launch.sh") + " ":
        raise ValueError("Supervisor PID does not own this exact future-method queue")
    os.kill(args.supervisor_pid, signal.SIGTERM)
    wait_exit(args.supervisor_pid)
    if process(args.producer_pid) != command:
        raise RuntimeError("Active native producer changed while cancelling future methods")
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "Cannot watch the owned ledger close event")
    stopped = False
    try:
        if libc.inotify_add_watch(fd, os.fsencode(raw), 0x00000008) < 0:
            raise OSError(ctypes.get_errno(), "Cannot watch the owned ledger")
        if not select.select([fd], [], [], args.wait_max_s)[0]:
            raise TimeoutError("No next completed input; producer continues and future queue remains cancelled")
        os.read(fd, 65536)
        os.kill(args.producer_pid, signal.SIGSTOP)
        stopped = True
        persisted, rows = complete_lines(raw)
        if len(rows) <= len(initial_rows):
            raise ValueError("Ledger event did not add a complete generated input")
        expected = [task_id(args.model, task(s, args.method)) for s in samples[:len(rows)]]
        if [row["key"] for row in rows] != expected or len(set(expected)) != len(rows):
            raise ValueError("Sealed source is not the exact ordered eval prefix")
        cfg = asdict(DecodeConfig(method=args.method))
        for row in rows:
            if (row["identity"] != ledger["identity"] or row["status"] != "ok"
                    or row["config"] != cfg or row["seed"] != stable_seed(row["sample"]["id"], args.model, 0)
                    or not row["tokens"] or len(row["tokens"]) > 32
                    or len(row["selected_log_probabilities"]) != len(row["tokens"])
                    or not all(math.isfinite(value) for value in row["selected_log_probabilities"])
                    or not math.isfinite(row["first_probability"])):
                raise ValueError("Sealed source identity/configuration/probability differs")
        if not persisted.endswith(b"\n"):
            raise ValueError("Preserve unfinished source write; no handoff is declared")
        os.kill(args.producer_pid, signal.SIGTERM)
        try:
            os.kill(args.producer_pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        stopped = False
        wait_exit(args.producer_pid)
        if raw.read_bytes() != persisted:
            raise ValueError("Source changed after the exact boundary stop")
        receipt = {
            "schema": "kdm_native_baseline_sealed_prefix_v1", "owner": args.owner,
            "model": args.model, "method": args.method, "dataset": "food101", "split": "eval",
            "whole_condition_complete": False, "completed": len(rows), "expected_full": 2424,
            "start": 0, "stop": len(rows), "remaining_start": len(rows), "remaining_stop": 2424,
            "remaining_sample_ids": [s["id"] for s in samples[len(rows):]],
            "remaining_keys": [task_id(args.model, task(s, args.method)) for s in samples[len(rows):]],
            "completed_keys_sha256": stable_hash(expected), "raw_path": str(raw),
            "raw_sha256": file_hash(raw), "identity_path": str(identity_path),
            "identity_sha256": file_hash(identity_path), "ledger_identity": ledger["identity"],
            "pilot_path": str(pilot), "pilot_sha256": file_hash(pilot),
            "claim_path": str(claim), "claim_sha256": file_hash(claim), "provenance": provenance,
            "producer_pid": args.producer_pid, "producer_command": command,
            "cancelled_supervisor_pid": args.supervisor_pid, "original_lock": original_lock,
            "stop_trigger": "next actual ledger IN_CLOSE_WRITE after fsync", "sealed_at_utc": now(),
            "helper_sha256": file_hash(Path(__file__)), "old_outputs_modified": False,
        }
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2)
        return {name: value for name, value in receipt.items()
                if name not in ("remaining_keys", "remaining_sample_ids", "provenance")}
    finally:
        os.close(fd)
        if stopped and process(args.producer_pid) == command:
            os.kill(args.producer_pid, signal.SIGCONT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--method", choices=("dola", "deco", "sid"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--producer-pid", type=int, required=True)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--owner", default="/root/native_baselines")
    parser.add_argument("--wait-max-s", type=int, default=60)
    args = parser.parse_args()
    if not 1 <= args.wait_max_s <= 60:
        parser.error("Use one bounded close-event wait of at most 60 seconds")
    (ROOT / OUTPUT / args.run_id).resolve().relative_to(ROOT / OUTPUT)
    print(json.dumps(execute(args), indent=2))


if __name__ == "__main__":
    main()
