#!/usr/bin/env python3
"""Explicit validator correction and CPU recovery for frozen Hallusion attempts.

The base protocol/model identity and original decoder remain unchanged. Every
v2 owner/receipt names the separately frozen correction and these new sources.
Recovery is explicit, requires the original owner process to be stopped, retains
the original pending/owner/failed bytes, and never loads a model or regenerates.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from workflows.hallusion_reference import generate_independent128 as base

BASE_PROTOCOL_SHA256 = "b2023acdb24a3907c243444022eb8624a12d665ede8e531e87b30f2dd1974b9c"
BASE_GENERATOR = "workflows/hallusion_reference/generate_independent128.py"
BASE_GENERATOR_SHA256 = "79dd9a85cbf79eb6f4bbc20b281e6e7d99c94fbe0a5e85e805c4746244cd8529"
CORRECTION = base.BASE + "/registration/validation_correction.json"
SOURCES = ("workflows/hallusion_reference/generate_independent128_v2.py",
           "workflows/hallusion_reference/test_independent128_v2.py")
POLICY = "sampling_log_probability_validated_separately_from_base_log_probability"


def _corrected_validator():
    # The frozen v1 function is copied in memory, deleting exactly the mistaken
    # comparison. All other conditions remain byte-for-byte identical. The
    # original module/file is never rewritten, and its SHA is checked below.
    source = inspect.getsource(base.validate_rows)
    mistaken = '                    or not math.isclose(step["sampling_log_probability"], logp, rel_tol=1e-12, abs_tol=1e-12)\n'
    if source.count(mistaken) != 1:
        raise ValueError("The frozen v1 validator comparison no longer has its registered form")
    namespace = dict(vars(base))
    exec(compile(source.replace(mistaken, ""), __file__ + ":corrected_v1_validator", "exec"), namespace)
    return namespace["validate_rows"]


validate_rows = _corrected_validator()
_original_completed_keys = base.completed_keys


def load_plan(args, root=ROOT):
    if args.protocol_sha256 != BASE_PROTOCOL_SHA256:
        raise ValueError("v2 requires the exact already-frozen v1 protocol")
    plan = base.load_plan(args, root=root)
    root = plan["root"]
    if base.file_hash(root / BASE_GENERATOR) != BASE_GENERATOR_SHA256:
        raise ValueError("Original frozen v1 generator bytes changed")
    if args.validation_correction != CORRECTION:
        raise ValueError("The append-only validation correction path is fixed")
    path = base.relative(root, args.validation_correction)
    digest = base.file_hash(path)
    if not base._sha(args.validation_correction_sha256) or digest != args.validation_correction_sha256:
        raise ValueError("Validation correction bytes differ from the explicitly frozen SHA256")
    correction = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema": "kdm_hallusion_independent128_validation_correction_v2",
                "base_protocol": base.PROTOCOL, "base_protocol_sha256": BASE_PROTOCOL_SHA256,
                "base_generator": BASE_GENERATOR, "base_generator_sha256": BASE_GENERATOR_SHA256,
                "validator_policy": POLICY}
    if any(correction.get(field) != value for field, value in expected.items()):
        raise ValueError("Validation correction is not bound to the original v1 experiment")
    sources = {name: base.file_hash(root / name) for name in SOURCES}
    if not base._same(correction.get("source_sha256"), sources) or not str(correction.get("reason", "")).strip():
        raise ValueError("Correction source identity or explicit approved reason is missing")
    plan["validation_correction"] = {"path": args.validation_correction, "sha256": digest,
                                     "source_sha256": sources}
    return plan


def completed_keys(plan):
    complete = _original_completed_keys(plan)
    ledger = plan["output"] / "completed_keys.jsonl"
    if not ledger.exists():
        return complete
    for name in {row["receipt"] for row in base.read_jsonl(ledger)}:
        receipt_path = base.relative(plan["output"], name)
        receipt = json.loads(receipt_path.read_text())
        owner = json.loads(base.relative(plan["output"], receipt["owner_path"]).read_text())
        correction = receipt.get("validation_correction")
        if owner.get("validation_correction") is not None:
            if not base._same(owner["validation_correction"], plan["validation_correction"]):
                raise ValueError("v2 owner validation registration differs")
            if not base._same(correction, owner["validation_correction"]):
                raise ValueError("v2 receipt does not bind its owner's validation registration")
        if correction is not None and not base._same(correction, plan["validation_correction"]):
            raise ValueError("Corrected receipt has an unregistered validation source")
        source = receipt.get("recovery_source")
        if source is not None:
            if correction is None:
                raise ValueError("Recovered receipt lacks its correction registration")
            pending = base.relative(plan["output"], source["pending_path"])
            recovery_path = base.relative(plan["output"], source["recovery_path"])
            recovery = json.loads(recovery_path.read_text())
            if (base.file_hash(pending) != source["pending_sha256"]
                    or receipt["raw_sha256"] != source["pending_sha256"]
                    or recovery["identity"] != plan["identity"]
                    or recovery["claim_identity"] != receipt["claim_identity"]
                    or not base._same(recovery["validation_correction"], correction)
                    or recovery.get("regenerated_rows") != 0 or recovery.get("model_execution") is not False):
                raise ValueError("Recovered original-pending/source binding differs")
            records = [row for row in recovery["sealed_pending"] if row["receipt_path"] == name]
            if (len(records) != 1 or records[0]["receipt_sha256"] != base.file_hash(receipt_path)
                    or records[0]["pending_sha256"] != source["pending_sha256"]
                    or records[0]["raw_sha256"] != receipt["raw_sha256"]
                    or records[0]["keys"] != receipt["keys"]):
                raise ValueError("Recovery record does not bind the sealed chunk")
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
            if not base._same(json.loads(path.read_text()), definition):
                raise ValueError("Output belongs to another original v1 experiment identity")
        else:
            base.write_once(path, definition)
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
                if release["claim_identity"] != base.stable_hash(previous) or release["reason"] != "stop_after_completed_chunk":
                    raise ValueError("Unbound ownership release")
                released = set(release["keys"])
                if not released <= set(previous["keys"]):
                    raise ValueError("Ownership release contains foreign keys")
            if pending & (set(previous["keys"]) - complete - released):
                raise ValueError("A pending key already has an owner; explicit stopped-claim recovery is required")
        if not pending:
            return None, [], None
        run = claims / args.claim_id
        run.mkdir(exist_ok=False)
        keys = [key for key in plan["selected"] if key in pending]
        owner = {"claim_id": args.claim_id, "owner": args.owner, "started_utc": base.now(),
                 **base.process_identity(), "keys": keys, "identity": plan["identity"],
                 "dataset_entries": {name: row["entry"] for name, row in plan["datasets"].items()},
                 "admission": admission, "batch_size": 1, "chunk_rows": args.chunk_rows,
                 "assignment_path": args.task_keys, "assignment_sha256": plan["assignment_sha256"],
                 "source_sha256": plan["definition"]["source_sha256"], "argv": sys.argv,
                 "validation_correction": plan["validation_correction"],
                 "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=plan["root"], text=True).strip()}
        base.write_once(run / "owner.json", owner)
        return run, keys, owner


def seal(plan, run, owner, number, pending_path, keys, eos, recovery_source=None):
    import fcntl
    checked = validate_rows(pending_path, plan, keys, base.stable_hash(owner), eos)
    raw = run / f"chunk_{number:05d}.jsonl"
    if raw.exists():
        raise ValueError("Refusing to replace a sealed chunk")
    if recovery_source is None:
        pending_path.rename(raw)
    else:
        # Recovery is append-only: the original pending evidence is retained.
        with pending_path.open("rb") as source, raw.open("xb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
    raw.chmod(0o444)
    path = run / f"chunk_{number:05d}.complete.json"
    receipt = {"status": "complete", "completed_utc": base.now(), "identity": plan["identity"],
               "claim_identity": base.stable_hash(owner), "keys": keys, "eos_token_ids": sorted(eos),
               "raw_path": str(raw.relative_to(plan["output"])), "raw_sha256": base.file_hash(raw),
               "owner_path": str((run / "owner.json").relative_to(plan["output"])),
               "owner_sha256": base.file_hash(run / "owner.json"), "validation": checked,
               "validation_correction": plan["validation_correction"]}
    if recovery_source is not None:
        receipt["recovery_source"] = recovery_source
    base.write_once(path, receipt)
    with (plan["output"] / "ownership.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = plan["output"] / "completed_keys.jsonl"
        recorded = {row["key"] for row in base.read_jsonl(ledger)} if ledger.exists() else set()
        if recorded & set(keys):
            raise ValueError("Completed ledger already contains a chunk key")
        with ledger.open("a", encoding="utf-8") as stream:
            for key in keys:
                stream.write(json.dumps({"key": key, "dataset": "hallusionbench", "identity": plan["identity"],
                    "receipt": str(path.relative_to(plan["output"])), "receipt_sha256": base.file_hash(path)}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"event": "sealed_chunk", "chunk": number, **checked,
                      "validation_correction_sha256": plan["validation_correction"]["sha256"]}), flush=True)
    return path, receipt


def stopped_owner_observation(owner):
    if owner["hostname"] != socket.gethostname():
        raise ValueError("Recovery must run on the original owner's host")
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    active = False
    observed_tick = None
    try:
        fields = Path(f'/proc/{owner["pid"]}/stat').read_text().rsplit(")", 1)[1].split()
        observed_tick = int(fields[19])
        active = current_boot == owner["boot_id"] and observed_tick == owner["start_tick"]
    except FileNotFoundError:
        pass
    observation = {"observed_utc": base.now(), "hostname": socket.gethostname(), "pid": owner["pid"],
                   "owner_start_tick": owner["start_tick"], "observed_start_tick": observed_tick,
                   "owner_boot_id": owner["boot_id"], "observed_boot_id": current_boot, "active": active}
    if active:
        raise ValueError("Original claim owner is still alive; no automatic takeover or recovery")
    return observation


def recover_claim(args, plan):
    if args.task_keys:
        raise ValueError("CPU recovery examines the original complete claim, not a task subset")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.recover_claim):
        raise ValueError("Invalid recovery claim identifier")
    run = plan["output"] / "claims" / args.recover_claim
    owner_path = run / "owner.json"
    owner = json.loads(owner_path.read_text())
    claim_identity = base.stable_hash(owner)
    if (owner["claim_id"] != args.recover_claim or owner["identity"] != plan["identity"]
            or owner["admission"]["host"] != plan["definition"]["source_host"]
            or not base._same(owner["source_sha256"], plan["definition"]["source_sha256"])
            or len(owner["keys"]) != len(set(owner["keys"])) or not set(owner["keys"]) <= set(plan["tasks"])):
        raise ValueError("Original stopped claim does not belong to the frozen v1 plan")
    observation = stopped_owner_observation(owner)
    recovery_path = run / "recovery.json"
    if recovery_path.exists():
        raise ValueError("This claim already has an append-only recovery record; verify instead")
    owner_sha = base.file_hash(owner_path)
    failed = run / "failed.json"
    failed_sha = base.file_hash(failed) if failed.exists() else None
    # The original completion reader now uses the corrected validator, but the
    # completed recovery record is checked only after all seals are registered.
    complete = _original_completed_keys(plan)
    owned_complete = set(owner["keys"]) & complete
    if owned_complete != set(owner["keys"][:len(owned_complete)]):
        raise ValueError("The original completed claim keys are not a generated prefix")
    records = []
    for pending in sorted(run.glob("chunk_*.pending.jsonl")):
        match = re.fullmatch(r"chunk_(\d{5})\.pending\.jsonl", pending.name)
        if match is None:
            raise ValueError("Unregistered pending chunk filename")
        rows = list(base.read_jsonl(pending))
        keys = [row["key"] for row in rows]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Empty or duplicate original pending chunk")
        next_keys = owner["keys"][len(owned_complete):len(owned_complete) + len(keys)]
        if keys != next_keys or set(keys) & complete:
            raise ValueError("Original pending rows overlap or skip the frozen claim's generated prefix")
        eos_values = {tuple(row.get("eos_token_ids", [])) for row in rows}
        if len(eos_values) != 1:
            raise ValueError("Original pending rows disagree about actual EOS identities")
        eos = set(next(iter(eos_values)))
        pending_sha = base.file_hash(pending)
        source = {"pending_path": str(pending.relative_to(plan["output"])), "pending_sha256": pending_sha,
                  "recovery_path": str(recovery_path.relative_to(plan["output"]))}
        receipt_path, receipt = seal(plan, run, owner, int(match[1]), pending, keys, eos, recovery_source=source)
        if base.file_hash(pending) != pending_sha or receipt["raw_sha256"] != pending_sha:
            raise ValueError("Recovery did not preserve the original pending bytes")
        records.append({**source, "raw_path": receipt["raw_path"], "raw_sha256": receipt["raw_sha256"],
                        "receipt_path": str(receipt_path.relative_to(plan["output"])),
                        "receipt_sha256": base.file_hash(receipt_path), "keys": keys, "rows": len(rows)})
        owned_complete.update(keys)
        complete.update(keys)
    remaining = [key for key in owner["keys"] if key not in complete]
    released_path = run / "released.json"
    if released_path.exists():
        release = json.loads(released_path.read_text())
        if (release["claim_identity"] != claim_identity or release["reason"] != "stop_after_completed_chunk"
                or release["keys"] != remaining):
            raise ValueError("Existing explicit ownership release differs from the remaining claim keys")
    else:
        base.write_once(released_path, {"claim_identity": claim_identity, "reason": "stop_after_completed_chunk",
            "released_utc": base.now(), "keys": remaining, "validation_correction": plan["validation_correction"],
            "scope": "explicit CPU sealing of existing trajectories; no model execution or retry"})
    recovery = {"schema": "kdm_hallusion_independent128_recovery_v2", "status": "recovered",
                "identity": plan["identity"], "claim_identity": claim_identity,
                "owner_path": str(owner_path.relative_to(plan["output"])), "owner_sha256": owner_sha,
                "validation_correction": plan["validation_correction"], "process_observation": observation,
                "sealed_pending": records, "completed_owned_rows": len(owned_complete), "released_keys": remaining,
                "released_path": str(released_path.relative_to(plan["output"])), "released_sha256": base.file_hash(released_path),
                "original_failed_path": str(failed.relative_to(plan["output"])) if failed.exists() else None,
                "original_failed_sha256": failed_sha, "recovered_utc": base.now(),
                "regenerated_rows": 0, "model_execution": False, "automatic_retry": False}
    base.write_once(recovery_path, recovery)
    if base.file_hash(owner_path) != owner_sha or (failed.exists() and base.file_hash(failed) != failed_sha):
        raise ValueError("Recovery changed original owner or failure evidence")
    verified = completed_keys(plan)
    if not owned_complete <= verified:
        raise ValueError("Recovered keys are missing their fully verified sealed evidence")
    return {"status": "CPU_claim_recovered", "claim_id": args.recover_claim, "model_identity": plan["identity"],
            "sealed_original_rows": sum(row["rows"] for row in records), "completed_owned_rows": len(owned_complete),
            "released_rows": len(remaining), "regenerated_rows": 0, "gpu_initialized": False,
            "recovery_path": str(recovery_path.relative_to(plan["root"])), "recovery_sha256": base.file_hash(recovery_path)}


def _install():
    # Only this new process is routed to the corrected validator and explicit
    # v2 provenance. Frozen v1 files/protocol/identity are not mutated.
    base.validate_rows = validate_rows
    base.completed_keys = completed_keys
    base.claim = claim
    base.seal = seal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--recover-claim", metavar="STOPPED_CLAIM_ID")
    parser.add_argument("--model", choices=base.MODELS, required=True)
    parser.add_argument("--methods", nargs="+", default=["direct"])
    parser.add_argument("--protocol", default=base.PROTOCOL)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--validation-correction", default=CORRECTION)
    parser.add_argument("--validation-correction-sha256", required=True)
    parser.add_argument("--output", default=base.BASE + "/runs")
    parser.add_argument("--registry", default="workflows/hallusion_blind/host_registry.json")
    parser.add_argument("--cards", default="")
    parser.add_argument("--claim-id", default="cpu_plan_check")
    parser.add_argument("--owner", default="/root")
    parser.add_argument("--task-keys")
    parser.add_argument("--chunk-rows", type=int, choices=(1, 8, 16), default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--k100-intern", action="store_true")
    args = parser.parse_args(argv)
    plan = load_plan(args)
    _install()
    if args.recover_claim:
        result = recover_claim(args, plan)
    elif args.check_plan:
        result = {"status": "CPU_plan_checked", "model_identity": plan["identity"],
                  "expected_model_rows": len(plan["tasks"]), "selected_rows": len(plan["selected"]),
                  "validation_correction": plan["validation_correction"], "gpu_initialized": False}
    elif args.verify:
        complete = completed_keys(plan)
        expected = set(plan["selected"])
        result = {"status": "complete" if expected <= complete else "incomplete", "model_identity": plan["identity"],
                  "expected_rows": len(expected), "completed_rows": len(expected & complete),
                  "validation_correction": plan["validation_correction"]}
    else:
        if not args.cards or args.claim_id == "cpu_plan_check":
            parser.error("--execute requires explicit --cards and a distinct --claim-id")
        result = base.execute(args, plan)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), flush=True)
    return 2 if args.verify and result["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
