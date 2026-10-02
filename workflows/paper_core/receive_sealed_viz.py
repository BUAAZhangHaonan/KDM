#!/usr/bin/env python3
"""Receive only the explicitly listed, immutable 512-row VizWiz conditions.

This local CPU transfer entry reads completion metadata, copies the three
original files for each named condition, and validates their source bindings.
It creates no labels and never opens a source file without a passed receipt.
The existing score_dev_viz entry performs the registered sample/config audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--host", choices=("6403", "RTX_Pro_6000"), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--selected-configs", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--central-dir", required=True)
    parser.add_argument("--executor-agent", required=True)
    parser.add_argument("--executor-model", required=True)
    parser.add_argument("--executor-effort", required=True)
    parser.add_argument("--executor-call-id", default="")
    args = parser.parse_args()
    source_root = PurePosixPath(args.source_root)
    if not source_root.is_absolute() or ".." in source_root.parts:
        raise ValueError("The named source root must be an absolute project path")
    if not re.fullmatch(r"/home/g203-4028/projects/knowledge-deficit-mitigation/outputs/paper_core_20261002_dev_viz/[A-Za-z0-9_/-]+", args.central_dir):
        raise ValueError("The destination must be within the independent core output directory")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if len(snapshot["models"]) != 1:
        raise ValueError("Receive one model at a time")
    model_snapshot = snapshot["models"][0]
    model = model_snapshot["model"]
    if model not in {"qwen25vl", "qwen35_4b", "minicpm26", "llava16_mistral", "gemma3_4b"}:
        raise ValueError("The snapshot is outside the registered core five models")
    selected_sha = sha256(args.selected_configs)
    args.local_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = args.local_dir / model / "raw"
    raw_dir.mkdir(parents=True)
    named = []
    keys = set()
    for condition in model_snapshot["completed"]:
        if not condition["passed"] or condition["rows"] != 512 or condition["required"] != 512:
            raise ValueError("A named source condition is not a complete 512-row run")
        raw_source = source_root / condition["raw"]
        if (PurePosixPath(condition["raw"]).is_absolute() or ".." in raw_source.parts
                or raw_source.suffix != ".jsonl" or raw_source.parts[-3:-1] != (model, "raw")):
            raise ValueError("The raw source does not match the named model")
        stem = str(raw_source)[:-len(".jsonl")]
        receipt_source = stem + ".complete.json"
        identity_source = stem + ".identity.json"
        if receipt_source != condition["receipt_path"] or condition["condition"] in keys:
            raise ValueError("The source receipt path or condition identity differs")
        keys.add(condition["condition"])
        named.append({"raw": str(raw_source), "receipt": receipt_source, "identity": identity_source, "metadata": condition})
    if len(named) != model_snapshot["complete_conditions"] or sum(x["metadata"]["rows"] for x in named) != model_snapshot["complete_rows"]:
        raise ValueError("Snapshot totals differ from its named conditions")
    if not named:
        raise ValueError("No completed condition was supplied")
    command = ["scp"] + [f"{args.host}:{item[k]}" for item in named for k in ("raw", "identity", "receipt")] + [str(raw_dir)]
    started = datetime.now(timezone.utc).isoformat()
    run(command)
    proofs = []
    for item in named:
        local = {k: raw_dir / PurePosixPath(item[k]).name for k in ("raw", "identity", "receipt")}
        receipt = json.loads(local["receipt"].read_text(encoding="utf-8"))
        identity = json.loads(local["identity"].read_text(encoding="utf-8"))
        definition = identity["definition"]
        raw_sha = sha256(local["raw"])
        if raw_sha != item["metadata"]["raw_sha256"] or raw_sha != receipt["raw_sha256"]:
            raise ValueError("Copied raw SHA differs from the source completion evidence")
        if not receipt["passed"] or receipt["rows"] != 512 or receipt["required"] != 512:
            raise ValueError("Copied receipt is not complete")
        core = {key: value for key, value in definition.items() if key not in {"shard", "n_shards", "base_config"}}
        definition_sha = hashlib.sha256(json.dumps(definition, sort_keys=True, ensure_ascii=False,
                                                  separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        core_sha = hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False,
                                            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if (identity["identity"] != definition_sha or receipt["identity"] != core_sha
                or receipt["identity"] != item["metadata"]["identity"]):
            raise ValueError("Copied identity differs from the source completion evidence")
        if definition["stage"] != "viz512" or definition["model"] != model or definition["selected_configs_sha256"] != selected_sha:
            raise ValueError("Copied source model, stage or original selected-config SHA differs")
        rows = [json.loads(line) for line in local["raw"].read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != 512 or len({row["key"] for row in rows}) != 512 or any(row["model"] != model for row in rows):
            raise ValueError("Copied source has missing or duplicate model/sample keys")
        proofs.append({"model": model, "condition": item["metadata"]["condition"], "rows": 512,
                       "source_host": args.host, "original_sources": {k: item[k] for k in ("raw", "identity", "receipt")},
                       "copied_sha256": {k: sha256(local[k]) for k in local}, "core_identity": core_sha,
                       "ledger_identity": identity["identity"],
                       "selected_configs_sha256": selected_sha,
                       "checkpoint": definition["runtime_spec"]["hf_model_id"], "dtype": definition["runtime_spec"]["dtype"]})
    central_raw = args.central_dir + "/" + model + "/raw"
    exclusive_command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "4028-root", "mkdir " + args.central_dir]
    run(exclusive_command)
    ssh_command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "4028-root", "mkdir -p " + central_raw]
    run(ssh_command)
    files = [str(raw_dir / PurePosixPath(item[k]).name) for item in named for k in ("raw", "identity", "receipt")]
    run(["scp", *files, "4028-root:" + central_raw + "/"])
    result = {"passed": True, "model": model, "complete_conditions": len(proofs), "complete_rows": 512 * len(proofs),
              "snapshot": str(args.snapshot), "snapshot_sha256": sha256(args.snapshot),
              "selected_configs": str(args.selected_configs), "selected_configs_sha256": selected_sha,
              "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "sources": proofs,
              "actual_transfer_commands": [command, exclusive_command, ssh_command, ["scp", *files, "4028-root:" + central_raw + "/"]],
              "actual_executor": {"agent": args.executor_agent, "model": args.executor_model,
                                  "effort": args.executor_effort, "call_id": args.executor_call_id},
              "source_raw_opened_without_receipt": 0, "GPU_queries": 0, "semantic_labels_created": 0}
    proof_path = args.local_dir / "TRANSFER_RECEIPT.json"
    proof_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run(["scp", str(proof_path), "4028-root:" + args.central_dir + "/"])
    print(json.dumps({key: result[key] for key in ("passed", "model", "complete_conditions", "complete_rows", "selected_configs_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
