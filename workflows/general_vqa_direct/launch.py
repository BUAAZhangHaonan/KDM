"""Launch explicitly assigned registered workers, preserving detached run logs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
JOBS = {
    "qwen25vl": ("4028", "0"), "minicpm26": ("4028", "1"),
    "qwen35_4b": ("4028", "4"), "gemma3_4b": ("4028", "5"),
    "llava16_mistral": ("4029", "1"), "phi35": ("4029", "2"),
    "onevision": ("6403", "0"), "qwen3vl": ("6403", "1"),
    "internvl35_8b": ("k100", "0"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=JOBS, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--host", choices=("4028", "4029", "6403", "k100"))
    parser.add_argument("--cards", help="Explicit allowed physical GPU assignment for a transferred shard")
    parser.add_argument("--sample-ids", help="Project-relative, non-overlapping frozen sample assignment")
    parser.add_argument("--registry", default="workflows/general_vqa_direct/host_registry.json")
    parser.add_argument("--output", default="outputs/general_vqa_direct/run_20261004")
    args = parser.parse_args()
    if bool(args.host) != bool(args.cards) or (args.host and len(args.models) != 1):
        raise ValueError("A transferred shard needs one model and explicit host plus physical cards")
    if not args.phase.replace("_", "").isalnum():
        raise ValueError("Invalid phase")
    reg_path = ROOT / args.registry
    reg_path.resolve().relative_to(ROOT.resolve())
    registry = json.loads(reg_path.read_text())
    logs = ROOT / args.output / "launches"
    logs.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        host, cards = JOBS[model]
        if args.host:
            host, cards = args.host, args.cards
        declared = registry["hosts"][host]
        if declared["hostname"] != socket.gethostname() or Path(declared["root"]) != ROOT:
            raise ValueError("Model was not assigned to this host/project")
        original = json.loads((ROOT / f"configs/runtime/{model}.json").read_text())
        python = registry.get("runtime_overrides", {}).get(host, {}).get(model, {}).get(
            "environment_python", original["environment_python"])
        claim_id = f"{model}_{host}_{args.phase}"
        command = ["bash", str(ROOT / "workflows/supplemental/remaining4/worker_registered.sh"),
                   str(ROOT), cards, python, str(reg_path), "workflows/general_vqa_direct/generate.py",
                   "--execute", "--model", model,
                   "--protocol", "data/general_vqa_direct_20261004/frozen/protocol.json",
                   "--datasets", *args.datasets, "--output", args.output,
                   "--registry", args.registry, "--cards", cards,
                   "--claim-id", claim_id, "--owner", "/root", "--chunk-rows", "64"]
        if model == "internvl35_8b" and host == "k100":
            command.append("--k100-intern")
        if args.sample_ids:
            command.extend(["--sample-ids", args.sample_ids])
        log_path = logs / (claim_id + ".log")
        receipt_path = logs / (claim_id + ".launch.json")
        if log_path.exists() or receipt_path.exists():
            raise FileExistsError("Launch identity already exists; inspect it instead of repeating")
        with log_path.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {"model": model, "host": host, "physical_gpus": cards, "pid": process.pid,
                   "started_utc": datetime.now(timezone.utc).isoformat(), "command": command,
                   "log": str(log_path.relative_to(ROOT)), "claim_id": claim_id,
                   "status": "dispatched_not_yet_accepted"}
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
