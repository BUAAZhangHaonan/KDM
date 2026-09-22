#!/usr/bin/env python3
"""Persistent all-model probe dispatch with fixed devices and no retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kdm.io import atomic_json, file_hash
from kdm.protocol import validate_freeze, validate_native_runtime_files
from kdm.execution import local_host, read_registry
from runner import load_inputs, AMENDMENT

LANES = {
    "remote_gpu1": {"host": "6403", "cards": [1], "models": ["qwen3vl", "glm46v"]},
    "gpu0": {"host": "4028", "cards": [0], "models": ["qwen35_9b"]},
    "gpu1": {"host": "4028", "cards": [1], "models": ["qwen35_4b", "qwen25vl"]},
    "pair45": {"host": "4028", "cards": [4, 5], "models": ["gemma3_12b", "llava15_13b", "internvl35_8b"]},
    "gpu4_after_pairs": {"host": "4028", "cards": [4], "models": ["gemma3_4b", "llava15_7b", "onevision", "llava16_mistral"]},
    "gpu5_after_pairs": {"host": "4028", "cards": [5], "models": ["minicpm26", "minicpm45", "phi35", "llava16_vicuna"]},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def plan():
    samples, names = load_inputs()
    panel = json.loads((ROOT / "configs/kdm/models.json").read_text())
    keys = [key for lane in LANES.values() for key in lane["models"]]
    if len(keys) != 16 or len(set(keys)) != 16 or set(keys) != {row["key"] for row in panel}:
        raise ValueError("Exactly the original 16 models required")
    registry = read_registry(ROOT)
    specs = {}
    freeze = validate_freeze(ROOT)
    for lane in LANES.values():
        for key in lane["models"]:
            specpath = ROOT / f"configs/runtime/{key}.json"
            spec = json.loads(specpath.read_text())
            if registry["model_hosts"][key] != lane["host"] or spec["gpu_count"] != len(lane["cards"]):
                raise ValueError("Host or GPU allocation differs from approved native spec")
            if file_hash(specpath) != freeze["files"][f"configs/runtime/{key}.json"]:
                raise ValueError("Runtime identity differs from original freeze")
            validate_native_runtime_files(ROOT, spec, key)
            specs[key] = spec
    return {"lanes": LANES, "models": 16, "samples_per_model": len(samples),
            "independent_per_model": 91670, "closed_per_model": 4848,
            "expected_independent_total": 1466720, "expected_closed_total": 77568,
            "expected_total_records": 1544288,
            "independent_config": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 32, "repeats": 10},
            "closed_classes": len(names), "splits": ["dev", "eval"],
            "closed_scope": "food101 only; 101 label-sequence scores per image, not one forward pass",
            "does_not_select_models": True, "no_automatic_retry": True,
            "failure_policy": "preserve failed model; allow independent model work to continue"}, specs


def execute(host, name):
    local_host(ROOT, host)
    schedule, specs = plan()
    if "/" in name or not name:
        raise ValueError("Run name must be a simple directory name")
    run = ROOT / "outputs/records/probes_v2" / name
    run.mkdir(parents=True, exist_ok=False)
    lockpath = ROOT / "outputs/locks/all_model_probes_v2.lock"
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = {key for lane in LANES.values() if lane["host"] == host for key in lane["models"]}
        report = {"host": host, "scheduler_pid": os.getpid(), "started_utc": now(),
                  "plan": schedule, "complete": False, "host_complete": False,
                  "amendment_path": AMENDMENT, "amendment_sha256": file_hash(ROOT / AMENDMENT),
                  "freeze_receipt_sha256": file_hash(ROOT / "outputs/records/preregistration_freeze.json"),
                  "workflow_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                  "jobs": {key: {"status": "not_started"} for key in sorted(active)}}
        mutex = threading.Lock()

        def update(key, **fields):
            with mutex:
                report["jobs"][key].update(fields)
                atomic_json(run / "status.json", report)

        def lane(lane_name):
            declaration = LANES[lane_name]
            cards = declaration["cards"]
            for key in declaration["models"]:
                command = ["bash", str(ROOT / "scripts/worker.sh"), str(ROOT), ",".join(map(str, cards)),
                           specs[key]["environment_python"], "-u",
                           str(ROOT / "workflows/all_model_probes_v2/runner.py"), "--model", key,
                           "--run-dir", str(run.relative_to(ROOT))]
                try:
                    with (run / f"{key}.log").open("x") as log:
                        update(key, status="starting", lane=lane_name, physical_gpus=cards,
                               command=command, started_utc=now())
                        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                        update(key, status="running", worker_pid=process.pid)
                        code = process.wait()
                        update(key, process_exit_code=code)
                        if code:
                            raise RuntimeError("Model execution failed; no retry: " + str(code))
                        receipt = json.loads((run / f"{key}_complete.json").read_text())
                        if receipt.get("complete") is not True or receipt.get("independent") != 91670 or receipt.get("closed") != 4848:
                            raise ValueError("Missing complete model coverage")
                        update(key, status="complete", finished_utc=now())
                except Exception as exc:
                    update(key, status="failed", error=type(exc).__name__ + ": " + str(exc), finished_utc=now())

        def pairs_then_singles():
            lane("pair45")
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lane, ["gpu4_after_pairs", "gpu5_after_pairs"]))

        atomic_json(run / "status.json", report)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = ([pool.submit(lane, "gpu0"), pool.submit(lane, "gpu1"), pool.submit(pairs_then_singles)]
                       if host == "4028" else [pool.submit(lane, "remote_gpu1")])
            for future in futures:
                future.result()
        report["finished_utc"] = now()
        report["host_complete"] = all(job["status"] == "complete" for job in report["jobs"].values())
        report["complete"] = False
        report["completion_scope"] = "host subset only; full panel requires both hosts and semantic annotations"
        atomic_json(run / "status.json", report)
        return 0 if report["host_complete"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", choices=["4028", "6403"])
    parser.add_argument("--name")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        if not args.host or not args.name:
            parser.error("--execute requires host and fresh run name")
        raise SystemExit(execute(args.host, args.name))
    print(json.dumps(plan()[0], indent=2))
