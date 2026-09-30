#!/usr/bin/env python3
"""Check the real native task plans and disjoint sample shards on CPU."""
from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from workflows.supplemental.remaining4.native import (
    CONFIG, MODELS, METHODS, file_hash, load_plan, task_id, within, write_new_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    path = within(ROOT, args.manifest)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {(model, method) for model in MODELS for method in METHODS}
    if {(row["model"], row["method"]) for row in manifest["conditions"]} != expected:
        raise ValueError("Native manifest does not cover the exact eight registered conditions")
    destination = within(ROOT, args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    checks, planned_keys = [], set()
    for row in manifest["conditions"]:
        if row["expected"] != 2424 or row["expected"] != row["reused"] + row["missing"]:
            raise ValueError("Native condition denominator or gap accounting differs")
        if file_hash(within(ROOT, row["missing_keys_path"])) != row["missing_keys_sha256"]:
            raise ValueError("Native explicit gap file changed")
        if not row["missing"]:
            checks.append({"model": row["model"], "method": row["method"], "missing": 0})
            continue
        options = dict(model=row["model"], method=row["method"], missing_keys=row["missing_keys_path"],
                       key_start=0, key_stop=None, shard=0, n_shards=1)
        plan = load_plan(Namespace(**options))
        write_new_json(destination / f"{row['model']}_{row['method']}_plan.json", plan["summary"])
        keys = {task_id(row["model"], task) for task in plan["tasks"]}
        if len(keys) != row["missing"] or planned_keys.intersection(keys):
            raise ValueError("Native model/method keys are missing or repeated across conditions")
        planned_keys.update(keys)
        shards = []
        if row["missing"] >= 8:
            for index in range(2):
                shard = load_plan(Namespace(**{**options, "shard": index, "n_shards": 2}))
                shards.append(shard["selected_keys"])
                write_new_json(destination / f"{row['model']}_{row['method']}_shard{index}_of2_plan.json", shard["summary"])
            if shards[0].intersection(shards[1]) or set.union(*shards) != keys:
                raise ValueError("Native sample shards are not disjoint or omit actual keys")
        if row["missing"] == 2424:
            counts = plan["summary"]["class_counts"]
            if len(counts) != 101 or set(counts.values()) != {24}:
                raise ValueError("Native complete condition violates the 101 by 24 quota")
        checks.append({"model": row["model"], "method": row["method"], "missing": len(keys),
                       "shard_counts": [len(shard) for shard in shards], "shards_disjoint": True,
                       "shards_union_complete": True, "registered_condition_rows": 2424,
                       "main_marker": "NONE", "reference_marker": "NONE", "guided": False,
                       "reference_guided": False, "gpu_count": plan["spec"]["gpu_count"],
                       "dtype": plan["spec"]["dtype"], "registered_versions": plan["spec"]["versions"]})
    if len(planned_keys) != manifest["missing_rows"] or manifest["planned_rows"] != 19392:
        raise ValueError("Native manifest total differs from actual expanded keys")
    import torch

    if torch.cuda.is_initialized():
        raise ValueError("CPU task-plan validation initialized CUDA")
    receipt = {"schema": "kdm_remaining4_native_cpu_plan_check_v1", "passed": True,
               "created_utc": datetime.now(timezone.utc).isoformat(), "manifest_path": str(path.relative_to(ROOT)),
               "manifest_sha256": file_hash(path), "config_sha256": file_hash(ROOT / CONFIG),
               "conditions": checks, "planned_rows": 19392, "actual_missing_keys": len(planned_keys),
               "reused_rows": manifest["reused_rows"], "cuda_initialized": False,
               "gpu_runtime_admission": "not_run", "real_operator_audit": "not_run",
               "limitations": "CPU original-source, registered-task and shard checks; operator audit runs during actual generation",
               "verifier_sha256": file_hash(Path(__file__))}
    write_new_json(destination / "verification.json", receipt)
    print(json.dumps({"passed": True, "conditions": len(checks), "actual_missing_keys": len(planned_keys),
                      "cuda_initialized": False, "verification": str((destination / "verification.json").relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
