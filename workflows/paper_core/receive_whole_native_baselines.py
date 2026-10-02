#!/usr/bin/env python3
"""Validate explicit completed native sources and seal CPU scoring views."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from kdm.io import file_hash, within
from workflows.paper_core.snapshot_native_baselines import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--staging", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = within(ROOT, args.manifest)
    staging = within(ROOT, args.staging)
    output = within(ROOT, args.output)
    output.relative_to(ROOT / "outputs/paper_core_20261002_dev_viz")
    inventory = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if inventory["schema"] != "kdm_native_baseline_immutable_sources_v1":
        raise ValueError("The explicit immutable native source inventory is required")
    output.mkdir(parents=True, exist_ok=False)
    seen, parts, proofs = set(), [], []
    for source in inventory["sources"]:
        key = source["model"], source["method"]
        if (key in seen or source["completed"] != 2424
                or source["start"] != 0 or source["stop"] != 2424):
            raise ValueError("A whole native condition is repeated or incomplete")
        seen.add(key)
        directory = staging / (source["model"] + "_" + source["method"])
        raw = directory / "raw" / Path(source["raw_path"]).name
        sidecar = raw.with_suffix(".identity.json")
        complete = directory / Path(source["receipt_path"]).name
        for name, path in (("raw", raw), ("identity", sidecar), ("receipt", complete)):
            if file_hash(path) != source[name + "_sha256"]:
                raise ValueError("Received immutable native bytes differ: " + str(path))
        receipt = json.loads(complete.read_text(encoding="utf-8"))
        if (receipt["passed"] is not True or receipt["completed"] != 2424
                or receipt["expected_full_part"] != 2424
                or receipt["raw_sha256"] != source["raw_sha256"]
                or receipt["ledger_identity"] != source["ledger_identity"]
                or receipt["condition"] != source["condition"]):
            raise ValueError("The actual completion receipt differs from its source inventory")
        view = output / (source["model"] + "_" + source["method"])
        snapshot(directory, view, 2424)
        view_receipt = json.loads((view / "receipt.json").read_text(encoding="utf-8"))
        if len(view_receipt["parts"]) != 1 or view_receipt["rows"] != 2424:
            raise ValueError("The isolated native whole-condition scoring view is incomplete")
        part = view_receipt["parts"][0]
        part.update(source_hostname=source["host"], source_raw_path=source["raw_path"],
                    source_live_at_snapshot=False, source_part_complete_claimed=True,
                    original_completion_receipt=source["receipt_path"],
                    original_completion_receipt_sha256=source["receipt_sha256"],
                    original_source_inventory_entry=source,
                    source_complete_at_utc=source["completed_at_utc"])
        parts.append(part)
        proofs.append({"path": str((view / "receipt.json").relative_to(ROOT)),
                       "sha256": file_hash(view / "receipt.json")})
    result = {"schema": "kdm_sealed_native_baseline_snapshot_v1", "passed": True,
              "parts": parts, "rows": 2424 * len(parts), "unique_keys": 2424 * len(parts),
              "whole_conditions_complete_claimed": True,
              "actual_complete_source_conditions": len(parts),
              "explicit_source_manifest": str(manifest_path.relative_to(ROOT)),
              "explicit_source_manifest_sha256": file_hash(manifest_path),
              "isolated_validation_receipts": proofs, "GPU_initialized": False,
              "snapshot_utc": datetime.now(timezone.utc).isoformat(),
              "runner_sha256": file_hash(Path(__file__)),
              "actual_command": [sys.executable, *sys.argv]}
    with (output / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("passed", "rows", "unique_keys",
                                           "actual_complete_source_conditions")}))


if __name__ == "__main__":
    main()
