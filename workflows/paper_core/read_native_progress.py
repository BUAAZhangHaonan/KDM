#!/usr/bin/env python3
"""One bounded CPU snapshot of complete native rows; never hash an active ledger."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    results = []
    for run in args.runs:
        directory = args.root / "outputs/paper_core_20260930" / run
        for path in sorted((directory / "raw").glob("*_native_vcd.jsonl")):
            rows = []
            with path.open("rb") as handle:
                for line in handle:
                    if line.endswith(b"\n"):
                        rows.append(json.loads(line))
            model = path.name.removesuffix("_native_vcd.jsonl")
            recent = [float(row["wall_s"]) for row in rows[-128:]]
            count = len(rows)
            gate_path = directory / f"{model}.gate_8.json"
            gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
            errors = path.with_suffix(".errors.jsonl")
            results.append({
                "model": model, "run": run, "raw": str(path),
                "completed_rows": count,
                "unique_keys": len({row["key"] for row in rows}),
                "expected_rows": 2424, "remaining_rows": 2424-count,
                "all_native_plain": all(row["method"] == "vcd" and
                    row["kind"] == "native_unguided" and not row["guided"] and
                    not row["reference_guided"] and row["marker"] == "NONE" and
                    row["reference_marker"] == "NONE" for row in rows),
                "recent_n": len(recent),
                "recent_mean_wall_s": statistics.mean(recent) if recent else None,
                "recent_median_wall_s": statistics.median(recent) if recent else None,
                "linear_remaining_s": statistics.mean(recent)*(2424-count) if recent else None,
                "gate_passed": gate.get("passed") if gate else None,
                "complete_receipt": (directory / f"{model}.complete.json").exists(),
                "errors_file": str(errors) if errors.exists() else None,
            })
    processes = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            values = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        command = [value.decode(errors="replace") for value in values if value]
        if any(value.endswith(("paper_core/native.py", "paper_core/native_a100.py")) for value in command):
            processes.append({"pid": int(entry.name), "argv": command})
    print(json.dumps({"schema": "kdm_native_cpu_snapshot_v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root), "rows": results, "processes": processes},
        ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
