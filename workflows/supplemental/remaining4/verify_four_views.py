#!/usr/bin/env python3
"""Verify finite supplemental exports using the accepted core CPU checks."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import atomic_json, file_hash, read_jsonl
from workflows.paper_core import verify_attribution_exports as accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, action="append", required=True)
    parser.add_argument("--expected-per-model", type=int, choices=(8, 101), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    natural_counts = Counter()
    seen = set()
    budgets = {}
    natural_sources = []
    for folder in args.directory:
        complete = json.loads((folder / "complete.json").read_text())
        if not complete["gpu_phase_complete_for_limit"]:
            raise ValueError("GPU source phase has no successful completion")
        model = complete["model"]
        rows = accepted.ledger(folder / "representative.events.jsonl", complete["outputs"]["representative"])
        for line, row in enumerate(rows, 1):
            sample_id = row["sample_id"]
            key = (model, sample_id)
            if key in seen or row["model"] != model:
                raise ValueError("Duplicate or mismatched supplemental representative key")
            seen.add(key)
            accepted.measure(row["measured"], model, sample_id, "representative101",
                             folder / "representative.events.jsonl", line)
            counts[model] += 1
        natural = accepted.ledger(folder / "natural_reference.events.jsonl", complete["outputs"]["natural_reference"])
        if {row["sample_id"] for row in natural} != {row["sample_id"] for row in rows}:
            raise ValueError("Natural reference and representative sources differ")
        for row in natural:
            result = row["noise_generated_r"]
            if (len(result["tokens"]) > 32 or row["config"]["method"] != "direct"
                    or row["config"]["temperature"] != 0.0 or row["config"]["top_p"] != 1.0
                    or row["model"] != model):
                raise ValueError("Natural reference generation configuration differs")
            natural_counts[model] += 1
        natural_sources.append({"directory": str(folder), "rows": len(natural),
                                "sha256": file_hash(folder / "natural_reference.events.jsonl")})
        budget = json.loads((folder / "budget.json").read_text())
        if budget["model"] != model or not 0 <= budget["used_gpu_seconds"] <= 2880:
            raise ValueError("Finite per-model budget exceeded")
        budgets[model] = max(budgets.get(model, 0), budget["used_gpu_seconds"])
    if counts != natural_counts or any(value != args.expected_per_model for value in counts.values()):
        raise ValueError("Representative and natural-reference key coverage is incomplete")
    for name, records in (("four_view_pairs.csv", accepted.PAIRS),):
        if records:
            with (args.out / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    with (args.out / "four_view_events.jsonl").open("w", encoding="utf-8") as stream:
        for row in accepted.EVENTS:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    receipt = {"schema": "kdm_remaining4_finite_four_view_verification_v1", "status": "passed",
               "representative_counts": dict(counts), "natural_counts": dict(natural_counts),
               "sources": accepted.SOURCES, "natural_sources": natural_sources,
               "positions": len(accepted.EVENTS), "candidate_pairs": len(accepted.PAIRS),
               "maximum_float64_residuals": dict(accepted.RESIDUAL), "cumulative_gpu_seconds": budgets,
               "verified_ties": accepted.ARGMAX_TIES, "new_GPU_forwards": 0,
               "source_verifier_sha256": file_hash(ROOT / "workflows/paper_core/verify_attribution_exports.py"),
               "semantic_labels_inferred_from_marker_proxy": False,
               "diagnostic_and_case_coverage": "separate actual-source finite phases"}
    atomic_json(args.out / "verification.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "representative_counts", "positions",
                                                  "candidate_pairs", "maximum_float64_residuals")}))


if __name__ == "__main__":
    main()
