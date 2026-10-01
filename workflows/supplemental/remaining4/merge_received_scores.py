"""Merge explicitly accepted immutable score cohorts without changing records."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import file_hash, within
from workflows.supplemental.remaining11.score import rows
from workflows.supplemental.remaining4.score_native import save_json, save_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = within(ROOT, args.output)
    output.relative_to(ROOT / "outputs/supplemental/remaining4")
    records, seen, inputs, counts = [], set(), [], Counter()
    for relative in args.score_dir:
        source = within(ROOT, relative)
        receipt_path, score_path = source / "receipt.json", source / "score_rows.jsonl.gz"
        receipt = json.loads(receipt_path.read_text())
        score_sha = file_hash(score_path)
        if not receipt.get("passed") or receipt["outputs"].get(score_path.name) != score_sha:
            raise ValueError("An immutable score cohort differs from its passed receipt")
        count = 0
        for _line, record, _sha in rows(score_path):
            if record["key"] in seen or record["dataset"] not in {"food101", "vizwiz"}:
                raise ValueError("The accepted cohorts repeat a generation key or change dataset scope")
            seen.add(record["key"])
            records.append(record)
            counts[record["dataset"]] += 1
            count += 1
        if count != receipt["rows"]:
            raise ValueError("Actual score objects differ from their registered cohort count")
        inputs.append({"path": str(score_path.relative_to(ROOT)), "sha256": score_sha,
                       "receipt_sha256": file_hash(receipt_path), "rows": count})
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "score_rows.jsonl.gz", records)
    receipt = {"schema": "kdm_explicit_immutable_score_union_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "rows": len(records),
        "unique_keys": len(seen), "counts": dict(counts), "source_score_inputs": inputs,
        "original_objects_identical": True, "new_scientific_judgments": 0,
        "reference_or_official_score_changed": False, "raw_reopened": False,
        "GPU_initialized": False, "runner_sha256": file_hash(Path(__file__)),
        "outputs": {p.name: file_hash(p) for p in output.iterdir()}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("passed", "rows", "unique_keys", "counts")}))


if __name__ == "__main__":
    main()
