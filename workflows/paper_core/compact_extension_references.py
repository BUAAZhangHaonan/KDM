#!/usr/bin/env python3
"""Copy accepted extension reference fields once, retaining exact source rows."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import file_hash, read_jsonl, within

SOURCE = "outputs/supplemental/remaining4/reference_full_20261001_1600/checkpoints/v2_reviewed_role_reference"
EXPECTED_SHA = "d3feac8fef2ade5967c0ff17762e15a17aac92e244f980c01d6ed360ff8d36a0"
MODELS = {"internvl35_8b", "onevision", "phi35", "qwen3vl"}
FIELDS = ("model", "dataset", "sample_id", "split", "target_class", "gold_rank",
          "attempt_count", "correct_count", "reference_G", "reference_complete",
          "historical_accepted_reference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = ROOT / SOURCE
    authority_path = source / "receipt.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    if (authority["passed"] is not True or authority["reference_complete"] != 19392
            or authority["reference_pending"] != 0):
        raise ValueError("The extension reference authority is not fully closed")
    samples = {s["id"]: s for s in read_jsonl(ROOT / "data/current/all.jsonl")
               if s["dataset"] == "food101"}
    output = within(ROOT, args.output)
    output.relative_to(ROOT / "outputs/paper_core_20261002_dev_viz")
    output.mkdir(parents=True, exist_ok=False)
    raw = source / "reference_G.jsonl"
    dest = output / "reference_rows.jsonl"
    digest, seen, counts = hashlib.sha256(), set(), Counter()
    with raw.open("rb") as stream, dest.open("x", encoding="utf-8") as target:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            r = json.loads(line)
            s = samples[r["sample_id"]]
            key = (r["model"], r["sample_id"])
            if (r["model"] not in MODELS or key in seen or r["dataset"] != "food101"
                    or r["split"] != s["split"] or r["target_class"] != s["class"]
                    or type(r["reference_G"]) is not bool or r["reference_complete"] is not True
                    or r["attempt_count"] != 10 or not 0 <= r["correct_count"] <= 10
                    or r["reference_G"] != (r["gold_rank"] > 1 and r["correct_count"] == 0)):
                raise ValueError("An accepted extension reference key or registered definition differs")
            seen.add(key)
            counts[(r["model"], r["split"])] += 1
            value = {k: r[k] for k in FIELDS}
            value.update(source_path=str(raw), source_sha256=EXPECTED_SHA,
                         source_line=line_number, source_line_sha256=hashlib.sha256(line).hexdigest())
            target.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    if (digest.hexdigest() != EXPECTED_SHA or len(seen) != 19392
            or set(counts) != {(m, split) for m in MODELS for split in ("dev", "eval")}
            or set(counts.values()) != {2424}):
        raise ValueError("The accepted reference SHA or complete per-model split coverage differs")
    receipt = {"schema": "kdm_exact_compact_reference_copy_v1", "passed": True,
               "rows": len(seen), "reference_pending": 0, "source_sha256": EXPECTED_SHA,
               "source_path": str(raw), "source_receipt_path": str(authority_path),
               "source_receipt_sha256": file_hash(authority_path),
               "outputs": {dest.name: file_hash(dest)}, "fields": list(FIELDS),
               "registered_reference_definition_changed": False, "semantic_judgments_added": 0,
               "GPU_initialized": False, "runner_sha256": file_hash(Path(__file__)),
               "completed_utc": datetime.now(timezone.utc).isoformat()}
    with (output / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
