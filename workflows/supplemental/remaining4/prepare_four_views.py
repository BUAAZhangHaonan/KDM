#!/usr/bin/env python3
"""Bind the registered representative inputs to immutable native responses."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import atomic_json, file_hash, read_jsonl, stable_seed, within

MODELS = ("internvl35_8b", "onevision", "phi35", "qwen3vl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--representative-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    destination = within(ROOT, args.out)
    destination.mkdir(parents=True, exist_ok=False)
    with args.representative_list.open(encoding="utf-8-sig", newline="") as stream:
        panel = list(csv.DictReader(stream))
    ids = {row["sample_id"] for row in panel}
    if len(panel) != 101 or len(ids) != 101 or len({row["target_class"] for row in panel}) != 101:
        raise ValueError("Representative panel must preserve the original 101 class inputs")
    samples = {row["id"]: row for row in read_jsonl(ROOT / "data/current/all.jsonl")}
    for row in panel:
        sample = samples[row["sample_id"]]
        if (sample["dataset"], sample["split"], sample["class"]) != ("food101", "eval", row["target_class"]):
            raise ValueError("Representative panel differs from the registered input")
    with gzip.open(args.scores, "rt", encoding="utf-8") as stream:
        selected = [row for row in map(json.loads, stream) if row["model"] in MODELS
                    and row["sample_id"] in ids and row["method"] == "vcd"
                    and row["kind"] == "native_unguided"]
    counts = Counter(row["model"] for row in selected)
    if counts != Counter({model: 101 for model in MODELS}):
        raise ValueError("Complete native source replies are required for all four panels")
    sources = {}
    for row in selected:
        path = within(ROOT, row["source_path"])
        if path not in sources:
            sources[path] = {record["key"]: record for record in read_jsonl(path)}
    records = {}
    for row in selected:
        raw = sources[within(ROOT, row["source_path"])][row["original_key"]]
        sample = samples[row["sample_id"]]
        if (raw["identity"] != row["source_identity"] or raw["text"] != row["answer"]
                or raw["sample"] != sample or raw["model"] != row["model"]
                or raw["seed"] != stable_seed(sample["id"], row["model"], 0)
                or raw["config"] != row["config"] or not raw["tokens"]):
            raise ValueError("Native response binding differs from the closed score source")
        key = (row["model"], sample["id"])
        if key in records:
            raise ValueError("Duplicate representative source key")
        records[key] = {"model": row["model"], "sample": sample, "seed": raw["seed"],
                        "native_vcd": {"text": raw["text"], "tokens": raw["tokens"],
                                       "terminated": raw["terminated"], "key": raw["key"],
                                       "config": raw["config"], "identity": raw["identity"]},
                        "source_path": row["source_path"], "source_line": row["source_line"],
                        "raw_line_sha256": row["raw_line_sha256"],
                        "verified_immutable_source_sha256": row["raw_source_sha256"],
                        "correct_canonical": row["canonical_name_in_primary_score"],
                        "abstain": row["abstain"]}
    output = destination / "representative_inputs.jsonl"
    with output.open("x", encoding="utf-8") as stream:
        for model in MODELS:
            for row in panel:
                stream.write(json.dumps(records[model, row["sample_id"]], ensure_ascii=False, allow_nan=False) + "\n")
    atomic_json(destination / "source_receipt.json", {
        "schema": "kdm_remaining4_representative_sources_v1", "models": list(MODELS),
        "rows": len(records), "inputs_per_model": 101, "new_gpu_forwards": 0,
        "scored_source": str(args.scores), "score_sha256": file_hash(args.scores),
        "panel_source": str(args.representative_list), "panel_sha256": file_hash(args.representative_list),
        "input_sha256": file_hash(output), "manifest_sha256": file_hash(ROOT / "data/current/all.jsonl"),
        "source_binding": "Selected raw identity, question, sample, seed, config and text checked; prior immutable file proofs reused",
        "new_guided_vcd_generations": 0, "diagnostic_and_case_sources": "Separate actual response comparisons required"})
    print(json.dumps({"output": str(output), "rows": len(records), "source_binding_passed": True}))


if __name__ == "__main__":
    main()
