#!/usr/bin/env python3
"""Export finite exact-QA source matches without creating semantic decisions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kdm.io import file_hash, read_jsonl, within
from workflows.main_results.score import qah
from workflows.paper_core.score_native import save_json, save_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pending", required=True)
    parser.add_argument("--literal-only", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pending_path, literal_path, output = [within(ROOT, value) for value in
                                         (args.pending, args.literal_only, args.output)]
    pending = list(read_jsonl(pending_path))
    literal = json.loads(literal_path.read_text(encoding="utf-8-sig"))
    requested = {}
    for row in [*pending, *literal]:
        key = qah(row["question"], row["answer"])
        if row["qa_key"] != key:
            raise ValueError("A pending lookup key differs from its complete question/answer")
        blind = {k: row[k] for k in ("qa_key", "question", "answer")}
        if key in requested and requested[key] != blind:
            raise ValueError("A pending QA key refers to different full text")
        requested[key] = blind
    path = ROOT / "outputs/paper_20260929/qa.parquet"
    frame = pd.read_parquet(path)
    if len(frame) != 44782 or frame.qa_key.duplicated().any():
        raise ValueError("The frozen QA source roster differs")
    qa_sha = file_hash(path)
    frame = frame.astype(object).where(pd.notnull(frame), None)
    matches, key_only, matched_keys = [], [], set()
    for position, row in enumerate(frame.to_dict("records"), 1):
        key = row["qa_key"]
        if key not in requested:
            continue
        if row["question"] is None or row["answer"] is None:
            key_only.append({"qa_key": key, "frozen_parquet_row": position,
                             "original_source_locator": {k: row[k] for k in ("qa_id", "raw_model", "raw_source_file_id", "raw_source_line")}})
            continue
        if (row["question"], row["answer"]) != (requested[key]["question"], requested[key]["answer"]):
            raise ValueError("A frozen QA hash match differs from the actual full text")
        matched_keys.add(key)
        variants = json.loads(row["review_variants_json"]) if row["review_variants_json"] is not None else None
        matches.append({"frozen_parquet_row": position, "original_frozen_QA_record": row,
                        "original_review_variants": variants,
                        "current_native_pending_QA": requested[key],
                        "source_path": str(path.relative_to(ROOT)), "source_sha256": qa_sha,
                        "current_native_scorer_already_loads_frozen_main_authority": True,
                        "additional_decision_not_created": True})
    source = ROOT / "outputs/annotations/main_results/reviewed_answers.jsonl"
    source_manifest = json.loads((source.parent / "source_manifest.json").read_text(encoding="utf-8"))
    if file_hash(source) != source_manifest["reviewed_answers_sha256"]:
        raise ValueError("The accepted exact-QA authority differs from its frozen receipt")
    authority_matches = []
    for line, row in enumerate(read_jsonl(source), 1):
        key = qah(row["question"], row["answer"])
        if key in requested:
            if (row["question"], row["answer"]) != (requested[key]["question"], requested[key]["answer"]):
                raise ValueError("An authority key differs from the complete pending QA")
            authority_matches.append({"qa_key": key, "source_line": line,
                                      "original_authority_record": row,
                                      "source_path": str(source.relative_to(ROOT)),
                                      "source_sha256": source_manifest["reviewed_answers_sha256"],
                                      "already_reused_by_current_native_scorer": True})
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "exact_full_QA_frozen_records.jsonl", matches)
    save_rows(output / "existing_main_authority_records.jsonl", authority_matches)
    save_rows(output / "unmatched_blind_QA.jsonl", [requested[k] for k in sorted(requested) if k not in matched_keys])
    save_rows(output / "all_still_pending_blind_QA.jsonl", [requested[k] for k in sorted(requested)])
    save_json(output / "key_only_raw_source_locators.json", key_only)
    receipt = {"schema": "kdm_finite_pending_frozen_exact_QA_lookup_v1", "passed": True,
               "requested_unique_QA": len(requested), "exact_full_QA_matches": len(matches),
               "existing_main_authority_matches_already_reused": len(authority_matches),
               "QA_key_only_without_frozen_full_text": len(key_only),
               "unmatched_full_QA": len(requested) - len(matches),
               "new_semantic_judgments": 0, "new_authority_acceptance": False,
               "source_qa_path": str(path.relative_to(ROOT)), "source_qa_sha256": qa_sha,
               "pending_source": str(pending_path.relative_to(ROOT)), "pending_source_sha256": file_hash(pending_path),
               "literal_only_source": str(literal_path.relative_to(ROOT)), "literal_only_source_sha256": file_hash(literal_path),
               "runner_sha256": file_hash(Path(__file__)), "actual_command": [sys.executable, *sys.argv],
               "created_utc": datetime.now(timezone.utc).isoformat(), "GPU_initialized": False,
               "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("passed", "requested_unique_QA", "exact_full_QA_matches",
                                             "existing_main_authority_matches_already_reused", "unmatched_full_QA")}))


if __name__ == "__main__":
    main()
