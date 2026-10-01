#!/usr/bin/env python3
"""Apply finite full-QA food reviews without reopening any raw source."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import file_hash, within
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.score import infer_qa, rows, score_target
from workflows.supplemental.remaining4.score_native import save_json, save_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, review_path, output = (within(ROOT, x) for x in (args.input, args.reviews, args.output))
    output.relative_to(ROOT / "outputs/supplemental/remaining4")
    source_receipt = json.loads((source.parent / "receipt.json").read_text())
    if (source_receipt.get("passed") is not True
            or file_hash(source) != source_receipt["outputs"][source.name]):
        raise ValueError("The input score bytes differ from their actual immutable receipt")
    patterns = frozen.compile_classes(sorted({r["class"] for _, r, _ in rows(ROOT / "data/current/all.jsonl")
                                              if r["dataset"] == "food101"}))
    decisions, expected = {}, set()
    for line, decision, _ in rows(review_path):
        key = frozen.qah(decision["question"], decision["answer"])
        span = decision.get("evidence_span") or decision.get("primary_answer_span") or decision.get("abstain_span")
        if (key != decision["qa_key"] or key in decisions or type(decision["abstain"]) is not bool
                or not isinstance(span, str) or not span or span not in decision["answer"]):
            raise ValueError("A finite reviewed complete QA or continuous primary span differs")
        decisions[key] = {**decision, "decision_source_path": str(review_path.relative_to(ROOT)),
                          "decision_source_line": line}
        for member in decision["source_memberships"]:
            expected.add((member["key"], member["source_identity"], member["raw_line_sha256"]))
    scored, delta, affected = [], [], set()
    for line, record, line_sha in rows(source):
        previous = dict(record)
        decision = decisions.get(record["qa_key"])
        if decision is not None:
            identity = record["key"], record["source_identity"], record["raw_line_sha256"]
            if record["dataset"] != "food101" or identity not in expected or identity in affected:
                raise ValueError("The finite review changed its source-bound Food member scope")
            inferred = infer_qa(record["question"], record["answer"], patterns, {}, {}, decisions)
            canonical, literal, reason = score_target(record["answer"], record["target_class"], inferred, patterns)
            if canonical not in (0, 1) or literal not in (0, 1):
                raise ValueError("The reviewed Food score remains unresolved")
            record.update(canonical_name_in_primary_score=canonical, literal_extracted_name_score=literal,
                          abstain=inferred["abstain"], score_reason=reason, primary_extraction=inferred["parsed"],
                          behavior_source=inferred["behavior_source"], decision_source=inferred["decision"])
            affected.add(identity)
            delta.append({"key": record["key"], "input_score_line": line, "input_score_line_sha256": line_sha,
                          "previous": previous, "updated": record})
        scored.append(record)
    if affected != expected or len({r["key"] for r in scored}) != len(scored):
        raise ValueError("Some reviewed source-bound members are missing from the actual input")
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "score_rows.jsonl.gz", scored)
    save_rows(output / "finite_review_delta.jsonl", delta)
    receipt = {"schema": "kdm_finite_food_review_application_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "rows": len(scored),
        "unique_keys": len({r["key"] for r in scored}), "reviewed_unique_QA": len(decisions),
        "affected_members": len(affected), "unaffected_objects_identical": len(scored) - len(affected),
        "food_canonical_pending": sum(r["canonical_name_in_primary_score"] is None for r in scored if r["dataset"] == "food101"),
        "food_literal_pending": sum(r["literal_extracted_name_score"] is None for r in scored if r["dataset"] == "food101"),
        "input_score_sha256": file_hash(source), "reviews_sha256": file_hash(review_path),
        "runner_sha256": file_hash(Path(__file__)), "raw_reopened": False, "GPU_initialized": False,
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("passed", "rows", "affected_members", "unaffected_objects_identical", "food_canonical_pending", "food_literal_pending")}))


if __name__ == "__main__":
    main()
