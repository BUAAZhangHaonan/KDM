"""Validate one completed actual-Luna batch without changing its judgments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workflows.general_vqa_direct.score import (
    check_decision, check_selected_batch, digest, rows, validate_review_author,
)
from workflows.general_vqa_direct.hallusion_scoring import HallusionScoring


def validate_batch(batch, author):
    assignment = json.loads((batch / "assignment.json").read_text())
    if assignment["owner"] != author:
        raise ValueError("Expected the actual assigned reviewer")
    path = batch / "decisions.jsonl"
    check_selected_batch(path, "qa_key")
    pending = {r["qa_key"]: r for _, r in rows(batch / "pending_review.jsonl")}
    hallusion = None
    hall_by_prompt = {}
    if any(r["dataset"] == "hallusionbench" for r in pending.values()):
        hallusion = HallusionScoring(ROOT)
        for sample in hallusion.samples:
            hall_by_prompt.setdefault(sample["prompt"], []).append(sample)
    for _, record in rows(path):
        source = pending[record["qa_key"]]
        if any(record.get(k) != source[k] for k in ("question", "answer")):
            raise ValueError("The full question or answer differs from the assignment")
        if record["qa_key"] != digest([source["question"], source["answer"]]):
            raise ValueError("QA key is not bound to the complete text")
        if (record.get("author") != author or record.get("model") != "gpt-5.6-luna"
                or record.get("effort") != "medium"):
            raise ValueError("Actual author/model/effort must be top-level wrapper fields")
        validate_review_author(record["author"])
        if "call_id" not in record or (record["call_id"] is not None and (
                not isinstance(record["call_id"], str) or not record["call_id"].strip())):
            raise ValueError("Top-level call_id must be null or a nonempty actual ID")
        samples = hall_by_prompt[source["question"]] if source["dataset"] == "hallusionbench" else [source]
        for sample in samples:
            check_decision(sample, source["answer"], record["decision"])
            if sample["dataset"] == "hallusionbench":
                hallusion.check_behavior_prediction(sample, record["decision"].get("predicted_answer"))
    return {"status": "passed", "rows": len(pending), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "author": author, "model": "gpt-5.6-luna", "effort": "medium",
            "checks": ["full unique assigned-key coverage", "full QA and key binding",
                       "top-level actual reviewer provenance", "check_decision",
                       "Hallusion question-kind extraction where applicable"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--author", required=True)
    args = parser.parse_args()
    print(json.dumps(validate_batch(args.batch, args.author)))
