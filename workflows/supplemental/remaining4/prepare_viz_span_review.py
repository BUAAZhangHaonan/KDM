"""Screen an explicitly bounded QA authority without creating semantic labels."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import file_hash, within
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.score import rows
from workflows.supplemental.remaining4.score_native import save_json, save_rows

DESCRIPTIVE = re.compile(r"\b(describe|description|tell me about|what (?:does|do) .*look like|what is happening|what.*doing)\b", re.I)
INTRO = re.compile(r"^(?:yes[, .]+)?(?:it|this|that|the (?:image|picture|photo|object|item|product|text|label|tag|name tag|bottle|can|box|package))\s+(?:is|are|looks|appears|seems|shows|reads|says|contains|depicts)", re.I)
WITHDRAWAL = re.compile(r"\b(?:I (?:don't|do not) know (?:what|which|when|where|how)|(?:cannot|can't|unable to)\s+(?:tell|read|identify|determine|see))\b", re.I)
QUOTED = re.compile(r'["“][^"”\n]{1,100}["”]')
BEHAVIOR_REASONS = {"possible_withdrawal_inside_complete_answer", "nonempty_answer_marked_invalid"}


def candidate_reasons(record):
    """Return review candidates, never a new label or a replacement answer."""
    if (record.get("span_selection_mode") == "actual_per_QA_model_choice"
            and record["actual_annotation"].get("read_full_question_answer") is True):
        return []
    question, answer, span = record["question"], record["answer"], record["answer_text"]
    answered = record["label"] in {"answer_assertive", "answer_uncertain"}
    reasons = []
    if answered and not DESCRIPTIVE.search(question):
        if INTRO.search(span):
            reasons.append("identification_answer_intro_in_selected_span")
        if QUOTED.search(span) and len(span.split()) > 3:
            reasons.append("quoted_named_answer_inside_sentence")
        if len(span.split()) > 7:
            reasons.append("long_identification_span")
        if len(answer.split()) > len(span.split()) + 6:
            reasons.append("multiple_sentence_identification_answer")
    if answered and WITHDRAWAL.search(answer):
        reasons.append("possible_withdrawal_inside_complete_answer")
    if record["label"] == "invalid" and answer.strip():
        reasons.append("nonempty_answer_marked_invalid")
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--expected-n", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    authority, output = (within(ROOT, value) for value in (args.authority, args.output))
    output.relative_to(ROOT / "outputs/supplemental/remaining4")
    source_receipt = json.loads((authority.parent / "receipt.json").read_text())
    if not source_receipt["passed"] or source_receipt["outputs"][authority.name] != file_hash(authority):
        raise ValueError("The bounded authority differs from its accepted source receipt")
    candidates, holds, seen, counts = [], [], set(), Counter()
    for line, record, line_sha in rows(authority):
        qa = frozen.qah(record["question"], record["answer"])
        if record["qa_key"] != qa or qa in seen:
            raise ValueError("The bounded QA authority has changed or repeats a key")
        seen.add(qa)
        reasons = candidate_reasons(record)
        if not reasons:
            continue
        counts.update(reasons)
        candidates.append({"qa_key": qa, "question": record["question"], "answer": record["answer"],
            "previous_label": record["label"], "previous_abstain": record["abstain"],
            "previous_answer_text": record["answer_text"], "candidate_reasons": reasons,
            "source_memberships": record["source_memberships"], "selected_source": record["selected_source"],
            "source_authority_line": line, "source_authority_line_sha256": line_sha,
            "screening_is_semantic_judgment": False, "official_answers_exposed": False})
        holds.append({"qa_key": qa, "pending_reason": "finite_primary_answer_or_behavior_review",
                      "candidate_reasons": reasons, "needs_behavior": bool(set(reasons) & BEHAVIOR_REASONS)})
    if len(seen) != args.expected_n:
        raise ValueError("The bounded QA count differs from the explicit population")
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "candidate_complete_QA.jsonl", candidates)
    save_rows(output / "span_and_behavior_review_holds.jsonl", holds)
    receipt = {"schema": "kdm_finite_Viz_span_review_scope_v1", "passed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "screened_unique_QA": len(seen),
        "candidate_unique_QA": len(candidates), "behavior_review_pending_QA": sum(r["needs_behavior"] for r in holds),
        "candidate_source_memberships": sum(len(r["source_memberships"]) for r in candidates),
        "screening_reason_counts": dict(counts), "scientific_labels_changed_by_screen": 0,
        "official_answers_exposed": False, "GPU_initialized": False, "raw_reopened": False,
        "source_authority": str(authority.relative_to(ROOT)), "source_authority_sha256": file_hash(authority),
        "runner_sha256": file_hash(Path(__file__)),
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("passed", "screened_unique_QA", "candidate_unique_QA", "behavior_review_pending_QA")}))


if __name__ == "__main__":
    main()
