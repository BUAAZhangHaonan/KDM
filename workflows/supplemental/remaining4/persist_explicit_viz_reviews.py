"""Bind explicit model judgments to their immutable complete-QA input.

This writer never chooses a semantic label or answer span. An invalid declared
span remains in the record and requires an actual reviewer decision.
"""
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
from kdm.scoring import lexical_label
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.score import rows
from workflows.supplemental.remaining4.score_native import save_json, save_rows

LABELS = {"answer_assertive", "answer_uncertain", "abstain", "invalid"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("queue", "queue-sha256", "decisions", "decisions-sha256", "attestation", "attestation-sha256", "output"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    args = parser.parse_args()
    queue, declared, attestation, output = (
        within(ROOT, value) for value in (args.queue, args.decisions, args.attestation, args.output)
    )
    output.relative_to(ROOT / "outputs/supplemental/remaining4")
    for path, expected in ((queue, args.queue_sha256), (declared, args.decisions_sha256),
                           (attestation, args.attestation_sha256)):
        if file_hash(path) != expected:
            raise ValueError("The explicitly bound source changed: " + str(path))
    evidence = json.loads(attestation.read_text(encoding="utf-8"))
    if (evidence.get("actual_complete_QA_read_reported_by_model") is not True
            or evidence.get("annotation_model") != "gpt-5.6-luna"
            or evidence.get("annotation_effort") != "medium"
            or evidence.get("first") != args.first or evidence.get("last") != args.last
            or evidence.get("source_queue_sha256") != args.queue_sha256
            or evidence.get("model_decisions_sha256") != args.decisions_sha256):
        raise ValueError("The actual model-read attestation does not bind this finite batch")
    targets = {}
    for line, record, line_sha in rows(queue):
        index = line - 1
        if args.first <= index <= args.last:
            if frozen.qah(record["question"], record["answer"]) != record["qa_key"]:
                raise ValueError("A complete-QA key does not match its immutable input")
            targets[index] = record, line_sha
    expected_indices = set(range(args.first, args.last + 1))
    if set(targets) != expected_indices:
        raise ValueError("The requested complete-QA interval is not fully present")
    normalized, pending, seen = [], [], set()
    for line, judgment, line_sha in rows(declared):
        index = judgment.get("candidate_index")
        if type(index) is not int or index not in targets or index in seen:
            raise ValueError("A model judgment repeats or exceeds its assigned interval")
        seen.add(index)
        label, abstain = judgment.get("label"), judgment.get("abstain")
        span, proof, needs_root = (judgment.get(key) for key in ("answer_text", "evidence_span", "needs_root"))
        if (label not in LABELS or type(abstain) is not bool or abstain != (label == "abstain")
                or not isinstance(span, str) or not isinstance(proof, str) or type(needs_root) is not bool):
            raise ValueError("A model judgment lacks its explicit label, boolean, or span")
        target, source_line_sha = targets[index]
        reasons = []
        if needs_root:
            reasons.append("model_requested_actual_root_review")
        if (span and span not in target["answer"]) or (proof and proof not in target["answer"]):
            reasons.append("declared_span_not_continuous_original_text")
        if label.startswith("answer_") and not span:
            reasons.append("declared_answer_has_no_primary_span")
        if label == "invalid" and target["answer"].strip() and lexical_label(target["answer"]) != "invalid":
            reasons.append("finite_nonempty_invalid_behavior_review")
        record = {
            "qa_key": target["qa_key"], "question": target["question"], "answer": target["answer"],
            "candidate_index": index, "label": label, "abstain": abstain, "A": abstain,
            "answer_text": span, "answer_text_span": span, "endorsed_short_answer_text": span,
            "evidence_span": proof, "needs_root": bool(reasons), "root_review_required": bool(reasons),
            "annotation_author": evidence["annotation_author"], "annotation_model": "gpt-5.6-luna",
            "annotation_effort": "medium", "annotation_session_id": evidence.get("annotation_session_id", ""),
            "annotation_call_id": evidence.get("annotation_call_id", ""),
            "read_full_question_answer": True, "span_selection_mode": "actual_per_QA_model_choice",
            "annotation_writer": "/root", "annotation_write_is_semantic_judgment": False,
            "official_answers_exposed": False, "model_declared_judgment": judgment,
            "source_queue": str(queue.relative_to(ROOT)), "source_queue_sha256": args.queue_sha256,
            "source_queue_line": index + 1, "source_queue_line_sha256": source_line_sha,
            "model_decisions_source": str(declared.relative_to(ROOT)), "model_decisions_sha256": args.decisions_sha256,
            "model_decisions_line": line, "model_decisions_line_sha256": line_sha,
            "read_attestation": str(attestation.relative_to(ROOT)), "read_attestation_sha256": args.attestation_sha256,
            "technical_review_reasons": reasons,
        }
        normalized.append(record)
        if reasons:
            pending.append(record)
    if seen != expected_indices:
        raise ValueError("The actual explicit decisions leave missing assigned QA keys")
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "decisions.jsonl", normalized)
    save_rows(output / "pending_full_QA.jsonl", pending)
    receipt = {
        "passed": True, "actual_model_decisions": len(normalized), "pending_root_QA": len(pending),
        "labels": dict(Counter(row["label"] for row in normalized)), "first": args.first, "last": args.last,
        "writer_generated_semantic_labels_or_spans": False, "original_declared_judgments_preserved": True,
        "actual_model_read_attestation_sha256": args.attestation_sha256,
        "source_queue_sha256": args.queue_sha256, "model_decisions_sha256": args.decisions_sha256,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {path.name: file_hash(path) for path in output.iterdir() if path.is_file()},
    }
    save_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
