"""Compose actual finite QA annotations with explicit correction precedence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
FLAGS = ("needs_root", "needs_root_review", "root_review_required")


def actual_span(row):
    for field in ("answer_text", "endorsed_short_answer_text", "answer_text_span"):
        value = row.get(field)
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
            value = value[0]
        if isinstance(value, str) and value:
            return value, field
    return None, None


def signature(row):
    return (row.get("label"), row.get("abstain"), actual_span(row)[0],
            tuple(bool(row.get(k)) for k in FLAGS))


def known_identity(decision):
    actual = (decision.get("annotation_model"), decision.get("annotation_effort", decision.get("effort")))
    rule = (str(decision.get("annotation_author", "")).startswith("rule:")
            and lexical_label(decision["answer"]) == decision.get("label")
            and decision.get("label") in LABELS)
    return actual in {("gpt-5.6-luna", "medium"), ("gpt-6.1-sol", "max")} or rule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path, output = (within(ROOT, x) for x in (args.sources, args.output))
    output.relative_to(ROOT / "outputs/supplemental/remaining4")
    config = json.loads(config_path.read_text())
    targets, queue_sources = {}, []
    for entry in config["target_queues"]:
        path = within(ROOT, entry["path"])
        if file_hash(path) != entry["sha256"]:
            raise ValueError("The explicitly finite target QA population changed")
        count = 0
        for line, record, _sha in rows(path):
            qa = frozen.qah(record["question"], record["answer"])
            if qa != record["qa_key"] or qa in targets:
                raise ValueError("The finite owner queues contain an invalid or repeated QA key")
            targets[qa] = record
            count += 1
        if count != entry["qa_count"]:
            raise ValueError("Target owner queue count differs")
        queue_sources.append({**entry, "actual_rows": count})
    if len(targets) != config["expected_unique_QA"]:
        raise ValueError("The actual finite target union differs from its explicit budget")
    chosen, history, source_manifest, counts = {}, defaultdict(list), [], Counter()
    for entry in sorted(config["entries"], key=lambda e: (e["priority"], e["path"])):
        path = within(ROOT, entry["path"])
        sha = file_hash(path)
        if sha != entry["sha256"]:
            raise ValueError("An actual annotation source changed: " + str(path))
        receipt = entry.get("receipt")
        if receipt:
            receipt_path = within(ROOT, receipt["path"])
            if receipt_path == path:
                raise ValueError("An annotation file is not an independent receipt")
            if file_hash(receipt_path) != receipt["sha256"]:
                raise ValueError("An annotation receipt changed")
        source_count, seen = 0, set()
        for line, decision, line_sha in rows(path):
            qa = frozen.qah(decision["question"], decision["answer"])
            target = targets.get(qa)
            if (qa != decision["qa_key"] or target is None or qa in seen
                    or (decision["question"], decision["answer"]) != (target["question"], target["answer"])):
                raise ValueError("An actual annotation changed or exceeded its complete-QA scope")
            seen.add(qa)
            pointer = {"path": entry["path"], "sha256": sha, "line": line, "line_sha256": line_sha,
                       "priority": entry["priority"], "receipt": receipt}
            history[qa].append(pointer)
            previous = chosen.get(qa)
            if previous and previous[0] == entry["priority"] and signature(previous[1]) != signature(decision):
                raise ValueError("Equal-priority actual annotations disagree: " + qa)
            chosen[qa] = entry["priority"], decision, pointer
            source_count += 1
        if source_count != entry["qa_count"]:
            raise ValueError("An actual annotation count differs from the finite source manifest")
        source_manifest.append({**entry, "actual_rows": source_count})
    accepted, pending = [], []
    for qa, target in sorted(targets.items()):
        source = chosen.get(qa)
        if source is None:
            pending.append({**target, "pending_reason": "no_actual_annotation"})
            continue
        _priority, decision, pointer = source
        label, behavior = decision.get("label"), decision.get("abstain")
        span, span_source_field = actual_span(decision)
        reason = None
        if any(decision.get(k) for k in FLAGS):
            reason = "actual_review_flag_remains"
        elif not known_identity(decision):
            reason = "actual_author_model_effort_unresolved"
        elif label not in LABELS or type(behavior) is not bool or behavior != (label == "abstain"):
            reason = "actual_semantic_label_or_behavior_disagrees"
        elif label in {"answer_assertive", "answer_uncertain"} and (
                not isinstance(span, str) or not span or span not in target["answer"]):
            reason = "actual_substantive_span_is_missing_or_noncontinuous"
        if reason:
            pending.append({**target, "pending_reason": reason, "actual_annotation": decision,
                            "authority_sources": history[qa]})
            continue
        if label in {"abstain", "invalid"}:
            span = ""
        if "A" in decision and (type(decision["A"]) is not bool or decision["A"] != behavior):
            counts["technical_A_alias_repairs"] += 1
        record = {"qa_key": qa, "question": target["question"], "answer": target["answer"],
            "dataset": "vizwiz", "label": label, "abstain": behavior, "A": behavior,
            "answer_text": span, "answer_text_span": span, "annotation_complete": True,
            "behavior_resolved": True, "quality_span_resolved": True, "needs_root": False,
            "needs_root_review": False, "root_review_required": False,
            "actual_annotation": decision, "authority_sources": history[qa], "selected_source": pointer,
            "source_memberships": target.get("source_memberships", []),
            "source_queues": target.get("source_queues", []),
            "annotation_author": decision.get("annotation_author"),
            "annotation_model": decision["annotation_model"],
            "annotation_effort": decision.get("annotation_effort", decision.get("effort")),
            "annotation_call_id": decision.get("annotation_call_id", decision.get("callID", "")),
            "annotation_session_id": decision.get("annotation_session_id", decision.get("sessionID", "")),
            "reported_source_A": decision.get("A"), "reported_source_answer_text": decision.get("answer_text"),
            "actual_span_source_field": span_source_field,
            "full_reply_fallback_used": False, "new_scientific_judgment": False}
        accepted.append(record)
        counts[label] += 1
        counts["final_author:" + str(decision.get("annotation_author"))] += 1
        counts["source_memberships"] += len(record["source_memberships"])
    if len(accepted) + len(pending) != len(targets):
        raise ValueError("The finite annotation union lost a target QA")
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "viz_exact_QA_authority.jsonl", accepted)
    save_rows(output / "pending_complete_QA.jsonl", pending)
    save_json(output / "actual_source_manifest.json", source_manifest)
    receipt = {"schema": "kdm_actual_finite_semantic_authority_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "unique_QA": len(targets),
        "closed_QA": len(accepted), "pending_QA": len(pending), "counts": dict(counts),
        "target_queues": queue_sources, "source_config_sha256": file_hash(config_path),
        "runner_sha256": file_hash(Path(__file__)), "original_annotations_preserved": True,
        "unknown_call_ids_invented": False, "new_scientific_judgments": 0,
        "unreviewed_labels_defaulted": False, "raw_reopened": False, "GPU_initialized": False,
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("passed", "unique_QA", "closed_QA", "pending_QA", "counts")}))


if __name__ == "__main__":
    main()
