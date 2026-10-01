"""Apply accepted exact-QA Viz annotations to immutable received score cohorts."""
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
from kdm.models.official_vqa_normalizer import VQAEval
from kdm.scoring import vqa_score
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.score import rows
from workflows.supplemental.remaining4.score_native import save_json, save_rows

FIELDS = ("abstain", "behavior_source", "decision_source", "semantic_label", "semantic_answer_text",
          "answer_quality_credit", "answer_quality_pending", "answer_quality_source")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", action="append", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    authority_path, output = (within(ROOT, p) for p in (args.authority, args.output))
    output.relative_to(ROOT / "outputs/supplemental/remaining4")
    authority_sha = file_hash(authority_path)
    authority_receipt = json.loads((authority_path.parent / "receipt.json").read_text())
    if (authority_receipt.get("schema") != "kdm_actual_finite_semantic_authority_v1"
            or not authority_receipt.get("passed") or authority_receipt["outputs"][authority_path.name] != authority_sha):
        raise ValueError("The actual finite QA authority differs from its passed receipt")
    authority = {}
    for line, row, line_sha in rows(authority_path):
        qa = frozen.qah(row["question"], row["answer"])
        if (row["qa_key"] != qa or qa in authority or not row["annotation_complete"]
                or type(row["abstain"]) is not bool or type(row["A"]) is not bool
                or row["A"] != row["abstain"]
                or row["abstain"] != (row["label"] == "abstain")
                or any(row.get(k) for k in ("needs_root", "needs_root_review", "root_review_required"))):
            raise ValueError("The accepted actual QA authority is duplicated or remains unresolved")
        if row["label"] in {"answer_assertive", "answer_uncertain"} and (
                not row["answer_text"] or row["answer_text"] not in row["answer"]):
            raise ValueError("An actual QA authority lacks a continuous substantive answer span")
        pointer = {"path": str(authority_path.relative_to(ROOT)), "sha256": authority_sha,
                   "line": line, "line_sha256": line_sha}
        authority[qa] = row, pointer
    if len(authority) != authority_receipt["closed_QA"]:
        raise ValueError("Actual accepted QA count differs from its receipt")
    normalizer = VQAEval(None, None)
    normalize = lambda text: normalizer.processDigitArticle(normalizer.processPunctuation(text))
    scored, seen, matched, pending, delta, inputs, counts = [], set(), set(), {}, [], [], Counter()
    for relative in args.score_dir:
        directory = within(ROOT, relative)
        source = directory / "score_rows.jsonl.gz"
        source_sha = file_hash(source)
        receipt = json.loads((directory / "receipt.json").read_text())
        if not receipt.get("passed") or receipt["outputs"][source.name] != source_sha:
            raise ValueError("An immutable input score cohort changed")
        cohort_count = 0
        for line, original, line_sha in rows(source):
            if original["key"] in seen:
                raise ValueError("Received cohorts repeat a generation key")
            seen.add(original["key"])
            record = dict(original)
            if record["dataset"] == "food101":
                if record != original:
                    raise ValueError("Food objects must stay exactly unchanged by Viz review application")
                counts["food_objects_identical"] += 1
            elif record["dataset"] == "vizwiz":
                counts["viz_rows"] += 1
                qa = frozen.qah(record["question"], record["answer"])
                if qa != record["qa_key"]:
                    raise ValueError("The source score QA differs from its complete actual content")
                if qa in authority:
                    decision, pointer = authority[qa]
                    refs = record["official_answers"]
                    if len(refs) != 10:
                        raise ValueError("The frozen official ten answers are incomplete")
                    quality_resolved = decision.get("quality_span_resolved", True)
                    behavior_resolved = decision.get("behavior_resolved", True)
                    quality = (0.0 if decision["label"] in {"abstain", "invalid"}
                               else float(vqa_score(decision["answer_text"], refs, normalize))) if quality_resolved else None
                    if quality is not None and not 0 <= quality <= 1:
                        raise ValueError("Official continuous Viz quality is nonfinite or out of range")
                    thin = {k: decision[k] for k in ("qa_key", "question", "answer", "label", "abstain", "answer_text",
                            "annotation_author", "annotation_model", "annotation_effort", "annotation_call_id", "annotation_session_id")}
                    thin.update(actual_authority_source=pointer, root_review_required=False, needs_root=False,
                                quality_span_resolved=quality_resolved,
                                behavior_resolved=behavior_resolved,
                                quality_span_pending_reason=decision.get("quality_span_pending_reason"),
                                span_selection_mode=decision.get("span_selection_mode"))
                    record.update(abstain=decision["abstain"] if behavior_resolved else None,
                        behavior_source="actual_closed_exact_QA_review" if behavior_resolved else "actual_review_finite_recheck_pending",
                        decision_source=thin, semantic_label=decision["label"], semantic_answer_text=decision["answer_text"],
                        answer_quality_credit=quality, answer_quality_pending=not quality_resolved, answer_quality_source=thin)
                    if any(record[k] != value for k, value in original.items() if k not in FIELDS):
                        raise ValueError("Viz review application changed an input, condition, source, reference or raw official credit")
                    delta.append({"key": record["key"], "qa_key": qa, "input_score_source": str(source.relative_to(ROOT)),
                        "input_score_line": line, "input_score_line_sha256": line_sha,
                        "previous": {k: original.get(k) for k in FIELDS}, "updated": {k: record.get(k) for k in FIELDS}})
                    matched.add(qa)
                    counts["actual_review_members_applied"] += 1
                counts["viz_quality_pending"] += record["answer_quality_credit"] is None
                counts["viz_behavior_pending"] += record["abstain"] is None
                if record["answer_quality_credit"] is None or record["abstain"] is None:
                    target = pending.setdefault(qa, {"qa_key": qa, "question": record["question"], "answer": record["answer"],
                        "dataset": "vizwiz", "source_memberships": [], "needs_behavior": record["abstain"] is None,
                        "needs_answer_span": record["answer_quality_credit"] is None})
                    member = {k: record[k] for k in
                        ("model", "key", "sample_id", "source_path", "source_line", "source_identity", "seed", "replicate")}
                    member.update({k: record[k] for k in ("raw_line_sha256", "source_line_sha256") if k in record})
                    if not member.get("raw_line_sha256", member.get("source_line_sha256")):
                        raise ValueError("A pending Viz member lacks its original line evidence")
                    target["source_memberships"].append(member)
            else:
                raise ValueError("Received score dataset is outside the selected scope")
            scored.append(record)
            cohort_count += 1
        if cohort_count != receipt["rows"]:
            raise ValueError("Input cohort count differs from its actual passed receipt")
        inputs.append({"path": str(source.relative_to(ROOT)), "sha256": source_sha, "rows": cohort_count,
                       "receipt_sha256": file_hash(directory / "receipt.json")})
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "score_rows.jsonl.gz", scored)
    save_rows(output / "actual_review_delta.jsonl.gz", delta)
    save_rows(output / "viz_pending_complete_QA.jsonl", list(pending.values()))
    receipt = {"schema": "kdm_finite_actual_Viz_review_application_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "rows": len(scored), "unique_keys": len(seen),
        "counts": dict(counts), "matched_unique_QA": len(matched), "pending_unique_Viz_QA": len(pending),
        "authority_path": str(authority_path.relative_to(ROOT)), "authority_sha256": authority_sha,
        "authority_receipt_sha256": file_hash(authority_path.parent / "receipt.json"), "input_cohorts": inputs,
        "Food_fields_changed": 0, "raw_official_Viz_credit_changed": 0, "unmatched_objects_changed": 0,
        "new_scientific_judgments": 0, "raw_reopened": False, "GPU_initialized": False,
        "runner_sha256": file_hash(Path(__file__)), "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("passed", "rows", "unique_keys", "counts", "matched_unique_QA", "pending_unique_Viz_QA")}))


if __name__ == "__main__":
    main()
