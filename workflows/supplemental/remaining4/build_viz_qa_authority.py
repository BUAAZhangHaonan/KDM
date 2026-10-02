"""Build finite exact-QA Viz authority from real final census and explicit reviews."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from kdm.io import file_hash, within
from kdm.scoring import lexical_label
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.score import rows
from workflows.supplemental.remaining4.score_native import save_json, save_rows

LABELS = {"answer_assertive", "answer_uncertain", "abstain", "invalid"}


def decision_value(decision, question, answer):
    if (decision["qa_key"] != frozen.qah(question, answer) or decision["question"] != question
            or decision["answer"] != answer or type(decision.get("abstain")) is not bool
            or decision.get("needs_root") or decision.get("root_review_required")):
        return None
    label = decision.get("label")
    if decision["abstain"]:
        return "abstain", ""
    if label == "invalid" or decision.get("status") == "invalid":
        return "invalid", ""
    answer_text = decision.get("answer_text")
    if not answer_text and decision.get("answer_text_span"):
        answer_text = decision["answer_text_span"]
    if answer_text and answer_text in answer:
        return label if label in {"answer_assertive", "answer_uncertain"} else None, answer_text
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--decision-file", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    score_dir, output = (within(ROOT, p) for p in (args.score_dir, args.output))
    score_path = score_dir / "score_rows.jsonl.gz"
    source_receipt = json.loads((score_dir / "receipt.json").read_text())
    sha = file_hash(score_path)
    if not source_receipt["passed"] or sha != source_receipt["outputs"][score_path.name]:
        raise ValueError("The finite existing score object population differs from its actual receipt")
    members, current, inline_decisions, total, models = defaultdict(list), {}, defaultdict(list), 0, Counter()
    source_keys = set()
    for line, row, line_sha in rows(score_path):
        if row["key"] in source_keys:
            raise ValueError("The finite existing score population contains a duplicate source key")
        source_keys.add(row["key"])
        total += 1
        models[row["model"] + ":" + row["dataset"]] += 1
        if row["dataset"] != "vizwiz":
            continue
        qa = frozen.qah(row["question"], row["answer"])
        if qa != row["qa_key"]:
            raise ValueError("A source exact-QA key differs from its complete original question and answer")
        if qa in current and current[qa] != (row["question"], row["answer"]):
            raise ValueError("An exact-QA key binds different complete original content")
        current[qa] = row["question"], row["answer"]
        members[qa].append({"model": row["model"], "key": row["key"], "sample_id": row["sample_id"],
            "source_path": row["source_path"], "source_line": row["source_line"], "source_identity": row["source_identity"],
            "raw_line_sha256": row.get("raw_line_sha256", row.get("source_line_sha256")),
            "score_source_path": str(score_path.relative_to(ROOT)), "score_source_line": line,
            "score_source_line_sha256": line_sha, "seed": row.get("seed"), "replicate": row["replicate"]})
        if row.get("decision_source"):
            inline_decisions[qa].append({"decision": row["decision_source"], "existing_score_source": members[qa][-1]})
    if total != source_receipt["rows"] or len(source_keys) != source_receipt["unique_keys"]:
        raise ValueError("Actual existing score population differs from its finite receipt")
    assets = json.loads((ROOT / "outputs/supplemental/remaining11/run_20260930_140337/assets/asset_manifest.json").read_text())
    census_path = Path(assets["census_final_labels"]["path"])
    exact_census, census_n = defaultdict(list), 0
    for line, row, line_sha in rows(census_path):
        census_n += 1
        if row["dataset"] != "vizwiz":
            continue
        qa = frozen.qah(row["question"], row["text"])
        if qa not in current:
            continue
        if row["label"] not in LABELS:
            raise ValueError("The actual final census has an undecided behavior label")
        exact_census[qa].append({"source_path": str(census_path), "source_line": line, "source_line_sha256": line_sha,
            "actual_annotation": row, "label": row["label"], "answer_text": row.get("answer_text", "")})
    if census_n != 293344:
        raise ValueError("The actual final census closure must have 293344 records")
    explicit, decision_sources = {}, []
    for relative in args.decision_file:
        path = within(ROOT, relative)
        source_sha, matched = file_hash(path), 0
        for line, decision, line_sha in rows(path):
            qa = decision["qa_key"]
            if qa not in current:
                continue
            question, answer = current[qa]
            if decision_value(decision, question, answer) is None:
                raise ValueError("An explicitly accepted exact-QA review has no closed behavior or substantive original span")
            explicit[qa] = {"decision": decision, "source_path": str(path.relative_to(ROOT)),
                "source_sha256": source_sha, "source_line": line, "source_line_sha256": line_sha}
            matched += 1
        decision_sources.append({"path": str(path.relative_to(ROOT)), "sha256": source_sha, "matched_target_QA": matched})
    authority, unresolved, counts, by_model = [], [], Counter(), defaultdict(Counter)
    for qa in sorted(current):
        question, answer = current[qa]
        label, answer_text, sources, reason, behavior = None, None, [], None, None
        if qa in explicit:
            source = explicit[qa]
            label, answer_text = decision_value(source["decision"], question, answer)
            behavior = source["decision"]["abstain"]
            sources, reason = [source], "explicit_actual_exact_QA_review"
        elif inline_decisions[qa]:
            values = {value for entry in inline_decisions[qa]
                      if (value := decision_value(entry["decision"], question, answer)) is not None}
            if len(values) == 1:
                label, answer_text = next(iter(values))
                behavior = label == "abstain"
                sources, reason = inline_decisions[qa], "existing_source_bound_exact_QA_decision"
            else:
                sources, reason = inline_decisions[qa], "existing_decision_span_or_behavior_unresolved"
        elif exact_census[qa]:
            variants = {(entry["label"], entry["answer_text"]) for entry in exact_census[qa]}
            sources = exact_census[qa]
            if len(variants) == 1:
                label, actual_text = next(iter(variants))
                if label in {"abstain", "invalid"} and actual_text == "":
                    answer_text, reason = "", "actual_final_census_exact_QA"
                elif label in {"answer_assertive", "answer_uncertain"} and actual_text and actual_text in answer:
                    answer_text, reason = actual_text, "actual_final_census_exact_QA"
                else:
                    label, reason = None, "actual_final_census_missing_or_noncontinuous_span"
            else:
                behavior_values = {entry["label"] == "abstain" for entry in exact_census[qa]}
                behavior = next(iter(behavior_values)) if len(behavior_values) == 1 else None
                reason = "actual_final_census_exact_QA_span_or_label_variants"
        else:
            actual_lexical = lexical_label(answer)
            if actual_lexical is not None:
                label, answer_text, reason = actual_lexical, "", "existing_registered_exact_phrase_rule"
                sources = [{"kind": "existing_registered_lexical_label", "code_path": "src/kdm/scoring.py",
                    "code_sha256": file_hash(ROOT / "src/kdm/scoring.py"), "evidence_span": answer}]
            else:
                reason = "no_actual_exact_QA_annotation_or_registered_exact_phrase"
        if label is not None:
            behavior = label == "abstain"
        complete = answer_text is not None and (label is not None or behavior is False)
        if complete and label in {"answer_assertive", "answer_uncertain"} and (not answer_text or answer_text not in answer):
            raise ValueError("An authority claims a substantive annotation without its actual continuous original span")
        record = {"qa_key": qa, "question": question, "answer": answer, "abstain": behavior,
            "label": label, "answer_text": answer_text if complete else None,
            "answer_text_span": answer_text if complete else None, "annotation_complete": complete,
            "behavior_resolved": type(behavior) is bool, "quality_span_resolved": complete,
            "authority_reason": reason, "authority_sources": sources, "source_memberships": members[qa],
            "full_reply_fallback_used": False, "new_scientific_judgment": False, "prepared_by": "/root/assets"}
        authority.append(record)
        counts["Viz_QA"] += 1
        counts["Viz_members"] += len(members[qa])
        counts["closed_QA" if complete else "pending_QA"] += 1
        counts["closed_members" if complete else "pending_members"] += len(members[qa])
        counts[reason + "_members"] += len(members[qa])
        if behavior is True:
            counts["actual_A_members"] += len(members[qa])
        for member in members[qa]:
            by_model[member["model"]]["Viz_members"] += 1
            by_model[member["model"]]["closed_members" if complete else "pending_members"] += 1
        if not complete:
            unresolved.append(record)
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "viz_exact_QA_authority.jsonl", authority)
    save_rows(output / "viz_unresolved_complete_QA_queue.jsonl", unresolved)
    receipt = {"schema": "kdm_finite_viz_exact_QA_authority_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "source_population_rows": total,
        "source_model_dataset_counts": dict(models), "counts": dict(counts), "by_model": {k: dict(v) for k, v in by_model.items()},
        "source_score_path": str(score_path.relative_to(ROOT)), "source_score_sha256": sha,
        "source_receipt_sha256": file_hash(score_dir / "receipt.json"), "actual_census_closure_rows": census_n,
        "actual_census_source_path": str(census_path), "actual_census_source_sha256": file_hash(census_path),
        "explicit_review_sources_in_order": decision_sources, "lexical_rule_source_sha256": file_hash(ROOT / "src/kdm/scoring.py"),
        "current_score_objects_changed": 0, "gold_used_for_behavior_or_span": False,
        "unannotated_long_reply_fallback": False, "new_annotations_created_by_adapter": 0,
        "old_raw_reopened": False, "GPU_initialized": False, "adapter_sha256": file_hash(Path(__file__)),
        "outputs": {p.name: file_hash(p) for p in output.iterdir()}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({"passed": True, "source_population_rows": total, "source_model_dataset_counts": dict(models),
                      "counts": dict(counts), "by_model": {k: dict(v) for k, v in by_model.items()}}), flush=True)


if __name__ == "__main__":
    main()
