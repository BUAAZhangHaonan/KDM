#!/usr/bin/env python3
"""Prepare complete reference QA for root; apply only explicit actual root rulings."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, stable_hash, stable_seed, within
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.score import infer_qa, rows, score_target
from workflows.supplemental.remaining4.review_phi_boundary69 import AUTHOR
from workflows.supplemental.remaining4.score_native import save_json, save_rows


def load_finite(folder):
    receipt = json.loads((folder / "finite_execution_receipt.json").read_text())
    for name in ("score_rows.jsonl.gz", "independent_score_rows_enriched.jsonl.gz", "boundary_queue.jsonl", "reference_G.jsonl"):
        if file_hash(folder / name) != receipt["outputs"][name]:
            raise ValueError("Finite reference score or original packet bytes differ")
    score = [row for _line, row, _sha in rows(folder / "independent_score_rows_enriched.jsonl.gz")]
    core = [row for _line, row, _sha in rows(folder / "score_rows.jsonl.gz")]
    packet = [row for _line, row, _sha in rows(folder / "boundary_queue.jsonl")]
    by_key = {row["key"]: row for row in score}
    if not receipt["passed"] or len(by_key) != receipt["rows"] or len(score) != len(core):
        raise ValueError("Actual finite score keys differ")
    proof_path = within(ROOT, score[0]["source_CPU_verification_path"])
    proof = json.loads(proof_path.read_text())
    if not proof["passed"] or file_hash(proof_path) != receipt["source_CPU_verification_sha256"]:
        raise ValueError("Actual original raw CPU source proof differs")
    parts = {(p["claim_id"], p["part"]): p for p in proof["parts"]}
    member_keys = set()
    for item in packet:
        if item["qa_key"] != frozen.qah(item["question"], item["answer"]):
            raise ValueError("Original complete QA hash differs")
        for member in item["memberships"]:
            row = by_key[member["key"]]
            part = parts[row["source_claim"], row["source_part"]]
            fields = ("model", "stage", "split", "sample_id", "target_class", "key", "source_path", "source_line", "source_identity", "raw_line_sha256")
            if (any(member[name] != row[name] for name in fields)
                    or (row["question"], row["answer"], row["qa_key"]) != (item["question"], item["answer"], item["qa_key"])
                    or row["model"] != "internvl35_8b" or row["kind"] != "independent_attempt" or not row["attempt"]
                    or row["source_identity"] != part["actual_source_identity"]
                    or within(ROOT, row["source_path"]) != within(ROOT, part["files"]["raw"]["received_path"])
                    or row["raw_source_sha256"] != part["files"]["raw"]["sha256"]
                    or row["source_CPU_verification_sha256"] != file_hash(proof_path)
                    or row["replicate"] not in range(10) or row["attempt_ordinal"] != row["replicate"] + 1
                    or row["seed"] != stable_seed(row["sample_id"], row["model"], row["replicate"])
                    or row["config_sha256"] != stable_hash(row["config"]) or row["key"] in member_keys):
                raise ValueError("Reference QA member identity, seed, original source hash or config differs")
            member_keys.add(row["key"])
    pending_keys = {r["key"] for r in score if r["canonical_name_in_primary_score"] is None or r["literal_extracted_name_score"] is None or r["abstain"] is None}
    if (len(packet) != receipt["boundary_QA"] or len({r["qa_key"] for r in packet}) != len(packet)
            or member_keys != pending_keys or receipt["canonical_scorer_source_sha256"] != file_hash(Path(frozen.__file__))):
        raise ValueError("Finite packet does not cover exactly the actual pending score members")
    classes = sorted({r["class"] for _line, r, _sha in rows(ROOT / "data/current/all.jsonl") if r["dataset"] == "food101"})
    if len(classes) != 101:
        raise ValueError("Registered class set differs")
    return receipt, score, core, packet, by_key, proof_path, member_keys, frozen.compile_classes(classes)


def prepare(folder, output):
    receipt, _score, _core, packet, by_key, proof_path, member_keys, patterns = load_finite(folder)
    summaries, text = [], []
    for index, item in enumerate(packet, 1):
        inferred = infer_qa(item["question"], item["answer"], patterns, {}, {}, {})
        members = [by_key[m["key"]] for m in item["memberships"]]
        summaries.append({"index": index, "qa_key": item["qa_key"], "question": item["question"], "answer": item["answer"],
            "rule_extraction": inferred["parsed"], "rule_name_boundary": inferred["name_boundary"],
            "recorded_score_reasons": sorted({r["score_reason"] for r in members}), "source_member_count": len(members),
            "canonical_pending_members": sum(r["canonical_name_in_primary_score"] is None for r in members),
            "literal_pending_members": sum(r["literal_extracted_name_score"] is None for r in members),
            "recorded_semantic_A": sorted({r["abstain"] for r in members}, key=str)})
        text.extend([f"## {index}. {item['qa_key']}", "", f"Question: {item['question']}", "", f"Answer: {item['answer']}", "",
            "Rule extraction: " + json.dumps(inferred["parsed"], ensure_ascii=False),
            "Rule status: " + ", ".join(summaries[-1]["recorded_score_reasons"]), ""])
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "qa_review_summary.jsonl", summaries)
    (output / "qa_review_summary.md").write_text("\n".join(text), encoding="utf-8")
    result = {"schema": "kdm_remaining4_reference_root_read_packet_v1", "passed": True, "QA": len(packet),
        "source_members_verified": len(member_keys), "source_packet_sha256": file_hash(folder / "boundary_queue.jsonl"),
        "source_score_sha256": file_hash(folder / "independent_score_rows_enriched.jsonl.gz"),
        "source_CPU_verification_sha256": file_hash(proof_path), "root_rulings_created": False, "raw_reopened": False,
        "outputs": {p.name: file_hash(p) for p in output.iterdir()}, "preparer_sha256": file_hash(Path(__file__))}
    save_json(output / "receipt.json", result)
    print(json.dumps(result, indent=2))


def apply_rulings(folder, luna_dir, rulings_path, review_output, output):
    receipt, score, core, packet, by_key, proof_path, member_keys, patterns = load_finite(folder)
    rulings = json.loads(rulings_path.read_text())
    actual_luna = list(rows(luna_dir / "decisions.jsonl"))
    luna_receipt = json.loads((luna_dir / "execution_receipt.json").read_text())
    queue_sha = file_hash(folder / "boundary_queue.jsonl")
    luna_sha = file_hash(luna_dir / "decisions.jsonl")
    if (rulings["actual_author"] != AUTHOR or rulings["prepared_by"] != "/root/assets"
            or rulings["source_packet_sha256"] != queue_sha or len(rulings["rulings"]) != len(packet)
            or len(actual_luna) != len(packet) or luna_receipt["queue_sha256"] != queue_sha
            or luna_receipt["sha256"] != luna_sha or luna_receipt["annotation_model"] != "gpt-5.6-luna"
            or luna_receipt["effort"] != "medium" or not luna_receipt["source_memberships_preserved"]
            or luna_receipt["source_memberships"] != len(member_keys)):
        raise ValueError("Explicit actual root rulings or successful Luna source write is not bound to this packet")
    root_decisions, proofs = [], []
    for index, (item, actual, ruling) in enumerate(zip(packet, actual_luna, rulings["rulings"]), 1):
        line, decision, line_sha = actual
        if (ruling["index"] != index or ruling["qa_key"] != item["qa_key"] or decision["qa_key"] != item["qa_key"]
                or (decision["question"], decision["answer"]) != (item["question"], item["answer"])
                or decision["source_memberships"] != item["memberships"] or type(ruling["abstain"]) is not bool
                or type(ruling["multiple_primary"]) is not bool or not ruling["reason"]
                or any(value not in item["answer"] for value in [ruling["span"], *ruling["names"]])):
            raise ValueError("Root order, full QA, original continuous spans or original Luna members differ")
        primary = None if ruling["multiple_primary"] else ruling["primary_name"]
        if not ruling["multiple_primary"] and (not primary or primary not in item["answer"]):
            raise ValueError("Actual unique primary name is not a complete original continuous span")
        relations = ruling.get("name_relations", [{"name": n, "span": n, "role": "coequal" if ruling["multiple_primary"] else "main"} for n in ruling["names"]])
        if any(not r["span"] or r["span"] not in item["answer"] for r in relations):
            raise ValueError("Root relations contain a non-original span")
        root = {**decision, "queue_index": index, "abstain": ruling["abstain"], "needs_root": False,
            "root_review_required": False, "multiple_primary": ruling["multiple_primary"],
            "status": "multiple_primary" if ruling["multiple_primary"] else "resolved_primary",
            "extracted_name": primary, "fullspan_primary_name": primary, "literal_full_name": primary, "literal_name": primary,
            "primary_answer_span": ruling["span"], "endorsed_primary_names": ruling["names"], "name_relations": relations,
            "name_scope_ambiguous": False, "canonical_override": None, "root_reason": ruling["reason"], "reason": ruling["reason"],
            "actual_author": AUTHOR, "root_review_author": AUTHOR, "prepared_by": "/root/assets",
            "annotation_model": AUTHOR["model"], "annotation_effort": AUTHOR["effort"], "effort": AUTHOR["effort"],
            "annotation_author": AUTHOR["agent"], "annotation_agent": AUTHOR["agent"],
            "annotation_session_id": AUTHOR["session_id"], "session_id": AUTHOR["session_id"], "annotation_call_id": "", "call_id": "",
            "source_luna_decision": decision, "source_luna_path": str((luna_dir / "decisions.jsonl").relative_to(ROOT)),
            "source_luna_sha256": luna_sha, "source_luna_line": line, "source_luna_line_sha256": line_sha,
            "source_packet_path": str((folder / "boundary_queue.jsonl").relative_to(ROOT)), "source_packet_sha256": queue_sha,
            "source_CPU_verification_path": str(proof_path.relative_to(ROOT)), "source_CPU_verification_sha256": file_hash(proof_path),
            "actual_root_rulings_path": str(rulings_path.relative_to(ROOT)), "actual_root_rulings_sha256": file_hash(rulings_path), "raw_reopened": False}
        for member in item["memberships"]:
            r = by_key[member["key"]]
            proofs.append({**member, "qa_key": item["qa_key"], "replicate": r["replicate"], "attempt_ordinal": r["attempt_ordinal"],
                "seed": r["seed"], "config_sha256": r["config_sha256"], "raw_source_sha256": r["raw_source_sha256"],
                "source_CPU_verification_sha256": file_hash(proof_path), "raw_reopened": False})
        root_decisions.append(root)
    review_output.mkdir(parents=True, exist_ok=False)
    decision_path = review_output / "root_reviewed.jsonl"
    save_rows(decision_path, root_decisions)
    save_rows(review_output / "source_member_verification.jsonl", proofs)
    decisions = {r["qa_key"]: {**r, "decision_source_path": str(decision_path), "decision_source_line": i} for i, r in enumerate(root_decisions, 1)}
    updated, replacements = [], {}
    for original in score:
        if original["key"] not in member_keys:
            updated.append(original)
            continue
        inferred = infer_qa(original["question"], original["answer"], patterns, {}, {}, decisions)
        canonical, literal, reason = score_target(original["answer"], original["target_class"], inferred, patterns)
        state = frozen.extract(inferred["decision"], patterns)
        if canonical is None or literal is None or type(inferred["abstain"]) is not bool:
            raise ValueError("Actual root ruling left a main score unresolved")
        if inferred["decision"]["multiple_primary"] and (not state["multi"] or canonical != 0 or literal != 0):
            raise ValueError("Frozen scoring did not respect actual multiple primary roles")
        changes = {"canonical_name_in_primary_score": canonical, "literal_extracted_name_score": literal,
            "literal_extracted_names": state["literal"], "abstain": inferred["abstain"], "score_reason": reason,
            "decision_source": inferred["decision"], "behavior_source": inferred["behavior_source"],
            "annotation_model": AUTHOR["model"], "annotation_effort": AUTHOR["effort"], "annotation_call_id": "",
            "annotation_session_id": AUTHOR["session_id"], "previous_score_source_path": str(folder.relative_to(ROOT)),
            "previous_score_source_sha256": file_hash(folder / "independent_score_rows_enriched.jsonl.gz")}
        replacements[original["key"]] = changes
        updated.append({**original, **changes, "previous_score_fields": {name: original.get(name) for name in changes}})
    if (set(replacements) != member_keys or any(before != after for before, after in zip(score, updated) if before["key"] not in member_keys)
            or any(r["canonical_name_in_primary_score"] is None or r["literal_extracted_name_score"] is None or r["abstain"] is None for r in updated)):
        raise ValueError("Finite root update changed unrelated objects or left pending scores")
    updated_by_key = {r["key"]: r for r in updated}
    references = []
    for _line, ref, _sha in rows(folder / "reference_G.jsonl"):
        attempts = [{**a, "canonical_name_in_primary_score": updated_by_key[a["key"]]["canonical_name_in_primary_score"],
                     "abstain": updated_by_key[a["key"]]["abstain"]} for a in ref["attempts"]]
        resolved = {a["replicate"] for a in attempts} == set(range(10)) and all(a["canonical_name_in_primary_score"] is not None for a in attempts)
        if ref["gold_rank"] is not None or ref["reference_G"] is not None or ref["reference_complete"]:
            raise ValueError("The finite InternVL source unexpectedly includes a registered rank or complete uniform GT")
        references.append({**ref, "attempts": attempts, "correct_count": sum(a["canonical_name_in_primary_score"] == 1 for a in attempts) if resolved else None})
    coverage = [{**r, "samples_with_all10_correctness_decided": sum(ref["split"] == r["split"] and ref["correct_count"] is not None for ref in references)}
                for _line, r, _sha in rows(folder / "attempt_coverage.jsonl")]
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "independent_score_rows_enriched.jsonl.gz", updated)
    save_rows(output / "score_rows.jsonl.gz", [{**r, **replacements.get(r["key"], {})} for r in core])
    save_rows(output / "reference_G.jsonl", references)
    save_rows(output / "attempt_coverage.jsonl", coverage)
    save_rows(output / "boundary_queue.jsonl", [])
    result = {**receipt, "completed_utc": datetime.now(timezone.utc).isoformat(), "canonical_pending": 0, "literal_pending": 0,
        "abstain_pending": 0, "boundary_QA": 0, "attempt_coverage": coverage, "updated_score_rows": len(member_keys),
        "unchanged_score_rows": len(score) - len(member_keys), "unchanged_score_objects_verified": True, "uniform_GT_complete": 0,
        "uniform_GT_pending": len(references), "raw_reopened": False, "actual_root_review_QA": len(packet),
        "actual_root_decision_path": str(decision_path.relative_to(ROOT)), "actual_root_decision_sha256": file_hash(decision_path),
        "root_ruling_preparer_sha256": file_hash(Path(__file__)), "previous_score_dir": str(folder.relative_to(ROOT)),
        "previous_finite_execution_receipt_sha256": file_hash(folder / "finite_execution_receipt.json"),
        "outputs": {p.name: file_hash(p) for p in output.iterdir()}}
    save_json(output / "finite_execution_receipt.json", result)
    save_json(review_output / "receipt.json", {"schema": "kdm_remaining4_reference_actual_root_review_v1", "passed": True,
        "actual_root_author": AUTHOR, "prepared_by": "/root/assets", "QA": len(packet), "source_members_verified": len(member_keys),
        "multiple_primary_QA": sum(r["multiple_primary"] for r in root_decisions), "root_decision_sha256": file_hash(decision_path),
        "source_packet_sha256": queue_sha, "source_luna_sha256": luna_sha, "source_CPU_verification_sha256": file_hash(proof_path),
        "raw_reopened": False, "score_receipt_sha256": file_hash(output / "finite_execution_receipt.json")})
    print(json.dumps({key: result[key] for key in ("rows", "updated_score_rows", "unchanged_score_rows", "canonical_pending", "literal_pending", "abstain_pending", "uniform_GT_pending")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "apply"))
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--luna-dir")
    parser.add_argument("--root-rulings")
    parser.add_argument("--review-output")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(within(ROOT, args.score_dir), within(ROOT, args.output))
    else:
        if not all((args.luna_dir, args.root_rulings, args.review_output)):
            parser.error("Actual Luna, explicit root rulings and exclusive review output are required")
        apply_rulings(within(ROOT, args.score_dir), within(ROOT, args.luna_dir), within(ROOT, args.root_rulings),
                      within(ROOT, args.review_output), within(ROOT, args.output))


if __name__ == "__main__":
    main()
