#!/usr/bin/env python3
"""Save actual root tail-eight rulings and update only their verified source rows."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, stable_seed, within
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.score import infer_qa, rows, score_target
from workflows.supplemental.remaining4.review_phi_boundary69 import AUTHOR
from workflows.supplemental.remaining4.score_native import save_json, save_rows

QUEUE_SHA = "d7c99b3d858178882f6e0c6b0c77c35f0c2577b8a4c9f3cb4010666e634e578c"
LUNA_SHA = "0d2f73602e1c4f995cda2a4b561952f0cc414a3d9ee379142af0ae3755699a5a"
# Actual parent root rulings, in the original complete-QA packet order.
RULINGS = [
 ("1daa598ec5dcc643b5aa432892e107604d28bef94a12556593634fea67f22109", "seared scallops", ["seared scallops"], False, "seared scallops为明确主菜，served over引入的grains和vegetables为附属。"),
 ("22a18bd74714d71d056dbb1dd0c33da81e70746199344a15af6cf4e4463688a7", "Beef steak and asparagus", ["Beef steak", "asparagus"], True, "Beef steak与asparagus由and同级并列，没有明确主配菜关系。"),
 ("3b9a1896b72b006575d60e8b2cfc22deff5fd0902d599395e70a12949d0397ed", "grilled steak", ["grilled steak"], False, "grilled steak为明确主菜，ribeye为切块细化说明，boiled potatoes为side。"),
 ("544950cb0dae78f9f6b648e1127400049bde763b312e3402da33d13bb92fd115", "grilled steak, a mound of rice, and some vegetables including carrots", ["grilled steak", "a mound of rice", "some vegetables including carrots"], True, "plate of food consisting列表没有明确主配菜关系，不能选第一个食物。"),
 ("6cdc9b742baf8ba3ba214ceaa2aebdfe3064ca653142640f13fe40e0395364ea", "grilled steak", ["grilled steak"], False, "grilled steak为明确主菜，rosemary为garnish。"),
 ("95b6fcaa52dac1ec8e9ebc7324a1ede2c618867e40e3aa8960b9951919fbf6e0", "fresh spring rolls", ["fresh spring rolls"], False, "已明确fresh spring rolls，summer rolls和Vietnamese spring rolls为对此名称的细化说明；保留原spring rolls词。"),
 ("a25573e3fe54af4c0a7d2fdd43bd4fc508656ffdf3ce932e23820086dde9e094", "pancake or crepe, a scoop of ice cream", ["pancake", "crepe", "a scoop of ice cream"], True, "pancake与crepe为具体竞争候选，并与ice cream同级，未明确唯一主菜。"),
 ("cd0bf8a841f6e1d174464ac05d178b0a16b3412e890673a3f62908ac1ae699ea", "fried dumplings", ["fried dumplings"], False, "已明确fried dumplings，vegetable和potstickers为填馅或类型说明，green onions为garnish。"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--luna-dir", required=True)
    parser.add_argument("--review-output", required=True)
    parser.add_argument("--score-output", required=True)
    args = parser.parse_args()
    previous, luna = within(ROOT, args.score_dir), within(ROOT, args.luna_dir)
    review_output, output = within(ROOT, args.review_output), within(ROOT, args.score_output)
    score_path, queue_path = previous / "score_rows.jsonl.gz", previous / "boundary_queue.jsonl"
    source_path = previous / "received_source_verification.json"
    prior_receipt = json.loads((previous / "receipt.json").read_text())
    prior_verify = json.loads((previous / "verification.json").read_text())
    source_proof = json.loads(source_path.read_text())
    if (prior_verify["passed"] is not True or prior_receipt["rows"] != 872
            or prior_receipt["canonical_pending"] != 8 or prior_receipt["literal_pending"] != 8
            or prior_receipt["abstain_pending"] != 0
            or file_hash(score_path) != prior_receipt["score_rows_sha256"]
            or file_hash(score_path) != prior_verify["score_rows_sha256"]
            or file_hash(source_path) != prior_verify["source_verification_sha256"]
            or file_hash(queue_path) != QUEUE_SHA or source_proof["passed"] is not True
            or prior_receipt["canonical_scorer_sha256"] != file_hash(Path(frozen.__file__))):
        raise ValueError("Finite actual CPU score, source proof or frozen scorer binding differs")
    packet = [item for _line, item, _sha in rows(queue_path)]
    actual_luna = list(rows(luna / "decisions.jsonl"))
    luna_receipt = json.loads((luna / "execution_receipt.json").read_text())
    if (len(packet) != 8 or len(actual_luna) != 8 or luna_receipt["queue_sha256"] != QUEUE_SHA
            or luna_receipt["rows"] != 8 or luna_receipt["unique_qa_keys"] != 8
            or luna_receipt["source_memberships"] != 8 or luna_receipt["abstain_count"] != 0
            or luna_receipt["sha256"] != LUNA_SHA or file_hash(luna / "decisions.jsonl") != LUNA_SHA
            or luna_receipt["annotation_model"] != "gpt-5.6-luna" or luna_receipt["effort"] != "medium"
            or luna_receipt["source_memberships_preserved"] is not True):
        raise ValueError("Actual successful Luna write receipt or original packet binding differs")
    records = [record for _line, record, _sha in rows(score_path)]
    by_key = {record["key"]: record for record in records}
    parts = {(item["claim_id"], item["part"]): item for item in source_proof["parts"]}
    if len(records) != 872 or len(by_key) != 872 or len(parts) != 2:
        raise ValueError("Finite tail score keys or source parts differ")
    root_decisions, member_proofs, changed_keys = [], [], set()
    for index, (item, actual, ruling) in enumerate(zip(packet, actual_luna, RULINGS), 1):
        line, luna_decision, luna_line_sha = actual
        qkey, span, names, multiple, reason = ruling
        question, answer = item["question"], item["answer"]
        if (item["qa_key"] != qkey or luna_decision["qa_key"] != qkey or line != index
                or qkey != frozen.qah(question, answer) or question != "What specific food is shown in this image?"
                or (luna_decision["question"], luna_decision["answer"]) != (question, answer)
                or luna_decision["source_memberships"] != item["memberships"]):
            raise ValueError("Actual root order, full QA or Luna memberships differ")
        for text in [span, *names]:
            if not text or text not in answer:
                raise ValueError("Actual root span is not continuous original text")
        primary = None if multiple else names[0]
        relations = [{"name": name, "span": name, "role": "coequal" if multiple else "main"} for name in names]
        if index == 7:
            relations[0]["role"] = relations[1]["role"] = "competing"
        details = {1: [("grains", "side"), ("vegetables", "side"), ("herbs", "side")],
                   3: [("ribeye", "description"), ("boiled potatoes", "side")],
                   5: [("rosemary", "side")],
                   6: [("summer rolls", "description"), ("Vietnamese spring rolls", "description")],
                   8: [("vegetable", "description"), ("potstickers", "description"), ("chopped green onions", "side")]}
        relations += [{"name": text, "span": text, "role": role} for text, role in details.get(index, [])]
        if any(relation["span"] not in answer for relation in relations):
            raise ValueError("Root relation contains a non-original span")
        decision = {**luna_decision, "queue_index": index, "abstain": False, "needs_root": False,
            "root_review_required": False, "multiple_primary": multiple,
            "status": "multiple_primary" if multiple else "resolved_primary", "extracted_name": primary,
            "primary_answer_span": span, "fullspan_primary_name": primary, "literal_full_name": primary,
            "endorsed_primary_names": names, "name_relations": relations,
            "name_scope_ambiguous": False, "canonical_override": None, "root_reason": reason, "reason": reason,
            "root_review_author": AUTHOR, "actual_author": AUTHOR, "prepared_by": "/root/assets",
            "annotation_model": AUTHOR["model"], "annotation_effort": AUTHOR["effort"], "effort": AUTHOR["effort"],
            "annotation_author": AUTHOR["agent"], "annotation_agent": AUTHOR["agent"],
            "annotation_session_id": AUTHOR["session_id"], "session_id": AUTHOR["session_id"],
            "annotation_call_id": "", "call_id": "", "source_luna_decision": luna_decision,
            "source_luna_path": str((luna / "decisions.jsonl").relative_to(ROOT)), "source_luna_sha256": LUNA_SHA,
            "source_luna_line": line, "source_luna_line_sha256": luna_line_sha,
            "source_packet_path": str(queue_path.relative_to(ROOT)), "source_packet_sha256": QUEUE_SHA,
            "source_CPU_verification_path": str((previous / "verification.json").relative_to(ROOT)),
            "source_CPU_verification_sha256": file_hash(previous / "verification.json"), "raw_reopened": False}
        for member in item["memberships"]:
            record = by_key[member["key"]]
            part = parts[record["source_claim"], record["source_part"]]
            fields = ("model", "method", "sample_id", "target_class", "key", "source_path", "source_line",
                      "source_identity", "raw_line_sha256", "source_claim", "source_part")
            if (any(member[field] != record[field] for field in fields)
                    or record["model"] != "phi35" or record["method"] != "m3id" or record["qa_key"] != qkey
                    or (record["question"], record["answer"]) != (question, answer)
                    or record["source_identity"] != part["actual_source_identity"]
                    or record["raw_source_sha256"] != part["files"]["raw"]["sha256"]
                    or record["source_path"] != part["files"]["raw"]["received_path"]
                    or record["seed"] != stable_seed(record["sample_id"], "phi35", 0)
                    or record["canonical_name_in_primary_score"] is not None
                    or record["literal_extracted_name_score"] is not None or record["key"] in changed_keys):
                raise ValueError("Root member differs from actual CPU key, SHA, identity, seed or QA proof")
            changed_keys.add(record["key"])
            member_proofs.append({**member, "qa_key": qkey, "seed": record["seed"],
                "raw_source_sha256": record["raw_source_sha256"], "CPU_source_proof_sha256": file_hash(source_path),
                "binding_verification": "prior_actual_raw_CPU_proof_to_verified_score_to_exact_QA_packet",
                "raw_reopened": False})
        root_decisions.append(decision)
    if len(changed_keys) != 8:
        raise ValueError("Finite root review does not cover exactly eight source rows")
    review_output.mkdir(parents=True, exist_ok=False)
    decision_path = review_output / "root_reviewed.jsonl"
    save_rows(decision_path, root_decisions)
    save_rows(review_output / "source_member_verification.jsonl", member_proofs)
    save_json(review_output / "receipt.json", {"schema": "kdm_remaining4_root_tail8_review_receipt_v1", "passed": True,
        "actual_root_author": AUTHOR, "prepared_by": "/root/assets", "QA": 8, "source_members": 8,
        "multiple_primary_QA": 3, "unique_primary_QA": 5, "canonical_override_nonnull": 0,
        "question_answer_identity_verified": True, "raw_reopened": False,
        "source_binding_scope": "reused_actual_raw_verified_CPU_hash_key_identity_seed_QA_chain",
        "queue_sha256": QUEUE_SHA, "source_score_sha256": file_hash(score_path), "luna_decision_sha256": LUNA_SHA,
        "root_decision_sha256": file_hash(decision_path),
        "source_member_proof_sha256": file_hash(review_output / "source_member_verification.jsonl"),
        "preparer_sha256": file_hash(Path(__file__)), "GPU_initialized": False, "new_API_calls": 0})
    decisions = {item["qa_key"]: {**item, "decision_source_path": str(decision_path), "decision_source_line": index}
                 for index, item in enumerate(root_decisions, 1)}
    classes = sorted({sample["class"] for _line, sample, _sha in rows(ROOT / "data/current/all.jsonl") if sample["dataset"] == "food101"})
    if len(classes) != 101:
        raise ValueError("Registered canonical class set differs")
    patterns = frozen.compile_classes(classes)
    updated, unchanged = [], 0
    for original in records:
        if original["key"] not in changed_keys:
            updated.append(original)
            unchanged += 1
            continue
        inferred = infer_qa(original["question"], original["answer"], patterns, {}, {}, decisions)
        canonical, literal, reason = score_target(original["answer"], original["target_class"], inferred, patterns)
        state = frozen.extract(inferred["decision"], patterns)
        if canonical is None or literal is None or inferred["abstain"] is not False:
            raise ValueError("Actual root decision left a field unresolved")
        if inferred["decision"]["multiple_primary"] and (not state["multi"] or canonical != 0 or literal != 0):
            raise ValueError("Frozen scoring failed to respect actual multiple primary roles")
        updated.append({**original, "canonical_name_in_primary_score": canonical,
            "literal_extracted_name_score": literal, "literal_extracted_names": state["literal"], "abstain": False,
            "score_reason": reason, "behavior_source": inferred["behavior_source"], "decision_source": inferred["decision"],
            "annotation_model": AUTHOR["model"], "annotation_effort": AUTHOR["effort"], "annotation_call_id": "",
            "semantic_primary_extraction": {"primary_name": inferred["decision"]["fullspan_primary_name"],
                "status": inferred["decision"]["status"], "rule": "actual_root_complete_QA_role_review"},
            "previous_score_fields": {field: original[field] for field in
                ("canonical_name_in_primary_score", "literal_extracted_name_score", "literal_extracted_names",
                 "abstain", "score_reason", "behavior_source", "decision_source")},
            "previous_score_source_path": str(score_path.relative_to(ROOT)),
            "previous_score_source_sha256": file_hash(score_path)})
    if unchanged != 864 or any(before != after for before, after in zip(records, updated) if before["key"] not in changed_keys):
        raise ValueError("Finite root update changed an unrelated score object")
    counts = [item for _line, item, _sha in rows(previous / "condition_counts.jsonl")]
    for item in counts:
        selected = [r for r in updated if (r["model"], r["method"]) == (item["model"], item["method"])]
        if len(selected) != item["received_n"] or Counter(r["target_class"] for r in selected) != item["received_category_counts"]:
            raise ValueError("Finite root update changed source coverage or class quotas")
        item.update(canonical_correct=sum(r["canonical_name_in_primary_score"] == 1 for r in selected),
                    canonical_incorrect=sum(r["canonical_name_in_primary_score"] == 0 for r in selected),
                    canonical_pending=sum(r["canonical_name_in_primary_score"] is None for r in selected),
                    literal_correct=sum(r["literal_extracted_name_score"] == 1 for r in selected),
                    literal_pending=sum(r["literal_extracted_name_score"] is None for r in selected),
                    abstain_true=sum(r["abstain"] is True for r in selected),
                    abstain_false=sum(r["abstain"] is False for r in selected),
                    abstain_pending=sum(r["abstain"] is None for r in selected))
    if any(item["canonical_pending"] or item["literal_pending"] or item["abstain_pending"] for item in counts):
        raise ValueError("Actual root review failed to close its finite source score")
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "score_rows.jsonl.gz", updated)
    save_rows(output / "boundary_queue.jsonl", [])
    save_rows(output / "condition_counts.jsonl", counts)
    shutil.copyfile(source_path, output / source_path.name)
    receipt = {**prior_receipt, "completed_utc": datetime.now(timezone.utc).isoformat(), "boundary_QA": 0,
        "canonical_pending": 0, "literal_pending": 0, "abstain_pending": 0, "condition_counts": counts,
        "behavior_source_counts": dict(Counter(r["behavior_source"] for r in updated)),
        "score_rows_sha256": file_hash(output / "score_rows.jsonl.gz"), "scorer_sha256": file_hash(Path(__file__)),
        "source_scorer_sha256": prior_receipt["scorer_sha256"], "incremental_QA_update_rows": 8,
        "incremental_review_QA": 8, "unchanged_score_rows": 864, "unchanged_score_objects_verified": True,
        "raw_reopened": False, "incremental_source_score_path": str(score_path.relative_to(ROOT)),
        "incremental_source_score_sha256": file_hash(score_path),
        "actual_root_decision_path": str(decision_path.relative_to(ROOT)), "actual_root_decision_sha256": file_hash(decision_path)}
    receipt["authority_inputs"] = {**receipt["authority_inputs"], "appended_root_tail8":
        {"path": str(decision_path.relative_to(ROOT)), "sha256": file_hash(decision_path)}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({"review_QA": 8, "source_members_verified": 8, "multiple_primary_QA": 3,
        "unique_primary_QA": 5, "updated_score_rows": 8, "unchanged_score_rows": 864, "rows": 872,
        "canonical_pending": 0, "literal_pending": 0, "abstain_pending": 0, "raw_reopened": False,
        "root_decisions": str(decision_path.relative_to(ROOT)), "score_output": str(output.relative_to(ROOT))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
