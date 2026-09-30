#!/usr/bin/env python3
"""Save the parent's actual 69 Phi rulings and update their 74 verified rows."""
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
from workflows.supplemental.remaining4.score_native import save_json, save_rows

AUTHOR = {"agent": "/root", "model": "gpt-6.1-sol", "effort": "max",
          "session_id": "01a0f0e4-4fe9-7940-b763-e0b428e5d5d8", "call_id": ""}
QUEUE_SHA = "696152a34bb0a2c74cf41942104555f6e5b60a51c044631a6c2564c0436a667d"
LUNA_SHA = "c653ac4938c4ccc4f23f513567a34815d1bb26b96784682c5d7d2cbc07facf66"

# The indices and semantic outcomes below are the parent's actual full-QA review.
# Components and spans are copied literally from the source answers.
MULTIPLE = {
 2: ("Sashimi and noodles", ["Sashimi", "noodles"], "coequal", "Sashimi与noodles为同级枚举，回复没有明确主配菜关系。"),
 4: ("seared steak, roasted butternut squash, grilled mushrooms", ["seared steak", "roasted butternut squash", "grilled mushrooms"], "coequal", "plated meal consisting列表没有明确主配菜关系，保留多个同级食物。"),
 5: ("Steak and mashed potatoes with gravy", ["Steak", "mashed potatoes"], "coequal", "Steak与mashed potatoes由and同级并列，gravy为with引入的附属。"),
 7: ("flatbread or pizza", ["flatbread", "pizza"], "competing", "flatbread与pizza为同级具体替代候选。"),
 10: ("grilled salmon, mashed potatoes, and steamed broccoli", ["grilled salmon", "mashed potatoes", "steamed broccoli"], "coequal", "plate of food consisting列表没有明确主配菜关系。"),
 16: ("Omelette and toast", ["Omelette", "toast"], "coequal", "Omelette与toast由and同级并列。"),
 17: ("French toast and sautéed mushrooms", ["French toast", "sautéed mushrooms"], "coequal", "French toast与sautéed mushrooms由and同级并列。"),
 21: ("Steak and egg with fries", ["Steak", "egg"], "coequal", "Steak与egg同级并列，fries为with引入的附属。"),
 24: ("grilled pork chop, snow peas, cauliflower, and carrots", ["grilled pork chop", "snow peas", "cauliflower", "carrots"], "coequal", "plate of food consisting列表没有明确主配菜关系。"),
 27: ("chocolate mousse or pudding", ["chocolate mousse", "pudding"], "competing", "chocolate mousse与pudding为同级具体替代候选。"),
 28: ("Ribs and French fries", ["Ribs", "French fries"], "coequal", "Ribs与French fries由and同级并列。"),
 29: ("Waffle cookies and ice cream", ["Waffle cookies", "ice cream"], "coequal", "Waffle cookies与ice cream由and同级并列。"),
 31: ("Steak and fries with gravy and corn", ["Steak", "fries"], "coequal", "Steak与fries同级并列，gravy和corn由with引入。"),
 32: ("steak, mashed potatoes, and coleslaw", ["steak", "mashed potatoes", "coleslaw"], "coequal", "Plate of food consisting列表没有明确主配菜关系。"),
 38: ("poached fruit, a scoop of ice cream", ["poached fruit", "a scoop of ice cream"], "coequal", "dessert plated列表同时列出poached fruit与ice cream，没有明确唯一主菜。"),
 41: ("shrimp, scallops, and possibly chicken or another type of white meat", ["shrimp", "scallops", "chicken", "another type of white meat"], "coequal", "dish consisting列表列出shrimp、scallops及肉类替代候选，没有明确唯一主菜。"),
 44: ("grilled steak, brown rice, steamed vegetables, and a gravy", ["grilled steak", "brown rice", "steamed vegetables", "a gravy"], "coequal", "plate consisting列表没有明确主配菜关系。"),
 47: ("raw minced meat, likely hamburger or ground beef", ["hamburger", "ground beef"], "competing", "hamburger与ground beef为具体竞争名称，不能按规范词选择hamburger。"),
 49: ("poached egg, a round dumpling or gnocchi", ["poached egg", "round dumpling", "gnocchi"], "coequal", "gourmet dish consisting列表存在同级食物和具体替代候选，没有唯一主菜。"),
 56: ("French toast and crispy fried onions", ["French toast", "crispy fried onions"], "coequal", "French toast与crispy fried onions由and同级并列。"),
 57: ("Ice cream and toast with strawberry jam", ["Ice cream", "toast"], "coequal", "Ice cream与toast同级并列，strawberry jam为with引入的附属。"),
 58: ("grilled salmon, steamed broccoli, sliced carrots, lemon wedges", ["grilled salmon", "steamed broccoli", "sliced carrots", "lemon wedges"], "coequal", "plate containing列表没有明确主配菜关系。"),
 62: ("one appears to be a pork chop and the other a lamb chop", ["pork chop", "lamb chop"], "coequal", "回复明确列出两种肉，pork chop与lamb chop为同级主菜。"),
 65: ("cheese omelette or a cheese-baked egg dish", ["cheese omelette", "cheese-baked egg dish"], "competing", "cheese omelette与cheese-baked egg dish为具体竞争候选。"),
}
UNIQUE_OVERRIDES = {
 6: ("cooked beef, likely a steak cut", "cooked beef, likely a steak cut", "同一cooked beef的进一步识别为likely a steak cut，保留完整明确名称及语气。"),
 15: ("lasagna", "lasagna", "lasagna为唯一明确具体名称，similar type of casserole为宽泛上位描述。"),
 22: ("mussels", "mussels", "mussels为实际主菜，accompanied by引入grilled cheese配菜；seafood dish为泛称。"),
 23: ("steamed dumplings", "steamed dumplings", "已明确steamed dumplings，xiaolongbao和soup dumplings是对此名称的细化。"),
 37: ("sushi roll", "sushi roll, specifically a California roll", "sushi roll为已明确的主菜，specifically California roll为细化，保留连续完整限定span。"),
 42: ("breaded and fried calamari rings", "breaded and fried calamari rings", "保留完整主菜breaded and fried calamari rings及所有修饰。"),
}
UNIQUE_DETAILS = {
 15: [("similar type of casserole", "description")],
 22: [("seafood dish", "description"), ("slices of lemon", "ingredient"), ("herbs", "ingredient"), ("grilled cheese sand", "side")],
 23: [("xiaolongbao", "description"), ("soup dumplings", "description")],
 37: [("California roll", "description")],
 42: [("lime wedge", "side"), ("cherry tomato", "side")],
}


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
    source_proof_path = previous / "received_source_verification.json"
    prior_receipt = json.loads((previous / "receipt.json").read_text())
    prior_verify = json.loads((previous / "verification.json").read_text())
    source_proof = json.loads(source_proof_path.read_text())
    if (prior_verify["passed"] is not True or prior_receipt["rows"] != 3960
            or prior_receipt["canonical_pending"] != 74 or prior_receipt["literal_pending"] != 74
            or prior_receipt["abstain_pending"] != 0
            or file_hash(score_path) != prior_receipt["score_rows_sha256"]
            or file_hash(score_path) != prior_verify["score_rows_sha256"]
            or file_hash(source_proof_path) != prior_verify["source_verification_sha256"]
            or file_hash(queue_path) != QUEUE_SHA or source_proof["passed"] is not True
            or prior_receipt["canonical_scorer_sha256"] != file_hash(Path(frozen.__file__))):
        raise ValueError("Finite actual CPU score, source proof or frozen scorer binding differs")
    actual_luna = list(rows(luna / "decisions.jsonl"))
    luna_receipt = json.loads((luna / "execution_receipt.json").read_text())
    packet = [record for _line, record, _sha in rows(queue_path)]
    if (len(actual_luna) != 69 or len(packet) != 69 or len(MULTIPLE) != 24
            or luna_receipt["rows"] != 69 or luna_receipt["unique_qa_keys"] != 69
            or luna_receipt["memberships"] != 74 or luna_receipt["abstain_count"] != 0
            or luna_receipt["sha256"] != LUNA_SHA or file_hash(luna / "decisions.jsonl") != LUNA_SHA
            or luna_receipt["annotation_model"] != "gpt-5.6-luna" or luna_receipt["effort"] != "medium"
            or luna_receipt["source_memberships_preserved"] is not True):
        raise ValueError("Actual successful Luna decision writes or finite packet identity differs")
    records = [record for _line, record, _sha in rows(score_path)]
    by_key = {record["key"]: record for record in records}
    parts = {(item["claim_id"], item["part"]): item for item in source_proof["parts"]}
    if len(records) != 3960 or len(by_key) != 3960 or len(parts) != 10:
        raise ValueError("Finite score keys or source part count differs")
    root_decisions, member_proofs, changed_keys = [], [], set()
    for index, (item, actual) in enumerate(zip(packet, actual_luna), 1):
        line, luna_decision, luna_line_sha = actual
        qkey, question, answer = item["qa_key"], item["question"], item["answer"]
        if (line != index or luna_decision["qa_key"] != qkey or qkey != frozen.qah(question, answer)
                or question != "What specific food is shown in this image?"
                or (luna_decision["question"], luna_decision["answer"]) != (question, answer)
                or luna_decision["source_memberships"] != item["memberships"]
                or luna_decision["annotation_model"] != "gpt-5.6-luna" or luna_decision["effort"] != "medium"):
            raise ValueError("Actual complete question, answer, packet order or Luna memberships differ")
        multiple = index in MULTIPLE
        if multiple:
            span, names, role, reason = MULTIPLE[index]
            primary, full_name = None, None
            relations = [{"name": name, "span": name, "role": role} for name in names]
        else:
            primary = luna_decision["extracted_name"]
            full_name = luna_decision["fullspan_primary_name"]
            span = luna_decision["primary_answer_span"]
            reason = "实际root完整复核保留Luna唯一主菜名称：" + primary + "。"
            if index in UNIQUE_OVERRIDES:
                primary, full_name, reason = UNIQUE_OVERRIDES[index]
                span = full_name
            names = [full_name]
            relations = [{"name": full_name, "span": full_name, "role": "main"}]
            relations += [{"name": name, "span": name, "role": role} for name, role in UNIQUE_DETAILS.get(index, [])]
        for text in [span, *names, *[relation["span"] for relation in relations]]:
            if not isinstance(text, str) or not text or text not in answer:
                raise ValueError("Actual root answer span is not continuous original text")
        decision = {**luna_decision, "qa_key": qkey, "question": question, "answer": answer,
            "queue_index": index, "abstain": False, "needs_root": False, "root_review_required": False,
            "multiple_primary": multiple, "status": "multiple_primary" if multiple else "resolved_primary",
            "extracted_name": primary, "primary_answer_span": span, "fullspan_primary_name": full_name,
            "literal_full_name": full_name, "endorsed_primary_names": names, "name_relations": relations,
            "name_scope_ambiguous": False, "canonical_override": None,
            "root_reason": reason, "reason": reason, "root_review_author": AUTHOR,
            "actual_author": AUTHOR, "prepared_by": "/root/assets", "annotation_model": AUTHOR["model"],
            "annotation_effort": AUTHOR["effort"], "effort": AUTHOR["effort"],
            "annotation_author": AUTHOR["agent"], "annotation_agent": AUTHOR["agent"],
            "annotation_session_id": AUTHOR["session_id"], "session_id": AUTHOR["session_id"],
            "annotation_call_id": "", "call_id": "", "source_luna_decision": luna_decision,
            "source_luna_path": str((luna / "decisions.jsonl").relative_to(ROOT)),
            "source_luna_sha256": LUNA_SHA, "source_luna_line": line, "source_luna_line_sha256": luna_line_sha,
            "source_memberships": item["memberships"], "source_packet_path": str(queue_path.relative_to(ROOT)),
            "source_packet_sha256": QUEUE_SHA,
            "source_CPU_verification_path": str((previous / "verification.json").relative_to(ROOT)),
            "source_CPU_verification_sha256": file_hash(previous / "verification.json"), "raw_reopened": False}
        for member in item["memberships"]:
            record = by_key[member["key"]]
            part = parts[record["source_claim"], record["source_part"]]
            fields = ("model", "method", "sample_id", "target_class", "key", "source_path", "source_line",
                      "source_identity", "raw_line_sha256", "source_claim", "source_part")
            if (any(member[field] != record[field] for field in fields)
                    or record["model"] != "phi35" or record["qa_key"] != qkey
                    or (record["question"], record["answer"]) != (question, answer)
                    or record["source_identity"] != part["actual_source_identity"]
                    or record["raw_source_sha256"] != part["files"]["raw"]["sha256"]
                    or record["source_path"] != part["files"]["raw"]["received_path"]
                    or record["seed"] != stable_seed(record["sample_id"], "phi35", 0)
                    or record["canonical_name_in_primary_score"] is not None
                    or record["literal_extracted_name_score"] is not None or record["key"] in changed_keys):
                raise ValueError("Root member differs from actual CPU-verified source key, SHA, identity, seed or QA")
            changed_keys.add(record["key"])
            member_proofs.append({**member, "qa_key": qkey, "seed": record["seed"],
                "raw_source_sha256": record["raw_source_sha256"], "CPU_source_proof_sha256": file_hash(source_proof_path),
                "binding_verification": "prior_actual_raw_CPU_proof_to_verified_score_to_exact_QA_packet",
                "raw_reopened": False})
        root_decisions.append(decision)
    if len(changed_keys) != 74 or len(root_decisions) != 69:
        raise ValueError("Finite root review does not cover exactly 69 QA and 74 source members")
    review_output.mkdir(parents=True, exist_ok=False)
    decision_path = review_output / "root_reviewed.jsonl"
    save_rows(decision_path, root_decisions)
    save_rows(review_output / "source_member_verification.jsonl", member_proofs)
    save_json(review_output / "receipt.json", {"schema": "kdm_remaining4_root69_review_receipt_v1", "passed": True,
        "actual_root_author": AUTHOR, "prepared_by": "/root/assets", "QA": 69, "source_members": 74,
        "multiple_primary_QA": 24, "unique_primary_QA": 45, "canonical_override_nonnull": 0,
        "question_answer_identity_verified": True, "raw_reopened": False,
        "source_binding_scope": "reused_actual_raw_verified_CPU_hash_key_identity_seed_QA_chain",
        "queue_sha256": QUEUE_SHA, "source_score_sha256": file_hash(score_path),
        "luna_decision_sha256": LUNA_SHA, "root_decision_sha256": file_hash(decision_path),
        "source_member_proof_sha256": file_hash(review_output / "source_member_verification.jsonl"),
        "preparer_sha256": file_hash(Path(__file__)), "new_API_calls": 0, "GPU_initialized": False})
    decisions = {item["qa_key"]: {**item, "decision_source_path": str(decision_path), "decision_source_line": index}
                 for index, item in enumerate(root_decisions, 1)}
    classes = sorted({record["target_class"] for record in records})
    if len(classes) != 101:
        raise ValueError("Registered 101-class canonical target set differs")
    patterns = frozen.compile_classes(classes)
    updated, unchanged = [], 0
    for original in records:
        if original["key"] not in changed_keys:
            updated.append(original)
            unchanged += 1
            continue
        inferred = infer_qa(original["question"], original["answer"], patterns, {}, {}, decisions)
        canonical, literal, reason = score_target(original["answer"], original["target_class"], inferred, patterns)
        if canonical is None or literal is None or inferred["abstain"] is not False:
            raise ValueError("Actual root decision left a field unresolved")
        state = frozen.extract(inferred["decision"], patterns)
        if inferred["decision"]["multiple_primary"] and (not state["multi"] or canonical != 0 or literal != 0):
            raise ValueError("Frozen source scoring did not respect actual multiple primary roles")
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
    if unchanged != 3886 or any(before != after for before, after in zip(records, updated) if before["key"] not in changed_keys):
        raise ValueError("Finite root update changed an unrelated score object")
    counts = [item for _line, item, _sha in rows(previous / "condition_counts.jsonl")]
    for item in counts:
        selected = [r for r in updated if (r["model"], r["method"]) == (item["model"], item["method"])]
        if len(selected) != item["received_n"] or Counter(r["target_class"] for r in selected) != item["received_category_counts"]:
            raise ValueError("Root update changed source keys or registered class quotas")
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
    shutil.copyfile(source_proof_path, output / source_proof_path.name)
    receipt = {**prior_receipt, "completed_utc": datetime.now(timezone.utc).isoformat(), "boundary_QA": 0,
        "canonical_pending": 0, "literal_pending": 0, "abstain_pending": 0, "condition_counts": counts,
        "behavior_source_counts": dict(Counter(r["behavior_source"] for r in updated)),
        "score_rows_sha256": file_hash(output / "score_rows.jsonl.gz"), "scorer_sha256": file_hash(Path(__file__)),
        "source_scorer_sha256": prior_receipt["scorer_sha256"], "incremental_QA_update_rows": 74,
        "incremental_review_QA": 69, "unchanged_score_rows": unchanged, "unchanged_score_objects_verified": True,
        "raw_reopened": False, "incremental_source_score_path": str(score_path.relative_to(ROOT)),
        "incremental_source_score_sha256": file_hash(score_path),
        "actual_root_decision_path": str(decision_path.relative_to(ROOT)), "actual_root_decision_sha256": file_hash(decision_path)}
    receipt["authority_inputs"] = {**receipt["authority_inputs"], "appended_root69":
        {"path": str(decision_path.relative_to(ROOT)), "sha256": file_hash(decision_path)}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({"review_QA": 69, "source_members_verified": 74, "multiple_primary_QA": 24,
        "unique_primary_QA": 45, "updated_score_rows": 74, "unchanged_score_rows": unchanged, "rows": 3960,
        "canonical_pending": 0, "literal_pending": 0, "abstain_pending": 0, "raw_reopened": False,
        "root_decisions": str(decision_path.relative_to(ROOT)), "score_output": str(output.relative_to(ROOT))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
