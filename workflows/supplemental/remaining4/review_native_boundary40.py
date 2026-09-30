#!/usr/bin/env python3
"""Persist the actual root 40-QA rulings and update only their verified score rows."""
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

# These semantic roles and reasons are the parent's actual complete-QA rulings.
# Exact QA key, primary continuous spans, side continuous spans, multiple, reason.
RULINGS = [
 ("0cbcec4d0ca312d4a91d18f5b14ee718932244c510c2fac4be2437703f3080b8", ["Shrimp", "pancakes"], [], True, "Shrimp与pancakes同级，唯一主菜未成立。"),
 ("0ee037e176db2de4cb7eb576f9e9e0435aa49ca394ac6ca0db875c9749c719d1", ["Steak"], ["rice", "carrots"], False, "Steak为主菜，rice与carrots为配菜。"),
 ("1971c94e272289bb64e50f885000883764a26a64582caa524527f2b69446c439", ["Breakfast burrito", "breakfast enchilada"], ["beans", "cheese", "salsa", "sour cream", "orange slice"], True, "Breakfast burrito与breakfast enchilada为竞争具体主菜，其余列表为配属。"),
 ("19ffc7a8502fba38089a3146efa6a914577cd1dd60d5bd8fb9c677b98a6afc5e", ["Salmon"], ["French fries", "lemon"], False, "Salmon为主菜，French fries与lemon为配菜；保留原文名称。"),
 ("27d2ad57878f36a6e8c333d7e0c5660fc4a29fe7892ef3872f052e71035f9df5", ["French fries", "apples"], [], True, "French fries与apples同级，未明确主配层级。"),
 ("30b47e5b5482d2408cacf6d96b1d643665163ff63377400b2e20acff739256a6", ["Steak"], ["asparagus", "fries"], False, "Steak为主菜，asparagus与fries为配菜。"),
 ("32455b403527e1d5fd209f338d22bd42ca9e4e064f0ed5d2850d4507934c7030", ["Steak"], ["onions"], False, "Steak为主菜，onions为配菜。"),
 ("34971b6db9ab0d55f10304d01fecc051089f386e7d0d47532928630ca2827753", ["Shrimp and mussel omelette"], ["poached egg"], False, "Shrimp和mussel为同一omelette的成分，完整主菜名称为Shrimp and mussel omelette，poached egg为副菜。"),
 ("43ab719feb3a470f01d9a60f16d05b60518ea201cd44b9f670d1b4e4131e9a5a", ["pulled pork sandwich"], ["waffle fries"], False, "pulled pork sandwich为完整主菜，waffle fries为配菜。"),
 ("46c3b084304e526e2fdd21558d2575e224f4b5b6373a62efcd50ad83c8693716", ["Hamburger"], ["loaded fries"], False, "Hamburger为主菜，loaded fries为配菜。"),
 ("5fa13bca1848812e58f615badb8aa8c495792d97343d6d92a541b72c6c88965d", ["Scallops", "shrimp"], ["spinach"], True, "Scallops与shrimp同级多主菜，spinach为配菜。"),
 ("6bc7b77dde03e08265129729bd25a76e74b6a14dffa1945d52e9db25a945ffad", ["Steak"], ["potatoes", "broccoli", "zucchini", "carrots", "coffee"], False, "Steak为主菜，其余列表为配菜或饮品。"),
 ("732a95bcd0ac11abd40bc3df9f98fcbc13b7eb842a87bb1b3672d6d79f200650", ["omelette", "hash browns"], ["toast"], True, "Breakfast plate为容器描述，omelette与hash browns同级食物，toast为配属。"),
 ("75e7755ac563f45c3a69ffa75ef5498b89c291d78e2d96a046eeb02c08f8af4d", ["Steak"], ["Broccoli"], False, "Steak为主菜，Broccoli为配菜。"),
 ("7aeca4d043786375bc5dedcf07e69d0ceea9f056302e8cc7782ff637a0a723c5", ["Steak burger"], ["baby carrots"], False, "完整主菜为Steak burger，baby carrots为配菜；评分沿用原101词边界。"),
 ("885296133ad525b3ff84d86c24fe30ada2fb1f2c88ee869351e06765c2029bf2", ["Eggs Benedict"], [], False, "Eggs Benedict为唯一明确具体名称，or a similar style为泛称；末句截断未实际撤回答案。"),
 ("89680f9efafe49e47d07dcc0dc79b2bf70cfcc5be8c6311b97ae149d1c1507f4", ["steak"], ["mashed potatoes"], False, "steak为主菜，mashed potatoes为配菜。"),
 ("8dd2935b9863a2fe0a7d0d222648d11795333176cb60060018fe1f69da3c8530", ["Steak"], ["greens"], False, "Steak为主菜，greens为配菜。"),
 ("8e8896fbdfa8b8bc137e9ab46a2c6aebfadba9e08dc1108b1fa8a235d3d3ec8a", ["Sushi", "fried shrimp"], [], True, "Sushi与fried shrimp同级，未明确唯一主菜。"),
 ("8fb5bff343b6c2265b865ea9fd857810e34d1f4b642f1372517d08f710d04229", ["pork chop"], ["Asparagus", "mashed potatoes"], False, "pork chop为主菜，Asparagus与mashed potatoes为配菜。"),
 ("913ae942c238206eace504267fe591fa9f28c3e5c771049d728b3f52bb6da2b8", ["Pepperoni and bacon pizza"], [], False, "Pepperoni和bacon修饰同一pizza，完整主菜名称为Pepperoni and bacon pizza。"),
 ("924a2a5b0a4a370bdaf9370cf69210af1b97bcacaf94e1a88746d28e87bae2f6", ["sandwich"], ["french fries"], False, "sandwich为主菜，french fries为配菜，主菜名称保留实际字形。"),
 ("93dcad4bbbc377403c2f8850f8aac57bd906856dd3f485dd82d686a9d9eeb154", ["meze platter"], ["hummus", "pita bread", "sliced cucumbers", "roasted red peppers", "olives"], False, "meze platter为主菜，featuring列表为成分和描述。"),
 ("9919e34f97e0ec267e6cbe9e3d5ab48cd2352eb894efa4dd9f663905553ac2b5", ["Ice cream", "meat"], [], True, "Ice cream与meat同级多主菜。"),
 ("9fdbbfeb13af5073d90d85f8ec0902f56b863f0537b004934fbcb27c9b85f8aa", ["Chicken", "steak"], [], True, "Chicken与steak同级，未明确主从关系。"),
 ("a65bd50e5b8c248d6cb159ce0a01e4b056c97c668b93c4ed552035e1d9d2a360", ["Pita chips", "hummus"], [], True, "Pita chips与hummus未明确层级，按同级主体处理。"),
 ("bc276e37e1bf086e0ce199d6ce512318c127ea49df6f97426aad458f2a01bddc", ["crab cake"], ["French fries"], False, "crab cake为主菜，French fries为配菜；完整名称保留单数原文。"),
 ("c1a1690f14fa6038e1f76fb2294974a579b029929232b66fd652e848dc5a7a3c", ["fried rice", "meat"], ["Broccoli", "carrots"], True, "fried rice与meat同级多主菜，Broccoli与carrots为配属。"),
 ("c4b6866352d56f48af557c22081f03fd12a7cd0b172ea59eced0fc04922649aa", ["Pork chop"], ["mashed potatoes"], False, "Pork chop为主菜，mashed potatoes为配菜。"),
 ("c55b9ec4b0107ec7c0a17e7d548a0ac8d1ad0d886c307f233ffb74d9ff01a335", ["cheese grilled sandwich"], ["french fries"], False, "完整主菜为cheese grilled sandwich，保留实际词序，french fries为配菜。"),
 ("ca0b29509693eba7d20dc99dcb503b9aa836812b6da3bda51a390970516f37d4", ["Rice", "garlic bread"], [], True, "Rice与garlic bread同级多主菜。"),
 ("cadf7e311fd2a4a4b5377a939278987e89c4f8fcbdd290a2700fc95d402c14c3", ["garlic bread", "croutons"], [], True, "Toasted bread slices为泛称描述，garlic bread与croutons是具体同级替代候选。"),
 ("cc7338f4370d341c0e9eeb3f2e1773e1b4d0269a61716902f448197afa30c726", ["Onion rings", "squid"], [], True, "Onion rings与squid同级，没有明确主配关系。"),
 ("d19d42428a20aa8057c150a3bae4b865227eed4f0841b4d0f7eee1d45d564c72", ["Strawberry and chocolate tiramisu"], [], False, "Strawberry和chocolate为同一tiramisu的双风味，完整名称作为唯一主菜。"),
 ("e2f2350e7f8757e7ec3b2b916514dde314ff9fdc6393a4e95fc7fc15661600c4", ["Ice cream", "cupcake"], [], True, "Ice cream与cupcake为同级双甜点。"),
 ("e6d50c7877f3e73739f7ac938debb4816b8b45cb516fdaa239414c945e938931", ["Breakfast burrito", "breakfast plate"], ["eggs", "salsa", "guacamole", "jalapeños", "potatoes"], True, "Breakfast burrito与breakfast plate为竞争主菜，后续列表为配属。"),
 ("e9162326d73c1a27398be9cd19fc62dd6a538a456bb900264143fbbcc06b770d", ["egg on top of meat"], ["French fries"], False, "完整主回答为egg on top of meat，French fries为配属。"),
 ("f4fbcef9ec262f03d906124a4214e28d7f89bf07dfc8b33a3b1f12fe80305bb7", ["steak"], ["Rice"], False, "steak为主菜，Rice为配菜。"),
 ("f7b15aafe8e661208aeaeccafac6495273355df57dc99df2c22ad176b16a9416", ["seared tuna tartare", "carpaccio"], [], True, "seared tuna tartare与carpaccio为竞争具体主菜。"),
 ("fb12f404021bbe9005974fb2c2d58effb3a538b61113697b24183204b0081856", ["Steak"], ["mashed potatoes", "carrots", "zucchini", "cucumber"], False, "Steak为主菜，其余列表为配菜。"),
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
    prior_receipt = json.loads((previous / "receipt.json").read_text())
    prior_verify = json.loads((previous / "verification.json").read_text())
    source_proof_path = previous / "received_source_verification.json"
    source_proof = json.loads(source_proof_path.read_text())
    score_path, queue_path = previous / "score_rows.jsonl.gz", previous / "boundary_queue.jsonl"
    if (prior_verify["passed"] is not True or prior_receipt["rows"] != 10928
            or file_hash(score_path) != prior_receipt["score_rows_sha256"]
            or file_hash(score_path) != prior_verify["score_rows_sha256"]
            or file_hash(source_proof_path) != prior_verify["source_verification_sha256"]
            or source_proof["passed"] is not True):
        raise ValueError("Previous actual raw-verified CPU score provenance differs")
    packet = [record for _line, record, _sha in rows(queue_path)]
    actual_luna = list(rows(luna / "decisions.jsonl"))
    luna_receipt = json.loads((luna / "execution_receipt.json").read_text())
    if (len(packet) != 40 or len(actual_luna) != 40 or len(RULINGS) != 40
            or luna_receipt["queue_sha256"] != file_hash(queue_path)
            or luna_receipt["decision_sha256"] != file_hash(luna / "decisions.jsonl")
            or luna_receipt["annotation_model"] != "gpt-5.6-luna" or luna_receipt["annotation_effort"] != "medium"
            or luna_receipt["missing"] or luna_receipt["duplicates"]):
        raise ValueError("Actual Luna execution, queue or successful decision-write proof differs")
    records = [record for _line, record, _sha in rows(score_path)]
    by_key = {record["key"]: record for record in records}
    parts = {(item["claim_id"], item["part"]): item for item in source_proof["parts"]}
    if len(records) != 10928 or len(by_key) != 10928 or len(parts) != 31:
        raise ValueError("Actual source score or sealed-part keys differ")
    root_decisions, member_proofs, changed_keys = [], [], set()
    for index, (item, actual, ruling) in enumerate(zip(packet, actual_luna, RULINGS), 1):
        line, luna_decision, luna_line_sha = actual
        qkey, main_spans, side_spans, multiple, reason = ruling
        if (item["qa_key"] != qkey or luna_decision["qa_key"] != qkey
                or qkey != frozen.qah(item["question"], item["answer"])
                or item["question"] != "What specific food is shown in this image?"
                or (luna_decision["question"], luna_decision["answer"]) != (item["question"], item["answer"])
                or luna_decision["queue_index"] != index or line != index
                or luna_decision["source_memberships"] != item["memberships"]):
            raise ValueError("Root order or actual complete-QA/Luna source membership differs")
        for span in main_spans + side_spans:
            if not span or span not in item["answer"]:
                raise ValueError("A root main or side span is not continuous original text")
        relations = [{"name": span, "span": span, "role": "coequal" if multiple else "main"} for span in main_spans]
        relations += [{"name": span, "span": span, "role": "side"} for span in side_spans]
        if index == 13:
            relations.append({"name": "Breakfast plate", "span": "Breakfast plate", "role": "container"})
        if index == 8:
            relations.extend({"name": span, "span": span, "role": "ingredient"} for span in ("Shrimp", "mussel"))
        if index == 12:
            next(relation for relation in relations if relation["name"] == "coffee")["role"] = "drink"
        if index == 32:
            relations.append({"name": "Toasted bread slices", "span": "Toasted bread slices", "role": "description"})
        primary = None if multiple else main_spans[0]
        decision = {"qa_key": qkey, "question": item["question"], "answer": item["answer"],
            "abstain": False, "needs_root": False, "status": "multiple_primary" if multiple else "resolved_primary",
            "primary_answer_span": primary, "fullspan_primary_name": primary, "literal_full_name": primary,
            "endorsed_primary_names": main_spans, "name_relations": relations, "name_scope_ambiguous": False,
            "canonical_override": "multiple_primary" if multiple else None, "root_reason": reason, "reason": reason,
            "root_review_author": AUTHOR, "prepared_by": "/root/assets", "annotation_model": AUTHOR["model"],
            "annotation_effort": AUTHOR["effort"], "annotation_agent": AUTHOR["agent"],
            "annotation_session_id": AUTHOR["session_id"], "annotation_call_id": "",
            "queue_index": index, "source_luna_decision": luna_decision,
            "source_luna_path": str((luna / "decisions.jsonl").relative_to(ROOT)), "source_luna_line": line,
            "source_luna_line_sha256": luna_line_sha, "source_memberships": item["memberships"],
            "source_packet_path": str(queue_path.relative_to(ROOT)), "source_packet_sha256": file_hash(queue_path),
            "source_CPU_verification_path": str((previous / "verification.json").relative_to(ROOT)),
            "source_CPU_verification_sha256": file_hash(previous / "verification.json"), "raw_reopened": False}
        for member in item["memberships"]:
            record = by_key[member["key"]]
            part = parts[record["source_claim"], record["source_part"]]
            fields = ("model", "method", "sample_id", "target_class", "key", "source_path", "source_line",
                      "source_identity", "raw_line_sha256", "source_claim", "source_part")
            if (any(member[field] != record[field] for field in fields)
                    or (record["question"], record["answer"]) != (item["question"], item["answer"])
                    or record["source_identity"] != part["actual_source_identity"]
                    or record["raw_source_sha256"] != part["files"]["raw"]["sha256"]
                    or record["source_path"] != part["files"]["raw"]["received_path"]
                    or record["seed"] != stable_seed(record["sample_id"], record["model"], 0)
                    or record["key"] in changed_keys):
                raise ValueError("Root source member differs from actual previously raw-verified CPU identity, seed or SHA")
            changed_keys.add(record["key"])
            member_proofs.append({**member, "qa_key": qkey, "seed": record["seed"],
                "raw_source_sha256": record["raw_source_sha256"],
                "CPU_source_proof_sha256": file_hash(source_proof_path),
                "binding_verification": "prior_actual_raw_CPU_proof_to_verified_score_to_exact_QA_packet",
                "raw_reopened": False})
        root_decisions.append(decision)
    if len(changed_keys) != 40:
        raise ValueError("Root finite review does not cover exactly forty source score rows")
    review_output.mkdir(parents=True, exist_ok=False)
    decision_path = review_output / "root_reviewed.jsonl"
    save_rows(decision_path, root_decisions)
    save_rows(review_output / "source_member_verification.jsonl", member_proofs)
    save_json(review_output / "receipt.json", {"schema": "kdm_remaining4_root40_review_receipt_v1", "passed": True,
        "actual_root_author": AUTHOR, "prepared_by": "/root/assets", "QA": 40, "source_members": len(changed_keys),
        "multiple_primary_QA": sum(item[3] for item in RULINGS), "unique_primary_QA": sum(not item[3] for item in RULINGS),
        "question_answer_identity_verified": True, "raw_reopened": False,
        "source_binding_scope": "reused_actual_raw_verified_CPU_hash_key_identity_seed_QA_chain",
        "queue_sha256": file_hash(queue_path), "source_score_sha256": file_hash(score_path),
        "luna_decision_sha256": file_hash(luna / "decisions.jsonl"), "root_decision_sha256": file_hash(decision_path),
        "source_member_proof_sha256": file_hash(review_output / "source_member_verification.jsonl"),
        "preparer_sha256": file_hash(Path(__file__)), "new_API_calls": 0, "GPU_initialized": False})
    decisions = {item["qa_key"]: {**item, "decision_source_path": str(decision_path), "decision_source_line": index}
                 for index, item in enumerate(root_decisions, 1)}
    patterns = frozen.compile_classes(sorted({record["target_class"] for record in records}))
    updated, unchanged = [], 0
    for original in records:
        if original["qa_key"] not in decisions:
            updated.append(original)
            unchanged += 1
            continue
        inferred = infer_qa(original["question"], original["answer"], patterns, {}, {}, decisions)
        canonical, literal, reason = score_target(original["answer"], original["target_class"], inferred, patterns)
        if canonical is None or literal is None or inferred["abstain"] is not False:
            raise ValueError("Actual root review left a primary, literal or abstention field unresolved")
        names = frozen.extract(inferred["decision"], patterns)["literal"]
        record = {**original, "canonical_name_in_primary_score": canonical, "literal_extracted_name_score": literal,
            "literal_extracted_names": names, "abstain": False, "score_reason": reason,
            "behavior_source": inferred["behavior_source"], "decision_source": inferred["decision"],
            "annotation_model": AUTHOR["model"], "annotation_effort": AUTHOR["effort"], "annotation_call_id": "",
            "semantic_primary_extraction": {"primary_name": inferred["decision"]["fullspan_primary_name"],
                "status": inferred["decision"]["status"], "rule": "actual_root_complete_QA_role_review"},
            "previous_score_fields": {field: original[field] for field in
                ("canonical_name_in_primary_score", "literal_extracted_name_score", "literal_extracted_names",
                 "abstain", "score_reason", "behavior_source", "decision_source")},
            "previous_score_source_path": str(score_path.relative_to(ROOT)), "previous_score_source_sha256": file_hash(score_path)}
        updated.append(record)
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "score_rows.jsonl.gz", updated)
    save_rows(output / "boundary_queue.jsonl", [])
    counts = [item for _line, item, _sha in rows(previous / "condition_counts.jsonl")]
    for item in counts:
        selected = [record for record in updated if (record["model"], record["method"]) == (item["model"], item["method"])]
        if len(selected) != item["received_n"] or Counter(record["target_class"] for record in selected) != item["received_category_counts"]:
            raise ValueError("Incremental root score update changed source coverage or class quotas")
        item.update(canonical_correct=sum(record["canonical_name_in_primary_score"] == 1 for record in selected),
                    canonical_incorrect=sum(record["canonical_name_in_primary_score"] == 0 for record in selected),
                    canonical_pending=sum(record["canonical_name_in_primary_score"] is None for record in selected),
                    literal_correct=sum(record["literal_extracted_name_score"] == 1 for record in selected),
                    literal_pending=sum(record["literal_extracted_name_score"] is None for record in selected),
                    abstain_true=sum(record["abstain"] is True for record in selected),
                    abstain_false=sum(record["abstain"] is False for record in selected),
                    abstain_pending=sum(record["abstain"] is None for record in selected))
    if unchanged != 10888 or any(record["canonical_pending"] or record["literal_pending"] or record["abstain_pending"] for record in counts):
        raise ValueError("Finite root review failed to close all fields or changed its update scope")
    save_rows(output / "condition_counts.jsonl", counts)
    shutil.copyfile(source_proof_path, output / source_proof_path.name)
    receipt = {**prior_receipt, "completed_utc": datetime.now(timezone.utc).isoformat(), "boundary_QA": 0,
        "canonical_pending": 0, "literal_pending": 0, "abstain_pending": 0, "condition_counts": counts,
        "behavior_source_counts": dict(Counter(record["behavior_source"] for record in updated)),
        "score_rows_sha256": file_hash(output / "score_rows.jsonl.gz"), "scorer_sha256": file_hash(Path(__file__)),
        "source_scorer_sha256": prior_receipt["scorer_sha256"], "incremental_QA_update_rows": 40,
        "unchanged_score_rows": unchanged, "raw_reopened": False,
        "incremental_source_score_path": str(score_path.relative_to(ROOT)), "incremental_source_score_sha256": file_hash(score_path),
        "actual_root_decision_path": str(decision_path.relative_to(ROOT)), "actual_root_decision_sha256": file_hash(decision_path)}
    receipt["authority_inputs"] = {**receipt["authority_inputs"], "appended_root40":
        {"path": str(decision_path.relative_to(ROOT)), "sha256": file_hash(decision_path)}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({"review_QA": 40, "source_members_verified": len(changed_keys), "multiple_primary_QA": 16,
        "updated_score_rows": 40, "unchanged_score_rows": unchanged, "rows": len(updated),
        "canonical_pending": 0, "literal_pending": 0, "abstain_pending": 0, "raw_reopened": False,
        "root_decisions": str(decision_path.relative_to(ROOT)), "score_output": str(output.relative_to(ROOT))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
