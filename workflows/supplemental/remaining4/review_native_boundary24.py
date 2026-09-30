#!/usr/bin/env python3
"""Append the actual root 24-QA decisions and the separate Ice cream cone ruling."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, stable_seed, within
from workflows.main_results.score import qah
from workflows.supplemental.remaining11.score import rows
from workflows.supplemental.remaining4.score_native import save_json, save_rows

# The parent root read every complete QA and supplied these exact final judgments.
# key, complete answer, primary spans, side spans, canonical class, multiple, reason.
RULINGS = [
 ("054360f70bd696a78aa30e865fb3117a567f92a4f0ea55a574f823c448742500", "Tomato, ham and french fries", ["ham", "french fries"], ["Tomato"], None, True, "ham与french fries为同级枚举，Tomato附属，没有唯一主菜。"),
 ("13bc0eedfd7da4afe80761b3fc61fb3f313b20554104279ee7b3d2af1bce1148", "Chicken curry and naan", ["Chicken curry"], ["naan"], "chicken_curry", False, "Chicken curry为主菜，naan为配菜。"),
 ("2049b6f7dd4464ebf7aeeeda8b9238a7afadb2bf90c1985e50c9079131e0cc44", "A bowl of beef and vegetable bibimbap with a fried egg on top.", ["bowl of beef and vegetable bibimbap"], ["fried egg"], "bibimbap", False, "beef和vegetable修饰同一bibimbap，fried egg为topping；完整主菜名称保留bowl表述。"),
 ("22aada6c6b7d2bb31d2d67d440b2bd092cf6c86cbe727f6c2b10ad4b6c4b0001", "Strawberry dessert with cream and cannoli.", ["Strawberry dessert"], ["cream", "cannoli"], None, False, "Strawberry dessert为泛称主菜，cream和cannoli附属；实际主菜名称不对应101类规范词。"),
 ("2dddc64b354e925dea870698a41f36e4f899010f848da39362b2a4b75ddf12a5", "sausage and French toast", ["sausage", "French toast"], [], None, True, "sausage与French toast同级列举，没有明确主从关系。"),
 ("59607630e4bf19b3b84001277cc71cb98c4628e5ab23593a6564cf1b9031ffce", "Pie and ice cream", ["Pie", "ice cream"], [], None, True, "Pie与ice cream为同级甜点主体，没有明确唯一主菜。"),
 ("650f7cdd7d1e144b874e01652c0329393cb0e3c703887fcea75b6ba879961d75", "Hash browns and omelette", ["Hash browns", "omelette"], [], None, True, "Hash browns与omelette同级列举，原文没有明确主从关系。"),
 ("769100956d26fffa6fbc5ed924a8944a52312398fe0a4cf1eea34f80ea5398ce", "Chicken sandwich and french fries", ["Chicken sandwich"], ["french fries"], None, False, "Chicken sandwich为主菜，french fries为配菜；保留实际未知菜名字形。"),
 ("78fbffa2f47942c96f04da6ab0a613f4787c690a5173dcdf394ac4c2643a286c", "Pizza and dip", ["Pizza"], ["dip"], "pizza", False, "Pizza为主菜，dip为配料。"),
 ("7efcb4faf81aa8660f939a0c4779906bbea3b34337a8b21d32c07c28e7fd4c08", "French fries and eggs", ["French fries", "eggs"], [], None, True, "French fries与eggs为同级主体，未明确唯一主菜。"),
 ("896921c6e4654a8bc5f63b10ff64234e14f733273c7613a8b84809ff040b9a30", "Onion rings and potato chips", ["Onion rings", "potato chips"], [], None, True, "Onion rings与potato chips为同级列举，没有明确主从关系。"),
 ("928f680d59b58d0ef6ce448a66d44e510af6dff7a99e55e25719a3940c73bf2a", "Breakfast burrito and potatoes", ["Breakfast burrito"], ["potatoes"], "breakfast_burrito", False, "Breakfast burrito为主菜，potatoes为配菜。"),
 ("98621d61c1da7d51d57b3c07291aff34076c54eaf7016e4c8cecf0eaad65ae56", "French toast and potatoes", ["French toast"], ["potatoes"], "french_toast", False, "French toast为主菜，potatoes为配菜；同一完整QA复用到两个源成员。"),
 ("b40cdeff3ef54150dbe522550a58edea281aa07dda624cbc0bbe2cd78fc5b0b3", "Hamburger and salad", ["Hamburger"], ["salad"], "hamburger", False, "Hamburger为主菜，salad为配菜。"),
 ("bb088683a012fed9350d9c1dd0f9602e6025ec20ea648d2ddea2b75bef4b0f38", "French fries and egg", ["French fries", "egg"], [], None, True, "French fries与egg同级列举，没有明确唯一主菜。"),
 ("bfb69c229061bd9cc6d8bc221b6f24527f1ea5b4de94b7392087404f8d308189", "Chicken patty, falafel wrap and pickles", ["Chicken patty", "falafel wrap"], ["pickles"], None, True, "Chicken patty与falafel wrap为同级主体，pickles为配料；唯一primary、canonical和fullspan均为空。"),
 ("c60b53342ed94b943987b9cac50b665049d0c903668dcc014b1043f83a4e3984", "Fish fillet and french fries", ["Fish fillet"], ["french fries"], None, False, "Fish fillet为主菜，french fries为配菜；不创造未登记类别。"),
 ("d0dea9e3fc3105daa90a0a0c8696ee05d676bfb87075c5795805ce9eaa2a0492", "No food is shown in the image. It features a lion in a grassy field with mountains in the background.", [], [], None, False, "完整回复明确No food并描述非食物，没有承诺具体Food答案，实际弃权。"),
 ("dced80be2ba86527f87bbd2781294dd1ec66cc06d7d2d3332259184589a1ae74", "Macaron and hamburger", ["Macaron", "hamburger"], [], None, True, "Macaron与hamburger同级列举，原文未明确主从关系。"),
 ("ded655ad4e8169ed36187a620d6e4cd181e6840fc717f5c1c7bf9a84a8a90a07", "Based on the visual evidence, the dish shown is **Oysters en Croute** or more specifically, **Oysters Rockefeller**.\n\nHowever, a closer look", ["Oysters Rockefeller"], [], "oysters", False, "or more specifically明确收窄到唯一Oysters Rockefeller；末句截断没有实际撤回答案。"),
 ("e3acea13c079317bed078101faf4e6f3d00b1e4d5c31f6979f39719d2841529e", "Pizza and salad", ["Pizza"], ["salad"], "pizza", False, "Pizza为主菜，salad为配菜。"),
 ("e45bb020dd3619199deda83f45fb4b88a05056d7a0160333564b3e74caacaf86", "Donut and cheesecake", ["Donut", "cheesecake"], [], None, True, "Donut与cheesecake为同级甜点主体，没有唯一主菜。"),
 ("e9f7b38e04eebc839df8c751d79af480b585d495a944f6abc1042f682cf43087", "The image shows a variety of donuts displayed in a bakery or donut shop. The man is holding a tray with different types of donuts.", ["variety of donuts"], [], "donuts", False, "不同形态均指donuts，完整主菜名称为variety of donuts，没有第二种竞争主菜。"),
 ("f4c056c126f3f2e37f54b11a903c5abde63a2d8553f75153caeaa950af550527", "Broccoli, carrot, fried rice and fish", ["fried rice", "fish"], ["Broccoli", "carrot"], None, True, "fried rice与fish为同级主体，Broccoli和carrot附属；单标签任务没有唯一主菜。"),
]


def verify_members(members, score_by_key, question, answer):
    wanted = {}
    for member in members:
        wanted.setdefault(member["source_path"], {})[member["source_line"]] = member
    actual = {}
    for path, lines in wanted.items():
        with within(ROOT, path).open("rb") as stream:
            for number, raw in enumerate(stream, 1):
                if number in lines:
                    actual[path, number] = (json.loads(raw), hashlib.sha256(raw).hexdigest())
                if number >= max(lines):
                    break
    for member in members:
        source, sha = actual[member["source_path"], member["source_line"]]
        scored = score_by_key[member["key"]]
        if (source["key"], source["identity"], source["model"], source["sample"]["id"],
            source["sample"]["question"], source["text"], source["seed"], sha) != (
            member["key"], member["source_identity"], member["model"], member["sample_id"],
            question, answer, scored["seed"], member["raw_line_sha256"]):
            raise ValueError("Actual source QA, key, model, sample, identity, seed or line SHA differs")
        if source["seed"] != stable_seed(member["sample_id"], member["model"], 0):
            raise ValueError("Source seed differs from its frozen sample/model/replicate rule")
    return len(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--luna-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    score, luna, output = within(ROOT, args.score_dir), within(ROOT, args.luna_dir), within(ROOT, args.output)
    packet = [row for _, row, _ in rows(score / "boundary_queue.jsonl")]
    original = [row for _, row, _ in rows(luna / "decisions.jsonl")]
    execution = json.loads((luna / "execution_receipt.json").read_text())
    if (len(packet) != 24 or len(original) != 24
            or file_hash(score / "boundary_queue.jsonl") != execution["queue_sha256"]
            or file_hash(luna / "decisions.jsonl") != execution["decision_sha256"]):
        raise ValueError("Actual 24-QA queue or Luna execution receipt differs")
    scored = {row["key"]: row for _, row, _ in rows(score / "score_rows.jsonl.gz")}
    if len(scored) != 3632:
        raise ValueError("This review is bound to its exact finite 3632 source rows")
    output.mkdir(parents=True, exist_ok=False)
    reviewed, checked = [], 0
    for index, (queued, old, ruling) in enumerate(zip(packet, original, RULINGS), 1):
        key, answer, primary, side, canonical, multiple, reason = ruling
        if (queued["qa_key"] != key or old["qa_key"] != key or old["queue_index"] != index
                or queued["answer"] != answer or old["answer"] != answer
                or qah(queued["question"], answer) != key or old["question"] != queued["question"]):
            raise ValueError("Ordered root ruling, actual full QA or source Luna membership differs")
        checked += verify_members(queued["memberships"], scored, queued["question"], answer)
        if any(span not in answer for span in primary + side):
            raise ValueError("Root evidence span is absent from the complete generated response")
        single = primary[0] if len(primary) == 1 else None
        abstain = index == 18
        decision = {**old, "primary_name": single, "canonical_name": canonical, "fullspan_primary_name": single,
            "primary_span": single, "literal_full_name": single, "endorsed_primary_names": primary,
            "name_relations": [{"name": name, "span": name, "role": "coequal" if multiple else "main"} for name in primary]
                + [{"name": name, "span": name, "role": "side"} for name in side],
            "canonical_override": "multiple_primary" if multiple else (canonical or "explicit_outside_101"),
            "name_scope_ambiguous": False, "abstain": abstain, "needs_root": False, "root_reviewed": True,
            "status": "multiple_primary" if multiple else ("abstain" if abstain else "resolved_primary"),
            "reason": reason, "root_reason": reason, "source_luna_decision": old,
            "annotation_model": "gpt-6.1-sol", "annotation_effort": "max", "annotation_agent": "/root",
            "annotation_session_id": "01a0f0e4-4fe9-7940-b763-e0b428e5d5d8", "annotation_call_id": "",
            "prepared_by": "/root/assets", "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "source_memberships": queued["memberships"], "actual_source_line_SHA_seed_verified": True}
        if abstain:
            decision["abstention_span"] = "No food is shown in the image."
        reviewed.append(decision)
    if checked != 25:
        raise ValueError("Root 24-QA source coverage differs from 25 actual members")
    save_rows(output / "root_reviewed24.jsonl", reviewed)
    literal = [record for record in scored.values() if record["literal_extracted_name_score"] is None
        and record["canonical_name_in_primary_score"] is not None and record["abstain"] is not None]
    if len(literal) != 3 or {r["answer"] for r in literal} != {"Ice cream cone"}:
        raise ValueError("Separate root Ice cream cone scope differs")
    members = [{key: record[key] for key in ("key", "model", "sample_id", "source_path", "source_line", "source_identity", "raw_line_sha256")}
        for record in literal]
    verified_cone = verify_members(members, scored, literal[0]["question"], "Ice cream cone")
    cone = {"qa_key": literal[0]["qa_key"], "question": literal[0]["question"], "answer": "Ice cream cone",
        "primary_name": "Ice cream cone", "canonical_name": "ice_cream", "fullspan_primary_name": "Ice cream cone",
        "primary_span": "Ice cream cone", "literal_full_name": "Ice cream cone", "canonical_override": "ice_cream",
        "endorsed_primary_names": ["Ice cream cone"], "name_relations": [{"name": "Ice cream cone", "span": "Ice cream cone", "role": "main"}],
        "name_scope_ambiguous": False, "abstain": False, "needs_root": False, "root_reviewed": True,
        "root_reason": "Ice cream cone为完整实际主菜名称，规范类别ice_cream；完整名称保留cone，literal按每目标独立比较为0。",
        "annotation_model": "gpt-6.1-sol", "annotation_effort": "max", "annotation_agent": "/root",
        "annotation_session_id": "01a0f0e4-4fe9-7940-b763-e0b428e5d5d8", "annotation_call_id": "",
        "prepared_by": "/root/assets", "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source_score_records": literal, "actual_source_line_SHA_seed_verified": True}
    save_rows(output / "root_literal_cone1.jsonl", [cone])
    save_rows(output / "literal_only_source3.jsonl", literal)
    result = {"reviewed_QA": 24, "source_memberships_verified": checked, "literal_QA": 1,
        "literal_source_memberships_verified": verified_cone, "source_luna_file": str(luna / "decisions.jsonl"),
        "source_luna_sha256": file_hash(luna / "decisions.jsonl"), "root_reviewed24_sha256": file_hash(output / "root_reviewed24.jsonl"),
        "root_literal_cone1_sha256": file_hash(output / "root_literal_cone1.jsonl"), "new_API_calls": 0, "GPU_initialized": False}
    save_json(output / "receipt.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
