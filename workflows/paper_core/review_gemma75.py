#!/usr/bin/env python3
"""Append root corrections after reading the finite 75 new Gemma native QA."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

# Actual semantic judgments; spans stay exactly in the generated response.
# Tuple: primary spans, other spans, competing distinct dishes, rationale.
CORRECTIONS = {
  2: (["Shrimp and egg breakfast tacos"], [], False, "Shrimp/egg为tacos配料，不拆成两个主菜。"),
  4: (["slice of New York cheesecake"], ["scoop of vanilla ice cream"], False, "with a scoop明确冰淇淋为配菜，cheesecake主菜。"),
  8: (["Grilled Cheese Sandwich"], ["brie cheese"], False, "inside的brie为三明治配料，末尾生成片段没有第二明确菜名。"),
  9: (["Tuna ceviche (or likely shrimp ceviche, given the shrimp visible)"], [], False, "两种原料候选都明确同一ceviche菜类，保留完整回答及其原料不确定性，不引入别名。"),
 14: (["ice cream", "chocolate cakes"], ["berry compote", "mint garnish"], True, "冰淇淋和蛋糕两项同级甜点主体，果酱/薄荷装饰。"),
 15: (["grilled cheese sandwich"], ["crispy potato wedges"], False, "三明治主体、土豆配菜，不按首个列举项选主菜。"),
 20: (["pot stickers"], ["dumplings", "jiaozi"], False, "also known as给同一主体名称解释，不是竞争；保留pot stickers原字形不换gyoza。"),
 24: (["Fried onion blossoms", "onion rings"], ["biscuit"], True, "or给两种具体食物候选，不作同义替换；饼为inside配料。"),
 25: (["Fried Chicken Wings"], ["red sauce"], False, "with后的酱料不是同级主菜，未完成Fried片段不补成新菜名。"),
 28: (["French Onion Soup Gratinée"], ["Onion Gratinée"], False, "括号为同一汤名简写，不是另一具体主菜。"),
 30: ([], [], False, "完整回复明确There is no food，不承诺具体食物答案，实际弃权。"),
 33: (["Filet Mignon", "Carpaccio"], ["arugula", "Parmesan cheese"], True, "首句or留下两个具体肉菜，后文强调carpaccio但未明确撤回首个候选。"),
 34: (["salsa", "guacamole"], [], True, "selection of两种蘸料同级，没有唯一主菜。"),
 35: (["chocolate-dipped wafer cake", "vanilla ice cream", "macaron"], ["strawberry", "chocolate sauce"], True, "甜点盘有多个具体甜点主体，不能任取其中一类。"),
 36: (["Steak"], ["Sweet Potato with Cheese"], False, "肉主体，土豆/奶酪为配菜，末尾未完成bullet无新增完整菜名。"),
 38: (["enchiladas", "huevos rancheros"], [], True, "both明确两个同级完整墨西哥菜。"),
 41: (["Salmon and Tuna Maki sushi"], [], False, "Salmon/Tuna修饰同一Maki sushi，不拆成两种鱼主菜。"),
 45: (["Chicken and Dumplings"], [], False, "保留完整复合菜名，不拆成竞争主菜。"),
 47: (["Fried chicken wings"], ["loaded fries"], False, "鸡翅主体、薯条配菜。"),
 49: (["layered chocolate cake", "mousse dessert"], [], True, "or明确蛋糕与慕斯两个竞争主菜。"),
 51: (["beignets", "funnel cakes"], ["Cinnamon-sugar fried dough balls"], True, "两个实际甜点名称竞争，不把第二明确候选当作泛称similar。"),
 52: (["Chocolate and pink glazed donuts"], [], False, "Chocolate/pink glazed都是donuts修饰，非两个主菜。"),
 54: (["steak", "shrimp with a sauce"], ["fried potatoes"], True, "mixed plate含两个同级肉/海鲜主体，土豆配菜。"),
 55: (["powdered sugar-dusted French toast"], ["scrambled eggs", "bacon"], False, "French toast为明确面包成菜主体，鸡蛋/培根配菜，breakfast plate仅容器泛称。"),
 56: (["croissants", "tarts", "cakes", "cookies", "filled buns", "donuts"], [], True, "wide variety明确多种点心同级，没有唯一具体主体。"),
 59: (["scallops", "white fish"], ["asparagus", "leek", "creamy foam sauce"], True, "possibly/or在两个海鲜主体间不确定，蔬菜和酱料配菜。"),
 60: (["Chicken and dumplings"], ["creamy sauce"], False, "in sauce是调味，复合菜名为唯一主体。"),
 61: (["assorted sushi rolls and slices"], [], False, "rolls/slices是同一sushi的形态，不是另一菜名。"),
 64: (["bowl of miso soup"], ["fried tofu sticks"], False, "with后的豆腐为汤配料，唯一miso soup。"),
 67: (["sashimi"], ["thinly sliced raw fish or seafood"], False, "后面为sashimi定义，非另一个主菜。"),
 68: (["Churros"], ["hot chocolate"], False, "churros主体、热巧克力为搭配饮料。"),
 71: (["lemon and orange macarons"], ["chocolate sauce"], False, "lemon/orange为macarons风味，with后的酱料不竞争。"),
 73: (["Tomato and pepper bruschetta"], [], False, "Tomato/pepper修饰同一bruschetta，不拆分。"),
 74: (["Seared Scallop", "Foie Gras"], ["pickled vegetables", "berry reduction"], True, "两种海鲜/肉主体同级appetizer，不任取foie gras；原Scallop单数保留。"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    decisions = [json.loads(line) for line in args.input.read_text().splitlines()]
    queue = {row["qa_key"]: row for row in map(json.loads, args.queue.read_text().splitlines())}
    if len(decisions) != 75 or set(queue) != {row["qa_key"] for row in decisions}:
        raise ValueError("Original 75-QA coverage differs")
    wanted = {}
    for original in decisions:
        qkey = hashlib.sha256(json.dumps([original["question"], original["answer"]],
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if qkey != original["qa_key"] or original["question"] != "What specific food is shown in this image?":
            raise ValueError("Complete QA identity differs")
        for member in queue[qkey]["memberships"]:
            wanted.setdefault(member["source_path"], {})[member["source_line"]] = member
    actual = {}
    for path, required in wanted.items():
        with Path(path).open("rb") as stream:
            for number, line in enumerate(stream, 1):
                if number in required:
                    actual[(path, number)] = (json.loads(line), hashlib.sha256(line).hexdigest())
                if number >= max(required):
                    break
    output = []
    for original in decisions:
        qkey, index = original["qa_key"], original["queue_index"]
        for member in queue[qkey]["memberships"]:
            raw, sha = actual[(member["source_path"], member["source_line"])]
            if (raw["key"], raw["identity"], raw["model"], raw["sample"]["question"], raw["text"], sha) != (
                member["key"], member["source_identity"], member["model"], original["question"], original["answer"], member["raw_line_sha256"]):
                raise ValueError("Actual source QA/identity/raw line SHA differs")
        updated = dict(original)
        if index in CORRECTIONS:
            primary, other, competing, reason = CORRECTIONS[index]
            updated.update(endorsed_primary_names=primary,
                name_relations=[{"name": name, "span": name, "role": "coequal" if competing else "main"} for name in primary]
                    + [{"name": name, "span": name, "role": "side"} for name in other],
                name_scope_ambiguous=False, abstain=index == 30, root_reason=reason)
            if competing:
                updated["canonical_override"] = "multiple_primary"
        updated.update(source_luna_decision=original, root_reviewed=True, needs_root=False,
            annotation_model="gpt-6.1-sol", annotation_effort="max", annotation_agent="/root",
            annotation_call_id="", annotation_session_id="01a0f0e4-4fe9-7940-b763-e0b428e5d5d8",
            root_reviewed_at_utc=datetime.now(timezone.utc).isoformat())
        updated.setdefault("root_reason", "已读取完整Q/A，保留原唯一主菜/配料/语义弃权判断。")
        for relation in updated["name_relations"]:
            if relation["span"] not in updated["answer"]:
                raise ValueError("Reviewed original evidence span absent")
        output.append(updated)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"reviewed_QA": 75, "corrected_QA": len(CORRECTIONS),
        "source_files_checked": len(wanted), "source_memberships_verified": len(actual),
        "abstention_QA": sum(row["abstain"] for row in output), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
