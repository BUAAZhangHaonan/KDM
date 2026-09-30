#!/usr/bin/env python3
"""Append the root's target-blind semantic review of the finite native QA batch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

# These are actual root judgments after reading every complete response, not a
# name-splitting rule. Indices are bound to the original QA cohort below.
# Each tuple is (complete primary spans, side/ingredient/explanation spans,
# explicit competing-primary status, reason).
JUDGMENTS = {
  1: (["chicken and dumplings"], [], False, "复合菜名保留，不拆成两个竞争回答。"),
  2: (["Beef and bacon poutine"], ["over-easy egg"], False, "主体为poutine，beef/bacon修饰；鸡蛋为with配菜。"),
  3: (["Greek salad"], ["chopped tomatoes", "cucumbers", "red onions", "feta cheese"], False, "明确Greek salad；consisting of后是配料。"),
  4: (["steak", "roast beef"], ["sliced beef"], True, "could be后的两个具体候选竞争，未唯一选菜；未实际拒答。"),
  5: (["slice of carrot cake"], [], False, "appears保留唯一明确carrot cake。"),
  6: (["chicken and dumplings soup"], [], False, "保留完整复合汤名，不作同义类映射。"),
  7: (["fish"], ["coleslaw", "french fries", "lemon wedge"], False, "鱼为主体，蔬菜/薯条/柠檬为配菜；不能据组合替换fish and chips类名。"),
  8: (["ice cream"], ["Strawberries", "pomegranate"], False, "over ice cream明确主体，水果为上层配料。"),
  9: (["Fried fish"], ["french fries"], False, "鱼主体、薯条配菜，不替换为fish and chips别名。"),
 10: (["sandwich"], ["french fries"], False, "泛称sandwich主体，薯条配菜；不推断club等子类。"),
 11: (["fried fish"], ["french fries", "coleslaw"], False, "鱼主体、薯条与凉拌菜配菜。"),
 12: (["Sandwich"], ["french fries"], False, "sandwich主体，薯条配菜，不推断子类。"),
 13: (["cake", "bread pudding"], ["whipped cream", "raspberries"], True, "possibly后的cake/bread pudding两候选竞争；顶层配料不参与候选。"),
 14: (["Ice cream"], ["fruit"], False, "冰淇淋主体，fruit泛指配料。"),
 15: (["Seaweed salad", "tartare"], [], True, "两项同级菜，没有唯一主菜指认。"),
 16: (["Ice cream"], ["dessert"], False, "dessert为泛称同一甜点，不是第二具体菜名。"),
 17: (["meatloaf", "hamburger patty"], ["greens", "sauce"], True, "possibly/or给出两个肉主体候选。"),
 18: (["Steak"], ["Eggs"], False, "steak为肉主体，鸡蛋为配菜；完整回复未拒答。"),
 19: (["onion rings", "calamari"], ["lemon wedge"], True, "including并列具体油炸食物，没有唯一主体；不把泛称fried seafood当单一规范类。"),
 20: (["lasagna"], ["Cheesy bread"], False, "lasagna为成菜主体，cheesy bread为面包配菜。"),
 21: (["Grilled steak"], ["baked potato", "butter", "parsley", "diced vegetables"], False, "肉主体与配菜/装饰明确区分。"),
 22: (["ravioli", "tortellini"], ["creamy yellow sauce"], True, "likely/or给出两种pasta候选，未唯一指认。"),
 23: (["Fried chicken"], ["french fries"], False, "鸡肉主体、薯条配菜；不推断具体鸡肉101子类。"),
 24: (["stuffed filet mignon"], ["Asparagus", "hollandaise", "grated cheese"], False, "filet mignon为肉主体，蔬菜和酱料为配菜/配料。"),
 25: (["Mussels", "fish fillet"], [], True, "两项海鲜同级主体，没有唯一选择。"),
 26: (["pulled pork sandwich"], ["slice of cornbread"], False, "三明治主体、面包配菜，原完整修饰名称保留。"),
 27: (["steak", "shrimp"], ["Asparagus", "grits"], True, "并列两个肉/海鲜主体；不重排文本创造shrimp and grits规范名称。"),
 28: (["Steak"], ["mushrooms"], False, "肉主体、蘑菇配菜。"),
 29: (["Shrimp", "steak"], [], True, "海鲜和肉同级，不选择第一个词为主菜。"),
 30: (["Pork chop"], ["potatoes"], False, "猪排主体、土豆配菜；原单数保留。"),
 31: (["Hummus"], ["Pita"], False, "hummus为蘸料成菜，pita为搭配面包。"),
 32: (["gourmet burger, which is a type of hamburger"], [], False, "which is a type of把hamburger作为同一gourmet burger的明确类别说明，不是竞争菜。"),
 33: (["slice of lasagna"], [], False, "slice of是份量修饰，唯一lasagna。"),
 34: (["bowl of bibimbap"], ["rice", "vegetables", "meat"], False, "consisting of为bibimbap配料说明。"),
 35: (["Caprese salad"], ["fresh mozzarella cheese", "ripe tomatoes", "fresh basil leaves"], False, "唯一沙拉主体，其余是配料。"),
 36: (["Ice cream", "frosted donut"], ["sprinkles"], True, "两项甜点同级主体，sprinkles为配料。"),
 37: (["Grilled steak"], ["sweet potato", "dipping sauce"], False, "肉主体，土豆与酱料配菜。"),
 38: (["Frozen yogurt", "pie"], [], True, "两个同级甜点，没有唯一主菜。"),
 39: (["grilled ham and tomato sandwich"], ["french fries"], False, "ham/tomato为sandwich配料；薯条为配菜。"),
 40: (["Cheeseburger"], ["French fries"], False, "汉堡主体、薯条配菜，原cheeseburger不替换hamburger。"),
 41: (["cheeseburger"], ["french fries"], False, "汉堡主体、薯条配菜，不作同义替换。"),
 42: (["Grilled cheese sandwich"], ["salad"], False, "三明治主体、泛称salad配菜。"),
 43: (["roasted pork", "dumplings"], [], True, "两项同级肉菜与饺子主体，没有唯一主菜。"),
 44: (["Banana bread", "ice cream"], [], True, "两种同级甜点，没有指明配料或唯一主体。"),
 45: (["Sandwich"], ["French fries"], False, "三明治主体、薯条配菜，不推断子类。"),
 46: (["omelette"], ["hash browns"], False, "鸡蛋成菜主体、土豆配菜。"),
 47: (["Grilled steak", "piece of chicken"], ["roasted potatoes"], True, "两个肉主体同级，土豆配菜。"),
 48: (["BBQ Poutine", "burger"], [], True, "两项完整成菜同级，没有唯一主菜。"),
 49: (["sandwich"], ["onion rings"], False, "三明治主体、洋葱圈配菜，不按目标挑候选。"),
 50: (["sushi", "shrimp", "other seafood items"], [], True, "various/including明确多项食物，并非唯一sushi主菜。"),
 51: (["Onion rings"], ["ketchup"], False, "洋葱圈为唯一食物主体、番茄酱为蘸料。"),
 52: (["Hot dogs"], ["French fries"], False, "热狗主体、薯条配菜；单复数按原文。"),
 53: (["Cheeseburger"], ["onion rings"], False, "汉堡主体、洋葱圈配菜。"),
 54: (["Bacon-wrapped ribeye steak"], ["sweet potato"], False, "保留肉主体完整修饰名，土豆配菜。"),
 55: (["steak", "beef ribs"], [], True, "两个肉主体同级，没有唯一选择。"),
 56: (["Grilled steak"], ["beans", "salsa", "lemon", "lettuce"], False, "肉主体与配菜/调味装饰区分。"),
 57: (["purple and white ice cream cone"], ["various fruits"], False, "ice cream主体，cone/颜色修饰保留，水果配料。"),
 58: (["plate of nachos"], ["glass of salsa"], False, "nachos主体、salsa蘸料。"),
 59: (["Ice cream", "carrot pudding"], [], True, "两项甜点同级，没有唯一主菜。"),
 60: (["dumplings"], ["filled dough"], False, "唯一dumplings，后文为成分说明，不替换gyoza。"),
 61: (["Spring rolls"], ["cheese dip"], False, "春卷主体、蘸料配菜。"),
 62: (["Shrimp", "steak"], [], True, "两项同级海鲜/肉主体。"),
 63: (["Scallops"], ["asparagus", "mashed potatoes", "lemon"], False, "扇贝主体、蔬菜/土豆/柠檬配菜。"),
 64: (["Sushi", "Tempura"], [], True, "两项同级成菜，没有唯一选择。"),
 65: (["Mussels"], ["rice"], False, "贻贝主体、米饭配菜。"),
 66: (["Ribs"], ["french fries"], False, "排骨主体、薯条配菜，不从泛称ribs推断baby back子类。"),
 67: (["slice of chocolate cake"], [], False, "唯一蛋糕，后文为外观/质地说明。"),
 68: (["eggs benedict"], ["hash brown", "bacon"], False, "鸡蛋成菜主体、土豆与培根配菜。"),
 69: (["Macarons", "chocolate eggs"], [], True, "两个同级具体甜食，没有唯一主体。"),
 70: (["bowl of ramen"], ["Japanese noodle soup", "seasoned broth", "noodles"], False, "ramen唯一成菜，其他是定义/配料。"),
 71: (["Salmon"], ["French fries", "lemon wedge"], False, "鱼主体、薯条和柠檬配菜，保留原名不替换grilled salmon。"),
 72: (["plate of churros"], ["fried dough pastry", "cup of chocolate sauce"], False, "churros为唯一成菜，定义与巧克力酱不竞争。"),
 73: (["enchiladas"], ["Garlic bread"], False, "enchiladas为完整主菜，面包为配菜。"),
 74: (["salmon"], ["french fries", "mashed potatoes"], False, "鱼主体、两种土豆配菜。"),
 75: (["Cheeseburger"], ["french fries"], False, "汉堡主体、薯条配菜，不作词汇替換。"),
 76: (["Omelette"], ["hash browns"], False, "鸡蛋成菜主体、土豆配菜。"),
 77: (["onion rings"], ["tomato slices", "pickles", "ranch dressing"], False, "洋葱圈主体、蔬菜与蘸料配菜。"),
 78: (["Sub sandwich"], ["onion rings"], False, "三明治主体、洋葱圈配菜，不替换三明治子类。"),
 79: (["Grilled steak"], ["rice pilaf", "steamed vegetables", "brown sauce"], False, "肉主体与配菜/酱料区分。"),
 80: (["ham and tomato omelette"], ["Hash brown", "toast"], False, "omelette为完整鸡蛋成菜，土豆和吐司为配菜。"),
 81: (["battered and fried seafood, likely scallops"], [], False, "likely给出唯一具体scallops；描述性泛称不是第二竞争主菜。"),
 82: (["Toast"], ["garlic bread crumbs", "mussels"], False, "with后的碎屑/贻贝为吐司配料；不能截出garlic bread作为主菜。"),
 83: (["BBQ ribs"], ["french fries", "biscuit"], False, "肉主体、薯条与饼配菜，不推断baby back ribs。"),
 84: ([], ["piece of cooked meat"], False, "明确cannot be conclusively identified，保存泛称描述但无已承诺具体答案。"),
 85: (["Greek salad"], ["leafy greens", "crumbled white cheese", "black olive", "cucumber slices"], False, "Greek salad唯一主体，其他为成分特征。"),
 86: (["Gumbo"], ["shrimp", "grits"], False, "Gumbo主菜，with后的虾/玉米粥为配料，不改为shrimp and grits类。"),
 87: (["soft serve ice cream"], ["frozen dessert", "milk", "cream", "sugar", "flavorings"], False, "ice cream唯一主体，泛称与成分解释不竞争。"),
 88: (["Fried chicken"], ["french fries", "ketchup"], False, "鸡肉主体、薯条与番茄酱配菜。"),
 89: (["Ice cream", "pastry"], [], True, "两个甜点同级，没有唯一选择。"),
 90: (["possibly miso soup"], ["traditional Japanese dish"], False, "possibly仍给出唯一具体miso soup；soup/broth是同一主体泛称。"),
 91: (["Eggs Benedict"], ["hash browns"], False, "鸡蛋成菜主体、土豆配菜。"),
 92: (["Asian savory pancake, commonly known as 'scallion pancakes' or 'Chinese pancakes'"], [], False, "commonly known as明确同一菜名说明，规范pancakes词形仍按原文，无同义替换。"),
 93: (["Ham and cheese sandwich"], ["french fries"], False, "ham/cheese为三明治配料，薯条配菜；不推断grilled。"),
 94: (["bowl of bibimbap"], ["rice", "vegetables", "fried egg"], False, "bibimbap主体，consisting of后为配料。"),
 95: (["paella"], ["Spanish rice dish", "seafood", "vegetables", "spices"], False, "paella唯一主体，其他为定义与配料。"),
 96: (["Curried pumpkin", "fried rice", "steamed buns"], [], True, "三个同级成菜，没有唯一主菜。"),
 97: (["Ham sandwich"], ["French fries"], False, "三明治主体、薯条配菜，不推断子类。"),
 98: (["steak"], ["egg"], False, "肉主体、鸡蛋配菜。"),
 99: (["grilled steak"], ["medium-rare doneness", "sear marks"], False, "肉主体，火候与外观不是食物配菜或竞争回答。"),
100: (["Falafel", "tabbouleh salad"], [], True, "两个明确同级成菜，没有唯一主体。"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = [json.loads(line) for line in args.input.read_text().splitlines()]
    queue = {row["qa_key"]: row for row in map(json.loads, args.queue.read_text().splitlines())}
    if len(source) != 100 or set(queue) != {row["qa_key"] for row in source}:
        raise ValueError("Original 100-QA cohort differs")
    wanted = {}
    for row in source:
        qkey = hashlib.sha256(json.dumps([row["question"], row["answer"]],
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if qkey != row["qa_key"] or row["question"] != "What specific food is shown in this image?":
            raise ValueError("Complete QA binding differs")
        for member in queue[qkey]["memberships"]:
            wanted.setdefault(member["source_path"], {})[member["source_line"]] = member
    checked = {}
    for path, required in wanted.items():
        with Path(path).open("rb") as stream:
            for number, line in enumerate(stream, 1):
                if number in required:
                    checked[(path, number)] = (json.loads(line), hashlib.sha256(line).hexdigest())
                if number >= max(required):
                    break
    output = []
    for original in source:
        index, answer = original["queue_index"], original["answer"]
        primary, secondary, competing, reason = JUDGMENTS[index]
        for span in primary + secondary:
            if span not in answer:
                raise ValueError(f"Absent reviewed complete span at {index}: {span}")
        for member in queue[original["qa_key"]]["memberships"]:
            row, sha = checked[(member["source_path"], member["source_line"])]
            if (row["key"], row["identity"], row["model"], row["sample"]["question"], row["text"], sha) != (
                member["key"], member["source_identity"], member["model"], original["question"], answer, member["raw_line_sha256"]):
                raise ValueError("Actual source identity/QA/line SHA differs")
        relations = [{"name": name, "span": name, "role": "coequal" if competing else "main"}
                     for name in primary]
        relations += [{"name": name, "span": name, "role": "side"} for name in secondary]
        reviewed = {**original, "endorsed_primary_names": primary, "name_relations": relations,
            "name_scope_ambiguous": False, "abstain": index == 84,
            "source_luna_decision": original, "root_reviewed": True, "needs_root": False,
            "root_reason": reason, "annotation_model": "gpt-6.1-sol", "annotation_effort": "max",
            "annotation_agent": "/root", "annotation_call_id": "",
            "annotation_session_id": "01a0f0e4-4fe9-7940-b763-e0b428e5d5d8",
            "root_reviewed_at_utc": datetime.now(timezone.utc).isoformat()}
        if competing:
            reviewed["canonical_override"] = "multiple_primary"
        output.append(reviewed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"reviewed_QA": len(output), "source_files_checked": len(wanted),
        "source_memberships_verified": len(checked), "abstention_QA": 1,
        "explicit_competing_primary_QA": sum(row.get("canonical_override") == "multiple_primary" for row in output),
        "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
