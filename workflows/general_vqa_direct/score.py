"""Benchmark correctness and full-response abstention on sealed Direct output.

Semantic review only interprets the response. Gold labels are joined after that
interpretation. Unresolved decisions remain missing, never wrong-by-default.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from kdm.scoring import lexical_label


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                yield number, json.loads(line)


def normalized(text):
    return unicodedata.normalize("NFKC", text).strip().replace("’", "'")


def formatted(text):
    value = normalized(text)
    for _ in range(4):
        before = value
        for left, right in [("**", "**"), ("`", "`"), ('"', '"')]:
            if value.startswith(left) and value.endswith(right) and len(value) > len(left) + len(right):
                value = value[len(left):-len(right)].strip()
        value = value.rstrip(".! \n\r\t")
        if value == before:
            break
    return value


def qa_key(sample, answer):
    return digest([sample["prompt"], answer])


def explicit_binary_decision(answer):
    """Accept an initial yes/no only when the complete reply keeps that decision.

    Revisions, conflicting polarity and inability/uncertainty are routed to the
    semantic reviewer. This rule never compares the response to the gold label.
    """
    value = normalized(answer)
    match = re.match(r"^\s*(?:\*\*|__)?(yes|no)\b(?:\*\*|__)?\s*[,.:!\-]?", value, re.I)
    if not match:
        return None
    prediction = match.group(1).lower()
    remainder = value[match.end():]
    boundary = (r"\b(?:cannot|can't|couldn't|unable|unclear|uncertain|unsure|"
                r"undetermined|ambiguous|inconclusive|hard to tell|difficult to tell|"
                r"not sure|not certain|not possible|not enough|insufficient|"
                r"however|but|although|yet|nevertheless|nonetheless|actually|correction|correct myself|"
                r"on second|instead|rather than|not necessarily|might|may|could|"
                r"possibly|perhaps|probably|appears?|seems?|likely)\b")
    if re.search(boundary, value, re.I):
        return None
    if re.search(r"\b(?:I|we)\s+(?:do\s+not|don't)\s+(?:know|see|have|think)", value, re.I):
        return None
    if re.search(r"\b(?:yes|no)\b", remainder, re.I):
        # Repeated same-polarity no in "No, there is no ..." is unambiguous.
        if prediction != "no" or re.search(r"\byes\b", remainder, re.I):
            return None
    if prediction == "yes" and re.search(r"\b(?:not|never|without|absent|neither)\b|n't\b", remainder, re.I):
        return None
    # A negative reply that later explicitly supplies a positive final answer
    # must be reviewed rather than assigned from its first word.
    if prediction == "no" and re.search(r"\b(?:answer|response)\s+(?:is|would be)\s+", remainder, re.I):
        return None
    return {"label": "answer_assertive", "abstain": False,
            "answer_text": answer, "predicted_answer": prediction,
            "evidence_span": answer,
            "decision_source": "explicit_binary_full_reply_guard_v1"}


def rule_decision(sample, answer):
    """Only whole-response, unambiguous formats are automatic."""
    literal = lexical_label(answer)
    if literal is not None:
        return {"label": literal, "abstain": literal == "abstain", "answer_text": "",
                "predicted_answer": None, "evidence_span": answer,
                "decision_source": "existing_kdm_whole_response_rule"}
    value = formatted(answer)
    dataset = sample["dataset"]
    if dataset in {"pope", "hallusionbench"} and value.casefold() in {"yes", "no"}:
        return {"label": "answer_assertive", "abstain": False, "answer_text": answer,
                "predicted_answer": value.casefold(), "evidence_span": answer,
                "decision_source": "complete_literal_yes_no"}
    if dataset in {"pope", "hallusionbench"}:
        binary = explicit_binary_decision(answer)
        if binary is not None:
            return binary
    options = sample.get("options", [])
    if options:
        match = re.fullmatch(r"(?:(?:The\s+(?:correct\s+)?answer\s+is|Answer\s*:|Option)\s*)?\(?([A-Z])\)?", value, re.I)
        if match and 0 <= ord(match.group(1).upper()) - 65 < len(options):
            return {"label": "answer_assertive", "abstain": False, "answer_text": answer,
                    "predicted_answer": match.group(1).upper(), "evidence_span": answer,
                    "decision_source": "complete_literal_option_letter"}
        indices = [i for i, option in enumerate(options) if formatted(str(option)).casefold() == value.casefold()]
        if len(indices) == 1:
            return {"label": "answer_assertive", "abstain": False, "answer_text": answer,
                    "predicted_answer": chr(65 + indices[0]), "evidence_span": answer,
                    "decision_source": "complete_unique_option_text"}
    if dataset == "mmmu" and sample.get("question_type") != "multiple-choice":
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\s*%)?", value):
            return {"label": "answer_assertive", "abstain": False, "answer_text": answer,
                    "predicted_answer": value, "evidence_span": answer,
                    "decision_source": "complete_numeric_open_answer"}
    return None


def check_decision(sample, answer, decision):
    if decision["label"] not in {"answer_assertive", "answer_uncertain", "abstain", "invalid"}:
        raise ValueError("Invalid semantic label")
    if type(decision["abstain"]) is not bool or decision["abstain"] != (decision["label"] == "abstain"):
        raise ValueError("Abstention and behavior label conflict")
    for field in ("answer_text", "evidence_span"):
        span = decision.get(field, "")
        if not isinstance(span, str) or span not in answer:
            raise ValueError("Semantic evidence must be a contiguous original span")
    if decision["label"] in {"answer_assertive", "answer_uncertain"}:
        prediction = decision.get("predicted_answer")
        if sample["dataset"] in {"pope", "hallusionbench"} and prediction not in {"yes", "no", None}:
            raise ValueError("Binary question prediction must be yes, no or explicitly unparsed")
        if sample.get("options") and prediction not in [chr(65+i) for i in range(len(sample["options"]))] + [None]:
            raise ValueError("Prediction is outside this question's original choices")


def correctness(sample, answer, decision, mmmu):
    if decision["label"] in {"abstain", "invalid"}:
        return 0
    prediction = decision.get("predicted_answer")
    if prediction is None:
        return 0
    if sample["dataset"] in {"pope", "hallusionbench"}:
        gold = sample["gold"]
        if gold in (0, "0"):
            gold = "no"
        elif gold in (1, "1"):
            gold = "yes"
        if gold not in {"yes", "no"}:
            raise ValueError("Unknown binary gold label")
        return int(prediction == gold)
    if sample["dataset"] == "scienceqa":
        gold = sample["gold"]
        letter = chr(65 + gold) if isinstance(gold, int) else str(gold)
        return int(prediction == letter)
    if sample["question_type"] == "multiple-choice":
        return int(mmmu.eval_multi_choice(sample["gold"], prediction))
    return int(mmmu.eval_open(sample["gold"], mmmu.parse_open_response(answer)))


def pope_official_prediction(answer):
    # Original RUCAIBox/POPE evaluate.py lines 8-20, retained independently.
    text = answer.split(".")[0].replace(",", "")
    words = text.split(" ")
    return "no" if any(word in words for word in ("No", "not", "no")) else "yes"


def load_mmmu(path):
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("general_vqa_official_mmmu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_rows(path, values):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", action="append", type=Path, default=[])
    parser.add_argument("--raw-list", type=Path, help="Validated immutable raw paths, one per line")
    parser.add_argument("--decisions", action="append", type=Path, default=[])
    parser.add_argument("--mmmu-eval", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.raw_list:
        args.raw.extend(Path(line) for line in args.raw_list.read_text().splitlines() if line.strip())
    if not args.raw or len(args.raw) != len(set(args.raw)):
        parser.error("At least one unique validated raw file is required")
    args.out.mkdir(parents=True, exist_ok=False)
    mmmu = load_mmmu(args.mmmu_eval)
    reviewed = {}
    for path in args.decisions:
        for line, record in rows(path):
            key = record["qa_key"]
            if key in reviewed and reviewed[key]["decision"] != record["decision"]:
                raise ValueError("Conflicting semantic decisions require explicit root resolution")
            if not record.get("author") or not record.get("model") or not record.get("effort"):
                raise ValueError("Actual semantic reviewer identity is required")
            reviewed[key] = {**record, "decision_path": str(path), "decision_line": line}
    scores, pending, seen, totals = [], {}, set(), defaultdict(Counter)
    for path in args.raw:
        for line, row in rows(path):
            identity = (row["model"], row["key"])
            if identity in seen:
                raise ValueError("Duplicate prediction key")
            seen.add(identity)
            sample, answer = row["sample"], row["text"]
            if sample["dataset"] == "mmmu" and mmmu is None:
                raise ValueError("The pinned official MMMU evaluator is required")
            key = qa_key(sample, answer)
            review = reviewed.get(key)
            if review is not None:
                if review["question"] != sample["prompt"] or review["answer"] != answer:
                    raise ValueError("Reviewed question or full answer mismatch")
                decision = review["decision"]
                source = {field: review.get(field) for field in ("author", "model", "effort", "call_id", "decision_path", "decision_line")}
            else:
                decision = rule_decision(sample, answer)
                source = {"author": "deterministic_rule", "model": None, "effort": None,
                          "rule": None if decision is None else decision.get("decision_source")}
            counter = totals[(row["model"], sample["dataset"])]
            counter["generated"] += 1
            counter["truncated"] += int(not row["terminated"])
            result = {"model": row["model"], "dataset": sample["dataset"], "sample_id": sample["id"],
                      "key": row["key"], "qa_key": key, "question": sample["prompt"], "answer": answer,
                      "gold": sample["gold"], "source_raw": str(path), "source_line": line,
                      "terminated": row["terminated"], "n_tokens": len(row["tokens"]), "wall_s": row["wall_s"],
                      "score": None, "abstain": None, "label": None, "decision_source": source}
            if sample["dataset"] == "pope":
                result["official_parser_prediction"] = pope_official_prediction(answer)
                result["official_parser_score"] = int(result["official_parser_prediction"] == sample["gold"])
            if decision is None:
                counter["pending"] += 1
                item = pending.setdefault(key, {"qa_key": key, "question": sample["prompt"], "answer": answer,
                    "dataset": sample["dataset"], "options": sample.get("options", []), "memberships": []})
                item["memberships"].append({"model": row["model"], "sample_id": sample["id"], "source_raw": str(path), "source_line": line})
            else:
                check_decision(sample, answer, decision)
                result.update(score=correctness(sample, answer, decision, mmmu), abstain=decision["abstain"],
                              label=decision["label"], predicted_answer=decision.get("predicted_answer"),
                              answer_text=decision["answer_text"], evidence_span=decision["evidence_span"])
                counter["resolved"] += 1
                counter["correct"] += result["score"]
                counter["abstentions"] += int(result["abstain"])
                counter["invalid"] += int(result["label"] == "invalid")
            scores.append(result)
    metrics = []
    for (model, dataset), counts in sorted(totals.items()):
        n = counts["generated"]
        metrics.append({"model": model, "dataset": dataset,
            **{name: counts[name] for name in ("generated", "truncated", "pending", "resolved", "correct", "abstentions", "invalid")},
            "accuracy": counts["correct"] / n if counts["pending"] == 0 else None,
            "abstention_rate": counts["abstentions"] / n if counts["pending"] == 0 else None})
    write_rows(args.out / "scores.jsonl", scores)
    write_rows(args.out / "pending_semantic.jsonl", list(pending.values()))
    fields = sorted({field for row in metrics for field in row})
    with (args.out / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    summary = {"generated": len(scores), "scored": sum(s["score"] is not None for s in scores),
               "pending_unique_QA": len(pending), "pending_rows": sum(len(p["memberships"]) for p in pending.values())}
    (args.out / "receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
