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
from workflows.general_vqa_direct.hallusion_scoring import HallusionScoring, OPEN_ANSWER_TYPES, validate_review_author


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig") as stream:
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
                r"on second|instead|rather than|not necessarily|might|may|could)\b")
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
    first_sentence = re.split(r"[.!?\n]", value, maxsplit=1)[0]
    uncertain = bool(re.search(r"\b(?:possibly|perhaps|probably|appears?|seems?|likely)\b", first_sentence, re.I))
    return {"label": "answer_uncertain" if uncertain else "answer_assertive", "abstain": False,
            "answer_text": answer, "predicted_answer": prediction,
            "evidence_span": answer,
            "decision_source": "explicit_binary_full_reply_guard_v2"}


def literal_choice(text, options):
    value = formatted(text)
    letter = re.fullmatch(r"\(?([A-Z])\)?", value, re.I)
    if letter and 0 <= ord(letter[1].upper()) - 65 < len(options):
        return letter[1].upper()
    paired = re.fullmatch(r"\(?([A-Z])\)?[.:]?\s+(.+)", value, re.I)
    if paired:
        index = ord(paired[1].upper()) - 65
        if 0 <= index < len(options) and formatted(paired[2]).casefold() == formatted(str(options[index])).casefold():
            return chr(65 + index)
    indices = [i for i, option in enumerate(options) if formatted(str(option)).casefold() == value.casefold()]
    return chr(65 + indices[0]) if len(indices) == 1 else None


def rule_decision(sample, answer):
    """Only whole-response, unambiguous formats are automatic."""
    literal = lexical_label(answer)
    if literal is not None:
        return {"label": literal, "abstain": literal == "abstain", "answer_text": "",
                "predicted_answer": None, "evidence_span": answer,
                "decision_source": "existing_kdm_whole_response_rule"}
    value = formatted(answer)
    dataset = sample["dataset"]
    binary = dataset == "pope" or (dataset == "hallusionbench" and sample.get("source_id") not in OPEN_ANSWER_TYPES)
    if binary and value.casefold() in {"yes", "no"}:
        return {"label": "answer_assertive", "abstain": False, "answer_text": answer,
                "predicted_answer": value.casefold(), "evidence_span": answer,
                "decision_source": "complete_literal_yes_no"}
    if binary:
        binary = explicit_binary_decision(answer)
        if binary is not None:
            return binary
    options = sample.get("options", [])
    if options:
        prediction = literal_choice(value, options)
        if prediction is not None:
            return {"label": "answer_assertive", "abstain": False, "answer_text": answer,
                    "predicted_answer": prediction, "evidence_span": answer,
                    "decision_source": "complete_literal_choice_letter_or_original_text"}
        markers = list(re.finditer(r"\b(?:the\s+(?:(?:correct|final)\s+)?answer\s+is\s*:?|(?:final\s+)?answer\s*:|option\s+(?=\(?[A-Z]\)?\s*[.!]?\s*$))\s*", answer, re.I))
        if markers:
            last = markers[-1]
            fragment = answer[last.end():].strip()
            prediction = literal_choice(fragment, options)
        if markers and prediction is not None:
            return {"label": "answer_assertive", "abstain": False, "answer_text": fragment,
                    "predicted_answer": prediction, "evidence_span": answer[last.start():].strip(),
                    "decision_source": "explicit_final_answer_matching_original_choice"}
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
        binary = sample["dataset"] == "pope" or (sample["dataset"] == "hallusionbench" and sample.get("source_id") not in OPEN_ANSWER_TYPES)
        if binary and prediction not in {"yes", "no", None}:
            raise ValueError("Binary question prediction must be yes, no or explicitly unparsed")
        if sample.get("options") and prediction not in [chr(65+i) for i in range(len(sample["options"]))] + [None]:
            raise ValueError("Prediction is outside this question's original choices")


def correctness(sample, answer, decision, mmmu):
    if sample["dataset"] == "hallusionbench":
        raise ValueError("Hallusion quality must use the full official reference independently of behavior")
    if decision["label"] in {"abstain", "invalid"}:
        return 0
    prediction = decision.get("predicted_answer")
    if prediction is None:
        return 0
    if sample["dataset"] == "pope":
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
    parser.add_argument("--annotations", type=Path, help="Completed disjoint batch directories")
    parser.add_argument("--quality-annotations", type=Path, help="Actual reference-aware Hallusion Luna batches")
    parser.add_argument("--mmmu-eval", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.raw_list:
        args.raw.extend(Path(line) for line in args.raw_list.read_text().splitlines() if line.strip())
    if not args.raw or len(args.raw) != len(set(args.raw)):
        parser.error("At least one unique validated raw file is required")
    if args.annotations:
        for original in sorted(args.annotations.glob("*/decisions.jsonl")):
            selection = original.with_name("ACTIVE.json")
            if selection.exists():
                selected = json.loads(selection.read_text())
                path = original.parent / selected["filename"]
                if path.parent != original.parent or hashlib.sha256(path.read_bytes()).hexdigest() != selected["sha256"]:
                    raise ValueError("Active annotation selection has a foreign path or mismatched hash")
                args.decisions.append(path)
                continue
            corrected = original.with_name("decisions_corrected.jsonl")
            if corrected.exists():
                args.decisions.append(corrected)
            elif not original.with_name("REJECTED.json").exists():
                args.decisions.append(original)
    if len(args.decisions) != len(set(args.decisions)):
        parser.error("A decision file was supplied twice")
    args.out.mkdir(parents=True, exist_ok=False)
    mmmu = load_mmmu(args.mmmu_eval)
    hallusion = HallusionScoring(ROOT)
    quality_reviews = {}
    if args.quality_annotations:
        for original in sorted(args.quality_annotations.glob("*/quality_decisions.jsonl")):
            path = original
            selection = original.with_name("ACTIVE.json")
            if selection.exists():
                selected = json.loads(selection.read_text())
                path = original.parent / selected["filename"]
                if path.parent != original.parent or hashlib.sha256(path.read_bytes()).hexdigest() != selected["sha256"]:
                    raise ValueError("Active quality selection has a foreign path or mismatched hash")
            elif original.with_name("REJECTED.json").exists():
                continue
            for line, record in rows(path):
                key = record["quality_key"]
                if key in quality_reviews:
                    raise ValueError("A quality key was reviewed in more than one active batch")
                quality_reviews[key] = {**record, "decision_path": str(path), "decision_line": line}
    reviewed = {}
    for path in args.decisions:
        for line, record in rows(path):
            key = record["qa_key"]
            if key in reviewed and reviewed[key]["decision"] != record["decision"]:
                raise ValueError("Conflicting semantic decisions require explicit root resolution")
            if not record.get("author") or not record.get("model") or not record.get("effort"):
                raise ValueError("Actual semantic reviewer identity is required")
            validate_review_author(record["author"])
            reviewed[key] = {**record, "decision_path": str(path), "decision_line": line}
    scores, pending, pending_quality, seen, totals = [], {}, {}, set(), defaultdict(Counter)
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
                      "identity": row["identity"], "dataset_identity": row["dataset_identity"],
                      "claim_identity": row["claim_identity"],
                      "engine": row.get("engine", "registered_hf_session"),
                      "gold": sample["gold"], "source_raw": str(path), "source_line": line,
                      "terminated": row["terminated"], "n_tokens": len(row["tokens"]), "wall_s": row["wall_s"],
                      "score": None, "abstain": None, "label": None, "decision_source": source}
            if sample["dataset"] == "pope":
                result["official_parser_prediction"] = pope_official_prediction(answer)
                result["official_parser_score"] = int(result["official_parser_prediction"] == sample["gold"])
            if sample["dataset"] == "hallusionbench":
                quality_key = hallusion.quality_key(sample, answer)
                quality_review = quality_reviews.get(quality_key)
                if quality_review is not None:
                    quality = hallusion.validate_quality(sample, answer, quality_review)
                    quality.update(decision_path=quality_review["decision_path"], decision_line=quality_review["decision_line"])
                else:
                    quality = hallusion.literal_binary_quality(sample, answer)
                result.update(quality_key=quality_key, quality=quality,
                              gt_answer_details=sample["gt_answer_details"], question_kind=hallusion.question_kind(sample))
                if quality is not None:
                    result["score"] = quality["score"]
                else:
                    request = hallusion.quality_request(sample, answer)
                    item = pending_quality.setdefault(quality_key, request)
                    item["memberships"].append({"model": row["model"], "sample_id": sample["id"], "source_raw": str(path), "source_line": line})
            if decision is None:
                counter["pending_behavior"] += 1
                item = pending.setdefault(key, {"qa_key": key, "question": sample["prompt"], "answer": answer,
                    "dataset": sample["dataset"], "options": sample.get("options", []),
                    "question_kind": hallusion.question_kind(sample) if sample["dataset"] == "hallusionbench" else sample.get("question_type"), "memberships": []})
                item["memberships"].append({"model": row["model"], "sample_id": sample["id"], "source_raw": str(path), "source_line": line})
            else:
                try:
                    check_decision(sample, answer, decision)
                except ValueError as error:
                    raise ValueError(f"{error}; qa_key={key}; source={source}") from error
                if sample["dataset"] == "hallusionbench":
                    hallusion.check_behavior_prediction(sample, decision.get("predicted_answer"))
                else:
                    result["score"] = correctness(sample, answer, decision, mmmu)
                result.update(abstain=decision["abstain"],
                              label=decision["label"], predicted_answer=decision.get("predicted_answer"),
                              answer_text=decision["answer_text"], evidence_span=decision["evidence_span"])
                counter["abstentions"] += int(result["abstain"])
                counter["invalid"] += int(result["label"] == "invalid")
            counter["pending_quality"] += int(result["score"] is None)
            counter["correct"] += result["score"] or 0
            complete = result["score"] is not None and result["abstain"] is not None
            counter["resolved" if complete else "pending"] += 1
            scores.append(result)
    metrics = []
    for (model, dataset), counts in sorted(totals.items()):
        n = counts["generated"]
        metrics.append({"model": model, "dataset": dataset,
            **{name: counts[name] for name in ("generated", "truncated", "pending", "pending_behavior", "pending_quality", "resolved", "correct", "abstentions", "invalid")},
            "accuracy": counts["correct"] / n if counts["pending_quality"] == 0 else None,
            "abstention_rate": counts["abstentions"] / n if counts["pending_behavior"] == 0 else None})
    write_rows(args.out / "scores.jsonl", scores)
    write_rows(args.out / "pending_semantic.jsonl", list(pending.values()))
    write_rows(args.out / "pending_quality.jsonl", list(pending_quality.values()))
    fields = sorted({field for row in metrics for field in row})
    with (args.out / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    summary = {"generated": len(scores), "scored": sum(s["score"] is not None and s["abstain"] is not None for s in scores),
               "quality_scored": sum(s["score"] is not None for s in scores),
               "behavior_scored": sum(s["abstain"] is not None for s in scores),
               "pending_unique_QA": len(pending), "pending_rows": sum(len(p["memberships"]) for p in pending.values()),
               "pending_quality_keys": len(pending_quality), "pending_quality_rows": sum(len(p["memberships"]) for p in pending_quality.values())}
    (args.out / "receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
