#!/usr/bin/env python3
"""Append six root primary-name decisions and derive a complete native Direct table."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workflows.main_results import score as frozen_score

MODELS = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b")
file_hash = frozen_score.sha


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def save_rows(path, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "xt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def save_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None

DECISIONS = {
    "fb18cbc3f4387529a23388f607ebf50166ed348204911418d9b2736f83fbb113":
        ("chocolate cake", "unique_main", "dessert plate with a chocolate cake, a scoop of ice cream, and a small piece of fruit", "蛋糕为主菜，冰淇淋球与水果为配菜。"),
    "685bf5f89aa3df2b6920c7ce3ecbdbc49fcf61528f1c07b21bb9442b1c690a3c":
        ("pita chips", "unique_main", "pita chips topped with hummus, olives, and nuts", "pita chips为主体；topped with之后均为配料。"),
    "cbfd8d9403bb6366e36b81d0e4b6d2230f8da209b0409a7acc8f044a1b4b5e35":
        ("Pita bread", "unique_main", "Pita bread with hummus and a slice of tomato", "回答以Pita bread为主菜，hummus与番茄为配菜。"),
    "603b5cc31bab20e633c29bfa557f818ae9a78f99426872dc10ed82c7629fd14b":
        ("Salad", "unique_main", "Salad with seared scallops", "主菜为泛称Salad；seared scallops在with配菜位置。"),
    "a0afa1e4131206fd66a9bc189fc5a7c31ea819a259357f3373efde20d1bccf9a":
        ("Japanese meal", "generic", "Japanese meal with a variety of vegetables and seaweed salad", "泛称Japanese meal是作答主体，没有明确指定唯一规范主菜；后续蔬菜与海藻沙拉为组成说明。"),
    "2e57ebbe9ae774c0eeacc00e9fb6e8b286cacaf4319e4aecf395af64217c0620":
        ("Asian-inspired stir-fry", "unique_main", "Asian-inspired stir-fry with steak, shrimp", "明确主菜为Asian-inspired stir-fry，steak与shrimp为原料；appears保留作答。"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--luna", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = list(read_rows(args.input))
    luna = {row["qa_key"]: row for row in read_rows(args.luna)}
    pending = {row["original_key"]: row for row in rows if row["correct_canonical"] is None}
    if set(pending) != set(DECISIONS) or set(luna) != set(DECISIONS):
        raise ValueError("Expected exactly the six source-bound pending replies")
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    patterns = frozen_score.compile_classes(manifest["canonical_classes"])
    reviews = {}
    for key, (name, kind, span, reason) in DECISIONS.items():
        row, original = pending[key], luna[key]
        if (row["question"], row["answer"]) != (original["question"], original["answer"]):
            raise ValueError("Luna question/complete answer differs from the actual source")
        if name not in row["answer"] or span not in row["answer"]:
            raise ValueError("Root primary/evidence span must occur in the complete answer")
        if row["abstain"] is not False or original["abstain"] is not False:
            raise ValueError("This finite root review resolves primary names only")
        decision = {
            "key": key, "question": row["question"], "answer": row["answer"],
            "source_identity": row["source_identity"], "source_path": row["source_path"],
            "source_line": row["source_line"], "source_luna_decision": original,
            "endorsed_primary_names": [name], "name_relations": {name: "main"},
            "primary_kind": kind, "name_scope_ambiguous": False, "abstain": False,
            "evidence_span": span, "primary_name_span": name, "reason": reason,
            "root_review_author": {"agent": "/root", "model": "gpt-6.1-sol",
                "effort": "max", "call_id": ""},
        }
        canonical, literal, score_reason = frozen_score.score_variant(
            row["answer"], row["target_class"], decision, patterns)
        if canonical is None or literal is None:
            raise ValueError("A completed explicit root review must resolve both scores")
        reviews[key] = (decision, canonical, literal, score_reason)
    updated = []
    for row in rows:
        if row["original_key"] in reviews:
            review, canonical, literal, reason = reviews[row["original_key"]]
            row = {**row, "prior_pending_score": {
                key: row[key] for key in ("correct_canonical", "correct_literal", "reason", "primary_extraction")},
                "correct_canonical": canonical, "correct_literal": literal,
                "reason": reason, "root_primary_review": review,
                "primary_extraction": {"status": review["primary_kind"],
                    "primary_name": review["endorsed_primary_names"][0],
                    "canonical_candidates": frozen_score.classes_for_name(
                        review["endorsed_primary_names"][0], patterns),
                    "rule": "explicit_root_review_of_complete_response"}}
        updated.append(row)
    if len(updated) != 12120 or len({(r["model"], r["sample_id"]) for r in updated}) != 12120:
        raise ValueError("Native Direct eval coverage differs")
    if any(r["correct_canonical"] is None or r["abstain"] is None for r in updated):
        raise ValueError("A pending main score cannot enter a completed table")
    quotas = Counter((r["model"], r["target_class"]) for r in updated)
    if len(quotas) != 505 or set(quotas.values()) != {24}:
        raise ValueError("Expected 101 classes by 24 eval inputs for every model")
    metrics = []
    for model in MODELS:
        group = [r for r in updated if r["model"] == model]
        correct = sum(r["correct_canonical"] for r in group)
        abstentions = sum(r["abstain"] for r in group)
        positives = sum(r["uniform_reference"] for r in group)
        tp = sum(r["abstain"] and r["uniform_reference"] for r in group)
        fp, fn = abstentions-tp, positives-tp
        metrics.append({"model": model, "dataset": "food101", "split": "eval",
            "method": "direct", "kind": "native_unguided", "marker": "NONE",
            "reference_marker": "NONE", "guided": False, "reference_guided": False,
            "replicate": 0, "n": len(group), "correct": correct,
            "accuracy": fraction(correct, len(group)), "abstentions": abstentions,
            "answer_errors": len(group)-correct-abstentions, "reference_positive": positives,
            "tp": tp, "fp": fp, "fn": fn, "precision": fraction(tp, abstentions),
            "recall": fraction(tp, positives), "f1": fraction(2*tp, abstentions+positives),
            "unnecessary_abstention_rate": fraction(fp, len(group)-positives),
            "precision_null_reason": "no_abstentions" if not abstentions else None,
            "primary_pending": 0, "abstention_pending": 0})
    args.out.mkdir(parents=True, exist_ok=False)
    save_rows(args.out / "root_primary6_review.jsonl", [r[0] for r in reviews.values()])
    save_rows(args.out / "score_rows.jsonl.gz", updated)
    with (args.out / "native_direct_conditions.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    receipt = {"schema": "kdm_native_direct_root_six_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": 12120, "unique_model_sample_keys": 12120, "root_review_rows": 6,
        "pending": 0, "source_scores": str(args.input),
        "source_scores_sha256": file_hash(args.input), "luna_source": str(args.luna),
        "luna_source_sha256": file_hash(args.luna),
        "frozen_scorer_sha256": file_hash(Path(frozen_score.__file__)),
        "new_generations": 0, "new_API_calls": 0, "GPU_initialized": False}
    save_json(args.out / "receipt.json", receipt)
    print(json.dumps({"receipt": receipt, "metrics": metrics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
