#!/usr/bin/env python3
"""Derived full-denominator metrics; never let missing annotations disappear."""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
from annotate import atomic_json, digest, file_hash, jsonlines

def score_run(root, out):
    sys.path.insert(0, str(root / "src"))
    from kdm.scoring import food_correct, vqa_score
    from kdm.models.official_vqa_normalizer import VQAEval
    evaluator = VQAEval(None, None)
    normalize = lambda text: evaluator.processDigitArticle(evaluator.processPunctuation(text))
    aliases = json.loads((root / "configs/kdm/food_aliases.json").read_text())
    identity = json.loads((out / "identity.json").read_text())
    expected_hashes = identity["definition"]
    for field, path in [
        ("frozen_scoring_sha256", root / "src/kdm/scoring.py"),
        ("frozen_aliases_sha256", root / "configs/kdm/food_aliases.json"),
        ("frozen_vqa_normalizer_sha256", root / "src/kdm/models/official_vqa_normalizer.py")]:
        if file_hash(path) != expected_hashes[field]: raise ValueError("Frozen scoring source changed")
    labels = {}
    for label in jsonlines(out / "labels.jsonl"):
        if label["identity"] != identity["identity"] or label["key"] in labels: raise ValueError("Invalid annotation identity or duplicate")
        labels[label["key"]] = label
    summary = defaultdict(lambda: {"expected": 0, "annotated": 0, "unresolved": 0,
        "abstain": 0, "invalid": 0, "answer_assertive": 0, "answer_uncertain": 0, "correct_score_sum": 0.0})
    consumed = set()
    receipt = json.loads((out / "queue.sources.json").read_text())
    for source in receipt["sources"]:
        path = root / source["path"]
        if file_hash(path) != source["sha256"]: raise ValueError("Original census has changed since queue creation")
        for raw in jsonlines(path):
            sample = raw["sample"]; label = labels.get(raw["key"]); value = 0.0
            groups = [(raw["model"], sample["dataset"], raw["guided"], sample["split"]),
                (raw["model"], sample["dataset"], raw["guided"], "all")]
            if label:
                if label["text"] != raw["text"] or label["raw_record_sha256"] != digest(raw): raise ValueError("Annotation does not bind to original response")
                consumed.add(raw["key"])
                if label["label"] in {"answer_assertive", "answer_uncertain"}:
                    value = float(food_correct(label["answer_text"], sample["class"], aliases)) if sample["dataset"] == "food101" else float(vqa_score(label["answer_text"], sample["gold"], normalize))
            for group in groups:
                counts = summary[group]; counts["expected"] += 1
                if label:
                    counts["annotated"] += 1; counts[label["label"]] += 1; counts["correct_score_sum"] += value
                else: counts["unresolved"] += 1
    if consumed != set(labels): raise ValueError("Annotations contain unknown census keys")
    rows = []
    for (model, dataset, guided, split), counts in sorted(summary.items()):
        n = counts["expected"]; u = counts["unresolved"]
        rows.append({"model": model, "dataset": dataset, "guided": guided, "split": split, **counts,
            "abstention_rate_lower": counts["abstain"]/n,
            "abstention_rate_upper": (counts["abstain"]+u)/n,
            "accuracy_lower": counts["correct_score_sum"]/n,
            "accuracy_upper": (counts["correct_score_sum"]+u)/n,
            "abstention_rate": counts["abstain"]/n if u == 0 else None,
            "accuracy": counts["correct_score_sum"]/n if u == 0 else None})
    report = {"schema": "deepseek_v2_full_denominator_scores", "annotation_identity": identity["identity"],
        "labels_sha256": file_hash(out / "labels.jsonl"), "queue_sources_sha256": file_hash(out / "queue.sources.json"),
        "scoring": "Frozen Food aliases and official VizWiz leave-one-annotator-out score. Abstain/invalid count zero. Unresolved rows remain in every denominator.",
        "limits": "Bounds reflect unresolved annotation only, not judge semantic error. Automatic annotation is not human review or model selection.",
        "rows": rows}
    atomic_json(out / "metrics.json", report)
    return report
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(); score_run(Path(a.root).resolve(), Path(a.out).resolve())

