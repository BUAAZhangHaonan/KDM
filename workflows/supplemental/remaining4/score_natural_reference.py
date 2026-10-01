#!/usr/bin/env python3
"""CPU-score the sealed four-model representative noisy-image replies."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash, stable_seed, within
from kdm.prompts import task_prompt
from workflows.main_results import score as frozen
from workflows.paper_core.score_native import save_rows
from workflows.supplemental.remaining11.score import infer_qa, load_authority, rows, score_target

MODELS = ("internvl35_8b", "onevision", "phi35", "qwen3vl")
AUTHORITY = "outputs/supplemental/remaining11/run_20260930_140337/assets/asset_manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authority-manifest", default=AUTHORITY)
    parser.add_argument("--decision-file", action="append", default=[])
    args = parser.parse_args()
    source, output = within(ROOT, args.input_dir), within(ROOT, args.output)
    verified_path = within(ROOT, args.verification)
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    if (verified["status"] != "passed"
            or verified["representative_counts"] != {model: 101 for model in MODELS}
            or verified["natural_counts"] != {model: 101 for model in MODELS}):
        raise ValueError("Four-model representative GPU export has not passed coverage checks")
    output.mkdir(parents=True, exist_ok=False)
    samples = {sample["id"]: sample for sample in read_jsonl(ROOT / "data/current/all.jsonl")
               if sample["dataset"] == "food101" and sample["split"] == "eval"}
    patterns = frozen.compile_classes(sorted({sample["class"] for sample in samples.values()}))
    authority_path = within(ROOT, args.authority_manifest)
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority = {"census_final_labels": authority["census_final_labels"], "historical_labels": []}
    decision_paths = [within(ROOT, value) for value in args.decision_file]
    reviews, behavior, _, _, decisions = load_authority(source, authority, decision_paths)
    records, pending, cache, receipts = [], {}, {}, []
    counts, seen = Counter(), set()
    for member in verified["natural_sources"]:
        folder = within(ROOT, member["directory"])
        if folder.parent.resolve() != source.resolve():
            raise ValueError("Verified natural-reference member is outside the selected source")
        raw = folder / "natural_reference.events.jsonl"
        if file_hash(raw) != member["sha256"]:
            raise ValueError("Sealed natural-reference bytes changed")
        meta_path = raw.with_suffix(".identity.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if stable_hash(meta["definition"]) != meta["identity"]:
            raise ValueError("Natural-reference ledger identity changed")
        part_rows = 0
        for line, row, line_sha in rows(raw):
            model, sid = row["model"], row["sample_id"]
            sample = samples[sid]
            if (model not in MODELS or (model, sid) in seen or row["status"] != "ok"
                    or row["identity"] != meta["identity"] or row["sample"] != sample
                    or row["source"]["sample"] != sample
                    or row["seed"] != stable_seed(sid, model, 0)
                    or row["prompt"] != task_prompt(sample["question"], "NONE", False)):
                raise ValueError("Natural-reference original sample/seed/prompt/identity differs")
            seen.add((model, sid))
            generated = row["noise_generated_r"]
            if generated["status"] != "ok":
                raise ValueError("Actual natural-reference generation failed")
            answer = generated["text"]
            qkey = frozen.qah(sample["question"], answer)
            if qkey not in cache:
                cache[qkey] = infer_qa(sample["question"], answer, patterns, reviews, behavior, decisions)
            inferred = cache[qkey]
            canonical, literal, reason = score_target(answer, sample["class"], inferred, patterns)
            record = {"model": model, "dataset": "food101", "split": "eval", "sample_id": sid,
                      "method": "direct", "kind": "natural_noisy_reference", "main_marker": "NONE",
                      "reference_marker": "NONE", "guided": False, "reference_guided": False,
                      "noise_step": 500, "seed": row["seed"], "qa_key": qkey,
                      "question": sample["question"], "answer": answer, "target_class": sample["class"],
                      "canonical_name_in_primary_score": canonical, "literal_extracted_name_score": literal,
                      "abstain": inferred["abstain"], "uniform_reference": None,
                      "reference_join_status": "awaiting_model_sample_reference_table",
                      "score_reason": reason, "behavior_source": inferred["behavior_source"],
                      "tokens": generated["tokens"], "terminated": generated["terminated"],
                      "config": row["config"],
                      "source": {"path": str(raw.relative_to(ROOT)), "line": line,
                                 "line_sha256": line_sha, "identity": row["identity"], "key": row["key"]},
                      "accepted_decision_path": inferred["decision"].get("decision_source_path")
                                                if inferred["decision"] else None}
            records.append(record)
            counts[model] += 1
            part_rows += 1
            if canonical is None or literal is None or inferred["abstain"] is None or inferred["name_boundary"]:
                item = pending.setdefault(qkey, {"qa_key": qkey, "question": sample["question"],
                                                "answer": answer, "source_members": []})
                item["source_members"].append({"model": model, "sample_id": sid,
                                               "target_class": sample["class"], "source": record["source"]})
        if part_rows != member["rows"]:
            raise ValueError("Sealed natural-reference row count differs")
        receipts.append({"path": str(raw.relative_to(ROOT)), "rows": part_rows,
                         "sha256": member["sha256"], "identity_sha256": file_hash(meta_path)})
    if counts != Counter({model: 101 for model in MODELS}):
        raise ValueError("Natural-reference four-model coverage differs")
    for model in MODELS:
        classes = Counter(samples[sid]["class"] for candidate, sid in seen if candidate == model)
        if len(classes) != 101 or set(classes.values()) != {1}:
            raise ValueError("Natural-reference class quotas differ")
    save_rows(output / "score_rows.jsonl.gz", records)
    save_rows(output / "pending_QA.jsonl", list(pending.values()))
    import pandas as pd
    pd.DataFrame(records).drop(columns=["source", "tokens", "config"]).to_parquet(output / "new_scores.parquet", index=False)
    metrics = []
    for model in MODELS:
        subset = [record for record in records if record["model"] == model]
        c_pending = sum(record["canonical_name_in_primary_score"] is None for record in subset)
        a_pending = sum(record["abstain"] is None for record in subset)
        correct = sum(record["canonical_name_in_primary_score"] == 1 for record in subset)
        abstentions = sum(record["abstain"] is True for record in subset)
        metrics.append({"model": model, "scope": "representative101_natural_reference", "n": 101,
                        "correct": correct, "accuracy": correct / 101 if not c_pending else None,
                        "abstentions": abstentions, "abstention_rate": abstentions / 101 if not a_pending else None,
                        "canonical_pending": c_pending,
                        "literal_pending": sum(record["literal_extracted_name_score"] is None for record in subset),
                        "abstain_pending": a_pending, "reference_pending": 101,
                        "precision": None, "recall": None,
                        "reference_metric_null_reason": "reference_not_joined"})
    pd.DataFrame(metrics).to_csv(output / "natural_reference_metrics.csv", index=False)
    receipt = {"schema": "kdm_remaining4_natural_reference_scoring_v1", "rows": len(records),
               "models": dict(counts), "boundary_QA": len(pending), "sources": receipts,
               "canonical_pending": sum(record["canonical_name_in_primary_score"] is None for record in records),
               "literal_pending": sum(record["literal_extracted_name_score"] is None for record in records),
               "abstain_pending": sum(record["abstain"] is None for record in records), "reference_pending": 404,
               "authority_manifest_sha256": file_hash(authority_path),
               "verification_sha256": file_hash(verified_path), "scorer_sha256": file_hash(Path(__file__)),
               "decision_files": [{"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)} for path in decision_paths],
               "GPU_initialized": False, "new_generations": 0, "new_API_calls": 0}
    atomic_json(output / "scoring_receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("rows", "models", "boundary_QA", "canonical_pending",
                                                   "literal_pending", "abstain_pending", "reference_pending")}))


if __name__ == "__main__":
    main()
