#!/usr/bin/env python3
"""Score sealed native baseline parts with existing canonical and QA authorities."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from kdm.io import file_hash, read_jsonl, stable_hash, within
from workflows.main_results import score as frozen
from workflows.paper_core.score_native import load_references, save_json, save_rows
from workflows.paper_core.score_dev_viz import native_decisions, NATIVE_RECEIPT, AUTHORITY
from workflows.supplemental.remaining11.score import infer_qa, load_authority, rows, score_target

FIELDS = ("method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")
COND = ("model", "dataset", "split", *FIELDS)
COMPACT = "outputs/paper_core_20261002_dev_viz/native_scoring_compact_references_20261003_0200"


def reference_map():
    _, result = load_references()
    paths = [{"path": "outputs/paper_20260929/references.parquet",
              "sha256": file_hash(ROOT / "outputs/paper_20260929/references.parquet")}]
    folder = ROOT / COMPACT
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
    path = folder / "reference_rows.jsonl"
    if (receipt["passed"] is not True or receipt["reference_pending"] != 0
            or receipt["rows"] != 19392 or file_hash(path) != receipt["outputs"][path.name]):
        raise ValueError("The source-bound compact extension references differ")
    for r in read_jsonl(path):
        if r["split"] == "eval":
            key = (r["model"], r["sample_id"])
            if key in result or r["reference_complete"] is not True or type(r["reference_G"]) is not bool:
                raise ValueError("An eval reference is duplicated or undecided")
            result[key] = r["reference_G"]
    if len(result) != 9 * 2424:
        raise ValueError("Nine-model eval reference coverage differs")
    paths.append({"path": str(path.relative_to(ROOT)), "sha256": receipt["outputs"][path.name],
                  "original_source_path": receipt["source_path"],
                  "original_source_sha256": receipt["source_sha256"]})
    return result, paths


def condition_metrics(records):
    groups = defaultdict(list)
    for r in records:
        groups[tuple(r[k] for k in COND)].append(r)
    metrics = []
    for key, group in groups.items():
        n = len(group)
        known = [r for r in group if r["canonical_name_in_primary_score"] in (0, 1)
                 and type(r["abstain"]) is bool]
        c = sum(r["canonical_name_in_primary_score"] == 1 for r in known)
        a = sum(r["abstain"] for r in known)
        tp = sum(r["abstain"] and r["uniform_reference"] for r in known)
        fp = a - tp
        fn = sum(not r["abstain"] and r["uniform_reference"] for r in known)
        rplus = sum(r["uniform_reference"] for r in group)
        complete = n == 2424 and len(known) == n
        value = dict(zip(COND, key))
        value.update(n=n, expected_n=2424, decided_rows=len(known), primary_complete=complete,
                     C=c, W_decided=len(known) - c - a, A=a, TP=tp, FP=fp, FN=fn,
                     reference_positive=rplus, J_numerator=c + tp, J_denominator=2424,
                     J=(c + tp) / 2424 if complete else None,
                     accuracy=c / 2424 if complete else None,
                     precision=tp / a if complete and a else None,
                     recall=tp / rplus if complete and rplus else None,
                     F1=2 * tp / (a + rplus) if complete and a + rplus else None,
                     metric_null_reason="" if complete else "condition_keys_or_semantic_labels_incomplete",
                     precision_null_reason="" if a else "zero_abstention_denominator",
                     recall_null_reason="" if rplus else "zero_reference_positive_denominator")
        metrics.append(value)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="append", required=True)
    parser.add_argument("--decision-file", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = within(ROOT, args.output)
    output.relative_to(ROOT / "outputs/paper_core_20261002_dev_viz")
    gt, references = reference_map()
    manifest_path = ROOT / AUTHORITY
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_authority = {"census_final_labels": manifest["census_final_labels"], "historical_labels": []}
    canonical_sha = file_hash(Path(frozen.__file__))
    existing_paths, authority_receipts = native_decisions(
        NATIVE_RECEIPT, references[0]["sha256"], canonical_sha)
    decision_paths = existing_paths + [within(ROOT, p) for p in args.decision_file]
    reviews, behavior, history, _, decisions = load_authority(output, source_authority, decision_paths)
    samples = {s["id"]: s for s in read_jsonl(ROOT / "data/current/all.jsonl")
               if s["dataset"] == "food101" and s["split"] == "eval"}
    patterns = frozen.compile_classes(sorted({s["class"] for s in samples.values()}))
    scored, pending, cache, seen, sources = [], {}, {}, set(), []
    for relative in args.snapshot:
        snapshot_path = within(ROOT, relative)
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot["schema"] != "kdm_sealed_native_baseline_snapshot_v1" or snapshot["passed"] is not True:
            raise ValueError("The native snapshot lacks an actual source validation receipt")
        sources.append({"path": str(snapshot_path.relative_to(ROOT)), "sha256": file_hash(snapshot_path),
                        "receipt": snapshot})
        for part in snapshot["parts"]:
            raw = within(ROOT, part["sealed_raw_path"])
            if file_hash(raw) != part["raw_sha256"]:
                raise ValueError("An immutable native part differs from its received source SHA")
            count = 0
            for line, row, line_sha in rows(raw):
                s = row["sample"]
                model = row["model"]
                if (row["key"] in seen or s != samples.get(s["id"])
                        or model != part["model"] or row["method"] != part["method"]
                        or row["identity"] != part["ledger_identity"]):
                    raise ValueError("A native score source overlaps or differs from its snapshot")
                seen.add(row["key"])
                count += 1
                qkey = frozen.qah(s["question"], row["text"])
                if qkey not in cache:
                    cache[qkey] = infer_qa(s["question"], row["text"], patterns, reviews, behavior, decisions)
                inferred = cache[qkey]
                canonical, literal, reason = score_target(row["text"], s["class"], inferred, patterns)
                record = {"model": model, "dataset": "food101", "split": "eval",
                          **{k: row[k] for k in FIELDS}, "main_marker": row["marker"],
                          "sample_id": s["id"], "key": row["key"], "qa_key": qkey,
                          "question": s["question"], "answer": row["text"], "target_class": s["class"],
                          "canonical_name_in_primary_score": canonical,
                          "literal_extracted_name_score": literal, "abstain": inferred["abstain"],
                          "uniform_reference": gt[(model, s["id"])], "reference_complete": True,
                          "score_reason": reason, "primary_extraction": inferred["parsed"],
                          "behavior_source": inferred["behavior_source"],
                          "behavior_history_sources": history.get(qkey, []),
                          "decision": inferred["decision"], "source_path": str(raw), "source_line": line,
                          "raw_line_sha256": line_sha, "raw_source_sha256": part["raw_sha256"],
                          "source_original_path": part["source_raw_path"], "source_host": part["source_hostname"],
                          "source_identity": row["identity"], "snapshot_receipt": str(snapshot_path),
                          "snapshot_receipt_sha256": sources[-1]["sha256"], "tokens": row["tokens"],
                          "terminated": row["terminated"], "seed": row["seed"], "config": row["config"],
                          "prompt": row["prompt"], "reference_prompt": row["reference_prompt"],
                          "annotation_model": inferred["annotation_model"],
                          "annotation_effort": inferred["annotation_effort"],
                          "annotation_call_id": inferred["annotation_call_id"]}
                scored.append(record)
                if canonical is None or inferred["abstain"] is None:
                    p = pending.setdefault(qkey, {"qa_key": qkey, "question": s["question"],
                                                   "answer": row["text"], "reasons": [], "memberships": []})
                    p["reasons"] = sorted(set(p["reasons"] + [reason, inferred["behavior_source"]]))
                    p["memberships"].append({"model": model, "sample_id": s["id"], "key": row["key"],
                                              "source_path": str(raw), "source_line": line,
                                              "raw_line_sha256": line_sha})
            if count != part["rows"]:
                raise ValueError("A native immutable part row count differs")
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "score_rows.jsonl.gz", scored)
    save_rows(output / "pending_complete_QA.jsonl", [pending[k] for k in sorted(pending)])
    values = condition_metrics(scored)
    pd.DataFrame(values).to_csv(output / "metrics_all.csv", index=False)
    receipt = {"schema": "kdm_native_baseline_parts_score_v1", "passed": True,
               "rows": len(scored), "unique_keys": len(seen), "unique_QA": len(cache),
               "canonical_pending": sum(r["canonical_name_in_primary_score"] is None for r in scored),
               "literal_pending": sum(r["literal_extracted_name_score"] is None for r in scored),
               "abstain_pending": sum(r["abstain"] is None for r in scored),
               "pending_unique_QA": len(pending), "reference_pending": 0,
               "complete_conditions": sum(v["primary_complete"] for v in values),
               "source_parts": sources, "reference_sources": references,
               "accepted_native_authority_receipts": authority_receipts,
               "actual_command": [sys.executable, *sys.argv],
               "additional_decision_sources": [
                   {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}
                   for path in decision_paths[len(existing_paths):]],
               "authority_manifest_sha256": file_hash(manifest_path),
               "canonical_scorer_sha256": canonical_sha, "runner_sha256": file_hash(Path(__file__)),
               "new_semantic_judgments_generated": 0, "GPU_initialized": False,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in
                      {"source_parts", "reference_sources", "accepted_native_authority_receipts", "outputs"}}))


if __name__ == "__main__":
    main()
