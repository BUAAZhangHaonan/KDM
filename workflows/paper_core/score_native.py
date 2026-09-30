#!/usr/bin/env python3
"""Score immutable core native VCD with frozen canonical functions and references."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.decoding import DecodeConfig
from kdm.io import file_hash, stable_hash, within
from kdm.pipeline import task_id
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.score import load_authority, infer_qa, rows, score_target
from workflows.paper_core.receive_native import MODELS, validate_received

COND = ("model", "dataset", "split", "method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")
REFERENCE_PATH = "outputs/paper_20260929/references.parquet"
DEFAULT_DIRECT = "outputs/paper_core_20260930/run_20260930_core_p0/native_direct_root6_20260930"


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_rows(path, values):
    opener = gzip.open if path.suffix == ".gz" else open
    options = {"compresslevel": 1} if path.suffix == ".gz" else {}
    with opener(path, "xt", encoding="utf-8", newline="\n", **options) as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def load_references():
    path = ROOT / REFERENCE_PATH
    frame = pd.read_parquet(path)
    if (len(frame) != 24240 or frame.duplicated(["model", "sample_id"]).any()
            or set(frame.model) != set(MODELS) or not frame.independent_attempts.eq(10).all()
            or not ((frame.gold_rank > 1) & frame.independent_correct_attempts.eq(0)).eq(frame.uniform_reference).all()):
        raise ValueError("Frozen uniform references differ from the registered complete ten-attempt rule")
    selected = frame[frame.split.eq("eval")]
    if len(selected) != 12120 or selected.groupby("model").size().to_dict() != dict.fromkeys(MODELS, 2424):
        raise ValueError("Frozen core eval reference denominator differs")
    return selected, {(row.model, row.sample_id): bool(row.uniform_reference) for row in selected.itertuples()}


def root_direct_decisions(folder):
    decisions = {}
    path = folder / "root_primary6_review.jsonl"
    final_pairs = {(row["question"], row["answer"]) for _, row, _ in rows(folder / "score_rows.jsonl.gz")}
    for line, row, _ in rows(path):
        qkey = frozen.qah(row["question"], row["answer"])
        original_luna = row["source_luna_decision"]
        if (type(row["abstain"]) is not bool or (row["question"], row["answer"]) not in final_pairs
                or (original_luna["question"], original_luna["answer"]) != (row["question"], row["answer"])
                or original_luna["qa_key"] != row["key"]):
            raise ValueError("Root Direct exact-QA decision differs from its complete source")
        author = row["root_review_author"]
        relations = row.get("name_relations", [])
        if isinstance(relations, dict):
            relations = [{"name": name, "role": role} for name, role in relations.items()]
        decisions[qkey] = {**row, "qa_key": qkey, "original_root_review_key": row["key"], "name_relations": relations,
                           "annotation_model": author["model"], "annotation_effort": author["effort"],
                           "annotation_call_id": author.get("call_id", ""), "decision_source_path": str(path),
                           "decision_source_line": line}
    if len(decisions) != 6:
        raise ValueError("Root Direct final authority must contain its exact six reviewed QA pairs")
    return decisions


def metrics(frame, condition):
    n = len(frame)
    correct = frame.canonical_name_in_primary_score
    abstain = frame.abstain
    reference = frame.uniform_reference.astype(bool)
    primary = n == 2424 and not correct.isna().any() and not abstain.isna().any()
    behavior = n == 2424 and not abstain.isna().any()
    literal_values = frame.literal_extracted_name_score
    literal_complete = n == 2424 and not literal_values.isna().any()
    c, a, r = int(correct.eq(1).sum()), int(abstain.eq(True).sum()), int(reference.sum())
    tp = int((abstain.eq(True) & reference).sum())
    fp = int((abstain.eq(True) & ~reference).sum())
    fn = int((abstain.eq(False) & reference).sum())
    tn = int((abstain.eq(False) & ~reference).sum())
    div = lambda numerator, denominator: numerator / denominator if denominator else None
    return {**condition, "expected_n": 2424, "n": n, "primary_complete": primary,
            "behavior_complete": behavior, "canonical_correct": c,
            "canonical_pending": int(correct.isna().sum()), "literal_pending": int(frame.literal_extracted_name_score.isna().sum()),
            "literal_correct": int(literal_values.eq(1).sum()), "literal_incorrect": int(literal_values.eq(0).sum()),
            "literal_complete": literal_complete,
            "literal_accuracy": div(int(literal_values.eq(1).sum()), n) if literal_complete else None,
            "abstain_pending": int(abstain.isna().sum()), "abstentions": a, "reference_positive": r,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "abstention_known_n": tp + fp + fn + tn,
            "reference_positive_behavior_pending": int((abstain.isna() & reference).sum()),
            "accuracy": div(c, n) if primary else None,
            "precision": div(tp, a) if behavior else None, "recall": div(tp, r) if behavior else None,
            "abstention_f1": div(2 * tp, a + r) if behavior else None,
            "metric_unit": "fraction", "reference_scope": "frozen_uniform_food_eval"}


def score_native(args):
    output = within(ROOT, args.output)
    output.mkdir(parents=True, exist_ok=False)
    received_path = within(ROOT, args.received_manifest) if args.received_manifest else None
    received = json.loads(received_path.read_text()) if received_path else {"parts": []}
    if received_path and (received["schema"] != "kdm_core_native_received_v1" or received["cpu_validation_passed"] is not True):
        raise ValueError("Core scorer requires a verified immutable received manifest")
    references, gt = load_references()
    samples = {row["id"]: row for _, row, _ in rows(ROOT / "data/current/all.jsonl")
               if row["dataset"] == "food101" and row["split"] == "eval"}
    patterns = frozen.compile_classes(sorted({row["class"] for row in samples.values()}))
    authority_manifest = json.loads(within(ROOT, args.authority_manifest).read_text())
    authority_manifest = {"census_final_labels": authority_manifest["census_final_labels"], "historical_labels": []}
    decision_paths = [within(ROOT, path) for path in args.decision_file]
    reviews, behavior, behavior_provenance, _historical, final_decisions = load_authority(
        ROOT / "outputs/paper_core_20260930", authority_manifest, decision_paths)
    direct = within(ROOT, args.direct_scores)
    decisions = {**root_direct_decisions(direct), **final_decisions}
    save_rows(output / "root_direct_authority_mapping.jsonl", [decisions[key] for key in sorted(decisions)
                                                             if decisions[key].get("original_root_review_key")])
    qa_cache, pending, scored, sources, source_counts, seen = {}, {}, [], [], Counter(), set()
    for part in received["parts"]:
        model = part["model"]
        raw = ROOT / part["received_raw_path"]
        verified = validate_received(raw.parent.parent, model, part["source"])
        if verified != part:
            raise ValueError("Received manifest differs from current immutable source validation")
        sources.append(part)
        for line, row, line_sha in rows(raw):
            sample, answer = row["sample"], row["text"]
            key = (model, sample["id"])
            if key in seen or sample != samples.get(sample["id"]):
                raise ValueError("Duplicate or unregistered native core model/sample")
            seen.add(key)
            qkey = frozen.qah(sample["question"], answer)
            if qkey not in qa_cache:
                qa_cache[qkey] = infer_qa(sample["question"], answer, patterns, reviews, behavior, decisions)
            inferred = qa_cache[qkey]
            canonical, literal, reason = score_target(answer, sample["class"], inferred, patterns)
            states = [frozen.extract(variant, patterns) for variant in inferred["variants"]]
            names = sorted({name for state in states for name in state["literal"]})
            if not states and inferred["parsed"]["primary_name"]:
                names = [inferred["parsed"]["primary_name"]]
            condition = {"model": model, "dataset": "food101", "split": "eval",
                         **{field: row[field] for field in COND[3:]}}
            record = {**condition, "main_marker": row["marker"], "sample_id": sample["id"], "key": row["key"],
                      "original_key": row["key"], "qa_key": qkey, "question": sample["question"], "answer": answer,
                      "target_class": sample["class"], "uniform_reference": gt[key],
                      "canonical_name_in_primary_score": canonical, "literal_extracted_name_score": literal,
                      "literal_extracted_names": names, "abstain": inferred["abstain"], "score_reason": reason,
                      "behavior_source": inferred["behavior_source"], "source_path": str(raw), "source_line": line,
                      "source_remote_path": part["source"]["files"]["raw"]["source_path"],
                      "source_remote_hostname": part["source"]["source_hostname"], "source_identity": row["identity"],
                      "generation_identity": part["generation_identity"], "raw_line_sha256": line_sha,
                      "raw_source_sha256": part["raw_sha256"], "received_complete_path": part["received_complete_path"],
                      "seed": row["seed"], "terminated": row["terminated"], "config": row["config"],
                      "config_sha256": stable_hash(row["config"]), "prompt": row["prompt"],
                      "reference_prompt": row["reference_prompt"], "primary_extraction": inferred["parsed"],
                      "behavior_history_sources": behavior_provenance.get(qkey, []),
                      "name_history_source": inferred["review"]["authority"] if inferred["review"] else None,
                      "decision_source_path": inferred["decision"].get("decision_source_path") if inferred["decision"] else None,
                      "decision_source_line": inferred["decision"].get("decision_source_line") if inferred["decision"] else None,
                      "annotation_model": inferred["annotation_model"], "annotation_effort": inferred["annotation_effort"],
                      "annotation_call_id": inferred["annotation_call_id"]}
            scored.append(record)
            source_counts[inferred["behavior_source"]] += 1
            if canonical is None or inferred["abstain"] is None:
                item = pending.setdefault(qkey, {"qa_key": qkey, "question": sample["question"], "answer": answer,
                                                 "reason": [], "memberships": []})
                item["reason"] = sorted(set(item["reason"] + [reason, inferred["behavior_source"]]))
                item["memberships"].append({"model": model, "sample_id": sample["id"], "key": row["key"],
                                             "target_class": sample["class"], "source_path": str(raw), "source_line": line,
                                             "source_identity": row["identity"], "raw_line_sha256": line_sha,
                                             "canonical_pending": canonical is None,
                                             "abstain_pending": inferred["abstain"] is None})
    reused_score_sources = []
    for name in args.reuse_score_rows:
        previous_path = within(ROOT, name)
        previous_receipt = json.loads((previous_path.parent / "scoring_receipt.json").read_text())
        if (previous_receipt["schema"] != "kdm_core_native_scoring_v1"
                or previous_receipt["uniform_reference_sha256"] != file_hash(ROOT / REFERENCE_PATH)
                or previous_receipt["canonical_scorer_sha256"] != file_hash(Path(frozen.__file__))):
            raise ValueError("Previous core score reference or canonical authority differs")
        previous_sha, reused_n, updated_n = file_hash(previous_path), 0, 0
        for line, original, line_sha in rows(previous_path):
            record = dict(original)
            model, sid, qkey = record["model"], record["sample_id"], record["qa_key"]
            sample = samples[sid]
            task = {"sample": sample, "method": "vcd", "marker": "NONE", "reference_marker": "NONE",
                    "guided": False, "reference_guided": False, "replicate": 0, "kind": "native_unguided"}
            if ((model, sid) in seen or model not in MODELS or record["key"] != task_id(model, task)
                    or any(record[field] != task[field] for field in COND[3:])
                    or record["target_class"] != sample["class"] or record["question"] != sample["question"]
                    or qkey != frozen.qah(record["question"], record["answer"])
                    or record["config"] != asdict(DecodeConfig(method="vcd"))
                    or record["uniform_reference"] != gt[(model, sid)]):
                raise ValueError("Previous immutable native score key, task or exact-QA differs")
            seen.add((model, sid))
            if qkey in final_decisions and (not args.literal_only or record["literal_extracted_name_score"] is None):
                if qkey not in qa_cache:
                    qa_cache[qkey] = infer_qa(record["question"], record["answer"], patterns, reviews, behavior, decisions)
                inferred = qa_cache[qkey]
                canonical, literal, reason = score_target(record["answer"], sample["class"], inferred, patterns)
                if args.literal_only and (canonical != record["canonical_name_in_primary_score"]
                        or inferred["abstain"] != record["abstain"] or literal is None):
                    raise ValueError("Literal-only closure changed an already decided primary value")
                states = [frozen.extract(variant, patterns) for variant in inferred["variants"]]
                names = sorted({name for state in states for name in state["literal"]})
                record.update(canonical_name_in_primary_score=canonical, literal_extracted_name_score=literal,
                              literal_extracted_names=names, abstain=inferred["abstain"], score_reason=reason,
                              behavior_source=inferred["behavior_source"], primary_extraction=inferred["parsed"],
                              decision_source_path=inferred["decision"].get("decision_source_path"),
                              decision_source_line=inferred["decision"].get("decision_source_line"),
                              annotation_model=inferred["annotation_model"], annotation_effort=inferred["annotation_effort"],
                              annotation_call_id=inferred["annotation_call_id"])
                updated_n += 1
            record["previous_score_provenance"] = {"path": str(previous_path.relative_to(ROOT)), "line": line,
                                                    "file_sha256": previous_sha, "line_sha256": line_sha}
            scored.append(record)
            reused_n += 1
        if reused_n != previous_receipt["native_rows"]:
            raise ValueError("Previous immutable score row count differs from its receipt")
        reused_score_sources.append({"path": str(previous_path.relative_to(ROOT)), "sha256": previous_sha,
                                    "rows": reused_n, "exact_QA_decision_updated_rows": updated_n,
                                    "source_receipt_path": str((previous_path.parent / "scoring_receipt.json").relative_to(ROOT)),
                                    "source_receipt_sha256": file_hash(previous_path.parent / "scoring_receipt.json"),
                                    "raw_rescanned": False})
    pending = {}
    source_counts = Counter(record["behavior_source"] for record in scored)
    for record in scored:
        if record["canonical_name_in_primary_score"] is None or record["abstain"] is None:
            item = pending.setdefault(record["qa_key"], {"qa_key": record["qa_key"], "question": record["question"],
                                                        "answer": record["answer"], "reason": [], "memberships": []})
            item["reason"] = sorted(set(item["reason"] + [record["score_reason"], record["behavior_source"]]))
            item["memberships"].append({key: record[key] for key in ("model", "sample_id", "key", "target_class", "source_path", "source_line", "source_identity", "raw_line_sha256")})
            item["memberships"][-1].update(canonical_pending=record["canonical_name_in_primary_score"] is None,
                                           abstain_pending=record["abstain"] is None)
    native = pd.DataFrame(scored)
    if len(native) != len(received["parts"]) * 2424 + sum(source["rows"] for source in reused_score_sources):
        raise ValueError("Native score rows differ from received full condition denominators")
    native_metrics = [metrics(group, dict(zip(COND, key)))
                      for key, group in native.groupby(list(COND), dropna=False, observed=True)]
    save_rows(output / "score_rows.jsonl.gz", scored)
    save_rows(output / "boundary_queue.jsonl", [pending[key] for key in sorted(pending)])
    pd.DataFrame(native_metrics).to_csv(output / "native_vcd_conditions.csv", index=False)
    references.to_csv(output / "frozen_uniform_references_eval.csv", index=False)
    result = {"schema": "kdm_core_native_scoring_v1", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "native_rows": len(scored), "native_models": [model for model in MODELS if model in set(native.model)],
              "conditions": len(native_metrics), "complete_primary_conditions": sum(row["primary_complete"] for row in native_metrics),
              "canonical_pending_rows": int(native.canonical_name_in_primary_score.isna().sum()),
              "abstain_pending_rows": int(native.abstain.isna().sum()), "boundary_QA": len(pending),
              "unique_QA": len({record["qa_key"] for record in scored}), "behavior_source_counts": dict(source_counts),
              "received_manifest": str(received_path.relative_to(ROOT)) if received_path else None,
              "received_manifest_sha256": file_hash(received_path) if received_path else None,
              "uniform_reference_path": REFERENCE_PATH, "uniform_reference_sha256": file_hash(ROOT / REFERENCE_PATH),
              "reference_eval_rows": len(references), "reference_join_missing": 0,
              "canonical_scorer_sha256": file_hash(Path(frozen.__file__)),
              "reused_inference_source_sha256": file_hash(ROOT / "workflows/supplemental/remaining11/score.py"),
              "scorer_sha256": file_hash(Path(__file__)), "new_generations": 0, "new_API_calls": 0, "GPU_initialized": False,
              "reused_previous_score_sources": reused_score_sources,
              "decision_files": [{"path": str(path), "sha256": file_hash(path)} for path in decision_paths]}
    save_json(output / "scoring_receipt.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return native, gt


def comparison_tables(args, native, gt):
    output = within(ROOT, args.output)
    export = ROOT / "outputs/paper_20260929"
    conditions = pd.read_csv(export / "conditions.csv")
    scores = pd.read_parquet(export / "scores.parquet")
    if len(conditions) != 352 or len(scores) != 853248 or scores.duplicated(["condition_id", "sample_id"]).any():
        raise ValueError("Frozen complete condition panel differs")
    frozen_panel = scores.merge(conditions[list(COND[:1]) + ["condition_id"] + list(COND[3:])], on="condition_id", validate="many_to_one")
    frozen_panel["dataset"], frozen_panel["split"] = "food101", "eval"
    frozen_panel = frozen_panel.rename(columns={"correct_canonical": "canonical_name_in_primary_score", "correct_literal": "literal_extracted_name_score"})
    frozen_panel["source_namespace"] = "frozen_paper_20260929"
    frozen_panel["source_condition_id"] = frozen_panel.condition_id
    direct_path = within(ROOT, args.direct_scores)
    receipt = json.loads((direct_path / "receipt.json").read_text())
    direct = pd.DataFrame([row for _, row, _ in rows(direct_path / "score_rows.jsonl.gz")])
    if (receipt["rows"] != 12120 or receipt["pending"] != 0 or len(direct) != 12120
            or direct.duplicated(["model", "sample_id"]).any() or direct.correct_canonical.isna().any() or direct.abstain.isna().any()):
        raise ValueError("Final root-reviewed native Direct coverage or decisions differ")
    direct = direct.rename(columns={"correct_canonical": "canonical_name_in_primary_score", "correct_literal": "literal_extracted_name_score"})
    direct["marker"], direct["reference_marker"], direct["replicate"] = "NONE", "NONE", 0
    direct["source_namespace"], direct["source_condition_id"] = "root_reviewed_exact_unguided_census", None
    native = native.copy()
    native["source_namespace"], native["source_condition_id"] = "P1_NATIVE_VCD", None
    columns = list(COND) + ["sample_id", "target_class", "canonical_name_in_primary_score", "literal_extracted_name_score",
                           "abstain", "uniform_reference", "source_namespace", "source_condition_id"]
    combined = pd.concat([frozen_panel[columns], direct[columns], native[columns]], ignore_index=True)
    expected_reference = [gt[(row.model, row.sample_id)] for row in combined.itertuples()]
    if not combined.uniform_reference.eq(expected_reference).all():
        raise ValueError("Condition rows do not share the exact frozen reference for model/sample")
    tables = []
    for key, group in combined.groupby(list(COND), dropna=False, observed=True):
        condition = dict(zip(COND, key))
        quota = group.groupby("target_class").size()
        if len(group) != 2424 or group.sample_id.duplicated().any() or len(quota) != 101 or not quota.eq(24).all():
            raise ValueError("Unified condition sample denominator or class quota differs")
        method, kind = condition["method"], condition["kind"]
        if not condition["guided"]:
            family = "Direct_unguided" if method == "direct" else "VCD_native_unguided"
        elif kind == "reference_instruction_removed":
            family = method.upper() + "_reference_guidance_removed"
        else:
            family = {"direct": "Direct_guided", "vcd": "VCD_guided", "m3id": "M3ID_guided",
                      "instruction_vcd": "IP_VCD", "instruction_m3id": "IP_M3ID", "cda_visual": "CDA_visual",
                      "dola": "DoLa", "deco": "DeCo", "sid": "registered_SID"}[method]
        tables.append({**metrics(group, condition), "method_family": family,
                       "configuration_scope": "same_marker" if condition["marker"] == condition["reference_marker"] else "cross_marker",
                       "source_namespace": group.source_namespace.iloc[0],
                       "source_condition_id": None if pd.isna(group.source_condition_id.iloc[0]) else int(group.source_condition_id.iloc[0])})
    table = pd.DataFrame(tables)
    table.to_csv(output / "all_core_conditions_same_eval.csv", index=False)
    table[table.configuration_scope.eq("same_marker")].to_csv(output / "same_marker_conditions.csv", index=False)
    table[table.method.isin(["vcd", "m3id", "sid"]) & table.kind.eq("main")].to_csv(output / "guided_cross_reference_conditions.csv", index=False)
    coverage = []
    for model in MODELS:
        have = native[native.model.eq(model)]
        coverage.append({"model": model, "dataset": "food101", "split": "eval", "method": "vcd", "kind": "native_unguided",
                         "expected": 2424, "received": len(have), "missing": 2424 - len(have),
                         "canonical_pending": int(have.canonical_name_in_primary_score.isna().sum()),
                         "abstain_pending": int(have.abstain.isna().sum()),
                         "raw_complete": len(have) == 2424, "primary_complete": len(have) == 2424 and not have.canonical_name_in_primary_score.isna().any() and not have.abstain.isna().any()})
    pd.DataFrame(coverage).to_csv(output / "native_vcd_coverage.csv", index=False)
    result = {"frozen_conditions": 352, "frozen_rows": 853248, "direct_unguided_conditions": 5,
              "direct_unguided_rows": 12120, "native_vcd_conditions": len(native.groupby(list(COND))),
              "native_vcd_rows": len(native), "table_conditions": len(table), "all_input_rows": len(combined),
              "complete_primary_conditions": int(table.primary_complete.sum()), "all_complete_condition_class_quota": "101x24",
              "all_reference_joins_match_frozen": True, "metric_unit": "fraction", "cross_reference_conditions": 192,
              "direct_final_receipt_path": str((direct_path / "receipt.json").relative_to(ROOT)),
              "direct_scores_sha256": file_hash(direct_path / "score_rows.jsonl.gz"),
              "frozen_panel_source": "outputs/paper_20260929/scores.parquet", "new_bootstrap_calculations": 0,
              "pair_statistics_entry": "workflows/paper_core/analyze_native.py"}
    save_json(output / "comparison_tables_receipt.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--received-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--authority-manifest", default="outputs/supplemental/remaining11/run_20260930_140337/assets/asset_manifest.json")
    parser.add_argument("--direct-scores", default=DEFAULT_DIRECT)
    parser.add_argument("--decision-file", action="append", default=[])
    parser.add_argument("--reuse-score-rows", action="append", default=[])
    parser.add_argument("--literal-only", action="store_true", help="Update only undecided literal fields from exact-QA decisions")
    args = parser.parse_args()
    if not args.received_manifest and not args.reuse_score_rows:
        parser.error("requires --received-manifest or --reuse-score-rows")
    native, gt = score_native(args)
    comparison_tables(args, native, gt)


if __name__ == "__main__":
    main()
