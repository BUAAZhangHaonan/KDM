#!/usr/bin/env python3
"""Score immutable registered core dev/Viz snapshots with existing authorities."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.decoding import DecodeConfig
from kdm.io import file_hash, stable_hash, stable_seed, within
from kdm.models.official_vqa_normalizer import VQAEval
from kdm.pipeline import task_id
from kdm.prompts import task_prompt
from kdm.scoring import lexical_label, vqa_score
from workflows.main_results import score as frozen
from workflows.paper_core.dev_viz import MODELS, planned_tasks, roster, condition as task_condition
from workflows.paper_core.score_native import save_json, save_rows
from workflows.supplemental.remaining11.score import load_authority, infer_qa, rows, score_target
from workflows.supplemental.remaining4.build_viz_qa_authority import decision_value
from workflows.supplemental.remaining4.score_received import load_viz_authority

FIELDS = ("method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")
COND = ("model", "dataset", "split", *FIELDS)
REFERENCE = "outputs/paper_20260929/references.parquet"
AUTHORITY = "outputs/supplemental/remaining11/run_20260930_140337/assets/asset_manifest.json"
NATIVE_RECEIPT = "outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v6_all5_literal_root_closed/scoring_receipt.json"
VIZ_OFFICIAL = "cache/assets/vizwiz/extracted/val.json"
LABELS = {"answer_assertive", "answer_uncertain", "abstain", "invalid"}
EXECUTOR = {"agent": "/root/dev_viz_inventory", "model": "gpt-6.1-sol", "effort": "max", "call_id": ""}


def native_decisions(receipt_path, reference_sha, canonical_sha):
    """Reuse only decision files bound to the accepted native scoring receipts."""
    paths, receipts, seen = [], [], set()

    def visit(path, expected_sha=None):
        path = within(ROOT, path)
        if path in seen:
            return
        sha = file_hash(path)
        if expected_sha is not None and sha != expected_sha:
            raise ValueError("An accepted native source receipt changed")
        value = json.loads(path.read_text())
        if (value["schema"] != "kdm_core_native_scoring_v1"
                or value["uniform_reference_sha256"] != reference_sha
                or value["canonical_scorer_sha256"] != canonical_sha):
            raise ValueError("An accepted native authority uses different frozen scoring")
        if expected_sha is None and (value["canonical_pending_rows"] != 0 or value["abstain_pending_rows"] != 0
                or value["complete_primary_conditions"] != value["conditions"]):
            raise ValueError("The final native authority is not closed")
        seen.add(path)
        for previous in value.get("reused_previous_score_sources", []):
            visit(previous["source_receipt_path"], previous["source_receipt_sha256"])
        for item in value.get("decision_files", []):
            decision_path = within(ROOT, item["path"])
            if file_hash(decision_path) != item["sha256"]:
                raise ValueError("An accepted native decision file changed")
            if decision_path not in paths:
                paths.append(decision_path)
        receipts.append({"path": str(path), "sha256": sha,
                         "role": "final_closed_authority" if expected_sha is None else "provenance_of_final_closure"})

    visit(receipt_path)
    return paths, receipts


def audit_raw(row, sample, model):
    if row["model"] != model or row["sample"] != sample or row["status"] != "ok":
        raise ValueError("Immutable raw model, sample or status differs")
    task = {"sample": sample, **{field: row[field] for field in FIELDS}}
    if row["key"] != task_id(model, task) or row["seed"] != stable_seed(sample["id"], model, row["replicate"]):
        raise ValueError("Immutable raw task key or stable seed differs")
    config = asdict(DecodeConfig(method=row["method"]))
    if row["method"] in {"m3id", "instruction_m3id"}:
        config["m3id_offset"] = len(row["offset_prompt_tokens"])
    if row["config"] != config or row["prompt"] != task_prompt(sample["question"], row["marker"], row["guided"]):
        raise ValueError("Immutable raw decode configuration or final prompt differs")
    reference_prompt = task_prompt(sample["question"], row["reference_marker"], row["reference_guided"]) if row["method"] != "direct" else None
    neutral_prompt = task_prompt(sample["question"], guided=False) if row["method"].startswith("instruction_") else None
    if row["reference_prompt"] != reference_prompt or row["neutral_prompt"] != neutral_prompt:
        raise ValueError("Immutable reference or neutral prompt differs")
    tokens, probabilities = row["tokens"], row["selected_log_probabilities"]
    if (not tokens or len(tokens) > 32 or len(tokens) != len(probabilities)
            or not isinstance(row["terminated"], bool) or not isinstance(row["text"], str)
            or not all(math.isfinite(value) for value in probabilities)
            or not math.isfinite(row["sequence_log_probability"])
            or not math.isfinite(row["first_probability"]) or not 0 <= row["first_probability"] <= 1):
        raise ValueError("Immutable generation token or finite probability evidence differs")
    return task


def snapshot_sources(args, samples, roster_path):
    records, sources, seen = [], [], set()
    expected_n = 404 if args.stage == "dev404" else 512
    input_dir = within(ROOT, args.input_dir) if args.input_dir else None
    receipts = sorted(input_dir.rglob("*.complete.json")) if input_dir else []
    if input_dir and not receipts:
        raise ValueError("The input snapshot directory has no immutable completed parts")
    plans = {}
    for receipt_path in receipts:
        part = json.loads(receipt_path.read_text())
        stem = receipt_path.name.removesuffix(".complete.json")
        raw = receipt_path.with_name(stem + ".jsonl")
        identity_path = receipt_path.with_name(stem + ".identity.json")
        sidecar = json.loads(identity_path.read_text())
        definition = sidecar["definition"]
        core = {key: value for key, value in definition.items() if key not in {"shard", "n_shards", "base_config"}}
        model = definition["model"]
        raw_sha = file_hash(raw)
        if (part["passed"] is not True or part["raw_sha256"] != raw_sha
                or sidecar["identity"] != stable_hash(definition) or part["identity"] != stable_hash(core)
                or definition["schema"] != "kdm_core_dev_viz_identity_v1" or definition["stage"] != args.stage
                or model not in MODELS or definition["noise_step"] != 500
                or definition["base_config"] != asdict(DecodeConfig()) or definition["decode"] != asdict(DecodeConfig())
                or definition["roster_sha256"] != file_hash(roster_path)
                or definition["frozen_spec"] != json.loads((ROOT / "configs/runtime" / (model + ".json")).read_text())
                or Path(part["raw"]).name != raw.name or part["required"] != expected_n):
            raise ValueError("Immutable completed part identity, source bytes or registered configuration differs")
        if args.stage == "viz512":
            if not args.selected_configs or definition["selected_configs_sha256"] != file_hash(within(ROOT, args.selected_configs)):
                raise ValueError("VizWiz source is not bound to the supplied frozen Food-dev selection")
        if model not in plans:
            plans[model] = {task_id(model, task): task for task in planned_tasks(model, args.stage, args.selected_configs)}
        source_rows = list(rows(raw))
        pilot = part.get("scope") == "pilot_only"
        if len(source_rows) != part["rows"] or (pilot and len(source_rows) != 8) or (not pilot and len(source_rows) != expected_n):
            raise ValueError("Immutable source row count or pilot/full scope differs")
        sample_ids = set()
        for line, row, line_sha in source_rows:
            sample = samples.get(row["sample"]["id"])
            if sample is None or row["identity"] != sidecar["identity"] or row["key"] not in plans[model]:
                raise ValueError("Immutable raw source is outside the registered exact sample/task identity")
            task = audit_raw(row, sample, model)
            if task != plans[model][row["key"]] or task_condition(task) != part["condition"] or row["method"] != part["method"] or row["marker"] != part["marker"]:
                raise ValueError("Immutable source condition differs from its completed receipt")
            if row["key"] in seen or sample["id"] in sample_ids:
                raise ValueError("Duplicate immutable source key or condition sample")
            seen.add(row["key"])
            sample_ids.add(sample["id"])
            records.append((row, {"source_path": str(raw), "source_line": line, "raw_line_sha256": line_sha,
                "raw_source_sha256": raw_sha, "source_identity": row["identity"], "generation_identity": part["identity"],
                "source_original_path": part["raw"], "complete_source_path": str(receipt_path),
                "identity_source_path": str(identity_path), "source_condition_id": part["condition"],
                "source_scope": "pilot_partial" if pilot else "complete_condition",
                "expected_n": expected_n, "source_raw_complete": not pilot,
                "model_checkpoint": definition["frozen_spec"]["hf_model_id"], "model_dtype": definition["frozen_spec"]["dtype"],
                "source_generation_author": definition["author"]}))
        expected_ids = {sample["id"] for sample in list(samples.values())[:8]} if pilot else set(samples)
        if sample_ids != expected_ids:
            raise ValueError("Immutable source pilot/full sample roster differs")
        sources.append({"model": model, "method": part["method"], "marker": part["marker"], "rows": len(source_rows),
                        "expected_n": expected_n, "raw_complete": not pilot, "raw_path": str(raw), "raw_sha256": raw_sha,
                        "complete_path": str(receipt_path), "complete_sha256": file_hash(receipt_path),
                        "identity_path": str(identity_path), "identity_sha256": file_hash(identity_path),
                        "core_identity": part["identity"], "ledger_identity": sidecar["identity"],
                        "generation_wall_s": part["generation_wall_s"], "scope": "pilot_partial" if pilot else "complete_condition",
                        "generation_entry_source_sha256": definition["source_sha256"]})
    return records, sources


def census_sources(args, samples):
    """Read the finite copied roster container; preserve the original census keys."""
    if not args.census_direct:
        return [], []
    container = within(ROOT, args.census_direct)
    container_sha = file_hash(container)
    expected_n = 404 if args.stage == "dev404" else 512
    manifest_path = ROOT / "data/responses/census/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    registered = {item["model"]: item for item in manifest["models"]}
    identities, records, grouped, seen = {}, [], defaultdict(list), set()
    for container_line, item, line_sha in rows(container):
        row = item["record"]
        if row["model"] not in MODELS or row["sample"]["id"] not in samples or row["guided"] is not False:
            continue
        model = row["model"]
        if model not in identities:
            identity_path = ROOT / "data/responses/census" / (model + ".identity.json")
            sidecar = json.loads(identity_path.read_text())
            definition = sidecar["definition"]
            runtime = json.loads((ROOT / "configs/runtime" / (model + ".json")).read_text())
            if (sidecar["identity"] != stable_hash(definition) or definition["backend"] != runtime
                    or definition["base_config"] != asdict(DecodeConfig())):
                raise ValueError("The original census backend or decoding identity differs")
            identities[model] = (sidecar, identity_path, runtime)
        sidecar, identity_path, runtime = identities[model]
        raw_path = ROOT / "data/responses/census" / (model + ".jsonl.gz")
        if (within(ROOT, item["source_path"]) != raw_path
                or item["source_identity"] != sidecar["identity"] or row["identity"] != sidecar["identity"]
                or item["original_source_sha256"] != registered[model]["source_sha256"]
                or item["identity_valid"] is not True or item["registered_backend_exact_match"] is not True
                or item["reuse_eligible"] is not True or item["reuse_difference_fields"]
                or row["method"] != "direct" or row["replicate"] != 0):
            raise ValueError("A copied census record differs from its original admitted source")
        audit_raw(row, samples[row["sample"]["id"]], model)
        if row["key"] in seen:
            raise ValueError("Duplicate copied census task")
        seen.add(row["key"])
        condition = {"model": model, "dataset": row["sample"]["dataset"], "split": row["sample"]["split"],
                     **{field: row[field] for field in FIELDS}}
        cid = stable_hash(condition)[:16]
        records.append((row, {"source_path": str(raw_path), "source_line": item["source_line"],
            "raw_line_sha256": None, "raw_record_stable_sha256": stable_hash(row),
            "raw_source_sha256": item["original_source_sha256"],
            "source_identity": row["identity"], "generation_identity": row["identity"],
            "source_original_path": str(raw_path), "complete_source_path": str(manifest_path),
            "identity_source_path": str(identity_path), "source_condition_id": cid,
            "source_scope": "existing_frozen_census", "expected_n": expected_n, "source_raw_complete": True,
            "model_checkpoint": runtime["hf_model_id"], "model_dtype": runtime["dtype"],
            "source_generation_author": None, "copied_container_path": str(container),
            "copied_container_line": container_line, "copied_container_line_sha256": line_sha,
            "copied_container_sha256": container_sha, "original_raw_reopened": False}))
        grouped[cid].append(row)
    sources = []
    for cid, group in grouped.items():
        if len(group) != expected_n or {row["sample"]["id"] for row in group} != set(samples):
            raise ValueError("The copied census condition is not a complete exact roster")
        model = group[0]["model"]
        sidecar, identity_path, _runtime = identities[model]
        sources.append({"model": model, "method": "direct", "marker": group[0]["marker"],
            "rows": len(group), "expected_n": expected_n, "raw_complete": True,
            "raw_path": str(ROOT / "data/responses/census" / (model + ".jsonl.gz")),
            "raw_sha256": registered[model]["source_sha256"], "complete_path": str(manifest_path),
            "complete_sha256": file_hash(manifest_path), "identity_path": str(identity_path),
            "identity_sha256": file_hash(identity_path), "core_identity": sidecar["identity"],
            "ledger_identity": sidecar["identity"], "scope": "existing_frozen_census",
            "copied_container_path": str(container), "copied_container_sha256": container_sha,
            "original_raw_reopened": False, "generation_wall_s": None})
    return records, sources


def viz_references(samples):
    """Bind current 512 samples to the official question, answers and answerability."""
    path = ROOT / VIZ_OFFICIAL
    official = {record["image"]: (index, record) for index, record in enumerate(json.loads(path.read_text()))}
    references = []
    for sample in samples.values():
        index, record = official[sample["source_record"]]
        ability = sample["annotated_answerable"]
        if (type(ability) is not int or ability not in (0, 1) or ability != record["answerable"]
                or sample["question"] != record["question"] or sample["official_answers"] != record["answers"]
                or len(record["answers"]) != 10 or sample["gold"] != [entry["answer"] for entry in record["answers"]]):
            raise ValueError("A VizWiz sample differs from its official answers or answerability")
        references.append({"sample_id": sample["id"], "annotated_answerable": ability,
            "official_reference": ability == 0, "question": sample["question"],
            "official_answers_json": json.dumps(record["answers"], ensure_ascii=False),
            "official_source_path": str(path), "official_record_index": index})
    if Counter(item["annotated_answerable"] for item in references) != {0: 166, 1: 346}:
        raise ValueError("The fixed VizWiz512 official answerability counts differ")
    return pd.DataFrame(references), path


def viz_inferences(source_records, authority_manifest, decision_paths, viz_authority_paths):
    """Reuse real exact-QA semantics; keep every unannotated substantive span pending."""
    targets = {frozen.qah(row["sample"]["question"], row["text"]):
               (row["sample"]["question"], row["text"]) for row, _source in source_records}
    census_path = within(ROOT, authority_manifest["census_final_labels"]["path"])
    census, census_n = defaultdict(list), 0
    for line, row, line_sha in rows(census_path):
        census_n += 1
        if row["dataset"] != "vizwiz":
            continue
        qkey = frozen.qah(row["question"], row["text"])
        if qkey not in targets:
            continue
        if row["label"] not in LABELS or targets[qkey] != (row["question"], row["text"]):
            raise ValueError("The actual final census has an undecided or mismatched exact QA")
        census[qkey].append({"source_path": str(census_path), "source_line": line,
            "source_line_sha256": line_sha, "actual_annotation": row, "label": row["label"],
            "answer_text": row.get("answer_text", "")})
    if census_n != 293344 or authority_manifest["census_final_labels"]["rows"] != census_n:
        raise ValueError("The registered final census closure must contain 293344 records")
    explicit = {}
    for path in decision_paths:
        sha = file_hash(path)
        for line, value, line_sha in rows(path):
            qkey = value.get("qa_key")
            if qkey not in targets:
                continue
            question, answer = targets[qkey]
            result = decision_value(value, question, answer)
            if result is None:
                continue
            explicit[qkey] = {"decision": value, "value": result, "source_path": str(path),
                "source_sha256": sha, "source_line": line, "source_line_sha256": line_sha}
    supplied = load_viz_authority(viz_authority_paths)
    inferred = {}
    lexical_sha = file_hash(ROOT / "src/kdm/scoring.py")
    span_normalizer = VQAEval(None, None)
    normalize_span = lambda text: span_normalizer.processDigitArticle(span_normalizer.processPunctuation(text))
    for qkey, (question, answer) in targets.items():
        label, span, abstain, source, reason = None, None, None, [], None
        if qkey in supplied:
            value = supplied[qkey]
            if value["question"] != question or value["answer"] != answer:
                raise ValueError("The supplied VizWiz authority differs from its complete QA")
            abstain = value["abstain"] if value.get("behavior_resolved", True) else None
            label = value.get("label")
            span = value.get("answer_text") if value.get("quality_span_resolved", True) else None
            source, reason = [value], "supplied_accepted_exact_QA_authority"
        elif qkey in explicit:
            value = explicit[qkey]
            label, span = value["value"]
            abstain = value["decision"]["abstain"]
            source, reason = [value], "explicit_actual_exact_QA_review"
        elif census[qkey]:
            source = census[qkey]
            variants = {(item["label"], item["answer_text"]) for item in source}
            if len(variants) == 1:
                label, actual_text = next(iter(variants))
                abstain = label == "abstain"
                if label in {"abstain", "invalid"} and actual_text == "":
                    span, reason = "", "actual_final_census_exact_QA"
                elif label in {"answer_assertive", "answer_uncertain"} and actual_text and actual_text in answer:
                    span, reason = actual_text, "actual_final_census_exact_QA"
                else:
                    label, reason = None, "actual_final_census_missing_or_noncontinuous_span"
            else:
                behaviors = {item["label"] == "abstain" for item in source}
                abstain = next(iter(behaviors)) if len(behaviors) == 1 else None
                reason = "actual_final_census_exact_QA_span_or_label_variants"
                substantive = all(item["label"] in {"answer_assertive", "answer_uncertain"}
                                  and item["answer_text"] and item["answer_text"] in answer for item in source)
                if substantive:
                    normalized_spans = {normalize_span(item["answer_text"]) for item in source}
                    if len(normalized_spans) == 1 and next(iter(normalized_spans)):
                        chosen = min(source, key=lambda item: (len(item["answer_text"]), item["source_line"]))
                        label, span = chosen["label"], chosen["answer_text"]
                        reason = "actual_final_census_VQA_equivalent_original_spans"
        else:
            label = lexical_label(answer)
            if label is not None:
                span, abstain, reason = "", label == "abstain", "existing_registered_exact_phrase_rule"
                source = [{"code_path": "src/kdm/scoring.py", "code_sha256": lexical_sha,
                           "evidence_span": answer}]
            else:
                reason = "no_actual_exact_QA_annotation_or_registered_exact_phrase"
        if abstain is True or label == "invalid":
            span = ""
        complete = type(abstain) is bool and span is not None
        if complete and not abstain and label != "invalid" and (not span or span not in answer):
            raise ValueError("A substantive VizWiz answer lacks its accepted continuous original span")
        inferred[qkey] = {"abstain": abstain, "label": label, "answer_text": span,
            "quality_span_resolved": complete, "behavior_source": reason, "authority_sources": source,
            "full_reply_fallback_used": False, "new_scientific_judgment": False}
    return inferred, {"actual_census_source_path": str(census_path), "actual_census_rows": census_n,
                      "actual_census_source_sha256": file_hash(census_path), "target_QA": len(targets)}


def viz_metrics(group, condition):
    n, expected = len(group), int(group.expected_n.iloc[0])
    quality, abstain, reference = group.quality_score, group.abstain, group.official_reference.astype(bool)
    behavior_complete = not abstain.isna().any()
    complete = behavior_complete and not quality.isna().any()
    a, r = int(abstain.eq(True).sum()), int(reference.sum())
    tp, fp = int((abstain.eq(True) & reference).sum()), int((abstain.eq(True) & ~reference).sum())
    fn, tn = int((abstain.eq(False) & reference).sum()), int((abstain.eq(False) & ~reference).sum())
    div = lambda numerator, denominator: numerator / denominator if denominator else None
    raw_complete = n == expected and bool(group.source_raw_complete.all())
    return {**condition, "n": n, "expected_n": expected, "raw_complete": raw_complete,
        "primary_scoring_complete": complete, "primary_complete": complete and raw_complete,
        "report_scope": "complete_condition" if n == expected else "pilot_partial",
        "FULL": int(group.quality_state.eq("FULL").sum()), "PARTIAL": int(group.quality_state.eq("PARTIAL").sum()),
        "ZERO": int(group.quality_state.eq("ZERO").sum()), "A": a,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "reference_positive": r,
        "official_unanswerable_n": r, "official_answerable_n": n - r,
        "quality_pending": int(quality.isna().sum()), "abstain_pending": int(abstain.isna().sum()),
        "official_raw_score_mean": float(group.official_raw_score.mean()),
        "official_raw_score_sum": float(group.official_raw_score.sum()),
        "answer_quality_mean": float(quality.mean()) if complete else None,
        "answer_quality_sum": float(quality.sum()) if complete else None,
        "answer_quality_reason": None if complete else "unresolved_answer_span_or_abstention",
        "precision": div(tp, a) if behavior_complete else None,
        "precision_reason": "unresolved_abstention" if not behavior_complete else ("zero_abstentions" if not a else None),
        "recall": div(tp, r) if behavior_complete else None,
        "recall_reason": "unresolved_abstention" if not behavior_complete else ("zero_official_unanswerables" if not r else None),
        "F1": div(2 * tp, a + r) if behavior_complete else None,
        "F1_reason": "unresolved_abstention" if not behavior_complete else ("zero_abstentions_and_reference_positives" if a + r == 0 else None),
        "answerable_abstention_rate": div(fp, n - r) if behavior_complete else None,
        "answerable_abstention_rate_reason": "unresolved_abstention" if not behavior_complete else ("zero_official_answerables" if n == r else None),
        "metric_denominator_scope": "actual_observed_inputs", "reference_scope": "VizWiz_official_answerability",
        "answer_quality_rule": "official_leave_one_annotator_out_on_accepted_answer_span;abstain_or_invalid_zero",
        "official_raw_score_rule": "official_leave_one_annotator_out_on_full_raw_reply_including_unanswerable",
        "metric_unit": "fraction"}


def food_metrics(group, condition):
    n, expected = len(group), int(group.expected_n.iloc[0])
    correct, abstain, reference = group.correct_canonical, group.abstain, group.uniform_reference.astype(bool)
    complete = not correct.isna().any() and not abstain.isna().any()
    behavior_complete = not abstain.isna().any()
    c, w, a = int((correct.eq(1) & abstain.eq(False)).sum()), int((correct.eq(0) & abstain.eq(False)).sum()), int(abstain.eq(True).sum())
    r = int(reference.sum())
    tp, fp = int((abstain.eq(True) & reference).sum()), int((abstain.eq(True) & ~reference).sum())
    fn, tn = int((abstain.eq(False) & reference).sum()), int((abstain.eq(False) & ~reference).sum())
    div = lambda numerator, denominator: numerator / denominator if denominator else None
    raw_complete = n == expected and bool(group.source_raw_complete.all())
    return {**condition, "n": n, "expected_n": expected, "raw_complete": raw_complete,
            "primary_scoring_complete": complete, "primary_complete": complete and raw_complete,
            "report_scope": "complete_condition" if n == expected else "pilot_partial",
            "C": c, "W": w, "A": a, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "reference_positive": r,
            "canonical_pending": int(correct.isna().sum()), "literal_pending": int(group.correct_literal.isna().sum()),
            "abstain_pending": int(abstain.isna().sum()), "accuracy": div(c, n) if complete else None,
            "accuracy_reason": None if complete else "unresolved_primary_decisions",
            "precision": div(tp, a) if behavior_complete else None,
            "precision_reason": "unresolved_abstention" if not behavior_complete else ("zero_abstentions" if not a else None),
            "recall": div(tp, r) if behavior_complete else None,
            "recall_reason": "unresolved_abstention" if not behavior_complete else ("zero_reference_positives" if not r else None),
            "F1": div(2 * tp, a + r) if behavior_complete else None,
            "F1_reason": "unresolved_abstention" if not behavior_complete else ("zero_abstentions_and_reference_positives" if a + r == 0 else None),
            "J": div(c + tp, n) if complete else None, "J_reason": None if complete else "unresolved_primary_decisions",
            "unnecessary_abstention_rate": div(fp, n - r) if behavior_complete else None,
            "unnecessary_abstention_rate_reason": "unresolved_abstention" if not behavior_complete else ("zero_reference_negatives" if n == r else None),
            "metric_denominator_scope": "actual_observed_inputs", "reference_scope": "frozen_uniform_Food_dev", "metric_unit": "fraction"}


def score(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU scoring requires CUDA_VISIBLE_DEVICES to be explicitly empty")
    start = time.perf_counter()
    output = within(ROOT, args.out)
    output.mkdir(parents=True, exist_ok=False)
    roster_path, selected_samples = roster(args.stage)
    samples = {sample["id"]: sample for sample in selected_samples}
    source_records, sources = snapshot_sources(args, samples, roster_path)
    direct_records, direct_parts = census_sources(args, samples)
    source_records += direct_records
    sources += direct_parts
    if not source_records:
        raise ValueError("No immutable source records selected")
    if len({row["key"] for row, _source in source_records}) != len(source_records):
        raise ValueError("An immutable snapshot and reused census bind the same task twice")
    validation_s = time.perf_counter() - start
    canonical_sha = file_hash(Path(frozen.__file__))
    accepted_paths, accepted_receipts = [], []
    reference_map, patterns = {}, None
    if args.stage == "dev404":
        reference_path = ROOT / REFERENCE
        reference_sha = file_hash(reference_path)
        reference_frame = pd.read_parquet(reference_path)
        if (len(reference_frame) != 24240 or reference_frame.duplicated(["model", "sample_id"]).any()
                or not reference_frame.independent_attempts.eq(10).all()
                or not ((reference_frame.gold_rank > 1) & reference_frame.independent_correct_attempts.eq(0)).eq(reference_frame.uniform_reference).all()):
            raise ValueError("Frozen reference counts or registered rule differs")
        references = reference_frame[reference_frame.sample_id.isin(samples)].copy()
        if len(references) != 2020 or references.uniform_reference.isna().any():
            raise ValueError("Fixed Food dev404 references are incomplete")
        reference_map = {(row.model, row.sample_id): row for row in references.itertuples()}
        classes = json.loads((ROOT / "data/manifest.json").read_text())["canonical_classes"]
        if len(classes) != 101:
            raise ValueError("Original Food-101 vocabulary differs")
        patterns = frozen.compile_classes(classes)
        accepted_paths, accepted_receipts = native_decisions(args.native_score_receipt, reference_sha, canonical_sha)
    else:
        references, reference_path = viz_references(samples)
        reference_sha = file_hash(reference_path)
        reference_map = {row.sample_id: row for row in references.itertuples()}
    authority_manifest_path = within(ROOT, args.authority_manifest)
    original_authority = json.loads(authority_manifest_path.read_text())
    authority_manifest = {"census_final_labels": original_authority["census_final_labels"], "historical_labels": []}
    decision_paths = accepted_paths + [within(ROOT, value) for value in args.decision_file]
    viz_authority_paths = [within(ROOT, value) for value in args.viz_authority_file]
    authority_start = time.perf_counter()
    viz_census_source = None
    if args.stage == "dev404":
        reviews, behavior, behavior_sources, _history, decisions = load_authority(output, authority_manifest, decision_paths)
        qa_cache = {}
    else:
        qa_cache, viz_census_source = viz_inferences(source_records, authority_manifest, decision_paths, viz_authority_paths)
        normalizer = VQAEval(None, None)
        normalize = lambda text: normalizer.processDigitArticle(normalizer.processPunctuation(text))
    authority_load_s = time.perf_counter() - authority_start
    scored, pending, memberships = [], {}, []
    scoring_start = time.perf_counter()
    for row, provenance in source_records:
        sample, answer = row["sample"], row["text"]
        qkey = frozen.qah(sample["question"], answer)
        if args.stage == "viz512":
            inferred = qa_cache[qkey]
            abstain, span = inferred["abstain"], inferred["answer_text"]
            raw_quality = float(vqa_score(answer, sample["official_answers"], normalize))
            quality = (0.0 if abstain is True or inferred["label"] == "invalid" else
                       float(vqa_score(span, sample["official_answers"], normalize))
                       if abstain is False and inferred["quality_span_resolved"] else None)
            if not 0 <= raw_quality <= 1 or (quality is not None and not 0 <= quality <= 1):
                raise ValueError("Official VizWiz answer quality lies outside [0, 1]")
            state = ("A" if abstain is True else "FULL" if quality == 1 else
                     "PARTIAL" if quality is not None and quality > 0 else "ZERO" if quality == 0 else None)
            ref = reference_map[sample["id"]]
            condition = {"model": row["model"], "dataset": sample["dataset"], "split": sample["split"],
                         **{field: row[field] for field in FIELDS}}
            authors = []
            for entry in inferred["authority_sources"]:
                annotation = entry.get("actual_annotation", entry.get("decision", entry))
                model = annotation.get("annotation_model", annotation.get("judge_response_model"))
                if model:
                    authors.append({"model": model, "effort": annotation.get("annotation_effort"),
                        "call_id": annotation.get("annotation_call_id", annotation.get("judge_response_id", "")),
                        "source_path": entry.get("source_path", entry.get("decision_source_path")),
                        "source_line": entry.get("source_line", entry.get("decision_source_line"))})
            record = {**condition, **provenance, "condition_id": stable_hash(condition)[:16], "main_marker": row["marker"],
                "sample_id": sample["id"], "key": row["key"], "qa_key": qkey, "question": sample["question"], "answer": answer,
                "answer_text": span, "label": inferred["label"], "image_source": sample["image_path"], "tokens": row["tokens"],
                "selected_log_probabilities": row["selected_log_probabilities"], "first_probability": row["first_probability"],
                "config": row["config"], "config_sha256": stable_hash(row["config"]), "seed": row["seed"],
                "prompt": row["prompt"], "reference_prompt": row["reference_prompt"], "neutral_prompt": row["neutral_prompt"],
                "offset_prompt_tokens": row.get("offset_prompt_tokens"), "terminated": row["terminated"],
                "official_raw_score": raw_quality, "quality_score": quality, "quality_state": state, "abstain": abstain,
                "official_reference": bool(ref.official_reference), "annotated_answerable": int(ref.annotated_answerable),
                "reference_source_path": str(reference_path), "official_record_index": int(ref.official_record_index),
                "official_answers": sample["official_answers"], "uniform_reference": None,
                "correct_canonical": None, "correct_literal": None, "target_class": None,
                "score_reason": inferred["behavior_source"], "behavior_source": inferred["behavior_source"],
                "semantic_authority_sources": inferred["authority_sources"], "annotation_authors": authors,
                "quality_span_resolved": inferred["quality_span_resolved"], "full_reply_fallback_used": False,
                "new_scientific_judgment": False, "rule_executor": EXECUTOR}
            scored.append(record)
            needs = {"quality": quality is None, "abstain": abstain is None}
            if any(needs.values()):
                item = pending.setdefault(qkey, {"qa_key": qkey, "question": sample["question"], "answer": answer,
                    "dataset": "vizwiz", "reason": [inferred["behavior_source"]], "needs": dict(needs)})
                item["needs"] = {key: item["needs"][key] or value for key, value in needs.items()}
                memberships.append({"qa_key": qkey, "model": row["model"], "sample_id": sample["id"], "key": row["key"],
                    "official_reference": bool(ref.official_reference), **provenance, "needs": needs})
            continue
        if qkey not in qa_cache:
            qa_cache[qkey] = infer_qa(sample["question"], answer, patterns, reviews, behavior, decisions)
        inferred = dict(qa_cache[qkey])
        if frozen.configured_marker(answer, row) and not inferred["variants"]:
            inferred.update(abstain=True, behavior_source="exact_actual_condition_marker")
        canonical, literal, reason = score_target(answer, sample["class"], inferred, patterns)
        if inferred["abstain"] is True and canonical not in (0, None):
            raise ValueError("Correct and abstained decisions overlap")
        ref = reference_map[(row["model"], sample["id"])]
        condition = {"model": row["model"], "dataset": sample["dataset"], "split": sample["split"],
                     **{field: row[field] for field in FIELDS}}
        states = [frozen.extract(variant, patterns) for variant in inferred["variants"]]
        names = sorted({name for state in states for name in state["literal"]})
        if not states and inferred["parsed"]["primary_name"]:
            names = [inferred["parsed"]["primary_name"]]
        record = {**condition, **provenance, "condition_id": stable_hash(condition)[:16], "main_marker": row["marker"],
            "sample_id": sample["id"], "key": row["key"], "qa_key": qkey, "question": sample["question"], "answer": answer,
            "target_class": sample["class"], "image_source": sample["image_path"], "tokens": row["tokens"],
            "selected_log_probabilities": row["selected_log_probabilities"], "first_probability": row["first_probability"],
            "config": row["config"], "config_sha256": stable_hash(row["config"]), "seed": row["seed"],
            "prompt": row["prompt"], "reference_prompt": row["reference_prompt"], "neutral_prompt": row["neutral_prompt"],
            "offset_prompt_tokens": row.get("offset_prompt_tokens"), "terminated": row["terminated"],
            "correct_canonical": canonical, "canonical_name_in_primary_score": canonical,
            "correct_literal": literal, "literal_extracted_name_score": literal, "literal_extracted_names": names,
            "abstain": inferred["abstain"], "uniform_reference": bool(ref.uniform_reference),
            "accepted_reference": bool(ref.accepted_reference), "gold_rank": int(ref.gold_rank),
            "independent_correct_attempts": int(ref.independent_correct_attempts), "independent_attempts": int(ref.independent_attempts),
            "reference_source_path": str(reference_path), "uniform_reference_source_line": int(ref.uniform_source_line),
            "score_reason": reason, "behavior_source": inferred["behavior_source"], "primary_extraction": inferred["parsed"],
            "behavior_history_sources": behavior_sources.get(qkey, []),
            "name_history_source": inferred["review"]["authority"] if inferred["review"] else None,
            "decision_source_path": inferred["decision"].get("decision_source_path") if inferred["decision"] else None,
            "decision_source_line": inferred["decision"].get("decision_source_line") if inferred["decision"] else None,
            "annotation_model": inferred["annotation_model"], "annotation_effort": inferred["annotation_effort"],
            "annotation_call_id": inferred["annotation_call_id"], "rule_executor": EXECUTOR}
        scored.append(record)
        needs = {"canonical": canonical is None, "literal": literal is None, "abstain": inferred["abstain"] is None}
        if any(needs.values()):
            item = pending.setdefault(qkey, {"qa_key": qkey, "question": sample["question"], "answer": answer,
                                            "dataset": "food101", "reason": [], "needs": dict.fromkeys(needs, False)})
            item["reason"] = sorted(set(item["reason"] + [reason, inferred["behavior_source"]]))
            item["needs"] = {key: item["needs"][key] or value for key, value in needs.items()}
            memberships.append({"qa_key": qkey, "model": row["model"], "sample_id": sample["id"], "key": row["key"],
                                "target_class": sample["class"], **provenance, "needs": needs})
    scoring_s = time.perf_counter() - scoring_start
    save_rows(output / "score_rows.jsonl.gz", scored)
    save_rows(output / "pending_qa.jsonl", [pending[key] for key in sorted(pending)])
    save_rows(output / "pending_memberships.jsonl", memberships)
    save_rows(output / "pending_luna_payload.jsonl", [{"qa_key": item["qa_key"], "question": item["question"],
                                                     "answer": item["answer"], "needs": item["needs"]} for item in pending.values()])
    frame = pd.DataFrame(scored)
    metric = food_metrics if args.stage == "dev404" else viz_metrics
    metrics = [metric(group, dict(zip(COND, key))) for key, group in frame.groupby(list(COND), dropna=False, observed=True)]
    pd.DataFrame(metrics).to_csv(output / "metrics_all.csv", index=False)
    columns = list(COND) + ["condition_id", "sample_id", "key", "qa_key", "question", "answer", "target_class",
        "correct_canonical", "correct_literal", "abstain", "uniform_reference", "source_path", "source_line",
        "source_identity", "generation_identity", "source_scope", "expected_n", "source_raw_complete", "score_reason",
        "behavior_source", "model_checkpoint", "model_dtype", "seed", "terminated", "config_sha256"]
    if args.stage == "dev404":
        columns += ["accepted_reference", "gold_rank", "independent_correct_attempts", "independent_attempts"]
    else:
        columns += ["answer_text", "label", "official_raw_score", "quality_score", "quality_state",
                    "official_reference", "annotated_answerable", "quality_span_resolved"]
    compact = frame[columns].copy()
    compact["correct_canonical"] = compact.correct_canonical.astype("Int8")
    compact["correct_literal"] = compact.correct_literal.astype("Int8")
    compact["abstain"] = compact.abstain.astype("boolean")
    compact.to_parquet(output / "new_scores.parquet", index=False)
    compact.to_csv(output / "new_scores.csv", index=False)
    pd.DataFrame(sources).to_csv(output / "sources.csv", index=False)
    references.to_csv(output / "frozen_references_used.csv", index=False)
    result = {"schema": "kdm_core_dev_viz_scoring_v1", "passed": True, "stage": args.stage,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "rows": len(scored), "conditions": len(metrics),
        "raw_complete_conditions": sum(item["raw_complete"] for item in metrics),
        "primary_complete_conditions": sum(item["primary_complete"] for item in metrics),
        "pilot_partial_conditions": sum(item["report_scope"] == "pilot_partial" for item in metrics),
        "canonical_pending_rows": int(frame.correct_canonical.isna().sum()) if args.stage == "dev404" else None,
        "literal_pending_rows": int(frame.correct_literal.isna().sum()) if args.stage == "dev404" else None,
        "quality_pending_rows": int(frame.quality_score.isna().sum()) if args.stage == "viz512" else None,
        "abstain_pending_rows": int(frame.abstain.isna().sum()), "pending_QA": len(pending), "pending_memberships": len(memberships),
        "unique_QA": len(qa_cache), "behavior_source_counts": dict(Counter(frame.behavior_source)),
        "timing": {"validation_s": validation_s, "authority_load_s": authority_load_s, "inference_and_scoring_s": scoring_s,
                   "elapsed_s": time.perf_counter() - start},
        "input_dir": str(within(ROOT, args.input_dir)) if args.input_dir else None,
        "census_direct_container": str(within(ROOT, args.census_direct)) if args.census_direct else None,
        "source_parts": sources, "reference_path": str(reference_path), "reference_sha256": reference_sha,
        "reference_join_missing": 0, "authority_manifest_path": str(authority_manifest_path),
        "authority_manifest_sha256": file_hash(authority_manifest_path), "canonical_scorer_sha256": canonical_sha,
        "reused_inference_source_sha256": file_hash(ROOT / "workflows/supplemental/remaining11/score.py"),
        "accepted_native_authority_receipts": accepted_receipts,
        "decision_files": [{"path": str(path), "sha256": file_hash(path)} for path in decision_paths],
        "viz_authority_files": [{"path": str(path), "sha256": file_hash(path)} for path in viz_authority_paths],
        "viz_final_census_source": viz_census_source,
        "official_vqa_normalizer_sha256": file_hash(ROOT / "src/kdm/models/official_vqa_normalizer.py"),
        "official_vqa_scoring_sha256": file_hash(ROOT / "src/kdm/scoring.py"),
        "runner_sha256": file_hash(Path(__file__)), "rule_executor": EXECUTOR,
        "actual_command": [sys.executable, *sys.argv], "new_generations": 0, "new_API_calls": 0,
        "GPU_initialized": False, "active_raw_opened": 0, "frozen_assets_modified": False,
        "outputs": {path.name: file_hash(path) for path in output.iterdir()}}
    save_json(output / "receipt.json", result)
    print(json.dumps({key: result[key] for key in ("passed", "rows", "conditions", "raw_complete_conditions",
        "primary_complete_conditions", "pilot_partial_conditions", "canonical_pending_rows", "literal_pending_rows",
        "quality_pending_rows", "abstain_pending_rows", "pending_QA", "unique_QA", "timing")}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir")
    parser.add_argument("--census-direct")
    parser.add_argument("--stage", choices=("dev404", "viz512"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--selected-configs")
    parser.add_argument("--decision-file", action="append", default=[])
    parser.add_argument("--viz-authority-file", action="append", default=[])
    parser.add_argument("--authority-manifest", default=AUTHORITY)
    parser.add_argument("--native-score-receipt", default=NATIVE_RECEIPT)
    args = parser.parse_args()
    if not args.input_dir and not args.census_direct:
        parser.error("requires an immutable --input-dir or finite copied --census-direct")
    score(args)


if __name__ == "__main__":
    main()
