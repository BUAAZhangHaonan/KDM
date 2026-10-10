#!/usr/bin/env python3
"""Score the exact existing selected-four unguided census eval replies on CPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.decoding import DecodeConfig
from kdm.io import file_hash, stable_hash, stable_seed, within
from kdm.models.official_vqa_normalizer import VQAEval
from kdm.pipeline import census_tasks, task_id
from kdm.prompts import task_prompt
from kdm.scoring import vqa_score
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.score import infer_qa, load_authority, rows, score_target
from workflows.supplemental.remaining11.vizwiz_score import metrics as viz_metrics, write_csv
from workflows.supplemental.remaining4.score_native import MODELS, DEFAULT_AUTHORITY, DEFAULT_AUTHORITY_SCORE, save_json, save_rows

OUTPUT_ROOT = "outputs/supplemental/remaining4/direct_cpu_20261001_1600"
AUTHORITY_RECEIPT = "outputs/supplemental/remaining4/reference_cpu_20261001_0325/scores/v2_root42_closed/finite_execution_receipt.json"
VIZ_ASSETS = "outputs/supplemental/remaining11/run_20260930_140337/assets/vizwiz/audit_20260930"
CONTEXT = ("model", "dataset", "split", "method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")


def output_path(value):
    path = within(ROOT, value)
    path.relative_to(ROOT / OUTPUT_ROOT)
    return path


def prepare(args):
    output = output_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = within(ROOT, DEFAULT_AUTHORITY)
    manifest = json.loads(manifest_path.read_text())
    sample_path = ROOT / "data/current/all.jsonl"
    sample_hash = file_hash(sample_path)
    samples = {r["id"]: r for _line, r, _sha in rows(sample_path) if r["split"] == "eval"}
    if len(samples) != 5925:
        raise ValueError("Full original Food2424/Viz3501 eval cohort required")
    source_entries = {r["model"]: r for r in manifest["census_sources"]}
    inputs, seen, proofs = [], set(), []
    config = asdict(DecodeConfig())
    for model in MODELS:
        source = source_entries[model]
        path, identity_path = Path(source["path"]), Path(source["identity_path"])
        meta = json.loads(identity_path.read_text())
        identity_hash = file_hash(identity_path)
        definition = meta["definition"]
        if (meta["identity"] != stable_hash(definition) or definition["model"] != model
                or definition["backend"]["key"] != model or definition["backend"].get("api")
                or definition["manifest_sha256"] != sample_hash or definition["base_config"] != config
                or set(source["raw_identities"]) != {meta["identity"]} or source["rows"] != 18334):
            raise ValueError("Original census model, identity, input cohort or parameters differ")
        total, selected = 0, []
        for line, raw, sha in rows(path):
            total += 1
            if raw["guided"] or raw["sample"]["split"] != "eval":
                continue
            sample = raw["sample"]
            task = next(t for t in census_tasks([sample]) if not t["guided"])
            if (sample != samples.get(sample["id"]) or raw["identity"] != meta["identity"]
                    or raw["model"] != model or raw["key"] != task_id(model, task)
                    or any(raw[field] != task[field] for field in CONTEXT[3:]) or raw["config"] != config
                    or raw["seed"] != stable_seed(sample["id"], model, 0)
                    or raw["prompt"] != task_prompt(sample["question"], "UNKNOWN", False)
                    or raw["reference_prompt"] is not None or type(raw["terminated"]) is not bool
                    or len(raw["tokens"]) > 32 or not isinstance(raw["text"], str) or raw["key"] in seen):
                raise ValueError("Original unguided census key/sample/prompt/config/seed/termination differs")
            probabilities = [*raw["selected_log_probabilities"], raw["sequence_log_probability"], raw["first_probability"]]
            if not all(math.isfinite(value) for value in probabilities) or not 0 <= raw["first_probability"] <= 1:
                raise ValueError("Original direct selected probabilities are nonfinite or outside range")
            seen.add(raw["key"])
            selected.append({"raw": raw, "source_path": str(path), "source_line": line,
                "source_identity_path": str(identity_path), "source_identity_sha256": identity_hash,
                "raw_line_sha256": sha, "source_identity": meta["identity"], "original_key": raw["key"]})
        if total != 18334 or len(selected) != 5925 or Counter(r["raw"]["sample"]["dataset"] for r in selected) != {"food101": 2424, "vizwiz": 3501}:
            raise ValueError("The original immutable census source lacks full selected eval keys")
        inputs.extend(selected)
        proofs.append({"model": model, "path": str(path), "identity_path": str(identity_path),
            "identity_sha256": identity_hash, "source_identity": meta["identity"], "original_definition": definition,
            "recorded_census_rows": total, "selected_eval_rows": len(selected), "parameters_changed": False,
            "raw_rehashed": False, "other_warehouse_raw_read": False})
    labels_path = Path(manifest["census_final_labels"]["path"])
    selected_labels, label_rows = {}, 0
    for line, label, sha in rows(labels_path):
        label_rows += 1
        if label["key"] in seen:
            if label["key"] in selected_labels:
                raise ValueError("Final closed census annotation duplicates an original source key")
            selected_labels[label["key"]] = {"label": label, "source_path": str(labels_path), "source_line": line, "line_sha256": sha}
    if label_rows != 293344 or set(selected_labels) != seen:
        raise ValueError("The final 293344 closure does not bind all selected unguided source replies")
    for item in inputs:
        raw, sample = item["raw"], item["raw"]["sample"]
        witness = selected_labels[raw["key"]]
        label = witness["label"]
        if (label["question"] != sample["question"] or label["text"] != raw["text"] or label["model"] != raw["model"]
                or label["sample_id"] != sample["id"] or label["raw_identity"] != raw["identity"] or label["guided"] is not False
                or label["dataset"] != sample["dataset"] or label["label"] not in {"answer_assertive", "answer_uncertain", "abstain", "invalid"}):
            raise ValueError("Final annotation and original raw exact-key/identity/full-QA binding differs")
        item["final_census_annotation"] = witness
    save_rows(output / "unguided_direct_source_bound.jsonl.gz", inputs)
    save_json(output / "source_verification.json", {"schema": "kdm_remaining4_unguided_direct_source_verification_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "rows": len(inputs), "unique_original_keys": len(seen), "models": list(MODELS),
        "Food_eval_per_model": 2424, "Viz_eval_per_model": 3501, "census_final_annotation_rows": label_rows,
        "selected_source_annotation_memberships": len(selected_labels), "sources": proofs,
        "manifest_path": str(manifest_path.relative_to(ROOT)), "manifest_sha256": file_hash(manifest_path), "sample_sha256": sample_hash,
        "outputs": {"unguided_direct_source_bound.jsonl.gz": file_hash(output / "unguided_direct_source_bound.jsonl.gz")},
        "adapter_sha256": file_hash(Path(__file__)), "GPU_initialized": False, "new_generations": 0, "raw_parameters_changed": False})
    print(json.dumps({"prepared_rows": len(inputs), "Food": 9696, "Viz": 14004, "new_generations": 0}), flush=True)


def score(args):
    prepared = output_path(args.prepared)
    source_receipt = json.loads((prepared / "source_verification.json").read_text())
    source_path = prepared / "unguided_direct_source_bound.jsonl.gz"
    if not source_receipt["passed"] or file_hash(source_path) != source_receipt["outputs"][source_path.name]:
        raise ValueError("Prepared census source bytes differ from actual verification")
    output = output_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads(within(ROOT, DEFAULT_AUTHORITY).read_text())
    authority_receipt_path = within(ROOT, AUTHORITY_RECEIPT)
    authority_receipt = json.loads(authority_receipt_path.read_text())
    if not authority_receipt["passed"] or any(authority_receipt[field] for field in ("canonical_pending", "literal_pending", "abstain_pending")):
        raise ValueError("Only accepted closed exact-QA decisions may be reused")
    decision_specs = list(authority_receipt["decision_files"])
    decision_specs.append({"path": authority_receipt["actual_root_decision_path"], "sha256": authority_receipt["actual_root_decision_sha256"]})
    for spec in decision_specs:
        if file_hash(within(ROOT, spec["path"])) != spec["sha256"]:
            raise ValueError("A previously closed source-bound decision file changed")
    decision_paths = [within(ROOT, spec["path"]) for spec in decision_specs]
    decision_paths.extend(within(ROOT, path) for path in args.decision_file)
    reviews, behavior, behavior_sources, _history, decisions = load_authority(
        within(ROOT, DEFAULT_AUTHORITY_SCORE), {"census_final_labels": original["census_final_labels"], "historical_labels": []}, decision_paths)
    classes = sorted({r["class"] for _line, r, _sha in rows(ROOT / "data/current/all.jsonl") if r["dataset"] == "food101"})
    if len(classes) != 101:
        raise ValueError("Original Food canonical class registration differs")
    patterns = frozen.compile_classes(classes)
    official_path = ROOT / VIZ_ASSETS / "official_reference.jsonl"
    official = {r["sample_id"]: r for _line, r, _sha in rows(official_path) if r["split"] == "eval"}
    evaluator = VQAEval(None, None)
    normalizer = lambda text: evaluator.processDigitArticle(evaluator.processPunctuation(text))
    groups, cache, pending, scored = defaultdict(list), {}, {}, []
    for _line, item, _sha in rows(source_path):
        raw, sample = item["raw"], item["raw"]["sample"]
        qkey = frozen.qah(sample["question"], raw["text"])
        witness, label = item["final_census_annotation"], item["final_census_annotation"]["label"]
        context = {"model": raw["model"], "dataset": sample["dataset"], "split": "eval", **{field: raw[field] for field in CONTEXT[3:]}}
        record = {**context, "main_marker": raw["marker"], "stage": "unguided_direct", "key": raw["key"], "original_key": raw["key"],
            "sample_id": sample["id"], "question": sample["question"], "answer": raw["text"], "qa_key": qkey,
            "source_identity": raw["identity"], **{k: item[k] for k in ("source_path", "source_line", "source_identity_path", "source_identity_sha256", "raw_line_sha256")},
            "seed": raw["seed"], "config": raw["config"], "prompt": raw["prompt"], "reference_prompt": raw["reference_prompt"],
            "terminated": raw["terminated"], "final_census_annotation": witness,
            "reference_G": None, "reference_complete": False, "knowledge_deficit_GT": None, "independent_reference_complete": False}
        if sample["dataset"] == "food101":
            if qkey not in cache:
                cache[qkey] = infer_qa(sample["question"], raw["text"], patterns, reviews, behavior, decisions)
            inferred = cache[qkey]
            canonical, literal, reason = score_target(raw["text"], sample["class"], inferred, patterns)
            names = sorted({name for variant in inferred["variants"] for name in frozen.extract(variant, patterns)["literal"]})
            if not names and inferred["parsed"]["primary_name"]:
                names = [inferred["parsed"]["primary_name"]]
            record.update({"target_class": sample["class"], "canonical_name_in_primary_score": canonical, "literal_extracted_name_score": literal,
                "literal_extracted_names": names, "abstain": inferred["abstain"], "score_reason": reason,
                "behavior_source": inferred["behavior_source"], "primary_extraction": inferred["parsed"],
                "behavior_history_sources": behavior_sources.get(qkey, []), "name_history_source": inferred["review"],
                "decision_source": inferred["decision"], "annotation_model": inferred["annotation_model"],
                "annotation_effort": inferred["annotation_effort"], "annotation_call_id": inferred["annotation_call_id"],
                "vqa_credit": None, "label": label["label"], "answer_text": None})
            unresolved = canonical is None or literal is None or inferred["abstain"] is None
        else:
            ref = official[sample["id"]]
            if ref["official_answers"] != sample["official_answers"] or ref["annotated_answerable"] != sample["annotated_answerable"]:
                raise ValueError("Original official ten-answer/answerability source differs")
            answer_text = label.get("answer_text", "")
            span_ok = not answer_text if label["label"] in {"abstain", "invalid"} else isinstance(answer_text, str) and bool(answer_text) and answer_text in raw["text"]
            credit = (0.0 if label["label"] in {"abstain", "invalid"} else float(vqa_score(answer_text, ref["official_answers"], normalizer))) if span_ok else None
            if credit is not None and (not math.isfinite(credit) or not 0 <= credit <= 1):
                raise ValueError("Official VQA consensus credit is nonfinite or outside range")
            record.update({"target_class": None, "canonical_name_in_primary_score": None, "literal_extracted_name_score": None,
                "literal_score_scope": "Food-name field not applicable to official VizWiz VQA", "literal_answer_text": answer_text,
                "label": label["label"], "answer_text": answer_text, "abstain": label["label"] == "abstain", "vqa_credit": credit,
                "score_reason": "final_census_official_VQA" if span_ok else "final_answer_span_unresolved",
                "behavior_source": "original_key_identity_bound_final_census", "annotated_answerable": ref["annotated_answerable"],
                "official_answers": ref["official_answers"], "official_source_path": ref["official_source_path"],
                "official_source_record": ref["official_source_record"], "official_record_sha256": ref["official_record_sha256"]})
            unresolved = not span_ok
        scored.append(record)
        groups[raw["model"], sample["dataset"]].append(record)
        if unresolved:
            packet = pending.setdefault(qkey, {"qa_key": qkey, "question": sample["question"], "answer": raw["text"],
                "primary_extraction": record.get("primary_extraction"), "reasons": [], "memberships": []})
            packet["reasons"] = sorted(set(packet["reasons"] + [record["score_reason"]]))
            packet["memberships"].append({k: record[k] for k in ("model", "dataset", "key", "sample_id", "target_class",
                "source_path", "source_line", "source_identity", "raw_line_sha256", "seed")})
    if len(scored) != 23700 or len({r["key"] for r in scored}) != 23700:
        raise ValueError("Full selected-four unguided eval key coverage differs")
    counts = []
    for (model, dataset), group in groups.items():
        context = {field: group[0][field] for field in CONTEXT}
        canonical_pending = sum(r["canonical_name_in_primary_score"] is None for r in group) if dataset == "food101" else None
        literal_pending = sum(r["literal_extracted_name_score"] is None for r in group) if dataset == "food101" else None
        abstain_pending = sum(r["abstain"] is None for r in group)
        if dataset == "food101":
            quota = Counter(r["target_class"] for r in group)
            if len(group) != 2424 or len(quota) != 101 or set(quota.values()) != {24}:
                raise ValueError("Full Direct Food eval denominator or per-class quotas differ")
            c = sum(r["canonical_name_in_primary_score"] == 1 for r in group)
            metrics = {**context, "main_marker": context["marker"], "n": len(group), "primary_metric": "canonical_accuracy",
                "primary_numerator": c, "primary_denominator": 2424, "primary_value": c / 2424 if canonical_pending == abstain_pending == 0 else None,
                "canonical_correct": c, "literal_correct": sum(r["literal_extracted_name_score"] == 1 for r in group),
                "abstain_n": sum(r["abstain"] is True for r in group), "abstain_rate": sum(r["abstain"] is True for r in group) / 2424 if abstain_pending == 0 else None,
                "reference_connected_n": 0, "reference_precision": None, "reference_recall": None}
        else:
            credit_pending = sum(r["vqa_credit"] is None for r in group)
            metrics = {**context, "main_marker": context["marker"], "n": len(group), "primary_metric": "official_VQA_credit",
                "primary_numerator": math.fsum(r["vqa_credit"] for r in group if r["vqa_credit"] is not None),
                "primary_denominator": 3501, "primary_value": None, "vqa_credit_pending": credit_pending}
            if credit_pending == abstain_pending == 0:
                metrics.update(viz_metrics(context, group))
                metrics["primary_value"] = metrics["all_input_vqa_accuracy"]
        counts.append({**metrics, "canonical_pending": canonical_pending, "literal_pending": literal_pending, "abstain_pending": abstain_pending})
    save_rows(output / "score_rows.jsonl.gz", scored)
    save_rows(output / "boundary_queue.jsonl", [pending[k] for k in sorted(pending)])
    save_rows(output / "condition_metrics.jsonl", counts)
    write_csv(output / "condition_metrics.csv", counts)
    receipt = {"schema": "kdm_remaining4_unguided_direct_score_v1", "passed": True, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(scored), "unique_keys": len(scored), "Food_rows": 9696, "Viz_rows": 14004,
        "Food_canonical_pending": sum(r["dataset"] == "food101" and r["canonical_name_in_primary_score"] is None for r in scored),
        "Food_literal_pending": sum(r["dataset"] == "food101" and r["literal_extracted_name_score"] is None for r in scored),
        "semantic_abstain_pending": sum(r["abstain"] is None for r in scored), "Viz_official_credit_pending": sum(r["dataset"] == "vizwiz" and r["vqa_credit"] is None for r in scored),
        "boundary_QA": len(pending), "condition_count": len(counts), "condition_metrics": counts,
        "source_verification_path": str((prepared / "source_verification.json").relative_to(ROOT)),
        "source_verification_sha256": file_hash(prepared / "source_verification.json"), "prepared_source_sha256": file_hash(source_path),
        "authority_receipt_sha256": file_hash(authority_receipt_path), "decision_files": [{"path": str(p.relative_to(ROOT)), "sha256": file_hash(p)} for p in decision_paths],
        "canonical_source_sha256": file_hash(Path(frozen.__file__)), "supplemental_scorer_sha256": file_hash(ROOT / "workflows/supplemental/remaining11/score.py"),
        "official_VQA_source_sha256": file_hash(ROOT / "src/kdm/scoring.py"), "official_normalizer_sha256": file_hash(ROOT / "src/kdm/models/official_vqa_normalizer.py"),
        "adapter_sha256": file_hash(Path(__file__)), "outputs": {p.name: file_hash(p) for p in output.iterdir()},
        "GPU_initialized": False, "new_generations": 0, "new_API_calls": 0, "scientific_parameters_changed": False,
        "old_outputs_changed": False, "uniform_reference_joined": False, "unknown_labels_defaulted_to_zero": False}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("rows", "Food_canonical_pending", "Food_literal_pending", "semantic_abstain_pending", "Viz_official_credit_pending", "boundary_QA")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "score"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--decision-file", action="append", default=[])
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        if args.prepared is None:
            parser.error("Scoring requires the actual prepared source-verification directory")
        score(args)


if __name__ == "__main__":
    main()
