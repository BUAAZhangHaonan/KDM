#!/usr/bin/env python3
"""Score source-bound VizWiz Direct baselines with the official VQA rules."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.decoding import DecodeConfig
from kdm.io import file_hash, read_jsonl, stable_hash, stable_seed
from kdm.models.official_vqa_normalizer import VQAEval
from kdm.pipeline import experiment_tasks, task_id
from kdm.prompts import task_prompt
from kdm.scoring import vqa_score
from workflows.supplemental.remaining11.inventory import MODELS

COND = ("model", "dataset", "split", "method", "kind", "marker", "reference_marker",
        "guided", "reference_guided", "replicate")


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_jsonl(path, values):
    opener = gzip.open if path.suffix == ".gz" else open
    options = {"compresslevel": 1} if path.suffix == ".gz" else {}
    with opener(path, "xt", encoding="utf-8", newline="\n", **options) as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def qah(question, answer):
    return hashlib.sha256((question + "\0" + answer).encode()).hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def accepted_coverage(path):
    counts, splits, unique = Counter(), Counter(), set()
    for row in read_jsonl(path):
        key = row["model"], row["sample_id"]
        if key in unique or type(row["gt"]) is not bool:
            raise ValueError("Historical accepted GT key/value differs")
        unique.add(key)
        counts[row["model"]] += 1
        splits[(row["model"], row["split"])] += 1
    return {"source_path": str(path), "source_sha256": file_hash(path), "actual_rows": sum(counts.values()),
            "unique_model_sample_keys": len(unique), "model_counts": dict(counts),
            "split_counts": [{"model": model, "split": split, "rows": n} for (model, split), n in sorted(splits.items())],
            "remaining11": [{"model": model, "historical_accepted_rows": counts[model],
                              "historical_accepted_reference": None if not counts[model] else "existing_source_rows_only"}
                             for model in MODELS],
            "mapping_rule": "exact registered model keys; no model aliases or transplantation across datasets"}


def metrics(context, rows):
    complete = len(rows) == 3501 and all(r["vqa_credit"] is not None and r["label"] is not None for r in rows)
    if not complete:
        raise ValueError("Only complete decided VizWiz eval conditions enter metrics")
    answered = [r for r in rows if r["label"] in {"answer_assertive", "answer_uncertain"}]
    abstained = [r for r in rows if r["label"] == "abstain"]
    invalid = [r for r in rows if r["label"] == "invalid"]
    answerable = [r for r in rows if r["annotated_answerable"] == 1]
    unanswerable = [r for r in rows if r["annotated_answerable"] == 0]
    if (len(answerable), len(unanswerable)) != (2365, 1136) or len(answered) + len(abstained) + len(invalid) != 3501:
        raise ValueError("VizWiz official/response denominators differ")
    credit = math.fsum(r["vqa_credit"] for r in rows)
    return {**context, "main_marker": context["marker"], "n": len(rows),
            "vqa_credit_numerator": credit, "all_input_vqa_accuracy": credit / len(rows),
            "answered_n": len(answered), "answer_coverage": len(answered) / len(rows),
            "answered_vqa_credit_numerator": math.fsum(r["vqa_credit"] for r in answered),
            "answered_vqa_accuracy": ratio(math.fsum(r["vqa_credit"] for r in answered), len(answered)),
            "abstain_n": len(abstained), "abstain_rate": len(abstained) / len(rows), "invalid_n": len(invalid),
            "human_answerable_n": len(answerable), "human_unanswerable_n": len(unanswerable),
            "direct_answerable_vqa_credit_numerator": math.fsum(r["vqa_credit"] for r in answerable),
            "direct_answerable_vqa_accuracy": math.fsum(r["vqa_credit"] for r in answerable) / len(answerable),
            "direct_answerable_positive_credit_n": sum(r["vqa_credit"] > 0 for r in answerable),
            "direct_answerable_full_credit_n": sum(r["vqa_credit"] == 1 for r in answerable),
            "abstentions_on_human_answerable_n": sum(r["label"] == "abstain" for r in answerable),
            "abstentions_on_human_unanswerable_n": sum(r["label"] == "abstain" for r in unanswerable),
            "human_unanswerable_share_of_abstentions": ratio(sum(r["label"] == "abstain" for r in unanswerable), len(abstained)),
            "independent_attempt_mean_correctness": None, "independent_reference_complete": False,
            "knowledge_deficit_GT": None, "full_method_effects_complete": False}


def make_figure(output, condition_metrics):
    paths = []
    if not condition_metrics:
        return paths
    labels = [row["model"] for row in condition_metrics]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(x - 0.18, [row["all_input_vqa_accuracy"] for row in condition_metrics], 0.36, label="Official VQA credit / all eval inputs")
    ax.bar(x + 0.18, [row["abstain_rate"] for row in condition_metrics], 0.36, label="Abstention rate")
    ax.set_xticks(x, labels, rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction / official consensus credit")
    ax.set_title("Remaining-model VizWiz: Direct / UNKNOWN (3,501 eval samples)")
    ax.legend()
    fig.tight_layout()
    (output / "figures").mkdir()
    for extension in ("png", "pdf"):
        path = output / "figures" / ("vizwiz_direct_unknown." + extension)
        fig.savefig(path, dpi=180)
        paths.append(str(path))
    plt.close(fig)
    return paths


def score(assets, output):
    manifest_path = assets / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "kdm_remaining11_vizwiz_asset_manifest_v1":
        raise ValueError("Registered VizWiz independent asset manifest required")
    samples = {r["id"]: r for r in read_jsonl(ROOT / "data/current/all.jsonl") if r["dataset"] == "vizwiz" and r["split"] == "eval"}
    if len(samples) != 3501 or Counter(r["annotated_answerable"] for r in samples.values()) != {0: 1136, 1: 2365}:
        raise ValueError("Registered full VizWiz eval split differs")
    sources = {s["model"]: s for s in manifest["sources"]}
    if set(sources) != set(MODELS):
        raise ValueError("All registered remaining-model baseline sources required")
    raw, required_qa, source_digests = {}, set(), {}
    for model in MODELS:
        path = Path(sources[model]["derived_formal_path"])
        sidecar = json.loads(path.with_suffix(".identity.json").read_text())
        if stable_hash(sidecar["definition"]) != sidecar["identity"]:
            raise ValueError("Derived source sidecar differs")
        source_digests[str(path)] = file_hash(path)
        rows = {}
        for number, row in enumerate(read_jsonl(path), 1):
            sample = row["sample"]
            task = {"sample": sample, "method": "direct", "kind": "main", "marker": "UNKNOWN", "reference_marker": "UNKNOWN", "guided": True, "reference_guided": True, "replicate": 0}
            if (sample != samples.get(sample["id"]) or row["key"] != task_id(model, task)
                    or row["identity"] != sidecar["identity"] or row["original_identity"] != sidecar["definition"]["source_identity"]
                    or row["model"] != model or any(row[field] != task[field] for field in COND[3:])
                    or row["config"] != asdict(DecodeConfig()) or row["seed"] != stable_seed(sample["id"], model, 0)
                    or row["prompt"] != task_prompt(sample["question"], "UNKNOWN", True)
                    or sample["id"] in rows):
                raise ValueError("Source-bound VizWiz formal identity/config/sample differs")
            rows[sample["id"]] = {"row": row, "derived_source_path": str(path), "derived_source_line": number}
            required_qa.add(qah(sample["question"], row["text"]))
        if set(rows) != set(samples):
            raise ValueError("VizWiz Direct source coverage incomplete")
        raw[model] = rows
    labels_path = Path(manifest["final_census_labels"]["original_path"])
    labels_by_original, qa_variants = {}, defaultdict(set)
    label_rows = 0
    for label in read_jsonl(labels_path):
        label_rows += 1
        key = qah(label["question"], label["text"])
        if key in required_qa:
            if label["key"] in labels_by_original:
                raise ValueError("Duplicate final census annotation key")
            labels_by_original[label["key"]] = label
            qa_variants[key].add((label["label"], label.get("answer_text", "")))
    if label_rows != 293344:
        raise ValueError("Complete 293,344-row final annotation closure required")
    official = {row["sample_id"]: row for row in read_jsonl(assets / "official_reference.jsonl")}
    official_eval = {sid: row for sid, row in official.items() if row["split"] == "eval"}
    if set(official_eval) != set(samples):
        raise ValueError("Official reference coverage differs")
    ev = VQAEval(None, None)
    normalizer = lambda text: ev.processDigitArticle(ev.processPunctuation(text))
    scored, groups, pending, cases = [], {}, [], []
    for model in MODELS:
        group = []
        case_counts = Counter()
        for sid, item in sorted(raw[model].items()):
            row, sample = item["row"], samples[sid]
            label = labels_by_original.get(row["original_key"])
            if label is None or label != row["historical_final_label"]:
                raise ValueError("Derived and original final census annotations differ")
            if (label["text"] != row["text"] or label["question"] != sample["question"]
                    or label["raw_identity"] != row["original_identity"] or label["model"] != model
                    or label["sample_id"] != sid or label["guided"] is not True):
                raise ValueError("Original final annotation source binding differs")
            reference = official_eval[sid]
            if reference["official_answers"] != sample["official_answers"] or reference["annotated_answerable"] != sample["annotated_answerable"]:
                raise ValueError("Official answers/answerability differ")
            behavior, answer = label["label"], label.get("answer_text", "")
            span_resolved = (not answer if behavior in {"abstain", "invalid"} else isinstance(answer, str) and bool(answer) and answer in row["text"])
            if behavior not in {"abstain", "invalid", "answer_assertive", "answer_uncertain"}:
                raise ValueError("Final behavior annotation remains unresolved")
            credit = (0.0 if behavior in {"abstain", "invalid"} else float(vqa_score(answer, reference["official_answers"], normalizer))) if span_resolved else None
            if credit is not None and (not math.isfinite(credit) or not 0 <= credit <= 1):
                raise ValueError("Official VQA credit is nonfinite/outside range")
            qkey = qah(sample["question"], row["text"])
            record = {"model": model, "dataset": "vizwiz", "split": "eval", "stage": "formal",
                "method": "direct", "kind": "main", "marker": "UNKNOWN", "main_marker": "UNKNOWN", "reference_marker": "UNKNOWN",
                "guided": True, "reference_guided": True, "replicate": 0, "key": row["key"], "sample_id": sid,
                "question": sample["question"], "answer": row["text"], "answer_text": answer,
                "label": behavior, "abstain": behavior == "abstain", "vqa_credit": credit,
                "annotated_answerable": reference["annotated_answerable"], "answer_type": sample["answer_type"],
                "official_answers": reference["official_answers"], "official_source_path": reference["official_source_path"],
                "official_source_record": reference["official_source_record"], "official_record_sha256": reference["official_record_sha256"],
                "knowledge_deficit_GT": None, "independent_reference_complete": False, "qa_key": qkey,
                "exact_QA_label_variants": [list(v) for v in sorted(qa_variants[qkey])],
                "score_authority": "original-key/source-identity-bound final census answer_text; exact QA witnesses retained",
                "original_key": row["original_key"], "original_identity": row["original_identity"], "derived_identity": row["identity"],
                "original_source_path": row["source_path"], "original_source_line": row["source_line"],
                "original_raw_line_sha256": row["source_row_sha256"], **{k: v for k, v in item.items() if k != "row"},
                "final_annotation_path": str(labels_path), "final_annotation_identity": label.get("identity"),
                "historical_annotation_provenance": {k: v for k, v in label.items() if k in {"label_source", "judge_response_model", "judge_response_id", "parent_annotation_identity", "source_annotation_identity", "model", "raw_identity", "raw_record_sha256"}},
                "seed": row["seed"], "config_sha256": stable_hash(row["config"]), "terminated": row["terminated"]}
            scored.append(record)
            group.append(record)
            if credit is None:
                pending.append({"model": model, "sample_id": sid, "original_key": row["original_key"], "qa_key": qkey, "reason": "final_answer_span_unresolved"})
            category = "abstention" if behavior == "abstain" else ("positive_official_credit" if credit and credit > 0 else "zero_official_credit")
            if case_counts[category] < 2:
                cases.append({"case_category": category, "selection": "first registered sample in lexical sample-ID order", **record})
                case_counts[category] += 1
        groups[model] = group
    condition_metrics, answerability = [], []
    for model, rows in groups.items():
        context = {field: rows[0][field] for field in COND}
        if len(rows) == 3501 and all(r["vqa_credit"] is not None for r in rows):
            condition_metrics.append(metrics(context, rows))
        for human in (0, 1):
            subgroup = [r for r in rows if r["annotated_answerable"] == human]
            answered = [r for r in subgroup if r["label"] in {"answer_assertive", "answer_uncertain"}]
            resolved = all(r["vqa_credit"] is not None for r in subgroup)
            answerability.append({**context, "annotated_answerable": human, "n": len(subgroup),
                "answered_n": len(answered), "abstain_n": sum(r["label"] == "abstain" for r in subgroup),
                "invalid_n": sum(r["label"] == "invalid" for r in subgroup), "answer_coverage": len(answered) / len(subgroup),
                "abstain_rate": sum(r["label"] == "abstain" for r in subgroup) / len(subgroup),
                "direct_answerable_mean_vqa_credit": math.fsum(r["vqa_credit"] for r in subgroup) / len(subgroup) if human and resolved else None,
                "independent_attempt_mean_correctness": None,
                "success_scope": "Direct consensus credit on human-answerable items" if human else "human unanswerability reported separately"})
    plan = json.loads((ROOT / "configs/kdm/method_plan.json").read_text())
    coverage = []
    for model in MODELS:
        for task in experiment_tasks((next(iter(samples.values())),), methods=plan[model]["vizwiz"]):
            context = {"model": model, "dataset": "vizwiz", "split": "eval", **{field: task[field] for field in COND[3:]}}
            available = task["method"] == "direct" and task["marker"] == "UNKNOWN"
            coverage.append({**context, "expected_n": 3501, "scored_n": 3501 if available else 0,
                "score_unresolved_n": sum(r["vqa_credit"] is None for r in groups[model]) if available else None,
                "missing_scored_rows": 0 if available else 3501,
                "condition_complete": available and not any(r["vqa_credit"] is None for r in groups[model])})
    if len(coverage) != 800 or len(scored) != 38511:
        raise ValueError("Registered VizWiz score/condition totals differ")
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "score_rows.jsonl.gz", scored)
    write_csv(output / "condition_metrics.csv", condition_metrics)
    write_csv(output / "answerability_metrics.csv", answerability)
    write_csv(output / "condition_coverage.csv", coverage)
    write_jsonl(output / "pending_score_rows.jsonl", pending)
    write_jsonl(output / "cases.jsonl", cases)
    write_json(output / "legacy_accepted_reference_coverage.json", accepted_coverage(ROOT / "data/reference_gt/legacy_accepted_reference_gt.jsonl"))
    write_json(output / "reference_status.json", {"official_reference_source": str(assets / "official_reference.jsonl"),
        "official_reference_eval_rows": 3501, "human_answerable_eval_rows": 2365, "human_unanswerable_eval_rows": 1136,
        "registered_independent_attempt_rows": 385110, "independent_attempt_scores_available_in_this_baseline_checkpoint": 0,
        "missing_independent_scored_rows": 385110, "knowledge_deficit_GT": None,
        "rule": "registered human answerability and human-answerable repeated-attempt credit are distinct evidence sources; no combined binary knowledge GT"})
    figure_paths = make_figure(output, condition_metrics)
    summary = {"schema": "kdm_remaining11_vizwiz_baseline_score_v1", "updated_utc": datetime.now(timezone.utc).isoformat(),
        "asset_manifest_path": str(manifest_path), "output_directory": str(output), "scored_rows": len(scored),
        "registered_formal_rows": 2800800, "registered_conditions": len(coverage),
        "complete_decided_conditions": len(condition_metrics), "pending_score_rows": len(pending),
        "missing_scored_formal_rows": sum(r["missing_scored_rows"] for r in coverage),
        "exact_QA_witness_count": len(required_qa), "original_final_label_rows_read": label_rows,
        "method_effects_computed": 0, "case_rows": len(cases), "figures": figure_paths,
        "input_sha256": {"asset_manifest": file_hash(manifest_path), "official_reference": file_hash(assets / "official_reference.jsonl"),
            "final_census_labels": file_hash(labels_path), "official_vqa_normalizer": file_hash(ROOT / "src/kdm/models/official_vqa_normalizer.py"),
            "vqa_scoring_function_source": file_hash(ROOT / "src/kdm/scoring.py"), "all_manifest": file_hash(ROOT / "data/current/all.jsonl")},
        "derived_source_sha256": source_digests, "scorer_sha256": file_hash(Path(__file__)),
        "authorship": "programmatic official VQA scoring; original final annotation authors and calls retained per record",
        "new_API_calls": 0, "GPU_initialized": False, "complete_registered_methods": False}
    report = ["# VizWiz剩余11模型Direct补充结果", "", f"已评分{len(scored):,}条来源绑定Direct/UNKNOWN回复，完整已决条件{len(condition_metrics)}个，每条件固定eval 3,501题。",
        "每题官方十答案、人工可回答性及原始回答span均保留。指标调用冻结official_vqa_normalizer和vqa_score；弃权与无效按原协议计0信用。", "",
        "| 模型 | 全输入官方VQA信用 | 弃权率 | 回答覆盖率 | 可回答题Direct信用 |", "|---|---:|---:|---:|---:|"]
    for r in condition_metrics:
        report.append(f"| {r['model']} | {r['all_input_vqa_accuracy']*100:.2f}% | {r['abstain_rate']*100:.2f}% | {r['answer_coverage']*100:.2f}% | {r['direct_answerable_vqa_accuracy']*100:.2f}% |")
    report += ["", "eval官方可回答2,365题、不可回答1,136题；分列行为与信用指标见answerability_metrics.csv。Direct信用为当前baseline观测。独立十次试答信用和成功率仍待补充，保留空值。",
        "原协议分别报告人工可回答性与可回答题独立试答成功率，合成二元知识缺失GT保持未定义。当前检查点未计算方法效应。",
        "800登记条件的覆盖与待评分缺项在condition_coverage.csv；新增生成数据应在独立后续评分检查点接续。", ""]
    with (output / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(report))
    summary["output_sha256"] = {path.name: file_hash(path) for path in output.iterdir() if path.is_file()}
    write_json(output / "summary.json", summary)
    write_json(output / "validation.json", {"actual_scored_rows": len(scored), "unique_formal_task_keys": len({r["key"] for r in scored}),
        "condition_full_sample_checks": len(groups), "rows_per_condition": 3501, "human_answerable_per_condition": 2365,
        "human_unanswerable_per_condition": 1136, "finite_official_credit_rows": sum(r["vqa_credit"] is not None for r in scored),
        "original_identity_key_question_answer_annotation_bindings": len(scored), "pending_rows": len(pending),
        "formal_expected_minus_scored": 2800800 - len(scored), "registered_condition_coverage_rows": len(coverage),
        "new_API_calls": 0, "GPU_initialized": False})
    print(json.dumps({key: value for key, value in summary.items() if key not in {"derived_source_sha256", "output_sha256"}}, ensure_ascii=False, indent=2))


def verify_output(output):
    summary = json.loads((output / "summary.json").read_text())
    for name, digest in summary["output_sha256"].items():
        if file_hash(output / name) != digest:
            raise ValueError("Saved VizWiz output digest differs: " + name)
    manifest_path = Path(summary["asset_manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    assets = manifest_path.parent
    input_paths = {"asset_manifest": manifest_path, "official_reference": assets / "official_reference.jsonl",
        "final_census_labels": Path(manifest["final_census_labels"]["original_path"]),
        "official_vqa_normalizer": ROOT / "src/kdm/models/official_vqa_normalizer.py",
        "vqa_scoring_function_source": ROOT / "src/kdm/scoring.py", "all_manifest": ROOT / "data/current/all.jsonl"}
    for name, digest in summary["input_sha256"].items():
        if file_hash(input_paths[name]) != digest:
            raise ValueError("Saved VizWiz input digest differs: " + name)
    official = {row["sample_id"]: row for row in read_jsonl(assets / "official_reference.jsonl") if row["split"] == "eval"}
    samples = {row["id"]: row for row in read_jsonl(ROOT / "data/current/all.jsonl") if row["dataset"] == "vizwiz" and row["split"] == "eval"}
    evaluator = VQAEval(None, None)
    normalizer = lambda text: evaluator.processDigitArticle(evaluator.processPunctuation(text))
    groups, keys, originals = defaultdict(list), set(), set()
    with gzip.open(output / "score_rows.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sample, reference = samples[row["sample_id"]], official[row["sample_id"]]
            task = {"sample": sample, **{field: row[field] for field in COND[3:]}}
            if row["key"] in keys or row["original_key"] in originals or row["key"] != task_id(row["model"], task):
                raise ValueError("Scored VizWiz source/task key is duplicated or differs")
            keys.add(row["key"])
            originals.add(row["original_key"])
            if (row["question"] != sample["question"] or row["official_answers"] != reference["official_answers"]
                    or row["annotated_answerable"] != reference["annotated_answerable"]
                    or row["qa_key"] != qah(row["question"], row["answer"])
                    or row["knowledge_deficit_GT"] is not None or row["independent_reference_complete"] is not False
                    or row["seed"] != stable_seed(row["sample_id"], row["model"], 0)
                    or row["config_sha256"] != stable_hash(asdict(DecodeConfig()))):
                raise ValueError("Scored VizWiz official/source/registered context differs")
            if row["label"] in {"abstain", "invalid"}:
                if row["answer_text"]:
                    raise ValueError("Abstention/invalid answer span differs")
                credit = 0.0
            else:
                if not row["answer_text"] or row["answer_text"] not in row["answer"]:
                    raise ValueError("Scored final answer span is unresolved")
                credit = float(vqa_score(row["answer_text"], row["official_answers"], normalizer))
            if not math.isfinite(row["vqa_credit"]) or not math.isclose(row["vqa_credit"], credit, abs_tol=1e-12):
                raise ValueError("Frozen official VQA credit recomputation differs")
            groups[row["model"]].append(row)
    if len(keys) != 38511 or set(groups) != set(MODELS):
        raise ValueError("Full 11-condition VizWiz scored coverage differs")
    with (output / "condition_metrics.csv").open(encoding="utf-8", newline="") as stream:
        saved = {row["model"]: row for row in csv.DictReader(stream)}
    for model, rows in groups.items():
        if len(rows) != 3501 or {row["sample_id"] for row in rows} != set(samples):
            raise ValueError("Full registered VizWiz eval sample denominator differs")
        recomputed = metrics({field: rows[0][field] for field in COND}, rows)
        for field, value in recomputed.items():
            observed = saved[model][field]
            if isinstance(value, bool):
                equal = observed == str(value)
            elif isinstance(value, (int, float)):
                equal = math.isclose(float(observed), value, abs_tol=1e-12)
            else:
                equal = observed == ("" if value is None else str(value))
            if not equal:
                raise ValueError("Saved VizWiz condition metric differs: " + model + "/" + field)
    with (output / "condition_coverage.csv").open(encoding="utf-8", newline="") as stream:
        coverage = list(csv.DictReader(stream))
    if len(coverage) != 800 or sum(int(row["missing_scored_rows"]) for row in coverage) != 2762289:
        raise ValueError("Full registered VizWiz condition coverage differs")
    result = {"verified_output_digests": len(summary["output_sha256"]), "verified_input_digests": len(input_paths),
        "recomputed_official_vqa_credit_rows": len(keys), "unique_formal_keys": len(keys),
        "unique_original_source_keys": len(originals), "complete_eval_conditions": len(groups),
        "registered_eval_rows_per_condition": 3501, "official_answerable_quota": 2365,
        "official_unanswerable_quota": 1136, "registered_conditions": len(coverage),
        "missing_scored_formal_rows": 2762289, "knowledge_deficit_GT": None,
        "figure_sha256": {Path(path).name: file_hash(Path(path)) for path in summary["figures"]},
        "verifier_sha256": file_hash(Path(__file__)), "new_API_calls": 0, "GPU_initialized": False}
    verification_path = output / "verification.json"
    if verification_path.exists():
        if json.loads(verification_path.read_text()) != result:
            raise ValueError("Existing verification receipt differs; preserve it and use a new checkpoint")
    else:
        write_json(verification_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-output", action="store_true")
    args = parser.parse_args()
    if args.verify_output:
        verify_output((ROOT / args.output).resolve())
    else:
        if args.assets is None:
            parser.error("--assets is required for scoring")
        score((ROOT / args.assets).resolve(), (ROOT / args.output).resolve())


if __name__ == "__main__":
    main()
