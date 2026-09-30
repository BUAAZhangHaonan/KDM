#!/usr/bin/env python3
"""Analyze immutable remaining-model scores under the frozen Food protocol."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gzip
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, read_jsonl
from kdm.pipeline import experiment_tasks, task_id
from workflows.main_results import analysis as shared
from workflows.main_results import preserve_abstention_control as control
from workflows.supplemental.remaining11.generate import MODELS

COND = ("model", "dataset", "split", "method", "kind", "marker",
        "reference_marker", "guided", "reference_guided", "replicate")
BOOT = shared.BOOT
SEED = shared.SEED
EXPECTED_CELLS = 832
EXPECTED_ROWS = 2016768
DENOMINATOR = 2424


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_jsonl(path, values):
    options = {"compresslevel": 1} if path.suffix == ".gz" else {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "xt", encoding="utf-8", newline="\n", **options) as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def write_csv(path, values):
    fields = list(dict.fromkeys(key for value in values for key in value))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def key_of(row):
    return tuple(row[name] for name in COND)


def context(key):
    return {**dict(zip(COND, key)), "main_marker": key[5]}


def direct_key(row):
    return (row["model"], row["marker"], row["guided"], row["replicate"], row["sample_id"])


def ratio(numerator, denominator):
    return control.fmt_ratio(numerator, denominator)


def load_registry():
    samples = {row["id"]: row for row in read_jsonl(ROOT / "data/current/all.jsonl")
               if row["dataset"] == "food101"}
    evaluation = {sid: row for sid, row in samples.items() if row["split"] == "eval"}
    classes = sorted({row["class"] for row in samples.values()})
    if len(samples) != 4848 or len(evaluation) != DENOMINATOR or len(classes) != 101:
        raise ValueError("Registered Food-101 manifest count differs")
    for split in ("dev", "eval"):
        if Counter(row["class"] for row in samples.values() if row["split"] == split) != Counter({name: 24 for name in classes}):
            raise ValueError("Registered Food-101 class quota differs")
    plan = json.loads((ROOT / "configs/kdm/method_plan.json").read_text())
    first = next(iter(evaluation.values()))
    expected = {}
    for model in MODELS:
        for task in experiment_tasks((first,), methods=plan[model]["food101"]):
            row = {"model": model, "dataset": "food101", "split": "eval", **task}
            key = key_of(row)
            if key in expected:
                raise ValueError("Duplicate registered condition")
            expected[key] = {name: task[name] for name in COND[3:]}
    if len(expected) != EXPECTED_CELLS or len(expected) * len(evaluation) != EXPECTED_ROWS:
        raise ValueError("Registered supplementary formal matrix differs")
    return samples, evaluation, classes, expected


def load_reference(path, samples):
    references = {}
    counts = defaultdict(Counter)
    for row in read_jsonl(path):
        key = row["model"], row["sample_id"]
        sample = samples.get(row["sample_id"])
        if row["model"] not in MODELS or sample is None or row["dataset"] != "food101":
            raise ValueError("Unregistered supplementary reference identity")
        if key in references or row["target_class"] != sample["class"] or row["split"] != sample["split"]:
            raise ValueError("Duplicate or mismatched supplementary reference")
        complete = row["reference_complete"]
        gt = row["reference_G"]
        if complete:
            attempts = row["attempts"]
            if row["gold_rank"] is None or row["correct_count"] is None or row["attempt_count"] != 10:
                raise ValueError("Complete reference lacks rank or independent coverage")
            if {a["replicate"] for a in attempts} != set(range(10)) or any(a["canonical_name_in_primary_score"] not in (0, 1) for a in attempts):
                raise ValueError("Complete reference has unresolved attempts")
            if row["correct_count"] != sum(a["canonical_name_in_primary_score"] == 1 for a in attempts):
                raise ValueError("Reference correct count differs from observed attempts")
            if type(gt) is not bool or gt != (row["gold_rank"] > 1 and row["correct_count"] == 0):
                raise ValueError("Reference differs from uniform registered rule")
        elif gt is not None:
            raise ValueError("Incomplete reference publishes a Boolean GT")
        references[key] = {"model": row["model"], "dataset": "food101", "sample_id": row["sample_id"],
                           "split": row["split"], "target_class": row["target_class"],
                           "gold_rank": row["gold_rank"], "correct_count": row["correct_count"],
                           "attempt_count": row["attempt_count"], "reference_G": gt,
                           "reference_complete": complete,
                           "missing_replicates": row["missing_replicates"],
                           "rank_source_identity": row["rank_source"].get("source_identity") if row["rank_source"] else None,
                           "rank_source_path": row["rank_source"].get("source_path") if row["rank_source"] else None,
                           "attempt_source_identities": sorted({a["source_identity"] for a in row["attempts"]}),
                           "historical_accepted_reference": row.get("historical_accepted_reference")}
        c = counts[(row["model"], row["split"])]
        c["rows"] += 1
        c["reference_complete"] += complete
        c["reference_positive"] += gt is True
        c["rank_missing"] += row["gold_rank"] is None
        c["attempts_missing"] += row["attempt_count"] != 10
        c["correct_count_unresolved"] += row["attempt_count"] == 10 and row["correct_count"] is None
    if set(references) != {(model, sid) for model in MODELS for sid in samples}:
        raise ValueError("Reference manifest must retain all 53,328 registered keys")
    return references, [{"model": model, "dataset": "food101", "split": split,
                         **dict(c), "reference_missing": c["rows"] - c["reference_complete"]}
                        for (model, split), c in sorted(counts.items())]


def load_scores(path, samples, expected):
    groups = defaultdict(dict)
    seen = set()
    stage_counts = Counter()
    fields = (*COND, "main_marker", "key", "sample_id", "target_class", "qa_key",
              "question", "answer", "canonical_name_in_primary_score", "literal_extracted_name_score",
              "literal_extracted_names", "abstain", "seed", "main_prompt_sha256",
              "reference_prompt_sha256", "config_sha256", "terminated", "source_path", "source_line",
              "source_identity", "original_key", "original_source_identity", "raw_line_sha256",
              "source_container", "source_member", "source_cohort", "score_reason", "behavior_source",
              "annotation_model", "annotation_effort", "annotation_call_id")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identity = row["model"], row["stage"], row["key"]
            if row["stage"] not in {"formal", "independent"}:
                raise ValueError("Unregistered score stage")
            if identity in seen:
                raise ValueError("Duplicate supplementary score task")
            seen.add(identity)
            stage_counts[row["stage"]] += 1
            sample = samples.get(row["sample_id"])
            if sample is None or row["model"] not in MODELS or row["dataset"] != "food101" or row["split"] != sample["split"] or row["target_class"] != sample["class"]:
                raise ValueError("Score identity differs from original manifest")
            if row["stage"] != "formal":
                continue
            key = key_of(row)
            if key not in expected:
                raise ValueError("Unregistered formal condition")
            task = {**expected[key], "sample": sample}
            if task_id(row["model"], task) != row["key"] or row["sample_id"] in groups[key]:
                raise ValueError("Formal task key or sample uniqueness differs")
            groups[key][row["sample_id"]] = {name: row.get(name) for name in fields}
    return groups, stage_counts


def group_status(key, rows, evaluation, classes):
    missing = sorted(set(evaluation) - set(rows))
    quota = Counter(r["target_class"] for r in rows.values())
    correctness = sum(r["canonical_name_in_primary_score"] not in (0, 1) for r in rows.values())
    abstention = sum(type(r["abstain"]) is not bool for r in rows.values())
    literal = sum(r["literal_extracted_name_score"] not in (0, 1) for r in rows.values())
    complete = not missing and quota == Counter({name: 24 for name in classes}) and not correctness and not abstention
    return {**context(key), "expected_n": DENOMINATOR, "observed_n": len(rows), "missing_sample_n": len(missing),
            "correctness_unresolved_n": correctness, "abstain_unresolved_n": abstention,
            "literal_unresolved_n": literal, "class_quota_complete": quota == Counter({name: 24 for name in classes}),
            "analysis_eligible": complete}, missing


def condition_metrics(key, rows, references, direct=None):
    values = list(rows.values())
    n = len(values)
    correct = sum(r["canonical_name_in_primary_score"] == 1 for r in values)
    answered = [r for r in values if not r["abstain"]]
    literal_unknown = sum(r["literal_extracted_name_score"] is None for r in values)
    known = [r for r in values if references[(r["model"], r["sample_id"])]["reference_complete"]]
    tp = sum(r["abstain"] and references[(r["model"], r["sample_id"])]["reference_G"] for r in known)
    fp = sum(r["abstain"] and not references[(r["model"], r["sample_id"])]["reference_G"] for r in known)
    fn = sum(not r["abstain"] and references[(r["model"], r["sample_id"])]["reference_G"] for r in known)
    complete = len(known) == n
    fixed = [r for r in values if direct is not None and direct[r["sample_id"]]["abstain"]
             and references[(r["model"], r["sample_id"])]["reference_G"] is True] if complete else []
    retained = sum(r["abstain"] for r in fixed)
    return {**context(key), "n": n, "canonical_correct": correct, "canonical_accuracy": correct / n,
            "literal_correct": sum(r["literal_extracted_name_score"] == 1 for r in values),
            "literal_unknown": literal_unknown,
            "literal_accuracy": sum(r["literal_extracted_name_score"] == 1 for r in values) / n if not literal_unknown else None,
            "abstain_n": n - len(answered), "answered_n": len(answered), "coverage": len(answered) / n,
            "selective_correct_n": sum(r["canonical_name_in_primary_score"] == 1 for r in answered),
            "selective_accuracy": ratio(sum(r["canonical_name_in_primary_score"] == 1 for r in answered), len(answered)),
            "reference_complete_n": len(known), "reference_missing_n": n - len(known),
            "reference_condition_complete": complete, "reference_positive_known_n": tp + fn,
            "abstention_tp_known_n": tp, "abstention_fp_known_n": fp, "abstention_fn_known_n": fn,
            "abstention_precision": ratio(tp, tp + fp) if complete else None,
            "abstention_recall": ratio(tp, tp + fn) if complete else None,
            "fixed_direct_reasonable_abstention_n": len(fixed) if complete and direct is not None else None,
            "fixed_direct_reasonable_abstention_retained_n": retained if complete and direct is not None else None,
            "fixed_direct_reasonable_abstention_retention": ratio(retained, len(fixed)) if complete and direct is not None else None,
            "source_identities": ";".join(sorted({r["source_identity"] for r in values})),
            "source_cohorts": ";".join(sorted({r["source_cohort"] for r in values}))}


def assert_pair(a, b):
    if set(a) != set(b) or len(a) != DENOMINATOR:
        raise ValueError("Comparison lacks the complete paired sample set")
    for sid in a:
        x, y = a[sid], b[sid]
        if (not x["main_prompt_sha256"] or x["main_prompt_sha256"] != y["main_prompt_sha256"]
                or x["seed"] is None or x["seed"] != y["seed"] or x["target_class"] != y["target_class"]):
            raise ValueError("Paired seed, main prompt, or target identity differs")


def compare(label, key, a, b, references, classes, draws, direct):
    assert_pair(a, b)
    by_class = defaultdict(list)
    retention = defaultdict(lambda: [0, 0])
    recall = defaultdict(lambda: [0, 0])
    fixed_difference = defaultdict(lambda: [0, 0])
    correction = {"all_numerator": 0, "all_denominator": 0, "specific_numerator": 0, "specific_denominator": 0}
    transitions = Counter()
    ref_complete = all(references[(key[0], sid)]["reference_complete"] for sid in a)
    for sid, x in a.items():
        y, d = b[sid], direct[sid]
        cls = x["target_class"]
        cx, cy, cd = (r["canonical_name_in_primary_score"] for r in (x, y, d))
        by_class[cls].append(cx - cy)
        transitions["baseline_correct_retained"] += cy == 1 and cx == 1
        transitions["baseline_correct_lost"] += cy == 1 and cx == 0
        transitions["new_correct_vs_baseline"] += cy == 0 and cx == 1
        transitions["baseline_wrong_n"] += cy == 0
        transitions["baseline_specific_wrong_n"] += cy == 0 and not y["abstain"]
        transitions["new_correct_vs_specific_wrong_baseline_n"] += cy == 0 and not y["abstain"] and cx == 1
        if cy == 1 and cd == 0:
            correction["all_denominator"] += 1
            correction["all_numerator"] += cx == 1
            if not d["abstain"]:
                correction["specific_denominator"] += 1
                correction["specific_numerator"] += cx == 1
        if ref_complete and references[(key[0], sid)]["reference_G"]:
            recall[cls][0] += int(x["abstain"]) - int(y["abstain"])
            recall[cls][1] += 1
            if y["abstain"]:
                retention[cls][0] += int(x["abstain"])
                retention[cls][1] += 1
            if d["abstain"]:
                fixed_difference[cls][0] += int(x["abstain"]) - int(y["abstain"])
                fixed_difference[cls][1] += 1
    acc = shared.bootstrap_class_ci(by_class, draws)
    result = {"comparison": label, "context": context(key), "method_a": next(iter(a.values()))["method"],
              "method_b": next(iter(b.values()))["method"], "n_paired": DENOMINATOR,
              "target_class_clusters": 101, "accuracy_difference_a_minus_b": acc,
              "accuracy_noninferiority_lower95_gt_minus_0_01": acc["ci95"][0] > -0.01,
              "reference_condition_complete": ref_complete, "transitions": dict(transitions),
              "correction_gain_all_inputs": transitions["new_correct_vs_baseline"] / DENOMINATOR,
              "correction_gain_on_baseline_wrong": ratio(transitions["new_correct_vs_baseline"], transitions["baseline_wrong_n"]),
              "correction_gain_on_specific_wrong_baseline": ratio(transitions["new_correct_vs_specific_wrong_baseline_n"], transitions["baseline_specific_wrong_n"]),
              "correction_retention_all": ratio(correction["all_numerator"], correction["all_denominator"]),
              "correction_retention_specific_answer": ratio(correction["specific_numerator"], correction["specific_denominator"]),
              "correction_counts": correction, "prompt_sha_mismatch_n": 0, "seed_mismatch_n": 0}
    for name, values in (("abstention_recall_difference_a_minus_b", recall),
                         ("baseline_reasonable_abstentions_retained_by_a", retention),
                         ("fixed_direct_reasonable_abstention_retention_difference_a_minus_b", fixed_difference)):
        result[name] = shared.bootstrap_ratio_diff_by_cluster(values, draws, classes) if ref_complete else None
    return result


def copy_control(key, treatment, direct):
    """Preserve the registered direct-abstained branch and provenance."""
    assert_pair(treatment, direct)
    selected, choices = {}, []
    for sid, row in treatment.items():
        original = direct[sid]
        source = original if original["abstain"] else row
        selected[sid] = source
        choices.append({**context(key), "sample_id": sid, "treatment_key": row["key"],
                        "direct_key": original["key"], "selected_key": source["key"],
                        "selected_source_method": source["method"], "selected_source_identity": source["source_identity"],
                        "selected_source_path": source["source_path"], "seed": source["seed"],
                        "main_prompt_sha256": source["main_prompt_sha256"],
                        "selection_reason": "direct_abstained" if original["abstain"] else "direct_answered",
                        "direct_abstain": original["abstain"], "selected_abstain": source["abstain"]})
    return selected, choices


def registered_pair_coverage(expected, eligible, copies):
    missing, controls = [], []
    expected_pair_count = 0
    for key in expected:
        d = context(key)
        if d["method"] == "direct":
            continue
        direct = {**d, "method": "direct", "kind": "main", "reference_marker": d["marker"],
                  "reference_guided": d["guided"]}
        dk = tuple(direct[name] for name in COND)
        expected_pair_count += 1
        if key not in eligible or dk not in eligible:
            missing.append({"comparison": "same_main_prompt_direct", "context": d,
                            "base_context": context(dk), "intervention_complete": key in eligible,
                            "comparator_complete": dk in eligible,
                            "reason": "registered_complete_conditions_not_available"})
        if d["method"] in {"vcd", "m3id"} and d["kind"] in {"main", "reference_instruction_removed"}:
            controls.append({**d, "expected_selection_rows": DENOMINATOR,
                             "treatment_complete": key in eligible, "direct_complete": dk in eligible,
                             "copy_control_complete": key in copies,
                             "missing_selection_rows": 0 if key in copies else DENOMINATOR})
        base = {"instruction_vcd": "vcd", "instruction_m3id": "m3id"}.get(d["method"])
        if base:
            b = {**d, "method": base, "kind": "main", "reference_guided": d["guided"]}
            bk = tuple(b[name] for name in COND)
            for label, ready in (("instruction_vs_base_method", bk in eligible),
                                 ("instruction_vs_copy_control", bk in copies)):
                expected_pair_count += 1
                if key not in eligible or dk not in eligible or not ready:
                    missing.append({"comparison": label, "context": d, "base_context": context(bk),
                                    "intervention_complete": key in eligible,
                                    "direct_complete": dk in eligible, "comparator_complete": ready,
                                    "reason": "registered_complete_conditions_not_available"})
    if len(controls) != 440 or expected_pair_count != 964:
        raise ValueError("Registered remaining-model comparison/control matrix differs")
    return missing, controls, expected_pair_count


def figures(out, metrics, comparisons):
    directory = out / "figures"
    directory.mkdir()
    paths = []
    baseline = [r for r in metrics if r["method"] == "direct" and r["kind"] == "main" and r["marker"] == "UNKNOWN"]
    if baseline:
        labels = [r["model"] for r in baseline]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.85), 4.8))
        ax.bar(x - 0.18, [r["canonical_accuracy"] for r in baseline], 0.36, label="Canonical accuracy")
        ax.bar(x + 0.18, [r["abstain_n"] / DENOMINATOR for r in baseline], 0.36, label="Abstention rate")
        ax.set_xticks(x, labels, rotation=40, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of all 2,424 eval samples")
        ax.set_title("Remaining-model Food-101: Direct / UNKNOWN")
        ax.legend()
        fig.tight_layout()
        for extension in ("png", "pdf"):
            path = directory / ("direct_unknown." + extension)
            fig.savefig(path, dpi=180)
            paths.append(str(path))
        plt.close(fig)
    pairs = [p for p in comparisons if p["comparison"] == "same_main_prompt_direct" and p["context"]["marker"] == "UNKNOWN" and p["context"]["reference_marker"] == "UNKNOWN"]
    if pairs:
        labels = [p["context"]["model"] + "/" + p["context"]["method"] + "/" + p["context"]["kind"] for p in pairs]
        values = [p["accuracy_difference_a_minus_b"]["estimate"] for p in pairs]
        low = [v - p["accuracy_difference_a_minus_b"]["ci95"][0] for v, p in zip(values, pairs)]
        high = [p["accuracy_difference_a_minus_b"]["ci95"][1] - v for v, p in zip(values, pairs)]
        fig, ax = plt.subplots(figsize=(9, max(4, len(pairs) * 0.35)))
        ax.errorbar(values, np.arange(len(pairs)), xerr=[low, high], fmt="o")
        ax.set_yticks(np.arange(len(pairs)), labels)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_xlabel("Canonical accuracy difference; 95% class-cluster paired bootstrap")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            path = directory / ("unknown_method_effects." + extension)
            fig.savefig(path, dpi=180)
            paths.append(str(path))
        plt.close(fig)
    return paths


def pct(value):
    return "未完成" if value is None else f"{value * 100:.2f}%"


def report_text(summary, metrics, ref_coverage, comparisons, appendix=False):
    lines = ["# 剩余11模型 Food-101 补充结果" if not appendix else "# 论文附录：剩余11模型补充数据", "",
             f"评分来源：`{summary['score_directory']}`。检查点：`{summary['output_directory']}`。",
             f"正式登记包含832个条件、2,016,768条样本记录；本检查点具有{summary['formal_observed_rows']:,}条正式评分，{summary['eligible_conditions']}个完整且主正确性及弃权已决的条件。", "",
             "Food-101每个完整条件固定使用eval 2,424张图像，101个类别各24张。当前主正确性采用canonical_name_in_primary_score；完整提取名称字面分数独立保存。",
             "参考GT固定为正确类别排名大于1且10次独立试答正确次数为0。参考覆盖不足的条件保留缺项，完整参考精确率、召回率与合理弃权保留率留空。", "",
             "| 模型 | 完整参考dev | 完整参考eval | eval参考缺项 |", "|---|---:|---:|---:|"]
    rc = {(r["model"], r["split"]): r for r in ref_coverage}
    for model in MODELS:
        lines.append(f"| {model} | {rc[model, 'dev']['reference_complete']} / 2424 | {rc[model, 'eval']['reference_complete']} / 2424 | {rc[model, 'eval']['reference_missing']} |")
    lines += ["", "完整Direct/UNKNOWN条件：", "", "| 模型 | 正确率 | 弃权率 | 回答覆盖率 | 参考完整数 |", "|---|---:|---:|---:|---:|"]
    for r in metrics:
        if r["method"] == "direct" and r["marker"] == "UNKNOWN":
            lines.append(f"| {r['model']} | {pct(r['canonical_accuracy'])} | {pct(r['abstain_n'] / DENOMINATOR)} | {pct(r['coverage'])} | {r['reference_complete_n']} / 2424 |")
    lines += ["", f"完整配对比较：{len(comparisons)}条。置信区间使用2,000次共享Food-101类别cluster重采样，随机种子20260929。每个比较匹配model、sample、main_marker、guided和replicate，并核对主提示哈希及seed。",
              f"完整复制控制：{summary['copy_control_conditions']} / {summary['copy_control_expected_conditions']}个条件。选择依据为Direct是否弃权，逐样本来源保存在`copy_control_selections.jsonl.gz`。",
              f"尚缺完整条件：{summary['ineligible_conditions']}个；精确条件字段、剩余样本数与未决标注数保存在`condition_coverage.csv`，缺失样本主键保存在`missing_keys.jsonl.gz`。", "",
              "当前完整条件支持描述性补充结果。方法效应仅从已完成的全条件配对计算；方法比较缺项记录于`missing_comparisons.jsonl`。",
              "逐样本合并表为`sample_results.jsonl.gz`，条件表为`condition_metrics.csv`，参考表为`reference_table.jsonl.gz`。图表使用上述完整条件生成，真实回答与来源案例保存在`cases.jsonl`。", ""]
    return "\n".join(lines)


def analyze(score_directory, output):
    score_directory = score_directory.resolve()
    summary_path = score_directory / "summary.json"
    score_path = score_directory / "score_rows.jsonl.gz"
    reference_path = score_directory / "reference_G.jsonl"
    input_summary = json.loads(summary_path.read_text())
    inputs = {"score_rows": file_hash(score_path), "reference_G": file_hash(reference_path), "score_summary": file_hash(summary_path),
              "input_manifest": file_hash(ROOT / "data/current/all.jsonl"), "method_plan": file_hash(ROOT / "configs/kdm/method_plan.json")}
    samples, evaluation, classes, expected = load_registry()
    references, ref_coverage = load_reference(reference_path, samples)
    groups, stage_counts = load_scores(score_path, samples, expected)
    if stage_counts["formal"] != input_summary["formal_rows"] or stage_counts["independent"] != input_summary["independent_rows"]:
        raise ValueError("Immutable score summary row counts differ")
    output.mkdir(parents=True, exist_ok=False)
    coverage, missing_keys, eligible, merged = [], [], {}, []
    for key, task_fields in sorted(expected.items(), key=lambda item: tuple(map(str, item[0]))):
        rows = groups.get(key, {})
        status, missing = group_status(key, rows, evaluation, classes)
        coverage.append(status)
        for sid in missing:
            task = {**task_fields, "sample": evaluation[sid]}
            missing_keys.append({**context(key), "stage": "formal", "sample_id": sid, "key": task_id(key[0], task)})
        if status["analysis_eligible"]:
            eligible[key] = rows
        for row in rows.values():
            reference = references[(row["model"], row["sample_id"])]
            merged.append({**row, "stage": "formal", "reference_G": reference["reference_G"],
                           "reference_complete": reference["reference_complete"], "reference_gold_rank": reference["gold_rank"],
                           "reference_correct_count": reference["correct_count"], "reference_attempt_count": reference["attempt_count"],
                           "condition_analysis_eligible": status["analysis_eligible"]})
    draws = np.random.RandomState(SEED).randint(0, 101, size=(BOOT, 101))
    direct_by_condition, direct_flat = {}, {}
    for key, rows in eligible.items():
        d = context(key)
        if d["method"] == "direct" and d["kind"] == "main" and d["marker"] == d["reference_marker"] and d["guided"] == d["reference_guided"]:
            direct_by_condition[(d["model"], d["marker"], d["guided"], d["replicate"])] = rows
            direct_flat.update({direct_key(row): row for row in rows.values()})
    metrics = [condition_metrics(key, rows, references, direct_by_condition.get((key[0], key[5], key[7], key[9])))
               for key, rows in eligible.items()]
    comparisons, missing_pairs, copies, selections, copy_metrics = [], [], {}, [], []
    for key, rows in eligible.items():
        d = context(key)
        if d["method"] == "direct":
            continue
        dk = (d["model"], d["marker"], d["guided"], d["replicate"])
        direct = direct_by_condition.get(dk)
        if direct is None:
            missing_pairs.append({"comparison": "same_main_prompt_direct", "context": d, "reason": "direct_comparator_incomplete_or_unresolved"})
            continue
        comparisons.append(compare("same_main_prompt_direct", key, rows, direct, references, classes, draws, direct))
        if d["method"] in {"vcd", "m3id"} and d["kind"] in {"main", "reference_instruction_removed"}:
            selected, choices = copy_control(key, rows, direct)
            copies[key] = selected
            selections.extend(choices)
            copy_metrics.append({**condition_metrics(key, selected, references, direct), "method": "preserve_direct_abstention_then_" + d["method"],
                                 "registered_treatment_method": d["method"], "selected_from_direct_n": sum(r["selection_reason"] == "direct_abstained" for r in choices)})
    control.CLASSES = classes
    full_gt_map = {(m, sid): {"gt": row["reference_G"], "target": row["target_class"], "cluster": row["target_class"]}
                   for (m, sid), row in references.items() if row["reference_complete"]}
    for key, rows in eligible.items():
        d = context(key)
        base = {"instruction_vcd": "vcd", "instruction_m3id": "m3id"}.get(d["method"])
        if base is None:
            continue
        bd = {**d, "method": base, "kind": "main", "reference_guided": d["guided"]}
        bk = tuple(bd[name] for name in COND)
        direct = direct_by_condition.get((d["model"], d["marker"], d["guided"], d["replicate"]))
        for name, other in (("instruction_vs_base_method", eligible.get(bk)), ("instruction_vs_copy_control", copies.get(bk))):
            if direct is None or other is None:
                missing_pairs.append({"comparison": name, "context": d, "base_context": context(bk), "reason": "registered_base_or_direct_incomplete_or_unresolved"})
                continue
            comparison = compare(name, key, rows, other, references, classes, draws, direct)
            if name == "instruction_vs_copy_control":
                comparison["method_b"] = "preserve_direct_abstention_then_" + base
            if name == "instruction_vs_copy_control" and comparison["reference_condition_complete"]:
                comparison["registered_control_metric_differences"] = control.bootstrap_differences(list(rows.values()), list(other.values()), full_gt_map, direct_flat, draws)
                comparison["instruction_metrics"] = control.count_metrics(list(rows.values()), full_gt_map, direct_flat)
                comparison["copy_control_metrics"] = control.count_metrics(list(other.values()), full_gt_map, direct_flat)
            comparisons.append(comparison)
    missing_pairs, copy_coverage, expected_pair_count = registered_pair_coverage(expected, eligible, copies)
    if len(comparisons) + len(missing_pairs) != expected_pair_count:
        raise ValueError("Registered paired comparison count does not conserve complete/missing cells")
    cases, categories = [], Counter()
    for row in merged:
        if not row["condition_analysis_eligible"]:
            continue
        category = "abstention" if row["abstain"] else ("correct_primary" if row["canonical_name_in_primary_score"] == 1 else "incorrect_primary")
        ck = row["model"], category
        if categories[ck] < 2:
            categories[ck] += 1
            cases.append({"case_category": category, "selection": "lexicographically_first_registered_complete_condition_then_source_order", **row})
    figures_paths = figures(output, metrics, comparisons)
    summary = {"schema": "kdm_remaining11_analysis_v1", "updated_utc": datetime.now(timezone.utc).isoformat(),
               "score_directory": str(score_directory), "output_directory": str(output),
               "formal_expected_rows": EXPECTED_ROWS, "formal_observed_rows": stage_counts["formal"],
               "formal_missing_rows": len(missing_keys), "formal_expected_conditions": EXPECTED_CELLS,
               "eligible_conditions": len(eligible), "ineligible_conditions": EXPECTED_CELLS - len(eligible),
               "formal_eligible_rows": len(eligible) * DENOMINATOR,
               "reference_expected_rows": len(references), "reference_complete_rows": sum(r["reference_complete"] for r in references.values()),
               "reference_missing_rows": sum(not r["reference_complete"] for r in references.values()),
               "paired_comparisons": len(comparisons), "missing_comparisons": len(missing_pairs),
               "expected_paired_comparisons": expected_pair_count,
               "copy_control_expected_conditions": len(copy_coverage),
               "copy_control_conditions": len(copies), "copy_control_selection_rows": len(selections),
               "case_rows": len(cases), "figures": figures_paths,
               "bootstrap": {"replicates": BOOT, "seed": SEED, "unit": "Food-101 target class", "clusters": 101, "paired": True},
               "input_sha256": inputs, "analysis_sha256": file_hash(Path(__file__)),
               "complete": len(eligible) == EXPECTED_CELLS and all(r["reference_complete"] for r in references.values())}
    write_csv(output / "condition_coverage.csv", coverage)
    write_csv(output / "condition_metrics.csv", metrics)
    write_csv(output / "reference_coverage.csv", ref_coverage)
    write_jsonl(output / "missing_keys.jsonl.gz", missing_keys)
    write_jsonl(output / "sample_results.jsonl.gz", merged)
    write_jsonl(output / "reference_table.jsonl.gz", references.values())
    write_jsonl(output / "paired_comparisons.jsonl", comparisons)
    write_jsonl(output / "missing_comparisons.jsonl", missing_pairs)
    write_jsonl(output / "copy_control_selections.jsonl.gz", selections)
    write_jsonl(output / "copy_control_metrics.jsonl", copy_metrics)
    write_jsonl(output / "copy_control_coverage.jsonl", copy_coverage)
    write_jsonl(output / "cases.jsonl", cases)
    for filename, appendix in (("report.md", False), ("appendix.md", True)):
        with (output / filename).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report_text(summary, metrics, ref_coverage, comparisons, appendix))
    summary["output_sha256"] = {path.name: file_hash(path) for path in output.iterdir() if path.is_file()}
    if inputs["score_summary"] != file_hash(summary_path):
        raise ValueError("Score input changed during analysis")
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def verify_checkpoint(output):
    """Verify real checkpoint artifacts and identity-pair calculations."""
    summary = json.loads((output / "summary.json").read_text())
    for filename, expected_sha in summary["output_sha256"].items():
        if file_hash(output / filename) != expected_sha:
            raise ValueError("Checkpoint output digest differs: " + filename)
    references = {(row["model"], row["sample_id"]): row
                  for row in shared.load_score(output / "reference_table.jsonl.gz")}
    complete = defaultdict(dict)
    for row in shared.load_score(output / "sample_results.jsonl.gz"):
        if row["condition_analysis_eligible"]:
            complete[key_of(row)][row["sample_id"]] = row
    if len(complete) != summary["eligible_conditions"]:
        raise ValueError("Checkpoint complete-condition count differs")
    classes = sorted({row["target_class"] for rows in complete.values() for row in rows.values()})
    if complete and len(classes) != 101:
        raise ValueError("Checkpoint target classes differ")
    draws = np.random.RandomState(SEED).randint(0, 101, size=(BOOT, 101))
    checks = []
    for key, rows in complete.items():
        if Counter(r["target_class"] for r in rows.values()) != Counter({name: 24 for name in classes}):
            raise ValueError("Checkpoint complete-condition quota differs")
        paired = compare("identity_pair_validation", key, rows, rows, references, classes, draws, rows)
        if paired["accuracy_difference_a_minus_b"]["estimate"] != 0 or paired["accuracy_difference_a_minus_b"]["ci95"] != [0.0, 0.0]:
            raise ValueError("Real response identity-pair bootstrap differs")
        selected, choices = copy_control(key, rows, rows)
        if set(selected) != set(rows) or any(selected[sid]["key"] != rows[sid]["key"] for sid in rows):
            raise ValueError("Real response identity selection differs")
        abstain_n = sum(r["abstain"] for r in rows.values())
        if sum(c["selection_reason"] == "direct_abstained" for c in choices) != abstain_n:
            raise ValueError("Registered copy branch does not conserve actual abstentions")
        checks.append({"context": context(key), "real_rows": len(rows), "actual_abstention_branch_n": abstain_n,
                       "accuracy_identity_difference": paired["accuracy_difference_a_minus_b"],
                       "selection_identity_preserved": True})
    write_json(output / "validation.json", {"schema": "kdm_remaining11_analysis_validation_v1",
               "actual_input": "immutable checkpoint sample/reference tables",
               "output_digest_checks": len(summary["output_sha256"]), "complete_condition_checks": len(checks),
               "real_identity_pair_rows": sum(c["real_rows"] for c in checks), "checks": checks,
               "validation_scope": "checkpoint artifacts, quotas, real identity-pair/bootstrap and selection invariants; complete intervention effects require available intervention cells"})
    print(json.dumps({"output_digest_checks": len(summary["output_sha256"]), "complete_condition_checks": len(checks),
                      "real_identity_pair_rows": sum(c["real_rows"] for c in checks)}, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--score-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-output", action="store_true")
    args = parser.parse_args()
    if args.verify_output:
        verify_checkpoint(args.output)
    else:
        if args.score_directory is None:
            parser.error("--score-directory is required for analysis")
        analyze(args.score_directory, args.output)


if __name__ == "__main__":
    main()
