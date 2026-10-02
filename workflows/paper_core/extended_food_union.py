"""Build a source-bound Food-only union without changing accepted score objects."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kdm.decoding import DecodeConfig
from kdm.io import file_hash, stable_hash, stable_seed, within
from kdm.pipeline import experiment_tasks, task_id
from kdm.prompts import task_prompt
from workflows.main_results import analysis as shared
from workflows.supplemental.remaining11.analysis import COND, compare, condition_metrics, context, group_status, key_of
from workflows.supplemental.remaining4.score_received import finite

BASE = ROOT / "outputs/paper_core_20261002_dev_viz"
OUT = BASE / "extended_food_union_cpu_20261002_2230"
PACKAGE = BASE / "review_package_staging_20261002_1700/package_v6"
REFERENCE = ROOT / "outputs/supplemental/remaining4/reference_full_20261001_1600/checkpoints/v2_reviewed_role_reference"
MODELS = ("internvl35_8b", "onevision", "phi35", "qwen3vl")
FIELDS = ("method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")
SOURCES = (
    ("old_food177333", ROOT / "outputs/supplemental/remaining4/received292541_20261002_0155_actual1000_foodGT",
     "38731cfc8e35e5e6ae7484de0d1e4185c81d6fadc6494d612f7926042f760efb", 292541, 177333),
    ("new_food53671", ROOT / "outputs/supplemental/remaining4/food53671_cpu_20261002_2140/reference/v1_food476_accepted_v2_join",
     "efa86be81d9caeb91cf7a4cd64bbcc39b02619cc60ad8d3ea7e72adb553f84d9", 53671, 53671),
)
CONFLICT = ROOT / "outputs/supplemental/remaining4/luna_content_batches_20261002_1345/score_owned1023_actual500_root48_20261002_1910_v2/unapplied_historical_shared_QA_conflicts.jsonl"


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    require(not path.exists(), "A finite Food union output must be exclusive")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, records):
    fields = list(dict.fromkeys(field for record in records for field in record))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def effective_holds(row):
    """Only current effective fields can block the accepted scientific decision."""
    flags = []
    for location, value in (("row", row), ("decision_source", row.get("decision_source"))):
        if not isinstance(value, dict):
            continue
        for field in ("needs_root", "root_review_required", "held", "on_hold", "label_pending"):
            if value.get(field) is True:
                flags.append(location + "." + field)
        for field in ("annotation_complete", "annotation_accepted", "eligible", "authority_eligible"):
            if value.get(field) is False:
                flags.append(location + "." + field + "=false")
    return flags


def historical_holds(value, historical=False):
    """Count explicit previous/variant flags without promoting them to current flags."""
    count = 0
    if isinstance(value, dict):
        for key, child in value.items():
            if historical and key in {"needs_root", "root_review_required", "held", "on_hold"} and child is True:
                count += 1
            else:
                previous = historical or key.startswith("previous_") or key in {
                    "source_variants", "review_variants", "source_luna_decision", "history", "old_decision",
                    "model_declared_original", "previous_actual_annotation"}
                count += historical_holds(child, previous)
    elif isinstance(value, list):
        count += sum(historical_holds(child, historical) for child in value)
    return count


def identity_proof(row, cache):
    binding = str(row.get("source_identity_path") or row["source_path"])
    if binding in cache:
        require(cache[binding]["identity"] == row["source_identity"] and cache[binding]["model"] == row["model"],
                "An explicit source identity path is reused by a different identity")
        return cache[binding]
    raw_path = within(ROOT, row["source_path"])
    sidecar = (within(ROOT, row["source_identity_path"]) if row.get("source_identity_path")
               else raw_path.with_suffix(".identity.json"))
    key = binding
    if key not in cache:
        proof = {"path": str(sidecar), "identity": row["source_identity"], "model": row["model"], "passed": False}
        if not sidecar.is_file():
            proof["hold_reason"] = "explicit_identity_sidecar_not_available"
        else:
            payload = load(sidecar)
            definition = payload["definition"]
            require(payload["identity"] == stable_hash(definition) == row["source_identity"]
                    and definition["model"] == row["model"], "Original identity sidecar content does not bind this score")
            registered = load(ROOT / ("configs/runtime/" + row["model"] + ".json"))
            if "registered_backend" in definition:
                backend = definition["registered_backend"]
                basis = "registered_backend"
            elif "backend" in definition:
                backend = definition["backend"]
                basis = "original_backend_immutable_components"
            else:
                backend = None
                basis = "unknown"
            if backend is None:
                proof["hold_reason"] = "original_identity_missing_explicit_backend"
            else:
                immutable_fields = ("key", "hf_model_id", "factory", "dtype", "thinking_mode",
                                    "model_config_sha256", "adapter_source_sha256", "input_limits")
                mismatch = [field for field in immutable_fields if backend.get(field) != registered.get(field)]
                for field in ("kwargs", "processor"):
                    ignored = {"model_path", "path"}
                    left = {k: v for k, v in backend.get(field, {}).items() if k not in ignored}
                    right = {k: v for k, v in registered.get(field, {}).items() if k not in ignored}
                    if left != right:
                        mismatch.append(field + "_immutable_values")
                weight_fields = ("filename", "size_bytes", "hub_revision", "hub_recorded_sha256")
                weights = lambda spec: [{k: item.get(k) for k in weight_fields} for item in spec.get("weights", [])]
                if weights(backend) != weights(registered):
                    mismatch.append("registered_weight_fingerprints")
                proof.update(sha256=file_hash(sidecar), binding_basis=basis,
                    original_definition_stage=definition.get("stage"), original_backend=backend,
                    immutable_component_mismatches=mismatch,
                    passed=not mismatch, weights_rehashed=False)
                if mismatch:
                    proof["hold_reason"] = "original_runtime_immutable_components_differ"
        cache[key] = proof
    require(cache[key]["identity"] == row["source_identity"] and cache[key]["model"] == row["model"],
            "An explicit source identity path is reused by a different identity")
    return cache[key]


def score_lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            require(line.endswith("\n"), "An accepted score has an incomplete final line")
            yield line_no, json.loads(line), line


def references():
    path = REFERENCE / "reference_G.jsonl"
    receipt = load(REFERENCE / "receipt.json")
    require(receipt["passed"] and receipt["reference_complete"] == 19392
            and receipt["reference_pending"] == 0 and receipt["outputs"][path.name] ==
            "d3feac8fef2ade5967c0ff17762e15a17aac92e244f980c01d6ed360ff8d36a0",
            "The Food reference is not the accepted complete reviewed-role v2 source")
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["model"], row["sample_id"]
            require(key not in result and row["reference_complete"] is True
                    and type(row["reference_G"]) is bool and row["attempt_count"] == 10
                    and row["reference_G"] == (row["gold_rank"] > 1 and row["correct_count"] == 0),
                    "A complete reference key or registered formula differs")
            result[key] = {k: row[k] for k in ("model", "sample_id", "split", "target_class",
                                             "gold_rank", "correct_count", "reference_G", "reference_complete")}
    require(len(result) == 19392, "Food reference count differs")
    return result, receipt["outputs"][path.name]


def audit(output, analyze=False):
    require(not output.exists(), "The Food cohort audit must have a new exclusive directory")
    all_samples = load_samples()
    eval_samples = {sid: s for sid, s in all_samples.items() if s["split"] == "eval"}
    method_plan = load(ROOT / "configs/kdm/method_plan.json")
    first = next(iter(eval_samples.values()))
    expected = {}
    for model in MODELS:
        for task in experiment_tasks((first,), methods=method_plan[model]["food101"]):
            if task["method"] in {"vcd", "m3id"}:
                continue
            cond = key_of({"model": model, "dataset": "food101", "split": "eval", **task})
            expected[cond] = {field: task[field] for field in FIELDS}
    require(len(expected) == 160, "Current retained registered Food condition count differs")
    refs, ref_sha = references()
    scoring_sha = file_hash(ROOT / "docs/SCORING.md")
    require(scoring_sha == file_hash(PACKAGE / "registered_current/docs/SCORING.md"),
            "Current and accepted-package scientific scoring protocols differ")
    with CONFLICT.open(encoding="utf-8") as stream:
        conflicts = [json.loads(line) for line in stream]
    require(len(conflicts) == 1 and conflicts[0]["sample_id"].startswith("vizwiz:"),
            "The bounded v6 historical conflict source changed")
    conflict_keys = {record["key"] for record in conflicts}
    conflict_qas = {record["qa_key"] for record in conflicts}
    seen, per_source, groups, findings, examples = {}, [], defaultdict(dict), [], {}
    identity_cache, effective_pending = {}, []
    for name, directory, expected_sha, expected_rows, expected_food in SOURCES:
        path = directory / "score_rows.jsonl.gz"
        receipt_path = directory / "receipt.json"
        receipt = load(receipt_path)
        require(receipt["passed"] and receipt["rows"] == expected_rows
                and receipt["outputs"][path.name] == expected_sha and file_hash(path) == expected_sha,
                "A finite accepted source differs from its actual passed receipt: " + name)
        counts, values, identities, source_paths = Counter(), defaultdict(Counter), set(), set()
        for number, row, original_line in score_lines(path):
            counts["all_rows"] += 1
            if row["dataset"] != "food101":
                continue
            counts["food_rows"] += 1
            model = row["model"]
            sample = all_samples.get(row["sample_id"])
            require(model in MODELS and sample is not None and row["split"] == sample["split"] == "eval"
                    and row["target_class"] == sample["class"] and row["question"] == sample["question"]
                    and row["stage"] == "formal" and finite(row),
                    "An accepted Food source has an unregistered input, roster, stage or finite value")
            cond = key_of(row)
            require(cond in expected and row["key"] == task_id(model, {"sample": sample, **expected[cond]})
                    and row["seed"] == stable_seed(sample["id"], model, row["replicate"])
                    and row["main_marker"] == row["marker"], "Food condition, task key, marker or seed differs")
            require(row["key"] not in seen and row["sample_id"] not in groups[cond],
                    "The old and new Food sources have repeated task/sample keys")
            config = {**asdict(DecodeConfig()), "method": row["method"]}
            if row["method"] == "instruction_m3id":
                require(type(row["config"].get("m3id_offset")) is int and row["config"]["m3id_offset"] > 0,
                        "Original instruction-M3ID offset is missing")
                config["m3id_offset"] = row["config"]["m3id_offset"]
            require(row["config"] == config and isinstance(row["source_identity"], str) and row["source_identity"],
                    "Food source identity or actual decode parameters differ from the registration")
            reference = refs[(model, row["sample_id"])]
            for field in ("reference_G", "reference_complete", "gold_rank", "correct_count"):
                counts[field + "_mismatch"] += row.get(field) != reference[field]
            counts["reference_source_mismatch"] += row.get("reference_join_source_sha256") != ref_sha
            for field in ("canonical_name_in_primary_score", "literal_extracted_name_score"):
                counts[field + "_pending"] += row.get(field) is None
                counts[field + "_invalid"] += row.get(field) not in (0, 1, None)
            counts["abstain_pending"] += row.get("abstain") is None
            counts["abstain_invalid"] += row.get("abstain") is not None and type(row["abstain"]) is not bool
            counts["reference_pending"] += row.get("reference_complete") is not True or type(row.get("reference_G")) is not bool
            holds = effective_holds(row)
            counts["effective_authority_held_members"] += bool(holds)
            counts["historical_nested_hold_flags_preserved"] += historical_holds(row.get("decision_source"))
            counts["v6_historical_Viz_conflict_key_overlap"] += row["key"] in conflict_keys
            counts["v6_historical_Viz_conflict_exact_QA_overlap"] += row["qa_key"] in conflict_qas
            proof = identity_proof(row, identity_cache) if analyze else None
            if proof is not None and not proof["passed"]:
                holds.append("runtime_identity." + proof["hold_reason"])
                counts["runtime_identity_held_members"] += 1
            if holds:
                effective_pending.append({"key": row["key"], "source_cohort": name, "source_line": number,
                                          "condition": context(cond), "actual_hold_reasons": holds})
            for field in ("score_version", "score_reason", "behavior_source", "stage"):
                values[field][json.dumps(row.get(field), ensure_ascii=False)] += 1
            counts["model|" + model] += 1
            identities.add(row["source_identity"])
            source_paths.add(row["source_path"])
            examples.setdefault(model, {"source_cohort": name, "source_score_path": str(path),
                "source_score_line": number, "actual_original_record": row})
            seen[row["key"]] = (name, number)
            groups[cond][row["sample_id"]] = {"row": row, "line": original_line,
                "source_score_path": str(path), "source_score_sha256": expected_sha,
                "source_score_line": number, "source_cohort": name, "effective_hold_reasons": holds}
        require(counts["all_rows"] == expected_rows and counts["food_rows"] == expected_food,
                "Actual finite Food/source row counts differ")
        blockers = {key: value for key, value in counts.items() if value and
                    (key.endswith("_pending") or key.endswith("_invalid") or key.endswith("_mismatch"))}
        if blockers:
            findings.append({"source_cohort": name, "actual_scientific_blockers": blockers})
        per_source.append({"id": name, "path": str(path), "sha256": expected_sha,
            "receipt_path": str(receipt_path), "receipt_sha256": file_hash(receipt_path),
            "counts": dict(counts), "score_field_versions": {k: dict(v) for k, v in values.items()},
            "original_runtime_identity_count": len(identities), "original_raw_source_path_count": len(source_paths)})
    require(len(seen) == 231004, "Finite Food union unique key count differs")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "audit.json", {"schema": "kdm_finite_food_union_source_audit_v1", "passed": not findings,
        "Food_union_unique_keys": len(seen), "old_new_key_overlap": 0, "retained_registered_conditions": len(expected),
        "observed_conditions": len(groups), "reference_source_sha256": ref_sha, "sources": per_source,
        "actual_scientific_blockers": findings, "original_records_mutated": 0,
        "SCORING_snapshot_sha256": scoring_sha, "explicit_score_version_status": "not present in either original source",
        "historical_v6_conflict_path": str(CONFLICT), "historical_v6_conflict_sha256": file_hash(CONFLICT),
        "historical_conflict_changes_applied": 0, "effective_held_members": len(effective_pending),
        "raw_reopened": False, "source_score_files_hashed": 2, "GPU_initialized": False,
        "new_semantic_judgments": 0, "runner_sha256": file_hash(Path(__file__)),
        "completed_utc": datetime.now(timezone.utc).isoformat()})
    write(output / "bounded_original_record_examples.json", examples)
    write(output / "effective_held_members.json", effective_pending)
    if analyze:
        write(output / "original_identity_sidecar_index.json", list(identity_cache.values()))
    print(json.dumps({"passed": not findings, "Food_rows": len(seen), "old_new_key_overlap": 0,
                      "observed_conditions": len(groups), "blockers": findings}, ensure_ascii=False))
    return groups, expected, eval_samples, refs, identity_cache, effective_pending


def analysis(output, groups, expected, evaluation, refs, identity_cache, pending):
    union_path = output / "Food_union_score_rows.jsonl.gz"
    index_path = output / "source_record_index.jsonl.gz"
    original_by_key, union_line = {}, 0
    with gzip.open(union_path, "xt", encoding="utf-8", compresslevel=1) as merged, gzip.open(
            index_path, "xt", encoding="utf-8", compresslevel=1) as index:
        for key in sorted(groups, key=lambda item: tuple(map(str, item))):
            for sid, member in sorted(groups[key].items()):
                union_line += 1
                merged.write(member["line"])
                digest = hashlib.sha256(member["line"].encode("utf-8")).hexdigest()
                original_by_key[member["row"]["key"]] = digest
                index.write(json.dumps({"union_line": union_line, "key": member["row"]["key"],
                    "source_cohort": member["source_cohort"], "source_score_path": member["source_score_path"],
                    "source_score_sha256": member["source_score_sha256"], "source_score_line": member["source_score_line"],
                    "original_score_line_sha256": digest}, separators=(",", ":")) + "\n")
    actual, seen = 0, set()
    for _, row, line in score_lines(union_path):
        require(row["key"] not in seen and hashlib.sha256(line.encode("utf-8")).hexdigest() == original_by_key[row["key"]],
                "A union record changed an original field or duplicated a key")
        seen.add(row["key"])
        actual += 1
    require(actual == len(seen) == 231004, "The Food-only row union loses a source record")
    classes = sorted({row["class"] for row in evaluation.values()})
    coverage, eligible, metrics, missing = [], {}, [], []
    for key in sorted(expected, key=lambda item: tuple(map(str, item))):
        members = groups.get(key, {})
        projected = {sid: {**member["row"], "source_cohort": member["source_cohort"]} for sid, member in members.items()}
        status, absent = group_status(key, projected, evaluation, classes)
        holds = Counter(reason for member in members.values() for reason in member["effective_hold_reasons"])
        status.update(condition_id=stable_hash(context(key)), reference_unresolved_n=sum(
            row["reference_complete"] is not True or type(row["reference_G"]) is not bool for row in projected.values()),
            effective_held_members=sum(bool(member["effective_hold_reasons"]) for member in members.values()),
            effective_hold_reasons=json.dumps(dict(holds), sort_keys=True),
            decode_config_variants=len({stable_hash(row["config"]) for row in projected.values()}))
        status["analysis_eligible"] = status["analysis_eligible"] and not status["effective_held_members"] and not status["reference_unresolved_n"]
        coverage.append(status)
        missing.extend({"condition_id": status["condition_id"], **context(key), "sample_id": sid,
                        "key": task_id(key[0], {"sample": evaluation[sid], **expected[key]})} for sid in absent)
        if status["analysis_eligible"]:
            eligible[key] = projected
    direct = {(key[0], key[5], key[7], key[9]): members for key, members in eligible.items()
              if key[3] == "direct" and key[4] == "main"}
    for key, members in eligible.items():
        baseline = direct.get((key[0], key[5], key[7], key[9]))
        metric = condition_metrics(key, members, refs, baseline)
        tp, fp, fn = (metric["abstention_" + k + "_known_n"] for k in ("tp", "fp", "fn"))
        metric.update(condition_id=stable_hash(context(key)), canonical_denominator=2424,
            abstention_precision_numerator=tp, abstention_precision_denominator=tp + fp,
            abstention_recall_numerator=tp, abstention_recall_denominator=tp + fn,
            abstention_F1=(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None),
            precision_undefined_reason="no_predicted_abstentions" if not tp + fp else None,
            recall_undefined_reason="no_reference_positives" if not tp + fn else None,
            retention_undefined_reason=("matching_complete_guided_Direct_unavailable" if baseline is None else
                "empty_guided_Direct_intersection_reference" if metric["fixed_direct_reasonable_abstention_n"] == 0 else None),
            JointJ_numerator=metric["canonical_correct"] + tp, JointJ_denominator=2424,
            JointJ=(metric["canonical_correct"] + tp) / 2424)
        metrics.append(metric)
    draws = np.random.RandomState(shared.SEED).randint(0, 101, size=(shared.BOOT, 101))
    comparisons, missing_pairs = [], []
    for key, members in eligible.items():
        if key[3] == "direct":
            continue
        baseline = direct.get((key[0], key[5], key[7], key[9]))
        if baseline is None:
            missing_pairs.append({"condition_id": stable_hash(context(key)), **context(key),
                                  "reason": "matching_Direct_not_complete_or_effective_authority_held"})
            continue
        paired = []
        stored_prompt_hashes_checked = 0
        for collection in (members, baseline):
            derived = {}
            for sid, row in collection.items():
                prompt_sha = hashlib.sha256(task_prompt(row["question"], row["marker"], row["guided"]).encode()).hexdigest()
                if row.get("main_prompt_sha256") is not None:
                    require(row["main_prompt_sha256"] == prompt_sha, "A saved actual main prompt differs from the registered recipe")
                    stored_prompt_hashes_checked += 1
                derived[sid] = {**row, "main_prompt_sha256": prompt_sha}
            paired.append(derived)
        result = compare("same_main_prompt_direct", key, paired[0], paired[1], refs, classes, draws, paired[1])
        result.update(condition_id=stable_hash(context(key)), base_condition_id=stable_hash(context(key_of(next(iter(baseline.values()))))),
            pairing_prompt_evidence="registered_task_prompt_question_marker_guided_equal; available stored SHA checked",
            stored_actual_main_prompt_hashes_checked=stored_prompt_hashes_checked,
            absent_prompt_SHA_added_to_original_records=False)
        comparisons.append(result)
    write_csv(output / "condition_coverage.csv", coverage)
    write_csv(output / "complete_condition_metrics.csv", metrics)
    write(output / "paired_comparisons.json", comparisons)
    write(output / "missing_complete_pairs.json", missing_pairs)
    with gzip.open(output / "missing_condition_keys.jsonl.gz", "xt", encoding="utf-8", compresslevel=1) as stream:
        for row in missing:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    model_table = []
    for model in MODELS:
        old_n = sum(member["source_cohort"] == "old_food177333" for key, members in groups.items() if key[0] == model for member in members.values())
        new_n = sum(member["source_cohort"] == "new_food53671" for key, members in groups.items() if key[0] == model for member in members.values())
        rows = [row for row in coverage if row["model"] == model]
        model_table.append({"model": model, "old_Food_rows": old_n, "new_Food_rows": new_n,
            "union_unique_rows": old_n + new_n, "registered_conditions": len(rows),
            "complete_scientifically_eligible_conditions": sum(row["analysis_eligible"] for row in rows),
            "full_coverage_held_conditions": sum(row["missing_sample_n"] == 0 and not row["analysis_eligible"] for row in rows),
            "partial_conditions": sum(row["missing_sample_n"] > 0 for row in rows),
            "missing_registered_sample_keys": sum(row["missing_sample_n"] for row in rows),
            "effective_held_members": sum(row["effective_held_members"] for row in rows)})
    write_csv(output / "per_model_compact.csv", model_table)
    receipt = {"schema": "kdm_extended_food_row_union_analysis_v1", "passed": True,
        "actual_Food_union_rows": actual, "unique_keys": len(seen), "old_new_key_overlap": 0,
        "per_model": model_table, "all_original_record_fields_and_JSON_line_bytes_equal": True,
        "complete_eligible_conditions": len(eligible), "partial_conditions": sum(row["missing_sample_n"] > 0 for row in coverage),
        "full_coverage_held_conditions": sum(row["missing_sample_n"] == 0 and not row["analysis_eligible"] for row in coverage),
        "effective_held_members": len(pending), "original_identity_sidecars_checked": len(identity_cache),
        "actual_same_main_prompt_Direct_paired_comparisons": len(comparisons),
        "bootstrap": shared.BOOT, "seed": shared.SEED, "cluster": "Food-101 class", "clusters": 101,
        "partial_cohort_used_as_full_denominator": False, "raw_reopened": False, "new_semantic_judgments": 0,
        "GPU_initialized": False, "v8_package_modified": False, "score_version_missing_preserved": True,
        "historical_v16_or_variant_authority_reinterpreted": False,
        "historical_current_hold_status_source": "v6 isolated Viz conflict and explicit current effective row/decision flags",
        "runner_sha256": file_hash(Path(__file__)), "completed_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {path.name: file_hash(path) for path in output.iterdir() if path.is_file()}}
    write(output / "receipt.json", receipt)
    lines = ["# 四扩展模型 Food 科学分析 cohort", "", f"原字段等值并入 {actual:,} 条唯一 Food eval 记录。",
        f"完整且当前来源已接受的条件 {len(eligible)} 个；部分条件 {receipt['partial_conditions']} 个；完整覆盖但受阻断的条件 {receipt['full_coverage_held_conditions']} 个。",
        "", "每个主效果条件固定 eval 2424、101类别各24；P/R/保留率零分母留空并说明原因。",
        "同措辞 Direct 配对使用相同 model/sample/marker/guided/replicate、注册问题及 seed；保存提示 SHA 如有则逐项核对，缺省提示 SHA 保持原对象缺省。",
        f"实际 {len(comparisons)} 条配对比较沿用 BOOT={shared.BOOT}/SEED={shared.SEED}、101类别 cluster 函数。",
        "", "旧、新 accepted 来源与行号见 source_record_index.jsonl.gz；score_version 原字段缺省保持缺省。",
        "当前判定只检查 effective 顶层状态；历史 source_variants 保留，其既往 flags 单独统计。v6 已隔离 Viz 冲突与 Food 的键和完整 QA 无交叉。",
        "", "未读 raw、未重判标签、未修改旧输出、五模型、论文或 v8 审阅包。"]
    (output / "README.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("passed", "actual_Food_union_rows", "per_model",
        "complete_eligible_conditions", "partial_conditions", "full_coverage_held_conditions",
        "actual_same_main_prompt_Direct_paired_comparisons")}, ensure_ascii=False))


def load_samples():
    samples = {}
    with (ROOT / "data/current/all.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["dataset"] == "food101":
                require(row["id"] not in samples, "The registered Food domain repeats a sample")
                samples[row["id"]] = row
    require(len(samples) == 4848 and Counter((s["split"], s["class"]) for s in samples.values())
            == Counter({(split, cls): 24 for split in ("dev", "eval") for cls in
                        {s["class"] for s in samples.values()}}), "Registered Food split/class quotas differ")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUT))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "This source union must run with GPUs hidden")
    output = within(ROOT, args.output)
    output.relative_to(BASE)
    groups, expected, evaluation, refs, identities, pending = audit(output, analyze=not args.audit_only)
    if not args.audit_only:
        require(not load(output / "audit.json")["actual_scientific_blockers"], "Scientific source differences must remain on hold")
        analysis(output, groups, expected, evaluation, refs, identities, pending)


if __name__ == "__main__":
    main()
