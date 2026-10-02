"""Append accepted core dev/selection/Viz evidence without rewriting older members."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import file_hash, within
from workflows.paper_core import append_core_review as append

BASE = ROOT / "outputs/paper_core_20261002_dev_viz"
SOURCE = BASE / "v9_union_stage_20261002_2305"
MODELS = {"qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b"}
HISTORY_FIELDS = {"behavior_history_sources", "semantic_authority_sources", "name_history_source"}
LARGE_TRACE_FIELDS = {"selected_log_probabilities"}
SCALARS = ("correct_canonical", "canonical_name_in_primary_score", "correct_literal",
           "literal_extracted_name_score", "abstain", "uniform_reference", "accepted_reference",
           "annotated_answerable", "answer_quality_credit", "official_consensus_raw_credit",
           "seed", "terminated", "first_probability", "source_line")
ARRAYS = ("tokens",)
SAMPLE = ("sample_id", "target_class", "image_source", "gold_rank", "independent_correct_attempts",
          "independent_attempts", "official_record_index", "uniform_reference_source_line")
IDENTITY = (
    "config", "prompt", "reference_prompt", "neutral_prompt", "offset_prompt_tokens",
    "source_identity", "generation_identity", "model_checkpoint", "model_dtype", "source_generation_author")
SOURCE_FIELDS = ("source_path", "source_original_path", "complete_source_path",
                 "identity_source_path", "reference_source_path", "decision_source_path", "decision_source_line")
NAMES = ("condition", "sample", "qa", "identity", "source", "label")
SELECTION_FILES = ("selected_configs.json", "dev_metrics.csv", "selected_operating_points.csv",
                   "best_observed_endpoints.csv", "paired_effects.csv", "transitions.csv",
                   "metrics_all.csv", "native_matched_fixed_sets.csv", "analysis_receipt.json")


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def score_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            require(line.endswith("\n"), "An accepted scientific source has an incomplete line")
            yield number, json.loads(line)


def restore_projection(projected, original, builder, top=True):
    """Only the explicitly declared source fields need an immutable server lookup."""
    if isinstance(original, dict):
        require(isinstance(projected, dict), "Projected source object changed type")
        restored = {}
        for field, value in original.items():
            if field in projected:
                restored[field] = restore_projection(projected[field], value, builder, False)
            else:
                require((top and field in HISTORY_FIELDS | LARGE_TRACE_FIELDS) or field in builder.RECORD_ONLY_FIELDS
                        or field.endswith("sha256") or field == "line_sha256",
                        "An undeclared scientific field was omitted: " + field)
                restored[field] = value
        require(set(projected) <= set(original), "A scientific projection introduced a field")
        return restored
    if isinstance(original, list):
        require(isinstance(projected, list) and len(projected) == len(original), "A scientific list changed length")
        return [restore_projection(a, b, builder, False) for a, b in zip(projected, original)]
    require(projected == original, "A scientific value changed")
    return projected


def decode_record(record, dictionaries):
    parts = {name: dictionaries[name][record[name + "_id"]] for name in NAMES}
    value = {key: child for name in NAMES for key, child in parts[name].items()
             if key != "ledger_original_present_fields"}
    value.update({field: record[field] for field in parts["label"]["ledger_original_present_fields"]})
    return value


def accepted_score(directory, group, expected_viz_rows=12800):
    receipt_path = directory / "receipt.json"
    receipt = load(receipt_path)
    require(receipt.get("passed") is True and receipt.get("pending_QA") == 0
            and receipt.get("pending_memberships") == 0 and receipt.get("abstain_pending_rows") == 0
            and receipt.get("reference_join_missing") == 0,
            "Only an actually closed, source-bound score cohort may be appended")
    if group == "core_dev32320":
        require(receipt.get("stage") == "dev404" and receipt.get("rows") == 32320
                and receipt.get("conditions") == receipt.get("raw_complete_conditions")
                == receipt.get("primary_complete_conditions") == 80
                and receipt.get("canonical_pending_rows") == receipt.get("literal_pending_rows") == 0,
                "The five-model dev panel is not the completed 32320-row source")
    else:
        require(group == "core_viz_scored_each_model" and receipt.get("stage") == "viz512"
                and receipt.get("quality_pending_rows") == 0,
                "The Viz cohort is not a fully quality/behavior-resolved accepted source")
        generated = [p for p in receipt["source_parts"] if p["method"] != "direct"]
        require(expected_viz_rows in {12800, 13312}
                and len(generated) == expected_viz_rows // 512
                and sum(p["rows"] for p in generated) == expected_viz_rows
                and all(p["rows"] == 512 and p["raw_complete"] for p in generated),
                "The explicitly authorized generated Viz conditions have not all completed 512 inputs")
    path = directory / "score_rows.jsonl.gz"
    table_path = directory / "new_scores.parquet"
    require(file_hash(path) == receipt["outputs"][path.name]
            and file_hash(table_path) == receipt["outputs"][table_path.name],
            "An accepted full/scientific score source changed bytes")
    return path, table_path, receipt_path, receipt


def annotation_summary(row):
    """Copy existing authors/locators; this function never assigns a behavior label."""
    keys = ("annotation_model", "annotation_effort", "annotation_call_id", "annotation_authors",
            "decision_source_path", "decision_source_line", "behavior_source", "score_reason")
    return {"actual_existing_author_fields": {key: row[key] for key in keys if key in row},
            "indexed_original_fields_at_same_bound_score_line": sorted((HISTORY_FIELDS | LARGE_TRACE_FIELDS) & set(row)),
            "final_behavior": row.get("abstain"),
            "full_source_object_requires_immutable_server_score_line": True}


def append_scores(directory, group, output, prefix, additions, builder, expected_viz_rows=12800):
    path, table_path, acceptance, receipt = accepted_score(directory, group, expected_viz_rows)
    table = pq.read_table(table_path)
    original_table_rows = table.to_pylist()
    destination = output / prefix / "compact" / group
    destination.mkdir(parents=True, exist_ok=False)
    dictionaries = {name: builder.Dictionary(name) for name in NAMES}
    annotation = builder.Dictionary("annotation_source")
    record_fields = SCALARS + ARRAYS
    records, original_fields, pending, models, conditions = [], set(), Counter(), Counter(), Counter()
    omitted_fields, histories, keys, identity_ids = set(), set(), set(), set()
    trace_counts = Counter()
    condition_fields = tuple(builder.COND_FIELDS) + ("condition_id", "source_condition_id", "expected_n",
                                                     "source_scope", "source_raw_complete")
    for number, original in score_rows(path):
        require(original.get("model") in MODELS and isinstance(original.get("question"), str)
                and isinstance(original.get("answer"), str) and isinstance(original.get("config"), dict),
                "A new scientific row lacks complete QA/model/configuration")
        require(original["key"] not in keys, "A new score cohort duplicates a task key")
        keys.add(original["key"])
        identity_ids.add(original["source_identity"])
        require(original.get("abstain") is not None, "A complete cohort contains an undecided behavior")
        if group == "core_dev32320":
            require(original.get("dataset") == "food101" and original.get("split") == "dev"
                    and original.get("correct_canonical") is not None and original.get("correct_literal") is not None
                    and original.get("uniform_reference") is not None,
                    "The accepted dev row changed dataset/split or a fixed scientific decision")
        else:
            require(original.get("dataset") == "vizwiz" and original.get("split") == "eval"
                    and original.get("answer_quality_credit") is not None
                    and original.get("annotated_answerable") is not None,
                    "Viz official continuous quality/answerability is missing")
        original_fields.update(original)
        histories.update(field for field in HISTORY_FIELDS if field in original)
        for field in LARGE_TRACE_FIELDS & set(original):
            require(isinstance(original[field], list), "An indexed ordinary trace field changed type")
            trace_counts[field] += len(original[field])
        reduced = {field: value for field, value in original.items() if field not in HISTORY_FIELDS | LARGE_TRACE_FIELDS}
        reduced = builder.remove_digests(reduced, "row", omitted_fields)
        parts = {
            "condition": {field: reduced[field] for field in condition_fields if field in reduced},
            "sample": {field: reduced[field] for field in SAMPLE if field in reduced},
            "qa": {field: reduced[field] for field in ("question", "answer")},
            "identity": {field: reduced[field] for field in IDENTITY if field in reduced},
            "source": {field: reduced[field] for field in SOURCE_FIELDS if field in reduced},
        }
        used = set().union(*(set(value) for value in parts.values())) | set(record_fields)
        parts["label"] = {field: value for field, value in reduced.items() if field not in used}
        parts["label"]["ledger_original_present_fields"] = [field for field in record_fields if field in reduced]
        record = {name + "_id": dictionaries[name].add(parts[name]) for name in NAMES}
        record.update(source_record_line=number, annotation_source_id=annotation.add(annotation_summary(original)),
                      **{field: reduced.get(field) for field in record_fields})
        restored = {
            field: value for name in NAMES for field, value in parts[name].items()
            if field != "ledger_original_present_fields"}
        restored.update({field: record[field] for field in parts["label"]["ledger_original_present_fields"]})
        require(restored == reduced and restore_projection(restored, original, builder) == original,
                "The compact scientific projection did not restore the complete bound original")
        require(number <= len(original_table_rows), "Full score and analysis parquet counts differ")
        compact_original = original_table_rows[number - 1]
        require({field: original[field] for field in compact_original} == compact_original,
                "An existing analysis parquet scientific value differs from its full score row")
        records.append(record)
        models[original["model"]] += 1
        conditions[original["condition_id"]] += 1
        for field in record_fields:
            pending[field] += field in original and original[field] is None
    require(len(records) == receipt["rows"] == table.num_rows and set(models) == MODELS,
            "The compact score source has incomplete actual coverage")
    for dictionary in dictionaries.values():
        dictionary.export(destination)
    annotation.export(destination)
    builder.checked_parquet(pa.Table.from_pylist(records), destination / "scores.parquet")
    values = {name: {row[name + "_id"]: json.loads(row["original_json"])
                     for row in pq.read_table(destination / (name + "_dictionary.parquet")).to_pylist()}
              for name in NAMES}
    actual_records = pq.read_table(destination / "scores.parquet").to_pylist()
    reconstructed_table = []
    for number, original in score_rows(path):
        record = actual_records[number - 1]
        require(record["source_record_line"] == number, "An actual compact source line changed")
        value = decode_record(record, values)
        require(restore_projection(value, original, builder) == original,
                "Actual compact dictionaries/ledger do not restore the bound original")
        reconstructed_table.append({field: original[field] for field in table.column_names})
    require(table.equals(pa.Table.from_pylist(reconstructed_table, schema=table.schema)),
            "Reconstructed analysis values/types differ from the existing scored parquet")
    identity_sources = {}
    raw_pointers = []
    for part in receipt["source_parts"]:
        identity_path = within(ROOT, part["identity_path"])
        expected = part.get("identity_sha256")
        require(expected and file_hash(identity_path) == expected, "An actual original identity sidecar changed")
        payload = load(identity_path)
        identifier = payload["identity"]
        if identifier not in identity_sources:
            identity_sources[identifier] = {"path": str(identity_path), "sha256": expected,
                                            "noise_step": payload.get("definition", {}).get("noise_step")}
            append.source_copy(identity_path, output, prefix + "/inputs/" + group + "/identities/"
                               + str(len(identity_sources) - 1) + ".json", additions,
                               "complete original accepted runtime identity", expected)
        raw = within(ROOT, part["raw_path"])
        pointer = append.omitted_pointer(raw, "Original raw/trace stays on server; accepted receipt binds its SHA and source lines",
                                         known_sha=part["raw_sha256"], source_count=part["rows"])
        pointer.update(record_locator="1-based original JSONL source_line in compact source dictionary",
                       complete_receipt_path=part["complete_path"], complete_receipt_sha256=part["complete_sha256"],
                       identity_path=str(identity_path), identity_sha256=expected)
        raw_pointers.append(pointer)
    require(identity_ids <= set(identity_sources), "A scientific row has no preserved full original identity")
    original_sha = receipt["outputs"][path.name]
    score_pointer = append.omitted_pointer(path, "Full original provenance/history is indexed by immutable gzip SHA and uncompressed line",
                                           known_sha=original_sha, source_count=len(records))
    score_pointer.update(record_locator="1-based uncompressed JSONL source_record_line",
                         indexed_ordinary_trace_fields=dict(trace_counts),
                         acceptance_receipt_path=str(acceptance), acceptance_receipt_sha256=file_hash(acceptance))
    append.write(destination / "SOURCE_POINTERS.json", [score_pointer, *raw_pointers])
    append.source_copy(acceptance, output, prefix + "/inputs/" + group + "/acceptance_receipt.json", additions,
                       "actual closed score cohort and original provider/receipt manifest")
    append.source_copy(directory / "sources.csv", output, prefix + "/inputs/" + group + "/sources.csv", additions,
                       "actual condition/source inventory", receipt["outputs"]["sources.csv"])
    append.source_copy(directory / "metrics_all.csv", output, prefix + "/inputs/" + group + "/metrics_all.csv", additions,
                       "existing score cohort metrics; not recomputed", receipt["outputs"]["metrics_all.csv"])
    result = {"id": group, "status": "accepted_compact_score_added", "rows": len(records),
              "per_model": dict(models), "condition_counts": dict(conditions), "pending_existing_fields": dict(pending),
              "original_fields": sorted(original_fields),
              "all_required_scoring_QA_and_generation_identity_values_preserved": True,
              "all_non_indexed_original_fields_embedded_equal": True,
              "actual_rows_reconstructed_equal_to_complete_bound_server_source": len(records),
              "actual_original_analysis_parquet_values_and_types_equal": True,
              "analysis_schema_base64": base64.b64encode(table.schema.serialize().to_pybytes()).decode("ascii"),
              "all_original_fields_fully_embedded": not histories and not omitted_fields and not trace_counts,
              "historical_provenance_fields_indexed_by_original_score_line": sorted(histories),
              "large_ordinary_trace_fields_indexed_by_original_score_line": dict(trace_counts),
              "source_only_digest_or_key_fields_indexed_by_original_score_line": sorted(omitted_fields),
              "original_source_path": str(path), "original_source_sha256": original_sha,
              "original_analysis_parquet_path": str(table_path),
              "original_analysis_parquet_sha256": receipt["outputs"][table_path.name],
              "full_original_reconstruction_requires_bound_server_score_lines": True,
              "source_record_line_one_based": True, "identity_sources": identity_sources,
              "field_availability": {"token_ids": "tokens field, exact original values", "noise_step": "full original identity definition",
                                     "noise_tensor_evidence": "unknown; no new tensor reconstruction",
                                     "actual_chat_prompt_token_ids": "unknown unless explicitly present in an existing source field",
                                     "termination": "exact existing terminated field; additional unstored termination reasons unknown"},
              "new_scores_or_semantic_judgments": 0}
    append.write(destination / "receipt.json", result)
    return result


def append_selection(directory, output, prefix, additions):
    receipt_path = directory / "analysis_receipt.json"
    receipt = load(receipt_path)
    require(receipt.get("passed") and receipt.get("selected_configs") == 20
            and receipt.get("dev_rows") == 32320 and receipt.get("full_five_model_panel") is True
            and set(receipt.get("models", [])) == MODELS and receipt.get("dev_only_selection") is True
            and receipt.get("bootstrap_replicates") == 2000 and receipt.get("bootstrap_seed") == 20260929,
            "Only the completed original five-model dev-only selection can be copied")
    counts = {}
    for name in SELECTION_FILES:
        path = directory / name
        require(path.is_file(), "A completed selection table is missing: " + name)
        if name == "selected_configs.json":
            values = load(path)
            require(isinstance(values, list) and len(values) == 20
                    and all(value.get("selection") == "dev_selected" for value in values),
                    "Actual dev-selected configuration identities are incomplete")
            counts[name] = len(values)
        elif path.suffix == ".csv":
            counts[name] = len(pd.read_csv(path))
        append.source_copy(path, output, prefix + "/inputs/frozen_dev_selection20/" + name, additions,
                           "completed original dev selection/eval/paired statistics; all bytes copied")
    require(counts["selected_operating_points.csv"] == 35 and counts["best_observed_endpoints.csv"] == 120
            and counts["dev_metrics.csv"] == 80,
            "Actual selected/best-observed/development table row counts differ")
    decisions = directory / "dev_decisions.csv"
    require(decisions.is_file() and len(pd.read_csv(decisions)) == 32320,
            "The actual selection decisions are not the complete original dev panel")
    append.write(output / prefix / "inputs/frozen_dev_selection20/OMITTED_DUPLICATE_DEV_DECISIONS.json",
                 append.omitted_pointer(decisions, "Scientific decisions already embedded in the core dev compact group; original selection CSV remains on server",
                                        known_sha=file_hash(decisions), source_count=32320))
    return {"id": "frozen_dev_selection20", "status": "accepted_table_added", "actual_rows": 20,
            "table_counts": counts, "original_selection_values_rules_and_frozen_times_unchanged": True,
            "source_directory": str(directory), "source_receipt_sha256": file_hash(receipt_path)}


def inspect(directory, source):
    path, table, receipt_path, receipt = accepted_score(directory, "core_dev32320")
    first = next(score_rows(path))[1]
    result = {"schema": "kdm_completed_core_append_source_inventory_v1", "source_package": str(source),
              "source_package_original_files": len(append.snapshot(source)),
              "actual_score_receipt_path": str(receipt_path), "actual_score_receipt_sha256": file_hash(receipt_path),
              "actual_rows": receipt["rows"], "actual_conditions": receipt["conditions"],
              "full_score_source": str(path), "full_score_source_sha256": receipt["outputs"][path.name],
              "full_score_bytes": path.stat().st_size, "original_analysis_parquet_schema": str(pq.read_schema(table)),
              "first_actual_full_score_fields": {key: type(value).__name__ for key, value in first.items()},
              "pure_history_fields_to_index": sorted(HISTORY_FIELDS & set(first)), "actual_source_parts": len(receipt["source_parts"]),
              "planned_or_incomplete_cohorts_included": False, "final_zip_created": False}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", default=str(SOURCE))
    parser.add_argument("--stage", choices=("dev-selection", "viz"))
    parser.add_argument("--dev-dir")
    parser.add_argument("--selection-dir")
    parser.add_argument("--viz-dir")
    parser.add_argument("--expected-viz-generated-rows", type=int, choices=(12800, 13312), default=12800)
    parser.add_argument("--output")
    parser.add_argument("--namespace")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--max-compressed-bytes", type=int, default=25000000)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "This finite packaging task must explicitly hide GPUs")
    require(0 < args.max_compressed_bytes <= 25000000, "The final review budget may not exceed 25,000,000 bytes")
    source = within(ROOT, args.source_package)
    require(source.is_dir(), "The existing source review package does not exist")
    if args.inspect:
        require(args.dev_dir, "Source inspection requires an accepted dev directory")
        inspect(within(ROOT, args.dev_dir), source)
        return
    require(args.stage and args.output and args.namespace, "An append needs an explicit stage/output/namespace")
    require(args.namespace.replace("_", "").isalnum(), "An append namespace must be simple and unique")
    if args.stage == "dev-selection":
        require(args.dev_dir and args.selection_dir and not args.viz_dir,
                "First append only the complete dev panel and the complete selected eval tables")
    else:
        require(args.viz_dir and not args.dev_dir and not args.selection_dir,
                "The subsequent Viz stage only appends its completed new score cohort")
        previous = {entry["id"] for path in (source / "appended").glob("*/ACTUAL_COMPLETION.json")
                    for entry in load(path) if entry.get("status") in {"accepted_compact_score_added", "accepted_table_added"}}
        require({"core_dev32320", "frozen_dev_selection20"} <= previous,
                "Append completed core dev/selection before the Viz stage")
    output = within(ROOT, args.output)
    output.relative_to(BASE)
    require(source != output and not output.exists(), "The new staging directory must be exclusive")
    started = time.monotonic()
    original = append.snapshot(source)
    require("STAGING_MANIFEST.json" in original and "compact/qa_dictionary.parquet" in original,
            "The source is not a validated existing review package")
    builder = append.module_from_package(source)
    shutil.copytree(source, output)
    prefix = "appended/" + args.namespace
    require(not (output / prefix).exists(), "The namespace already exists in the original package")
    append.write(output / prefix / "ORIGINAL_V6_FILES.json", original)
    additions, completion = [], []
    if args.stage == "dev-selection":
        completion.append(append_scores(within(ROOT, args.dev_dir), "core_dev32320", output, prefix, additions, builder))
        completion.append(append_selection(within(ROOT, args.selection_dir), output, prefix, additions))
    else:
        completion.append(append_scores(within(ROOT, args.viz_dir), "core_viz_scored_each_model", output, prefix,
                                        additions, builder, args.expected_viz_generated_rows))
    missing = {}
    for path in sorted((source / "appended").glob("*/MISSING_INPUTS.json")):
        for entry in load(path):
            missing[entry["id"]] = entry
    for path in sorted((source / "appended").glob("*/ACTUAL_COMPLETION.json")):
        for entry in load(path):
            if entry.get("status") in {"accepted_compact_score_added", "accepted_table_added", "completed_existing_evidence"}:
                missing.pop(entry["id"], None)
    for entry in completion:
        missing.pop(entry["id"], None)
    append.write(output / prefix / "MISSING_INPUTS.json", [missing[key] for key in sorted(missing)])
    append.write(output / prefix / "ACTUAL_COMPLETION.json", completion)
    append.write(output / prefix / "OMITTED_SOURCE_POINTERS.json", [])
    append.source_copy(Path(__file__), output, prefix + "/scripts/append_completed_core.py", additions,
                       "actual thin three-input packaging adapter; no scoring or semantic decisions")
    append.write(output / prefix / "ADDED_SOURCES.json", additions)
    (output / prefix / "README.zh.md").write_text(
        "# 核心补充数据独立追加\n\n旧包全部文件、字典 ID 和科学结果保持逐字节一致。"
        "本命名空间只追加 ACTUAL_COMPLETION.json 中已验收来源，不将未完成实验计入完成量。\n\n"
        "compact 字典与 ledger 保留既有逐样本评分/参考、全文 QA、token IDs、首位概率、配置、随机种子和终止状态。"
        "逐行实际反解与原 analysis parquet 值/类型均已核对。完整原身份 sidecar 在 inputs。"
        "确实没有保存的噪声张量、聊天 prompt token IDs 或额外终止原因保持 unknown。\n\n"
        "普通 dev/Viz 的 selected_log_probabilities 逐步轨迹留服务器；字段名与实际条目数见 compact receipt 和 SOURCE_POINTERS.json。"
        "关键机制 trace、既有12案例与全部旧包成员保持。原始 key/digest 与明确列出的历史 provenance 嵌套没有全部重复嵌入。"
        "完整原 JSON 对象需要按 compact receipt 的原 score SHA 与 source_record_line 从服务器恢复；"
        "每条最终判断、初始和纠正历史均可在同一原行定位。annotation_source 字典仅复制既有作者字段和源行，"
        "打包代理不是标注作者。raw/完整 trace 的既有 SHA、实际路径和原始行号见 SOURCE_POINTERS.json。\n\n"
        "新增小表保持原字节与分母。此阶段未重新标注/评分、未使用 GPU/GT/API、未修改论文、未生成最终 ZIP。"
        "压缩上界是实际逐成员 DEFLATE 与保守元数据预算，不能当作最终 ZIP 大小。\n", encoding="utf-8")
    members = [(p, str(p.relative_to(output))) for p in sorted(output.rglob("*")) if p.is_file()]
    bound = append.compressed_bound(members) + 65536
    require(bound <= args.max_compressed_bytes, "Actual compact DEFLATE bound exceeds the finite review budget")
    require(append.snapshot(source) == original and all(file_hash(output / name) == proof["sha256"]
                                                     for name, proof in original.items()),
            "An existing review member changed during the append")
    result = {"schema": "kdm_completed_core_append_cpu_v10", "passed": True, "stage": args.stage,
              "source_package": str(source), "staging": str(output), "namespace": prefix,
              "original_files": len(original), "original_bytes": sum(value["bytes"] for value in original.values()),
              "original_members_byte_equal": True, "old_dictionary_ID_bytes_changed": 0,
              "accepted_actual_groups": [entry["id"] for entry in completion], "missing_inputs": len(missing),
              "compressed_zip_upper_bound_bytes": bound, "max_compressed_bytes": args.max_compressed_bytes,
              "compressed_estimate_method": "actual raw DEFLATE per file plus conservative metadata/proof reserve",
              "final_zip_created": False, "scores_rewritten": False, "new_semantic_judgments": 0,
              "GPU_initialized": False, "official_GT_read": False, "new_generations": 0, "new_API_calls": 0,
              "papers_modified": False, "compiler_author": {"agent": "/root/package_union_append",
              "model": "gpt-6.1-sol", "effort": "max", "call_id": ""}, "source_sha256": file_hash(Path(__file__)),
              "actual_argv": sys.argv, "cpu_python": sys.executable, "elapsed_seconds": time.monotonic() - started,
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    append.write(output / prefix / "APPEND_RECEIPT.json", result)
    preview = append.verify(output, prefix, check_manifest=False)
    append.write(output / prefix / "VERIFICATION.json", {key: value for key, value in preview.items()
                 if key not in {"actual_package_files", "actual_package_bytes"}})
    append.write(output / prefix / "PACKAGE_MANIFEST.json", {"schema": "kdm_append_only_core_member_manifest_v10",
                 "files": append.snapshot(output), "final_zip_created": False})
    verified = append.verify(output, prefix)
    actual_bound = append.compressed_bound([(p, str(p.relative_to(output))) for p in sorted(output.rglob("*")) if p.is_file()])
    require(actual_bound <= bound, "Actual proof files exceeded the declared finite reserve")
    print(json.dumps({**result, "actual_verification": verified, "actual_all_members_DEFLATE_bound_bytes": actual_bound,
                      "append_receipt_sha256": file_hash(output / prefix / "APPEND_RECEIPT.json"),
                      "package_manifest_sha256": file_hash(output / prefix / "PACKAGE_MANIFEST.json")},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
