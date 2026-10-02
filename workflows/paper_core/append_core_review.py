"""Append completed CPU evidence to an exclusive, byte-preserved review copy."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import file_hash, within

BASE = ROOT / "outputs/paper_core_20261002_dev_viz"
SOURCE = BASE / "review_package_staging_20261002_1700/package_v6"
DEST = BASE / "package_v7_staging_20261002_2045"
FUTURE = (
    ("core_dev32320", 32320, "完整已决 dev 逐样本评分及原条件/配置/QA/身份"),
    ("frozen_dev_selection20", 20, "原始 dev 选择的 20 项，保留真实选择规则与源条件"),
    ("core_viz_scored_each_model", None, "各核心模型完整 Viz 已决评分与官方连续质量来源"),
    ("extended_food53671", 53671, "新增 Food 已决逐样本评分、完整身份与真实 QA"),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    require(not path.exists(), "An append would overwrite an existing package member: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def snapshot(path):
    result = {}
    for member in sorted(path.rglob("*")):
        require(not member.is_symlink(), "A review member must have explicit regular-file bytes")
        if member.is_file():
            result[str(member.relative_to(path))] = {"bytes": member.stat().st_size, "sha256": file_hash(member)}
    return result


def source_copy(path, output, relative, index, scope, expected_sha=None):
    require(path.is_file(), "A completed source file is missing: " + str(path))
    target = output / relative
    target.resolve().relative_to(output.resolve())
    require(not target.exists(), "An added source collides with an original package member: " + relative)
    actual = file_hash(path)
    require(expected_sha is None or actual == expected_sha, "An accepted CPU source changed: " + str(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    require(file_hash(target) == actual, "Copied evidence bytes differ from their source")
    index.append({"source_path": str(path), "package_path": relative, "source_sha256": actual,
                  "bytes": path.stat().st_size, "source_scope": scope})


def omitted_pointer(path, reason, known_sha=None, source_count=None):
    require(path.is_file(), "An omitted source pointer is not an actual file: " + str(path))
    return {"source_path": str(path), "source_bytes": path.stat().st_size,
            "source_sha256": known_sha, "source_sha256_status": "existing receipt" if known_sha else "not rehashed",
            "source_mtime_ns": path.stat().st_mtime_ns, "source_count": source_count,
            "package_included": False, "reason": reason}


def append_existing(output, prefix, index):
    completed, omitted = [], []
    cda = BASE / "cda_audit_20261002_1700/results"
    receipt = load(cda / "execution_receipt.json")
    validation = load(cda / "summary_verification.json")
    require(receipt["new_generations"] == receipt["new_API_calls"] == 0
            and receipt["GPU_initialized"] is False and validation["summary_rows_checked"] == 177
            and validation["max_response_or_step_count_error"] == 0,
            "The completed CDA CPU evidence does not match its actual receipt")
    for entry in receipt["data_files"]:
        source_copy(cda / entry["name"], output, prefix + "/evidence/cda/" + entry["name"],
                    index, "completed frozen-five CDA CPU audit", entry["sha256"])
    source_copy(cda / "execution_receipt.json", output, prefix + "/evidence/cda/execution_receipt.json",
                index, "completed CDA audit receipt")
    for name in receipt["omitted_full_data"]:
        omitted.append(omitted_pointer(cda / name, "Full trace remains at its original server path"))
    completed.append({"id": "cda_existing_cpu", "status": "completed_existing_evidence",
                      "summary_rows_checked": 177, "behavior_conditions": 20, "source_directory": str(cda)})
    replay = BASE / "replay_audit_20261002_1700/final"
    verification = load(replay / "verification.json")
    require(verification["audited_positions"] == verification["exact_source_matches"] == 9
            and verification["gpu_forwards"] == 0 and verification["frozen_outputs_or_scores_modified"] is False,
            "Replay source validation is not the completed nine-position CPU audit")
    for member in sorted(replay.iterdir()):
        require(member.is_file(), "Unexpected directory inside the explicit replay final source")
        expected = verification.get("outputs", {}).get(member.name, {}).get("sha256")
        source_copy(member, output, prefix + "/evidence/replay/" + member.name, index,
                    "completed CPU replay source comparison; causal explanation retains its original unknowns", expected)
    completed.append({"id": "replay_source9_cpu", "status": "completed_existing_evidence",
                      "positions": 9, "paths": 6, "causal_reason_status": verification["causal_reason_status"],
                      "source_directory": str(replay)})
    four = BASE / "four_view_existing_summary_20261002_1900/final"
    analysis = load(four / "analysis_receipt.json")
    require(analysis["passed"] and analysis["source_pair_rows"] == 1049
            and analysis["saved_event_positions"] == 814, "Existing four-view CPU analysis has not passed")
    for member in sorted(four.rglob("*")):
        require(not member.is_symlink(), "An analysis source must be explicit regular-file evidence")
        if member.is_file():
            source_copy(member, output, prefix + "/evidence/four_view/" + str(member.relative_to(four)), index,
                        "existing four-view algebra, plots and frozen beef behavior/source join")
    completed.append({"id": "four_view_existing_cpu_summary", "status": "completed_existing_evidence",
                      "positions": 814, "pairs": 1049, "source_directory": str(four)})
    coverage = BASE / "generation_coverage_refresh_20261002_1920/report_v3_current_registered_queues/final_tables"
    state = load(coverage / "CURRENT_STATE.json")
    require(state["schema"] == "kdm_current_source_bound_formal_key_coverage_v3"
            and state["raw_files_opened"] == state["new_scoring_or_labels"] == 0,
            "Coverage is not the completed source-key-only v3 report")
    for member in sorted(coverage.iterdir()):
        if member.name == "source_bound_uncovered_registered_keys.jsonl.gz":
            omitted.append(omitted_pointer(member, "Complete source-bound key difference stays on the server",
                                           source_count=state["uncovered_key_export_rows"]))
        else:
            require(member.is_file() and member.suffix in {".csv", ".json", ".md"},
                    "Unexpected file in the explicit compact generation coverage source")
            source_copy(member, output, prefix + "/evidence/generation_coverage/" + member.name, index,
                        "current registered/source key coverage; unknown Viz counts remain null")
    completed.append({"id": "extended_registered_key_coverage_cpu", "status": "completed_existing_evidence",
                      "metadata_claims": state["metadata_claims_checked"], "source_directory": str(coverage),
                      "scope": state["full_scope_qualification"]})
    return completed, omitted


def module_from_package(package):
    source = package / "scripts/build_existing_review_staging.py"
    spec = importlib.util.spec_from_file_location("existing_review_compact_builder", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_rows(path):
    if path.suffix == ".parquet":
        for number, row in enumerate(pq.read_table(path).to_pylist(), 1):
            yield number, row
    else:
        require(path.suffix in {".jsonl", ".gz"}, "An accepted score source must be JSONL or parquet")
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                require(line.endswith("\n"), "An accepted score source has an incomplete final line")
                yield number, json.loads(line)


def append_score(group, package, output, prefix, index, builder):
    """A separate namespace keeps every old dictionary and ledger byte unchanged."""
    path = within(ROOT, group["source_path"])
    require(file_hash(path) == group["source_sha256"], "An accepted score source SHA changed")
    acceptance = within(ROOT, group["acceptance_receipt_path"])
    require(file_hash(acceptance) == group["acceptance_receipt_sha256"], "The score acceptance receipt changed")
    receipt = load(acceptance)
    checks = group["acceptance_checks"]
    require(checks and all(receipt.get(field) == expected for field, expected in checks.items()),
            "The explicit accepted score receipt checks did not pass")
    identity_index = {}
    for member in group["identity_members"]:
        source = within(ROOT, member["path"])
        require(file_hash(source) == member["sha256"], "A full original identity sidecar changed")
        payload = load(source)
        require("identity" in payload and "definition" in payload, "A full source identity is required")
        identity_index[payload["identity"]] = member
        source_copy(source, output, prefix + "/inputs/" + group["id"] + "/identities/" +
                    str(len(identity_index) - 1) + ".json", index, "complete original runtime identity", member["sha256"])
    require(identity_index, "A scored addition requires its complete original identity sidecars")
    names = ("condition", "sample", "qa", "identity", "source", "label")
    dictionaries = {name: builder.Dictionary(name) for name in names}
    sample_fields = ("sample_id", "target_class")
    qa_fields = ("question", "answer")
    source_fields = tuple(builder.SOURCE_FIELDS) + ("key", "original_key", "qa_key", "source_line",
                                                   "raw_line_sha256", "config_sha256")
    scalar_fields = tuple(builder.SCALAR_FIELDS)
    records, unique, model_counts, decision_counts = [], set(), Counter(), Counter()
    source_only_fields = set()
    for number, row in source_rows(path):
        require(isinstance(row.get("question"), str) and isinstance(row.get("answer"), str)
                and row.get("sample_id") is not None and row.get("config") is not None
                and row.get("source_identity") in identity_index,
                "An accepted score row lacks full QA, explicit config or bound original identity")
        key = tuple(builder.canonical_json(row[field]) for field in group["unique_key_fields"])
        require(key not in unique, "An accepted new score cohort repeats a declared sample/task key")
        unique.add(key)
        for field, value in group.get("exact_row_fields", {}).items():
            require(row.get(field) == value, "An accepted score row changed its declared dataset/split/condition")
        for field in group["require_decided_fields"]:
            require(field in row and row[field] is not None, "An allegedly complete score has an undecided field")
        parts = {
            "condition": {field: row[field] for field in builder.COND_FIELDS if field in row},
            "sample": {field: row[field] for field in sample_fields if field in row},
            "qa": {field: row[field] for field in qa_fields},
            "identity": {field: row[field] for field in builder.IDENTITY_FIELDS if field in row},
            "source": {field: row[field] for field in source_fields if field in row},
        }
        used = set().union(*(set(value) for value in parts.values())) | set(scalar_fields)
        parts["label"] = {field: value for field, value in row.items() if field not in used}
        parts["label"]["scalar_original_present_fields"] = [field for field in scalar_fields if field in row]
        restored = {field: value for name in names for field, value in parts[name].items()
                    if field != "scalar_original_present_fields"}
        restored.update({field: row[field] for field in parts["label"]["scalar_original_present_fields"]})
        require(restored == row, "Compact split changed an original accepted scientific or provenance field")
        projected = {name: builder.remove_digests(parts[name], name, source_only_fields) for name in names}
        reconstructed_projection = {field: value for name in names for field, value in projected[name].items()
                                    if field != "scalar_original_present_fields"}
        reconstructed_projection.update({field: row[field] for field in parts["label"]["scalar_original_present_fields"]})
        require(reconstructed_projection == builder.remove_digests(row, "row", set()),
                "An indexed source-only projection changed an original scientific or identity field")
        record = {name + "_id": dictionaries[name].add(projected[name]) for name in names}
        record.update(source_record_line=number, **{field: row.get(field) for field in scalar_fields})
        records.append(record)
        model_counts[row["model"]] += 1
        for field in scalar_fields:
            decision_counts[field + "_pending"] += field in row and row[field] is None
    require(len(records) == group["expected_rows"] and records,
            "A complete new score cohort does not have its declared actual row count")
    destination = output / prefix / "compact" / group["id"]
    destination.mkdir(parents=True, exist_ok=False)
    for dictionary in dictionaries.values():
        dictionary.export(destination)
    table = pa.Table.from_pylist(records)
    builder.checked_parquet(table, destination / "scores.parquet")
    restored = pq.read_table(destination / "scores.parquet").to_pylist()
    require(restored == records, "Actual typed compact scores do not reproduce their accepted original values")
    source_copy(acceptance, output, prefix + "/inputs/" + group["id"] + "/acceptance_receipt.json",
                index, "accepted source cohort receipt", group["acceptance_receipt_sha256"])
    payload = {"id": group["id"], "status": "accepted_compact_score_added", "namespace": prefix + "/" + group["id"],
               "rows": len(records), "per_model": dict(model_counts), "decision_pending": dict(decision_counts),
               "all_original_values_equal_before_source_only_projection": True, "old_IDs_rewritten": False,
               "all_original_fields_fully_embedded": not source_only_fields,
               "all_non_digest_original_fields_compactly_embedded_equal": True,
               "source_only_fields_require_bound_original_score_line": bool(source_only_fields),
               "source_only_digest_or_identifier_fields": sorted(source_only_fields),
               "source_only_original_fields_locator": "immutable original_source_sha256 plus 1-based source_record_line",
               "all_scientific_fields_complete_QA_and_identity_config_equal": True,
               "original_source_path": str(path), "original_source_sha256": group["source_sha256"],
               "source_record_line_one_based": True, "full_QA_and_original_identity_preserved": True}
    write(destination / "receipt.json", payload)
    return payload


def append_planned(group, package, output, prefix, index, builder):
    if group["status"] != "accepted":
        return {**group, "status": "missing", "completion_claimed": False}
    require(group["id"].replace("_", "").isalnum(), "An addition needs a simple unique group ID")
    if group["kind"] == "scored_records":
        return append_score(group, package, output, prefix, index, builder)
    require(group["kind"] == "accepted_table", "Unsupported explicit future addition kind")
    source = within(ROOT, group["source_path"])
    if source.suffix == ".csv":
        count = len(pd.read_csv(source))
    elif source.suffix == ".parquet":
        count = pq.read_metadata(source).num_rows
    else:
        values = load(source)
        require(isinstance(values, list), "An accepted JSON selection must be an explicit row array")
        count = len(values)
    require(count == group["expected_rows"] and count > 0, "An accepted selection table has the wrong actual row count")
    receipt_path = within(ROOT, group["acceptance_receipt_path"])
    require(file_hash(receipt_path) == group["acceptance_receipt_sha256"], "An accepted table receipt changed")
    receipt = load(receipt_path)
    require(group["acceptance_checks"] and all(receipt.get(k) == v for k, v in group["acceptance_checks"].items()),
            "An accepted table has not passed its explicit original receipt checks")
    source_copy(source, output, prefix + "/inputs/" + group["id"] + "/" + source.name,
                index, "complete accepted original table", group["source_sha256"])
    source_copy(receipt_path, output, prefix + "/inputs/" + group["id"] + "/acceptance_receipt.json",
                index, "accepted original table receipt", group["acceptance_receipt_sha256"])
    return {"id": group["id"], "status": "accepted_table_added", "actual_rows": count,
            "original_values_and_selection_rule_unchanged": True}


def compressed_bound(paths):
    """Estimate a standard ZIP32 upper bound without writing a ZIP archive."""
    total = 4096 + 131072
    for path, name in paths:
        compressor = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-15)
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(compressor.compress(chunk))
        size += len(compressor.flush())
        total += size + 100 + 2 * len(name.encode("utf-8"))
    return total


def verify(output, prefix, check_manifest=True):
    receipt = load(output / prefix / "APPEND_RECEIPT.json")
    old = load(output / prefix / "ORIGINAL_V6_FILES.json")
    source = Path(receipt["source_package"])
    require(snapshot(source) == old, "An original package member changed during the CPU append")
    for name, proof in old.items():
        target = output / name
        require(target.is_file() and target.stat().st_size == proof["bytes"] and file_hash(target) == proof["sha256"],
                "A copied original package file differs in bytes: " + name)
    for entry in load(output / prefix / "ADDED_SOURCES.json"):
        require(file_hash(output / entry["package_path"]) == entry["source_sha256"], "An appended source copy differs")
    actual = snapshot(output)
    if check_manifest:
        manifest_path = output / prefix / "PACKAGE_MANIFEST.json"
        manifest = load(manifest_path)
        require({name: proof for name, proof in actual.items() if name != str(manifest_path.relative_to(output))} == manifest["files"],
                "The actual review package does not match its append-only manifest")
    require(receipt["original_members_byte_equal"] and receipt["compressed_zip_upper_bound_bytes"]
            <= receipt.get("max_compressed_bytes", 25000000),
            "The append-only copy or compressed review size target failed")
    return {"passed": True, "copied_old_files": len(old), "copied_old_bytes": sum(r["bytes"] for r in old.values()),
            "actual_package_files": len(actual), "actual_package_bytes": sum(r["bytes"] for r in actual.values()),
            "old_file_or_ID_bytes_changed": 0, "compressed_zip_upper_bound_bytes": receipt["compressed_zip_upper_bound_bytes"],
            "missing_inputs": len(load(output / prefix / "MISSING_INPUTS.json")), "final_zip_created": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", default=str(SOURCE))
    parser.add_argument("--output", default=str(DEST))
    parser.add_argument("--namespace", default="append_core_v7")
    parser.add_argument("--append-plan")
    parser.add_argument("--skip-existing-cpu", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--max-compressed-bytes", type=int, default=25000000)
    args = parser.parse_args()
    require(0 < args.max_compressed_bytes <= 25000000, "The explicit review budget must be at most 25 MB")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Explicitly hide GPUs for this CPU packaging task")
    output = within(ROOT, args.output)
    output.relative_to(BASE)
    require(args.namespace.replace("_", "").isalnum(), "A package namespace must be simple and unique")
    prefix = "appended/" + args.namespace
    if args.verify_only:
        print(json.dumps(verify(output, prefix), ensure_ascii=False))
        return
    source = within(ROOT, args.source_package)
    require(source.is_dir() and source != output and not output.exists(), "The staging destination must be new and exclusive")
    started = time.monotonic()
    old = snapshot(source)
    require(old and "STAGING_MANIFEST.json" in old and "compact/qa_dictionary.parquet" in old,
            "The source package is not an existing complete compact review staging")
    shutil.copytree(source, output)
    require(not (output / prefix).exists(), "This namespace already exists in the original package")
    additions, completion, omitted = [], [], []
    write(output / prefix / "ORIGINAL_V6_FILES.json", old)
    if not args.skip_existing_cpu:
        completion, omitted = append_existing(output, prefix, additions)
    if args.append_plan:
        plan_path = within(ROOT, args.append_plan)
        groups = load(plan_path)["groups"]
        source_copy(plan_path, output, prefix + "/ACTUAL_APPEND_PLAN.json", additions, "explicit accepted/missing source plan")
    else:
        groups = [{"id": name, "status": "missing", "expected_rows": count, "source_path": None,
                   "missing_reason": "Await root-provided accepted actual source and acceptance receipt", "required_input": purpose}
                  for name, count, purpose in FUTURE]
    require(len({g["id"] for g in groups}) == len(groups), "The future append plan repeats a group ID")
    builder = module_from_package(source)
    for group in groups:
        completion.append(append_planned(group, source, output, prefix, additions, builder))
    inherited_missing = {}
    for path in sorted((source / "appended").glob("*/MISSING_INPUTS.json")):
        for entry in load(path):
            inherited_missing[entry["id"]] = {**entry, "inherited_missing_source_path": str(path),
                "inherited_missing_source_sha256": file_hash(path)}
    accepted_states = {"accepted_compact_score_added", "accepted_table_added", "completed_existing_evidence"}
    for path in sorted((source / "appended").glob("*/ACTUAL_COMPLETION.json")):
        for entry in load(path):
            if entry["status"] in accepted_states:
                inherited_missing.pop(entry["id"], None)
    for entry in completion:
        if entry["status"] in accepted_states:
            inherited_missing.pop(entry["id"], None)
        elif entry["status"] == "missing":
            inherited_missing[entry["id"]] = entry
    missing = [inherited_missing[key] for key in sorted(inherited_missing)]
    write(output / prefix / "MISSING_INPUTS.json", missing)
    write(output / prefix / "ACTUAL_COMPLETION.json", completion)
    write(output / prefix / "OMITTED_SOURCE_POINTERS.json", omitted)
    source_copy(Path(__file__), output, prefix + "/scripts/append_core_review.py", additions, "actual CPU append source")
    write(output / prefix / "ADDED_SOURCES.json", additions)
    template = {"groups": [{**r, "kind": "accepted_table" if r["id"] == "frozen_dev_selection20" else "scored_records",
                            "source_sha256": None, "acceptance_receipt_path": None,
                            "acceptance_receipt_sha256": None, "acceptance_checks": {}, "identity_members": [],
                            "unique_key_fields": ["key"], "require_decided_fields": [], "exact_row_fields": {}}
                           for r in missing]}
    write(output / prefix / "FUTURE_APPEND_PLAN_TEMPLATE.json", template)
    completed_lines = []
    for entry in completion:
        if entry["status"] in accepted_states:
            count = entry.get("rows", entry.get("actual_rows"))
            details = "；实际逐样本行数 " + str(count) if count is not None else "；实际范围见 ACTUAL_COMPLETION.json"
            completed_lines.append("- " + entry["id"] + "：" + entry["status"] + details)
    missing_lines = ["- " + entry["id"] + "：待追加已验收来源；登记目标行数 " + str(entry.get("expected_rows"))
                     + "；" + entry.get("required_input", entry.get("missing_reason", "见 MISSING_INPUTS.json")) for entry in missing]
    overview = ("# 本次 CPU 追加审阅证据\n\n命名空间：`" + prefix + "`。来源审阅包：`" + str(source)
        + "`。全部来源包文件和原字典 ID 字节保持；新增资料见本目录 ACTUAL_COMPLETION.json 与 ADDED_SOURCES.json。\n\n"
        + "## 本次实际已追加\n\n" + ("\n".join(completed_lines) if completed_lines else "本次没有新增已完成组。")
        + "\n\n## 当前审阅包待追加输入\n\n" + ("\n".join(missing_lines) if missing_lines else "本轮显式登记的追加输入均已接入。")
        + "\n\n待追加输入仅表示本包缺少实际 accepted 产物；不据此推断服务器实验的当前完成状态。既有 CPU、冻结评分、Luna500/root48 和历史冲突证据通过上述来源包原路径与原 manifest 查询。\n\n"
        + "完整 trace/raw/图像通过 OMITTED_SOURCE_POINTERS.json 或既有源行号定位。scored_records 在独立 compact namespace 直接保存完整问题与全文回复、条件身份、模型身份、配置、随机种子、科学评分和参考字段，并验证与 accepted 原记录等值。完整身份 sidecar 保存在 inputs 下。\n\n"
        + "纯记录 key/qa_key 及来源 digest 字段仅通过 compact receipt 中的 original_source_path、original_source_sha256 和 scores.parquet 中的 1-based source_record_line 查询原服务器记录；逐项集合见 source_only_digest_or_identifier_fields。这些字段没有全部嵌入包内，完整原对象复原需要读取已绑定的服务器源行。既有 v6/v7 文件与字典 ID 保持字节等值。\n\n"
        + "未来按 FUTURE_APPEND_PLAN_TEMPLATE.json 填入实际源 SHA、真实验收检查和完整身份，每次创建新 output/namespace。\n\n"
        + "最新追加与继承缺项以本目录文件为准。未生成最终 ZIP，未修改论文或评分。\n")
    readme = output / prefix / "README.zh.md"
    require(not readme.exists(), "Append README collides with original content")
    readme.write_text(overview, encoding="utf-8")
    current = [(p, str(p.relative_to(output))) for p in sorted(output.rglob("*")) if p.is_file()]
    bound = compressed_bound(current)
    require(bound <= args.max_compressed_bytes, "Actual DEFLATE upper bound exceeds the explicit review byte budget")
    require(snapshot(source) == old and all(file_hash(output / name) == proof["sha256"] for name, proof in old.items()),
            "An original file or dictionary ID changed during the actual append")
    result = {"schema": "kdm_append_core_review_cpu_v7", "passed": True, "source_package": str(source),
              "staging": str(output), "namespace": prefix, "original_files": len(old),
              "original_bytes": sum(r["bytes"] for r in old.values()), "original_members_byte_equal": True,
              "old_dictionary_ID_bytes_changed": 0, "completed_CPU_groups": sum(r["status"] == "completed_existing_evidence" for r in completion),
              "missing_inputs": len(missing), "scientific_completion_claimed_for_missing": False,
              "compressed_zip_upper_bound_bytes": bound, "max_compressed_bytes": args.max_compressed_bytes,
              "compressed_estimate_method": "raw DEFLATE per file plus conservative ZIP32 headers and 128 KiB metadata reserve",
              "final_zip_created": False, "scores_rewritten": False, "GPU_initialized": False,
              "new_generations": 0, "new_API_calls": 0, "papers_modified": False,
              "source_sha256": file_hash(Path(__file__)), "actual_argv": sys.argv, "cpu_python": sys.executable,
              "elapsed_seconds": time.monotonic() - started, "completed_utc": datetime.now(timezone.utc).isoformat()}
    write(output / prefix / "APPEND_RECEIPT.json", result)
    preview = verify(output, prefix, check_manifest=False)
    write(output / prefix / "VERIFICATION.json", {key: value for key, value in preview.items()
          if key not in {"actual_package_files", "actual_package_bytes"}})
    write(output / prefix / "PACKAGE_MANIFEST.json", {"schema": "kdm_append_only_package_member_manifest_v7",
          "source_original_manifest_sha256": old["STAGING_MANIFEST.json"]["sha256"], "files": snapshot(output),
          "final_zip_created": False})
    actual = verify(output, prefix)
    print(json.dumps({**result, "actual_output_verification": actual,
                      "verification_file_sha256": file_hash(output / prefix / "VERIFICATION.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
