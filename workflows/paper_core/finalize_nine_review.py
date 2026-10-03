#!/usr/bin/env python3
"""Combine accepted Food, dev, selected Viz and bounded details without rescoring."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workflows.paper_core.assemble_nine_final_review import ARCHIVES, BASE, DETAILS, DETAILS_SHA, digest, save_json, logical_archive_members

MATH = BASE / "native_baselines/remaining4_original_finite_details_cpu_math_summary_20261003_2328/summary.json"
MATH_SHA = "e50c9097f99b71a6f5f9c01f00b97345186c446fbcd9d5677c5f2c1293f01b0a"
MODELS = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b",
          "internvl35_8b", "onevision", "phi35", "qwen3vl")
COND = ("model", "dataset", "split", "method", "kind", "marker", "reference_marker",
        "guided", "reference_guided", "replicate")
PRIOR_ROOT = {"README.zh.md", "ACTUAL_TASK_STATUS.json", "PACKAGE_RECEIPT.json", "PACKAGE_MANIFEST.json",
              "PACKAGE_MANIFEST.before_import_cache_repair.json"}


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def inside(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    value = value.resolve()
    value.relative_to(ROOT.resolve())
    return value


def copy_bound(source: Path, target: Path, expected: str, provenance: list, purpose: str) -> None:
    require(digest(source) == expected, f"An explicit accepted source changed: {source}")
    require(not target.exists(), f"A destination would be overwritten: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    require(digest(target) == expected, f"Copy differs: {target}")
    provenance.append({"source_path": str(source), "source_sha256": expected,
                       "packaged_path": str(target), "operation": "exact byte copy", "purpose": purpose})


def import_zip(role: str, source: Path, expected: str, package: Path, provenance: list) -> None:
    for row, data in logical_archive_members(role, source, expected):
        name = row["original_member"]
        if role == "Food_v5":
            relative = (Path("prior_snapshot/Food_v5") / name
                        if name in PRIOR_ROOT else Path(name))
        else:
            relative = Path("dev_and_J") / name
        target = package / relative
        require(not target.exists(), f"Input namespaces collide: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        expected_member = hashlib.sha256(data).hexdigest()
        require(digest(target) == expected_member, "Imported member changed")
        provenance.append({"source_archive": str(source), "source_archive_sha256": expected,
                           "source_member": row["replacement_member"], "logical_source_member": name,
                           "packaged_path": str(relative),
                           "source_sha256": expected_member, "operation": "exact member byte copy"})


def import_details(package: Path, provenance: list) -> dict:
    folder = package / "bounded_details_remaining4"
    copy_bound(DETAILS, folder / "completed_only_manifest.json", DETAILS_SHA, provenance, "actual eight completed scopes")
    copy_bound(MATH, folder / "math_summary.json", MATH_SHA, provenance, "accepted saved-pair CPU identities")
    details = load(DETAILS)
    math = load(MATH)
    require(details["passed"] is True and details["source_scopes"] == details["expected_scopes"] == 8,
            "All eight actual bounded scopes are required")
    require(all(row["complete"] is True and row["active"] is False for row in details["observations"]),
            "A bounded scope is not immutable complete")
    require(math["passed"] is True and math["diagnostic_samples"] == 103
            and math["actual_token_candidate_pairs"] == 175 and math["actual_registered_case_inputs"] == 11,
            "Actual diagnostic/case scope differs")
    copy_bound(inside(math["pair_rows"]["path"]), folder / "actual_registered_candidate_pairs.csv",
               math["pair_rows"]["sha256"], provenance, "all 175 actual candidate-pair rows")
    source_count = 0
    for row in details["sources"]:
        for kind, binding in row["bindings"].items():
            source = inside(binding["path"])
            target = folder / row["model"] / row["phase"] / kind / source.name
            copy_bound(source, target, binding["sha256"], provenance, f"{row['model']} {row['phase']} {kind}")
            source_count += 1
    return {"completed_scopes": 8, "selected_inputs": 116, "diagnostic_samples": 103,
            "no_divergence_inputs": 13, "actual_candidate_pairs": 175, "case_inputs": 11,
            "complete_case_paths": sum(row["actual_complete_paths"] for row in details["sources"]),
            "bound_source_files": source_count, "actual_math_source": math}


def validate_dev(package: Path) -> dict:
    folder = package / "dev_and_J"
    receipt = load(folder / "dev_selection/receipt.json")
    main = load(folder / "Food_eval_J/receipt.json")
    require(receipt["passed"] is True and receipt["rows"] == receipt["unique_keys"] == 21008
            and receipt["complete_new_conditions"] == 52 and receipt["pending_rows"] == 0
            and receipt["total_dev_selected_model_methods"] == 18, "The actual dev selection is incomplete")
    require(main["passed"] is True and main["actual_operating_points"] == 69
            and main["IP_dev_selected_points"] == 18 and main["eval_extrema_used"] is False,
            "The current main table is not based on all dev-selected points")
    points = pd.read_csv(folder / "Food_eval_J/main69_actual_operating_points.csv")
    require(len(points) == 69 and set(points.model) == set(MODELS)
            and not points.duplicated(["model", "method"]).any(), "Main operating-point identities differ")
    metrics = pd.read_csv(package / "main/metrics_all_complete.csv").set_index("condition_id")
    for row in points.itertuples(index=False):
        require(row.condition_id in metrics.index, "A selected eval condition is absent from accepted Food")
        source = metrics.loc[row.condition_id]
        for field in ("model", "method", "marker", "C", "W", "A", "TP", "FP", "FN", "J"):
            require(getattr(row, field) == source[field], f"A current main point changed: {field}")
    require(set(metrics.model) == set(MODELS), "The nine checkpoint roster differs")
    names = metrics[["model", "checkpoint"]].drop_duplicates()
    require(len(names) == 9 and names.model.is_unique, "Registered checkpoint names are ambiguous")
    names.to_csv(package / "MODEL_CHECKPOINTS.csv", index=False)
    return {"new_dev_rows": 21008, "new_dev_conditions": 52, "selected_model_methods": 18,
            "actual_main_points": 69, "models": names.to_dict("records"),
            "source_created_utc": receipt["created_utc"]}


def import_viz(source: Path, expected_receipt: str, package: Path, provenance: list) -> dict:
    receipt_path = source / "receipt.json"
    require(digest(receipt_path) == expected_receipt, "The final Viz receipt changed")
    receipt = load(receipt_path)
    require(receipt["passed"] is True and receipt["total_rows"] == 28672 and receipt["total_conditions"] == 56
            and receipt["main_conditions"] == 45 and receipt["mechanism_conditions"] == 11
            and receipt["quality_pending_rows"] == receipt["abstain_pending_rows"] == 0,
            "The final selected Viz panel is not genuinely complete")
    folder = package / "vizwiz_final"
    copy_bound(receipt_path, folder / "receipt.json", expected_receipt, provenance, "actual final Viz source receipt")
    for name in ("new25_metrics.csv", "main45_metrics.csv", "all56_metrics.csv", "mechanism11_metrics.csv",
                 "condition_coverage56.csv", "source_score_receipts.json"):
        copy_bound(source / name, folder / name, receipt["outputs"][name], provenance, "accepted Viz table/source copy")
    score_path = source / "new25_scores.jsonl.gz"
    require(digest(score_path) == receipt["outputs"][score_path.name], "New Viz score objects changed")
    records = []
    keys = set()
    groups = {}
    with gzip.open(score_path, "rt", encoding="utf-8", newline="") as stream:
        for line, original in enumerate(stream, 1):
            record = json.loads(original)
            require(record["key"] not in keys, "New Viz repeats a source key")
            require(record["dataset"] == "vizwiz" and record["quality_score"] is not None
                    and type(record["abstain"]) is bool, "A new Viz decision is unresolved")
            keys.add(record["key"])
            condition = tuple(record[field] for field in COND)
            groups.setdefault(condition, []).append(record)
            records.append({"source_record_line": line, "key": record["key"], "model": record["model"],
                            "sample_id": record["sample_id"], "source_record_line_sha256": hashlib.sha256(original.encode()).hexdigest(),
                            "complete_original_record_json_line": original})
    require(len(records) == len(keys) == 12800 and len(groups) == 25, "New Viz source coverage differs")
    domains = []
    for group in groups.values():
        ids = {row["sample_id"] for row in group}
        require(len(group) == len(ids) == 512 and sum(bool(row["official_reference"]) for row in group) == 166,
                "The real fixed512 official166/346 cohort differs")
        domains.append(ids)
    require(all(ids == domains[0] for ids in domains), "New methods use different Viz inputs")
    old = pd.read_parquet(package / "support/vizwiz/core5_512/new_scores.parquet")
    require(len(old) == 15872, "The old31 actual Viz records are absent")
    table = pa.Table.from_pylist(records)
    parquet_path = folder / "new25_complete_records.parquet"
    pq.write_table(table, parquet_path, compression="zstd", compression_level=12)
    restored = pq.read_table(parquet_path).to_pylist()
    require(restored == records, "The complete Viz source records are not byte-reversible")
    save_json(folder / "columnar_complete_record_receipt.json", {
        "passed": True, "rows": 12800, "source_path": str(score_path),
        "source_sha256": receipt["outputs"][score_path.name], "output_sha256": digest(parquet_path),
        "source_line_and_complete_JSON_bytes_equal": True, "source_fields_dropped": [],
        "old31_payload": "support/vizwiz/core5_512/new_scores.parquet",
        "old31_not_recopied_or_rescored": True,
    })
    for field in ("authority", "ownership", "roster"):
        source_path = inside(receipt[field + "_path"])
        copy_bound(source_path, folder / "source_metadata" / (field + source_path.suffix),
                   receipt[field + "_sha256"], provenance, f"accepted Viz {field} binding")
    for number, binding in enumerate(receipt["new_score_sources"], 1):
        copy_bound(inside(binding["receipt_path"]), folder / "source_metadata" / f"score_receipt_{number:02d}.json",
                   binding["receipt_sha256"], provenance, "accepted constituent Viz score receipt")
    return {"rows": 28672, "conditions": 56, "main_rows": 23040, "main_conditions": 45,
            "mechanism_rows": 5632, "mechanism_conditions": 11, "new_rows": 12800,
            "old_rows": 15872, "quality_pending": 0, "abstain_pending": 0,
            "fixed_roster_n": 512, "official_unanswerable_n": 166, "official_answerable_n": 346,
            "source_completed_utc": receipt["completed_utc"], "receipt_path": str(receipt_path),
            "receipt_sha256": expected_receipt}


def copy_snapshots(values: list[str], package: Path, provenance: list) -> list:
    result = []
    for value in values:
        specification = json.loads(value)
        path = inside(specification["path"])
        name = specification["name"]
        require(PurePosixPath(name).name == name, "Snapshot name must be one filename")
        copy_bound(path, package / "current_source_checkpoints" / name, specification["sha256"], provenance,
                   "explicit final execution or coverage snapshot")
        result.append(specification)
    return result


def import_root_acceptance(package: Path, provenance: list) -> dict:
    path = package / "current_source_checkpoints/FINAL_ACCEPTANCE_ROOT_20261003_2340.json"
    acceptance = load(path)
    require(acceptance["Food_eval_rows"] == 1170792 and acceptance["Food_conditions"] == 483
            and acceptance["Viz_eval_rows"] == 28672 and acceptance["Viz_conditions"] == 56
            and acceptance["dev_selected_IP_configs"] == 18
            and acceptance["all_main_key_score_reference_pending"] == 0,
            "The finite root acceptance differs from current accepted panels")
    for role, binding in acceptance["sources"].items():
        source = inside(binding["path"])
        if source.suffix == ".json":
            copy_bound(source, package / "current_source_checkpoints/root_acceptance_sources" / (role + ".json"),
                       binding["sha256"], provenance, "explicit finite root acceptance source")
    return acceptance


def write_readme(package: Path, status: dict) -> None:
    math = status["bounded_details"]["actual_math_source"]
    maxima = {"finite": max(row["finite_decomposition_closure_max"] for row in math["models"]),
              "IP": max(row["finite_IP_difference_closure_max"] for row in math["models"])}
    limitations = "\n".join("- " + value for value in status["root_acceptance"]["limitations"])
    (package / "README.zh.md").write_text(f"""# KDM 九模型完整结果审阅包

本包根据已接受的 Food、开发集选择、Viz512 和有限归因结果复制整理。实际构包 UTC 为 {status['created_utc']}；输入结果自己的实际时间保存在各原回执，文件名仅用于定位。

## 当前主结果

Food eval 包含 483 条件、1,170,792 条正式评分。主比较为 123 条件、298,152 条；每条件固定 2,424 图像、101 类各 24。其余 360 条件、872,640 条用于注册机制矩阵。完整原问答、逐样本评分、参考、条件、配置、种子、身份及来源字典均保留。九个登记 checkpoint 名称见 MODEL_CHECKPOINTS.csv。

当前主表是 [九模型 J 比较](dev_and_J/Food_eval_J/J_main_comparison9.csv) 与 [69 个实际工作点](dev_and_J/Food_eval_J/main69_actual_operating_points.csv)：51 个登记原生/控制基线，以及九模型实际 dev-selected 的 18 个 IP-VCD/IP-M3ID 条件。每个 IP 条件的全部 Acc/P/R/F1/J 分子分母来自同一保存的 eval 工作点。J=(正确作答 C+参考支持弃权 TP)/N；零分母按原表保留空值。

新增开发集候选实际完成 52 条件、21,008 条、每条件 dev404；保留五个原 IP-VCD 冻结选择，实际总计 18 个模型—方法选择。开发资料见 dev_and_J/dev_selection，旧登记开发记录见 support/dev_selection。原 [eval best_observed](main/best_observed_joint_operating_points.csv) 和 [各指标独立极值](main/best_observed_independent_endpoints.csv) 单列。figures 中继承的图示对应原 best_observed 工作点，其数值未改。

## VizWiz 第二任务

实际已闭合 56×512=28,672 条，主条件 45 个、机制条件 11 个。每条件官方可回答 346、不可回答 166。见 [主表](vizwiz_final/main45_metrics.csv)、[全部条件](vizwiz_final/all56_metrics.csv)、[真实覆盖](vizwiz_final/condition_coverage56.csv)。主质量沿用实际 annotated answer_text 的官方连续质量；行为弃权主质量为零，完整回复的 official raw 分数独立保存。没有将连续分数二元化或改用 Food 参考。

旧五模型 31 条件、15,872 条原 parquet 与来源保持 support/vizwiz/core5_512。新增 25 条件、12,800 条完整原始评分对象存于 vizwiz_final/new25_complete_records.parquet；complete_original_record_json_line 可逐行恢复所有原字段和 JSON 字节，原评分文件、SHA、1-based 行号由邻接回执绑定。没有重复嵌入旧31数据。

## 有限四路归因

原五模型已有 814 首位/诊断位置、1,049 候选对、24 完整路径保留原包资料。扩展四模型本次八个实际完成 scope 独立保存在 bounded_details_remaining4：116 选定输入中实际分岔诊断 103，13 个无分岔保持该状态；175 对实际候选、11 个真实案例和22完整路径。空 strata 和未有的案例类型由原来源如实记录。

保存的有限四项数学最大闭合残差 {maxima['finite']:.17g}，IP 差最大残差 {maxima['IP']:.17g}。actual_registered_candidate_pairs.csv、math_summary.json 保留原支持集、共同支持、实际 argmax 与交互代理。词元边际和 argmax 代理的适用范围是保存前缀，完整回答行为另由真实语义评分记录。named-six 来源、原回答 tokens/text、预算、身份、准入和 complete 证明都按原 SHA 复制。

## 来源与读取

prior_snapshot/Food_v5 保留 v5 的 README、实际状态和原包回执；这些是历史输入说明。当前覆盖见 FINAL_ACTUAL_STATUS.json 和 current_source_checkpoints 中的明确源。旧标签冲突隔离及实际 Luna/root/provider 证据保留原目录和追加来源，不重新评分或生成。

FINAL_PACKAGE_MANIFEST.json 逐文件记录 SHA、字节数和来源；相同 SHA 的原图/数据仅在 ZIP 中存储一次，IDENTICAL_ASSET_ALIASES.json 保留所有逻辑路径。解压后运行 `python scripts/verify_final_nine_review.py PACKAGE --materialize` 可按明确相同 SHA 映射恢复重复路径并验收所有文件。完整独占 staging 中已保留每个物理文件。科学字段、原字典 ID、完整 QA 和来源未删减。

科学范围：Food 的评价与选择分别使用注册 eval2424/dev404；Viz 当前结果限注册512面板；有限归因按实际选定集合和保存前缀报告。完整方法主结果、机制矩阵、病例路径与代理测量各自保留分母和来源。

## root 有限验收与科学缺项

[独立 root 验收](current_source_checkpoints/FINAL_ACCEPTANCE_ROOT_20261003_2340.json) 保存原始完整裁定与来源。Viz 主表每模型五个方法；原五模型 CDA 使用冻结选定配置，扩展四模型 CDA 使用原始 UNKNOWN，精确配置身份保持原表。Food 的注册方法集合单独保存。原 UNKNOWN 有限机制测量与 dev-selected marker 主结果分别报告。原 CDA Eq4 全向量熵未保存，现有权重结果照原数据保留；历史九项 replay 的来源未知字段也保持原值。

root 原始适用范围与缺项逐项保存：

{limitations}

实际构包和独立验证源码见 scripts/final_review_build_source；完整复制路径、源 SHA、命令和字节验收记录保存在本包 manifest 与外部 BUILD_RECEIPT.json。
""", encoding="utf-8")


def alias_plan(files: dict) -> dict:
    candidates = {}
    aliases = {}
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".pdf", ".parquet", ".csv", ".gz"}
    ordered = sorted(files, key=lambda name: (0 if name.startswith(("main/", "figures/", "support/")) else 1, name))
    for name in ordered:
        if Path(name).suffix.lower() not in suffixes:
            continue
        content = files[name]["sha256"]
        if content in candidates:
            aliases[name] = {"canonical_path": candidates[content], "sha256": content, "bytes": files[name]["bytes"]}
        else:
            candidates[content] = name
    return aliases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viz-dir", required=True)
    parser.add_argument("--viz-receipt-sha", required=True)
    parser.add_argument("--snapshot", action="append", default=[], help="Explicit JSON {name,path,sha256}")
    parser.add_argument("--snapshot-plan", help="Explicit JSON list of {name,path,sha256} source files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = inside(args.output)
    output.relative_to(BASE.resolve())
    require(not output.exists(), "Use an exclusive final staging directory")
    output.mkdir(parents=True)
    package = output / "package"
    package.mkdir()
    provenance = []
    for role, source, expected, _ in ARCHIVES:
        import_zip(role, source, expected, package, provenance)
    details = import_details(package, provenance)
    dev = validate_dev(package)
    viz = import_viz(inside(args.viz_dir), args.viz_receipt_sha, package, provenance)
    specifications = list(args.snapshot)
    if args.snapshot_plan:
        snapshot_plan = load(inside(args.snapshot_plan))
        require(type(snapshot_plan) is list, "The source snapshot plan must be an explicit list")
        specifications.extend(json.dumps(record, ensure_ascii=False) for record in snapshot_plan)
    snapshots = copy_snapshots(specifications, package, provenance)
    root_acceptance = import_root_acceptance(package, provenance)
    old = load(package / "prior_snapshot/Food_v5/PACKAGE_RECEIPT.json")
    require(old["passed"] is True and old["Food_full_conditions"] == 483 and old["Food_full_rows"] == 1170792,
            "The accepted full Food input panel differs")
    status = {"schema": "kdm_nine_final_review_actual_status_v1", "passed": True,
              "created_utc": datetime.now(timezone.utc).isoformat(), "Food_conditions": 483,
              "Food_rows": 1170792, "Food_main_conditions": 123, "Food_main_rows": 298152,
              "Food_mechanism_conditions": 360, "Food_mechanism_rows": 872640,
              "dev": dev, "Viz": viz, "bounded_details": details, "source_snapshots": snapshots,
              "root_acceptance": root_acceptance,
              "semantic_labels_rescored": 0, "new_GPU_or_generation": 0,
              "file_names_used_as_actual_timestamps": False, "frozen_inputs_modified": False}
    save_json(package / "FINAL_ACTUAL_STATUS.json", status)
    write_readme(package, status)
    tools = package / "scripts"
    copy_bound(Path(__file__).with_name("verify_final_nine_review.py"), tools / "verify_final_nine_review.py",
               digest(Path(__file__).with_name("verify_final_nine_review.py")), provenance, "standalone final verifier")
    for name in ("finalize_nine_review.py", "assemble_nine_final_review.py", "verify_final_nine_review.py"):
        source = Path(__file__).with_name(name)
        copy_bound(source, tools / "final_review_build_source" / name, digest(source), provenance,
                   "actual final construction/verification source bytes")
    save_json(package / "FINAL_SOURCE_COPY_INDEX.json", provenance)
    files = {str(path.relative_to(package)).replace("\\", "/"):
             {"sha256": digest(path), "bytes": path.stat().st_size}
             for path in sorted(package.rglob("*")) if path.is_file()}
    aliases = alias_plan(files)
    save_json(package / "IDENTICAL_ASSET_ALIASES.json", aliases)
    manifest = {"schema": "kdm_nine_final_review_package_manifest_v1", "files": files,
                "logical_file_count": len(files), "source_archives": [
                    {"path": str(path), "sha256": expected} for _, path, expected, _ in ARCHIVES],
                "byte_equal_original_source_members": sum("source_member" in row for row in provenance),
                "all_input_dictionary_IDs_preserved": True, "all_scientific_fields_preserved": True,
                "alias_file": "IDENTICAL_ASSET_ALIASES.json", "aliases_sha256": digest(package / "IDENTICAL_ASSET_ALIASES.json"),
                "runner_sha256": digest(Path(__file__)), "created_utc": status["created_utc"]}
    save_json(package / "FINAL_PACKAGE_MANIFEST.json", manifest)
    archive = output / "KDM_Nine_Complete_Review.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                relative = str(path.relative_to(package)).replace("\\", "/")
                if relative not in aliases:
                    target.write(path, relative)
    with zipfile.ZipFile(archive) as final:
        require(final.testzip() is None, "Final archive CRC failed")
        for name, binding in files.items():
            actual_name = aliases[name]["canonical_path"] if name in aliases else name
            require(hashlib.sha256(final.read(actual_name)).hexdigest() == binding["sha256"],
                    f"A logical ZIP member differs: {name}")
    receipt = {"schema": "kdm_nine_final_review_build_execution_v1", "passed": True,
               "ZIP_path": str(archive), "ZIP_sha256": digest(archive), "ZIP_bytes": archive.stat().st_size,
               "target_bytes": 25000000, "within_target_size": archive.stat().st_size < 25000000,
               "logical_files": len(files), "identical_assets_deduplicated": len(aliases),
               "alias_uncompressed_bytes": sum(row["bytes"] for row in aliases.values()),
               "all_original_source_members_byte_equal": True,
               "new25_source_rows_byte_reversible": True, "Food_full_rows": 1170792,
               "Food_main_conditions": 123, "Food_main_dev_selected_workpoints": 69,
               "Viz_actual_conditions": 56, "Viz_actual_rows": 28672,
               "Viz_main_conditions": 45, "Viz_mechanism_conditions": 11,
               "bounded_completed_scopes": 8, "raw_score_or_paper_modified": False,
               "new_scoring_or_GPU_generation": False, "actual_command": sys.argv,
               "runner_sha256": digest(Path(__file__)), "completed_utc": datetime.now(timezone.utc).isoformat()}
    save_json(output / "BUILD_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
