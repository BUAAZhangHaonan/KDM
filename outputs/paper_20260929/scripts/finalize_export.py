"""Finalize the existing-data paper export using exported tables and source locators.

This performs CPU serialization and count checks. It reads frozen scores as flags,
never invokes scoring, model inference, or annotation.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/paper_20260929"
KEY = ("model", "method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def key(row):
    return tuple(
        str(row[k]).lower() == "true" if k in ("guided", "reference_guided")
        else int(row[k]) if k == "replicate" else row[k]
        for k in KEY
    )


def finalize():
    checks = {"kind": "CPU checks of serialized export, existing-key joins, and frozen count comparisons", "executed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "checks": [], "discrepancies": []}

    def check(name, actual, expected):
        if hasattr(actual, "item"):
            actual = actual.item()
        ok = actual == expected
        record = {"name": name, "actual": actual, "expected": expected, "passed": ok}
        checks["checks"].append(record)
        if not ok:
            checks["discrepancies"].append(record)

    table_checks = read_json(OUT / "table_checks.json")
    runtime_checks = read_json(OUT / "runtime_checks.json")
    case_checks = read_json(OUT / "case_checks.json")
    conditions = read_csv(OUT / "conditions.csv")
    prompt_config = read_json(OUT / "prompts_and_configs.json")
    prompt_conditions = {key(r): r for r in prompt_config["conditions"]}
    for row in conditions:
        binding = prompt_conditions[key(row)]
        check(f"prompt/config condition {row['condition_id']}", int(row["condition_id"]), binding["condition_id"])
        row["prompt_id"] = binding["prompt_ids"][0] if len(binding["prompt_ids"]) == 1 else None
        row["config_id"] = binding["config_ids"][0] if len(binding["config_ids"]) == 1 else None
        row["prompt_ids"] = json.dumps(binding["prompt_ids"])
        row["config_ids"] = json.dumps(binding["config_ids"])
    write_csv(OUT / "conditions.csv", conditions)

    # Add a direct raw-text locator to every deduplicated QA without reading raw files.
    database = sqlite3.connect(f"file:{OUT / 'work/index.sqlite'}?mode=ro", uri=True)
    qa_locations = {
        int(row[0]): row[1:] for row in database.execute(
            "SELECT qa_id, model, source_file_id, source_line FROM "
            "(SELECT qa_id,model,source_file_id,source_line,"
            "ROW_NUMBER() OVER (PARTITION BY qa_id ORDER BY condition_id,sample_id) AS ordinal FROM responses) "
            "WHERE ordinal=1"
        )
    }
    qa_rows = pq.read_table(OUT / "qa.parquet").to_pylist()
    for row in qa_rows:
        model, source_id, line = qa_locations[row["qa_id"]]
        row.update(raw_model=model, raw_source_file_id=source_id, raw_source_line=line)
    pq.write_table(pa.Table.from_pylist(qa_rows), OUT / "qa.parquet", compression="zstd")
    check("QA direct raw locators", len(qa_locations), len(qa_rows))

    sources = read_csv(OUT / "sources.csv")
    existing_source_paths = {r["real_path"] for r in sources}
    next_source_id = max(int(r["source_file_id"]) for r in sources) + 1
    for segment in runtime_checks["sources"]:
        archived = segment["source_archive_path"]
        if archived in existing_source_paths:
            continue
        sources.append({
            "source_file_id": next_source_id, "role": "original_formal_segment",
            "project_relative_path": segment["source_path"], "real_path": archived,
            "bytes": segment.get("source_recorded_bytes"), "known_rows": segment["expected_rows"],
            "exists": True, "existence_basis": "existing manifest and source-bound execution metadata",
            "runtime_source_id": segment["source_id"], "consolidated_path": segment["consolidated_path"],
            "consolidated_first_line": segment["segment_first_line"],
            "consolidated_last_line": segment["segment_last_line"],
        })
        next_source_id += 1
        existing_source_paths.add(archived)
    write_csv(OUT / "sources.csv", sources)

    # Work on compact flags only. The long responses remain in the case/QA files.
    columns = ["condition_id", "sample_id", "target_class", "correct_canonical", "correct_literal", "abstain", "uniform_reference", "accepted_reference", "source_file_id", "source_line", "qa_id", "detail_id", "binding_id", "behavior", "score_reason"]
    scores = pq.read_table(OUT / "scores.parquet", columns=columns).to_pandas()
    check("scores parquet rows", len(scores), 853248)
    check("scores conditions", scores.condition_id.nunique(), 352)
    check("unique condition/sample keys", int(scores.duplicated(["condition_id", "sample_id"]).sum()), 0)
    for field in ("correct_canonical", "correct_literal", "abstain", "uniform_reference", "accepted_reference"):
        check(f"decided {field}", int(scores[field].isna().sum()), 0)
    check("rows per condition", sorted(scores.groupby("condition_id").size().unique().tolist()), [2424])
    check("unique samples per condition", sorted(scores.groupby("condition_id").sample_id.nunique().unique().tolist()), [2424])
    check("classes per condition", sorted(scores.groupby("condition_id").target_class.nunique().unique().tolist()), [101])
    check("inputs per condition/class", sorted(scores.groupby(["condition_id", "target_class"]).size().unique().tolist()), [24])
    check("condition split values", sorted({r["split"] for r in conditions}), ["eval"])
    metrics = {key(r): r for r in read_csv(OUT / "baseline_tables/condition_metrics.csv")}
    score_groups = {int(cid): frame.set_index("sample_id") for cid, frame in scores.groupby("condition_id")}
    for row in conditions:
        group = score_groups[int(row["condition_id"])]
        expected = metrics[key(row)]
        for target, source in (("correct_canonical", "canonical"), ("correct_literal", "literal")):
            positive = int(group[target].sum())
            check(f"condition {row['condition_id']} {source} correct", positive, int(expected[f"{source}_correct"]))
            check(f"condition {row['condition_id']} {source} incorrect", len(group)-positive, int(expected[f"{source}_incorrect"]))
        check(f"condition {row['condition_id']} abstain", int(group.abstain.sum()), int(expected["abstain_true"]))

    references = pq.read_table(OUT / "references.parquet").to_pandas()
    check("reference rows", len(references), 24240)
    check("unique reference model/sample keys", int(references.duplicated(["model", "sample_id"]).sum()), 0)
    condition_models = {int(r["condition_id"]): r["model"] for r in conditions}
    joined = scores[["condition_id", "sample_id", "uniform_reference", "accepted_reference"]].copy()
    joined["model"] = joined.condition_id.map(condition_models)
    joined = joined.merge(references[["model", "sample_id", "split", "uniform_reference", "accepted_reference"]], on=["model", "sample_id"], how="left", suffixes=("", "_reference"), validate="many_to_one")
    check("reference joins", int(joined.split.notna().sum()), 853248)
    check("reference joined eval rows", int((joined.split == "eval").sum()), 853248)
    for reference in ("uniform_reference", "accepted_reference"):
        check(f"preserved {reference}", int((joined[reference] != joined[reference+"_reference"]).sum()), 0)
    checks["reference_split_counts"] = {str(k): int(v) for k, v in references.groupby("split").size().items()}
    del joined

    controls = pq.read_table(OUT / "control_selections.parquet").to_pandas()
    check("control selections rows", len(controls), 484800)
    check("control conditions", controls.control_condition_id.nunique(), 200)
    check("control condition/sample uniqueness", int(controls.duplicated(["control_condition_id", "sample_id"]).sum()), 0)
    selected = controls.merge(scores[["condition_id", "sample_id", "source_file_id", "source_line", "qa_id", "abstain"]], left_on=["selected_condition_id", "sample_id"], right_on=["condition_id", "sample_id"], how="left", suffixes=("", "_selected"), validate="many_to_one")
    check("controls joined to formal scores", int(selected.condition_id.notna().sum()), 484800)
    check("control selected source files", int((selected.selected_source_file_id != selected.source_file_id).sum()), 0)
    check("control selected source lines", int((selected.selected_source_line != selected.source_line_selected).sum()), 0)
    check("control selected QA", int((selected.qa_id != selected.qa_id_selected).sum()), 0)
    check("control selected abstain", int((selected.selected_abstain != selected.abstain).sum()), 0)
    del selected

    # Retain the uploaded paper decisions and compare all 40 existing paired points.
    context = OUT / "context"
    with zipfile.ZipFile(context / "KDM_Paper_Start_20260929.zip") as package:
        for name in package.namelist():
            relative = Path(name).relative_to("KDM_Paper_Start_20260929")
            if relative.as_posix() in {"01_论文主线与结果决策.md", "04_CODEX_原会话导出提示词.md", "context/用户科研写作要求_原文.md", "checks/source_mapping.json", "tables/instruction_pairs_all40.csv"}:
                target = context / "paper_start" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(package.read(name))
        name = next(n for n in package.namelist() if n.endswith("/tables/instruction_pairs_all40.csv"))
        uploaded_pairs = list(csv.DictReader(io.StringIO(package.read(name).decode("utf-8-sig"))))
    check("uploaded paper pair rows", len(uploaded_pairs), 40)
    points = {int(p["source_line"]): p for p in table_checks["instruction_pair_point_checks"]}
    paired_rows = []
    for row in uploaded_pairs:
        point = points[int(row["source_jsonl_line"])]
        a = score_groups[point["condition_a"]].sort_index()
        b = score_groups[point["condition_b"]].reindex(a.index)
        direct = score_groups[point["direct_condition_id"]].reindex(a.index)
        corrections = b.correct_canonical & ~direct.correct_canonical
        specific = corrections & ~direct.abstain
        values = {
            "accuracy_delta_pp": float((a.correct_canonical.astype(int)-b.correct_canonical.astype(int)).sum())*100/len(a),
            "all_input_corrections_retained": int((corrections & a.correct_canonical).sum()),
            "all_input_corrections_total": int(corrections.sum()),
            "specific_answer_corrections_retained": int((specific & a.correct_canonical).sum()),
            "specific_answer_corrections_total": int(specific.sum()),
            "baseline_correct_retained": int((b.correct_canonical & a.correct_canonical).sum()),
            "baseline_correct_lost": int((b.correct_canonical & ~a.correct_canonical).sum()),
            "baseline_correct_unknown_in_a": 0,
            "new_correct_vs_baseline": int((~b.correct_canonical & a.correct_canonical).sum()),
        }
        for short, field in (("uniform", "uniform_reference"), ("accepted", "accepted_reference")):
            original = direct.abstain & direct[field]
            numerator = int((original & a.abstain).sum()-(original & b.abstain).sum())
            denominator = int(original.sum())
            values[f"{short}_retention_net_count"] = numerator
            values[f"{short}_original_supported"] = denominator
            values[f"{short}_retention_delta_pp"] = numerator*100/denominator if denominator else None
        comparisons = {}
        for field, actual in values.items():
            text = row[field]
            expected = float(text) if text not in ("", "None", "nan") else None
            ok = actual is None and expected is None or actual is not None and expected is not None and abs(actual-expected) < 1e-9
            comparisons[field] = {"actual": actual, "uploaded": expected, "passed": ok}
            if not ok:
                checks["discrepancies"].append({"source_jsonl_line": point["source_line"], "field": field, "actual": actual, "uploaded": expected})
        paired_rows.append({"source_jsonl_line": point["source_line"], "condition_a": point["condition_a"], "condition_b": point["condition_b"], "direct_condition_id": point["direct_condition_id"], "values": comparisons})
    checks["uploaded_40_pair_points"] = paired_rows
    check("uploaded pair point comparisons", sum(len(r["values"]) for r in paired_rows), 600)

    protocol_dir = context / "final_project"
    protocol_dir.mkdir(exist_ok=True)
    for relative in ("docs/SCORING.md", "docs/PROTOCOL.md", "docs/PROTOCOL_REGISTER.md", "docs/RESULTS.md", "docs/REPRODUCIBILITY.md", "data/manifest.json", "data/responses/formal/manifest.json", "configs/kdm/models.json", "configs/kdm/study.json", "configs/kdm/method_plan.json", *(f"configs/runtime/{m}.json" for m in ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b"))):
        source = ROOT / relative
        if source.is_file():
            target = protocol_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    review_package = ROOT.parent / "knowledge-deficit-mitigation-archive/20260929/review_package/KDM_Review.zip"
    if review_package.exists():
        with zipfile.ZipFile(review_package) as package:
            for name in package.namelist():
                if name.endswith("LARGE_ASSETS.json"):
                    (context / "LARGE_ASSETS.json").write_bytes(package.read(name))
                    break

    cases = [json.loads(line) for line in (OUT / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    for case in cases:
        case["image"]["package_path"] = "cases/images/" + Path(case["image"]["copied_path"]).name
    (OUT / "cases.jsonl").write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")

    availability = []
    for row in read_csv(OUT / "runtime_availability.csv"):
        availability.append({"category": "runtime_or_cache", "details_json": json.dumps(row, ensure_ascii=False)})
    for row in read_csv(OUT / "case_availability.csv"):
        availability.append({"category": "case", "details_json": json.dumps(row, ensure_ascii=False)})
    for row in table_checks["field_availability"]:
        availability.append({"category": "stored_score_field", "details_json": json.dumps(row, ensure_ascii=False)})
    availability += [
        {"category": "behavior_granularity", "details_json": json.dumps({"final_score_behavior_counts": scores.behavior.value_counts(dropna=False).to_dict(), "existing_auxiliary_behavior_counts": table_checks["stored_behavior_labels"], "location": "scores.behavior and score_details.stored_behavior_label", "meaning": "Frozen values preserved. Auxiliary null labels remain null; score.abstain is the final decided flag."}, ensure_ascii=False)},
        {"category": "QA_text", "details_json": json.dumps({"available_QA": 14134, "total_QA": len(qa_rows), "raw_locators": len(qa_locations), "location": "qa.parquet raw_source_file_id/raw_source_line", "meaning": "Review text is present for 14134 unique QA. Every QA has a direct original-response locator; all 12 cases include full responses."}, ensure_ascii=False)},
    ]
    if (OUT / "runtime_comparisons.csv").exists():
        runtime_pairs = read_csv(OUT / "runtime_comparisons.csv")
        checks["runtime_comparisons"] = {"rows": len(runtime_pairs), "available": sum(r["availability"] == "available" for r in runtime_pairs), "unavailable": sum(r["availability"] != "available" for r in runtime_pairs)}
        runtime_by_id = {int(r["runtime_record_id"]): r for r in read_csv(OUT / "runtime_records.csv")}
        compared_ids = set()
        ratio_differences = 0
        for row in runtime_pairs:
            for side in ("vcd", "ip_vcd"):
                if row[f"{side}_runtime_record_id"]:
                    compared_ids.add(int(row[f"{side}_runtime_record_id"]))
            if row["availability"] == "available":
                vcd = runtime_by_id[int(row["vcd_runtime_record_id"])]
                ip = runtime_by_id[int(row["ip_vcd_runtime_record_id"])]
                expected = float(ip["wall_s_sum"]) / float(vcd["wall_s_sum"])
                ratio_differences += abs(expected-float(row["ip_over_vcd_wall_s_ratio"])) > 1e-12
            if row["availability"] != "available":
                availability.append({"category": "runtime_comparison", "details_json": json.dumps(row, ensure_ascii=False)})
        check("runtime comparison ratios from recorded wall_s", ratio_differences, 0)
        check("runtime comparison original-record coverage", len(compared_ids), sum(r["method"] in ("vcd", "instruction_vcd") for r in runtime_by_id.values()))
    write_csv(OUT / "availability.csv", availability)
    checks["table_export"] = table_checks
    checks["runtime_export"] = {k: runtime_checks[k] for k in ("formal_rows", "conditions", "runtime_records", "sum_runtime_rows", "wall_s_observed_rows", "tokens_observed_rows", "input_content_hashes_computed", "formal_full_scans")}
    checks["case_export"] = case_checks
    checks["source_files_count"] = len(sources)
    checks["passed"] = not checks["discrepancies"] and table_checks["passed"] and case_checks["status"] == "complete"
    write_json(OUT / "export_checks.json", checks)

    schemas = []
    for path in sorted(OUT.glob("*.parquet")):
        fields = pq.read_schema(path)
        schemas.append(f"### {path.name}\n\n| Field | Arrow type |\n|---|---|\n" + "\n".join(f"| {field.name} | `{field.type}` |" for field in fields))
    schema_text = """# Export schema and source mapping

All line locators are 1-based lines in the decompressed JSONL. Integer dictionary IDs are local to this export. Strings and existing semantic values are copied verbatim; Parquet Boolean columns use true/false and retain native nulls when present. CSV empty cells denote unavailable fields, and CSV list/object cells use JSON.

## Main table mapping

`scores.parquet` primary key: `(condition_id, sample_id)`. Join `condition_id` to `conditions.csv` for model and split. The condition key is the exact tuple `(model, method, kind, marker, reference_marker, guided, reference_guided, replicate)`, sorted to assign IDs. `marker` is also exported as `main_marker`. Each of the 352 conditions has 2,424 eval samples.

| Export field | Frozen field/source |
|---|---|
| correct_canonical | score_rows.canonical_name_in_primary_score (0/1 to bool) |
| correct_literal | score_rows.literal_extracted_name_score (0/1 to bool) |
| abstain | score_rows.abstain |
| uniform_reference | uniform_reference_gt.gt joined on model/sample_id |
| accepted_reference | reference_gt.gt joined on model/sample_id |
| sample_id, target_class, seed, behavior, score_reason | same-named score_rows fields |
| source_file_id / source_line | dictionary encoding of score_rows.source_path / existing source_line |
| score_source_line | line in the frozen score_rows.jsonl.gz |
| qa_id | dictionary encoding of score_rows.qa_key |
| detail_id | dictionary of the copied extraction/review fields plus matched existing automatic_behavior fields |
| binding_id | dictionary of existing main/reference/neutral prompt and config source hashes |

`qa.parquet` retains existing qa_key, exact review question/answer and review variants when recorded. `raw_model/raw_source_file_id/raw_source_line` always locate one original formal response for that QA. Review text may be null. The complete cases include every requested full answer. To recover any other full answer, read the located line from the gzip listed in sources.csv. Its existing `key` is the original response key. One QA can occur in many scored rows with different targets; correctness remains per score row.

`score_details.parquet` uses JSON strings for each original field so lists, strings, null, false and empty lists keep their distinct values. Apply `json.loads` per cell. `canonical_name_in_primary` and literal extraction can be null even when the separate frozen score is decided. `stored_behavior_*` holds the auxiliary automatic_behavior file's original values and source; final `scores.abstain/behavior` remains the paper scoring source.

`conditions.prompt_id/config_id` connects to the arrays in prompts_and_configs.json. `prompt_ids/config_ids` preserves all actual variants per condition; the present export has one of each per condition. The JSON stores actual main/reference/neutral prompt text, complete DecodeConfig values and first/last source lines. `bindings.json` retains already-recorded hashes and locators.

## References and controls

`references.parquet` key: `(model, sample_id)`; split is retained for all 24,240 dev/eval rows. `gold_rank` maps to uniform gold_rank; `independent_correct_attempts` to uniform independent_correct_attempts_exact; `independent_attempts` to uniform attempts. Both source records are additionally retained verbatim as `uniform_original_json` and `accepted_original_json`, with separate source-file IDs/lines. Main retention denominator is fixed to the condition's Direct-abstain AND uniform_reference set. Historical accepted-reference results use the separate accepted field.

`control_conditions.csv` defines 200 IDs in a separate control namespace. `control_selections.parquet` key: `(control_condition_id,sample_id)`. Join `(selected_condition_id,sample_id)` to scores to obtain the frozen selected response. `base_condition_id` is the treatment condition and `direct_condition_id` its original Direct. The selected file/line and QA are retained. `source_line` here is the original selections.jsonl.gz line. No new selection rule is executed.

## Sources, runtime, cases and checks

`sources.csv.source_file_id` is the global export dictionary. Runtime's historical `source_id` is a separate 21-segment namespace bridged by `sources.runtime_source_id`. The current consolidated gzip paths are active sources. Archived segments use `archive.tar::member` notation, with preserved consolidated line ranges. Paths containing older host names describe historical records and were read from the central archive.

`runtime_records.csv` records 1,147 condition/source/identity groups. n_rows, wall_s_sum/mean/min/max and tokens_sum/mean/min/max come from saved per-response wall_s and len(tokens). Source ranges are filtered by the explicit model/condition/identity in each record. wall_s spans sample image loading, session generation and cleanup; source model initialization and ledger writing sit outside this timer. The source has no separate prefill/decode timings or peak memory fields. `runtime_availability.csv` and `availability.csv` preserve field-specific availability and JSON-key sources. Runtime comparison rows, when present, explicitly identify matching device/configuration/input sets and formulas. Existing scalar caches and their sizes/coverage are listed for later selection.

`cases.jsonl` includes 12 original inputs, Direct/VCD/IP-VCD/copied-control full response records, all ten saved independent attempts, reference values and image paths/dimensions. `cases/images/` contains byte-identical source images. `case_group_availability.csv` records all eligible counts and selected counts for the stated frozen-flag combinations. Group labels describe those combinations; the behavior values remain the original labels.

`baseline_tables/` copies the existing 352-condition metrics, 372 paired comparisons and control summaries. `context/paper_start/tables/instruction_pairs_all40.csv` is copied from the uploaded paper package. Confidence intervals are retained from those existing files. `export_checks.json` records CPU row/key/count/join checks and comparison of all 40 point estimates. The executed export scripts are in scripts/.

## Arrow schemas

"""
    (OUT / "schema.md").write_text(schema_text + "\n\n".join(schemas) + "\n", encoding="utf-8")

    script_out = OUT / "scripts"
    script_out.mkdir(exist_ok=True)
    for source in sorted((ROOT / "scripts/paper_20260929").glob("*.py")):
        shutil.copyfile(source, script_out / source.name)
    payload_files = [p for p in OUT.rglob("*") if p.is_file() and "work" not in p.relative_to(OUT).parts and p.name not in ("version_before.json", "git_inventory.md") and not p.name.startswith("KDM_Paper_Data_Export")]
    total = sum(p.stat().st_size for p in payload_files)
    (OUT / "README.md").write_text(f"""# KDM Paper Data Export — 2026-09-29

论文分析入口为 scores.parquet + conditions.csv。所有评分、参考值、控制选择和回答来自当前冻结资产；本次执行CPU格式转换、已有键连接、计数核对及图片复制。

| 内容 | 已导出数量 |
|---|---:|
| 正式逐样本评分 | 853,248 |
| 完整正式条件 | 352，每条件2,424个eval输入 |
| 参考记录 | 24,240，dev/eval各12,120 |
| 复制控制 | 200条件，484,800条既有选择 |
| 去重QA / 评分细节字典 | 44,782 / 8,222 |
| 原始真实案例 / 图片 | 12 / 12 |
| 案例独立试答 | 120 |
| 运行耗时记录 | 1,147组，覆盖853,248条回答 |
| 已有配对比较 / 本包40组点估计核对 | 372 / 40 |

当前数据文件合计约 {total/1_000_000:.2f} MB（未压缩，最终压缩体积见version_receipt/package receipt）。导出检查结论：{'通过，计数差异0' if checks['passed'] else '具体差异见export_checks.json'}。

正文核心计数：LLaVA/UNKNOWN的Direct、VCD、IP-VCD正确数689、852、852；uniform原有集合287，保留33、61。MiniCPM/UNCLEAR对应733、873、903；uniform原有集合374，保留239、249。MiniCPM的VCD纠正226个，IP-VCD保留212个，复制控制正确722个。accepted历史参考字段分别保留在表中。

已有字段覆盖：全部主正确性、字面正确性和弃权值已决。QA表保存14,134条已有复核全文，其余QA提供直接原始文件/行号；12个案例含完整原文。既有抽取主名称有779,420行null，直接保留。辅助行为表有334行空标签，最终score.abstain字段完整。已有耗时覆盖全体回答，细分prefill/decode、峰值显存和batch size的可用情况逐项列于availability.csv。sources.csv提供当前路径与历史归档映射。

## 读取

```python
import pandas as pd
s = pd.read_parquet('scores.parquet')
c = pd.read_csv('conditions.csv')
r = pd.read_parquet('references.parquet')
wide = s.merge(c, on='condition_id', validate='many_to_one')
# 方法配对使用同一个sample_id以及完整匹配的条件。
sel = pd.read_parquet('control_selections.parquet')
control_rows = sel.merge(s, left_on=['selected_condition_id','sample_id'],
                        right_on=['condition_id','sample_id'], validate='many_to_one')
```

schema.md列出字段类型、原字段映射、联合主键和空值含义。cases.jsonl与cases/images/可直接用于案例排版。正文主线和用户写作要求位于context/paper_start；最终协议与评分说明位于context/final_project。version_receipt.md记录工作目录、真实Git状态和正常推送结果。

## CPU复现入口

在规范项目根目录依次执行scripts/paper_20260929/export_tables.py、export_runtime.py、export_cases.py、export_runtime.py --comparisons-only、finalize_export.py。Python依赖为pyarrow、pandas、Pillow；SQLite来自标准库。export_tables.py针对新的导出目录进行独占创建，重复执行应另设输出目录。已有模型输出与评分保持冻结。
""", encoding="utf-8")
    database.close()
    print(json.dumps({"checks_passed": checks["passed"], "discrepancies": len(checks["discrepancies"]), "qa_raw_locators": len(qa_locations), "uploaded_pairs": len(paired_rows), "payload_bytes_before_readme": total}))


if __name__ == "__main__":
    finalize()
