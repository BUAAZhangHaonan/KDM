#!/usr/bin/env python3
"""Package the genuinely closed Food main panel and finite supporting evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kdm.io import file_hash, within
from workflows.paper_core.assemble_nine_main_food import MODELS, output_json

BASE = ROOT / "outputs/paper_core_20261002_dev_viz"
LABELS = {"direct": "Direct", "dola": "DoLa", "deco": "DeCo", "cda_visual": "CDA",
          "vcd": "Native VCD", "sid": "SID", "instruction_vcd": "IP-VCD", "instruction_m3id": "IP-M3ID"}


def markdown(frame):
    values = [["" if pd.isna(x) else str(x).replace("|", "\\|") for x in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
                     + ["| " + " | ".join(row) + " |" for row in values])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("main", "figures", "mechanism", "support", "matrix-status", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = within(ROOT, args.output)
    output.relative_to(BASE)
    inputs = {name: within(ROOT, getattr(args, name)) for name in ("main", "figures", "mechanism", "support")}
    for name, folder in inputs.items():
        receipt = json.loads((folder / "receipt.json").read_text())
        if receipt["passed"] is not True:
            raise ValueError("An input projection is not accepted: " + name)
        for relative, expected in receipt["outputs"].items():
            if file_hash(folder / relative) != expected:
                raise ValueError("An accepted input projection changed: " + name + "/" + relative)
    main_receipt = json.loads((inputs["main"] / "receipt.json").read_text())
    if (main_receipt["rows"], main_receipt["conditions"]) != (298152, 123):
        raise ValueError("The full nine-model Food main comparison is required")
    matrix_status = within(ROOT, args.matrix_status)
    status = json.loads(matrix_status.read_text())
    if status.get("Food_main_complete") is not True or status.get("Food_main_conditions") != 123:
        raise ValueError("The supplied actual status does not certify the Food main panel")
    output.mkdir(parents=True, exist_ok=False)
    for name, folder in inputs.items():
        shutil.copytree(folder, output / name)
    shutil.copyfile(matrix_status, output / "ACTUAL_TASK_STATUS.json")
    tools = output / "scripts"
    tools.mkdir()
    for name in ("export_main_compact.py", "export_frozen_registered_mechanism.py", "prepare_review_support.py",
                 "render_main_joint.py", "recover_frozen_qa.py", "verify_nine_review_package.py", "build_nine_review_package.py"):
        shutil.copyfile(Path(__file__).with_name(name), tools / name)

    metrics = pd.read_csv(output / "main/metrics_all_complete.csv")
    selected = pd.read_csv(output / "main/best_observed_joint_operating_points.csv")
    dev = pd.read_csv(output / "main/dev_selected_operating_points.csv")
    baselines = metrics[metrics.method.isin(["direct", "dola", "deco", "cda_visual", "sid", "vcd"])].copy()
    points = pd.concat([baselines, selected], ignore_index=True)
    if len(metrics) != 123 or len(selected) != 18 or len(dev) != 5 or points.duplicated(["model", "method"]).any():
        raise ValueError("The complete main methods or actual selection identities differ")
    table = points.pivot(index="model", columns="method", values="J").reindex(MODELS)
    table = table.reindex(columns=list(LABELS)).rename(columns=LABELS)
    checkpoints = metrics[["model", "checkpoint"]].drop_duplicates().set_index("model").checkpoint
    numeric = table.reset_index()
    numeric.insert(1, "checkpoint", numeric.model.map(checkpoints))
    numeric.to_csv(output / "nine_model_main_joint_J.csv", index=False)
    formatted = table.map(lambda x: "—" if pd.isna(x) else f"{100*x:.2f}").reset_index()
    formatted["model"] = formatted.model.map(checkpoints)
    supporting = []
    for _, point in selected.iterrows():
        for method in ("vcd", "cda_visual"):
            baseline = baselines[(baselines.model == point.model) & (baselines.method == method)].iloc[0]
            row = {"model": point.model, "method": point.method, "marker": point.marker,
                   "selection": point.selection, "baseline": method,
                   "condition_id": point.condition_id, "baseline_condition_id": baseline.condition_id,
                   "N": 2424, "delta_J_pp": 100*(point.J-baseline.J)}
            for field in ("C", "W", "A", "TP", "FP", "FN", "reference_positive"):
                row["method_"+field] = int(point[field]); row["baseline_"+field] = int(baseline[field])
                row["delta_"+field] = int(point[field]-baseline[field])
            supporting.append(row)
    pd.DataFrame(supporting).to_csv(output / "necessary_and_unnecessary_abstention_changes.csv", index=False)
    behavior_table = pd.DataFrame(supporting)
    behavior_table = behavior_table[(behavior_table.method == "instruction_vcd") & (behavior_table.baseline == "vcd")]
    behavior_table = behavior_table[["model", "delta_C", "delta_TP", "delta_FP", "delta_J_pp"]].copy()
    behavior_table["delta_J_pp"] = behavior_table.delta_J_pp.map(lambda x: f"{x:+.2f}")
    dev_table = dev[["model", "marker", "C", "W", "A", "TP", "FP", "J"]].copy()
    dev_table["J"] = dev_table.J.map(lambda x: f"{100*x:.2f}")
    dev_table.to_csv(output / "core5_actual_dev_selected_IP_VCD.csv", index=False)
    native = baselines[baselines.method.eq("vcd")][["model", "condition_id", "J"]].rename(
        columns={"condition_id": "native_condition_id", "J": "native_J"})
    dev_native = dev.merge(native, on="model", validate="one_to_one")
    if len(dev_native) != 5 or not dev_native.method.eq("instruction_vcd").all():
        raise ValueError("The five actual dev-selected IP-VCD native comparisons differ")
    dev_native["delta_J_pp"] = 100 * (dev_native.J - dev_native.native_J)
    dev_native.to_csv(output / "core5_actual_dev_selected_vs_native_J.csv", index=False)
    dev_native_higher = int(dev_native.delta_J_pp.gt(0).sum())
    dev_native_mean_delta_pp = float(dev_native.delta_J_pp.mean())

    cda = pd.read_csv(output / "support/cda/CDA_UNKNOWN_weight_state_reference_summary.csv")
    cda = cda[(cda.panel == "full_eval") & (cda.aggregation == "condition_all")].copy()
    cda_table = cda[["model", "responses", "steps", "zero_sum_steps_fraction", "negative_wa_steps_fraction", "wa_first_position_mean"]].copy()
    for field in ("zero_sum_steps_fraction", "negative_wa_steps_fraction"):
        cda_table[field] = cda_table[field].map(lambda x: f"{100*x:.2f}%")
    cda_table["wa_first_position_mean"] = cda_table.wa_first_position_mean.map(lambda x: f"{x:.4f}")
    cda_table.to_csv(output / "core5_original_CDA_weight_behavior.csv", index=False)
    all_cda = pd.read_csv(output / "support/cda/CDA_UNKNOWN_weight_state_reference_summary.csv")
    cda_state = all_cda[(all_cda.panel == "full_eval") & (all_cda.aggregation == "condition_state_reference")].copy()
    cda_state.to_csv(output / "core5_original_CDA_weight_behavior_by_state_reference.csv", index=False)
    q25 = cda_state[cda_state.model == "qwen25vl"].groupby("state").agg(
        responses=("responses", "sum"), first_weight_sum=("wa_first_sum", "sum"),
        first_weight_count=("responses_with_first_weights", "sum"))
    q25["wa_first_mean"] = q25.first_weight_sum / q25.first_weight_count
    q25_table = q25[["responses", "wa_first_mean"]].reset_index()
    q25_table["wa_first_mean"] = q25_table.wa_first_mean.map(lambda x: f"{x:.4f}")
    aggregate = json.loads((output / "support/mechanism/core5/aggregate_summary.json").read_text())
    proxy = aggregate["representative_pair_counts"]
    proxy_table = pd.DataFrame([{"panel": "representative101", "support": "guided", "N_proxy_pairs": proxy["both_guided"],
                                "marker_relative_advantage_lower": proxy["both_guided_negative"],
                                "marker_relative_advantage_higher": proxy["both_guided_positive"],
                                "zero": proxy["both_guided_zero"]}])
    proxy_table.to_csv(output / "third_term_proxy_direction.csv", index=False)
    macro = pd.read_csv(output / "figures/descriptive_macro_J.csv")
    display_macro = macro.copy()
    display_macro["method"] = display_macro.method.map(LABELS)
    display_macro["baseline_method"] = display_macro.baseline_method.map(LABELS)
    display_macro["mean_J_delta"] = display_macro.mean_J_delta.map(lambda x: f"{100*x:+.2f} pp")
    by_pair = macro.set_index(["method", "baseline_method"])
    ip_vcd_native = by_pair.loc[("instruction_vcd", "vcd")]
    ip_vcd_cda = by_pair.loc[("instruction_vcd", "cda_visual")]
    ip_m3id_cda = by_pair.loc[("instruction_m3id", "cda_visual")]
    llava_effect = next(row for row in supporting if row["model"] == "llava16_mistral"
                        and row["method"] == "instruction_vcd" and row["baseline"] == "vcd")
    q25_dev = dev[dev.model.eq("qwen25vl")].iloc[0]
    q25_observed = selected[selected.model.eq("qwen25vl") & selected.method.eq("instruction_vcd")].iloc[0]
    same_selected = dev.merge(selected, on=["model", "method"], suffixes=("_dev", "_eval"))
    same_selected_count = int(same_selected.condition_id_dev.eq(same_selected.condition_id_eval).sum())
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    text = f"""# KDM 九模型补充结果审阅包

生成时间：{datetime.now(timezone.utc).isoformat()}。代码：{source_commit}。

Food-101九模型主比较已完整：123条件、298,152条，全部使用同一2,424张eval图像、冻结评分和模型级参考。包含六模型实际注册的SID；未注册支持的SID记空。联合指标J=(正确作答C+参考支持弃权TP)/N。原五模型853,248条冻结资产和论文正文未修改。

## 1. 原生基线与本文方法的联合评价

{markdown(formatted)}

数值为J百分比。每个IP方法的一行全部指标来自同一个J最高的实际eval条件（best_observed），四措辞的各指标独立极值只在单独表中。主表保留Direct、DoLa、DeCo、原始UNKNOWN配置CDA、原生VCD及实际注册SID。guided VCD/M3ID只进入机制目录。九模型的最佳观测表不冒充开发集选择结果。

![联合指标](figures/nine_model_joint_score.png)

{markdown(display_macro)}

IP-VCD相较原生VCD在{int(ip_vcd_native.improved_models)}/9模型提高J，等权平均提高{100*ip_vcd_native.mean_J_delta:.2f}个百分点，其中{int(ip_vcd_native.positive_saved_CI_models)}个模型的保存配对95%区间完全高于0；相较CDA在{int(ip_vcd_cda.improved_models)}/9模型提高J，平均提高{100*ip_vcd_cda.mean_J_delta:.2f}个百分点，{int(ip_vcd_cda.positive_saved_CI_models)}个区间完全高于0。IP-M3ID相较CDA在{int(ip_m3id_cda.improved_models)}/9模型提高J、平均提高{100*ip_m3id_cda.mean_J_delta:.2f}个百分点。这些是注册eval四措辞中的最佳观测结果，开发集工作点另列。

宏平均是九模型等权描述性均值；各模型的真实配对区间来自类别聚类bootstrap（2,000次、seed 20260929），没有用平均区间充当跨模型显著性。精确分子、分母和同一次运行的Acc/P/R/F1/保留率见main表。

## 2. 必要弃权与不必要弃权的变化

necessary_and_unnecessary_abstention_changes.csv逐模型提供C/W/A/TP/FP/FN、参考阳性数及差值；J提高可以由正确回答增加或参考支持弃权增加形成。未将所有新增弃权都视为收益。transitions.csv保留C/E/A九格及参考正负分层。

{markdown(behavior_table)}

例如LLaVA-Mistral中，IP-VCD正确回答相对原生VCD变化{llava_effect['delta_C']:+d}条，参考支持弃权变化{llava_effect['delta_TP']:+d}条，J变化{llava_effect['delta_J_pp']:+.2f}个百分点；不必要弃权变化{llava_effect['delta_FP']:+d}条也被单列。联合指标显示这项实际取舍，完整分项保留。

## 3. 第三项压低弃权的实际条件

{markdown(proxy_table)}

对保存的非marker−marker候选边际，交互项=正常指令增量−参考指令增量，IP减guided=负交互项。交互项为负时降低marker相对优势；方向也可能相反。表格是已有首位词元代理，不能替代完整回复的语义弃权率。共同支持内的该代理子集为0，不报告它的分布。代表、诊断和完整路径分开保存。

## 4. 实际开发集选择

{markdown(dev_table)}

这五个IP-VCD配置确实仅用Food dev404选择并冻结，eval结果与best_observed分开。IP-M3ID及扩展四模型的dev选择尚未完成；相应主表使用明确标记的注册eval观测集合。registered_dev_decisions.parquet保存已完成32,320条开发记录与原来源，当前主表只采用上述五个实际IP-VCD工作点。

这五个实际dev选定工作点相对原生VCD，在{dev_native_higher}/5模型提高eval J，五模型等权描述性平均提高{dev_native_mean_delta_pp:.2f}个百分点。core5_actual_dev_selected_vs_native_J.csv保留两侧条件身份、各项分子和真实差值；这个均值不作为跨模型显著性结论。

其中Qwen2.5-VL的dev选定{q25_dev.marker}，eval J为{100*q25_dev.J:.2f}%；其eval最佳观测为{100*q25_observed.J:.2f}%。五个dev选定IP-VCD配置中，{same_selected_count}个与本轮eval最高J条件一致。两类选择分别记账。

## 5. 原始CDA权重与行为

{markdown(cda_table)}

只汇报五模型UNKNOWN单配置的12,120条实际回答。权重统计按C/E/A和参考正负分组；Eq6/7的重算残差及零和扩展、负权重均保留。现存trace缺完整分支向量，不能宣布Eq4全向量闭合，也不从分支数推断实测耗时。观测权重与行为的关系是描述性关系。

Qwen2.5-VL的首位弃权分支权重按实际行为汇总如下：

{markdown(q25_table)}

该模型实际弃权组的首位弃权权重均值高于正确作答组和错误作答组；这是保存trace的相关关系，不能单凭权重符号判定整句是否弃权。完整模型、行为与参考分层见core5_original_CDA_weight_behavior_by_state_reference.csv。

## 覆盖与目录

- main/：298,152条紧凑逐样本评分、完整问答字典、条件、源路径与行号、全部主指标、实际配对统计及转换。
- mechanism/：冻结五模型200个注册VCD/M3ID矩阵和Ref-off条件，共484,800条。扩展四模型相应160条件继续运行，实际去重进度见ACTUAL_TASK_STATUS.json。
- support/vizwiz/：五模型512题已完成31个条件、15,872条；主比较20条件与机制11条件分开。官方回答分数保留连续值；每条件166不可回答、346可回答。
- support/references/：Food九模型43,632个dev/eval参考记录；扩展四模型193,920个独立试答结果和来源。原五模型242,400次试答的来源由冻结参考及原source字典定位。
- support/mechanism/、support/cases/、support/replay_source_audit/：真实四路位置、稀疏候选、12案例路径和必要图像。历史6路径9处argmax差异与浮点并列检查分开；没有补造缺失噪声或缓存证据。

本包交付完整Food主比较及已经完成的支持证据；扩展矩阵、扩展VizWiz方法面板、扩展诊断/完整案例仍有具体缺项，详见实际状态。原生M3ID的五模型独立基线未在本轮明确53条件清单内完成，不以历史guided结果替代。大型raw、token IDs、逐步概率和完整运行身份保留服务器原路径，紧凑记录通过source文件SHA和一基行号连接；历史原名与来源保留。

离线核对：在本包目录执行 `python scripts/verify_nine_review_package.py --package .`。它核对文件清单、九模型键覆盖、101×24配额、J分子、参考连接与两类选择身份。所有计划量和正在运行量都未计作完成量。
"""
    (output / "README.zh.md").write_text(text, encoding="utf-8")
    output_json(output / "PACKAGE_RECEIPT.json", {
        "schema": "kdm_nine_closed_main_review_package_v1", "passed": True,
        "Food_main_complete": True, "Food_main_rows": 298152, "Food_main_conditions": 123,
        "frozen_mechanism_rows": 484800, "frozen_mechanism_conditions": 200,
        "extension_mechanism_complete": False, "all_nine_VizWiz_methods_complete": False,
        "actual_dev_selected_IP_VCD_models": 5, "actual_commit": source_commit,
        "source_projections": {k: {"path": str(v.relative_to(ROOT)), "receipt_sha256": file_hash(v / "receipt.json")} for k,v in inputs.items()},
        "actual_matrix_status_source": str(matrix_status.relative_to(ROOT)), "actual_matrix_status_sha256": file_hash(matrix_status),
        "GPU_initialized": False, "frozen_source_or_paper_modified": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "actual_command": [sys.executable, *sys.argv]})
    manifest = {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": file_hash(path)}
                for path in sorted(output.rglob("*")) if path.is_file()}
    output_json(output / "PACKAGE_MANIFEST.json", manifest)
    subprocess.run([sys.executable, str(tools / "verify_nine_review_package.py"), "--package", str(output)], check=True)
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(output.rglob("*")):
            if path.is_file(): zipped.write(path, path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("The completed archive fails CRC verification")
    if archive.stat().st_size > 25*1024*1024:
        raise ValueError("The actual review package exceeds 25 MiB")
    print(json.dumps({"passed": True, "Food_main_rows": 298152, "Food_main_conditions": 123,
                      "archive": str(archive.relative_to(ROOT)), "archive_sha256": file_hash(archive),
                      "archive_bytes": archive.stat().st_size, "files": len(manifest)}))


if __name__ == "__main__":
    main()
