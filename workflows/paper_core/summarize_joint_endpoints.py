#!/usr/bin/env python3
"""Summarize actual correct-or-reasonable-abstention endpoints from closed P1."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b")
MARKERS = ("UNKNOWN", "UNCLEAR", "UNSURE", "I cannot identify it")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = (ROOT / args.input).resolve(), (ROOT / args.output).resolve()
    if ROOT not in source.parents or ROOT not in output.parents:
        raise ValueError("Input/output must stay inside this repository")
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 20 or {r["model"] for r in rows} != set(MODELS):
        raise ValueError("Expected twenty actual four-marker, five-model rows")
    selected = []
    lines = [
        "# 回答与合理弃权的联合评价",
        "",
        "J=(正确回答数+参考支持的弃权数)/2424。IP四措辞全部来自冻结eval实际运行；下表每模型选择J最高的同一次运行，所有计数保留该条件身份。并列按U/C/S/I固定顺序。这是eval最佳观测，尚未做dev配置选择；没有新生成。",
        "",
        "| 注册checkpoint | 原生VCD J | IP最佳J | IP措辞/CID | 差值（百分点） |",
        "|---|---:|---:|---|---:|",
    ]
    for model in MODELS:
        items = [r for r in rows if r["model"] == model]
        if len(items) != 4 or {r["marker"] for r in items} != set(MARKERS):
            raise ValueError("Marker coverage or condition uniqueness differs")
        for item in items:
            if item["dataset"] != "food101" or item["split"] != "eval":
                raise ValueError("A row is outside the registered eval set")
            n = int(item["J_denominator"])
            if n != 2424 or int(item["paired_n"]) != n:
                raise ValueError("Actual paired denominator differs")
            for side in ("native", "ip"):
                total = int(item[side + "_correct"]) + int(item[side + "_tp"])
                if total != int(item[side + "_J_numerator"]) or not 0 <= total <= n:
                    raise ValueError("Actual disjoint outcome sum differs")
                if abs(total / n - float(item[side + "_J"])) > 1e-15:
                    raise ValueError("Saved ratio differs from actual counts")
        if len({(r["native_condition_key"], r["native_J_numerator"]) for r in items}) != 1:
            raise ValueError("Native comparison identity differs across markers")
        best = max(Fraction(int(r["ip_J_numerator"]), 2424) for r in items)
        ties = [r for r in items if Fraction(int(r["ip_J_numerator"]), 2424) == best]
        chosen = min(ties, key=lambda r: MARKERS.index(r["marker"]))
        native = Fraction(int(chosen["native_J_numerator"]), 2424)
        delta = float(best - native) * 100
        result = {**chosen, "selection": "best_observed_eval_joint_J",
                  "dev_selected": False, "delta_percentage_points": delta,
                  "tie_condition_ids": json.dumps([r["ip_condition_id"] for r in ties])}
        selected.append(result)
        lines.append(
            f"| {chosen['hf_model_id']} | {int(chosen['native_J_numerator'])}/2424="
            f"{float(native)*100:.4f}% | {int(chosen['ip_J_numerator'])}/2424="
            f"{float(best)*100:.4f}% | {chosen['marker']}/{chosen['ip_condition_id']} | "
            f"{delta:+.4f} |"
        )
    lines += [
        "",
        "前四模型的IP最佳观测联合指标高于原生VCD；Gemma低0.6601个百分点。该表体现正确回答与合理弃权的合计，不表示所有IP措辞均改善，也不将四项分别最优端点拼成同一工作点。主菜正确性、语义弃权及uniform参考沿用已闭合评分，旧资产未改。",
        "",
        "开发集选择未完成，本表未宣称实际部署配置已经确定。未新增置信区间或选择校正。",
    ]
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "joint_best_observed_endpoints.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    report = output / "joint_evaluation.md"
    report.write_text("\n".join(lines) + "\n")
    receipt = {
        "schema": "kdm_frozen_joint_endpoint_summary_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(source.relative_to(ROOT)), "input_sha256": sha(source),
        "actual_rows_verified": 20, "actual_models": 5, "n_per_condition": 2424,
        "new_generations": 0, "GPU_initialized": False, "dev_selected": False,
        "source_script_sha256": sha(Path(__file__)),
        "output_sha256": {p.name: sha(p) for p in (csv_path, report)},
    }
    (output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": True, "rows": 5, "output": str(output.relative_to(ROOT)),
                      "joint_delta_pp": {r["model"]: r["delta_percentage_points"]
                                         for r in selected}}))


if __name__ == "__main__":
    main()
