#!/usr/bin/env python3
"""Render J-only comparisons from complete, source-bound nine-model results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kdm.io import file_hash, within
from workflows.paper_core.assemble_nine_main_food import MODELS, output_json

DISPLAY = ("Qwen2.5-VL-7B", "Qwen3.5-4B", "LLaVA-Mistral-7B", "MiniCPM-V-2.6",
           "Gemma-3-4B", "InternVL3.5-8B", "OneVision-7B", "Phi-3.5-V", "Qwen3-VL-8B")
METHODS = ("direct", "vcd", "dola", "deco", "sid", "cda_visual", "instruction_vcd", "instruction_m3id")
LABELS = ("Direct", "VCD", "DoLa", "DeCo", "SID", "CDA", "IP-VCD", "IP-M3ID")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-panel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = (within(ROOT, p) for p in (args.main_panel, args.output))
    output.relative_to(ROOT / "outputs/paper_core_20261002_dev_viz")
    proof = json.loads((source / "receipt.json").read_text())
    if not proof["passed"] or not proof["complete_main_panel"] or proof["complete_conditions"] != 123:
        raise ValueError("The complete nine-model main panel is required")
    files = ("metrics_all_complete.csv", "best_observed_joint_operating_points.csv", "paired_main_comparisons.csv")
    for name in files:
        if file_hash(source / name) != proof["outputs"][name]:
            raise ValueError("An accepted main comparison source changed: " + name)
    all_metrics = pd.read_csv(source / files[0])
    selected = pd.read_csv(source / files[1])
    effects = pd.read_csv(source / files[2])
    if len(selected) != 18 or selected.duplicated(["model", "method"]).any() or set(selected.model) != set(MODELS):
        raise ValueError("Both IP methods require a single J-observed configuration on every model")
    if not selected.n.eq(2424).all() or not selected.selection_split.eq("eval").all() or selected.dev_selected.any():
        raise ValueError("Eval observations cannot be described as dev-selected results")
    baselines = all_metrics[all_metrics.method.isin(METHODS[:6])]
    if baselines.duplicated(["model", "method"]).any():
        raise ValueError("Native baselines must use one actual registered configuration")
    points = pd.concat([baselines, selected], ignore_index=True)
    if not points.J.eq((points.C + points.TP) / points.n).all():
        # CSV roundtrips can differ by the last floating digit only.
        if not np.allclose(points.J, (points.C + points.TP) / points.n, rtol=0, atol=1e-15):
            raise ValueError("The joint score does not match its actual numerator/denominator")
    table = points.pivot(index="model", columns="method", values="J").reindex(index=MODELS, columns=METHODS)
    paired = effects[effects.metric.eq("J") & effects.baseline_method.isin(["vcd", "cda_visual"])
                     & effects.selection.eq(selected.selection.iloc[0])].copy()
    if len(paired) != 36 or paired.duplicated(["model", "method", "baseline_method"]).any():
        raise ValueError("All 36 saved native-VCD/CDA J comparisons are required")
    if not paired.bootstrap_replicates.eq(2000).all() or not paired.bootstrap_seed.eq(20260929).all():
        raise ValueError("The registered paired bootstrap identity changed")
    for _, pair in paired.iterrows():
        method = table.loc[pair.model, pair.method]
        baseline = table.loc[pair.model, pair.baseline_method]
        if not np.isclose(pair.delta, method - baseline, atol=1e-15, rtol=0):
            raise ValueError("The saved paired J effect uses a different actual configuration")
    output.mkdir(parents=True, exist_ok=False)
    points.to_csv(output / "main_joint_points.csv", index=False)
    table.to_csv(output / "joint_score_table.csv", index=True)
    paired.to_csv(output / "joint_paired_figure_data.csv", index=False)
    macro = paired.groupby(["method", "baseline_method"], sort=True).agg(
        mean_J_delta=("delta", "mean"), improved_models=("delta", lambda x: int(x.gt(0).sum())),
        models=("model", "nunique"), positive_saved_CI_models=("ci95_lower", lambda x: int(x.gt(0).sum())))
    macro.to_csv(output / "descriptive_macro_J.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(11.6, 5.7), layout="constrained")
    values = table.to_numpy() * 100
    im = ax.imshow(np.ma.masked_invalid(values), cmap="Blues", vmin=0, vmax=70, aspect="auto")
    ax.set_xticks(range(len(METHODS)), LABELS)
    ax.set_yticks(range(len(MODELS)), DISPLAY)
    for i in range(len(MODELS)):
        maximum = np.nanmax(values[i])
        for j in range(len(METHODS)):
            value = values[i, j]
            text = "n/a" if not np.isfinite(value) else f"{value:.2f}"
            best = np.isfinite(value) and np.isclose(value, maximum, rtol=0, atol=1e-12)
            ax.text(j, i, text, ha="center", va="center", color="white" if value > 43 else "#182537",
                    weight="bold" if best else "normal", fontsize=10)
    ax.axvline(5.5, color="#263f54", linewidth=1.4)
    ax.set_title("Food-101: joint score J (%)", loc="left", fontsize=14, pad=12)
    fig.colorbar(im, ax=ax, fraction=.028, pad=.025, label="J (%)")
    fig.supxlabel("N=2,424/model. IP columns: highest observed eval J among four registered phrases; all values use that same configuration.", fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(output / ("nine_model_joint_score." + extension), dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), sharey=True, layout="constrained")
    colors = {"instruction_vcd": "#2171b5", "instruction_m3id": "#169c79"}
    names = {"instruction_vcd": "IP-VCD", "instruction_m3id": "IP-M3ID"}
    for ax, baseline in zip(axes, ("vcd", "cda_visual")):
        ax.axvline(0, color="#7c838c", linewidth=.9)
        for method, offset in (("instruction_vcd", -.13), ("instruction_m3id", .13)):
            block = paired[paired.method.eq(method) & paired.baseline_method.eq(baseline)].set_index("model").loc[list(MODELS)]
            y = np.arange(len(MODELS)) + offset
            delta = block.delta.to_numpy() * 100
            lo, hi = block.ci95_lower.to_numpy() * 100, block.ci95_upper.to_numpy() * 100
            ax.errorbar(delta, y, xerr=np.vstack((delta - lo, hi - delta)), fmt="o", markersize=4.8,
                        capsize=2.6, color=colors[method], label=names[method])
        ax.set_yticks(range(len(MODELS)), DISPLAY)
        ax.set_xlabel("Joint-score change (percentage points)")
        ax.set_title("Compared with " + ("native VCD" if baseline == "vcd" else "CDA visual transfer"), loc="left")
        ax.grid(axis="x", alpha=.17)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1.18), ncol=2)
    fig.supxlabel("Saved paired 95% class-bootstrap intervals: 2,000 replicates, seed 20260929. Eval-observed phrase choices; dev-selected results are reported separately.", fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(output / ("nine_model_joint_effects." + extension), dpi=220)
    plt.close(fig)
    output_json(output / "receipt.json", {
        "schema": "kdm_complete_nine_main_J_figures_v1", "passed": True, "models": 9,
        "actual_IP_operating_points": 18, "paired_J_comparisons": 36,
        "selection": "eval best-observed J, one actual configuration per model/method",
        "no_guided_VCD_or_M3ID_baselines": True, "unsupported_SID_cells_are_null": True,
        "macro_means_are_descriptive_not_averaged_CIs": True, "GPU_initialized": False,
        "source_receipt_sha256": file_hash(source / "receipt.json"),
        "sources": [{"path": str((source / name).relative_to(ROOT)), "sha256": file_hash(source / name)} for name in files],
        "runner_sha256": file_hash(Path(__file__)), "actual_command": [sys.executable, *sys.argv],
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"passed": True, "models": 9, "IP_points": 18, "paired_J_comparisons": 36}))


if __name__ == "__main__":
    main()
