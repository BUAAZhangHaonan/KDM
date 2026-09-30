#!/usr/bin/env python3
"""Render source-bound core native accuracy and paired cluster intervals on CPU."""
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, within
from workflows.paper_core.receive_native import MODELS
from workflows.paper_core.score_native import save_json

LABELS = {"qwen25vl": "Qwen2.5-VL", "qwen35_4b": "Qwen3.5-4B", "llava16_mistral": "LLaVA-1.6-Mistral",
          "minicpm26": "MiniCPM-V2.6", "gemma3_4b": "Gemma3-4B"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    score, analysis, output = within(ROOT, args.score_dir), within(ROOT, args.analysis_dir), within(ROOT, args.output)
    output.mkdir(parents=True, exist_ok=False)
    metrics_path = score / "all_core_conditions_same_eval.csv"
    stats_path = analysis / "native_paired_statistics.json"
    metrics = pd.read_csv(metrics_path)
    direct = metrics[metrics.method_family.eq("Direct_unguided")].set_index("model")
    native = metrics[metrics.method_family.eq("VCD_native_unguided") & metrics.primary_complete].set_index("model")
    paired = {row["model"]: row for row in json.loads(stats_path.read_text())}
    if len(direct) != 5 or set(native.index) != set(paired):
        raise ValueError("Figure sources lack exact Direct baselines or matching completed native pairs")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 160, "savefig.dpi": 200, "font.family": "DejaVu Sans"})
    x, width = np.arange(5), .34
    fig, ax = plt.subplots(figsize=(10.5, 4.3), layout="constrained")
    base_values = [float(direct.loc[model, "accuracy"]) * 100 for model in MODELS]
    ax.bar(x - width / 2, base_values, width, color="#5477a8", label="Direct (unguided)")
    available_x, available_y = [], []
    for index, model in enumerate(MODELS):
        if model in native.index:
            available_x.append(x[index] + width / 2)
            available_y.append(float(native.loc[model, "accuracy"]) * 100)
        else:
            ax.text(x[index] + width / 2, 1.5, "Pending", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.bar(available_x, available_y, width, color="#c56a4b", label="Native VCD (unguided)")
    ax.set_xticks(x, [LABELS[model] for model in MODELS])
    ax.set_ylabel("Canonical primary-name accuracy (%)")
    ax.set_ylim(0, max(base_values + available_y) + 10)
    ax.set_title("Food-101 eval · 2,424 inputs per complete condition")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    fig.savefig(output / "native_accuracy.png")
    fig.savefig(output / "native_accuracy.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.7, 4.1), layout="constrained")
    for index, model in enumerate(MODELS):
        if model in paired:
            stat = paired[model]["accuracy_native_minus_Direct"]
            value, lower, upper = stat["estimate"] * 100, stat["ci95"][0] * 100, stat["ci95"][1] * 100
            ax.errorbar(value, index, xerr=[[value - lower], [upper - value]], fmt="o", color="#a85036", capsize=4, markersize=6)
        else:
            ax.text(.2, index, "Pending primary scoring", va="center", fontsize=9, color="#666666")
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set_yticks(np.arange(5), [LABELS[model] for model in MODELS])
    ax.set_ylim(4.5, -.5)
    ax.set_xlabel("Accuracy difference: native VCD − unguided Direct (percentage points)")
    ax.set_title("Paired 95% CI · 2,000 shared resamples of 101 Food-101 classes")
    ax.grid(axis="x", alpha=.18)
    fig.savefig(output / "native_accuracy_difference_ci.png")
    fig.savefig(output / "native_accuracy_difference_ci.pdf")
    plt.close(fig)
    result = {"schema": "kdm_core_native_figures_v1", "completed_native_conditions": len(native),
              "pending_native_conditions": 5 - len(native), "all_complete_condition_n": 2424,
              "table_path": str(metrics_path.relative_to(ROOT)), "table_sha256": file_hash(metrics_path),
              "pair_statistics_path": str(stats_path.relative_to(ROOT)), "pair_statistics_sha256": file_hash(stats_path),
              "files": {path.name: file_hash(path) for path in sorted(output.iterdir())},
              "renderer_source_sha256": file_hash(Path(__file__)), "visual_review": "pending"}
    save_json(output / "figure_receipt.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
