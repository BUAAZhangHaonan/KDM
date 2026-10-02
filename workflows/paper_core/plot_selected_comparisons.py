"""Plot accepted paired Food comparisons without rerunning inference/statistics."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = ["qwen25vl", "qwen35_4b", "minicpm26", "llava16_mistral", "gemma3_4b"]
DISPLAY = ["Qwen2.5-VL", "Qwen3.5-4B", "MiniCPM-2.6", "LLaVA-Mistral", "Gemma3-4B"]
COMPARISONS = [
    ("dev_selected_vs_native_VCD", "Native VCD"),
    ("IP_vs_same_budget_dev_selected_guided_VCD", "Dev-selected guided VCD"),
    ("IP_vs_same_marker_guided_VCD", "Same-marker guided VCD"),
]
METRICS = [("accuracy", "Accuracy", "#2471a3", -0.12), ("J", "Joint score J", "#c66b14", 0.12)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    src = args.selection_dir.resolve()
    out = args.out.resolve()
    if out.exists() or out == src or src in out.parents:
        raise ValueError("Use a new independent output directory.")

    paired_file = src / "paired_effects.csv"
    selected_file = src / "selected_operating_points.csv"
    receipt_file = src / "analysis_receipt.json"
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if (receipt.get("passed") is not True or receipt.get("selected_configs") != 20
            or receipt.get("dev_rows") != 32320 or receipt.get("frozen_selections_verified") is not True
            or set(receipt.get("models", [])) != set(MODELS)):
        raise ValueError("Selection source is not accepted.")
    paired, fields = read_csv(paired_file)
    selected, _ = read_csv(selected_file)
    if len(paired) != 120 or len(selected) != 35:
        raise ValueError("Require the completed five-model 120/35 tables.")
    configs = [row for row in selected if row["selection"] == "dev_selected"]
    if len(configs) != 20 or {row["model"] for row in configs} != set(MODELS):
        raise ValueError("Require all 20 frozen dev-selected configurations.")
    for row in selected:
        if (int(row["n"]) != 2424
                or sum(int(row[k]) for k in ("C", "W", "A")) != 2424
                or int(row["TP"]) + int(row["FP"]) != int(row["A"])
                or int(row["J_numerator"]) != int(row["C"]) + int(row["TP"])):
            raise ValueError("Source counts do not close.")

    index = {}
    for row in paired:
        if row["method"] == "instruction_vcd" and row["metric"] in {"accuracy", "J"}:
            key = (row["comparison"], row["model"], row["metric"])
            if key in index:
                raise ValueError("Duplicate plotted comparison.")
            if (int(row["paired_n"]) != 2424 or int(row["bootstrap_replicates"]) != 2000
                    or int(row["bootstrap_seed"]) != 20260929 or row["bootstrap_unit"] != "food_class"):
                raise ValueError("Paired source differs from the registered analysis.")
            values = [float(row[key]) for key in ("delta", "ci95_lower", "ci95_upper")]
            if not all(math.isfinite(value) for value in values) or not values[1] <= values[0] <= values[2]:
                raise ValueError("Invalid saved paired interval.")
            index[key] = row
    expected = {(cmp, model, metric) for cmp, _ in COMPARISONS for model in MODELS
                for metric, *_ in METRICS}
    if set(index) != expected:
        raise ValueError("Require exactly 30 saved IP-VCD comparisons.")

    out.mkdir(parents=True)
    plotted = [index[(cmp, model, metric)] for cmp, _ in COMPARISONS for model in MODELS
               for metric, *_ in METRICS]
    with (out / "figure_data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(plotted)

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.7), sharey=True)
    low = min(float(row["ci95_lower"]) * 100 for row in plotted) - 1.5
    high = max(float(row["ci95_upper"]) * 100 for row in plotted) + 1.5
    for ax, (comparison, title) in zip(axes, COMPARISONS):
        ax.axvline(0, color="#888888", linewidth=0.8, zorder=1)
        for metric, label, color, offset in METRICS:
            rows = [index[(comparison, model, metric)] for model in MODELS]
            point = [float(row["delta"]) * 100 for row in rows]
            lower = [value - float(row["ci95_lower"]) * 100 for value, row in zip(point, rows)]
            upper = [float(row["ci95_upper"]) * 100 - value for value, row in zip(point, rows)]
            ax.errorbar(point, [i + offset for i in range(5)], xerr=[lower, upper],
                        fmt="o", markersize=4.5, capsize=2, color=color, label=label, zorder=3)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(low, high)
        ax.set_xlabel("IP-VCD minus baseline (percentage points)")
        ax.grid(axis="x", color="#eeeeee", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(5), DISPLAY)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle("Food-101 eval: dev-selected IP-VCD", fontsize=13, y=0.99)
    fig.text(0.01, 0.035,
             "N=2,424 per model; saved 95% class-bootstrap intervals (2,000 replicates; seed 20260929).\n"
             "J=(correct answers + reference-supported abstentions)/N. "
             "Gemma: selected guide UNSURE; IP and matched guide UNCLEAR.",
             fontsize=8.5, va="bottom")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(out / "ip_selected_food_comparisons.png", dpi=180, facecolor="white")
    fig.savefig(out / "ip_selected_food_comparisons.pdf", facecolor="white")
    plt.close(fig)

    result = {
        "schema": "kdm_existing_paired_comparison_plot_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_rows": 120, "plotted_rows": 30, "selected_configs": 20, "eval_n": 2424,
        "input_files": [{"path": str(path), "sha256": sha(path)}
                        for path in (paired_file, selected_file, receipt_file)],
        "code_sha256": sha(Path(__file__).resolve()),
        "new_inference": 0, "new_semantic_judgments": 0, "new_bootstrap_or_tests": 0,
        "delta_orientation": "IP minus baseline", "intervals": "original saved pointwise 95%",
        "full_source_condition_fields_preserved_in_figure_data": True,
        "outputs": {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()},
    }
    (out / "plot_receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
