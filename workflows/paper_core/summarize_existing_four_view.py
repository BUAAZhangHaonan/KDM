"""Summarize saved candidate-pair proxies and case traces without model inference."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys

import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PANEL_ORDER = ("representative101", "diagnostic198", "case12")
MODEL_ORDER = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b")
MODEL_LABELS = {"qwen25vl": "Qwen2.5-VL", "qwen35_4b": "Qwen3.5", "llava16_mistral": "LLaVA-Mistral",
                "minicpm26": "MiniCPM26", "gemma3_4b": "Gemma3"}
POSITION_GROUPS = ("all_positions", "first_position", "later_position")
SUPPORT_SCOPES = ("all_proxy", "guided_support", "common_support")
NUMERIC_FIELDS = ("natural_visual", "normal_instruction", "reference_instruction", "interaction", "ip_minus_guided")
QUANTILES = (("min", 0), ("q05", .05), ("q25", .25), ("median", .5), ("q75", .75), ("q95", .95), ("max", 1))
KEY = ("model", "sample_id", "panel", "position", "source_path", "source_line")
ZERO_TOLERANCE = 1e-12
ALGEBRA_TOLERANCE = 1e-12
BEEF_CASE = "food101:beef_carpaccio_eval_039.jpg"
PALETTE = {"positive": "#C6D4EA", "negative": "#F59092", "zero": "#E5D8D4"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv_with_lines(path):
    frame = pd.read_csv(path, keep_default_na=False)
    lines = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        while True:
            start = reader.line_num + 1
            record = next(reader, None)
            if record is None:
                break
            lines.append((start, reader.line_num))
    if len(lines) != len(frame):
        raise ValueError("CSV record/physical-line mapping differs")
    frame["source_csv_record"] = np.arange(1, len(frame) + 1)
    frame["source_csv_line"] = [first for first, _ in lines]
    frame["source_csv_end_line"] = [last for _, last in lines]
    return frame


def boolean_column(frame, field):
    values = frame[field].map({True: True, False: False, "True": True, "False": False})
    if values.isna().any():
        raise ValueError(f"Unexpected boolean values in {field}")
    frame[field] = values.astype(bool)


def source_events(path):
    values = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, text in enumerate(stream, 1):
            value = json.loads(text)
            value["events_artifact_line"] = line_number
            values.append(value)
    return values


def summary(subset, identity, provenance):
    n = len(subset)
    zero = subset.ip_minus_guided.abs().le(ZERO_TOLERANCE)
    positive = subset.ip_minus_guided.gt(ZERO_TOLERANCE)
    negative = subset.ip_minus_guided.lt(-ZERO_TOLERANCE)
    result = {**identity, "proxy_pair_n": n, "unique_sample_n": subset.sample_id.nunique(),
              "unique_model_n": subset.model.nunique(),
              "unique_model_input_n": len(subset[["model", "sample_id"]].drop_duplicates()),
              "recorded_position_n": subset.events_artifact_line.nunique(),
              "unique_prefix_position_n": len(subset[["model", "sample_id", "position", "prefix_tokens_json"]].drop_duplicates()),
              "ip_minus_guided_positive_n": int(positive.sum()), "ip_minus_guided_negative_n": int(negative.sum()),
              "ip_minus_guided_zero_n": int(zero.sum()),
              "ip_minus_guided_exact_positive_n": int(subset.ip_minus_guided.gt(0).sum()),
              "ip_minus_guided_exact_negative_n": int(subset.ip_minus_guided.lt(0).sum()),
              "ip_minus_guided_exact_zero_n": int(subset.ip_minus_guided.eq(0).sum()),
              "source_csv_lines": ";".join(map(str, sorted(subset.source_csv_line.unique()))),
              "empty_reason": "no_proxy_pair_in_source_subset" if not n else "", **provenance}
    if result["ip_minus_guided_positive_n"] + result["ip_minus_guided_negative_n"] + result["ip_minus_guided_zero_n"] != n:
        raise ValueError("Proxy sign counts do not close")
    for field in NUMERIC_FIELDS:
        for label, quantile in QUANTILES:
            result[f"{field}_{label}"] = float(subset[field].quantile(quantile)) if n else None
    return result


def position_filter(frame, group):
    if group == "first_position":
        return frame[frame.position.eq(0)]
    if group == "later_position":
        return frame[frame.position.gt(0)]
    return frame


def figures(proxy, out, runtime, provenance):
    spec = importlib.util.spec_from_file_location("easyplot_runtime", runtime)
    easyplot = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ValueError("The supplied plotting runtime cannot be loaded")
    spec.loader.exec_module(easyplot)
    labels = {"positive": "Marker weakened", "negative": "Marker strengthened", "zero": "Numerical zero"}
    with easyplot.publication_context(font_family="DejaVu Sans", base_size=8,
                                     glyphs="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-()%") as font:
        fig, axes = easyplot.make_scientific_plate(nrows=2, ncols=2, width_mm=190, height_mm=116, dpi=300)
        for panel_index, panel in enumerate(PANEL_ORDER[:2]):
            for support_index, scope in enumerate(SUPPORT_SCOPES[:2]):
                ax = axes[panel_index, support_index]
                source = proxy[proxy.panel.eq(panel)]
                if scope == "guided_support":
                    source = source[source.pair_in_guided_support]
                x = np.arange(len(MODEL_ORDER))
                bottoms = np.zeros(len(x), dtype=float)
                groups = [source[source.model.eq(model)] for model in MODEL_ORDER]
                for direction in ("positive", "negative", "zero"):
                    heights = []
                    for group in groups:
                        delta = group.ip_minus_guided
                        count = {"positive": delta.gt(ZERO_TOLERANCE).sum(), "negative": delta.lt(-ZERO_TOLERANCE).sum(),
                                 "zero": delta.abs().le(ZERO_TOLERANCE).sum()}[direction]
                        heights.append(float(count / len(group)) if len(group) else 0.)
                    ax.bar(x, heights, bottom=bottoms, width=.62, color=PALETTE[direction],
                           edgecolor="#566169", linewidth=.45, label=labels[direction])
                    bottoms += heights
                for index, group in enumerate(groups):
                    ax.text(index, 1.025, f"n={len(group)}", ha="center", va="bottom", fontsize=7)
                    if not len(group):
                        ax.text(index, .5, "no pair", ha="center", va="center", rotation=90, fontsize=6.5)
                ax.set_ylim(0, 1.13)
                ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100"])
                ax.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=20, ha="right")
                ax.set_ylabel("Recorded proxy pairs (%)")
                ax.text(.5, 1.13, f"{panel}: {scope}", transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
                easyplot.format_axes(ax)
                easyplot.add_panel_tag(ax, "abcd"[panel_index * 2 + support_index], x=-.10, y=1.13)
        axes[0, 0].legend(loc="upper left", bbox_to_anchor=(-.1, 1.43), ncol=3, frameon=False)
        fig.canvas.draw()
        easyplot.export_figure(fig, out / "proxy_margin_direction", formats=("png", "svg"), dpi=300,
                              provenance={**provenance, "font": font, "palette_hex": PALETTE,
                                          "unit": "recorded nonmarker-minus-marker candidate pair", "interval": "none: descriptive counts",
                                          "selection_status": "registered_measurement", "source_panels": list(PANEL_ORDER[:2])})
        easyplot.plt.close(fig)

        fig, axes = easyplot.make_scientific_plate(nrows=1, ncols=2, width_mm=190, height_mm=90, dpi=300)
        supported_delta = proxy.loc[proxy.pair_in_guided_support, "ip_minus_guided"]
        shared_min, shared_max = float(supported_delta.min()), float(supported_delta.max())
        shared_padding = max(.1, (shared_max - shared_min) * .06)
        for panel_index, panel in enumerate(PANEL_ORDER[:2]):
            ax = axes[0, panel_index]
            source = proxy[proxy.panel.eq(panel) & proxy.pair_in_guided_support]
            tick_labels = []
            for index, model in enumerate(MODEL_ORDER):
                group = source[source.model.eq(model)]
                tick_labels.append(f"{MODEL_LABELS[model]} (n={len(group)})")
                if len(group):
                    quantiles = group.ip_minus_guided.quantile([.05, .25, .5, .75, .95])
                    ax.scatter(group.ip_minus_guided, np.full(len(group), index), s=5, color="#568FC3", alpha=.18, edgecolors="none")
                    ax.plot([quantiles.loc[.05], quantiles.loc[.95]], [index, index], color="#566169", linewidth=.7)
                    ax.plot([quantiles.loc[.25], quantiles.loc[.75]], [index, index], color="#568FC3", linewidth=3.2, solid_capstyle="butt")
                    ax.scatter([quantiles.loc[.5]], [index], s=21, color="#C6D4EA", edgecolors="#252A2E", linewidths=.6, zorder=4)
            ax.axvline(0, color="#566169", linewidth=.6, linestyle="--")
            ax.set_yticks(np.arange(len(MODEL_ORDER)), tick_labels)
            ax.invert_yaxis()
            ax.set_xlabel("IP - guided margin, nonmarker - marker\n(log units)")
            ax.set_xlim(shared_min - shared_padding, shared_max + shared_padding)
            ax.text(.5, 1.06, f"{panel}: guided support", transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
            easyplot.format_axes(ax)
            easyplot.add_panel_tag(ax, "ab"[panel_index], x=-.08, y=1.06)
        fig.canvas.draw()
        easyplot.export_figure(fig, out / "proxy_margin_quantiles", formats=("png", "svg"), dpi=300,
                              provenance={**provenance, "font": font, "unit": "recorded candidate pair; repeated model-input positions",
                                          "interval": "thin range: q05-q95; thick range: q25-q75; point: median; not confidence intervals",
                                          "selection_status": "registered_measurement"})
        easyplot.plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--easyplot-runtime", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = {name: (ROOT / value).resolve() for name, value in vars(args).items()}
    for path in paths.values():
        if not path.is_relative_to(ROOT):
            raise ValueError("All sources and outputs must remain within the project")
    out = paths["out"]
    out.mkdir(parents=True, exist_ok=False)
    hashes = {name: sha(paths[name]) for name in ("source", "events", "cases", "easyplot_runtime")}
    command = "CUDA_VISIBLE_DEVICES='' " + shlex.join([sys.executable, *sys.argv])
    provenance = {"source_csv": str(paths["source"].relative_to(ROOT)), "source_csv_sha256": hashes["source"],
                  "source_events": str(paths["events"].relative_to(ROOT)), "source_events_sha256": hashes["events"],
                  "script_sha256": sha(Path(__file__)), "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
                  "numeric_zero_abs_tolerance": ZERO_TOLERANCE, "selection_status": "registered_measurement",
                  "alpha": 1, "candidate_margin": "nonmarker candidate a minus marker candidate b"}
    pair = read_csv_with_lines(paths["source"])
    events = source_events(paths["events"])
    cases = read_csv_with_lines(paths["cases"])
    if len(pair) != 1049 or len(events) != 814 or len(cases) != 24:
        raise ValueError("The existing 1049-pair / 814-position / 24-path artifact contract differs")
    for field in ("a_nonmarker_b_marker_proxy", "pair_in_guided_support", "pair_in_common_support"):
        boolean_column(pair, field)
    for field in NUMERIC_FIELDS:
        pair[field] = pd.to_numeric(pair[field], errors="raise").astype(np.float64)
    if not np.isfinite(pair[list(NUMERIC_FIELDS)].to_numpy()).all():
        raise ValueError("Candidate margins contain nonfinite values")
    pair["interaction_closure_residual"] = pair.interaction - pair.normal_instruction + pair.reference_instruction
    pair["ip_guided_closure_residual"] = pair.ip_minus_guided + pair.interaction
    residuals = pair[["interaction_closure_residual", "ip_guided_closure_residual"]].abs().max().to_dict()
    if any(value > ALGEBRA_TOLERANCE for value in residuals.values()):
        raise ValueError("Saved candidate-pair algebra does not close at float64 tolerance")
    event_meta = pd.DataFrame([{**{key: event[key] for key in KEY}, "events_artifact_line": event["events_artifact_line"],
                               "path_condition": event["path_condition"],
                               "prefix_tokens_json": json.dumps(event["prefix_tokens"], separators=(",", ":"))}
                              for event in events])
    pair = pair.merge(event_meta[event_meta.panel.isin(pair.panel.unique())], on=list(KEY), how="left", validate="many_to_one")
    if pair.events_artifact_line.isna().any():
        raise ValueError("Candidate pairs cannot be linked to their saved positions")
    pair.events_artifact_line = pair.events_artifact_line.astype(int)
    pair["ip_margin_numeric_sign"] = np.where(pair.ip_minus_guided.gt(ZERO_TOLERANCE), "positive",
                                               np.where(pair.ip_minus_guided.lt(-ZERO_TOLERANCE), "negative", "zero"))
    for name, value in provenance.items():
        pair[name] = value
    proxy = pair[pair.a_nonmarker_b_marker_proxy].copy()
    if (proxy.candidate_a == proxy.candidate_b).any():
        raise ValueError("A marker proxy pair has identical candidates")
    proxy.to_csv(out / "proxy_pairs_annotated.csv", index=False)
    pair.to_csv(out / "all_pair_algebra_and_sources.csv", index=False)
    summaries, joint, coverage = [], [], []
    for panel in PANEL_ORDER:
        for model in ("ALL", *MODEL_ORDER):
            selected = proxy[proxy.panel.eq(panel)]
            source_positions = event_meta[event_meta.panel.eq(panel)]
            paths_in_cases = cases[cases.model.isin(MODEL_ORDER)] if panel == "case12" else cases.iloc[:0]
            if model != "ALL":
                selected = selected[selected.model.eq(model)]
                source_positions = source_positions[source_positions.model.eq(model)]
                paths_in_cases = paths_in_cases[paths_in_cases.model.eq(model)]
            coverage.append({"panel": panel, "model": model, "recorded_event_position_n": len(source_positions),
                             "unique_model_input_n": len(source_positions[["model", "sample_id"]].drop_duplicates()),
                             "unique_sample_n": source_positions.sample_id.nunique(), "saved_case_path_n": len(paths_in_cases),
                             "candidate_proxy_pair_n": len(selected), **provenance})
            for position_group in POSITION_GROUPS:
                positions = position_filter(selected, position_group)
                identity = {"panel": panel, "model": model, "position_group": position_group}
                for scope in SUPPORT_SCOPES:
                    subset = positions if scope == "all_proxy" else positions[positions["pair_in_" + scope]]
                    summaries.append(summary(subset, {**identity, "support_scope": scope}, provenance))
                for guided in (False, True):
                    for common in (False, True):
                        subset = positions[positions.pair_in_guided_support.eq(guided) & positions.pair_in_common_support.eq(common)]
                        joint.append(summary(subset, {**identity, "pair_in_guided_support": guided,
                                                       "pair_in_common_support": common}, provenance))
    pd.DataFrame(summaries).to_csv(out / "proxy_pair_summary.csv", index=False)
    pd.DataFrame(joint).to_csv(out / "proxy_joint_support_summary.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "event_panel_model_coverage.csv", index=False)
    examples = []
    for sign in ("positive", "negative"):
        available = proxy[proxy.panel.eq("representative101") & proxy.pair_in_guided_support & proxy.ip_margin_numeric_sign.eq(sign)]
        if available.empty:
            raise ValueError(f"No saved representative guided-supported {sign} proxy example exists")
        example = available.sort_values("source_csv_record").iloc[0].to_dict()
        example["marker_relative_effect"] = "weakened" if sign == "positive" else "strengthened"
        example["selection_rule"] = "first source CSV record with this sign in representative guided support"
        examples.append(example)
    pd.DataFrame(examples).to_csv(out / "direction_examples2.csv", index=False)
    beef_paths = cases[cases.sample_id.eq(BEEF_CASE)].copy()
    if len(beef_paths) != 2:
        raise ValueError("The original beef_carpaccio case must retain its two full paths")
    case_provenance = {**provenance, "source_csv": str(paths["cases"].relative_to(ROOT)), "source_csv_sha256": hashes["cases"]}
    for name, value in case_provenance.items():
        beef_paths[name] = value
    beef_paths.to_csv(out / "beef_carpaccio_original_paths2.csv", index=False)
    beef_events = [event for event in events if event["sample_id"] == BEEF_CASE]
    if len(beef_events) != 9:
        raise ValueError("The original beef_carpaccio case must retain its nine saved positions")
    with (out / "beef_carpaccio_original_positions9.jsonl").open("x", encoding="utf-8") as stream:
        for event in beef_events:
            stream.write(json.dumps({**event, "summary_provenance": case_provenance}, ensure_ascii=False, allow_nan=False) + "\n")
    case_rows = []
    for event in beef_events:
        path = beef_paths[beef_paths.condition.eq(event["path_condition"])].iloc[0]
        tokens = json.loads(path.tokens)
        actual_branch = {"vcd_unknown": "guided", "instruction_vcd_unknown": "ip"}[event["path_condition"]]
        if event["prefix_tokens"] != tokens[:event["position"]] or tokens[event["position"]] != event["argmax"][actual_branch]:
            raise ValueError("The original beef case prefix or saved path argmax differs")
        case_rows.append({"model": event["model"], "sample_id": event["sample_id"], "path_condition": event["path_condition"],
                          "position": event["position"], "historical_token_id": tokens[event["position"]],
                          "prefix_tokens": json.dumps(event["prefix_tokens"]), "seed": event["seed"],
                          **{f"argmax_{key}": value for key, value in event["argmax"].items()},
                          **event["support"], "guided_common_argmax": event["distributions"]["guided_on_common_support"]["argmax"],
                          "ip_common_argmax": event["distributions"]["ip_on_common_support"]["argmax"],
                          "source_path": event["source_path"], "source_line": event["source_line"],
                          "events_artifact_line": event["events_artifact_line"], "prefix_matches_original_path": True,
                          "historical_matches_saved_path_argmax": True, **case_provenance})
    pd.DataFrame(case_rows).to_csv(out / "beef_carpaccio_original_position_table9.csv", index=False)
    figures(proxy, out, paths["easyplot_runtime"], provenance)
    if hashes != {name: sha(paths[name]) for name in hashes}:
        raise ValueError("An existing input changed during the CPU summary")
    top = [row for row in summaries if row["model"] == "ALL" and row["position_group"] == "all_positions"]
    receipt = {"passed": True, "generated_utc": datetime.now(timezone.utc).isoformat(), "source_hashes": hashes,
               "source_pair_rows": len(pair), "proxy_pair_rows": len(proxy), "nonproxy_pair_rows": len(pair) - len(proxy),
               "saved_event_positions": len(events), "saved_case_paths": len(cases),
               "proxy_common_support_pairs": int(proxy.pair_in_common_support.sum()),
               "algebra_max_abs_residuals": residuals, "algebra_abs_tolerance": ALGEBRA_TOLERANCE,
               "numeric_zero_abs_tolerance": ZERO_TOLERANCE, "panel_summaries": top,
               "unique_input_unit": "model plus sample_id; candidate pairs/positions are repeated readings",
               "position_unit": "saved event record; identical first prefixes in case paths remain separately recorded",
               "case_pair_margin_gap": "case12 has 101 saved positions and 24 paths; candidate-pair CSV contains zero case12 rows",
               "limits": "Token proxies and local margins do not constitute full semantic abstention rates or population causal effects; historical argmax-source audit remains separate",
               "command": command, **provenance, "software": {"python": sys.version, "numpy": np.__version__,
                                                                "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
               "new_model_generations": 0, "GPU_initialized": False, "new_dev_selected_configuration": False}
    (out / "analysis_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "pair_rows": len(pair), "proxy_rows": len(proxy), "residuals": residuals,
                      "panel_counts": [{key: row[key] for key in ("panel", "support_scope", "proxy_pair_n", "unique_model_input_n",
                                        "recorded_position_n", "ip_minus_guided_positive_n", "ip_minus_guided_negative_n",
                                        "ip_minus_guided_zero_n")} for row in top]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
