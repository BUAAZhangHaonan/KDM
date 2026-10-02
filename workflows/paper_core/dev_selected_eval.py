"""Freeze dev-selected phrases, then read the corresponding frozen Food eval rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import atomic_json, file_hash, within
from kdm.prompts import MARKERS
from workflows.main_results.analysis import BOOT, SEED, bootstrap_class_ci, bootstrap_ratio_diff_by_cluster
from workflows.paper_core.dev_viz import DEV_METHODS, MODELS, roster
from workflows.paper_core.score_native import DEFAULT_DIRECT
from workflows.paper_core.select_phrase import select
from workflows.supplemental.remaining11.score import rows

NATIVE = "outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v6_all5_literal_root_closed/score_rows.jsonl.gz"
FIELDS = ("model", "method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")


def ratio(a, b):
    return a / b if b else None


def checked(frame, expected_ids):
    if len(frame) != len(expected_ids) or set(frame.sample_id) != expected_ids or frame.sample_id.duplicated().any():
        raise ValueError("A complete frozen eval condition must cover the exact 2424 IDs")
    if frame[["correct_canonical", "abstain", "uniform_reference"]].isna().any().any():
        raise ValueError("Unresolved values cannot enter the selected eval comparison")
    if (frame.correct_canonical.eq(1) & frame.abstain.eq(True)).any():
        raise ValueError("Correct and abstention events must be disjoint")
    quota = frame.groupby("target_class").size()
    if len(quota) != 101 or not quota.eq(24).all():
        raise ValueError("Eval class quotas differ from 101 by 24")
    return frame.set_index("sample_id", drop=False).sort_index()


def metrics(frame, identity, direct):
    correct = frame.correct_canonical.astype(bool)
    abstain = frame.abstain.astype(bool)
    reference = frame.uniform_reference.astype(bool)
    n, c, a, r = len(frame), int(correct.sum()), int(abstain.sum()), int(reference.sum())
    tp = int((abstain & reference).sum())
    fp = a - tp
    fixed = direct.abstain.astype(bool) & reference
    retained = int((abstain & fixed).sum())
    fixed_n = int(fixed.sum())
    return dict(**identity, n=n, C=c, W=n-c-a, A=a, TP=tp, FP=fp, FN=r-tp, TN=n-r-fp,
                reference_positive=r, accuracy=c/n, precision=ratio(tp,a), recall=ratio(tp,r),
                F1=ratio(2*tp,a+r), J=(c+tp)/n, J_numerator=c+tp,
                unnecessary_abstention_rate=ratio(fp,n-r), unnecessary_abstention_denominator=n-r,
                fixed_reasonable_retained=retained, fixed_reasonable_denominator=fixed_n,
                fixed_reasonable_retention=ratio(retained,fixed_n),
                fixed_reasonable_baseline_condition_id=direct.attrs.get("condition_id"),
                fixed_reasonable_baseline_marker=direct.attrs.get("condition_identity", {}).get("marker", "NONE"),
                fixed_reasonable_baseline_guided=direct.attrs.get("condition_identity", {}).get("guided", False),
                precision_undefined_reason="no_abstentions" if not a else "",
                recall_undefined_reason="no_reference_positive" if not r else "",
                fixed_retention_undefined_reason="empty_fixed_direct_reasonable_set" if not fixed_n else "")


def paired(a, b, identity, draws, fixed_direct=None):
    if not a.index.equals(b.index) or not a.uniform_reference.equals(b.uniform_reference):
        raise ValueError("Paired comparison input or reference identity differs")
    ca, cb = a.correct_canonical.astype(int), b.correct_canonical.astype(int)
    aa, ab, ref = a.abstain.astype(bool), b.abstain.astype(bool), a.uniform_reference.astype(bool)
    ja, jb = ca + (aa & ref).astype(int), cb + (ab & ref).astype(int)
    classes = sorted(a.target_class.unique())
    effects = []
    for name, delta in (("accuracy", ca-cb), ("J", ja-jb)):
        by_class = {cls: delta[a.target_class.eq(cls)].tolist() for cls in classes}
        summary = bootstrap_class_ci(by_class, draws)
        effects.append(dict(**identity, metric=name, numerator=int(delta.sum()), denominator=len(a),
                            delta=summary["estimate"], ci95_lower=summary["ci95"][0], ci95_upper=summary["ci95"][1],
                            bootstrap_replicates=BOOT, bootstrap_seed=SEED, bootstrap_unit="food_class"))
    recall = {cls: (int(((aa.astype(int)-ab.astype(int))*ref)[a.target_class.eq(cls)].sum()),
                    int(ref[a.target_class.eq(cls)].sum())) for cls in classes}
    summary = bootstrap_ratio_diff_by_cluster(recall, draws, classes)
    effects.append(dict(**identity, metric="abstention_recall", numerator=summary["numerator"],
                        denominator=summary["denominator"], delta=summary["estimate"],
                        ci95_lower=summary["ci95"][0], ci95_upper=summary["ci95"][1],
                        bootstrap_replicates=BOOT, bootstrap_seed=SEED, bootstrap_unit="food_class"))
    if fixed_direct is not None:
        if not fixed_direct.index.equals(a.index) or not fixed_direct.uniform_reference.equals(a.uniform_reference):
            raise ValueError("The fixed reasonable-abstention set has different input or reference identity")
        fixed = fixed_direct.abstain.astype(bool) & ref
        counts = {cls: (int(((aa.astype(int)-ab.astype(int))*fixed)[a.target_class.eq(cls)].sum()),
                        int(fixed[a.target_class.eq(cls)].sum())) for cls in classes}
        summary = bootstrap_ratio_diff_by_cluster(counts, draws, classes)
        effects.append(dict(**identity, metric="fixed_reasonable_retention", numerator=summary["numerator"],
                            denominator=summary["denominator"], delta=summary["estimate"],
                            ci95_lower=summary["ci95"][0], ci95_upper=summary["ci95"][1],
                            method_retained=int((aa & fixed).sum()), baseline_retained=int((ab & fixed).sum()),
                            fixed_reasonable_baseline_condition_id=fixed_direct.attrs["condition_id"],
                            undefined_reason="empty_fixed_direct_reasonable_set" if not summary["denominator"] else "",
                            bootstrap_replicates=BOOT, bootstrap_seed=SEED, bootstrap_unit="food_class"))
    before = np.where(ab, "A", np.where(cb.eq(1), "C", "E"))
    after = np.where(aa, "A", np.where(ca.eq(1), "C", "E"))
    transitions = []
    for reference in (False, True):
        for old in ("C", "E", "A"):
            for new in ("C", "E", "A"):
                count = int(((ref.to_numpy() == reference) & (before == old) & (after == new)).sum())
                transitions.append(dict(**identity, uniform_reference=reference, from_state=old,
                                        to_state=new, n=count, all_input_n=len(a), reference_stratum_n=int(ref.eq(reference).sum())))
    if sum(row["n"] for row in transitions) != len(a):
        raise ValueError("Paired C/W/A transition counts do not close")
    return effects, transitions


def validate_dev_panel(dev, models, dev_ids):
    required = ["model", "method", "marker", "split", "sample_id", "correct_canonical", "abstain", "uniform_reference"]
    if len(dev_ids) != 404:
        raise ValueError("The registered dev roster must contain exactly 404 inputs")
    if set(required) - set(dev.columns):
        raise ValueError("The dev score panel lacks required identifiers or decisions")
    if not dev.model.isin(MODELS).all():
        raise ValueError("The dev score panel contains an unregistered model")
    panel = dev[dev.model.isin(models)].copy()
    if len(panel) != 6464 * len(models) or set(panel.model) != set(models):
        raise ValueError("Every requested model requires its complete 6464-row dev panel")
    if panel[required].isna().any().any():
        raise ValueError("The requested dev models contain unresolved identifiers or decisions")
    if set(panel.split) != {"dev"}:
        raise ValueError("Dev selection must not read eval decisions")
    if panel.duplicated(["model", "method", "marker", "sample_id"]).any():
        raise ValueError("The requested dev panel contains duplicate decision keys")
    groups = panel.groupby(["model", "method", "marker"])
    expected = {(model, method, marker) for model in models for method in DEV_METHODS for marker in MARKERS}
    if set(groups.groups) != expected:
        raise ValueError("Each requested model requires the four-method four-phrase dev panel")
    for key, group in groups:
        if len(group) != 404 or set(group.sample_id) != dev_ids:
            raise ValueError(f"The registered dev404 roster is incomplete for {key}")
    return panel, required


def preserve_frozen_selections(recomputed, frozen, models):
    if not isinstance(frozen, list) or not frozen:
        raise ValueError("Frozen selections must be a nonempty selected_configs JSON list")
    indexed = {}
    for row in frozen:
        if not isinstance(row, dict):
            raise ValueError("Each frozen selection must be a configuration record")
        key = (row.get("model"), row.get("method"))
        if key[0] not in MODELS or key[1] not in DEV_METHODS or row.get("marker") not in MARKERS or key in indexed:
            raise ValueError("Frozen selections contain an unregistered or duplicate configuration")
        indexed[key] = row
    expected = {(model, method) for model in models for method in DEV_METHODS}
    if {key for key in indexed if key[0] in models} != expected:
        raise ValueError("Frozen selections must cover all methods of every requested model")
    compare = ("model", "method", "marker", "n", "correct", "abstentions", "reference_positive", "tp", "successes",
               "selection_utility", "accuracy", "precision", "recall", "marker_order", "selection_split", "selection_rule")
    chosen = []
    for current in recomputed:
        previous = indexed[(current["model"], current["method"])]
        if any(key not in previous or previous[key] != current[key] for key in compare):
            raise ValueError(f"Frozen selection or dev counts changed for {current['model']}/{current['method']}")
        if previous.get("selection") != "dev_selected" or any(
                not isinstance(previous.get(key), str) or not previous[key]
                for key in ("selection_frozen_at_utc", "source_scores", "source_scores_sha256")):
            raise ValueError("Frozen selections lack their original selection time or score provenance")
        chosen.append(dict(previous))
    if len(chosen) != len(expected):
        raise ValueError("Recomputed dev choices do not cover the requested frozen configurations")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-scores", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--native-scores", default=NATIVE)
    parser.add_argument("--direct-scores", default=DEFAULT_DIRECT + "/score_rows.jsonl.gz")
    parser.add_argument("--models", nargs="+", choices=MODELS,
                        help="Freeze only the complete dev panels of these registered models; default: all five")
    parser.add_argument("--frozen-selections", help="Verify and retain a previously frozen selected_configs JSON")
    args = parser.parse_args()
    models = args.models if args.models is not None else list(MODELS)
    if len(models) != len(set(models)):
        parser.error("--models must contain distinct registered models")
    if BOOT != 2000 or SEED != 20260929:
        raise ValueError("The preregistered paired bootstrap identity changed")
    out = within(ROOT, args.out)
    out.mkdir(parents=True, exist_ok=False)
    dev_path = within(ROOT, args.dev_scores)
    dev = pd.read_parquet(dev_path)
    rename = {"canonical_name_in_primary_score": "correct_canonical"}
    dev = dev.rename(columns={key: value for key, value in rename.items() if key in dev and value not in dev})
    _, dev_samples = roster("dev404")
    dev_ids = {row["id"] for row in dev_samples}
    source_dev_rows = len(dev)
    dev, required = validate_dev_panel(dev, models, dev_ids)
    dev[required].to_csv(out / "dev_decisions.csv", index=False)
    dev_metrics, chosen = select(dev[required], dev_ids)
    analyzed_at = datetime.now(timezone.utc).isoformat()
    dev_sha = file_hash(dev_path)
    frozen_path = within(ROOT, args.frozen_selections) if args.frozen_selections else None
    if frozen_path:
        chosen = preserve_frozen_selections(chosen, json.loads(frozen_path.read_text(encoding="utf-8")), models)
    else:
        for row in chosen:
            row.update(selection="dev_selected", selection_frozen_at_utc=analyzed_at,
                       source_scores=str(dev_path.relative_to(ROOT)), source_scores_sha256=dev_sha)
    selection_times = sorted({row["selection_frozen_at_utc"] for row in chosen})
    selected_at = selection_times[0] if len(selection_times) == 1 else None
    if len(chosen) != 4 * len(models):
        raise ValueError("Expected one dev-selected configuration per model and method")
    dev_metrics.to_csv(out / "dev_metrics.csv", index=False)
    atomic_json(out / "selected_configs.json", chosen)

    frozen_dir = ROOT / "outputs/paper_20260929"
    conditions = pd.read_csv(frozen_dir / "conditions.csv")
    scores = pd.read_parquet(frozen_dir / "scores.parquet")
    native = pd.DataFrame([row for _, row, _ in rows(within(ROOT, args.native_scores))]).rename(columns={
        "canonical_name_in_primary_score": "correct_canonical", "literal_extracted_name_score": "correct_literal"})
    direct_plain = pd.DataFrame([row for _, row, _ in rows(within(ROOT, args.direct_scores))])
    native = native[native.model.isin(models)].copy()
    direct_plain = direct_plain[direct_plain.model.isin(models)].copy()
    restored_direct_fields = []
    for key, value in {"marker": "NONE", "reference_marker": "NONE", "replicate": 0}.items():
        if key not in direct_plain:
            direct_plain[key] = value
            restored_direct_fields.append(key)
    direct_identity_source = "workflows/paper_core/score_native.py:comparison_tables:301:accepted_unguided_Direct_identity"
    reference_frame = pd.read_parquet(frozen_dir / "references.parquet")
    reference = {(row.model, row.sample_id): bool(row.uniform_reference) for row in reference_frame.itertuples()
                 if row.split == "eval" and row.model in models}
    eval_ids = set(native.sample_id)
    if len(eval_ids) != 2424 or len(native) != 2424 * len(models):
        raise ValueError("Native eval baseline is incomplete for the requested models")
    draws = np.random.RandomState(SEED).randint(0, 101, size=(BOOT, 101))
    pool, direct_cache, native_cache, native_metadata, metrics_all = {}, {}, {}, {}, []
    selected_metrics, effects, transitions, best_rows, native_matched = [], [], [], [], []

    def frozen_condition(model, method, marker):
        kind = "instruction_preserving" if method in {"instruction_vcd", "cda_visual"} else "main"
        ref_guided = method in {"vcd", "m3id", "direct"}
        found = conditions[conditions.model.eq(model) & conditions.method.eq(method) & conditions.kind.eq(kind)
                           & conditions.marker.eq(marker) & conditions.reference_marker.eq(marker)
                           & conditions.guided.eq(True) & conditions.reference_guided.eq(ref_guided)
                           & conditions.replicate.eq(0)]
        if len(found) != 1:
            raise ValueError("Selected method must match one complete registered eval identity")
        meta = found.iloc[0].to_dict()
        cid = int(meta["condition_id"])
        frame = checked(scores[scores.condition_id.eq(cid)].copy(), eval_ids)
        if any(bool(row.uniform_reference) != reference[(model, row.sample_id)] for row in frame.itertuples()):
            raise ValueError("Frozen eval reference differs from the accepted uniform reference")
        frame.attrs["condition_id"] = cid
        frame.attrs["condition_identity"] = {key: meta[key] for key in FIELDS}
        return meta, frame

    for model in models:
        model_spec = json.loads((ROOT / f"configs/runtime/{model}.json").read_text())
        checkpoint = model_spec["hf_model_id"]
        n = checked(native[native.model.eq(model)].copy(), eval_ids)
        d = checked(direct_plain[direct_plain.model.eq(model)].copy(), eval_ids)
        native_cache[model] = n
        for method in DEV_METHODS:
            for marker in MARKERS:
                meta, frame = frozen_condition(model, method, marker)
                if (model, marker) not in direct_cache:
                    direct_cache[(model, marker)] = frozen_condition(model, "direct", marker)[1]
                identity = {key: meta[key] for key in FIELDS}
                identity.update(condition_id=int(meta["condition_id"]), checkpoint=checkpoint, split="eval", dataset="food101", main_marker=marker,
                                selection="registered_eval_observation", score_source="outputs/paper_20260929/scores.parquet")
                metric = metrics(frame, identity, direct_cache[(model, marker)])
                pool[(model, method, marker)] = (identity, frame, metric)
                metrics_all.append(metric)
        for marker in MARKERS:
            guided_direct = direct_cache[(model, marker)]
            identity = {**guided_direct.attrs["condition_identity"], "condition_id": guided_direct.attrs["condition_id"],
                        "checkpoint": checkpoint, "dataset": "food101", "split": "eval", "main_marker": marker,
                        "selection": "registered_eval_observation", "score_source": "outputs/paper_20260929/scores.parquet"}
            metrics_all.append(metrics(guided_direct, identity, guided_direct))
        for method, frame in (("native_vcd", n), ("direct_unguided", d)):
            d.attrs["condition_identity"] = {key: d[key].iloc[0] for key in FIELDS}
            if any(frame[key].nunique(dropna=False) != 1 for key in FIELDS):
                raise ValueError("A native baseline contains more than one original condition identity")
            identity = {key: frame[key].iloc[0] for key in FIELDS}
            identity.update(display_method=method, main_marker=identity["marker"],
                            condition_id=None, checkpoint=checkpoint, dataset="food101", split="eval",
                            selection="single_registered_native_configuration", score_source=args.native_scores if method == "native_vcd" else args.direct_scores)
            identity.update(condition_identity_restored_fields="|".join(restored_direct_fields) if method == "direct_unguided" else "",
                            condition_identity_recovery_source=direct_identity_source if method == "direct_unguided" and restored_direct_fields else "")
            metric = metrics(frame, identity, d)
            metrics_all.append(metric)
            selected_metrics.append(metric)
            if method == "native_vcd":
                native_metadata[model] = identity
    for choice in chosen:
        model, method, marker = choice["model"], choice["method"], choice["marker"]
        identity, frame, metric = pool[(model, method, marker)]
        selected_metrics.append({**metric, "selection": "dev_selected", "dev_n": choice["n"],
                                 "dev_C": choice["correct"], "dev_TP": choice["tp"], "dev_J": choice["selection_utility"],
                                 "selection_frozen_at_utc": choice["selection_frozen_at_utc"],
                                 "selection_source_scores": choice["source_scores"],
                                 "selection_source_scores_sha256": choice["source_scores_sha256"]})
        fixed_direct = direct_cache[(model, marker)]
        native_matched.append({**metrics(native_cache[model], native_metadata[model], fixed_direct),
                               "matched_method": method, "dev_selected_marker": marker,
                               "matched_selected_condition_id": identity["condition_id"],
                               "fixed_set_scope": "same_marker_guided_Direct_uniform_R_positive"})
        pairs = [("dev_selected_vs_native_VCD", native_cache[model], None)]
        if method == "instruction_vcd":
            direct_identity = {**fixed_direct.attrs["condition_identity"], "condition_id": fixed_direct.attrs["condition_id"],
                               "checkpoint": identity["checkpoint"], "dataset": "food101", "split": "eval", "main_marker": marker,
                               "selection": "matched_to_dev_selected_IP", "score_source": "outputs/paper_20260929/scores.parquet"}
            selected_metrics.append(metrics(fixed_direct, direct_identity, fixed_direct))
            guided_choice = next(row for row in chosen if row["model"] == model and row["method"] == "vcd")
            guided_identity, guided, _ = pool[(model, "vcd", guided_choice["marker"])]
            pairs.append(("IP_vs_same_budget_dev_selected_guided_VCD", guided, guided_identity["condition_id"]))
            same_identity, same_guided, _ = pool[(model, "vcd", marker)]
            pairs.append(("IP_vs_same_marker_guided_VCD", same_guided, same_identity["condition_id"]))
        for label, base, base_cid in pairs:
            context = {**identity, "comparison": label, "baseline_condition_id": base_cid,
                       "selection": "dev_selected", "paired_n": 2424,
                       "selection_frozen_at_utc": choice["selection_frozen_at_utc"],
                       "fixed_set_direct_condition_id": fixed_direct.attrs["condition_id"]}
            pair_effects, pair_transitions = paired(frame, base, context, draws, fixed_direct)
            effects.extend(pair_effects)
            transitions.extend(pair_transitions)
        observations = [pool[(model, method, item)][2] for item in MARKERS]
        for name in ("accuracy", "precision", "recall", "F1", "J", "fixed_reasonable_retention"):
            valid = [row for row in observations if row[name] is not None]
            winner = min(valid, key=lambda row: (-row[name], MARKERS.index(row["marker"]))) if valid else None
            numerator_field, denominator_field = {
                "accuracy": ("C", "n"), "precision": ("TP", "A"), "recall": ("TP", "reference_positive"),
                "J": ("J_numerator", "n"), "fixed_reasonable_retention": ("fixed_reasonable_retained", "fixed_reasonable_denominator"),
                "F1": (None, None)}[name]
            numerator = (2*winner["TP"] if name == "F1" else winner[numerator_field]) if winner else None
            denominator = (winner["A"]+winner["reference_positive"] if name == "F1" else winner[denominator_field]) if winner else 0
            best_rows.append(dict(model=model, method=method, metric=name, selection="best_observed_eval",
                                  marker=winner["marker"] if winner else None, condition_id=winner["condition_id"] if winner else None,
                                  value=winner[name] if winner else None, dev_selected_marker=marker,
                                  dev_selected_condition_id=identity["condition_id"], dev_selected_value=metric[name],
                                  best_minus_selected=winner[name]-metric[name] if winner and metric[name] is not None else None,
                                  numerator=numerator, denominator=denominator, all_input_n=2424,
                                  undefined_reason="no_candidate_with_nonzero_metric_denominator" if not winner else ""))
    pd.DataFrame(metrics_all).to_csv(out / "metrics_all.csv", index=False)
    pd.DataFrame(selected_metrics).to_csv(out / "selected_operating_points.csv", index=False)
    pd.DataFrame(native_matched).to_csv(out / "native_matched_fixed_sets.csv", index=False)
    pd.DataFrame(best_rows).to_csv(out / "best_observed_endpoints.csv", index=False)
    pd.DataFrame(effects).to_csv(out / "paired_effects.csv", index=False)
    pd.DataFrame(transitions).to_csv(out / "transitions.csv", index=False)
    atomic_json(out / "analysis_receipt.json", dict(passed=True, dev_rows=len(dev), selected_configs=len(chosen),
                models=models, registered_models=list(MODELS), full_five_model_panel=set(models) == set(MODELS),
                source_dev_rows=source_dev_rows, expected_dev_rows=6464 * len(models), dev_scores_sha256=dev_sha,
                eval_selected_method_conditions=4 * len(models), native_baseline_conditions=len(models), guided_direct_conditions=4 * len(models),
                guided_direct_selected_workpoints=len(models), native_matched_fixed_sets=len(native_matched), paired_comparisons=len(effects)//4,
                selection_frozen_at_utc=selected_at, dev_only_selection=True, bootstrap_replicates=BOOT, bootstrap_seed=SEED,
                selection_frozen_times_utc=selection_times, analysis_created_at_utc=analyzed_at,
                frozen_selections_verified=frozen_path is not None,
                frozen_selections_source=str(frozen_path.relative_to(ROOT)) if frozen_path else None,
                frozen_selections_source_sha256=file_hash(frozen_path) if frozen_path else None,
                frozen_scores_sha256=file_hash(frozen_dir / "scores.parquet"), native_source_sha256=file_hash(within(ROOT, args.native_scores)),
                direct_source_sha256=file_hash(within(ROOT, args.direct_scores)),
                direct_condition_identity_restored_fields=restored_direct_fields,
                direct_condition_identity_recovery_source=direct_identity_source,
                direct_condition_identity_recovery_source_sha256=file_hash(ROOT / "workflows/paper_core/score_native.py"),
                script_sha256=file_hash(Path(__file__)), new_model_generations=0, GPU_initialized=False))
    print(json.dumps(dict(passed=True, models=models, selected_configs=len(chosen), output=str(out.relative_to(ROOT))), allow_nan=False))


if __name__ == "__main__":
    main()
