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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-scores", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--native-scores", default=NATIVE)
    parser.add_argument("--direct-scores", default=DEFAULT_DIRECT + "/score_rows.jsonl.gz")
    args = parser.parse_args()
    if BOOT != 2000 or SEED != 20260929:
        raise ValueError("The preregistered paired bootstrap identity changed")
    out = within(ROOT, args.out)
    out.mkdir(parents=True, exist_ok=False)
    dev_path = within(ROOT, args.dev_scores)
    dev = pd.read_parquet(dev_path)
    rename = {"canonical_name_in_primary_score": "correct_canonical"}
    dev = dev.rename(columns={key: value for key, value in rename.items() if key in dev and value not in dev})
    required = ["model", "method", "marker", "split", "sample_id", "correct_canonical", "abstain", "uniform_reference"]
    _, dev_samples = roster("dev404")
    dev_ids = {row["id"] for row in dev_samples}
    if len(dev) != 32320 or set(dev.model) != set(MODELS) or set(dev.method) != set(DEV_METHODS):
        raise ValueError("Dev selection requires the complete symmetric five-model four-method panel")
    if set(dev.split) != {"dev"}:
        raise ValueError("Dev selection must not read eval decisions")
    dev[required].to_csv(out / "dev_decisions.csv", index=False)
    dev_metrics, chosen = select(dev[required], dev_ids)
    selected_at = datetime.now(timezone.utc).isoformat()
    for row in chosen:
        row.update(selection="dev_selected", selection_frozen_at_utc=selected_at,
                   source_scores=str(dev_path.relative_to(ROOT)), source_scores_sha256=file_hash(dev_path))
    if len(chosen) != 20:
        raise ValueError("Expected one dev-selected configuration per model and method")
    dev_metrics.to_csv(out / "dev_metrics.csv", index=False)
    atomic_json(out / "selected_configs.json", chosen)

    frozen_dir = ROOT / "outputs/paper_20260929"
    conditions = pd.read_csv(frozen_dir / "conditions.csv")
    scores = pd.read_parquet(frozen_dir / "scores.parquet")
    native = pd.DataFrame([row for _, row, _ in rows(within(ROOT, args.native_scores))]).rename(columns={
        "canonical_name_in_primary_score": "correct_canonical", "literal_extracted_name_score": "correct_literal"})
    direct_plain = pd.DataFrame([row for _, row, _ in rows(within(ROOT, args.direct_scores))])
    restored_direct_fields = []
    for key, value in {"marker": "NONE", "reference_marker": "NONE", "replicate": 0}.items():
        if key not in direct_plain:
            direct_plain[key] = value
            restored_direct_fields.append(key)
    direct_identity_source = "workflows/paper_core/score_native.py:comparison_tables:301:accepted_unguided_Direct_identity"
    reference_frame = pd.read_parquet(frozen_dir / "references.parquet")
    reference = {(row.model, row.sample_id): bool(row.uniform_reference) for row in reference_frame.itertuples() if row.split == "eval"}
    eval_ids = set(native.sample_id)
    if len(eval_ids) != 2424 or len(native) != 12120:
        raise ValueError("Native eval baseline is not the verified complete five-model result")
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

    for model in MODELS:
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
                                 "selection_frozen_at_utc": selected_at})
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
                eval_selected_method_conditions=20, native_baseline_conditions=5, guided_direct_conditions=20,
                guided_direct_selected_workpoints=5, native_matched_fixed_sets=len(native_matched), paired_comparisons=len(effects)//4,
                selection_frozen_at_utc=selected_at, dev_only_selection=True, bootstrap_replicates=BOOT, bootstrap_seed=SEED,
                frozen_scores_sha256=file_hash(frozen_dir / "scores.parquet"), native_source_sha256=file_hash(within(ROOT, args.native_scores)),
                direct_source_sha256=file_hash(within(ROOT, args.direct_scores)),
                direct_condition_identity_restored_fields=restored_direct_fields,
                direct_condition_identity_recovery_source=direct_identity_source,
                direct_condition_identity_recovery_source_sha256=file_hash(ROOT / "workflows/paper_core/score_native.py"),
                script_sha256=file_hash(Path(__file__)), new_model_generations=0, GPU_initialized=False))
    print(json.dumps(dict(passed=True, selected_configs=20, output=str(out.relative_to(ROOT))), allow_nan=False))


if __name__ == "__main__":
    main()
