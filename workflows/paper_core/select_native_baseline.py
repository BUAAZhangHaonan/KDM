#!/usr/bin/env python3
"""Compare complete native VCD with endpoint-wise observed same-marker IP-VCD."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, stable_hash, within
from workflows.main_results.analysis import BOOT, SEED, bootstrap_class_ci
from workflows.paper_core.receive_native import MODELS
from workflows.paper_core.score_native import load_references, rows, save_json

MARKERS = ("UNKNOWN", "UNCLEAR", "UNSURE", "I cannot identify it")
ENDPOINTS = ("accuracy", "reasonable_abstention_retention", "abstention_precision", "abstention_recall")
DEFAULT_SCORE = "outputs/paper_core_20260930/run_20260930_core_p1_cpu/scores/v6_all5_literal_root_closed"
FROZEN = ROOT / "outputs/paper_20260929"
FROZEN_RECOMPUTED = ROOT / "outputs/paper_core_20260930/run_20260930_core_p0/frozen/all_conditions_352.csv"


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def validate_frame(frame, reference, model):
    frame = frame.set_index("sample_id").sort_index()
    target = reference[reference.model.eq(model)].set_index("sample_id").sort_index()
    if len(frame) != 2424 or frame.index.duplicated().any() or not frame.index.equals(target.index):
        raise ValueError(f"Incomplete or different eval sample keys: {model}")
    if not frame.target_class.equals(target.target_class) or not frame.uniform_reference.equals(target.uniform_reference):
        raise ValueError(f"Frozen category or uniform-reference binding differs: {model}")
    quotas = frame.groupby("target_class").size()
    if len(quotas) != 101 or not quotas.eq(24).all():
        raise ValueError(f"Food-101 eval class quota differs: {model}")
    for field in ("correct_canonical", "abstain", "uniform_reference"):
        if frame[field].isna().any() or not frame[field].map(lambda x: x in (0, 1, False, True)).all():
            raise ValueError(f"Unresolved primary or behavior labels: {model}/{field}")
    return frame


def metrics(frame, fixed):
    correct = frame.correct_canonical.astype(bool)
    abstain = frame.abstain.astype(bool)
    reference = frame.uniform_reference.astype(bool)
    tp, fp = int((abstain & reference).sum()), int((abstain & ~reference).sum())
    endpoints = {
        "accuracy": (int(correct.sum()), len(frame)),
        "reasonable_abstention_retention": (int((abstain & fixed).sum()), int(fixed.sum())),
        "abstention_precision": (tp, tp + fp),
        "abstention_recall": (tp, int(reference.sum())),
    }
    return {"correct": int(correct.sum()), "abstentions": int(abstain.sum()),
            "reference_positive": int(reference.sum()), "tp": tp, "fp": fp,
            "fn": int((~abstain & reference).sum()), "endpoints": endpoints,
            "joint_correct_or_reasonable_abstention": ratio(int(correct.sum()) + tp, len(frame))}


def display(numerator, denominator):
    return f"{100 * numerator / denominator:.4f}% ({numerator}/{denominator})" if denominator else "NA (0/0: zero denominator)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", default=DEFAULT_SCORE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-paired-from")
    args = parser.parse_args()
    score, output = within(ROOT, args.score_dir), within(ROOT, args.output)
    output.mkdir(parents=True, exist_ok=False)
    if (BOOT, SEED) != (2000, 20260929):
        raise ValueError("Registered class-cluster bootstrap differs")
    source_verification = json.loads((score / "verification.json").read_text())
    score_hash = file_hash(score / "score_rows.jsonl.gz")
    frozen_scores_hash = file_hash(FROZEN / "scores.parquet")
    if (source_verification["passed"] is not True or source_verification["native_rows"] != 12120
            or source_verification["score_rows_sha256"] != score_hash
            or any(source_verification["frozen_counts_and_metrics_mismatches"].values())):
        raise ValueError("Core native scores lack complete frozen-zero-diff verification")
    native = pd.DataFrame([row for _, row, _ in rows(score / "score_rows.jsonl.gz")])
    native = native.rename(columns={"canonical_name_in_primary_score": "correct_canonical"})
    if (len(native) != 12120 or set(native.model) != set(MODELS)
            or not native.dataset.eq("food101").all() or not native.split.eq("eval").all()
            or not native.method.eq("vcd").all() or not native.kind.eq("native_unguided").all()
            or not native.marker.eq("NONE").all() or not native.reference_marker.eq("NONE").all()
            or native.guided.any() or native.reference_guided.any() or not native.replicate.eq(0).all()):
        raise ValueError("Native main-baseline condition identity differs")
    references, _ = load_references()
    model_inventory_path = ROOT / "configs/kdm/models.json"
    model_inventory = {item["key"]: item for item in json.loads(model_inventory_path.read_text())}
    reused_statistics = {}
    if args.reuse_paired_from:
        previous_folder = within(ROOT, args.reuse_paired_from)
        previous_verification = json.loads((previous_folder / "verification.json").read_text())
        if (previous_verification["passed"] is not True
                or previous_verification["sources"]["native_scores"]["sha256"] != score_hash
                or previous_verification["sources"]["frozen_scores"]["sha256"] != frozen_scores_hash
                or previous_verification["bootstrap_replicates"] != BOOT
                or previous_verification["bootstrap_seed"] != SEED):
            raise ValueError("Previous paired statistics do not bind to identical score sources and registered bootstrap")
        previous_statistics_path = previous_folder / "native_vs_selected_IP_accuracy_paired.json"
        if file_hash(previous_statistics_path) != previous_verification["outputs"][previous_statistics_path.name]:
            raise ValueError("Previous paired-statistics file differs from its receipt")
        reused_statistics = {item["model"]: item for item in json.loads(previous_statistics_path.read_text())}
        if set(reused_statistics) != set(MODELS):
            raise ValueError("Previous paired-statistics model coverage differs")
    conditions = pd.read_csv(FROZEN / "conditions.csv")
    ips = conditions[conditions.method.eq("instruction_vcd") & conditions.kind.eq("instruction_preserving")
                     & conditions.marker.eq(conditions.reference_marker) & conditions.guided
                     & ~conditions.reference_guided & conditions.replicate.eq(0) & conditions.split.eq("eval")]
    directs = conditions[conditions.method.eq("direct") & conditions.kind.eq("main") & conditions.guided
                         & conditions.reference_guided & conditions.replicate.eq(0) & conditions.split.eq("eval")]
    if (len(conditions) != 352 or len(ips) != 20 or len(directs) != 20
            or set(ips.marker) != set(MARKERS) or set(directs.marker) != set(MARKERS)):
        raise ValueError("Expected complete four same-marker IP-VCD/Direct cells for each core model")
    selected_ids = sorted(set(ips.condition_id) | set(directs.condition_id))
    frozen_rows = pd.read_parquet(FROZEN / "scores.parquet", filters=[("condition_id", "in", selected_ids)],
                                  columns=["condition_id", "sample_id", "target_class", "correct_canonical",
                                           "abstain", "uniform_reference", "seed"])
    if len(frozen_rows) != 40 * 2424:
        raise ValueError("Complete selected frozen-condition rows differ")
    previous = pd.read_csv(FROZEN_RECOMPUTED).set_index("condition_id")
    results, retention, members, registry, statistics, compact, all_metrics = [], [], [], [], [], [], []
    draws = np.random.RandomState(SEED).randint(0, 101, size=(BOOT, 101))
    for model in MODELS:
        a = validate_frame(native[native.model.eq(model)].copy(), references, model)
        source_ids = sorted(set(a.source_identity))
        config_ids = sorted(set(a.config_sha256))
        native_condition = {"model": model, "dataset": "food101", "split": "eval", "method": "vcd",
                            "kind": "native_unguided", "marker": "NONE", "reference_marker": "NONE",
                            "guided": False, "reference_guided": False, "replicate": 0}
        native_key = "supplemental_native_vcd:" + stable_hash(native_condition)
        registry.append({**native_condition, "hf_model_id": model_inventory[model]["hf_model_id"],
                         "display_name": model_inventory[model]["display_name"],
                         "condition_key": native_key, "frozen_condition_id": None,
                         "n": 2424, "primary_pending": 0, "behavior_pending": 0,
                         "score_source_path": str((score / "score_rows.jsonl.gz").relative_to(ROOT)),
                         "score_source_sha256": score_hash, "generation_source_identities": source_ids,
                         "config_sha256": config_ids, "selection_scope": "main_baseline_authorized_native_unguided_VCD",
                         "replaces_guided_VCD_only_in_new_main_comparison": True})
        candidates = []
        for marker in MARKERS:
            ip_meta = ips[ips.model.eq(model) & ips.marker.eq(marker)]
            d_meta = directs[directs.model.eq(model) & directs.marker.eq(marker)]
            if len(ip_meta) != 1 or len(d_meta) != 1:
                raise ValueError(f"Non-unique comparator identity: {model}/{marker}")
            ip_meta, d_meta = ip_meta.iloc[0], d_meta.iloc[0]
            ip_id, direct_id = int(ip_meta.condition_id), int(d_meta.condition_id)
            b = validate_frame(frozen_rows[frozen_rows.condition_id.eq(ip_id)].copy(), references, model)
            d = validate_frame(frozen_rows[frozen_rows.condition_id.eq(direct_id)].copy(), references, model)
            if not a.seed.equals(b.seed) or not a.seed.equals(d.seed):
                raise ValueError(f"Paired deterministic source seeds differ: {model}/{marker}")
            fixed = d.abstain.astype(bool) & d.uniform_reference.astype(bool)
            native_metrics, ip_metrics = metrics(a, fixed), metrics(b, fixed)
            old = previous.loc[ip_id]
            for key, value in {"n": len(b), "correct": ip_metrics["correct"], "abstentions": ip_metrics["abstentions"],
                               "reference_positive": ip_metrics["reference_positive"], "tp": ip_metrics["tp"],
                               "fp": ip_metrics["fp"], "fn": ip_metrics["fn"],
                               "fixed_original_reference": int(fixed.sum()),
                               "retained": ip_metrics["endpoints"]["reasonable_abstention_retention"][0]}.items():
                if int(old[key]) != value:
                    raise ValueError(f"Previously computed frozen endpoint count changed: {ip_id}/{key}")
            identity = {"model": model, "hf_model_id": model_inventory[model]["hf_model_id"],
                        "display_name": model_inventory[model]["display_name"],
                        "dataset": "food101", "split": "eval", "marker": marker,
                        "ip_condition_id": ip_id, "ip_complete_condition_key": ip_meta.complete_condition_key,
                        "ip_method": "instruction_vcd", "ip_kind": "instruction_preserving",
                        "ip_guided": True, "ip_reference_guided": False,
                        "ip_main_marker": marker, "ip_reference_marker": marker, "replicate": 0,
                        "ip_prompt_id": int(ip_meta.prompt_id), "ip_config_id": int(ip_meta.config_id),
                        "fixed_direct_condition_id": direct_id, "fixed_direct_complete_condition_key": d_meta.complete_condition_key,
                        "native_condition_key": native_key, "native_frozen_condition_id": None, "paired_n": 2424,
                        "fixed_set_definition": "frozen guided Direct abstain AND frozen uniform_reference"}
            rn, rd = native_metrics["endpoints"]["reasonable_abstention_retention"]
            pn, pdn = ip_metrics["endpoints"]["reasonable_abstention_retention"]
            retention.append({**identity, "native_retained_numerator": rn, "ip_retained_numerator": pn,
                              "fixed_set_denominator": rd, "native_retention": ratio(rn, rd), "ip_retention": ratio(pn, pdn),
                              "null_reason": "no_frozen_Direct_reasonable_abstentions" if not rd else None})
            for sid in fixed[fixed].index:
                members.append({"model": model, "marker": marker, "sample_id": sid, "target_class": d.at[sid, "target_class"],
                                "fixed_direct_condition_id": direct_id, "ip_condition_id": ip_id,
                                "direct_abstain": True, "uniform_reference": True,
                                "native_abstain": bool(a.at[sid, "abstain"]), "ip_abstain": bool(b.at[sid, "abstain"])})
            candidates.append({"identity": identity, "native": native_metrics, "ip": ip_metrics, "frame": b})
            all_metrics.append({**identity, "native_correct": native_metrics["correct"], "ip_correct": ip_metrics["correct"],
                                "native_tp": native_metrics["tp"], "ip_tp": ip_metrics["tp"],
                                "native_J_numerator": native_metrics["correct"] + native_metrics["tp"],
                                "ip_J_numerator": ip_metrics["correct"] + ip_metrics["tp"], "J_denominator": 2424,
                                "native_J": native_metrics["joint_correct_or_reasonable_abstention"],
                                "ip_J": ip_metrics["joint_correct_or_reasonable_abstention"]})
        model_compact = {"model": model, "hf_model_id": model_inventory[model]["hf_model_id"]}
        for endpoint in ENDPOINTS:
            valid = [x for x in candidates if x["ip"]["endpoints"][endpoint][1]]
            if not valid:
                raise ValueError(f"All observed IP endpoints undefined: {model}/{endpoint}")
            best_value = max(Fraction(*x["ip"]["endpoints"][endpoint]) for x in valid)
            ties = [x for x in valid if Fraction(*x["ip"]["endpoints"][endpoint]) == best_value]
            chosen = min(ties, key=lambda x: MARKERS.index(x["identity"]["marker"]))
            nn, nd = chosen["native"]["endpoints"][endpoint]
            pn, pdn = chosen["ip"]["endpoints"][endpoint]
            item = {**chosen["identity"], "endpoint": endpoint, "native_numerator": nn, "native_denominator": nd,
                    "native_value": ratio(nn, nd), "native_null_reason": "no_native_abstentions" if not nd else None,
                    "ip_numerator": pn, "ip_denominator": pdn, "ip_value": ratio(pn, pdn),
                    "ip_null_reason": None, "selection": "independently_best_observed_eval_endpoint",
                    "selection_dataset": "food101", "selection_split": "eval", "dev_selected": False,
                    "tied_condition_ids": json.dumps(sorted(x["identity"]["ip_condition_id"] for x in ties)),
                    "tie_candidates": json.dumps([{ "condition_id": x["identity"]["ip_condition_id"],
                                                    "marker": x["identity"]["marker"],
                                                    "numerator": x["ip"]["endpoints"][endpoint][0],
                                                    "denominator": x["ip"]["endpoints"][endpoint][1]}
                                                   for x in sorted(ties, key=lambda x: MARKERS.index(x["identity"]["marker"]))]),
                    "tie_rule": "UNKNOWN_UNCLEAR_UNSURE_I_cannot_identify_it_among_exact_ratio_ties",
                    "endpoint_choices_are_independent_configurations": True}
            results.append(item)
            model_compact[endpoint] = f"{display(nn, nd)} -> {display(pn, pdn)} [{item['marker']}; CID {item['ip_condition_id']}]"
            if endpoint == "accuracy":
                if reused_statistics:
                    prior = reused_statistics[model]
                    if (prior["ip_condition_id"] != item["ip_condition_id"]
                            or prior["native_condition_key"] != native_key or prior["paired_n"] != 2424):
                        raise ValueError("Selected accuracy comparison identity differs from reusable previous statistics")
                    ci = prior["accuracy_difference"]
                else:
                    delta = chosen["frame"].correct_canonical.astype(int) - a.correct_canonical.astype(int)
                    values = {target: delta[a.target_class.eq(target)].tolist() for target in sorted(a.target_class.unique())}
                    ci = bootstrap_class_ci(values, draws)
                statistics.append({**chosen["identity"], "comparison": "observed_best_Acc_IP_VCD_minus_native_VCD",
                                   "accuracy_difference": ci, "difference_unit": "fraction",
                                   "bootstrap_replicates": BOOT, "bootstrap_seed": SEED,
                                   "bootstrap_unit": "Food-101 target class", "classes": 101,
                                   "class_quota": 24, "selection_adjusted": False,
                                   "CI_scope": "paired_response_difference_for_selected_observed_configuration"})
        compact.append(model_compact)
    pd.DataFrame(results).to_csv(output / "main_endpoint_comparison.csv", index=False)
    pd.DataFrame(compact).to_csv(output / "main_endpoint_compact.csv", index=False)
    pd.DataFrame(retention).to_csv(output / "retention_all_four_markers.csv", index=False)
    pd.DataFrame(members).to_csv(output / "retention_fixed_set_memberships.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(output / "joint_correct_or_reasonable_abstention_all_markers.csv", index=False)
    save_json(output / "native_vs_selected_IP_accuracy_paired.json", statistics)
    sources = {"native_scores": {"path": str((score / "score_rows.jsonl.gz").relative_to(ROOT)), "sha256": score_hash},
               "native_verification": {"path": str((score / "verification.json").relative_to(ROOT)), "sha256": file_hash(score / "verification.json")},
               "frozen_conditions": {"path": str((FROZEN / "conditions.csv").relative_to(ROOT)), "sha256": file_hash(FROZEN / "conditions.csv")},
               "frozen_scores": {"path": str((FROZEN / "scores.parquet").relative_to(ROOT)), "sha256": frozen_scores_hash},
               "frozen_reference": {"path": "outputs/paper_20260929/references.parquet", "sha256": file_hash(FROZEN / "references.parquet")},
               "previous_frozen_counts": {"path": str(FROZEN_RECOMPUTED.relative_to(ROOT)), "sha256": file_hash(FROZEN_RECOMPUTED)},
               "model_inventory": {"path": str(model_inventory_path.relative_to(ROOT)), "sha256": file_hash(model_inventory_path)}}
    if args.reuse_paired_from:
        sources["reused_paired_statistics"] = {"path": str(previous_statistics_path.relative_to(ROOT)),
                                               "sha256": file_hash(previous_statistics_path)}
    save_json(output / "selected_main_baseline.json", {"schema": "kdm_core_selected_main_baseline_v1",
              "selected_main_baseline": "native_unguided_VCD", "created_utc": datetime.now(timezone.utc).isoformat(),
              "condition_registry": registry, "frozen_guided_VCD_retained_as_legacy_comparison": True,
              "frozen_panel_modified": False, "paper_body_modified": False, "sources": sources,
              "endpoint_selection": "best_observed_eval_endpoint_separately_for_each_endpoint",
              "exact_ratio_tie_marker_order": list(MARKERS),
              "IP_scope": "four_registered_same_marker_instruction_preserving_VCD_conditions",
              "retention_fixed_set": "each frozen guided Direct marker abstain intersect uniform_reference",
              "zero_denominator_value": None})
    verification = {"schema": "kdm_core_main_native_endpoint_verification_v1", "passed": True,
                    "native_rows": 12120, "native_conditions": 5, "frozen_filtered_rows": len(frozen_rows),
                    "frozen_selected_conditions": 40, "endpoint_rows": len(results), "marker_retention_rows": len(retention),
                    "fixed_set_memberships": len(members), "class_quota": "101x24_each_condition",
                    "primary_pending": 0, "behavior_pending": 0, "reference_join_missing": 0,
                    "recomputed_frozen_IP_endpoint_mismatches": 0, "paired_accuracy_comparisons": len(statistics),
                    "reused_paired_accuracy_comparisons": len(reused_statistics),
                    "bootstrap_replicates": BOOT, "bootstrap_seed": SEED,
                    "selection_adjusted_CI": False, "frozen_raw_scanned": False,
                    "new_API_calls": 0, "new_generations": 0, "GPU_initialized": False,
                    "script_sha256": file_hash(Path(__file__)), "sources": sources,
                    "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / "verification.json", verification)
    headers = ["hf_model_id", *ENDPOINTS]
    table = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n"
    table += "".join("| " + " | ".join(str(row[k]) for k in headers) + " |\n" for row in compact)
    with (output / "main_endpoint_compact.md").open("x", encoding="utf-8") as stream:
        stream.write("# Five-core native VCD main comparison\n\nNative -> independently best observed IP-VCD endpoint; all eval cells n=2424.\n\n")
        stream.write(table)
        stream.write("\nRetention uses the same frozen guided Direct ∩ uniformR fixed set at the selected IP marker. "
                     "Zero denominators are undefined. Endpoint choices are different observed eval configurations, not a single tuned condition. "
                     "Paired confidence intervals do not correct for selecting the best observed eval accuracy. "
                     "The frozen 352 conditions and paper body are unchanged.\n")
    print(table, flush=True)
    print(json.dumps({key: verification[key] for key in ("passed", "native_rows", "endpoint_rows", "marker_retention_rows", "fixed_set_memberships", "primary_pending", "behavior_pending")}), flush=True)


if __name__ == "__main__":
    main()
