"""Receive externally resolved complete-response labels; freeze cal and export DU.

No lexical or yes/no scoring, abstention heuristic, semantic model, or u-label
construction is implemented here. Missing labels prevent point export/selection.
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path

from wording_protocol import *
from generate_wording import build_plan, complete_chunks


def raw_index(registration, series, phase, selection=None, models=None):
    result = {}
    for model in model_scope(models):
        plan = build_plan(registration, phase, model, selection)
        complete = complete_chunks(Path(series) / "runs" / phase / model, plan)
        for key, value in complete.items():
            require(key not in result, "Duplicate raw task in series")
            result[key] = value
    return result


def expected_keys(registration, phase, selection=None, models=None):
    return {task_key(registration, phase, c["condition_identity"], sid)
            for c in phase_conditions(registration, phase, selection, models)
            for sid in phase_sample_ids(registration, phase)}


def receive_scores(registration, series, phase, score_file, selection=None, models=None):
    raw = raw_index(registration, series, phase, selection, models)
    expected = expected_keys(registration, phase, selection, models)
    require(set(raw) == expected, f"Raw phase incomplete: {len(raw)}/{len(expected)}")
    scores, seen = read_csv(score_file), set()
    for score in scores:
        key = score["key"]
        require(key in raw and key not in seen, "Foreign/duplicate scored response")
        seen.add(key)
        row = raw[key]["row"]
        require(not strict_bool(row.get("synthetic_test_only", False), "synthetic raw"), "Synthetic raw cannot enter formal results")
        require(score["phase"] == phase and score["model"] == row["model"]
                and score["sample_id"] == row["sample"]["id"]
                and score["condition_identity"] == row["condition_identity"], "Score task binding differs")
        require(score["registration_identity"] == registration["identity"]
                and score["reference_original_sha256"] == REFERENCE_SHA
                and score["selection_identity"] == (row["selection_identity"] or ""), "Score registration/reference/pretest-selection differs")
        require(score["raw_record_sha256"] == stable_hash(row)
                and score["response_sha256"] == response_hash(row), "Score does not bind the complete actual response")
        require(strict_bool(score["quality_resolved"], "quality_resolved")
                and strict_bool(score["abstain_resolved"], "abstain_resolved"), "Unresolved labels cannot enter DU")
        require(score["quality"] in ("0", "1", "2") and bool(score.get("reviewer", "").strip()), "Resolved registered quality and real reviewer required")
        strict_bool(score["abstain"], "abstain")
        require(score.get("synthetic_test_only", "False") in ("False", "false", "0", ""), "Synthetic labels cannot enter formal results")
    require(seen == expected, f"Labels incomplete: {len(seen)}/{len(expected)}")
    return scores, raw


def aggregate(registration, phase, scores, selection=None, models=None):
    grouped = collections.defaultdict(list)
    for score in scores:
        grouped[score["condition_identity"]].append(score)
    conditions = phase_conditions(registration, phase, selection, models)
    require(set(grouped) == {c["condition_identity"] for c in conditions}, "Condition label coverage differs")
    return [metric_from_scores(registration, c, phase, grouped[c["condition_identity"]]) for c in conditions]


def state(registration, score):
    abstain = strict_bool(score["abstain"], "abstain")
    u = registration["u"][(score["model"], score["sample_id"])]
    return "TP" if abstain and u else "FP" if abstain else "C" if int(score["quality"]) == 1 else "W"


def paired_effects(registration, phase, scores, metrics, selection=None, models=None):
    import numpy as np
    by_score = {(s["condition_identity"], s["sample_id"]): s for s in scores}
    by_metric = {s["condition_identity"]: s for s in metrics}
    active = phase_conditions(registration, phase, selection, models)
    ids = phase_sample_ids(registration, phase)
    components = list(dict.fromkeys(registration["split_by_id"][sid]["component_id"] for sid in ids))
    expected_groups = 56 if phase == "calibration" else 102
    require(len(components) == expected_groups, "Bootstrap component roster differs")
    component_index = {component: i for i, component in enumerate(components)}
    memberships = np.array([component_index[registration["split_by_id"][sid]["component_id"]] for sid in ids])
    sizes = np.bincount(memberships, minlength=len(components))
    strata = {}
    for sid in ids:
        meta = registration["split_by_id"][sid]
        component = component_index[meta["component_id"]]
        strata.setdefault((meta["category"], meta["subcategory"]), set()).add(component)
    require(sum(len(v) for v in strata.values()) == len(components), "A bootstrap component crosses domains")
    repetitions = registration["protocol"]["bootstrap"]["repetitions"]
    contrasts, transitions = [], []
    for ip in active:
        if ip["method"] not in IP_METHODS:
            continue
        baselines = [("matching_guided_direct", c) for c in active if c["model"] == ip["model"]
                     and c["role"] == "guided_direct" and c["marker"] == ip["marker"]]
        baselines += [("native_" + c["method"], c) for c in active
                      if c["model"] == ip["model"] and c["role"] == "native_control"]
        require(len(baselines) == 3, "Paired IP comparison lacks guided Direct/native Direct/VCD")
        for baseline_role, baseline in baselines:
            states_ip = [state(registration, by_score[(ip["condition_identity"], sid)]) for sid in ids]
            states_base = [state(registration, by_score[(baseline["condition_identity"], sid)]) for sid in ids]
            delta = np.array([int(a in ("C", "TP")) - int(b in ("C", "TP")) for a, b in zip(states_ip, states_base)])
            component_delta = np.bincount(memberships, weights=delta, minlength=len(components))
            seed = stable_seed("wording_DU_20261009", phase, ip["condition_identity"], baseline["condition_identity"])
            rng = np.random.default_rng(seed)
            weights = np.zeros((repetitions, len(components)), dtype=np.int64)
            for stratum in sorted(strata):
                columns = sorted(strata[stratum])
                weights[:, columns] = rng.multinomial(len(columns), np.full(len(columns), 1 / len(columns)), size=repetitions)
            sampled_n = weights @ sizes
            require(np.all(sampled_n > 0), "Zero bootstrap denominator")
            boot = (weights @ component_delta) / sampled_n
            lower, upper = np.quantile(boot, [0.025, 0.975])
            m_ip, m_base = by_metric[ip["condition_identity"]], by_metric[baseline["condition_identity"]]
            numerator = m_ip["DU_numerator"] - m_base["DU_numerator"]
            require(int(delta.sum()) == numerator, "Paired DU/metric identity differs")
            contrasts.append({"phase": phase, "model": ip["model"], "ip_method": ip["method"], "marker": ip["marker"],
                              "ip_condition_identity": ip["condition_identity"], "baseline_role": baseline_role,
                              "baseline_condition_identity": baseline["condition_identity"], "n_pairs": len(ids),
                              "delta_C_count": m_ip["C"] - m_base["C"], "delta_TP_count": m_ip["TP"] - m_base["TP"],
                              "delta_FP_count_diagnostic_only": m_ip["FP"] - m_base["FP"],
                              "delta_DU_numerator": numerator, "delta_DU": numerator / len(ids),
                              "delta_answer_coverage": m_ip["answer_coverage"] - m_base["answer_coverage"],
                              "delta_DU_ci95_lower": float(lower), "delta_DU_ci95_upper": float(upper),
                              "n_clusters": len(components), "bootstrap_repetitions": repetitions, "seed": seed,
                              "interval": "pointwise_95_percentile_domain_stratified_paired_component_bootstrap",
                              "uncertainty": "conditional_on_supplied_quality_and_frozen_u; fixed9_models"})
            counts = collections.Counter(zip(states_base, states_ip))
            require(sum(counts.values()) == len(ids), "Paired state counts differ")
            for (before, after), n in sorted(counts.items()):
                transitions.append({"phase": phase, "model": ip["model"], "ip_method": ip["method"], "marker": ip["marker"],
                                    "ip_condition_identity": ip["condition_identity"], "baseline_role": baseline_role,
                                    "baseline_condition_identity": baseline["condition_identity"],
                                    "baseline_state": before, "IP_state": after, "n": n, "n_pairs": len(ids)})
    return contrasts, transitions


def make_queue(registration, series, phase, destination, selection=None, allow_partial=False, models=None):
    raw = raw_index(registration, series, phase, selection, models)
    expected = expected_keys(registration, phase, selection, models)
    require(allow_partial or set(raw) == expected, "Complete raw phase required unless --allow-partial queue is explicit")
    references = {(r["model"], r["sample_id"]): r for r in read_csv(registration["asset_path"]("quality_reference"))}
    queue = []
    for key, binding in raw.items():
        row = binding["row"]
        ref = references[(row["model"], row["sample"]["id"])]
        queue.append({"key": key, "phase": phase, "model": row["model"], "method": row["method"], "marker": row["marker"],
                      "condition_identity": row["condition_identity"], "sample_id": row["sample"]["id"],
                      "registration_identity": registration["identity"], "selection_identity": row["selection_identity"],
                      "reference_original_sha256": REFERENCE_SHA,
                      "raw_record_sha256": stable_hash(row), "response_sha256": response_hash(row),
                      "raw_path": binding["raw_path"], "raw_sha256": binding["raw_sha256"], "raw_line": binding["raw_line"],
                      "question": row["sample"]["question"], "complete_actual_response": row["text"],
                      "reference_gt_answer": ref["gt_answer"], "reference_gt_details": ref["gt_answer_details"],
                      "reference_sufficient": ref["reference_sufficient"],
                      "tokens_n": len(row["tokens"]), "finish_reason": row["finish_reason"], "truncated": row["truncated"],
                      "quality": "", "abstain": "", "quality_resolved": "", "abstain_resolved": "", "reviewer": "",
                      "review_model": "", "review_effort": "", "review_source_sha256": ""})
    require(queue, "No sealed raw rows to queue")
    write_csv(destination, queue)
    return {"status": "queue_written_labels_blank", "sealed_raw_rows": len(queue), "expected_rows": len(expected), "new_semantic_decisions": 0}


def export_metrics(registration, series, phase, score_file, out, selection=None, models=None):
    out = Path(out).resolve()
    require(not out.exists(), "New exclusive result directory required")
    scope = model_scope(models)
    scores, raw = receive_scores(registration, series, phase, score_file, selection, scope)
    metrics = aggregate(registration, phase, scores, selection, scope)
    effects, transitions = paired_effects(registration, phase, scores, metrics, selection, scope)
    out.mkdir(parents=True)
    write_csv(out / "condition_metrics_DU.csv", metrics)
    write_csv(out / "paired_DU_effects_bootstrap.csv", effects)
    write_csv(out / "paired_state_transitions.csv", transitions)
    bindings = [{"key": key, "raw_path": value["raw_path"], "raw_sha256": value["raw_sha256"], "raw_line": value["raw_line"],
                 "raw_record_sha256": stable_hash(value["row"]), "response_sha256": value["row"]["response_sha256"]}
                for key, value in raw.items()]
    write_csv(out / "scored_raw_bindings.csv", bindings)
    receipt = {"status": "complete_supplied_labels", "phase": phase, "registration_identity": registration["identity"],
               "models": list(scope),
               "selection_identity": selection["selection_identity"] if selection else None,
               "score_file": str(Path(score_file).resolve()), "score_file_sha256": file_hash(score_file),
               "n_scored_rows": len(scores), "n_conditions": len(metrics),
               "n_per_condition": 317 if phase == "calibration" else 634,
               "reference_original_sha256": REFERENCE_SHA,
               "bootstrap_groups": 56 if phase == "calibration" else 102, "bootstrap_repetitions": 2000,
               "new_semantic_decisions": 0, "GPU_initialized": False,
               "scope": registration["protocol"]["holdout_scope"],
               "output_sha256": {path.name: file_hash(path) for path in out.iterdir() if path.is_file()}}
    write_once(out / "metric_receipt.json", receipt)
    return metrics, receipt


def freeze_selection(registration, metrics_dir, score_file, destination, models=None):
    destination = Path(destination).resolve()
    require(not destination.exists(), "Selection directory must be new; no test-driven replacement")
    metrics_dir = Path(metrics_dir).resolve()
    receipt = read_json(metrics_dir / "metric_receipt.json")
    scope = model_scope(models)
    require(receipt["status"] == "complete_supplied_labels" and receipt["phase"] == "calibration"
            and receipt["registration_identity"] == registration["identity"]
            and model_scope(receipt.get("models")) == scope and receipt["n_conditions"] == 14 * len(scope)
            and receipt["n_scored_rows"] == 14 * 317 * len(scope), "Full scoped calibration-only receipt required")
    require(file_hash(score_file) == receipt["score_file_sha256"], "Calibration label source changed")
    metrics_file = metrics_dir / "condition_metrics_DU.csv"
    require(file_hash(metrics_file) == receipt["output_sha256"][metrics_file.name], "Calibration metrics changed")
    metrics = read_csv(metrics_file)
    winners, candidates, guided = choose_winners(registration, metrics, scope)
    destination.mkdir(parents=True)
    shutil.copyfile(metrics_file, destination / "calibration_metrics.csv")
    shutil.copyfile(score_file, destination / "calibration_scored_rows.csv")
    shutil.copyfile(metrics_dir / "metric_receipt.json", destination / "calibration_metric_receipt.json")
    selection = {"schema": "kdm_hallusion_wording_selection_v1", "status": "frozen", "frozen_utc": now(),
                 "models": list(scope),
                 "registration_identity": registration["identity"], "selection_split": "calibration", "n": 317,
                 "marker_order": list(MARKERS), "rule": "maximize integer C+TP; original marker order breaks ties",
                 "calibration_metrics_file": "calibration_metrics.csv", "calibration_metrics_sha256": file_hash(metrics_file),
                 "calibration_labels_file": "calibration_scored_rows.csv", "calibration_labels_sha256": file_hash(score_file),
                 "calibration_receipt_file": "calibration_metric_receipt.json",
                 "calibration_receipt_sha256": file_hash(destination / "calibration_metric_receipt.json"),
                 "winners": winners, "candidates": candidates, "guided_direct_condition_ids": guided,
                 "holdout_outputs_read": False, "synthetic_test_only": False,
                 "reference_original_sha256": REFERENCE_SHA}
    selection["selection_identity"] = stable_hash(selection)
    write_once(destination / "frozen_selection.json", selection)
    load_selection(registration, destination / "frozen_selection.json")
    write_csv(destination / "selected_IP_conditions.csv", winners)
    write_csv(destination / "all_four_marker_calibration_counts.csv", candidates)
    return {"status": "frozen", "winners": len(winners), "matching_guided_direct_conditions": len(guided),
            "selection_identity": selection["selection_identity"], "holdout_outputs_read": False}


def merge_selections(registration, sources, destination):
    require(len(sources) == 9, "Final selection merge requires exactly nine model files")
    items = [load_selection(registration, source) for source in sources]
    by_model = {}
    for source, item in zip(sources, items):
        scope = model_scope(item.get("models"))
        require(len(scope) == 1 and item["schema"] == "kdm_hallusion_wording_selection_v1"
                and scope[0] not in by_model, "Each registered model requires one complete unique cal seal")
        by_model[scope[0]] = (Path(source).resolve(), item)
    require(set(by_model) == set(MODELS), "Nine-model roster incomplete")
    destination = Path(destination).resolve()
    require(not destination.exists(), "New exclusive final selection directory required")
    destination.mkdir(parents=True)
    winners, candidates, guided, bindings = [], [], set(), []
    for model in MODELS:
        source, item = by_model[model]
        target = destination / "model_sources" / model
        target.mkdir(parents=True)
        for field in ("calibration_metrics_file", "calibration_labels_file", "calibration_receipt_file"):
            filename = item[field]
            shutil.copyfile(path_within(source.parent, filename), target / filename)
        shutil.copyfile(source, target / "frozen_selection.json")
        bindings.append({"model": model, "path": str((target / "frozen_selection.json").relative_to(destination)),
                         "sha256": file_hash(target / "frozen_selection.json"), "selection_identity": item["selection_identity"]})
        winners.extend(item["winners"]); candidates.extend(item["candidates"])
        guided.update(item["guided_direct_condition_ids"])
    selection = {"schema": "kdm_hallusion_wording_selection_merge_v1", "status": "frozen", "frozen_utc": now(),
                 "models": list(MODELS), "registration_identity": registration["identity"],
                 "selection_split": "calibration", "n": 317, "marker_order": list(MARKERS),
                 "rule": "each model independently maximizes integer C+TP with fixed marker-order tie; complete9 union",
                 "model_selections": bindings, "winners": winners, "candidates": candidates,
                 "guided_direct_condition_ids": sorted(guided), "holdout_outputs_read": False,
                 "synthetic_test_only": False, "reference_original_sha256": REFERENCE_SHA}
    selection["selection_identity"] = stable_hash(selection)
    write_once(destination / "frozen_selection.json", selection)
    load_selection(registration, destination / "frozen_selection.json")
    write_csv(destination / "selected_IP_conditions.csv", winners)
    write_csv(destination / "all_four_marker_calibration_counts.csv", candidates)
    return {"status": "frozen_complete9_union", "models": 9, "winners": 18, "candidates": 72,
            "selection_identity": selection["selection_identity"], "holdout_outputs_read": False}


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("queue", "export", "select"):
        p = sub.add_parser(name)
        p.add_argument("--protocol", type=Path, default=package / "registration/protocol.json")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--model", choices=MODELS, help="Only this model; all14 cal conditions/N317 are still required")
        if name != "select":
            p.add_argument("--series-root", type=Path, required=True)
            p.add_argument("--phase", choices=("calibration", "holdout"), required=True)
            p.add_argument("--selection", type=Path)
        if name in ("export", "select"):
            p.add_argument("--scores", type=Path, required=True)
        if name == "queue":
            p.add_argument("--allow-partial", action="store_true")
        if name == "select":
            p.add_argument("--calibration-metrics-dir", type=Path, required=True)
    p = sub.add_parser("merge-selections")
    p.add_argument("--protocol", type=Path, default=package / "registration/protocol.json")
    p.add_argument("--selection-files", type=Path, nargs=9, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registration = load_registration(args.protocol)
    selection = load_selection(registration, args.selection) if getattr(args, "selection", None) else None
    scope = (args.model,) if getattr(args, "model", None) else None
    if args.command == "queue":
        result = make_queue(registration, args.series_root, args.phase, args.output, selection, args.allow_partial, scope)
    elif args.command == "export":
        _, result = export_metrics(registration, args.series_root, args.phase, args.scores, args.output, selection, scope)
    elif args.command == "select":
        result = freeze_selection(registration, args.calibration_metrics_dir, args.scores, args.output, scope)
    else:
        result = merge_selections(registration, args.selection_files, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
