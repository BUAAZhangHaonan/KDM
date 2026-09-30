#!/usr/bin/env python3
"""CPU comparison identities and complete native-versus-Direct paired statistics."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, within
from kdm.prompts import task_prompt
from workflows.main_results.analysis import BOOT, SEED, bootstrap_class_ci, bootstrap_ratio_diff_by_cluster
from workflows.paper_core.receive_native import MODELS
from workflows.paper_core.score_native import COND, DEFAULT_DIRECT, load_references, rows, save_json, save_rows


def novel_boundaries(score, prior, output):
    known, existing, sources = set(), {}, []
    for path in prior:
        keys = set()
        for _, row, _ in rows(path):
            keys.add(row["qa_key"])
            existing[row["qa_key"]] = row
        known.update(keys)
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": file_hash(path), "unique_QA": len(keys)})
    original = [row for _, row, _ in rows(score / "boundary_queue.jsonl")]
    novel = [row for row in original if row["qa_key"] not in known]
    overlap = [{**row, "earlier_queue_memberships": existing[row["qa_key"]]["memberships"]}
               for row in original if row["qa_key"] in known]
    save_rows(output / "boundary_novel.jsonl", novel)
    save_rows(output / "boundary_already_queued.jsonl", overlap)
    receipt = {"source_boundary_queue": str((score / "boundary_queue.jsonl").relative_to(ROOT)),
               "source_boundary_sha256": file_hash(score / "boundary_queue.jsonl"),
               "original_unique_QA": len(original), "novel_unique_QA": len(novel),
               "already_queued_unique_QA": len(overlap), "earlier_queues": sources,
               "already_queued_implies_adjudicated": False}
    save_json(output / "boundary_delta_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--direct-scores", default=DEFAULT_DIRECT)
    parser.add_argument("--prior-boundary", action="append", default=[])
    args = parser.parse_args()
    score, output = within(ROOT, args.score_dir), within(ROOT, args.output)
    output.mkdir(parents=True, exist_ok=False)
    if BOOT != 2000 or SEED != 20260929:
        raise ValueError("Frozen bootstrap registration differs")
    receipt = json.loads((score / "scoring_receipt.json").read_text())
    native = pd.DataFrame([row for _, row, _ in rows(score / "score_rows.jsonl.gz")])
    direct_folder = within(ROOT, args.direct_scores)
    direct = pd.DataFrame([row for _, row, _ in rows(direct_folder / "score_rows.jsonl.gz")])
    references, gt = load_references()
    if len(native) != receipt["native_rows"] or len(direct) != 12120:
        raise ValueError("Paired source counts differ from completed scoring receipts")
    frozen_conditions = pd.read_csv(ROOT / "outputs/paper_20260929/conditions.csv")
    mapping = []
    for condition in frozen_conditions.to_dict("records"):
        if condition["method"] == "direct":
            continue
        baseline = frozen_conditions[(frozen_conditions.model == condition["model"])
                                     & frozen_conditions.method.eq("direct")
                                     & frozen_conditions.marker.eq(condition["marker"])]
        if len(baseline) != 1:
            raise ValueError("Frozen method lacks its exact same-main-prompt Direct condition")
        mapping.append({"model": condition["model"], "comparison": "frozen_same_main_prompt_Direct",
                        "method": condition["method"], "kind": condition["kind"], "main_marker": condition["marker"],
                        "reference_marker": condition["reference_marker"], "guided": condition["guided"],
                        "reference_guided": condition["reference_guided"], "replicate": condition["replicate"],
                        "method_condition_id": condition["condition_id"], "direct_condition_id": int(baseline.iloc[0].condition_id),
                        "baseline_condition_id": int(baseline.iloc[0].condition_id), "baseline_method": "direct",
                        "expected_n": 2424, "identity_scope": "frozen_paper_20260929", "new_pair_calculation": False})
        if condition["method"] in {"instruction_vcd", "instruction_m3id"}:
            base_method = "vcd" if condition["method"] == "instruction_vcd" else "m3id"
            for baseline_kind, comparison in (("main", "frozen_IP_vs_same_marker_guided"),
                                               ("reference_instruction_removed", "frozen_IP_vs_reference_guidance_removed")):
                candidates = frozen_conditions[(frozen_conditions.model == condition["model"])
                                                & frozen_conditions.method.eq(base_method)
                                                & frozen_conditions.kind.eq(baseline_kind)
                                                & frozen_conditions.marker.eq(condition["marker"])
                                                & frozen_conditions.reference_marker.eq(condition["marker"])]
                if len(candidates) != 1:
                    raise ValueError("Frozen IP comparison lacks its exact registered baseline")
                base = candidates.iloc[0]
                mapping.append({"model": condition["model"], "comparison": comparison,
                                "method": condition["method"], "kind": condition["kind"],
                                "main_marker": condition["marker"], "reference_marker": condition["reference_marker"],
                                "guided": condition["guided"], "reference_guided": condition["reference_guided"],
                                "replicate": condition["replicate"], "method_condition_id": condition["condition_id"],
                                "baseline_condition_id": int(base.condition_id), "baseline_method": base_method,
                                "baseline_kind": baseline_kind, "baseline_guided": bool(base.guided),
                                "baseline_reference_guided": bool(base.reference_guided), "expected_n": 2424,
                                "identity_scope": "frozen_paper_20260929", "new_pair_calculation": False})
    for model in MODELS:
        mapping.append({"model": model, "comparison": "native_VCD_vs_unguided_Direct", "method": "vcd",
                        "kind": "native_unguided", "main_marker": "NONE", "reference_marker": "NONE",
                        "guided": False, "reference_guided": False, "replicate": 0,
                        "method_condition_id": None, "direct_condition_id": None, "expected_n": 2424,
                        "baseline_condition_id": None, "baseline_method": "direct", "baseline_guided": False,
                        "identity_scope": "P1_NATIVE_VCD_vs_exact_census_Direct", "new_pair_calculation": True})
    pd.DataFrame(mapping).to_csv(output / "comparison_identity_map.csv", index=False)
    rng = np.random.RandomState(SEED)
    draws = rng.randint(0, 101, size=(BOOT, 101))
    summaries, transitions, pending = [], [], []
    for model in MODELS:
        a, b = native[native.model.eq(model)].copy(), direct[direct.model.eq(model)].copy()
        canonical_pending = int(a.canonical_name_in_primary_score.isna().sum())
        abstain_pending = int(a.abstain.isna().sum())
        if (len(a) != 2424 or canonical_pending or abstain_pending):
            pending.append({"model": model, "method": "vcd", "kind": "native_unguided", "expected_n": 2424,
                            "received": len(a), "missing": 2424 - len(a), "canonical_pending": canonical_pending,
                            "abstain_pending": abstain_pending, "new_effect_calculated": False})
            continue
        if len(b) != 2424 or b.correct_canonical.isna().any() or b.abstain.isna().any():
            raise ValueError("Native paired Direct is incomplete")
        a, b = a.set_index("sample_id").sort_index(), b.set_index("sample_id").sort_index()
        if not a.index.equals(b.index) or not a.target_class.equals(b.target_class) or a.index.duplicated().any():
            raise ValueError("Native paired model/sample/category identities differ")
        if not a.seed.equals(b.seed) or not a.uniform_reference.equals(b.uniform_reference):
            raise ValueError("Native paired seed or reference differs")
        if any(row.prompt != task_prompt(row.question, guided=False) for row in a.itertuples()):
            raise ValueError("Native main prompt differs from the source-verified unguided Direct prompt")
        quota = a.groupby("target_class").size()
        if len(quota) != 101 or not quota.eq(24).all():
            raise ValueError("Native paired class clusters differ from 101 by 24")
        classes = sorted(quota.index)
        correct_a, correct_b = a.canonical_name_in_primary_score.astype(int), b.correct_canonical.astype(int)
        abstain_a, abstain_b, reference = a.abstain.astype(bool), b.abstain.astype(bool), a.uniform_reference.astype(bool)
        delta = correct_a - correct_b
        acc_by_class = {name: delta[a.target_class.eq(name)].tolist() for name in classes}
        recall_data, preservation = {}, {}
        fixed = abstain_b & reference
        for name in classes:
            mask = a.target_class.eq(name)
            recall_data[name] = (int(((abstain_a.astype(int) - abstain_b.astype(int)) * reference)[mask].sum()),
                                 int(reference[mask].sum()))
            preservation[name] = (int((abstain_a & fixed & mask).sum()), int((fixed & mask).sum()))
        before = np.where(abstain_b, "A", np.where(correct_b.eq(1), "C", "E"))
        after = np.where(abstain_a, "A", np.where(correct_a.eq(1), "C", "E"))
        for stratum in (False, True):
            for source in ("C", "E", "A"):
                for target in ("C", "E", "A"):
                    count = int(((reference.to_numpy() == stratum) & (before == source) & (after == target)).sum())
                    transitions.append({"model": model, "comparison": "native_VCD_vs_unguided_Direct",
                                        "uniform_reference": stratum, "reference_stratum_n": int(reference.eq(stratum).sum()),
                                        "from_state": source, "to_state": target, "n": count, "all_input_n": 2424})
        summaries.append({"model": model, "comparison": "native_VCD_vs_unguided_Direct", "paired_n": 2424,
                          "class_clusters": 101, "main_prompt_equal": True, "seeds_equal": True,
                          "accuracy_native_minus_Direct": bootstrap_class_ci(acc_by_class, draws),
                          "uniform_abstention_recall_native_minus_Direct": bootstrap_ratio_diff_by_cluster(recall_data, draws, classes),
                          "fixed_Direct_reasonable_abstention_retained": bootstrap_ratio_diff_by_cluster(preservation, draws, classes),
                          "correct_retained": int((correct_a.eq(1) & correct_b.eq(1)).sum()),
                          "correct_lost": int((correct_a.eq(0) & correct_b.eq(1)).sum()),
                          "new_correct": int((correct_a.eq(1) & correct_b.eq(0)).sum()),
                          "bootstrap_replicates": BOOT, "bootstrap_seed": SEED, "bootstrap_unit": "Food-101 target class"})
    save_json(output / "native_paired_statistics.json", summaries)
    save_rows(output / "native_pairs_pending.jsonl", pending)
    pd.DataFrame(transitions, columns=["model", "comparison", "uniform_reference", "reference_stratum_n", "from_state", "to_state", "n", "all_input_n"]).to_csv(output / "native_transitions_reference_stratified.csv", index=False)
    for model in {row["model"] for row in transitions}:
        if sum(row["n"] for row in transitions if row["model"] == model) != 2424:
            raise ValueError("Native complete C/E/A transitions do not cover all paired samples")
    delta_receipt = novel_boundaries(score, [within(ROOT, path) for path in args.prior_boundary], output)
    result = {"schema": "kdm_core_native_pair_analysis_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "native_score_rows": len(native), "direct_rows": len(direct), "reference_eval_rows": len(references),
              "completed_native_pairs": len(summaries), "pending_native_pairs": len(pending),
              "comparison_identity_rows": len(mapping), "transition_rows": len(transitions),
              "bootstrap_replicates": BOOT, "bootstrap_seed": SEED, "bootstrap_unit": "Food-101 target class",
              "score_sha256": file_hash(score / "score_rows.jsonl.gz"), "direct_score_sha256": file_hash(direct_folder / "score_rows.jsonl.gz"),
              "new_API_calls": 0, "new_generations": 0, "GPU_initialized": False,
              "boundary_delta": delta_receipt, "analysis_source_sha256": file_hash(Path(__file__))}
    save_json(output / "analysis_receipt.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
