"""Read-only CPU evidence for the frozen five-model core experiment supplement."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from kdm.decoding import DecodeConfig
from kdm.io import file_hash, stable_hash, stable_seed
from kdm.pipeline import task_id
from kdm.prompts import task_prompt
from workflows.main_results import score as frozen_score

MODELS = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b")
FIELDS = ("model", "method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")
DEFAULT_LABELS = ROOT / "outputs/supplemental/remaining11/run_20260930_140337/recovered/outputs/annotations/luna_census_remaining_v1/remaining108_plus2_20260924/census_merged_v1/labels.jsonl"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_rows(path, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    options = {"compresslevel": 1} if path.suffix == ".gz" else {}
    with opener(path, "xt", encoding="utf-8", newline="\n", **options) as stream:
        for row in rows:
            stream.write(json.dumps(clean(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def fraction(a, b):
    return a / b if b else None


def family(row):
    method, kind = row["method"], row["kind"]
    if method == "vcd":
        return "VCD_reference_guidance_removed" if kind == "reference_instruction_removed" else "VCD_with_abstention_guidance"
    if method == "m3id":
        return "M3ID_reference_guidance_removed" if kind == "reference_instruction_removed" else "M3ID_with_abstention_guidance"
    return {"direct": "Direct_with_abstention_guidance", "instruction_vcd": "IP_VCD",
            "instruction_m3id": "IP_M3ID", "cda_visual": "CDA_visual_no_momentum",
            "dola": "DoLa", "deco": "DeCo", "sid": "registered_SID"}[method]


def frozen_audit(package, output, export):
    output.mkdir(parents=True, exist_ok=False)
    conditions = pd.read_csv(export / "conditions.csv")
    bundled = pd.read_csv(package / "data/conditions.csv")
    if not conditions[list(FIELDS)].equals(bundled[list(FIELDS)]) or not np.array_equal(conditions.condition_id, bundled.condition_id):
        raise ValueError("Bundle and server condition identities differ")
    with np.load(package / "data/frozen_decisions.npz", allow_pickle=False) as data:
        arrays = {field: data[field] for field in data.files}
    scores = pd.read_parquet(export / "scores.parquet")
    references = pd.read_parquet(export / "references.parquet")
    required = ["condition_id", "sample_id", "correct_canonical", "correct_literal", "abstain", "uniform_reference", "qa_id"]
    if len(scores) != 853248 or scores.duplicated(["condition_id", "sample_id"]).any() or scores[required].isna().any().any():
        raise ValueError("Frozen score coverage/nulls/uniqueness differs")
    if len(references) != 24240 or references.duplicated(["model", "sample_id"]).any():
        raise ValueError("Frozen reference coverage differs")
    condition_index = {int(cid): i for i, cid in enumerate(arrays["condition_ids"])}
    sample_index = {sid: i for i, sid in enumerate(arrays["sample_ids"])}
    ci = scores.condition_id.map(condition_index).to_numpy()
    si = scores.sample_id.map(sample_index).to_numpy()
    if pd.isna(ci).any() or pd.isna(si).any():
        raise ValueError("Frozen bundle/server score key intersection differs")
    checks = {}
    for field in ("correct_canonical", "correct_literal", "abstain", "uniform_reference", "qa_id"):
        checks[field + "_mismatches"] = int(np.count_nonzero(scores[field].to_numpy() != arrays[field][ci, si]))
    quota = scores.groupby(["condition_id", "target_class"], observed=True).size()
    if len(quota) != 352 * 101 or not (quota == 24).all() or any(checks.values()):
        raise ValueError("Frozen panel or 101-by-24 class quota differs")
    expected_gt = (references.gold_rank > 1) & (references.independent_correct_attempts == 0)
    if not (references.independent_attempts == 10).all() or not (expected_gt == references.uniform_reference).all():
        raise ValueError("Uniform reference rule differs")
    checks.update(actual_score_rows=len(scores), conditions=len(conditions), samples=len(sample_index),
                  reference_rows=len(references), class_quota_checks=len(quota),
                  reference_rule_mismatches=0, reference_ten_attempt_rows=len(references))
    reference_table = references.groupby(["model", "split"], observed=True).agg(
        n=("sample_id", "size"), uniform_positive=("uniform_reference", "sum"), historical_accepted_positive=("accepted_reference", "sum")).reset_index()
    reference_table.to_csv(output / "reference_counts.csv", index=False)
    recomputed = output.parent / "outputs/cpu_recomputed"
    metrics = pd.read_csv(recomputed / "all_conditions_352.csv")
    recompute_checks = json.loads((recomputed / "checks.json").read_text())
    if any(recompute_checks.values()) or len(metrics) != 352:
        raise ValueError("Package recomputation is incomplete")
    metrics = metrics.merge(conditions[["condition_id", "guided", "reference_guided", "replicate", "split", "prompt_id", "config_id"]], on="condition_id", validate="one_to_one")
    metrics["method_family"] = [family(row) for row in metrics.to_dict("records")]
    metrics["configuration_scope"] = np.where(metrics.marker == metrics.reference_marker, "same_marker", "cross_marker")
    metrics.to_csv(output / "all_conditions_352.csv", index=False)
    paired = metrics[metrics.method.isin(["vcd", "m3id", "sid"]) & metrics.kind.eq("main")]
    paired[paired.marker.eq(paired.reference_marker)].to_csv(output / "guided_same_marker_4_per_model.csv", index=False)
    paired.to_csv(output / "guided_cross_reference_16_per_model.csv", index=False)
    prompt_config = json.loads((export / "prompts_and_configs.json").read_text())
    prompts = {row["prompt_id"]: row["value"] for row in prompt_config["prompts"]}
    configs = {row["config_id"]: row["value"] for row in prompt_config["configs"]}
    inventory = []
    for row in conditions.to_dict("records"):
        inventory.append({**row, "method_family": family(row),
            "same_marker": row["marker"] == row["reference_marker"], "available": True,
            "main_prompt": prompts[row["prompt_id"]]["prompt"],
            "reference_prompt": prompts[row["prompt_id"]]["reference_prompt"],
            "neutral_prompt": prompts[row["prompt_id"]]["neutral_prompt"],
            "decode_config": json.dumps(configs[row["config_id"]], sort_keys=True),
            "source_namespace": "frozen_paper_20260929", "scope": "registered_condition"})
    for model in MODELS:
        inventory.append({"condition_id": None, "model": model, "method": "vcd", "kind": "native_unguided",
            "method_family": "VCD_without_abstention_guidance", "guided": False, "reference_guided": False,
            "marker": "UNKNOWN", "reference_marker": "UNKNOWN", "replicate": 0, "split": "eval",
            "n": 0, "expected_n": 2424, "available": False, "scope": "new_generation_required",
            "source_namespace": None, "decode_config": json.dumps(asdict(DecodeConfig(method="vcd")), sort_keys=True)})
    pd.DataFrame(inventory).to_csv(output / "method_inventory.csv", index=False)
    direct = {(row.model, row.marker): int(row.condition_id) for row in conditions.itertuples() if row.method == "direct"}
    tuples = {tuple(getattr(row, field) for field in FIELDS): int(row.condition_id) for row in conditions.itertuples()}
    transitions = []
    for row in conditions.itertuples():
        pairs = []
        if row.method != "direct":
            pairs.append(("same_main_prompt_Direct", direct[(row.model, row.marker)], int(row.condition_id)))
        if row.method in {"instruction_vcd", "instruction_m3id"}:
            base = "vcd" if row.method == "instruction_vcd" else "m3id"
            pairs.append(("IP_vs_same_marker_guided", tuples[(row.model, base, "main", row.marker, row.marker, True, True, 0)], int(row.condition_id)))
            pairs.append(("IP_vs_reference_guidance_removed", tuples[(row.model, base, "reference_instruction_removed", row.marker, row.marker, True, False, 0)], int(row.condition_id)))
        for name, before_id, after_id in pairs:
            i, j = condition_index[before_id], condition_index[after_id]
            before = np.where(arrays["abstain"][i], "A", np.where(arrays["correct_canonical"][i], "C", "E"))
            after = np.where(arrays["abstain"][j], "A", np.where(arrays["correct_canonical"][j], "C", "E"))
            reference = arrays["uniform_reference"][i]
            if not np.array_equal(reference, arrays["uniform_reference"][j]):
                raise ValueError("Paired reference set differs")
            for stratum in (False, True):
                for a in ("C", "E", "A"):
                    for b in ("C", "E", "A"):
                        mask = (before == a) & (after == b) & (reference == stratum)
                        transitions.append({"comparison": name, "model": row.model, "main_marker": row.marker,
                            "reference_marker": row.reference_marker, "method": row.method, "kind": row.kind,
                            "before_condition_id": before_id, "after_condition_id": after_id,
                            "uniform_reference": stratum, "reference_stratum_denominator": int((reference == stratum).sum()),
                            "from_state": a, "to_state": b, "n": int(mask.sum()), "all_input_denominator": 2424})
    pd.DataFrame(transitions).to_csv(output / "transitions_reference_stratified.csv", index=False)
    save_json(output / "panel_validation.json", {**checks, "recompute_checks": recompute_checks,
        "export_schema_path": str(export / "schema.md"), "export_source_dictionary": str(export / "sources.csv"),
        "input_sha256": {str(path): file_hash(path) for path in (package / "data/frozen_decisions.npz", export / "scores.parquet", export / "references.parquet", export / "conditions.csv")}})
    tests = []
    for name in ("test_four_view_math.py", "test_select_phrase.py"):
        command = [sys.executable, str(package / "tests" / name)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        tests.append({"command": command, "exit_code": completed.returncode, "stdout": completed.stdout,
                      "stderr": completed.stderr, "evidence_type": "synthetic_math_or_selection_test", "empirical_model_generation": False})
        if completed.returncode:
            raise ValueError("Package synthetic test failed: " + name)
    save_json(output / "mathematical_tests.json", tests)
    return conditions, scores, references, arrays, checks


def score_existing(answer, target, label, variants, patterns):
    abstain = label["label"] == "abstain"
    root_variants = [v for v in variants if v.get("_root_overlay") or v.get("_source") == "root_independent_gap_review"]
    followup = [v for v in variants if str(v.get("_source", "")).startswith("followup_")]
    binary_variants = root_variants or followup or variants
    binary = {v.get("abstain") for v in binary_variants if type(v.get("abstain")) is bool}
    if len(binary) > 1:
        return {"correct_canonical": None, "correct_literal": None, "abstain": None, "reason": "frozen_exact_QA_binary_conflict"}
    if len(binary) == 1:
        abstain = next(iter(binary))
    inferred = frozen_score.parse_target_blind_primary(label.get("answer_text") or answer, patterns)
    if abstain:
        return {"correct_canonical": 0, "correct_literal": 0, "abstain": True, "reason": "source_bound_final_abstention", "primary_extraction": inferred}
    if label["label"] == "invalid" and not variants:
        return {"correct_canonical": 0, "correct_literal": 0, "abstain": False, "reason": "source_bound_final_invalid", "primary_extraction": inferred}
    result = []
    for variant in variants:
        variant = {**variant, "abstain": abstain}
        override = variant.get("canonical_override")
        if override in {"explicit_outside_101", "multiple_primary"}:
            result.append((0, 0, "frozen_QA_primary_override"))
        elif isinstance(override, str) and override:
            full = variant.get("literal_full_name")
            literal = int(frozen_score.fmt(full) in {frozen_score.fmt(target), frozen_score.fmt(target.replace("_", " "))}) if isinstance(full, str) else 0
            override_literal = variant.get("literal_score_override")
            literal = override_literal if type(override_literal) is int else literal
            result.append((int(override == target), literal, "frozen_root_QA_primary_class"))
        else:
            result.append(frozen_score.score_variant(answer, target, variant, patterns))
    exact = frozen_score.fmt(answer) in {frozen_score.fmt(target), frozen_score.fmt(target.replace("_", " "))}
    if exact and not root_variants:
        canonical = literal = 1
        reason = "whole_response_exact_class"
    elif result:
        canonical_values, literal_values = {r[0] for r in result}, {r[1] for r in result}
        canonical = next(iter(canonical_values)) if len(canonical_values) == 1 else None
        literal = next(iter(literal_values)) if len(literal_values) == 1 else None
        reason = "frozen_exact_QA_name_reuse" if canonical is not None else "frozen_exact_QA_name_unresolved"
    elif inferred["status"] == "unique_class":
        canonical = int(inferred["canonical_candidates"][0] == target)
        literal = int(frozen_score.fmt(inferred["primary_name"]) in {frozen_score.fmt(target), frozen_score.fmt(target.replace("_", " "))})
        reason = "final_census_primary_span_current_canonical_rule"
    elif inferred["status"] == "multiple_classes":
        canonical = literal = 0
        reason = "multiple_primary_canonical_classes"
    elif label.get("answer_text") and not frozen_score.classes_for_name(label["answer_text"], patterns):
        canonical = literal = 0
        reason = "final_census_primary_span_outside_original_101"
    elif frozen_score.target_absent(answer, target):
        canonical = literal = 0
        reason = "target_absent_from_full_response"
    else:
        canonical = literal = None
        reason = "current_primary_scope_unresolved"
    return {"correct_canonical": canonical, "correct_literal": literal, "abstain": abstain,
            "reason": reason, "primary_extraction": inferred}


def native_direct_audit(output, references, labels_path):
    output.mkdir(parents=True, exist_ok=False)
    samples = {r["id"]: r for r in read_rows(ROOT / "data/current/food101.jsonl") if r["split"] == "eval"}
    if len(samples) != 2424 or Counter(r["class"] for r in samples.values()) != {r["class"]: 24 for r in samples.values()}:
        raise ValueError("Current Food eval sample quota differs")
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    patterns = frozen_score.compile_classes(manifest["canonical_classes"])
    reviews = {(r["question"], r["answer"]): r["variants"] for r in read_rows(ROOT / "outputs/annotations/main_results/reviewed_answers.jsonl")}
    sources, required_keys, mismatch, rows = [], set(), [], []
    for model in MODELS:
        path = ROOT / "data/responses/census" / (model + ".jsonl.gz")
        sidecar_path = path.with_name(model + ".identity.json")
        sidecar = json.loads(sidecar_path.read_text())
        runtime = json.loads((ROOT / "configs/runtime" / (model + ".json")).read_text())
        identity_valid = stable_hash(sidecar["definition"]) == sidecar["identity"]
        backend_match = sidecar["definition"]["backend"] == runtime
        found = set()
        for number, row in enumerate(read_rows(path), 1):
            sample = row["sample"]
            if sample["dataset"] != "food101" or sample["split"] != "eval" or row["guided"]:
                continue
            task = {"sample": sample, "method": "direct", "kind": "unguided", "marker": "UNKNOWN", "reference_marker": "UNKNOWN", "guided": False, "reference_guided": False, "replicate": 0}
            errors = []
            if sample != samples.get(sample["id"]): errors.append("current_sample_mismatch")
            if row["key"] != task_id(model, task): errors.append("original_task_key_mismatch")
            if row["config"] != asdict(DecodeConfig()): errors.append("decode_config_mismatch")
            if row["prompt"] != task_prompt(sample["question"], guided=False): errors.append("unguided_prompt_mismatch")
            if row["seed"] != stable_seed(sample["id"], model, 0): errors.append("seed_mismatch")
            if row["identity"] != sidecar["identity"] or not identity_valid: errors.append("source_identity_mismatch")
            if not backend_match: errors.append("registered_backend_mismatch")
            if row["model"] != model or row["status"] != "ok": errors.append("raw_model_or_status_mismatch")
            if sample["id"] in found: errors.append("duplicate_sample")
            found.add(sample["id"])
            if errors:
                mismatch.append({"model": model, "sample_id": sample["id"], "key": row["key"], "source_path": str(path), "source_line": number, "differences": errors})
                continue
            rows.append({"original_response": row, "source_path": str(path), "source_line": number})
            required_keys.add(row["key"])
        sources.append({"model": model, "source_path": str(path), "source_identity": sidecar["identity"],
                        "identity_valid": identity_valid, "registered_backend_exact_match": backend_match,
                        "expected_eval_samples": 2424, "actual_eval_samples": len(found), "missing_sample_ids": sorted(set(samples) - found)})
    labels = {}
    total_labels = 0
    for label in read_rows(labels_path):
        total_labels += 1
        if label["key"] in required_keys:
            if label["key"] in labels:
                raise ValueError("Duplicate final closed census label key")
            labels[label["key"]] = label
    if total_labels != 293344:
        raise ValueError("Final full census closure differs")
    reference = {(r.model, r.sample_id): bool(r.uniform_reference) for r in references.itertuples() if r.split == "eval"}
    scored, pending = [], []
    for item in rows:
        raw = item["original_response"]
        label = labels.get(raw["key"])
        if label is None or (label["raw_identity"], label["question"], label["text"], label["model"], label["sample_id"], label["guided"]) != (raw["identity"], raw["sample"]["question"], raw["text"], raw["model"], raw["sample"]["id"], False):
            raise ValueError("Source-bound final census annotation differs")
        variants = reviews.get((raw["sample"]["question"], raw["text"]), [])
        decision = score_existing(raw["text"], raw["sample"]["class"], label, variants, patterns)
        record = {"model": raw["model"], "sample_id": raw["sample"]["id"], "dataset": "food101", "split": "eval",
            "method": "direct", "kind": "unguided", "guided": False, "reference_guided": False,
            "original_key": raw["key"], "source_identity": raw["identity"], "question": raw["sample"]["question"],
            "answer": raw["text"], "target_class": raw["sample"]["class"], "uniform_reference": reference[(raw["model"], raw["sample"]["id"])],
            "original_final_label": label, "frozen_QA_variants": variants, "seed": raw["seed"], "decode_config": raw["config"],
            "terminated": raw["terminated"], "tokens": raw["tokens"], "source_path": item["source_path"], "source_line": item["source_line"],
            "score_authority": "frozen canonical functions with exact-QA reuse and source-bound closed census primary span", **decision}
        scored.append(record)
        if decision["correct_canonical"] is None or decision["abstain"] is None:
            pending.append(record)
    metrics = []
    for model in MODELS:
        group = [r for r in scored if r["model"] == model]
        complete = len(group) == 2424 and all(r["correct_canonical"] is not None and r["abstain"] is not None for r in group)
        behavior_complete = len(group) == 2424 and all(r["abstain"] is not None for r in group)
        correct = sum(r["correct_canonical"] == 1 for r in group)
        na = sum(r["abstain"] is True for r in group)
        nr = sum(r["uniform_reference"] for r in group)
        tp = sum(r["abstain"] is True and r["uniform_reference"] for r in group)
        fp, fn = na - tp, nr - tp
        metrics.append({"model": model, "method_family": "Direct_without_abstention_guidance", "expected_n": 2424,
            "n": len(group), "primary_complete": complete, "behavior_complete": behavior_complete,
            "canonical_correct_known": correct, "canonical_pending": sum(r["correct_canonical"] is None for r in group),
            "abstentions": na, "reference_positive": nr, "tp": tp, "fp": fp, "fn": fn,
            "accuracy": fraction(correct, 2424) if complete else None,
            "precision": fraction(tp, na) if behavior_complete else None,
            "recall": fraction(tp, nr) if behavior_complete else None,
            "abstention_f1": fraction(2 * tp, na + nr) if behavior_complete else None})
    save_rows(output / "source_reuse_rows.jsonl.gz", rows)
    save_rows(output / "score_rows.jsonl.gz", scored)
    save_rows(output / "pending_primary_rows.jsonl", pending)
    save_rows(output / "source_differences.jsonl", mismatch)
    pd.DataFrame(metrics).to_csv(output / "native_direct_conditions.csv", index=False)
    receipt = {"sources": sources, "matched_rows": len(scored), "original_unique_keys": len(required_keys),
        "source_difference_rows": len(mismatch), "pending_primary_rows": len(pending), "complete_conditions": sum(r["primary_complete"] for r in metrics),
        "label_closure_rows": total_labels, "closed_label_source_path": str(labels_path), "new_generations": 0,
        "new_API_calls": 0, "GPU_initialized": False, "frozen_scorer_source_sha256": file_hash(Path(frozen_score.__file__))}
    save_json(output / "reuse_validation.json", receipt)
    return receipt


def cda_audit(package, output, conditions, scores, export):
    output.mkdir(parents=True, exist_ok=False)
    representatives = set(pd.read_csv(package / "configs/attribution_representative101.csv").sample_id)
    if len(representatives) != 101:
        raise ValueError("Registered 101 representative inputs required")
    condition_map = {int(row.condition_id): row for row in conditions.itertuples() if row.method == "cda_visual"}
    selected = scores[scores.condition_id.isin(condition_map) & scores.sample_id.isin(representatives)]
    if len(selected) != 2020:
        raise ValueError("Finite CDA representative trace coverage differs")
    source_dict = pd.read_csv(export / "sources.csv").set_index("source_file_id")
    locators = defaultdict(dict)
    for row in selected.itertuples():
        locators[int(row.source_file_id)][int(row.source_line)] = row
    trace_records, summaries = [], defaultdict(lambda: {"responses": 0, "steps": 0, "recomputable_steps": 0,
        "missing_calibration_steps": 0, "zero_sum_steps": 0, "negative_wa_steps": 0, "wp_sum": 0.0, "wc_sum": 0.0, "wa_sum": 0.0,
        "max_equation_6_7_error": 0.0, "max_weights_sum_error": 0.0})
    located = 0
    for source_id, wanted in locators.items():
        path = Path(source_dict.loc[source_id, "real_path"])
        stop = max(wanted)
        with gzip.open(path, "rb") as stream:
            for number, line in enumerate(stream, 1):
                if number > stop:
                    break
                if number not in wanted:
                    continue
                score = wanted[number]
                condition = condition_map[int(score.condition_id)]
                raw = json.loads(line)
                if raw["sample"]["id"] != score.sample_id or any(raw[field] != getattr(condition, field) for field in FIELDS):
                    raise ValueError("CDA trace locator/condition identity differs")
                state = "A" if score.abstain else ("C" if score.correct_canonical else "E")
                group = summaries[(condition.model, condition.marker, state)]
                group["responses"] += 1
                residuals = []
                for step in raw["trace"]:
                    group["steps"] += 1
                    necessary = ("h_prior", "h_context", "h_null_prior", "h_null_context", "rp", "rc", "wp", "wc", "wa", "weights")
                    if not all(name in step for name in necessary):
                        group["missing_calibration_steps"] += 1
                        continue
                    hp, hc, hnp, hnc = (step[name] for name in necessary[:4])
                    if not np.isfinite([hp, hc, hnp, hnc]).all() or hnp <= 0 or hnc <= 0:
                        raise ValueError("Saved CDA null entropy invalid")
                    rp, rc = max(hp - hnp, 0) / hnp, max(hc - hnc, 0) / hnc
                    total = rp + rc
                    wp, wc = (rp * rp / total, rc * rc / total) if total > 0 else (0.0, 0.0)
                    wa = 1 - wp - wc
                    error = max(abs(step[name] - value) for name, value in zip(("rp", "rc", "wp", "wc", "wa"), (rp, rc, wp, wc, wa)))
                    error = max(error, max(abs(a - b) for a, b in zip(step["weights"], (wp, wc, wa))))
                    if error > 1e-10 or step["negative_wa"] != (wa < 0) or step["zero_sum_extension"] != (total == 0):
                        raise ValueError("CDA saved equations 6/7 differ")
                    group["recomputable_steps"] += 1
                    group["zero_sum_steps"] += total == 0
                    group["negative_wa_steps"] += wa < 0
                    for name, value in (("wp", wp), ("wc", wc), ("wa", wa)):
                        group[name + "_sum"] += value
                    group["max_equation_6_7_error"] = max(group["max_equation_6_7_error"], error)
                    group["max_weights_sum_error"] = max(group["max_weights_sum_error"], abs(sum(step["weights"]) - 1))
                    residuals.append(error)
                trace_records.append({"model": condition.model, "condition_id": int(score.condition_id), "sample_id": score.sample_id,
                    "marker": condition.marker, "state": state, "uniform_reference": bool(score.uniform_reference),
                    "source_path": str(path), "source_line": number, "raw_line_sha256": hashlib.sha256(line).hexdigest(),
                    "source_identity": raw["identity"], "original_key": raw["key"], "text": raw["text"], "tokens": raw["tokens"],
                    "trace": raw["trace"], "terminated": raw["terminated"], "equations_6_7_recomputed_steps": len(residuals),
                    "max_equations_6_7_error": max(residuals) if residuals else None,
                    "h_abstention_available": all("h_abstention" in s for s in raw["trace"]), "full_branch_logits_available": False})
                located += 1
    if located != 2020:
        raise ValueError("CDA selected trace source coverage incomplete")
    summary_rows = []
    for (model, marker, state), group in summaries.items():
        summary_rows.append({"model": model, "marker": marker, "state": state, "panel": "representative101_per_condition", **group,
            **{name + "_token_weighted_mean": fraction(group[name + "_sum"], group["recomputable_steps"]) for name in ("wp", "wc", "wa")},
            "zero_sum_fraction": fraction(group["zero_sum_steps"], group["recomputable_steps"]),
            "negative_wa_fraction": fraction(group["negative_wa_steps"], group["recomputable_steps"])})
    pd.DataFrame(summary_rows).to_csv(output / "trace_state_weight_summary.csv", index=False)
    save_rows(output / "source_bound_trace_rows.jsonl.gz", trace_records)
    receipt = {"selected_response_rows": located, "expected_response_rows": 2020, "conditions": 20,
        "representative_inputs_per_condition": 101, "total_steps": sum(r["steps"] for r in summary_rows),
        "recomputed_steps": sum(r["recomputable_steps"] for r in summary_rows),
        "missing_calibration_steps": sum(r["missing_calibration_steps"] for r in summary_rows),
        "max_equations_6_7_error": max(r["max_equation_6_7_error"] for r in summary_rows),
        "scope": "registered representative101 subset of existing twenty CDA conditions",
        "available_calibration_entropies": ["h_prior", "h_context", "h_null_prior", "h_null_context"],
        "abstention_entropy_and_full_branch_logits": "not_saved; equation4_vector_recomputation_unavailable",
        "zero_sum_rule": "wp=wc=0, wa=1 continuous extension", "negative_abstention_weights_preserved": True,
        "new_generations": 0, "new_API_calls": 0, "GPU_initialized": False, "cda_source_sha256": file_hash(ROOT / "src/kdm/cda.py")}
    save_json(output / "trace_validation.json", receipt)
    return receipt


def historical_alignment(package, output, run_output):
    """Align only the three named historical labels, without rescoring their census."""
    output.mkdir(parents=True, exist_ok=False)
    aliases = {"Gemma3-12B": "gemma3_12b", "Gemma3-4B": "gemma3_4b", "GLM-4.6V": "glm46v"}
    current_samples = {row["id"]: row for row in read_rows(ROOT / "data/current/all.jsonl")}
    audit_rows, source_records = [], {}
    for user_name, model in aliases.items():
        path = ROOT / "data/responses/census" / (model + ".jsonl.gz")
        sidecar_path = path.with_name(model + ".identity.json")
        sidecar = json.loads(sidecar_path.read_text())
        runtime_path = ROOT / "configs/runtime" / (model + ".json")
        runtime = json.loads(runtime_path.read_text())
        groups = defaultdict(lambda: {"n": 0, "sample_ids": set(), "identities": Counter(), "terminated_n": 0,
            "source_difference_n": 0, "configuration_hashes": set()})
        for row in read_rows(path):
            sample = row["sample"]
            group = groups[(sample["dataset"], sample["split"], row["guided"])]
            task = {"sample": sample, "method": "direct", "kind": "census" if row["guided"] else "unguided",
                "marker": "UNKNOWN", "reference_marker": "UNKNOWN", "guided": row["guided"],
                "reference_guided": row["guided"], "replicate": 0}
            differs = (sample != current_samples.get(sample["id"]) or row["key"] != task_id(model, task)
                or row["seed"] != stable_seed(sample["id"], model, 0)
                or row["prompt"] != task_prompt(sample["question"], guided=row["guided"])
                or row["config"] != asdict(DecodeConfig()) or row["status"] != "ok" or row["model"] != model)
            group["source_difference_n"] += differs
            group["n"] += 1
            group["sample_ids"].add(sample["id"])
            group["identities"][row["identity"]] += 1
            group["terminated_n"] += bool(row["terminated"])
            group["configuration_hashes"].add(stable_hash(row["config"]))
        source_records[model] = {"model_user_label": user_name, "registered_model": model,
            "hf_model_id": runtime["hf_model_id"], "source_path": str(path), "identity_path": str(sidecar_path),
            "runtime_path": str(runtime_path), "sidecar_identity": sidecar["identity"],
            "sidecar_identity_valid": stable_hash(sidecar["definition"]) == sidecar["identity"],
            "registered_backend_exact_match": sidecar["definition"]["backend"] == runtime,
            "groups": []}
        for (dataset, split, guided), group in sorted(groups.items()):
            expected = {sid for sid, sample in current_samples.items() if sample["dataset"] == dataset and sample["split"] == split}
            record = {"model_user_label": user_name, "model": model, "dataset": dataset, "split": split,
                "guided": guided, "n": group["n"], "unique_sample_ids": len(group["sample_ids"]),
                "current_expected_sample_ids": len(expected), "missing_sample_ids_n": len(expected - group["sample_ids"]),
                "extra_sample_ids_n": len(group["sample_ids"] - expected), "source_difference_n": group["source_difference_n"],
                "terminated_n": group["terminated_n"], "max_token_budget_reached_n": group["n"] - group["terminated_n"],
                "source_path": str(path), "row_identities": json.dumps(dict(group["identities"]), sort_keys=True),
                "configuration_hashes": json.dumps(sorted(group["configuration_hashes"]))}
            audit_rows.append(record)
            source_records[model]["groups"].append(record)
    pd.DataFrame(audit_rows).to_csv(output / "raw_current_alignment.csv", index=False)
    core = pd.read_csv(run_output / "frozen/all_conditions_352.csv")
    native = pd.read_csv(run_output / "native_direct/native_direct_conditions.csv")
    supplemental = ROOT / "outputs/supplemental/remaining11/run_20260930_140337"
    food_table = supplemental / "analysis/checkpoint_v7_20260930_1845/condition_metrics.csv"
    viz_table = supplemental / "scores/vizwiz_direct_baseline_20260930/condition_metrics.csv"
    food_metrics, viz_metrics = pd.read_csv(food_table), pd.read_csv(viz_table)
    records = []
    for dataset, filename in (("food101", "historical_food16_user_provided.csv"), ("vizwiz", "historical_vizwiz16_user_provided.csv")):
        table_path = package / "tables" / filename
        for old in pd.read_csv(table_path).to_dict("records"):
            if old["model_user_label"] not in aliases:
                continue
            model = aliases[old["model_user_label"]]
            source = source_records[model]
            record = {"dataset": dataset, "model": model, "model_user_label": old["model_user_label"],
                "hf_model_id": source["hf_model_id"], **{"historical_" + key: value for key, value in old.items() if key != "model_user_label"},
                "historical_table_path": str(table_path), "historical_original_source_identity": None,
                "historical_numbers_recomputed": False, "historical_identity_limit": "user-provided table does not bind original raw identity",
                "current_census_path": source["source_path"], "current_census_sidecar_identity": source["sidecar_identity"],
                "current_scoring_scope": "original101 canonical_name_in_primary_score" if dataset == "food101" else "official consensus; abstention/invalid zero",
                "between_scope_accuracy_difference_is_method_effect": False}
            current = None
            if model == "gemma3_4b" and dataset == "food101":
                current = core[core.model.eq(model) & core.method.eq("direct") & core.marker.eq("UNKNOWN")].iloc[0].to_dict()
                un = native[native.model.eq(model)].iloc[0].to_dict()
                record.update(current_native_unguided_accuracy=un["accuracy"], current_native_unguided_primary_pending=un["canonical_pending"],
                    current_native_unguided_n=un["n"], current_native_unguided_abstentions=un["abstentions"],
                    current_native_unguided_table=str(run_output / "native_direct/native_direct_conditions.csv"))
                record.update(current_guided_n=current["n"], current_guided_accuracy=current["accuracy"] / 100,
                    current_guided_abstentions=current["abstentions"], current_guided_table=str(run_output / "frozen/all_conditions_352.csv"),
                    current_guided_split="eval", current_guided_condition_id=current["condition_id"])
            elif model != "gemma3_4b":
                table = food_metrics if dataset == "food101" else viz_metrics
                selected = table[table.model.eq(model) & table.method.eq("direct") & table.marker.eq("UNKNOWN")]
                if len(selected) != 1:
                    raise ValueError("Historical alignment current result is ambiguous")
                current = selected.iloc[0].to_dict()
                record.update(current_guided_n=current["n"],
                    current_guided_accuracy=current["canonical_accuracy"] if dataset == "food101" else current["all_input_vqa_accuracy"],
                    current_guided_abstentions=current["abstain_n"], current_guided_split=current["split"],
                    current_guided_table=str(food_table if dataset == "food101" else viz_table),
                    current_guided_source_identities=current.get("source_identities"),
                    current_guided_reference_complete=current.get("reference_condition_complete") if dataset == "food101" else False)
            else:
                record.update(current_guided_n=None, current_guided_accuracy=None,
                    current_metric_unavailable_reason="no core Gemma3-4B VizWiz canonical table in this bounded CPU audit")
            record["historical_and_current_guided_denominator_equal"] = old["n"] == record.get("current_guided_n")
            records.append(record)
    pd.DataFrame(records).to_csv(output / "historical_named_models_alignment.csv", index=False)
    receipt = {"historical_table_rows": len(records), "actual_named_raw_rows": sum(row["n"] for row in audit_rows),
        "current_raw_alignment_groups": len(audit_rows), "raw_source_difference_rows": sum(row["source_difference_n"] for row in audit_rows),
        "sources": source_records, "unknown_original_historical_identities": 6,
        "known_scope_differences": ["historical Food uses dev+eval4848 and aliases; current canonical result eval2424",
            "historical VizWiz uses official validation4319; current supplemental table eval3501",
            "current named supplemental result identities may differ from current census sidecars; lineage is retained separately"],
        "new_generations": 0, "new_API_calls": 0, "GPU_initialized": False}
    save_json(output / "historical_validation.json", receipt)
    return receipt


def verify_existing_output(output):
    """Verify the derived P0 artifacts without rescanning their frozen sources."""
    frozen = pd.read_csv(output / "frozen/all_conditions_352.csv")
    assert len(frozen) == 352 and int(frozen.n.sum()) == 853248 and (frozen.n == 2424).all()
    assert (frozen.correct + frozen.wrong + frozen.abstentions == frozen.n).all()
    assert (frozen.tp + frozen.fp == frozen.abstentions).all()
    assert (frozen.tp + frozen.fn == frozen.reference_positive).all()
    assert np.allclose(frozen.accuracy, frozen.correct / frozen.n * 100)
    assert np.allclose(frozen.recall, frozen.tp / frozen.reference_positive * 100)
    assert np.allclose(frozen.abstention_f1, 2 * frozen.tp / (frozen.abstentions + frozen.reference_positive) * 100)
    same = pd.read_csv(output / "frozen/guided_same_marker_4_per_model.csv")
    cross = pd.read_csv(output / "frozen/guided_cross_reference_16_per_model.csv")
    assert (same.marker == same.reference_marker).all()
    assert (same.groupby(["model", "method"]).size() == 4).all()
    assert (cross.groupby(["model", "method"]).size() == 16).all()
    transitions = pd.read_csv(output / "frozen/transitions_reference_stratified.csv")
    pair_cols = ["comparison", "before_condition_id", "after_condition_id"]
    sums = transitions.groupby(pair_cols).n.sum()
    assert (sums == 2424).all() and (transitions.groupby(pair_cols).size() == 18).all()
    by_stratum = transitions.groupby(pair_cols + ["uniform_reference"]).agg(n=("n", "sum"), expected=("reference_stratum_denominator", "first"))
    assert (by_stratum.n == by_stratum.expected).all()
    native_rows = list(read_rows(output / "native_direct/score_rows.jsonl.gz"))
    native = pd.DataFrame(native_rows)
    assert len(native) == 12120 and not native.duplicated(["model", "sample_id"]).any()
    assert (native.groupby(["model", "target_class"]).size() == 24).all()
    assert not native.abstain.isna().any() and native.correct_canonical.dropna().isin([0, 1]).all()
    assert not ((native.correct_canonical == 1) & native.abstain).any()
    pending = list(read_rows(output / "native_direct/pending_primary_rows.jsonl"))
    assert {row["original_key"] for row in pending} == set(native.loc[native.correct_canonical.isna() | native.abstain.isna(), "original_key"])
    native_table = pd.read_csv(output / "native_direct/native_direct_conditions.csv")
    trace_rows = list(read_rows(output / "cda_trace/source_bound_trace_rows.jsonl.gz"))
    trace = pd.DataFrame(trace_rows)
    assert len(trace) == 2020 and not trace.duplicated(["condition_id", "sample_id"]).any()
    assert (trace.groupby("condition_id").size() == 101).all()
    assert sum(row["equations_6_7_recomputed_steps"] for row in trace_rows) == 18545
    assert max(row["max_equations_6_7_error"] for row in trace_rows) == 0
    inventory = pd.read_csv(output / "frozen/method_inventory.csv").to_dict("records")
    source_receipt = json.loads((output / "native_direct/reuse_validation.json").read_text())
    identities = {row["model"]: row["source_identity"] for row in source_receipt["sources"]}
    for row in native_table.to_dict("records"):
        inventory.append({"model": row["model"], "method": "direct", "kind": "unguided", "marker": "UNKNOWN",
            "reference_marker": "UNKNOWN", "guided": False, "reference_guided": False, "replicate": 0, "split": "eval",
            "method_family": row["method_family"], "same_marker": True, "available": True, "n": row["n"], "expected_n": 2424,
            "primary_complete": row["primary_complete"], "primary_pending": row["canonical_pending"],
            "main_prompt": task_prompt("What specific food is shown in this image?", guided=False), "reference_prompt": None,
            "neutral_prompt": None, "decode_config": json.dumps(asdict(DecodeConfig()), sort_keys=True),
            "source_namespace": "existing_census_source_bound_reuse", "source_identity": identities[row["model"]],
            "source_scores": str(output / "native_direct/score_rows.jsonl.gz"), "scope": "existing_registered_native_Direct"})
    pd.DataFrame(inventory).to_csv(output / "method_inventory_362_rows.csv", index=False)
    receipt = {"verified_utc": datetime.now(timezone.utc).isoformat(), "frozen_conditions": len(frozen),
        "frozen_score_rows": int(frozen.n.sum()), "same_marker_conditions": len(same), "cross_marker_conditions": len(cross),
        "reference_stratified_pairs": len(sums), "reference_stratified_transition_rows": len(transitions),
        "native_direct_rows": len(native), "native_direct_unique_model_sample_keys": len(native),
        "native_direct_class_quota_checks": int(len(native.groupby(["model", "target_class"]))),
        "native_direct_primary_pending": len(pending), "native_direct_complete_conditions": int(native_table.primary_complete.sum()),
        "cda_representative_trace_rows": len(trace), "cda_trace_steps": 18545, "cda_max_equations_6_7_error": 0,
        "method_inventory_rows": len(inventory), "new_native_VCD_rows_available": 0, "missing_native_VCD_rows": 12120,
        "synthetic_tests_are_empirical_generation": False, "historical_tables_are_recomputed_experience": False,
        "source_rescan_required": False, "code_sha256": file_hash(Path(__file__))}
    save_json(output / "P0_CPU_ACCEPTANCE.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--census-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--export", type=Path, default=ROOT / "outputs/paper_20260929")
    parser.add_argument("--history-only", action="store_true")
    parser.add_argument("--verify-output", action="store_true")
    args = parser.parse_args()
    output, package, export = args.output.resolve(), args.package.resolve(), args.export.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.verify_output:
        print(json.dumps(clean(verify_existing_output(output)), ensure_ascii=False, indent=2))
        return
    if args.history_only:
        history = historical_alignment(package, output / "historical_alignment", output)
        print(json.dumps(clean({key: value for key, value in history.items() if key != "sources"}), ensure_ascii=False, indent=2))
        return
    conditions, scores, references, arrays, frozen = frozen_audit(package, output / "frozen", export)
    native = native_direct_audit(output / "native_direct", references, args.census_labels.resolve())
    cda = cda_audit(package, output / "cda_trace", conditions, scores, export)
    save_json(output / "P0_CPU_SUMMARY.json", {"updated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": frozen, "native_direct": native, "cda_trace": cda,
        "new_GPU_generations": 0, "new_API_calls": 0, "scientific_parameters_changed": False,
        "authorship": "programmatic CPU audit and existing source-bound annotations; no new semantic model judgment",
        "code_sha256": file_hash(Path(__file__))})
    print(json.dumps(clean({"frozen": frozen, "native_direct": native, "cda_trace": cda}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
