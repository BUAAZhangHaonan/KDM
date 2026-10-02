#!/usr/bin/env python3
"""Assemble explicit accepted Food sources into the registered nine-model main panel."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kdm.io import file_hash, read_jsonl, stable_hash, within
from workflows.main_results.analysis import BOOT, SEED
from workflows.paper_core.dev_selected_eval import metrics, paired
from workflows.paper_core.extended_food_union import effective_holds
from workflows.paper_core.score_native_baseline_parts import reference_map

FIVE = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b")
EXTRA = ("internvl35_8b", "onevision", "phi35", "qwen3vl")
MODELS = FIVE + EXTRA
MARKERS = ("UNKNOWN", "UNCLEAR", "UNSURE", "I cannot identify it")
FIELDS = ("model", "dataset", "split", "method", "kind", "marker",
          "reference_marker", "guided", "reference_guided", "replicate")


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            require(line.endswith("\n"), "An accepted source has an incomplete JSONL record")
            yield number, json.loads(line), hashlib.sha256(line.encode("utf-8")).hexdigest()


def output_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def condition(row):
    value = {k: row[k] for k in FIELDS}
    for name in ("guided", "reference_guided"):
        require(type(value[name]) in (bool, np.bool_), "A condition flag is not Boolean")
        value[name] = bool(value[name])
    value["replicate"] = int(value["replicate"])
    # The accepted core census omits inactive marker fields. Its explicit
    # recovery is the same one used by dev_selected_eval.py.
    if value["method"] == "direct" and not value["guided"]:
        value.update(marker="NONE", reference_marker="NONE")
    return value


def condition_id(value):
    return stable_hash(value)


def main_metrics(frame, identity, fixed_direct):
    if fixed_direct is not None:
        return metrics(frame, identity, fixed_direct)
    correct, abstain, reference = frame.correct_canonical.eq(1), frame.abstain.astype(bool), frame.uniform_reference.astype(bool)
    n, c, a, r = len(frame), int(correct.sum()), int(abstain.sum()), int(reference.sum())
    tp = int((abstain & reference).sum())
    fp = a - tp
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    return {**identity, "n": n, "C": c, "W": n - c - a, "A": a, "TP": tp, "FP": fp,
            "FN": r - tp, "TN": n - r - fp, "reference_positive": r,
            "accuracy": c / n, "precision": ratio(tp, a), "recall": ratio(tp, r),
            "F1": ratio(2 * tp, a + r), "J": (c + tp) / n, "J_numerator": c + tp,
            "unnecessary_abstention_rate": ratio(fp, n - r), "unnecessary_abstention_denominator": n - r,
            "fixed_reasonable_retained": None, "fixed_reasonable_denominator": None,
            "fixed_reasonable_retention": None, "fixed_reasonable_baseline_condition_id": None,
            "precision_undefined_reason": "no_abstentions" if not a else "",
            "recall_undefined_reason": "no_reference_positive" if not r else "",
            "fixed_retention_undefined_reason": "registered_guided_Direct_fixed_set_unavailable"}


def requested_conditions(plan):
    result = {}
    for model in MODELS:
        definitions = [("direct", "unguided", "NONE", False),
                       ("dola", "native_unguided", "NONE", False),
                       ("deco", "native_unguided", "NONE", False),
                       ("vcd", "native_unguided", "NONE", False),
                       ("cda_visual", "instruction_preserving", "UNKNOWN", True)]
        if "sid" in plan[model]["food101"]:
            definitions.append(("sid", "native_unguided", "NONE", False))
        definitions += [(method, "instruction_preserving", marker, True)
                        for method in ("instruction_vcd", "instruction_m3id") for marker in MARKERS]
        for method, kind, marker, guided in definitions:
            identity = dict(model=model, dataset="food101", split="eval", method=method,
                            kind=kind, marker=marker, reference_marker=marker, guided=guided,
                            reference_guided=False, replicate=0)
            result[condition_id(identity)] = identity
    require(len(result) == 123, "The main comparison must contain exactly 123 registered conditions")
    return result


def frozen_records(manifest, source_info):
    score_path = within(ROOT, manifest["frozen_scores"])
    condition_path = within(ROOT, manifest["frozen_conditions"])
    frame = pd.read_parquet(score_path)
    definitions = pd.read_csv(condition_path)
    require(len(frame) == 853248 and len(definitions) == 352
            and not frame.duplicated(["condition_id", "sample_id"]).any(),
            "The accepted five-model frozen panel differs")
    metadata = definitions.set_index("condition_id").to_dict("index")
    source_info.update(id="frozen_five", path=str(score_path.relative_to(ROOT)),
                       sha256=file_hash(score_path), rows=len(frame), accepted_frozen=True,
                       conditions_path=str(condition_path.relative_to(ROOT)),
                       conditions_sha256=file_hash(condition_path))
    require(source_info["sha256"] == manifest["frozen_scores_sha256"],
            "The accepted frozen score publication SHA differs")
    for position, row in enumerate(frame.to_dict("records"), 1):
        meta = metadata[row["condition_id"]]
        original = {**row, **{k: meta[k] for k in FIELDS if k in meta}, "dataset": "food101", "split": "eval"}
        yield position, original, None


def source_records(source, source_info):
    path = within(ROOT, source["path"])
    receipt_path = within(ROOT, source["receipt"])
    receipt = load(receipt_path)
    digest = file_hash(path)
    require(source["expected_sha256"] == digest, "An explicit accepted source file changed")
    require(source["expected_receipt_sha256"] == file_hash(receipt_path), "An accepted receipt changed")
    checks = source.get("receipt_require", {"passed": True})
    require(all(receipt.get(k) == value for k, value in checks.items()),
            "The accepted source receipt does not satisfy its explicit validation")
    source_info.update(id=source["id"], path=str(path.relative_to(ROOT)), sha256=digest,
                       receipt=str(receipt_path.relative_to(ROOT)), receipt_sha256=file_hash(receipt_path),
                       receipt_schema=receipt.get("schema"), receipt_validation=checks)
    yield from json_rows(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paired", action="store_true", help="Compute registered class bootstrap comparisons")
    args = parser.parse_args()
    manifest_path = within(ROOT, args.manifest)
    manifest = load(manifest_path)
    require(manifest["schema"] == "kdm_nine_main_food_explicit_sources_v1", "An explicit source manifest is required")
    output = within(ROOT, args.output)
    output.relative_to(ROOT / "outputs/paper_core_20261002_dev_viz")
    output.mkdir(parents=True, exist_ok=False)
    samples = {s["id"]: s for s in read_jsonl(ROOT / "data/current/all.jsonl")
               if s["dataset"] == "food101" and s["split"] == "eval"}
    require(len(samples) == 2424 and len(Counter(s["class"] for s in samples.values())) == 101
            and set(Counter(s["class"] for s in samples.values()).values()) == {24},
            "The frozen Food eval roster must contain 101 classes by 24 images")
    references, reference_sources = reference_map()
    require(BOOT == 2000 and SEED == 20260929, "The registered bootstrap identity changed")
    wanted = requested_conditions(load(ROOT / "configs/kdm/method_plan.json"))
    identity_holds, overlays = {}, {}
    for relative in manifest.get("effective_hold_files", []):
        path = within(ROOT, relative)
        for item in load(path):
            identity_holds[item["key"]] = item["actual_hold_reasons"]
    overlay_sources = []
    for item in manifest.get("accepted_overlays", []):
        path = within(ROOT, item["path"])
        require(file_hash(path) == item["sha256"], "An explicitly accepted correction source changed")
        for line, row, digest in json_rows(path):
            binding = row["old_score_source"], row["old_score_physical_line"]
            require(binding not in overlays, "An accepted append-only correction is repeated")
            overlays[binding] = {**row, "correction_source": str(path.relative_to(ROOT)),
                                 "correction_source_line": line, "correction_source_sha256": item["sha256"],
                                 "correction_line_sha256": digest}
        overlay_sources.append({"path": str(path.relative_to(ROOT)), "sha256": item["sha256"]})
    selected, fixed, source_summaries, conflicts, duplicates = {}, {}, [], [], []
    entries = [{"id": "frozen_five", "format": "frozen_parquet"}, *manifest["sources"]]
    overlay_applied = []
    for source in entries:
        info, candidates, auxiliaries, source_conflicts = {}, {}, {}, []
        iterator = frozen_records(manifest, info) if source.get("format") == "frozen_parquet" else source_records(source, info)
        count, relevant, pending = 0, 0, 0
        for line, original, line_sha in iterator:
            count += 1
            if original.get("model") not in MODELS or original.get("dataset", "food101") != "food101" or original.get("split", "eval") != "eval":
                continue
            row = dict(original)
            if row["method"] == "direct" and row.get("guided") is False:
                row.setdefault("marker", "NONE")
                row.setdefault("reference_marker", "NONE")
                row.setdefault("replicate", 0)
            identity = condition(row)
            cid = condition_id(identity)
            auxiliary = row["method"] == "direct" and row["guided"] and row["kind"] == "main" and row["marker"] == row["reference_marker"]
            if cid not in wanted and not auxiliary:
                continue
            relevant += 1
            sample = samples.get(row["sample_id"])
            require(sample is not None and row["target_class"] == sample["class"], "A main source has an unregistered sample or target")
            if "question" in row:
                require(row["question"] == sample["question"], "A source question differs from the frozen roster")
            c = row.get("canonical_name_in_primary_score", row.get("correct_canonical"))
            literal = row.get("literal_extracted_name_score", row.get("correct_literal"))
            abstain = row.get("abstain")
            require(c in (0, 1, None) and literal in (0, 1, None) and (abstain is None or type(abstain) in (bool, np.bool_)), "A score has an invalid decision type")
            c = None if c is None else int(c)
            literal = None if literal is None else int(literal)
            reference = references[(row["model"], row["sample_id"])]
            for field in ("uniform_reference", "reference_G"):
                if row.get(field) is not None:
                    require(type(row[field]) in (bool, np.bool_) and bool(row[field]) == reference, "An original reference conflicts with its frozen model reference")
            holds = effective_holds(row)
            if source.get("apply_effective_identity_holds", False):
                holds += identity_holds.get(row.get("key"), [])
            overlay = overlays.get((str(ROOT / info["path"]), line))
            if overlay is not None:
                require(row["qa_key"] == overlay["qa_key"] and c == overlay["old_canonical"] and abstain == overlay["old_abstain"], "An accepted correction does not bind the original score")
                c, literal, abstain = overlay["new_canonical"], overlay["new_literal"], overlay["new_abstain"]
                overlay_applied.append({"source": info["id"], "source_line": line, **overlay})
            require(not (c == 1 and abstain is True), "Correct answers and abstentions must be disjoint")
            decided = c is not None and literal is not None and abstain is not None and not holds
            pending += not decided
            compact = {**identity, "condition_id": cid, "sample_id": row["sample_id"],
                       "target_class": row["target_class"], "correct_canonical": c,
                       "correct_literal": literal, "abstain": None if abstain is None else bool(abstain),
                       "uniform_reference": reference, "reference_complete": True,
                       "source_id": info["id"], "source_score_path": info["path"],
                       "source_score_sha256": info["sha256"], "source_score_line": line,
                       "source_score_line_sha256": line_sha, "source_task_key": row.get("key", row.get("original_key")),
                       "source_condition_id": row.get("condition_id"), "source_identity": row.get("source_identity"),
                       "source_full_condition_identity": {k: original.get(k) for k in FIELDS},
                       "source_file_id": row.get("source_file_id"), "source_raw_line": row.get("source_line"),
                       "qa_key": row.get("qa_key"), "question": row.get("question"), "answer": row.get("answer"),
                       "seed": row.get("seed"), "config": row.get("config", row.get("decode_config")),
                       "tokens": row.get("tokens"), "terminated": row.get("terminated"),
                       "original_source_preserved": True, "accepted_complete_decision": decided,
                       "effective_hold_reasons": holds, "applied_correction": overlay}
            key = cid, row["sample_id"]
            pool = auxiliaries if auxiliary else candidates
            require(key not in pool, "An explicit source repeats a main condition/sample key")
            pool[key] = compact
        info.update(rows_read=count, main_or_fixed_set_rows=relevant, pending_or_held_rows=pending)
        for pool, existing in ((candidates, selected), (auxiliaries, fixed)):
            for key, candidate in pool.items():
                old = existing.get(key)
                if old is None:
                    continue
                compare = ("target_class", "correct_canonical", "correct_literal", "abstain", "uniform_reference")
                mismatched = [name for name in compare if old[name] is not None
                              and candidate[name] is not None and old[name] != candidate[name]]
                for name in ("qa_key", "question", "answer", "config", "seed"):
                    if old.get(name) is not None and candidate.get(name) is not None and old[name] != candidate[name]:
                        mismatched.append(name)
                if mismatched:
                    source_conflicts.append({"condition_id": key[0], "sample_id": key[1],
                        "old_source": old["source_id"], "new_source": info["id"], "fields": mismatched,
                        "old_source_line": old["source_score_line"], "new_source_line": candidate["source_score_line"]})
        info["conflicts"] = len(source_conflicts)
        if source_conflicts:
            conflicts.extend(source_conflicts)
            info["source_accepted_into_derived_panel"] = False
        else:
            info["source_accepted_into_derived_panel"] = True
            for pool, existing in ((candidates, selected), (auxiliaries, fixed)):
                for key, candidate in pool.items():
                    old = existing.get(key)
                    if old is None:
                        existing[key] = candidate
                    else:
                        duplicates.append({"condition_id": key[0], "sample_id": key[1],
                            "prior_source": old["source_id"], "additional_source": candidate["source_id"],
                            "prior_line": old["source_score_line"], "additional_line": candidate["source_score_line"],
                            "same_score_and_QA": True})
                        if not old["accepted_complete_decision"] and candidate["accepted_complete_decision"]:
                            candidate["prior_ineligible_source"] = old
                            existing[key] = candidate
        source_summaries.append(info)
    coverage, complete, missing = [], {}, []
    for cid, identity in wanted.items():
        records = [r for (condition_key, _), r in selected.items() if condition_key == cid]
        ids = {r["sample_id"] for r in records}
        closed = ids == set(samples) and all(r["accepted_complete_decision"] for r in records)
        counts = Counter(r["target_class"] for r in records)
        if closed:
            require(len(counts) == 101 and set(counts.values()) == {24}, "A full condition lacks its 101-by-24 class quota")
            complete[cid] = pd.DataFrame(records).set_index("sample_id", drop=False).sort_index()
        coverage.append({**identity, "condition_id": cid, "expected_n": 2424,
                         "observed_n": len(records), "decided_n": sum(r["accepted_complete_decision"] for r in records),
                         "missing_n": 2424 - len(records), "pending_or_held_n": sum(not r["accepted_complete_decision"] for r in records),
                         "full_condition_complete": closed, "class_count": len(counts),
                         "source_ids": json.dumps(sorted({r["source_id"] for r in records}))})
        missing += [{"condition_id": cid, **identity, "sample_id": sid, "gap_kind": "no_bound_score_source"}
                    for sid in sorted(set(samples) - ids)]
        missing += [{"condition_id": cid, **identity, "sample_id": r["sample_id"],
                     "gap_kind": "bound_score_pending_or_held", "source_id": r["source_id"],
                     "effective_hold_reasons": r["effective_hold_reasons"]} for r in records if not r["accepted_complete_decision"]]
    fixed_frames = {}
    for model in MODELS:
        for marker in MARKERS:
            group = [r for r in fixed.values() if r["model"] == model and r["marker"] == marker]
            if len(group) == 2424 and all(r["accepted_complete_decision"] for r in group):
                frame = pd.DataFrame(group).set_index("sample_id", drop=False).sort_index()
                frame.attrs["condition_id"] = group[0]["source_condition_id"] or group[0]["condition_id"]
                frame.attrs["condition_identity"] = {name: group[0][name] for name in FIELDS}
                fixed_frames[(model, marker)] = frame
    control_path = within(ROOT, manifest["frozen_control_selections"])
    controls = pd.read_parquet(control_path, columns=["sample_id", "direct_condition_id", "direct_abstain"])
    controls = controls.drop_duplicates(["direct_condition_id", "sample_id"])
    for (model, marker), frame in fixed_frames.items():
        if model not in FIVE:
            continue
        cid = frame.attrs["condition_id"]
        observed_control = controls[controls.direct_condition_id.eq(cid)].set_index("sample_id").sort_index()
        require(len(observed_control) == 2424 and observed_control.index.equals(frame.index)
                and observed_control.direct_abstain.astype(bool).equals(frame.abstain.astype(bool)),
                "The original frozen control selection does not preserve its guided Direct fixed set")
    all_metrics = []
    for cid, frame in complete.items():
        identity = {**wanted[cid], "condition_id": cid, "checkpoint": load(ROOT / ("configs/runtime/" + wanted[cid]["model"] + ".json"))["hf_model_id"],
                    "main_marker": wanted[cid]["marker"], "score_source": "explicit_source_record_index"}
        marker = identity["marker"] if identity["guided"] else "UNKNOWN"
        baseline = fixed_frames.get((identity["model"], marker))
        value = main_metrics(frame, identity, baseline)
        literal_correct = int(frame.correct_literal.eq(1).sum())
        value.update(J_denominator=len(frame), accuracy_numerator=value["C"], accuracy_denominator=len(frame),
                     precision_numerator=value["TP"], precision_denominator=value["A"],
                     recall_numerator=value["TP"], recall_denominator=value["reference_positive"],
                     F1_numerator=2 * value["TP"], F1_denominator=value["A"] + value["reference_positive"],
                     F1_undefined_reason="zero_abstentions_and_reference_positive" if value["A"] + value["reference_positive"] == 0 else "",
                     literal_correct=literal_correct, literal_denominator=len(frame),
                     literal_accuracy=literal_correct / len(frame), literal_pending=0,
                     canonical_pending=0, abstain_pending=0, reference_pending=0)
        value.update(fixed_reasonable_baseline_marker=marker, fixed_reasonable_baseline_guided=True,
                     fixed_set_source_scope="same_marker_original_guided_Direct_R_positive",
                     source_ids=json.dumps(sorted(set(frame.source_id))), selection="registered_configuration")
        all_metrics.append(value)
    observed, endpoint_rows = [], []
    for model in MODELS:
        for method in ("instruction_vcd", "instruction_m3id"):
            choices = [r for r in all_metrics if r["model"] == model and r["method"] == method]
            if len(choices) != 4:
                continue
            chosen = min(choices, key=lambda r: (-r["J"], MARKERS.index(r["marker"])))
            observed.append({**chosen, "selection": "best_observed_eval_joint_J", "selection_split": "eval", "dev_selected": False})
            for metric in ("J", "accuracy", "precision", "recall", "fixed_reasonable_retention"):
                eligible = [r for r in choices if r[metric] is not None]
                if eligible:
                    best = min(eligible, key=lambda r: (-r[metric], MARKERS.index(r["marker"])))
                    endpoint_rows.append({**best, "selection": "best_observed_eval_independent_endpoint",
                                          "selected_metric": metric, "selected_metric_value": best[metric], "dev_selected": False})
    dev_selected = []
    for relative in manifest.get("dev_selected_configs", []):
        path = within(ROOT, relative)
        for choice in load(path):
            if choice["method"] not in {"instruction_vcd", "instruction_m3id"}:
                continue
            require(choice["model"] in FIVE and choice["selection"] == "dev_selected"
                    and choice["selection_split"] == "dev" and choice["n"] == 404,
                    "A dev-selected source lacks its actual dev404 selection identity")
            matching = [r for r in all_metrics if r["model"] == choice["model"] and r["method"] == choice["method"] and r["marker"] == choice["marker"]]
            if matching:
                require(len(matching) == 1, "A dev-selected working point is ambiguous")
                dev_selected.append({**matching[0], "selection": "dev_selected", "dev_selected": True,
                                     "dev_selection_source": str(path.relative_to(ROOT)),
                                     "dev_selection_source_sha256": file_hash(path),
                                     "actual_frozen_dev_selection": json.dumps(choice, ensure_ascii=False, sort_keys=True)})
    effects, transitions = [], []
    if args.paired:
        draws = np.random.RandomState(SEED).randint(0, 101, size=(BOOT, 101))
        for point in [*observed, *dev_selected]:
            model = point["model"]
            a = complete[point["condition_id"]]
            baselines = [r for r in all_metrics if r["model"] == model and r["method"] in {"direct", "vcd", "cda_visual", "dola", "deco", "sid"}]
            for baseline in baselines:
                b = complete[baseline["condition_id"]]
                identity = {"model": model, "method": point["method"], "marker": point["marker"],
                            "selection": point["selection"], "condition_id": point["condition_id"],
                            "baseline_method": baseline["method"], "baseline_condition_id": baseline["condition_id"],
                            "comparison": "same_frozen_food_eval_main_baseline"}
                effect_rows, transition_rows = paired(
                    a, b, identity, draws,
                    fixed_direct=fixed_frames.get((model, point["marker"])))
                effects.extend(effect_rows)
                transitions.extend(transition_rows)
    pd.DataFrame(coverage).to_csv(output / "condition_coverage.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(output / "metrics_all_complete.csv", index=False)
    pd.DataFrame(observed).to_csv(output / "best_observed_joint_operating_points.csv", index=False)
    pd.DataFrame(endpoint_rows).to_csv(output / "best_observed_independent_endpoints.csv", index=False)
    pd.DataFrame(dev_selected).to_csv(output / "dev_selected_operating_points.csv", index=False)
    pd.DataFrame(effects).to_csv(output / "paired_main_comparisons.csv", index=False)
    pd.DataFrame(transitions).to_csv(output / "transitions.csv", index=False)
    with gzip.open(output / "missing_or_pending_keys.jsonl.gz", "xt", encoding="utf-8", compresslevel=1) as stream:
        for row in missing:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    score_frame = pd.DataFrame(list(selected.values()))
    # Historical provider records use heterogeneous nested annotation schemas.
    # Preserve their complete provenance as JSON instead of coercing those
    # mixed list/scalar fields into an Arrow struct or changing any decision.
    for name in ("applied_correction", "prior_ineligible_source"):
        if name in score_frame:
            score_frame[name + "_json"] = score_frame[name].map(
                lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
                if isinstance(value, dict) else None)
            score_frame.drop(columns=[name], inplace=True)
    score_frame.to_parquet(output / "main_score_rows.parquet", index=False)
    output_json(output / "sources.json", source_summaries)
    output_json(output / "source_conflicts.json", conflicts)
    output_json(output / "duplicate_score_source_bindings.json", duplicates)
    output_json(output / "accepted_corrections_applied.json", overlay_applied)
    receipt = {"schema": "kdm_nine_main_food_accepted_source_panel_v1", "passed": not conflicts,
               "requested_conditions": 123, "requested_rows": 298152, "actual_bound_score_rows": len(selected),
               "complete_conditions": len(complete), "complete_condition_rows": 2424 * len(complete),
               "missing_or_pending_rows": len(missing), "complete_main_panel": len(complete) == 123 and not conflicts,
               "best_observed_joint_points": len(observed), "actual_dev_selected_points": len(dev_selected),
               "reference_sources": reference_sources, "accepted_overlay_sources": overlay_sources,
               "frozen_control_selections": {"path": str(control_path.relative_to(ROOT)), "sha256": file_hash(control_path)},
               "accepted_overlay_members_applied": len(overlay_applied), "source_conflict_count": len(conflicts),
               "source_manifest": str(manifest_path.relative_to(ROOT)), "source_manifest_sha256": file_hash(manifest_path),
               "runner_sha256": file_hash(Path(__file__)), "actual_command": [sys.executable, *sys.argv],
               "created_utc": datetime.now(timezone.utc).isoformat(), "GPU_initialized": False,
               "new_semantic_judgments": 0, "original_source_objects_modified": 0,
               "incomplete_conditions_in_main_metrics": 0, "matrix_conditions_in_main_metrics": 0,
               "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    output_json(output / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("passed", "requested_conditions", "actual_bound_score_rows",
                                             "complete_conditions", "missing_or_pending_rows", "complete_main_panel")}))


if __name__ == "__main__":
    main()
