#!/usr/bin/env python3
"""Audit the independently registered VizWiz queue without GPU execution."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.decoding import DecodeConfig
from kdm.execution import resolve_image_path
from kdm.io import file_hash, read_jsonl, stable_hash, stable_seed
from kdm.pipeline import census_tasks, experiment_tasks, task_id
from kdm.prompts import task_prompt
from workflows.supplemental.remaining11.inventory import MODELS, CENSUS_PREFIX, independent_task

CONDITION_FIELDS = ("model", "dataset", "split", "method", "kind", "marker", "reference_marker",
                    "guided", "reference_guided", "replicate")
ARCHIVE = Path("/home/g203-4028/projects/knowledge-deficit-mitigation-archive/20260929/historical_outputs.tar")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def dump_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def row_stream(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for number, raw_line in enumerate(stream, 1):
            yield number, json.loads(raw_line), hashlib.sha256(raw_line).hexdigest()


def source_runtime_differences(source, current):
    fields = ("factory", "hf_model_id", "dtype", "thinking_mode", "versions", "input_limits")
    differences = {field: {"source": source.get(field), "current": current.get(field)}
                   for field in fields if source.get(field) != current.get(field)}
    for field in ("dtype", "model_path"):
        a, b = source.get("kwargs", {}).get(field), current.get("kwargs", {}).get(field)
        if a != b:
            differences["kwargs." + field] = {"source": a, "current": b}
    weights = lambda spec: {(r["filename"], r.get("hub_recorded_sha256")) for r in spec.get("weights", [])}
    if weights(source) != weights(current):
        differences["weights"] = {"source": sorted(weights(source)), "current": sorted(weights(current))}
    if source.get("processor", {}).get("files") != current.get("processor", {}).get("files"):
        differences["processor_files"] = True
    return differences


def recover_if_needed(member_name, run, output, archive_files):
    existing = run / "recovered" / member_name
    if existing.exists():
        return existing, None
    entry = archive_files.get(member_name)
    if entry is None:
        raise ValueError("Required exact member missing from finite archive manifest: " + member_name)
    destination = output / "recovered" / member_name
    if Path(member_name).is_absolute() or ".." in Path(member_name).parts:
        raise ValueError("Unsafe exact recovery member")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:") as archive:
        member = archive.getmember(member_name)
        if not member.isfile() or member.size != entry["bytes"]:
            raise ValueError("Exact archive member type/size differs")
        source = archive.extractfile(member)
        if source is None:
            raise ValueError("Exact archive member is unreadable")
        with source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
    return destination, {"member": member_name, "path": str(destination), "bytes": entry["bytes"],
                         "source_container": str(ARCHIVE), "historical_recorded_sha256": entry["sha256"]}


def official_references(samples, output):
    path = ROOT / "cache/assets/vizwiz/extracted/val.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    by_image = {row["image"]: row for row in original}
    if len(original) != 4319 or len(by_image) != 4319:
        raise ValueError("Original official VizWiz validation coverage differs")
    answerability = Counter()
    image_count = 0
    with (output / "official_reference.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for sid, sample in sorted(samples.items()):
            row = by_image[sample["source_record"]]
            if (sample["question"] != row["question"] or sample["official_answers"] != row["answers"]
                    or sample["gold"] != [a["answer"] for a in row["answers"]]
                    or sample["annotated_answerable"] != row["answerable"]
                    or sample["answer_type"] != row.get("answer_type") or len(row["answers"]) != 10):
                raise ValueError("Registered sample differs from official annotation: " + sid)
            split = "dev" if int(hashlib.sha256(row["image"].encode()).hexdigest()[:8], 16) % 5 == 0 else "eval"
            if split != sample["split"] or sample["cluster"] != row["image"]:
                raise ValueError("Registered VizWiz identity/split differs")
            resolve_image_path(sample["image_path"], ROOT)
            image_count += 1
            answerability[(sample["split"], row["answerable"])] += 1
            dump_line(stream, {"dataset": "vizwiz", "sample_id": sid, "split": sample["split"],
                              "question": sample["question"], "annotated_answerable": row["answerable"],
                              "answer_type": row.get("answer_type"), "official_answers": row["answers"],
                              "official_source_path": str(path), "official_source_record": row["image"],
                              "official_record_sha256": stable_hash(row), "knowledge_deficit_GT": None,
                              "GT_status": "official_answerability_separate_from_registered_ten_attempt_summary"})
    return {"path": str(path), "sha256": file_hash(path), "samples": len(samples),
            "actual_image_paths_verified": image_count,
            "answerability_counts": [{"split": split, "annotated_answerable": answerable, "n": n}
                                     for (split, answerable), n in sorted(answerability.items())],
            "normalizer_path": str(ROOT / "src/kdm/models/official_vqa_normalizer.py"),
            "normalizer_sha256": file_hash(ROOT / "src/kdm/models/official_vqa_normalizer.py"),
            "score_function": "kdm.scoring.vqa_score: leave-one-annotator-out ten-answer official-normalized consensus",
            "registered_probe_summary": "kdm.analysis.probe_summary: human-unanswerable mean correctness remains null; positive/full-correct credit reported separately"}


def audit(run, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "remaining_keys").mkdir()
    (output / "derived_formal").mkdir()
    samples = {row["id"]: row for row in read_jsonl(ROOT / "data/current/all.jsonl") if row["dataset"] == "vizwiz"}
    if len(samples) != 4319 or Counter(row["split"] for row in samples.values()) != {"dev": 818, "eval": 3501}:
        raise ValueError("VizWiz requires all original dev/eval samples")
    evaluation = {sid: sample for sid, sample in samples.items() if sample["split"] == "eval"}
    inclusion_path = run / "assets/registered_inclusion.json"
    inclusion = json.loads(inclusion_path.read_text())["rows"]
    selected = {row["model"]: row for row in inclusion if row["dataset"] == "vizwiz" and row["selected"]}
    if set(selected) != set(MODELS):
        raise ValueError("Current registered remaining11 VizWiz inclusion differs")
    method_path = ROOT / "configs/kdm/method_plan.json"
    methods = json.loads(method_path.read_text())
    archive_manifest_path = ARCHIVE.with_name("historical_outputs_manifest.json")
    archive_files = {row["path"]: row for row in json.loads(archive_manifest_path.read_text())["files"]}
    finite_raw = json.loads((run / "assets/historical_raw_index.json").read_text())["files"]
    official = official_references(samples, output)
    labels_path, recovered_label = recover_if_needed(CENSUS_PREFIX + "labels.jsonl", run, output, archive_files)
    labels = {}
    for _, row, _ in row_stream(labels_path):
        if row["model"] not in selected or row["dataset"] != "vizwiz":
            continue
        if row["key"] in labels or row["sample_id"] not in samples:
            raise ValueError("Duplicate or unregistered final VizWiz census label")
        labels[row["key"]] = row
    if len(labels) != 11 * 4319 * 2:
        raise ValueError("Final census VizWiz labels are not complete")
    food_manifest = json.loads((run / "assets/asset_manifest.json").read_text())
    historical_probe_evidence = []
    for source in food_manifest["sources"]:
        if source["stage"] == "independent":
            if source["counts"].get("outside_food101", 0):
                raise ValueError("Historical probe has non-Food rows that require individual VizWiz validation")
            if source["rows"] != source["food_rows"]:
                raise ValueError("Historical source audit cannot exclude VizWiz")
            historical_probe_evidence.append({"model": source["model"], "path": source["path"],
                "identity": source["identity"], "source_cohort": source["cohort"],
                "actual_rows": source["rows"], "actual_food_rows": source["food_rows"], "actual_vizwiz_rows": 0,
                "trailing_newline": source["trailing_newline"], "source_validation_errors": source["validation_error_count"],
                "evidence": "previous_actual_raw_dataset/key audit; source unchanged in exclusive recovered directory",
                "source_audit_path": str(run / "assets/source_audits" / f"{source['model']}_independent_{source['source_id']}.json")})
    relevant_raw = [r for r in finite_raw if any("/" + model + "/" in r["path"] for model in MODELS)
                    and r["path"].endswith(".jsonl")]
    expected_history = {entry["source_member"] for entry in food_manifest["sources"] if entry["stage"] in {"candidate", "independent"}}
    unclassified_raw = [entry["path"] for entry in relevant_raw if entry["path"] not in expected_history and "/current/" not in entry["path"]]
    if unclassified_raw:
        raise ValueError("Finite historical model raw index contains unaudited sources: " + str(unclassified_raw))
    if list((run / "raw").glob("*/vizwiz")):
        raise ValueError("New VizWiz raw exists; its immutable completed shards must be audited before generating gaps")
    gap_rows, sources, condition_rows, recovered = [], [], [], []
    if recovered_label:
        recovered.append(recovered_label)
    total_labels = 0
    model_order = ("qwen35_9b",) + tuple(model for model in MODELS if model != "qwen35_9b")
    for model in model_order:
        census_path = ROOT / "data/responses/census" / (model + ".jsonl.gz")
        sidecar_path = census_path.with_name(model + ".identity.json")
        sidecar = json.loads(sidecar_path.read_text())
        if stable_hash(sidecar["definition"]) != sidecar["identity"]:
            raise ValueError("Canonical census sidecar identity differs")
        runtime_path = ROOT / "configs/runtime" / (model + ".json")
        runtime = json.loads(runtime_path.read_text())
        differences = source_runtime_differences(sidecar["definition"]["backend"], runtime)
        if differences:
            raise ValueError("Canonical census/current runtime differences: " + str(differences))
        formal_definition = {"schema": "kdm_vizwiz_census_to_formal_source_mapping_v1", "model": model,
            "backend": sidecar["definition"]["backend"], "source_path": str(census_path),
            "source_identity": sidecar["identity"], "source_sidecar_path": str(sidecar_path),
            "source_kind": "census", "derived_kind": "main", "dataset": "vizwiz",
            "rule": "exact guided UNKNOWN eval; only kind/key change; original identity and labels retained"}
        derived_identity = stable_hash(formal_definition)
        derived_path = output / "derived_formal" / (model + "_direct_unknown.jsonl")
        label_output = output / (model + "_census_final_labels.jsonl")
        counts = Counter()
        source_keys, formal_keys = set(), set()
        expected_census = {task_id(model, task) for task in census_tasks(samples.values())}
        with derived_path.open("x", encoding="utf-8", newline="\n") as target, label_output.open("x", encoding="utf-8", newline="\n") as label_target:
            for number, row, line_sha in row_stream(census_path):
                sample = row["sample"]
                if sample["dataset"] != "vizwiz":
                    continue
                key, sid = row["key"], sample["id"]
                if key not in expected_census or key in source_keys or sample != samples[sid]:
                    raise ValueError("Canonical VizWiz census task/sample differs")
                source_keys.add(key)
                label = labels[key]
                if (row["model"] != model or row["identity"] != sidecar["identity"]
                        or row.get("status", "ok") != "ok" or row["seed"] != stable_seed(sid, model, row["replicate"])
                        or row["config"] != asdict(DecodeConfig())
                        or row["prompt"] != task_prompt(sample["question"], row["marker"], row["guided"])
                        or label["text"] != row["text"] or label["question"] != sample["question"]
                        or label["raw_identity"] != row["identity"] or label["sample_id"] != sid
                        or label["split"] != sample["split"] or label["guided"] != row["guided"]):
                    raise ValueError("VizWiz original response or final-label binding differs")
                if type(row["terminated"]) is not bool or len(row["tokens"]) > 32:
                    raise ValueError("VizWiz native termination/token budget differs")
                if len(row["tokens"]) != len(row["selected_log_probabilities"]):
                    raise ValueError("VizWiz native token/probability lengths differ")
                if not all(math.isfinite(value) for value in row["selected_log_probabilities"]):
                    raise ValueError("VizWiz native selected probability is nonfinite")
                if label["label"] not in {"abstain", "answer_assertive", "answer_uncertain", "invalid"}:
                    raise ValueError("Final VizWiz behavior remains unresolved")
                answer_text = label.get("answer_text", "")
                span_resolved = (not answer_text if label["label"] in {"abstain", "invalid"}
                                 else isinstance(answer_text, str) and bool(answer_text) and answer_text in row["text"])
                counts["raw_rows"] += 1
                counts["guided_rows"] += row["guided"]
                counts["unguided_rows"] += not row["guided"]
                counts["token_budget_exhausted"] += not row["terminated"]
                counts["answer_span_unresolved"] += not span_resolved
                counts["guided_abstentions"] += row["guided"] and label["label"] == "abstain"
                dump_line(label_target, {**label, "current_source_path": str(census_path), "current_source_line": number,
                    "current_raw_line_sha256": line_sha, "legacy_raw_record_sha256": label.get("raw_record_sha256"),
                    "legacy_hash_encoding": "historical field preserved; source binding verified by original identity/key/question/full answer",
                    "current_answer_span_resolved": bool(span_resolved), "official_reference_file": str(output / "official_reference.jsonl")})
                total_labels += 1
                if row["guided"] and sample["split"] == "eval":
                    task = {"sample": sample, "method": "direct", "kind": "main", "marker": "UNKNOWN",
                            "reference_marker": "UNKNOWN", "guided": True, "reference_guided": True, "replicate": 0}
                    formal_key = task_id(model, task)
                    if formal_key in formal_keys:
                        raise ValueError("Duplicate derived VizWiz formal key")
                    formal_keys.add(formal_key)
                    dump_line(target, {**row, "key": formal_key, "kind": "main", "identity": derived_identity,
                        "original_key": key, "original_identity": row["identity"], "source_identity": row["identity"],
                        "source_path": str(census_path), "source_line": number, "source_row_sha256": line_sha,
                        "source_row_sha256_encoding": "exact decompressed UTF-8 line including original newline",
                        "derivation": "vizwiz_guided_unknown_eval_census_to_formal_direct",
                        "historical_final_label": label, "official_reference_file": str(output / "official_reference.jsonl")})
        if source_keys != expected_census or len(formal_keys) != 3501 or counts["guided_abstentions"] != selected[model]["confirmed_abstentions"]:
            raise ValueError("Canonical VizWiz source coverage/inclusion differs")
        write_json(derived_path.with_suffix(".identity.json"), {"identity": derived_identity, "definition": formal_definition})
        sources.append({"model": model, "dataset": "vizwiz", "stage": "census", "path": str(census_path),
            "identity_path": str(sidecar_path), "identity": sidecar["identity"], "source_cohort": sidecar["definition"]["backend"]["factory"],
            "source_member": "outputs/raw/current/" + model + "/census.jsonl", "source_container": str(ARCHIVE),
            "rows": len(source_keys), "coverage_complete": True, "physical_rows_json_valid": True, **dict(counts),
            "final_label_path": str(label_output), "final_label_rows": counts["raw_rows"],
            "derived_formal_path": str(derived_path), "derived_formal_identity": derived_identity,
            "formal_reused_rows": len(formal_keys), "runtime_differences": differences})
        task_list = list(experiment_tasks((next(iter(evaluation.values())),), methods=methods[model]["vizwiz"]))
        for task in task_list:
            condition_rows.append({"model": model, "dataset": "vizwiz", "split": "eval",
                **{field: task[field] for field in CONDITION_FIELDS[3:]}, "expected_n": 3501,
                "reused_n": 3501 if task["method"] == "direct" and task["marker"] == "UNKNOWN" else 0})
        for stage in ("independent", "formal"):
            key_path = output / "remaining_keys" / (model + "_" + stage + ".jsonl")
            expected = len(evaluation) * (10 if stage == "independent" else len(task_list))
            reused = 0 if stage == "independent" else len(formal_keys)
            remaining = 0
            with key_path.open("x", encoding="utf-8", newline="\n") as stream:
                for sample in evaluation.values():
                    tasks = (independent_task(sample, replicate) for replicate in range(10)) if stage == "independent" else experiment_tasks((sample,), methods=methods[model]["vizwiz"])
                    for task in tasks:
                        key = task_id(model, task)
                        if stage == "formal" and key in formal_keys:
                            continue
                        dump_line(stream, {"key": key, "model": model, "dataset": "vizwiz", "stage": stage,
                                           "sample_id": sample["id"], "split": sample["split"]})
                        remaining += 1
            if remaining + reused != expected:
                raise ValueError("VizWiz missing-key totals do not conserve registered tasks")
            gap_rows.append({"model": model, "dataset": "vizwiz", "stage": stage, "selected": True,
                "split": "eval", "expected_rows": expected, "reusable_rows": reused, "remaining_rows": remaining,
                "remaining_keys_file": str(key_path), "registered_methods": ";".join(methods[model]["vizwiz"]),
                "label_status": "final_census_closed; new_independent_and_methods_unavailable",
                "reference_status": "official_answerability_available; registered_ten_attempt_summary_missing; knowledge_GT_undefined"})
            print(json.dumps({"event": "vizwiz_missing_keys_written", **gap_rows[-1]}, ensure_ascii=False), flush=True)
    with (output / "asset_gaps.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(gap_rows[0])); writer.writeheader(); writer.writerows(gap_rows)
    with (output / "condition_coverage.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(condition_rows[0])); writer.writeheader(); writer.writerows(condition_rows)
    totals = {stage: {field: sum(int(row[field]) for row in gap_rows if row["stage"] == stage)
                      for field in ("expected_rows", "reusable_rows", "remaining_rows")}
              for stage in ("independent", "formal")}
    if len(condition_rows) != 800 or totals["formal"]["expected_rows"] != 2800800:
        raise ValueError("Registered VizWiz formal matrix differs")
    manifest = {"schema": "kdm_remaining11_vizwiz_asset_manifest_v1", "updated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "vizwiz", "samples": 4319, "splits": {"dev": 818, "eval": 3501},
        "registered_models": list(MODELS), "formal_registered_conditions": len(condition_rows),
        "totals": totals, "gaps": gap_rows, "sources": sources, "official_reference": official,
        "final_census_labels": {"original_path": str(labels_path), "original_member": CENSUS_PREFIX + "labels.jsonl",
                                "source_container": str(ARCHIVE), "selected_rows": total_labels, "unresolved_behavior_rows": 0},
        "historical_probe_dataset_evidence": historical_probe_evidence, "new_exact_recoveries": recovered,
        "finite_archive_raw_index": str(run / "assets/historical_raw_index.json"),
        "finite_relevant_model_raw_entries": len(relevant_raw), "unclassified_model_raw_entries": unclassified_raw,
        "input_sha256": {"all_manifest": file_hash(ROOT / "data/current/all.jsonl"),
            "method_plan": file_hash(method_path), "registered_inclusion": file_hash(inclusion_path)},
        "scientific_unresolved": ["registered VizWiz ten-attempt output/credit summaries remain missing",
            "original protocol does not define a binary knowledge-deficit GT from official answerability and repeated credits",
            "no Food-101 closed-rank rule is applied to VizWiz"],
        "GPU_execution": False, "new_API_calls": 0, "complete_generation": False}
    write_json(output / "asset_manifest.json", manifest)
    write_json(output / "validation.json", {"registered_samples_verified": len(samples), "actual_image_paths_verified": 4319,
        "official_annotation_records_verified": 4319, "original_census_task_keys_verified": sum(s["rows"] for s in sources),
        "original_identity_and_final_label_bindings_verified": total_labels,
        "derived_formal_original_key_identity_preserved": sum(s["formal_reused_rows"] for s in sources),
        "registered_missing_keys_written": sum(row["remaining_rows"] for row in gap_rows),
        "registered_formal_condition_count": 800, "source_validation_errors": 0,
        "script_sha256": file_hash(Path(__file__))})
    print(json.dumps({"output": str(output), "totals": totals, "final_label_rows": total_labels,
                      "new_exact_recoveries": len(recovered)}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = (ROOT / args.run).resolve()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(run / "assets/vizwiz"):
        raise ValueError("VizWiz inventory output must remain in its independent assets directory")
    audit(run, output)


if __name__ == "__main__":
    main()
