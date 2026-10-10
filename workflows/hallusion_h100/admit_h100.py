"""Minimal new-H100 gate on four original conditions of the first cal question.

This is an execution tool, not an automatic startup action. check-plan does not
import any backend. execute records actual token/branch agreement; a mismatch
blocks reuse and returns requires_new_baselines while admitting explicit fresh
generation if model, runtime, source, image and hardware checks passed.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import socket
import sys
from pathlib import Path

from wording_protocol import *
from generate_wording import (validate_runtime_files, validate_worker_locks, load_shared,
                              validate_prediction, logical_runtime)


def validate_capture(registration, path, anchors_path, expected_sha):
    capture = read_json(path)
    require(capture.get("status") == "pass", "Frozen anchor capture did not pass")
    require(capture.get("raw36_sha256") == expected_sha == file_hash(anchors_path), "Frozen capture/package SHA differs")
    require(capture.get("original_manifest_sha256") == file_hash(registration["asset_path"]("manifest"))
            and capture.get("frozen_split_sha256") == file_hash(registration["asset_path"]("split")),
            "Frozen capture manifest/split differs")
    require(capture.get("sample_id") == phase_sample_ids(registration, "calibration")[0]
            and capture.get("selection") == "original_manifest_order_first_calibration_sample"
            and capture.get("models") == 9 and capture.get("raw_rows") == 36
            and capture.get("all_four_methods_same_question_per_model") is True
            and capture.get("all_nine_models_same_question") is True,
            "Frozen capture is not the registered first-cal four-method gate")
    require(set(capture.get("selection_does_not_use", [])) ==
            {"answers", "quality", "abstention", "u", "tokens", "finish_reason", "runtime"},
            "Gate capture selection scope differs")
    return capture


def anchors_for_model(registration, path, expected_sha, model):
    require(file_hash(path) == expected_sha, "Frozen gate capture SHA differs")
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for text in stream:
            wrapper = json.loads(text)
            row = wrapper["raw_record"]
            if row["model"] == model:
                # source_row_sha256 binds the literal original JSONL line bytes
                # without CR/LF. The extraction does not retain those bytes, so
                # reserializing this object cannot reproduce that source hash.
                # The validated whole-capture SHA binds the source SHA/line and
                # object together; this separate digest names the parsed object.
                require(all(isinstance(wrapper.get(field), str) and len(wrapper[field]) == 64
                            and all(char in "0123456789abcdef" for char in wrapper[field])
                            for field in ("source_sha256", "source_row_sha256")), "Missing literal original source SHA binding")
                require(type(wrapper.get("source_line")) is int and wrapper["source_line"] >= 1,
                        "Missing original source line")
                wrapper["canonical_raw_record_sha256"] = stable_hash(row)
                rows.append(wrapper)
    first_id = phase_sample_ids(registration, "calibration")[0]
    require(len(rows) == 4 and {r["raw_record"]["method"] for r in rows} == {"direct", "vcd", *IP_METHODS}, "Exactly four original method anchors required")
    require(all(r["raw_record"]["sample"]["id"] == first_id for r in rows), "Gate must use first source-order cal sample, without outcome selection")
    original = read_json(registration["asset_path"]("original_conditions"))
    for wrapper in rows:
        row = wrapper["raw_record"]
        condition = next(c for c in original if c["model"] == model and c["method"] == row["method"])
        require(row["condition_identity"] == stable_hash(condition), "Gate anchor original condition identity differs")
        require(row["sample"] == registration["sample_by_id"][first_id], "Gate sample/image metadata differs")
        require(all(row[field] == condition[field] for field in ("marker", "reference_marker", "guided", "reference_guided", "replicate", "kind")), "Gate anchor fields differ")
        cfg = copy.deepcopy(condition["config"])
        if row["method"] == "instruction_m3id":
            require(isinstance(row.get("offset_prompt_tokens"), list), "Old M3ID offset missing")
            cfg["m3id_offset"] = len(row["offset_prompt_tokens"])
        require(row["config"] == cfg and row["seed"] == stable_seed(first_id, model, 0), "Gate numerical parameters/seed differ")
        require(row["status"] == "ok" and 1 <= len(row["tokens"]) <= 128, "Gate original output invalid")
        require(row["prompt"] == guided_prompt(row["sample"]["prompt"], row["marker"], row["guided"]), "Gate original prompt differs")
    order = {method: i for i, method in enumerate(("direct", "vcd") + IP_METHODS)}
    return sorted(rows, key=lambda r: order[r["raw_record"]["method"]])


def compare_gate(old, new):
    differences = [field for field in ("tokens", "text", "terminated", "truncated", "finish_reason",
                                      "prompt", "reference_prompt", "neutral_prompt", "config", "seed", "offset_prompt_tokens")
                   if old.get(field) != new.get(field)]
    branch_scope = []
    for name, branch in old.get("branch_inputs", {}).items():
        branch_scope.append(name)
        actual = new.get("branch_inputs", {}).get(name, {})
        for field in ("prompt", "input_ids_sha256", "input_token_count", "processed_image_tensors", "noise_tensor_sha256"):
            if field in branch and actual.get(field) != branch[field]:
                differences.append("branch_inputs." + name + "." + field)
    if old["method"] in ("vcd", "instruction_vcd"):
        if not old.get("branch_inputs", {}).get("reference", {}).get("noise_tensor_sha256"):
            differences.append("missing_original_processed_noise_evidence")
    return {"exact": not differences, "differences": differences,
            "old_branch_evidence_scope": branch_scope,
            "input_evidence_missing": not bool(branch_scope),
            "old_response_sha256": response_hash(old), "new_response_sha256": response_hash(new),
            "old_n_tokens": len(old["tokens"]), "new_n_tokens": len(new["tokens"])}


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol", type=Path, default=package / "registration/protocol.json")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("/code/knowledge-deficit-mitigation"))
    parser.add_argument("--runtime-spec", type=Path, required=True)
    parser.add_argument("--cards", default="0")
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--anchors-sha256", required=True)
    parser.add_argument("--capture-validation", type=Path, required=True,
                        help="Data agent's actual passed frozen-source capture receipt")
    parser.add_argument("--reuse-sources", type=Path,
                        help="JSON array of copied complete old chunks/receipt/owner bindings; optional")
    parser.add_argument("--output", type=Path, required=True, help="New model-specific admission directory")
    args = parser.parse_args()
    registration = load_registration(args.protocol)
    anchors = anchors_for_model(registration, args.anchors, args.anchors_sha256, args.model)
    capture = validate_capture(registration, args.capture_validation, args.anchors, args.anchors_sha256)
    if args.check_plan:
        print(json.dumps({"status": "CPU_gate_plan_checked", "model": args.model,
                          "sample_id": phase_sample_ids(registration, "calibration")[0],
                          "four_methods": [r["raw_record"]["method"] for r in anchors],
                          "capture_sha256": args.anchors_sha256,
                          "canonical_record_sha256": [r["canonical_raw_record_sha256"] for r in anchors],
                          "source_row_sha_scope": "original literal JSONL line without CR/LF; preserved, not reconstructed",
                          "output_token_cap": 128, "new_semantic_decisions": 0, "GPU_initialized": False}, indent=2))
        return 0
    project, out = args.project_root.resolve(), args.output.resolve()
    require(out.is_relative_to(project) and not out.exists(), "New exclusive admission directory required")
    actual, runtime = validate_runtime_files(registration, project, args.model, args.runtime_spec)
    physical = validate_worker_locks(project, args.cards.split(","), actual)
    engine = load_shared(project)
    out.mkdir(parents=True)
    write_once(out / "started.json", {"started_utc": now(), "model": args.model, "hostname": socket.gethostname(),
               "registration_identity": registration["identity"], "runtime": runtime, "physical_GPUs": physical,
               "anchors_sha256": args.anchors_sha256, "capture_validation_sha256": file_hash(args.capture_validation),
               "batch_size": 1, "output_token_cap": 128, "argv": sys.argv})
    checks = []
    original = read_json(registration["asset_path"]("original_conditions"))
    try:
        backend = engine.make_backend(actual, "cuda:0")
        eos = set(backend.eos)
        require(bool(eos), "Actual EOS IDs missing")
        for index, wrapper in enumerate(anchors):
            old = wrapper["raw_record"]
            condition = next(c for c in original if c["model"] == args.model and c["method"] == old["method"])
            task = {key: old[key] for key in ("sample", "method", "marker", "reference_marker", "guided", "reference_guided", "kind", "replicate", "condition_identity")}
            generated = engine.json_safe(engine.run_one(backend, task, engine.DecodeConfig(**condition["config"]), args.model))
            generated["status"] = "ok"
            validate_prediction(generated, condition["config"], eos, fresh=True)
            # Existing source EOS IDs may be unavailable in an extracted row;
            # compare its recorded termination against actual EOS without inventing
            # an original EOS list. That evidence limit is retained explicitly.
            check = compare_gate(old, generated)
            try:
                validate_prediction(old, condition["config"], eos, fresh=False)
            except ValueError as exc:
                # The original output is never reused when its recorded EOS
                # behavior disagrees with the actual registered backend. This
                # evidence difference does not prohibit explicit fresh runs.
                check["exact"] = False
                check["differences"].append("old_prediction_vs_actual_EOS_contract")
                check["old_prediction_validation_difference"] = str(exc)
            check.update(method=old["method"], marker=old["marker"], sample_id=old["sample"]["id"],
                         original_condition_identity=old["condition_identity"], source_root_relative_path=wrapper["source_root_relative_path"],
                         source_sha256=wrapper["source_sha256"], source_line=wrapper["source_line"],
                         source_row_sha256=wrapper["source_row_sha256"],
                         canonical_raw_record_sha256=wrapper["canonical_raw_record_sha256"],
                         old_eos_id_list_available="eos_token_ids" in wrapper)
            if "eos_token_ids" in wrapper and set(wrapper["eos_token_ids"]) != eos:
                check["exact"] = False
                check["differences"].append("original_vs_actual_eos_token_ids")
            write_once(out / f"gate_{index:02d}_{old['method']}.json", {"source": wrapper, "generated": generated, "check": check})
            checks.append(check)
        all_exact = all(check["exact"] for check in checks)
        gate = {"schema": "wording_new_h100_four_condition_gate_v1", "model": args.model,
                "registration_identity": registration["identity"], "expected_conditions": 4, "completed_conditions": 4,
                "all_exact": all_exact, "checks": checks, "finished_utc": now(),
                "scope": "one fixed first-cal question, four prior registered conditions; not all951 replayed",
                "new_semantic_decisions": 0}
        write_once(out / "four_condition_gate.json", gate)
        sources = read_json(args.reuse_sources) if args.reuse_sources else []
        require(isinstance(sources, list), "Reuse sources must be an explicit JSON array")
        admission = {"schema": "kdm_hallusion_wording_host_admission_v1",
                     "status": "passed" if all_exact else "requires_new_baselines",
                     "generation_admitted": True, "registration_identity": registration["identity"],
                     "model": args.model, "hostname": socket.gethostname(), "project_root": str(project),
                     "runtime": runtime, "physical_GPUs": physical, "eos_token_ids": sorted(eos),
                     "output_token_cap": 128, "reference_original_sha256": REFERENCE_SHA,
                     "new_host_gate": {"path": str(out / "four_condition_gate.json"), "sha256": file_hash(out / "four_condition_gate.json")},
                     "trajectory_reuse": {"status": "passed" if all_exact else "requires_new_baselines", "sources": sources},
                     "fresh_mode": "explicit fresh generation with same registered prompt/config/batch/dtype is allowed",
                     "automatic_fallback": False, "scope": gate["scope"]}
        write_once(out / "host_admission.json", admission)
        print(json.dumps({"status": admission["status"], "generation_admitted": True, "reuse_admitted": all_exact,
                          "model": args.model, "host_admission": str(out / "host_admission.json"),
                          "differences": [c for c in checks if not c["exact"]]}, indent=2))
        return 0
    except BaseException as exc:
        write_once(out / "failed.json", {"status": "failed", "error": type(exc).__name__ + ": " + str(exc),
                                         "model": args.model, "automatic_retry": False, "automatic_parameter_change": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
