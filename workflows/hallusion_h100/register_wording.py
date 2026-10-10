"""Freeze the authorized four-marker Hallusion317/634 independent series."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path

from wording_protocol import (MODELS, MARKERS, IP_METHODS, GUIDED_SUFFIX, REFERENCE_SHA, POLICY_SHA,
                              require, file_hash, stable_hash, read_json, read_csv, write_csv)


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=package.parent / "kdm_hallusion_du_analysis_20261009")
    parser.add_argument("--output", type=Path, default=package / "registration")
    parser.add_argument("--refresh-draft", action="store_true",
                        help="Refresh implementation binding only before any run, score, selection or result exists.")
    args = parser.parse_args()
    analysis, out = args.analysis_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol_file = out / "protocol.json"
    if protocol_file.exists():
        require(args.refresh_draft, "Registration already exists; no implicit replacement")
        for name in ("runs", "scoring", "selection", "results"):
            directory = out.parent / name
            require(not directory.exists() or not any(p.is_file() for p in directory.rglob("*")),
                    f"Cannot refresh after real {name} assets exist")
    protocol_source = analysis / "protocol"
    reference = analysis / "references/main_quality/derived_complete_root51_20261009/reference.csv"
    policy = analysis / "references/REFERENCE_POLICY_APPROVED_v2.json"
    require(file_hash(reference) == REFERENCE_SHA and file_hash(policy) == POLICY_SHA, "Approved reference bytes differ")
    sources = {
        "original_conditions": (protocol_source / "sources/hallusion/conditions.json", "source_conditions.json"),
        "original_protocol": (protocol_source / "sources/hallusion/protocol.json", "source_protocol.json"),
        "manifest": (protocol_source / "sources/dataset/manifest_hallusionbench.jsonl", "manifest.jsonl"),
        "split": (protocol_source / "group_split_proposal.csv", "split.csv"),
        "reference_policy": (policy, "reference_policy.json"),
        "split_description": (protocol_source / "split_feasibility.json", "split_feasibility.json"),
    }
    for name, (source, dest) in sources.items():
        shutil.copyfile(source, out / dest)
    original_protocol = read_json(out / "source_protocol.json")
    require(original_protocol["dataset_entry"]["manifest_sha256"] == file_hash(out / "manifest.jsonl"), "Original manifest source differs")
    require(original_protocol["conditions_sha256"] == file_hash(out / "source_conditions.json"), "Original condition source differs")
    require(original_protocol["guided_suffix"] == GUIDED_SUFFIX and original_protocol["plain_prompt_suffix"] == "", "Native base prompt differs")
    original_conditions = read_json(out / "source_conditions.json")
    conditions = []
    for model in MODELS:
        for method in IP_METHODS + ("direct", "vcd"):
            old = [c for c in original_conditions if c["model"] == model and c["method"] == method]
            require(len(old) == 1, f"Missing/duplicate source condition: {model}/{method}")
            old = old[0]
            source_hash = stable_hash(old)
            if method in IP_METHODS:
                role, markers = "IP_candidate", MARKERS
            elif method == "direct":
                role, markers = "guided_direct", MARKERS
            else:
                role, markers = "native_control", ("NONE",)
            for marker in markers:
                condition = {key: copy.deepcopy(old[key]) for key in (
                    "model", "method", "marker", "reference_marker", "guided", "reference_guided", "kind", "replicate", "config", "checkpoint")}
                condition.update(marker=marker, reference_marker=marker,
                                 guided=role != "native_control", reference_guided=False,
                                 kind="guided_direct_control" if role == "guided_direct" else old["kind"], role=role,
                                 original_condition_identity=source_hash, original_registered_marker=old["marker"])
                condition["condition_identity"] = stable_hash(condition)
                conditions.append(condition)
            if method == "direct":
                condition = {key: copy.deepcopy(old[key]) for key in (
                    "model", "method", "marker", "reference_marker", "guided", "reference_guided", "kind", "replicate", "config", "checkpoint")}
                condition.update(role="native_control", original_condition_identity=source_hash,
                                 original_registered_marker=old["marker"])
                condition["condition_identity"] = stable_hash(condition)
                conditions.append(condition)
    require(len(conditions) == 126, "Condition cardinality changed")
    (out / "conditions.json").write_text(json.dumps(conditions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Direct field copying, not a semantic reference reconstruction. Discard all
    # independent-trial text/QA fields before storing the portable u table.
    u_rows, quality_reference = [], []
    with reference.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            require(row["should_abstain"] in ("True", "False"), "Unresolved supplied u")
            u_rows.append({"model": row["model"], "sample_id": row["sample_id"],
                           "u": int(row["should_abstain"] == "True"), "source_status": row["status"],
                           "u_decision_status": row["u_decision_status"], "reference_original_sha256": REFERENCE_SHA})
            quality_reference.append({key: row[key] for key in (
                "model", "sample_id", "gt_answer", "gt_answer_details", "reference_sufficient", "status", "reference_origin")}
                | {"reference_original_sha256": REFERENCE_SHA})
    require(len(u_rows) == 8559 and len({(r["model"], r["sample_id"]) for r in u_rows}) == 8559, "Unique8559 supplied reference rows required")
    for filename, rows in (("frozen_u.csv", u_rows), ("frozen_quality_reference.csv", quality_reference)):
        dest = out / filename
        if dest.exists():
            require(args.refresh_draft, "Reference extraction already exists")
            dest.unlink()
        write_csv(dest, rows)
    source_hashes = read_json(out / "source_hashes.json")
    require(source_hashes["workflows/hallusion_blind/generate128.py"] == file_hash(protocol_source / "sources/hallusion/active_generate128.py"), "run_one source snapshot changed")
    assets = {name: {"path": dest, "sha256": file_hash(out / dest)} for name, (_, dest) in sources.items()}
    for name, filename in (("conditions", "conditions.json"), ("u_reference", "frozen_u.csv"),
                           ("quality_reference", "frozen_quality_reference.csv"), ("shared_sources", "source_hashes.json")):
        assets[name] = {"path": filename, "sha256": file_hash(out / filename)}
    for model in MODELS:
        name = f"source_runtime/{model}.json"
        spec = read_json(out / name)
        require(spec["key"] == model and not spec.get("api"), "Registered native model required")
        require(spec["dtype"] == ("float16" if model == "onevision" else "bfloat16"), "Frozen dtype differs")
        assets["runtime_" + model] = {"path": name, "sha256": file_hash(out / name)}
    implementation = ["wording_protocol.py", "register_wording.py", "generate_wording.py", "du_metrics.py", "admit_h100.py"]
    definition = {
        "schema": "kdm_hallusion_wording317_634_v1", "status": "registered_pending_host_admission",
        "models": list(MODELS), "markers": list(MARKERS), "IP_methods": list(IP_METHODS),
        "output_token_cap": 128, "termination_policy": "EOS_or_128_generated_tokens",
        "selection_rule": "maximize integer C+TP on calibration317; tie uses fixed marker order",
        "selection_scheduling": "independent complete14x317 model seals, then its634; final complete9 union is required",
        "calibration_n": 317, "calibration_groups": 56, "holdout_n": 634, "holdout_groups": 102,
        "full_n": 951, "full_groups": 158,
        "holdout_scope": "internal_exploratory_all951_previously_observed; external_unseen_confirmation_required",
        "base_prompt_policy": "exact_original_manifest_prompt_plus_only_original_registered_guided_suffix",
        "guided_suffix": GUIDED_SUFFIX, "batch_size": 1,
        "shared_engine": "workflows/hallusion_blind/generate128.py:run_one",
        "original_full951_entry_modified": False, "new_parameter_grid": False,
        "reference_original_sha256": REFERENCE_SHA, "reference_policy_sha256": POLICY_SHA,
        "u_extraction": "8559 exact supplied model/sample_id/should_abstain fields; no semantic decisions",
        "score_policy": "complete actual response; quality correct1/incorrect0/unclear2; abstain separately supplied",
        "DU": "(C+TP)/N; C excludes all abstentions; TP=abstain and independent u=1",
        "native_controls": ["direct", "vcd"],
        "reuse_policy": "disabled until passed new-host admission AND exact-source/token-compatibility proof",
        "bootstrap": {"repetitions": 2000, "seed": 20261009, "unit": "registered component_id",
                      "strata": "category/subcategory", "estimator": "paired pooled sample DU difference"},
        "record_budget": {"calibration_IP": 22824, "calibration_guided_direct": 11412,
                          "calibration_native_controls": 5706, "holdout_IP": 11412,
                          "holdout_guided_direct_upper": 11412, "holdout_native_controls": 11412,
                          "all_no_reuse_upper": 74178, "new_no_native_generation_with_exact_prior_IP_reuse_upper": 51354},
        "assets": assets, "implementation_sha256": {name: file_hash(package / name) for name in implementation},
    }
    definition["registration_identity"] = stable_hash(definition)
    protocol_file.write_text(json.dumps(definition, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "registered_pending_host_admission", "registration_identity": definition["registration_identity"],
                      "conditions": len(conditions), "u_labels": len(u_rows), "GPU_initialized": False}, indent=2))


if __name__ == "__main__":
    main()
