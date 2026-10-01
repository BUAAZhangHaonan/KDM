#!/usr/bin/env python3
"""Prepare the original named six diagnostic strata for native VCD versus IP."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from kdm.io import file_hash, stable_hash, stable_seed, within
from workflows.paper_core.run_attribution_details import longest_common_prefix
from workflows.supplemental.remaining11.score import rows
from workflows.supplemental.remaining4.score_native import save_json, save_rows

BASE = "outputs/supplemental/remaining4"
PAIR = BASE + "/ip_vcd_source_join_20261001_1900"
REFERENCE = BASE + "/reference_full_20261001_1600/checkpoints/v2_reviewed_role_reference"
NAMES = ("supported_abstention_recovered", "supported_abstention_preserved", "wrong_answer_corrected",
         "correct_answer_corrupted", "correct_answer_preserved", "unnecessary_abstention_introduced")
PREDICATES = ("baseA and R and nextC", "baseA and R and nextA", "baseE and nextC", "baseC and nextE",
              "baseC and nextC", "not baseA and nextA and not R")
OLD = BASE + "/detail_sources_20261001_qwen3vl"


def state(record):
    correct, abstain = record["canonical_name_in_primary_score"], record["abstain"]
    if correct not in (0, 1) or type(abstain) is not bool or correct == 1 and abstain:
        raise ValueError("A real score has unresolved or incompatible C/E/A state")
    return "A" if abstain else "C" if correct else "E"


def stratum_of(base, nxt, reference):
    conditions = (base == "A" and reference and nxt == "C", base == "A" and reference and nxt == "A",
                  base == "E" and nxt == "C", base == "C" and nxt == "E", base == "C" and nxt == "C",
                  base != "A" and nxt == "A" and not reference)
    return [name for name, accepted in zip(NAMES, conditions) if accepted]


def oriented_pair(branches, left, right):
    a, b = branches[left], branches[right]
    if a["abstain"] != b["abstain"]:
        path_a, path_b = (left, right) if a["abstain"] else (right, left)
        meaning = "abstention_path_minus_answer_path"
    elif a["correct_canonical"] != b["correct_canonical"]:
        path_a, path_b = (left, right) if a["correct_canonical"] else (right, left)
        meaning = "correct_path_minus_wrong_path"
    else:
        path_a, path_b = left, right
        meaning = "source_defined_nonsemantic_token_contrast"
    tokens_a, tokens_b = branches[path_a]["tokens"], branches[path_b]["tokens"]
    position = longest_common_prefix(tokens_a, tokens_b)
    if position >= min(len(tokens_a), len(tokens_b)):
        return None, {"path_a": path_a, "path_b": path_b, "position": position,
            "tokens_identical": tokens_a == tokens_b, "complete_prefix_without_saved_competing_token": tokens_a != tokens_b,
            "reason": "no_actual_next_token_divergence; no termination token synthesized"}
    if tokens_a[:position] != tokens_b[:position] or tokens_a[position] == tokens_b[position]:
        raise ValueError("An actual diagnostic contrast lacks a common prefix and distinct saved next tokens")
    return {"path_a": path_a, "path_b": path_b, "position": position,
        "candidate_a": tokens_a[position], "candidate_b": tokens_b[position], "actual_shared_prefix": tokens_a[:position],
        "orientation": path_a + "_minus_" + path_b, "signed_orientation": meaning,
        "semantic_candidate_orientation": meaning != "source_defined_nonsemantic_token_contrast"}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strata-registration", required=True)
    parser.add_argument("--prior-budget", required=True)
    args = parser.parse_args()
    output, registration_path, budget_path = [within(ROOT, p) for p in (args.output, args.strata_registration, args.prior_budget)]
    with registration_path.open(newline="", encoding="utf-8-sig") as stream:
        registration = list(csv.DictReader(stream))
    if not registration or not {row["stratum"] for row in registration} <= set(NAMES):
        raise ValueError("The original registration contains an unknown named diagnostic stratum")
    pair_path = within(ROOT, PAIR) / "native_Direct_IP_sample_pairs.jsonl.gz"
    pair_receipt = json.loads((pair_path.parent / "receipt.json").read_text())
    ref_path = within(ROOT, REFERENCE) / "reference_G.jsonl"
    ref_sha = file_hash(ref_path)
    ref_receipt = json.loads((ref_path.parent / "receipt.json").read_text())
    old_path = within(ROOT, OLD) / "selected_actual_sources.jsonl"
    old_receipt = json.loads((old_path.parent / "source_receipt.json").read_text())
    budget = json.loads(budget_path.read_text())
    if (not pair_receipt["passed"] or file_hash(pair_path) != pair_receipt["outputs"][pair_path.name]
            or not ref_receipt["passed"] or ref_receipt["reference_complete"] != 19392 or ref_receipt["reference_pending"]
            or ref_sha != ref_receipt["outputs"][ref_path.name]
            or not old_receipt["passed"] or file_hash(old_path) != old_receipt["outputs"][old_path.name]
            or budget["model"] != "qwen3vl" or budget["gpu_count"] != 1 or not 0 <= budget["used_gpu_seconds"] < 2880):
        raise ValueError("Actual original source, reference and cumulative budget proofs are required")
    samples = {r["id"]: r for _line, r, _sha in rows(ROOT / "data/current/all.jsonl") if r["dataset"] == "food101" and r["split"] == "eval"}
    references = {r["sample_id"]: r for _line, r, _sha in rows(ref_path) if r["model"] == "qwen3vl" and r["split"] == "eval"}
    pairs = {r["sample_id"]: r for _line, r, _sha in rows(pair_path) if r["model"] == "qwen3vl"}
    cache = {r["sample"]["id"]: r for _line, r, _sha in rows(old_path)}
    if set(samples) != set(pairs) or set(samples) != set(references) or len(samples) != 2424:
        raise ValueError("The actual three-path comparison does not cover the full registered eval population")
    populations, transitions = defaultdict(list), Counter()
    for sid in sorted(samples):
        base, nxt = state(pairs[sid]["native_vcd"]["score"]), state(pairs[sid]["IP"]["score"])
        reference = references[sid]["reference_G"]
        if type(reference) is not bool or not references[sid]["reference_complete"]:
            raise ValueError("A named diagnostic stratum cannot use an unresolved reference")
        transitions[base + "_to_" + nxt] += 1
        for name in stratum_of(base, nxt, reference):
            populations[name].append(sid)
    selected = sorted({sid for name in NAMES for sid in populations[name][:8]})
    branches_by_sample, requests, branch_locations = {}, defaultdict(dict), {}
    cached_count = 0
    for sid in selected:
        if sid in cache:
            if cache[sid]["sample"] != samples[sid] or cache[sid]["reference_sha256"] != ref_sha:
                raise ValueError("An already verified finite raw source cache has a different sample/reference version")
            branches_by_sample[sid] = cache[sid]["branches"]
            cached_count += 3
            continue
        for name, label in (("native_vcd", "native_vcd"), ("ip_vcd", "IP"), ("unguided_direct", "unguided_direct")):
            wrapped = pairs[sid][label]
            record, path = wrapped["score"], within(ROOT, wrapped["score"]["source_path"])
            requests[path][record["source_line"]] = wrapped
            branch_locations[sid, name] = path, record["source_line"]
    extracted, source_reads = {}, []
    for path, requested in requests.items():
        exemplar = requested[min(requested)]["score"]
        identity_path = within(ROOT, exemplar["source_identity_path"]) if exemplar["kind"] == "unguided" else path.with_suffix(".identity.json")
        identity, identity_sha = json.loads(identity_path.read_text()), file_hash(identity_path)
        if identity["identity"] != stable_hash(identity["definition"]):
            raise ValueError("An original raw source identity definition differs")
        visited, found = 0, set()
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            for line, actual in enumerate(islice(stream, max(requested)), 1):
                visited = line
                if line not in requested:
                    continue
                wrapped, raw = requested[line], json.loads(actual)
                record, sha = wrapped["score"], hashlib.sha256(actual).hexdigest()
                if (sha != record["raw_line_sha256"] or raw["key"] != record["key"] or raw["identity"] != record["source_identity"]
                        or raw["identity"] != identity["identity"] or raw["sample"] != samples[record["sample_id"]]
                        or raw["seed"] != stable_seed(record["sample_id"], "qwen3vl", 0) or raw["text"] != record["answer"]
                        or raw["terminated"] != record["terminated"] or not raw["tokens"]
                        or "config" in record and record["config"] != raw["config"]
                        or "config_sha256" in record and stable_hash(raw["config"]) != record["config_sha256"]):
                    raise ValueError("A selected original raw response differs from its actual score/seed/config/source proof")
                extracted[path, line] = {"tokens": raw["tokens"], "text": raw["text"], "terminated": raw["terminated"],
                    "abstain": record["abstain"], "correct_canonical": record["canonical_name_in_primary_score"],
                    "correct_literal": record["literal_extracted_name_score"], "config": raw["config"],
                    "prompt": raw["prompt"], "reference_prompt": raw.get("reference_prompt"), "source_key": raw["key"],
                    "source_path": str(path.relative_to(ROOT)), "source_line": line, "source_identity": raw["identity"],
                    "raw_line_sha256": sha, "source_identity_path": str(identity_path.relative_to(ROOT)),
                    "source_identity_sha256": identity_sha, "original_score_source": wrapped}
                found.add(line)
        if found != set(requested):
            raise ValueError("An exact selected source line is absent or truncated")
        source_reads.append({"path": str(path.relative_to(ROOT)), "selected_source_lines": sorted(found),
            "selected_JSON_objects_parsed": len(found), "last_line_traversed": visited,
            "unselected_JSON_objects_parsed": 0, "full_raw_hash_performed": False})
    diagnostics, pool, no_divergence = [], [], []
    for sid in selected:
        branches = branches_by_sample.get(sid)
        if branches is None:
            branches = {name: extracted[branch_locations[sid, name]] for name in ("native_vcd", "ip_vcd", "unguided_direct")}
        contrast = []
        for left, right in (("native_vcd", "ip_vcd"), ("unguided_direct", "ip_vcd")):
            pair, missing = oriented_pair(branches, left, right)
            if pair is not None:
                contrast.append(pair)
            else:
                no_divergence.append({"model": "qwen3vl", "sample_id": sid, **missing})
        ref = references[sid]
        memberships = [name for name in NAMES if sid in populations[name][:8]]
        row = {"model": "qwen3vl", "sample": samples[sid], "seed": stable_seed(sid, "qwen3vl", 0), "marker": "UNKNOWN",
            "reference_version_bound": True, "reference_path": str(ref_path.relative_to(ROOT)), "reference_sha256": ref_sha,
            "reference_G": ref["reference_G"], "gold_rank": ref["gold_rank"], "correct_count": ref["correct_count"],
            "registered_strata": memberships, "comparison": "native_vcd_to_instruction_preserving_vcd",
            "comparison_scope": "supplemental native baseline panel; original core guided baseline registration unchanged",
            "branches": branches, "pairs": contrast}
        pool.append(row)
        if contrast:
            diagnostics.append(row)
    cases, absent, used = [], [], set()
    criteria = (("actual_recovery", "wrong_answer_corrected"), ("new_unnecessary_abstention", "unnecessary_abstention_introduced"),
                ("actual_answer_switch", "correct_answer_corrupted"))
    for case_type, stratum in criteria:
        eligible = [row for row in pool if stratum in row["registered_strata"] and row["sample"]["id"] not in used]
        if eligible:
            row = eligible[0]
            cases.append({**row, "case_type": case_type})
            used.add(row["sample"]["id"])
        else:
            absent.append({"case_type": case_type, "reason": "no actual distinct source in its named registered stratum selection"})
    coverage = [{"stratum": name, "predicate": predicate, "full_population_n": len(populations[name]),
        "uniform_reference_positive": sum(references[sid]["reference_G"] for sid in populations[name]),
        "selected_n": len(populations[name][:8]), "selected_sample_ids": populations[name][:8],
        "selection_rule": "ascending sample_id, first at most eight; empty strata remain empty"} for name, predicate in zip(NAMES, PREDICATES)]
    if len(selected) > 48 or len(diagnostics) > 48 or len(cases) > 3:
        raise ValueError("The actual registered finite source selection exceeds its bound")
    output.mkdir(parents=True, exist_ok=False)
    for filename, data in (("diagnostic_inputs.jsonl", diagnostics), ("case_inputs.jsonl", cases),
            ("registered_six_strata_coverage.jsonl", coverage), ("selected_actual_sources.jsonl", pool),
            ("no_token_divergence.jsonl", no_divergence), ("missing_case_types.jsonl", absent)):
        save_rows(output / filename, data)
    result = {"schema": "kdm_remaining4_registered_named_detail_sources_v2", "passed": True, "model": "qwen3vl",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "full_eval_population": 2424, "nine_grid_counts": dict(transitions),
        "registered_six_strata": coverage, "selected_unique_samples": len(selected), "diagnostic_inputs": len(diagnostics),
        "complete_case_inputs": len(cases), "actual_token_pairs": sum(len(row["pairs"]) for row in diagnostics),
        "no_actual_token_divergence_pairs": len(no_divergence), "cached_verified_raw_objects_reused": cached_count,
        "new_exact_raw_objects_parsed": len(extracted), "new_source_reads": source_reads,
        "original_strata_registration_path": str(registration_path.relative_to(ROOT)), "original_strata_registration_sha256": file_hash(registration_path),
        "original_core_comparison": "guided_vcd_to_instruction_preserving_vcd", "supplemental_comparison": "native_vcd_to_instruction_preserving_vcd",
        "reference_path": str(ref_path.relative_to(ROOT)), "reference_sha256": ref_sha,
        "source_pair_sha256": file_hash(pair_path), "cached_real_raw_source_sha256": file_hash(old_path),
        "prior_budget_path": str(budget_path.relative_to(ROOT)), "prior_budget_sha256": file_hash(budget_path),
        "prior_used_gpu_seconds": budget["used_gpu_seconds"], "old_outputs_overwritten": False,
        "termination_tokens_synthesized": False, "GPU_initialized": False, "mechanism_measurements_completed": False,
        "adapter_sha256": file_hash(Path(__file__)), "outputs": {p.name: file_hash(p) for p in output.iterdir()}}
    save_json(output / "source_receipt.json", result)
    print(json.dumps({key: result[key] for key in ("passed", "selected_unique_samples", "diagnostic_inputs", "complete_case_inputs", "actual_token_pairs", "cached_verified_raw_objects_reused", "new_exact_raw_objects_parsed")}), flush=True)


if __name__ == "__main__":
    main()
