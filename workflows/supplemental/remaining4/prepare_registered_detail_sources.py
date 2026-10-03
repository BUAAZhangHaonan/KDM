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
NAMES = ("supported_abstention_recovered", "supported_abstention_preserved", "wrong_answer_corrected",
         "correct_answer_corrupted", "correct_answer_preserved", "unnecessary_abstention_introduced")
PREDICATES = ("baseA and R and nextC", "baseA and R and nextA", "baseE and nextC", "baseC and nextE",
              "baseC and nextC", "not baseA and nextA and not R")


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
    parser.add_argument("--model", choices=("internvl35_8b","onevision","phi35","qwen3vl"), required=True)
    parser.add_argument("--accepted-main", required=True)
    parser.add_argument("--accepted-main-sha256", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--reference-complete", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strata-registration", required=True)
    parser.add_argument("--prior-budget", required=True)
    args = parser.parse_args()
    output, registration_path, budget_path = [within(ROOT, p) for p in (args.output, args.strata_registration, args.prior_budget)]
    with registration_path.open(newline="", encoding="utf-8-sig") as stream:
        registration = list(csv.DictReader(stream))
    if not registration or not {row["stratum"] for row in registration} <= set(NAMES):
        raise ValueError("The original registration contains an unknown named diagnostic stratum")
    import pandas as pd
    main_path=within(ROOT,args.accepted_main)
    if file_hash(main_path)!=args.accepted_main_sha256:
        raise ValueError("The accepted main score source SHA differs")
    ref_path=within(ROOT,args.reference_root)/"reference_G.jsonl"
    ref_sha=file_hash(ref_path)
    ref_receipt=json.loads((ref_path.parent/"receipt.json").read_text())
    budget=json.loads(budget_path.read_text())
    if (not ref_receipt["passed"] or ref_receipt["reference_complete"]!=args.reference_complete
            or ref_receipt["reference_pending"] or ref_receipt["outputs"][ref_path.name]!=ref_sha
            or budget["model"]!=args.model or budget["gpu_count"]!=(2 if args.model=="internvl35_8b" else 1)
            or not 0<=budget["used_gpu_seconds"]<2880):
        raise ValueError("The final reference or original remaining per-model budget differs")
    frame=pd.read_parquet(main_path)
    frame=frame[(frame.model==args.model)&(frame.dataset=="food101")&(frame.split=="eval")&(frame.replicate==0)]
    samples={x["id"]:x for _line,x,_sha in rows(ROOT/"data/current/all.jsonl")if x["dataset"]=="food101"and x["split"]=="eval"}
    pairs={sid:{}for sid in samples};references={};main_records={}
    configurations=(("native_vcd","vcd","native_unguided","NONE",False),("IP","instruction_vcd","instruction_preserving","UNKNOWN",True),("unguided_direct","direct","unguided","NONE",False))
    for label,method,kind,marker,guided in configurations:
        sub=frame[(frame.method==method)&(frame.kind==kind)&(frame.marker==marker)&(frame.reference_marker==marker)&(frame.guided==guided)&(~frame.reference_guided)]
        if len(sub)!=2424 or sub.sample_id.nunique()!=2424 or set(sub.sample_id)!=set(samples):
            raise ValueError("An accepted exact original UNKNOWN/native/unguided condition is not complete2424")
        for accepted in sub.to_dict("records"):
            sid=accepted["sample_id"]
            if not accepted["reference_complete"]or not accepted["accepted_complete_decision"]:
                raise ValueError("A selected population contains an unaccepted score/reference")
            record={"canonical_name_in_primary_score":int(accepted["correct_canonical"]),"literal_extracted_name_score":int(accepted["correct_literal"]),"abstain":bool(accepted["abstain"])}
            ref={"reference_G":bool(accepted["uniform_reference"]),"reference_complete":True,"gold_rank":None,"correct_count":None}
            if sid in references and references[sid]["reference_G"]!=ref["reference_G"]:
                raise ValueError("Accepted original three paths disagree on their final reference")
            references[sid]=ref;pairs[sid][label]={"score":record};main_records[sid,label]=accepted
    if len(samples)!=2424:
        raise ValueError("The fixed Food eval population differs")
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
    score_requests=defaultdict(dict)
    for sid in selected:
        for label in ("native_vcd","IP","unguided_direct"):
            meta=main_records[sid,label];score_requests[within(ROOT,meta["source_score_path"])][int(meta["source_score_line"])]=(sid,label,meta)
    selected_score_reads=[]
    for score_path,wanted in score_requests.items():
        if len({meta["source_score_sha256"]for _sid,_label,meta in wanted.values()})!=1 or file_hash(score_path)!=next(iter(wanted.values()))[2]["source_score_sha256"]:
            raise ValueError("An accepted score container differs from its exact main binding")
        opener=gzip.open if score_path.suffix==".gz"else open;found=set()
        with opener(score_path,"rb")as stream:
            for line,data in enumerate(islice(stream,max(wanted)),1):
                if line not in wanted:continue
                sid,label,meta=wanted[line];original=json.loads(data)
                if hashlib.sha256(data).hexdigest()!=meta["source_score_line_sha256"]or original["key"]!=meta["source_task_key"]or original["sample_id"]!=sid or original["model"]!=args.model:
                    raise ValueError("A selected accepted score line/key/model differs")
                record=dict(original)
                record["accepted_main_original_score_fields"]={k:original.get(k)for k in ["canonical_name_in_primary_score","literal_extracted_name_score","abstain","reference_G"]}
                record.update(canonical_name_in_primary_score=int(meta["correct_canonical"]),literal_extracted_name_score=int(meta["correct_literal"]),abstain=bool(meta["abstain"]),reference_G=bool(meta["uniform_reference"]),reference_complete=True)
                record["accepted_main_binding"]={k:meta[k]for k in ["source_score_path","source_score_sha256","source_score_line","source_score_line_sha256","source_task_key","condition_id","qa_key","applied_correction_json"]}
                record["accepted_main_binding"].update(path=args.accepted_main,sha256=args.accepted_main_sha256)
                pairs[sid][label]["score"]=record
                if label=="IP":references[sid].update(gold_rank=original.get("gold_rank"),correct_count=original.get("correct_count"))
                found.add(line)
        if found!=set(wanted):raise ValueError("A selected accepted score line is absent")
        selected_score_reads.append({"path":str(score_path.relative_to(ROOT)),"selected_lines":sorted(found),"JSON_objects_parsed":len(found),"unselected_JSON_objects_parsed":0})
    branches_by_sample, requests, branch_locations = {}, defaultdict(dict), {}
    cached_count = 0
    for sid in selected:
        for name, label in (("native_vcd", "native_vcd"), ("ip_vcd", "IP"), ("unguided_direct", "unguided_direct")):
            wrapped = pairs[sid][label]
            record, path = wrapped["score"], within(ROOT, wrapped["score"]["source_path"])
            requests[path][record["source_line"]] = wrapped
            branch_locations[sid, name] = path, record["source_line"]
    extracted, source_reads = {}, []
    for path, requested in requests.items():
        exemplar = requested[min(requested)]["score"]
        identity_path = within(ROOT, exemplar["source_identity_path"]) if exemplar.get("source_identity_path") else path.with_suffix(".identity.json")
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
                        or raw["seed"] != stable_seed(record["sample_id"], args.model, 0) or raw["text"] != record["answer"]
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
                no_divergence.append({"model": args.model, "sample_id": sid, **missing})
        ref = references[sid]
        memberships = [name for name in NAMES if sid in populations[name][:8]]
        row = {"model": args.model, "sample": samples[sid], "seed": stable_seed(sid, args.model, 0), "marker": "UNKNOWN",
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
    result = {"schema": "kdm_remaining4_registered_named_detail_sources_v3_accepted_main_overlay", "passed": True, "model": args.model,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "full_eval_population": 2424, "nine_grid_counts": dict(transitions),
        "registered_six_strata": coverage, "selected_unique_samples": len(selected), "diagnostic_inputs": len(diagnostics),
        "complete_case_inputs": len(cases), "actual_token_pairs": sum(len(row["pairs"]) for row in diagnostics),
        "no_actual_token_divergence_pairs": len(no_divergence), "cached_verified_raw_objects_reused": cached_count,
        "new_exact_raw_objects_parsed": len(extracted), "new_source_reads": source_reads,
        "original_strata_registration_path": str(registration_path.relative_to(ROOT)), "original_strata_registration_sha256": file_hash(registration_path),
        "original_core_comparison": "guided_vcd_to_instruction_preserving_vcd", "supplemental_comparison": "native_vcd_to_instruction_preserving_vcd",
        "reference_path": str(ref_path.relative_to(ROOT)), "reference_sha256": ref_sha,
        "accepted_main_score_rows_path":args.accepted_main,"accepted_main_score_rows_sha256":args.accepted_main_sha256,"selected_score_source_reads":selected_score_reads, "cached_real_raw_source_sha256": None,
        "prior_budget_path": str(budget_path.relative_to(ROOT)), "prior_budget_sha256": file_hash(budget_path),
        "prior_used_gpu_seconds": budget["used_gpu_seconds"], "old_outputs_overwritten": False,
        "termination_tokens_synthesized": False, "GPU_initialized": False, "mechanism_measurements_completed": False,
        "adapter_sha256": file_hash(Path(__file__)), "outputs": {p.name: file_hash(p) for p in output.iterdir()}}
    save_json(output / "source_receipt.json", result)
    print(json.dumps({key: result[key] for key in ("passed", "selected_unique_samples", "diagnostic_inputs", "complete_case_inputs", "actual_token_pairs", "cached_verified_raw_objects_reused", "new_exact_raw_objects_parsed")}), flush=True)


if __name__ == "__main__":
    main()
