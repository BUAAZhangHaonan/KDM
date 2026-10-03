#!/usr/bin/env python3
"""Measure bounded diagnostics and real paths supplied by a sealed source list."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workflows/paper_core"))
from PIL import Image
from kdm.execution import resolve_image_path
from kdm.io import Ledger, atomic_json, file_hash, read_jsonl, stable_hash, stable_seed, within
from kdm.pipeline import make_backend
from workflows.paper_core.attribution import measure_attribution, measure_case
from workflows.paper_core.run_attribution_details import _BudgetBackend, longest_common_prefix
from workflows.supplemental.remaining4.native import MODELS, source_inputs
from workflows.supplemental.remaining11 import execution


def source_plan(model, path, phase):
    rows = [row for row in read_jsonl(path) if row["model"] == model]
    limit = 48 if phase == "diagnostic" else 3
    if not rows or len(rows) > limit or len({row["sample"]["id"] for row in rows}) != len(rows):
        raise ValueError("A nonempty bounded unique actual source list is required")
    samples = {row["id"]: row for row in read_jsonl(ROOT / "data/current/all.jsonl")}
    for row in rows:
        sample = row["sample"]
        if (sample != samples.get(sample["id"]) or sample["dataset"] != "food101"
                or sample["split"] != "eval" or row["seed"] != stable_seed(sample["id"], model, 0)
                or row["marker"] != "UNKNOWN" or not row["reference_version_bound"]):
            raise ValueError("Detail sample, seed, marker or final reference binding differs")
        branches = row["branches"]
        if not {"native_vcd", "ip_vcd"} <= set(branches):
            raise ValueError("Real native and IP responses are required")
        for branch in branches.values():
            if (not branch["tokens"] or branch["abstain"] not in (True, False)
                    or branch["correct_canonical"] not in (0, 1)
                    or not branch["raw_line_sha256"] or not branch["source_identity"]):
                raise ValueError("An actual saved token path and resolved behavior are required")
        if phase == "diagnostic":
            pairs = row["pairs"]
            if not 0 <= len(pairs) <= 2:
                raise ValueError("Each diagnostic input needs at most two explicit path pairs")
            for pair in pairs:
                left, right = (branches[pair[name]] for name in ("path_a", "path_b"))
                a, b = left["tokens"], right["tokens"]
                position = longest_common_prefix(a, b)
                if (position != pair["position"] or position >= min(len(a), len(b))
                        or a[position] == b[position]):
                    raise ValueError("Diagnostic position is not the first real token divergence")
                if left["abstain"] != right["abstain"] and left["abstain"] is not True:
                    raise ValueError("The registered candidate a must follow the abstaining path")
                if (not left["abstain"] and not right["abstain"]
                        and left["correct_canonical"] != right["correct_canonical"]
                        and left["correct_canonical"] != 1):
                    raise ValueError("The registered candidate a must follow the correct path in a C/E pair")
    return rows


def first_position_cache(model, verification_path):
    verification = json.loads(verification_path.read_text())
    if verification.get("status") != "passed":
        raise ValueError("First-position reuse requires an actually passed verification")
    cached = {}
    for source in verification["sources"]:
        path = within(ROOT, source["path"])
        if (path.name != "representative.events.jsonl"
                or not path.parent.name.startswith(model + "_")):
            continue
        if file_hash(path) != source["sha256"]:
            raise ValueError("A previously verified first-position source differs")
        identity_path = within(ROOT, source["identity_path"])
        if file_hash(identity_path) != source["identity_sha256"]:
            raise ValueError("A previously verified first-position identity differs")
        records = list(read_jsonl(path))
        if len(records) != source["rows"]:
            raise ValueError("A first-position source has a different actual row count")
        for line, row in enumerate(records, 1):
            measured = row["measured"]
            if (row["model"] != model or row["sample_id"] in cached
                    or measured["prefixes"] != [[]] or measured["marker"] != "UNKNOWN"
                    or measured["noise_step"] != 500 or measured["alpha"] != 1
                    or measured["beta"] != 0.1 or len(measured["events"]) != 1
                    or measured["events"][0]["position"] != 0):
                raise ValueError("An actual first-position cache is not a unique registered measurement")
            cached[row["sample_id"]] = (row, {
                "path": source["path"], "sha256": source["sha256"], "line": line,
                "verification_path": str(verification_path.relative_to(ROOT)),
                "verification_sha256": file_hash(verification_path)})
    return cached


def diagnostic_groups(row):
    groups = {(): []}
    for pair in row["pairs"]:
        left = row["branches"][pair["path_a"]]
        prefix = tuple(left["tokens"][:pair["position"]])
        groups.setdefault(prefix, []).append(pair)
    return groups



def cached_first_measurement(previous, row, token_pairs=()):
    """Exact CPU reuse of cached four-view candidate rows; no new first measurement."""
    import copy, math
    record, provenance=previous
    value=record["measured"]
    if (record["source"]["sample"]!=row["sample"] or record["seed"]!=row["seed"]
            or value["question"]!=row["sample"]["question"]
            or record["source"]["native_vcd"]["tokens"]!=row["branches"]["native_vcd"]["tokens"]):
        raise ValueError("An already measured first position differs; do not remeasure404")
    result=copy.deepcopy(value);event=result["events"][0];candidates={x["token"]:x for x in event["candidate_rows"]}
    existing={tuple(x)for x in result["candidate_pairs"]}
    derived=[]
    for a,b in token_pairs:
        if (a,b)in existing:continue
        if a not in candidates or b not in candidates or not event["math_pairs"]:
            raise ValueError("Cached first position lacks a requested candidate; preserve gap, no remeasurement")
        template=copy.deepcopy(event["math_pairs"][0]);pa,pb=candidates[a],candidates[b];lp_a=pa["logp"];lp_b=pb["logp"]
        assert all(math.isfinite(x)for x in list(lp_a.values())+list(lp_b.values()))
        dm={k:lp_a[k]-lp_b[k]for k in ['g','h','c','r']};alpha=template['alpha'];native=(1+alpha)*dm['c']-alpha*dm['r'];guided=(1+alpha)*dm['g']-alpha*dm['h'];ip=dm['g']+alpha*(dm['c']-dm['r']);interaction=alpha*((dm['g']-dm['c'])-(dm['h']-dm['r']))
        template.update(candidate_a=a,candidate_b=b,logp_a=dict(lp_a),logp_b=dict(lp_b),base_pair_margins=dm,clean_guide_margin_change=dm['g']-dm['c'],reference_guide_margin_change=dm['h']-dm['r'],unguided_visual_margin_change=alpha*(dm['c']-dm['r']),native_pair_margin=native,guided_pair_margin=guided,ip_pair_margin=ip,interaction_pair_margin=interaction,ip_minus_guided_margin=ip-guided,pair_in_guided_support=pa['Sg_member']and pb['Sg_member'],pair_in_native_support=pa['Sc_member']and pb['Sc_member'],guided_support_a=pa['Sg_member'],guided_support_b=pb['Sg_member'],native_support_a=pa['Sc_member'],native_support_b=pb['Sc_member'],decomposition_closure=guided-native-(dm['g']-dm['c'])-interaction,ip_difference_closure=ip-guided+interaction)
        for item in template['interpolation']:item['pair_margin']=ip+item['interaction_weight']*interaction
        # Prefix-wide support/argmax/normalizers retain their actual cached full-vocabulary values.
        template['CPU_derived_from_actual_cached_candidate_rows']=True;event['math_pairs'].append(template);result['candidate_pairs'].append([a,b]);derived.append([a,b])
    result.update(cache_reused_from=provenance,CPU_derived_candidate_pairs=derived,diagnostic_source_meta={"input":row})
    return result


def case_with_cached_first(backend,image,row,name,previous):
    """Replay saved nonzero prefixes and reuse the already measured first prefix."""
    import copy
    branch=row['branches'][name];tokens=branch['tokens'];first=cached_first_measurement(previous,row);event=copy.deepcopy(first['events'][0])
    if tokens[0]not in {x['token']for x in event['candidate_rows']}:
        raise ValueError('Cached case first token is absent; do not repeat404')
    event.update(observed_token=tokens[0],cache_reused_from=previous[1],source_meta={"input":row,"path_name":name})
    if len(tokens)>1:
        result=measure_attribution(backend,image,row['sample']['question'],[tokens[:i]for i in range(1,len(tokens))],row['seed'],observed_tokens=tokens,observed_text=branch['text'],source_meta={"input":row,"path_name":name})
        result['events'].insert(0,event);result['prefixes'].insert(0,[])
    else:result=first;result['events']=[event];result.update(observed_tokens=tokens,observed_text=branch['text'])
    result.update(response_tokens=tokens,response_text=branch['text'],terminated=bool(tokens[-1]in backend.eos),truncated=bool(tokens[-1]not in backend.eos and len(tokens)>=32),first_position_cache_reused_from=previous[1])
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--phase", choices=("diagnostic", "cases"), required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prior-budget", required=True)
    parser.add_argument("--first-position-verification")
    parser.add_argument("--physical-gpus", required=True)
    parser.add_argument("--host-registry", required=True)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()
    inputs, prior_path, out = (within(ROOT, value) for value in
                              (args.inputs, args.prior_budget, args.out))
    out.relative_to(ROOT / "outputs/supplemental/remaining4")
    rows = source_plan(args.model, inputs, args.phase)
    cache = {}
    if not args.first_position_verification:
        raise ValueError("Finite detail planning requires verified first-position sources; do not repeat404")
    cache = first_position_cache(args.model, within(ROOT, args.first_position_verification))
    for row in rows:
        previous=cache.get(row["sample"]["id"])
        if previous:
            pairs=diagnostic_groups(row)[()] if args.phase=="diagnostic" else []
            token_pairs=list(dict.fromkeys((row["branches"][x["path_a"]]["tokens"][0],row["branches"][x["path_b"]]["tokens"][0])for x in pairs))
            value=cached_first_measurement(previous,row,token_pairs)
            if args.phase=="cases"and any(row["branches"][name]["tokens"][0]not in {x["token"]for x in value["events"][0]["candidate_rows"]}for name in ("native_vcd","ip_vcd")):
                raise ValueError("Cached case first token is absent; preserve gap without repeat404")
    cards = args.physical_gpus.split(",")
    prior = json.loads(prior_path.read_text())
    if (prior["model"] != args.model or prior["gpu_count"] != len(cards)
            or not 0 <= prior["used_gpu_seconds"] < 2880):
        raise ValueError("Original per-model attribution budget is exhausted or mismatched")
    identity = {"schema": "kdm_remaining4_attribution_details_v1", "model": args.model,
                "phase": args.phase, "inputs_sha256": file_hash(inputs),
                "prior_budget_sha256": file_hash(prior_path), "claim_id": args.claim_id,
                "owner": args.owner, "cap_gpu_seconds": 2880, "actual_input_count": len(rows),
                "new_guided_vcd_generation": False, "runner_sha256": file_hash(Path(__file__))}
    identity.update(first_positions_planned=len(rows) if args.phase == "diagnostic" else 0,
        additional_positions_planned=sum(len(diagnostic_groups(row)) - 1 for row in rows)
            if args.phase == "diagnostic" else 0,
        verified_first_position_cache_inputs=len(cache),
        diagnostic_inputs_without_divergence=sum(not row["pairs"] for row in rows)
            if args.phase == "diagnostic" else 0)
    if args.check_plan:
        print(json.dumps({**identity, "gpu_initialized": False, "measured_inputs": 0}))
        return
    out.mkdir(parents=True, exist_ok=False)
    execution.REGISTRY = str(within(ROOT, args.host_registry).relative_to(ROOT))
    os.environ["KDM_SUPPLEMENTAL_DATASET"] = "food101"
    _, frozen_spec, _, provenance = source_inputs(args.model, "vcd")
    admitted = execution.validate_supplemental_runtime(
        ROOT, frozen_spec, args.model, cards, "attribution", args.claim_id, args.owner)
    identity.update(provenance=provenance, admission=admitted["execution"],
                    measurement_sha256=file_hash(ROOT / "workflows/paper_core/attribution.py"))
    atomic_json(out / "admission.json", identity)
    ledger = Ledger(out / (args.phase + ".events.jsonl"), identity)
    started = time.perf_counter()
    reused_first, new_first, additional = 0, 0, 0
    try:
        backend = _BudgetBackend(make_backend(admitted["runtime_spec"], "cuda:0"), started,
                                 2880 - prior["used_gpu_seconds"], cards)
        for row in rows:
            sample, measured = row["sample"], []
            with Image.open(resolve_image_path(sample["image_path"], ROOT)) as source:
                image = source.convert("RGB")
                if args.phase == "diagnostic":
                    for prefix, pairs in diagnostic_groups(row).items():
                        native = row["branches"]["native_vcd"]
                        token_pairs = list(dict.fromkeys(
                            (row["branches"][pair["path_a"]]["tokens"][len(prefix)],
                             row["branches"][pair["path_b"]]["tokens"][len(prefix)])
                            for pair in pairs))
                        previous = cache.get(sample["id"]) if not prefix else None
                        if previous:
                            value=cached_first_measurement(previous,row,token_pairs)
                            measured.append({**value,"diagnostic_source_meta":{"input":row,"candidate_orientations":pairs}})
                            reused_first+=1
                            continue
                        observed = row["branches"][pairs[0]["path_a"]]["tokens"] if pairs else native["tokens"]
                        measured.append(measure_attribution(
                            backend, image, sample["question"], [list(prefix)], row["seed"],
                            diagnostic_positions=[len(prefix)], observed_tokens=observed,
                            candidate_pairs=token_pairs,
                            candidate_token_ids=[branch["tokens"][0] for branch in row["branches"].values()],
                            source_meta={"input": row, "candidate_orientations": pairs,
                                "position_kind": "first" if not prefix else "first_actual_divergence"}))
                        if prefix:
                            additional += 1
                        else:
                            new_first += 1
                else:
                    for name in ("native_vcd", "ip_vcd"):
                        branch = row["branches"][name]
                        previous=cache.get(sample["id"])
                        if previous:measured.append(case_with_cached_first(backend,image,row,name,previous));reused_first+=1
                        else:measured.append(measure_case(backend,image,sample["question"],branch["tokens"],row["seed"],observed_text=branch["text"],source_meta={"input":row,"path_name":name}))
            ledger.add(stable_hash([args.model, sample["id"], args.phase]),
                       {"model": args.model, "sample_id": sample["id"], "phase": args.phase,
                        "source": row, "measured": measured, "status": "ok"})
    finally:
        elapsed = (time.perf_counter() - started) * len(cards)
        atomic_json(out / "budget.json", {"model": args.model, "gpu_count": len(cards),
                    "used_gpu_seconds": prior["used_gpu_seconds"] + elapsed,
                    "last_call_gpu_seconds": elapsed, "cap_gpu_seconds": 2880,
                    "updated_utc": datetime.now(timezone.utc).isoformat()})
    atomic_json(out / "complete.json", {"model": args.model, "phase": args.phase,
                "expected_inputs": len(rows), "completed_inputs": len(ledger.keys),
                "passed": len(rows) == len(ledger.keys), "total_all_model_cases_claimed": False,
                "first_positions_reused": reused_first, "first_positions_new": new_first,
                "additional_unique_positions_measured": additional,
                "diagnostic_inputs_without_divergence": identity["diagnostic_inputs_without_divergence"]})


if __name__ == "__main__":
    main()
