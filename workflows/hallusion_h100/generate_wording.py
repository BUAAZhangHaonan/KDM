"""Independent Hallusion wording runner; delegates generation to frozen run_one.

check-plan and verify are CPU-only. execute is never called automatically.
Old trajectories require the new host's exact-token reuse admission; fresh
generation remains available under the same registered parameters when the
hardware produces a different greedy trajectory.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import importlib.util
import importlib.metadata
import json
import math
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from wording_protocol import *


def logical_runtime(spec):
    kwargs = copy.deepcopy(spec["kwargs"])
    kwargs.pop("model_path", None)
    processor = copy.deepcopy(spec.get("processor", {}))
    processor.pop("path", None)
    weights = [{k: v for k, v in weight.items() if k != "mtime_ns"} for weight in spec.get("weights", [])]
    return {"key": spec["key"], "hf_model_id": spec.get("hf_model_id"), "factory": spec["factory"],
            "kwargs_without_model_path": kwargs, "dtype": spec["dtype"], "versions": spec["versions"],
            "thinking_mode": spec.get("thinking_mode"), "gpu_count": spec["gpu_count"],
            "processor_without_path": processor, "weights_without_copy_mtime": weights,
            "model_config_sha256": spec.get("model_config_sha256"), "api": spec.get("api", False)}


def validate_project_sources(registration, project):
    project = Path(project).resolve()
    for path, digest in read_json(registration["asset_path"]("shared_sources")).items():
        require(file_hash(project / path) == digest, f"Original source changed: {path}")
    manifest = read_json(registration["asset_path"]("original_protocol"))["dataset_entry"]
    require(file_hash(project / manifest["manifest"]) == manifest["manifest_sha256"], "Original project manifest differs")


def validate_runtime_files(registration, project, model, runtime_path, check_environment=True):
    project, runtime_path = Path(project).resolve(), Path(runtime_path).resolve()
    validate_project_sources(registration, project)
    actual = read_json(runtime_path)
    original = read_json(registration["asset_path"]("runtime_" + model))
    require(logical_runtime(actual) == logical_runtime(original), "Model/backend/dtype/packages/processor/device-map identity changed")
    require(actual["key"] == model and not actual.get("api"), "Native registered model required")
    if check_environment:
        require(Path(sys.executable).resolve() == Path(actual["environment_python"]).resolve(), "Use this model's registered environment Python")
        versions = {package: importlib.metadata.version(package) for package in actual["versions"]}
        require(versions == actual["versions"], "Actual installed package versions differ")
    checkpoint = Path(actual["kwargs"]["model_path"])
    for weight in actual["weights"]:
        path = checkpoint / weight["filename"]
        require(path.is_file() and path.stat().st_size == weight["size_bytes"], "Missing/incomplete registered weight: " + str(path))
    if actual.get("model_config_sha256"):
        require(file_hash(checkpoint / "config.json") == actual["model_config_sha256"], "Model config differs")
    processor = actual.get("processor", {})
    for filename, digest in processor.get("files", {}).items():
        require(file_hash(Path(processor["path"]) / filename) == digest, "Processor/tokenizer file changed: " + filename)
    # The original native16 proof remains portable only with identical logical
    # runtime and adapter sources. The new Hallusion gate is recorded separately.
    interface = original.get("interface_verification", {})
    require(interface.get("status") == "passed", "Original native16 registration not passed")
    proof_path = path_within(project, interface["record"])
    proof = read_json(proof_path)
    require(proof.get("passed") is True and proof.get("completed") == proof.get("expected") == 16, "Original native16 evidence incomplete")
    require(logical_runtime(proof["spec"]) == logical_runtime(original), "Original native16 proof is for another logical runtime")
    interface_manifest = project / "data/current/interface16.jsonl"
    require(proof["manifest_sha256"] == file_hash(interface_manifest), "Original native16 input identity changed")
    return actual, {"original_runtime_sha256": file_hash(registration["asset_path"]("runtime_" + model)),
                    "actual_runtime_path": str(runtime_path), "actual_runtime_sha256": file_hash(runtime_path),
                    "logical_runtime_sha256": stable_hash(logical_runtime(actual)),
                    "portable_native16_path": str(proof_path), "portable_native16_sha256": file_hash(proof_path)}


def validate_worker_locks(project, cards, spec):
    require(os.name == "posix", "GPU execution requires the Linux H100 worker")
    require(len(cards) == spec["gpu_count"] and len(cards) == len(set(cards)), "GPU allocation cardinality differs")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ",".join(cards), "Visible GPU allocation differs")
    for index, card in enumerate(sorted(cards, key=int)):
        expected = Path(project).resolve() / "outputs/locks" / f"gpu_{card}.lock"
        require(Path(os.readlink(f"/proc/self/fd/{20 + index}")).resolve() == expected, "Missing inherited per-GPU worker lock")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True)
    inventory = {fields[0].strip(): {"uuid": fields[1].strip(), "name": fields[2].strip()}
                 for line in query.splitlines() if len(fields := line.split(",")) == 3}
    require(all(card in inventory and "H100" in inventory[card]["name"] for card in cards), "Allocated cards are not the authorized H100 host")
    return {card: inventory[card] for card in cards}


def load_shared(project):
    project = Path(project).resolve()
    sys.path[:0] = [str(project / "src"), str(project)]
    path = project / "workflows/hallusion_blind/generate128.py"
    spec = importlib.util.spec_from_file_location("kdm_frozen_hallusion128", path)
    require(spec is not None and spec.loader is not None, "Cannot load original run_one")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.ROOT.resolve() == project, "Shared runner project root differs")
    require(module.guided_prompt("exact plain", "UNKNOWN", True) == guided_prompt("exact plain", "UNKNOWN", True), "Original suffix differs")
    return module


def load_admission(registration, project, model, path, permit_reuse=False, check_environment=True):
    path = Path(path).resolve()
    admission = read_json(path)
    require(admission.get("schema") == "kdm_hallusion_wording_host_admission_v1", "Wrong host admission schema")
    require(admission.get("generation_admitted") is True
            and admission.get("status") in ("passed", "requires_new_baselines"), "New host generation is not admitted")
    require(admission["registration_identity"] == registration["identity"] and admission["model"] == model, "Admission registration/model differs")
    require(admission["hostname"] == socket.gethostname()
            and Path(admission["project_root"]).resolve() == Path(project).resolve(), "Admission belongs to another host/project")
    require(admission["output_token_cap"] == 128 and admission["reference_original_sha256"] == REFERENCE_SHA, "Admission budget/reference differs")
    spec, runtime = validate_runtime_files(registration, project, model, admission["runtime"]["actual_runtime_path"], check_environment)
    require(runtime == admission["runtime"], "Admission model/native-interface files changed")
    gate = admission["new_host_gate"]
    gate_path = Path(gate["path"])
    require(file_hash(gate_path) == gate["sha256"], "New host gate evidence changed")
    proof = read_json(gate_path)
    require(proof["registration_identity"] == registration["identity"] and proof["model"] == model
            and proof["expected_conditions"] == proof["completed_conditions"] == 4,
            "New host gate does not cover Direct/VCD and the two prior IP operating points")
    if permit_reuse:
        require(admission["trajectory_reuse"]["status"] == "passed" and proof["all_exact"] is True,
                "Reuse admission failed: use explicit fresh mode after reporting requires_new_baselines")
    admission["admission_file_sha256"] = file_hash(path)
    return spec, admission


def build_plan(registration, phase, model, selection=None, condition_ids=None, sample_ids=None, shard=0, n_shards=1):
    require(model in MODELS and n_shards >= 1 and 0 <= shard < n_shards, "Invalid model/shard")
    conditions = phase_conditions(registration, phase, selection, (model,))
    if condition_ids is not None:
        require(len(condition_ids) == len(set(condition_ids)) and set(condition_ids) <= {c["condition_identity"] for c in conditions}, "Foreign/duplicate condition assignment")
        conditions = [c for c in conditions if c["condition_identity"] in set(condition_ids)]
    ids = phase_sample_ids(registration, phase)
    if sample_ids is not None:
        require(len(sample_ids) == len(set(sample_ids)) and set(sample_ids) <= set(ids), "Foreign/duplicate phase sample assignment")
        ids = [sid for sid in ids if sid in set(sample_ids)]
    ids = [sid for sid in ids if int(stable_hash(sid)[:8], 16) % n_shards == shard]
    tasks = {}
    first, rest = [], []
    for condition in conditions:
        keys = []
        for sid in ids:
            key = task_key(registration, phase, condition["condition_identity"], sid)
            task = {field: condition[field] for field in ("method", "marker", "reference_marker", "guided", "reference_guided", "kind", "replicate")}
            task.update(condition_identity=condition["condition_identity"], sample=registration["sample_by_id"][sid])
            tasks[key] = task
            keys.append(key)
        first.extend(keys[:8])
        rest.extend(keys[8:])
    selection_identity = selection["selection_identity"] if selection else None
    if selection and selection.get("schema") == "kdm_hallusion_wording_selection_merge_v1":
        selection_identity = next(item["selection_identity"] for item in selection["model_selections"]
                                  if item["model"] == model)
    return {"registration": registration, "phase": phase, "model": model, "conditions": conditions,
            "tasks": tasks, "selected": first + rest,
            "selection_identity": selection_identity,
            "full_phase_n": 317 if phase == "calibration" else 634}


def validate_prediction(row, cfg, eos, fresh=False):
    require(row.get("status") == "ok", "Failed result cannot be scored")
    tokens = row.get("tokens")
    require(isinstance(tokens, list) and 1 <= len(tokens) <= 128
            and all(type(token) is int and token >= 0 for token in tokens), "Output tokens outside cap128")
    require(isinstance(row.get("text"), str), "Complete actual response text required")
    ended = tokens[-1] in eos
    require(type(row.get("terminated")) is bool and row["terminated"] == ended
            and type(row.get("truncated")) is bool and row["truncated"] == (not ended)
            and row["finish_reason"] == ("eos" if ended else "length")
            and not any(token in eos for token in tokens[:-1])
            and (ended or len(tokens) == 128), "EOS/length contract differs")
    expected = copy.deepcopy(cfg)
    if cfg["method"] == "instruction_m3id":
        offset = row.get("offset_prompt_tokens")
        require(isinstance(offset, list) and all(type(t) is int for t in offset), "Missing native M3ID plain-prompt offset")
        expected["m3id_offset"] = len(offset)
    require(row["config"] == expected, "Original numerical config differs")
    if fresh:
        trace, selected = row.get("trace"), row.get("selected_log_probabilities")
        require(isinstance(trace, list) and [r["token"] for r in trace] == tokens, "Fresh token trace differs")
        require(isinstance(selected, list) and len(selected) == len(tokens)
                and all(math.isfinite(value) for value in selected), "Fresh selected-token probabilities incomplete")


def validate_raw(row, plan, key, eos):
    task = plan["tasks"][key]
    require(row["key"] == key and row["registration_identity"] == plan["registration"]["identity"]
            and row["phase"] == plan["phase"] and row["model"] == plan["model"]
            and row["selection_identity"] == plan["selection_identity"]
            and all(row.get(field) == value for field, value in task.items()), "Raw task/phase/selection binding differs")
    condition = plan["registration"]["condition_by_id"][task["condition_identity"]]
    require(row["prompt"] == guided_prompt(task["sample"]["prompt"], task["marker"], task["guided"]), "Base prompt/suffix changed")
    require(row["seed"] == stable_seed(task["sample"]["id"], plan["model"], 0), "Original seed changed")
    fresh = row["generation_source"] == "new_host_frozen_run_one"
    require(fresh or row["generation_source"] == "reused_host_admitted_exact128", "Unknown generation source")
    validate_prediction(row, condition["config"], eos, fresh)
    if fresh:
        branches = row["branch_inputs"]
        expected = {"main": row["prompt"]}
        if task["method"] in ("vcd",) + IP_METHODS:
            expected["reference"] = task["sample"]["prompt"]
        if task["method"] in IP_METHODS:
            expected["neutral"] = task["sample"]["prompt"]
        require(set(branches) == set(expected) and all(branches[k]["prompt"] == v for k, v in expected.items()), "Actual branch prompts differ")
        if task["method"] in ("vcd", "instruction_vcd"):
            require(bool(branches["reference"].get("noise_tensor_sha256")), "Processed-noise evidence missing")
    else:
        require(row["reuse_binding"]["new_host_reuse_admitted"] is True, "Old output reused without admission")
    require(row["response_sha256"] == response_hash(row), "Complete response hash changed")


def old_source_rows(project, source):
    raw = path_within(project, source["raw_path"])
    receipt_path = path_within(project, source["receipt_path"])
    owner_path = path_within(project, source["owner_path"])
    require(file_hash(raw) == source["raw_sha256"] and file_hash(receipt_path) == source["receipt_sha256"]
            and file_hash(owner_path) == source["owner_sha256"], "Original chunk/owner/receipt file changed")
    receipt, owner = read_json(receipt_path), read_json(owner_path)
    rows = list(read_jsonl(raw))
    require(receipt["status"] == "complete" and receipt["raw_sha256"] == source["raw_sha256"]
            and receipt["owner_sha256"] == source["owner_sha256"] and receipt["claim_identity"] == stable_hash(owner), "Original seal/ownership differs")
    require(receipt["keys"] == [r["key"] for r in rows], "Original sealed key order differs")
    eos = set(receipt["eos_token_ids"])
    require(eos, "Original EOS IDs missing")
    for line, row in enumerate(rows, 1):
        require(row["claim_identity"] == receipt["claim_identity"] and row["identity"] == receipt["identity"], "Original row ownership differs")
        yield row, {**source, "line": line, "original_key": row["key"], "original_identity": row["identity"],
                    "original_claim_identity": row["claim_identity"], "eos_token_ids": sorted(eos)}


def bounded_capture_rows(project, source, registration, plan):
    paths = {name: path_within(project, source[name + "_path"]) for name in ("capture", "validation", "owners")}
    require(all(file_hash(path) == source[name + "_sha256"] for name, path in paths.items()),
            "Bounded original capture/proof bytes changed")
    receipt = read_json(paths["validation"])
    require(receipt.get("status") == "pass" and receipt.get("schema") == "hallusion_original_four_condition_bounded_capture_v1"
            and receipt["raw_sha256"] == source["capture_sha256"] and receipt["owners_sha256"] == source["owners_sha256"]
            and receipt["original_manifest_sha256"] == file_hash(registration["asset_path"]("manifest"))
            and receipt["original_conditions_sha256"] == file_hash(registration["asset_path"]("original_conditions"))
            and receipt["frozen_split_sha256"] == file_hash(registration["asset_path"]("split"))
            and receipt["source_rows_validated_sha_line"] is True and receipt["source_chains_status"] == "pass"
            and receipt["complete_raw_records"] is True and receipt["new_semantic_decisions"] == 0,
            "Passed bounded source-chain/capture proof required")
    require(receipt["raw_rows"] in (22824, 34236)
            and receipt["role_counts"]["core_native951_IPcal317"] == 22824
            and receipt["role_counts"].get("prior_IP_holdout634", 0) == receipt["raw_rows"] - 22824,
            "Bounded original row scope differs")
    owners = read_json(paths["owners"])
    allowed = {(task["method"], task["marker"], task["guided"], task["sample"]["id"])
               for task in plan["tasks"].values()}
    seen, count = set(), 0
    with gzip.open(paths["capture"], "rt", encoding="utf-8") as stream:
        for capture_line, text in enumerate(stream, 1):
            wrapper = json.loads(text)
            row = wrapper["raw_record"]
            count += 1
            source_key = (row["method"], row["marker"], row["guided"], row["sample"]["id"])
            # In calibration, original-marker IP holdout rows are never entered
            # into the usable index. Holdout tasks only exist after its model's
            # complete pretest calibration seal has been loaded and verified.
            if row["model"] != plan["model"] or source_key not in allowed:
                continue
            require(wrapper["canonical_raw_record_sha256"] == stable_hash(row), "Captured complete raw object changed")
            proof = owners[wrapper["source_sha256"]]
            seal, owner = proof["receipt"], proof["owner"]
            line = wrapper["source_line"]
            require(proof["source_chain_validation_status"] == "pass"
                    and seal["status"] == "complete" and seal["raw_sha256"] == wrapper["source_sha256"]
                    and seal["owner_sha256"] == proof["owner_sha256"]
                    and seal["claim_identity"] == stable_hash(owner) == row["claim_identity"]
                    and seal["identity"] == row["identity"] and type(line) is int
                    and 1 <= line <= len(seal["keys"]) and seal["keys"][line - 1] == row["key"],
                    "Original bounded row seal/owner/source line differs")
            require(isinstance(wrapper["source_row_sha256"], str) and len(wrapper["source_row_sha256"]) == 64,
                    "Original literal JSONL line SHA missing")
            eos = seal.get("eos_token_ids")
            require(isinstance(eos, list) and eos and all(type(token) is int for token in eos),
                    "Selected original source has no stored EOS IDs; requires explicit fresh generation")
            require(source_key not in seen, "Duplicate original bounded reuse row")
            seen.add(source_key)
            yield row, {**source, "capture_line": capture_line, "line": line,
                        "original_source_path": wrapper["source_path"], "original_source_sha256": wrapper["source_sha256"],
                        "original_source_row_sha256": wrapper["source_row_sha256"],
                        "canonical_raw_record_sha256": wrapper["canonical_raw_record_sha256"],
                        "original_receipt_sha256": proof["receipt_sha256"], "original_owner_sha256": proof["owner_sha256"],
                        "original_key": row["key"], "original_identity": row["identity"],
                        "original_claim_identity": row["claim_identity"], "eos_token_ids": eos}
    require(count == receipt["raw_rows"], "Bounded original capture row count differs")


def load_reuse_index(project, admission, plan, sources=None):
    require(admission["trajectory_reuse"]["status"] == "passed", "No new-host reuse admission")
    index = {}
    for source in admission["trajectory_reuse"]["sources"] if sources is None else sources:
        reader = bounded_capture_rows(project, source, plan["registration"], plan) if source.get("kind") == "bounded_capture" else old_source_rows(project, source)
        for row, binding in reader:
            if row["model"] != admission["model"] or row["method"] not in ("direct", "vcd") + IP_METHODS:
                continue
            key = (row["method"], row["marker"], row["guided"], row["sample"]["id"])
            require(key not in index, "Duplicate original reuse task")
            index[key] = (row, binding)
    return index


def exact_old_for_task(plan, task, old):
    condition = plan["registration"]["condition_by_id"][task["condition_identity"]]
    require(old["sample"] == task["sample"] and old["model"] == plan["model"]
            and all(old.get(f) == task[f] for f in ("method", "marker", "reference_marker", "guided", "reference_guided", "replicate")), "Reuse input/marker/model differs")
    require(old["prompt"] == guided_prompt(task["sample"]["prompt"], task["marker"], task["guided"])
            and old["seed"] == stable_seed(task["sample"]["id"], plan["model"], 0), "Reuse prompt/seed differs")
    require(old["config"]["max_tokens"] == 128, "Only existing exact cap128 trajectories may be reused")
    cfg = copy.deepcopy(condition["config"])
    if task["method"] == "instruction_m3id":
        require(isinstance(old.get("offset_prompt_tokens"), list), "Old M3ID offset evidence missing")
        cfg["m3id_offset"] = len(old["offset_prompt_tokens"])
    require(old["config"] == cfg, "Reuse numerical configuration differs")


def complete_chunks(folder, plan):
    completed, seen = {}, set()
    claims = folder / "claims"
    if not claims.exists():
        return completed
    for receipt_path in sorted(claims.glob("*/chunk_*.complete.json")):
        receipt = read_json(receipt_path)
        raw = path_within(folder, receipt["raw_path"])
        owner_path = path_within(folder, receipt["owner_path"])
        require(receipt["status"] == "complete" and receipt["registration_identity"] == plan["registration"]["identity"]
                and file_hash(raw) == receipt["raw_sha256"] and file_hash(owner_path) == receipt["owner_sha256"], "Completed chunk/source identity differs")
        owner = read_json(owner_path)
        require(stable_hash(owner) == receipt["claim_identity"], "Completed claim changed")
        rows = list(read_jsonl(raw))
        require(receipt["keys"] == [r["key"] for r in rows], "Completed chunk coverage/order differs")
        eos = set(receipt["eos_token_ids"])
        for line, row in enumerate(rows, 1):
            key = row["key"]
            require(key not in seen and row["claim_identity"] == receipt["claim_identity"], "Duplicate or foreign completed row")
            seen.add(key)
            # Other explicit assignments of this same model/phase can coexist.
            require(row["registration_identity"] == plan["registration"]["identity"] and row["phase"] == plan["phase"], "Foreign phase in run directory")
            if key in plan["tasks"]:
                validate_raw(row, plan, key, eos)
                completed[key] = {"row": row, "raw_path": str(raw), "raw_sha256": receipt["raw_sha256"],
                                  "raw_line": line, "receipt_path": str(receipt_path)}
    return completed


def execute(args, registration, plan):
    import fcntl
    project, series = args.project_root.resolve(), args.series_root.resolve()
    require(series.is_relative_to(project / "outputs") and "hallusion_blind128_20261006" not in series.parts, "Independent output root required")
    spec, admission = load_admission(registration, project, args.model, args.admission,
                                    permit_reuse=args.generation_mode != "fresh")
    require(args.generation_mode != "fresh" or args.reuse_sources is None, "Fresh mode cannot consume old source overrides")
    sources = read_json(args.reuse_sources) if args.reuse_sources else None
    require(sources is None or isinstance(sources, list), "Explicit reuse-sources JSON array required")
    cards = args.cards.split(",")
    resources = validate_worker_locks(project, cards, spec) if args.generation_mode != "reuse-only" else {}
    folder = series / "runs" / args.phase / args.model
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "ownership.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        completed = complete_chunks(folder, plan)
        pending = [key for key in plan["selected"] if key not in completed]
        claims = folder / "claims"
        claims.mkdir(exist_ok=True)
        for previous in claims.glob("*/owner.json"):
            owner = read_json(previous)
            active = set(owner["keys"])
            release_path = previous.parent / "released.json"
            if release_path.exists():
                release = read_json(release_path)
                require(release["claim_identity"] == stable_hash(owner) and set(release["keys"]) <= active, "Invalid explicit ownership release")
                active -= set(release["keys"])
            require(not (set(pending) & (active - set(completed))), "Pending key still belongs to another claim; no automatic retry/takeover")
        if not pending:
            return {"status": "already_complete", "selected_rows": len(plan["selected"])}
        run = claims / args.claim_id
        run.mkdir(exist_ok=False)
        owner = {"schema": "wording_claim_v1", "registration_identity": registration["identity"],
                 "selection_identity": plan["selection_identity"], "phase": args.phase, "model": args.model,
                 "claim_id": args.claim_id, "keys": pending, "started_utc": now(), "pid": os.getpid(),
                 "hostname": socket.gethostname(), "admission_sha256": admission["admission_file_sha256"],
                 "generation_mode": args.generation_mode, "physical_GPUs": resources, "batch_size": 1,
                 "reuse_sources_file": str(args.reuse_sources.resolve()) if args.reuse_sources else None,
                 "reuse_sources_file_sha256": file_hash(args.reuse_sources) if args.reuse_sources else None,
                 "runtime": admission["runtime"], "argv": sys.argv}
        write_once(run / "owner.json", owner)
    engine = backend = None
    eos = set(admission["eos_token_ids"])
    chunk, number, key = [], 0, None
    def seal():
        nonlocal chunk, number
        path = run / f"chunk_{number:05d}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for row in chunk:
                validate_raw(row, plan, row["key"], eos)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        path.chmod(0o444)
        receipt = {"status": "complete", "completed_utc": now(), "registration_identity": registration["identity"],
                   "claim_identity": stable_hash(owner), "keys": [r["key"] for r in chunk],
                   "eos_token_ids": sorted(eos), "raw_path": str(path.relative_to(folder)), "raw_sha256": file_hash(path),
                   "owner_path": str((run / "owner.json").relative_to(folder)), "owner_sha256": file_hash(run / "owner.json")}
        write_once(run / f"chunk_{number:05d}.complete.json", receipt)
        print(json.dumps({"event": "sealed_chunk", "phase": args.phase, "model": args.model, "rows": len(chunk), "chunk": number}), flush=True)
        chunk, number = [], number + 1
    try:
        reuse = load_reuse_index(project, admission, plan, sources) if args.generation_mode != "fresh" else {}
        for index, key in enumerate(pending):
            task = plan["tasks"][key]
            condition = registration["condition_by_id"][task["condition_identity"]]
            source_key = (task["method"], task["marker"], task["guided"], task["sample"]["id"])
            if source_key in reuse:
                old, binding = reuse[source_key]
                exact_old_for_task(plan, task, old)
                require(set(binding["eos_token_ids"]) == eos, "Old/new EOS IDs differ")
                result = copy.deepcopy(old)
                result.update(generation_source="reused_host_admitted_exact128", wall_s=0.0,
                              reuse_binding={**binding, "new_host_reuse_admitted": True,
                                             "admission_sha256": admission["admission_file_sha256"]})
            else:
                require(args.generation_mode != "reuse-only", "Exact admitted original trajectory missing; no automatic fresh fallback in reuse-only mode")
                if backend is None:
                    engine = load_shared(project)
                    backend = engine.make_backend(spec, "cuda:0")
                    require(set(backend.eos) == eos, "Admitted and actual EOS IDs differ")
                before = time.perf_counter()
                result = engine.run_one(backend, task, engine.DecodeConfig(**condition["config"]), args.model)
                result = engine.json_safe(result)
                result.update(generation_source="new_host_frozen_run_one", wall_s=time.perf_counter() - before)
            # Only the independent series wrapper changes. Full model output,
            # branch inputs, config, seed and complete text remain untouched.
            row = {**result, **task, "key": key, "model": args.model, "phase": args.phase,
                   "registration_identity": registration["identity"], "selection_identity": plan["selection_identity"],
                   "claim_identity": stable_hash(owner), "admission_sha256": admission["admission_file_sha256"],
                   "runtime_sha256": admission["runtime"]["actual_runtime_sha256"], "status": "ok"}
            row["response_sha256"] = response_hash(row)
            validate_raw(row, plan, key, eos)
            chunk.append(row)
            if len(chunk) == args.chunk_rows or index == len(pending) - 1:
                seal()
                if (run / "STOP_AFTER_CHUNK").exists() and index < len(pending) - 1:
                    write_once(run / "released.json", {"claim_identity": stable_hash(owner), "keys": pending[index + 1:],
                                                        "reason": "explicit_stop_after_completed_chunk", "released_utc": now()})
                    return {"status": "stopped_after_completed_chunk", "completed_rows": index + 1}
        receipt = {"status": "complete", "completed_rows": len(pending), "completed_utc": now(),
                   "registration_identity": registration["identity"], "claim_identity": stable_hash(owner)}
        write_once(run / "complete.json", receipt)
        return receipt
    except BaseException as exc:
        write_once(run / "failed.json", {"status": "failed", "key": key, "failed_utc": now(),
                   "error": type(exc).__name__ + ": " + str(exc), "traceback": traceback.format_exc(),
                   "automatic_retry": False, "automatic_parameter_change": False, "claim_identity": stable_hash(owner)})
        raise


def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-plan", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol", type=Path, default=package / "registration/protocol.json")
    parser.add_argument("--phase", choices=("calibration", "holdout"), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("/code/knowledge-deficit-mitigation"))
    parser.add_argument("--series-root", type=Path, default=Path("/code/knowledge-deficit-mitigation/outputs/hallusion_h100_20261009"))
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--cards", default="0")
    parser.add_argument("--claim-id")
    parser.add_argument("--generation-mode", choices=("fresh", "admitted-reuse", "reuse-only"), default="fresh")
    parser.add_argument("--reuse-sources", type=Path, help="Explicit late-bound source array; still requires passed model gate")
    parser.add_argument("--condition-ids", type=Path, help="JSON array of exclusive registered condition IDs")
    parser.add_argument("--sample-ids", type=Path, help="JSON array of exclusive IDs within the registered phase")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--chunk-rows", type=int, choices=(1, 8, 16), default=8)
    args = parser.parse_args()
    registration = load_registration(args.protocol)
    selection = load_selection(registration, args.selection) if args.selection else None
    plan = build_plan(registration, args.phase, args.model, selection,
                      read_json(args.condition_ids) if args.condition_ids else None,
                      read_json(args.sample_ids) if args.sample_ids else None, args.shard, args.n_shards)
    if args.execute:
        require(args.admission is not None and args.claim_id is not None, "Execution requires explicit host admission and claim ID")
        result = execute(args, registration, plan)
    elif args.verify:
        folder = args.series_root / "runs" / args.phase / args.model
        complete = complete_chunks(folder, plan)
        n = len(set(plan["selected"]) & set(complete))
        result = {"status": "complete" if n == len(plan["selected"]) else "incomplete", "expected_rows": len(plan["selected"]), "completed_rows": n}
    else:
        result = {"status": "CPU_plan_checked", "registration_identity": registration["identity"],
                  "phase": args.phase, "model": args.model, "full_phase_n": plan["full_phase_n"],
                  "conditions_in_assignment": len(plan["conditions"]), "selected_rows": len(plan["selected"]),
                  "output_token_cap": 128, "GPU_initialized": False}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 2 if args.verify and result["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
