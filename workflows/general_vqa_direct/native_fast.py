#!/usr/bin/env python3
"""Opt-in native greedy Direct, gated against eight sealed original outputs.

The original runner, inputs and model adapters remain unchanged. This entry
reuses their plan/admission/claims/sealing and the native call in models/verify.py.
Old ledgers and claims are read-only; the dispatcher owns cross-host assignments.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from workflows.general_vqa_direct import generate as base

FAST_MODELS = ("qwen25vl", "qwen35_4b", "llava16_mistral", "gemma3_4b",
               "onevision", "qwen3vl", "phi35")
ENGINE = "registered_hf_native_generate_greedy_v1"
SOURCE = "workflows/general_vqa_direct/native_fast.py"


def plans(args):
    plan = base.load_plan(args)
    plan["definition"] = copy.deepcopy(plan["definition"])
    plan["definition"]["engine"] = ENGINE
    plan["definition"]["source_sha256"][SOURCE] = base.file_hash(ROOT / SOURCE)
    plan["identity"] = base.stable_hash(plan["definition"])
    original_args = copy.copy(args)
    original_args.output, original_args.sample_ids = args.reference_output, None
    protocol = json.loads(base.relative(ROOT, args.protocol).read_text())
    original_args.datasets = list(protocol["datasets"])
    original = base.load_plan(original_args)
    if original["output"] == plan["output"]:
        raise ValueError("Native output must have a separate source/engine directory")
    return plan, original


def source_state(original):
    """Validate completed chunks and read unreleased ownership without writes."""
    complete = base.completed_keys(original)
    folder, occupied, bindings = original["output"], set(), {}
    ledger = folder / "completed_keys.jsonl"
    if ledger.exists():
        bindings["completed_keys.jsonl"] = base.file_hash(ledger)
    claims = folder / "claims"
    for directory in sorted(claims.iterdir()) if claims.exists() else []:
        if not directory.is_dir():
            continue
        path = directory / "owner.json"
        owner = json.loads(path.read_text())
        if owner["identity"] != original["identity"]:
            raise ValueError("Original claim has a foreign source identity")
        bindings[str(path.relative_to(folder))] = base.file_hash(path)
        released = set()
        path = directory / "released.json"
        if path.exists():
            release = json.loads(path.read_text())
            released = set(release["keys"])
            if (release["claim_identity"] != base.stable_hash(owner)
                    or release["reason"] != "stop_after_completed_chunk"
                    or not released <= set(owner["keys"])):
                raise ValueError("Original ownership release is not bound to its claim")
            bindings[str(path.relative_to(folder))] = base.file_hash(path)
        occupied.update(set(owner["keys"]) - complete - released)
    return complete, occupied, {"output": str(folder.relative_to(ROOT)),
                                "identity": original["identity"], "files": bindings}


def reference_eight(args, original):
    complete, _, state = source_state(original)
    ledger = list(base.read_jsonl(original["output"] / "completed_keys.jsonl"))
    ordered = [row["key"] for row in ledger if row["key"] in complete]
    if args.gate_sample_ids:
        ids = base.ids_from_file(base.relative(ROOT, args.gate_sample_ids))
        by_id = {item["sample"]["id"]: key for key, item in original["tasks"].items()}
        ordered = [by_id[sample_id] for sample_id in ids]
    else:
        ordered = ordered[:8]
    if len(ordered) != 8 or len(set(ordered)) != 8 or not set(ordered) <= complete:
        raise ValueError("Gate requires exactly eight distinct, verified sealed original keys")
    entries = {row["key"]: row for row in ledger}
    rows, bindings, cache = [], [], {}
    for key in ordered:
        entry = entries[key]
        receipt_path = base.relative(original["output"], entry["receipt"])
        receipt = json.loads(receipt_path.read_text())
        raw = base.relative(original["output"], receipt["raw_path"])
        if str(raw) not in cache:
            cache[str(raw)] = {row["key"]: row for row in base.read_jsonl(raw)}
        rows.append(cache[str(raw)][key])
        bindings.append({"key": key, "receipt": entry["receipt"],
                         "receipt_sha256": entry["receipt_sha256"],
                         "raw_path": receipt["raw_path"], "raw_sha256": receipt["raw_sha256"]})
    return rows, {"source": state, "sealed_references": bindings}


def native_generate(session, cfg, eos, decode):
    if cfg.method != "direct" or cfg.temperature != 0 or cfg.top_p != 1 or cfg.max_tokens <= 0:
        raise ValueError("Native fast entry only admits the frozen greedy Direct condition")
    backend = session.b
    if session.text_only or session.ohs or backend.em.mt in {"minicpmv", "internvl_chat"}:
        raise ValueError("This native fast revision admits unmodified HF image-text inputs only")
    # The current manual Direct loop applies no logits penalties or constraints.
    generation = backend.model.generation_config
    neutral = {"min_length": (0, None), "min_new_tokens": (0, None),
               "no_repeat_ngram_size": (0, None), "bad_words_ids": (None, []),
               "force_words_ids": (None, []), "forced_bos_token_id": (None,),
               "forced_eos_token_id": (None,), "suppress_tokens": (None, []),
               "begin_suppress_tokens": (None, []), "sequence_bias": (None, {}),
               "num_return_sequences": (1, None), "penalty_alpha": (None, 0)}
    if any(getattr(generation, name, allowed[0]) not in allowed for name, allowed in neutral.items()):
        raise ValueError("Native checkpoint generation config would alter plain greedy Direct")
    inputs = dict(session.inputs)
    offset = inputs["input_ids"].shape[-1]
    kwargs = {"do_sample": False, "temperature": 0.0, "top_p": 1.0,
              "num_beams": 1, "num_return_sequences": 1, "repetition_penalty": 1.0,
              "max_new_tokens": cfg.max_tokens, "eos_token_id": sorted(eos),
              "pad_token_id": backend.tokenizer.pad_token_id or backend.tokenizer.eos_token_id,
              "use_cache": True, "return_dict_in_generate": False, "output_scores": False,
              "output_attentions": False, "output_hidden_states": False}
    with backend.torch.inference_mode():
        output = backend.model.generate(**inputs, **kwargs)
        tokens = output[0, offset:].tolist()
    if (not tokens or len(tokens) > cfg.max_tokens or any(token in eos for token in tokens[:-1])
            or (tokens[-1] not in eos and len(tokens) != cfg.max_tokens)):
        raise ValueError("Native token sequence violates original EOS/budget semantics")
    return {"status": "ok", "tokens": tokens, "text": decode(tokens),
            "terminated": tokens[-1] in eos, "engine": ENGINE,
            "probability_recording": "not_requested_not_measured"}


def session_for(backend, sample):
    from PIL import Image
    from workflows.general_vqa_direct.inputs import direct_session
    images = []
    for filename, sha in zip(sample["image_paths"], sample["image_sha256"]):
        path = base.relative(ROOT, filename)
        if base.file_hash(path) != sha:
            raise ValueError("Image differs from the frozen manifest: " + filename)
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return direct_session(backend, images, sample)


def run_gate(args, plan, original, backend, references, reference_binding, admission, load_wall):
    path = plan["output"] / "gates" / (args.claim_id + ".json")
    if path.exists():
        raise ValueError("Refusing to replace a previous gate result")
    report = {"schema": "kdm_native_direct_exact8_gate_v1", "status": "failed", "passed": False,
              "identity": plan["identity"], "engine": ENGINE, "model": args.model,
              "source_sha256": plan["definition"]["source_sha256"], "reference": reference_binding,
              "admission": admission, "model_load_wall_s": load_wall, "rows": [],
              "scope": "Exact eight recorded samples; other inputs are not empirically covered by this gate",
              "checkpoint_generation_config": backend.model.generation_config.to_dict(),
              "executed_model_generation": False, "started_utc": base.now(), **base.process_identity()}
    try:
        for old in references:
            start = time.perf_counter()
            session = session_for(backend, old["sample"])
            evidence = session.general_vqa_input_evidence
            if evidence != old["input_evidence"]:
                raise ValueError("Gate input IDs or frozen prompt/image ordering differs: " + old["key"])
            cfg = original["datasets"][old["sample"]["dataset"]]["cfg"]
            result = native_generate(session, cfg, set(backend.eos), backend.decode)
            report["executed_model_generation"] = True
            row = {"key": old["key"], "sample_id": old["sample"]["id"],
                   "dataset": old["sample"]["dataset"], "dataset_identity": old["dataset_identity"],
                   "config": asdict(cfg), "seed": old["seed"], "input_evidence": evidence,
                   "input_ids": session.inputs["input_ids"].detach().cpu().tolist(),
                   "tokens": result["tokens"], "reference_tokens": old["tokens"],
                   "terminated": result["terminated"], "reference_terminated": old["terminated"],
                   "tokens_equal": result["tokens"] == old["tokens"],
                   "termination_equal": result["terminated"] == old["terminated"],
                   "native_wall_s": time.perf_counter() - start, "reference_wall_s": old["wall_s"]}
            report["rows"].append(row)
            del session
            if not row["tokens_equal"] or not row["termination_equal"]:
                raise ValueError("Native gate token/EOS mismatch: " + old["key"])
        report.update(status="passed", passed=True, completed=8,
                      native_mean_input_wall_s=sum(row["native_wall_s"] for row in report["rows"]) / 8,
                      reference_mean_input_wall_s=sum(row["reference_wall_s"] for row in report["rows"]) / 8)
    except BaseException as exc:
        report.update(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["completed_utc"] = base.now()
        path.parent.mkdir(parents=True, exist_ok=True)
        base.write_once(path, report)
    return {"path": str(path.relative_to(ROOT)), "sha256": base.file_hash(path),
            "passed": True, "completed": 8, "reference_identity": original["identity"]}


def check_assignment(plan, original):
    complete, occupied, binding = source_state(original)
    pending = set(plan["selected"]) - base.completed_keys(plan)
    if pending & complete or pending & occupied:
        raise ValueError("Native assignment intersects original completed or unreleased claimed keys")
    return binding


def execute(args, plan, original, references, reference_binding):
    if not args.gate_only and set(plan["selected"]) <= base.completed_keys(plan):
        return {"status": "already_complete", "selected_rows": len(plan["selected"])}
    actual, admission = base.admit(args, plan)
    if not args.gate_only:
        admission["original_assignment_check"] = check_assignment(plan, original)
    started = time.perf_counter()
    backend = base.make_backend(actual, "cuda:0")
    load_wall = time.perf_counter() - started
    gate = run_gate(args, plan, original, backend, references, reference_binding, admission, load_wall)
    if args.gate_only:
        return {"status": "actual_exact8_gate_passed", "gate": gate}
    # The worker still holds its original GPU flock. Recheck source assignments
    # after the gate, then reuse the resident model with the unchanged writer.
    admission["original_assignment_check"] = check_assignment(plan, original)
    admission.update(native_fast_gate=gate, resident_model_load_wall_s=load_wall,
                     model_reuse="loaded once before exact8 gate; execute loader is a resident lookup")
    old_admit, old_factory, old_generate = base.admit, base.make_backend, base.generate
    try:
        base.admit = lambda _args, _plan: (actual, admission)
        base.make_backend = lambda _actual, _device: backend
        def generate(session, reference, cfg, eos, decode, seed=0, neutral_main=None):
            if reference is not None or neutral_main is not None:
                raise ValueError("Native fast Direct cannot use reference/neutral branches")
            return {**native_generate(session, cfg, eos, decode), "native_fast_gate": gate}
        base.generate = generate
        return base.execute(args, plan)
    finally:
        base.admit, base.make_backend, base.generate = old_admit, old_factory, old_generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-plan", action="store_true")
    mode.add_argument("--gate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--model", choices=FAST_MODELS, required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--sample-ids", help="Required for production; root-assigned never-claimed or released IDs")
    parser.add_argument("--reference-output", required=True, help="Read-only original output parent, with authentic sealed chunks and owner/ledger")
    parser.add_argument("--gate-sample-ids", help="Optional exactly eight original completed IDs; default is first eight in original ledger")
    parser.add_argument("--output", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--cards", required=True)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--chunk-rows", type=int, choices=(64, 128), default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    args.k100_intern = False
    if args.execute and not args.sample_ids:
        parser.error("--execute requires an explicit --sample-ids assignment")
    plan, original = plans(args)
    if args.verify:
        complete = base.completed_keys(plan)
        expected = set(plan["selected"])
        result = {"status": "complete" if expected <= complete else "incomplete",
                  "expected_rows": len(expected), "completed_rows": len(expected & complete)}
    else:
        references, binding = reference_eight(args, original)
        if args.check_plan:
            assignment = check_assignment(plan, original) if args.sample_ids else None
            result = {"status": "CPU_plan_checked_GPU_gate_not_run", "identity": plan["identity"],
                      "engine": ENGINE, "selected_rows": len(plan["selected"]),
                      "gate_keys": [row["key"] for row in references], "reference": binding,
                      "assignment_check": assignment, "gpu_initialized": False}
        else:
            result = execute(args, plan, original, references, binding)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 2 if args.verify and result["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
