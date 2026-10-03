#!/usr/bin/env python3
"""Explicit K100 single-map admission and registered InternVL matrix continuation."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from kdm.io import file_hash, read_jsonl, stable_seed, within
from kdm.models.hf import vcd_processed_noise
from kdm.pipeline import run_tasks, task_id
from kdm.prompts import task_prompt
from kdm.protocol import validate_environment
from workflows.paper_core.native_audit import input_description
from workflows.supplemental.remaining11 import execution, generate
from workflows.supplemental.remaining4.internvl_k100_single import expected_map
from workflows.supplemental.remaining4.native_audit import NativeAuditBackend, NativeAuditSession
from workflows.supplemental.remaining4.registered_matrix import restored_scope


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def single_spec(frozen):
    result = copy.deepcopy(frozen)
    original = {key: (0 if key not in ("language_model.model.norm", "language_model.lm_head")
                      and not (key.startswith("language_model.model.layers.")
                               and int(key.rsplit(".", 1)[1]) >= 18) else 1)
                for key in expected_map()}
    if (result["key"] != "internvl35_8b" or result["gpu_count"] != 2
            or result["factory"] != "kdm.models.internvl_dual:InternVLDualBackend"
            or result["kwargs"]["device_map"] != original
            or result["kwargs"]["max_memory"] != {"0": "22GiB", "1": "22GiB"}
            or result["kwargs"]["dtype"] != "bfloat16"
            or result["versions"] != {"torch": "2.9.0+cu128", "transformers": "5.17.0"}):
        raise ValueError("The original InternVL dual identity or software differs")
    result["gpu_count"] = 1
    result["factory"] = "workflows.supplemental.remaining4.internvl_k100_single:InternVLK100SingleBackend"
    result["kwargs"]["device_map"] = expected_map()
    result["kwargs"]["max_memory"] = {"0": "44GiB"}
    return result


class MatrixObserver(NativeAuditBackend):
    def __init__(self, backend, tasks, cfg):
        super().__init__(backend, "internvl35_8b", tasks, cfg)
        self.sessions = 0

    def session(self, image, prompt, reference="clean", seed=0, need_layers=False):
        index, role = divmod(self.sessions, 2)
        if index >= len(self.tasks):
            raise ValueError("The gate requested more than its eight two-branch tasks")
        task = self.tasks[index]
        self.cfg = replace(self.cfg, method=task["method"])
        expected = task_prompt(task["sample"]["question"],
                              task["reference_marker"] if role else task["marker"],
                              task["reference_guided"] if role else task["guided"])
        mode = ("noise" if task["method"] == "vcd" else "text_only") if role else "clean"
        if prompt != expected or reference != mode or need_layers:
            raise ValueError("The registered prompt, reference type, or layer mode differs")
        if role and seed != stable_seed(task["sample"]["id"], self.model, task["replicate"]):
            raise ValueError("The registered reference perturbation seed differs")
        session = self.backend.session(image, prompt, reference=reference, seed=seed, need_layers=False)
        self.sessions += 1
        if not role:
            self.records.append({"key": task_id(self.model, task), "sample_id": task["sample"]["id"],
                "condition": {key: value for key, value in task.items() if key != "sample"},
                "main_prompt": prompt, "clean_inputs": input_description(session.inputs),
                "main_prompt_token_ids": session.inputs["input_ids"].detach().cpu().tolist(),
                "offset_prompt_tokens": list(self.backend.encode(prompt)), "step_checks": []})
        else:
            record = self.records[index]
            if mode == "noise":
                clean_reference = self.backend.em.build(image, prompt)
                expected_pixels = vcd_processed_noise(clean_reference["pixel_values"], seed)
                expected_other = {key: value for key, value in clean_reference.items() if key != "pixel_values"}
                actual_other = {key: value for key, value in session.inputs.items() if key != "pixel_values"}
                if (input_description(expected_pixels) != input_description(session.inputs["pixel_values"])
                        or input_description(expected_other) != input_description(actual_other)):
                    raise ValueError("The actual reference differs from the unchanged processed-tensor VCD noise")
                record.update(noise_step=500, expected_noise_recreated_exactly=True,
                              reference_prompt_clean_inputs=input_description(clean_reference))
            elif (session.text_only is not True
                  or input_description(session.inputs) != input_description(self.backend._text_inputs(prompt))):
                raise ValueError("The actual M3ID reference differs from the original text-only adapter")
            record.update(reference_prompt=prompt, reference_mode=mode, seed=seed,
                          reference_inputs=input_description(session.inputs),
                          reference_prompt_token_ids=session.inputs["input_ids"].detach().cpu().tolist())
        return NativeAuditSession(self, session, index, reference)

    def proof(self, rows, source_rows):
        if len(rows) != 8 or self.sessions != 16 or self.clean_steps:
            raise ValueError("The gate lacks eight complete, finite two-branch generations")
        sources = {row["key"]: row for row in source_rows}
        for record, row, task in zip(self.records, rows, self.tasks):
            source = sources[row["key"]]
            if (row["key"] != task_id(self.model, task) or record["key"] != row["key"]
                    or row["seed"] != source["seed"] or row["config"] != source["config"]
                    or row["prompt"] != source["prompt"]
                    or row["reference_prompt"] != source["reference_prompt"]
                    or len(record["step_checks"]) != len(row["tokens"])
                    or len(row["trace"]) != len(row["tokens"])
                    or row["terminated"] != (row["tokens"][-1] in self.backend.eos)):
                raise ValueError("The eight gate identity, seed, config, prompt, trace, or termination differs")
            for check, token, trace in zip(record["step_checks"], row["tokens"], row["trace"]):
                if (token != check["pipeline_argmax"] or trace["weight"] != check["weight"]
                        or trace["active"] != check["active"]
                        or check["official_logp_gap_at_pipeline_argmax"] > 1e-8):
                    raise ValueError("An actual gate token differs from the registered score operator")
            if task["method"] == "m3id" and row["offset_prompt_tokens"] != source["offset_prompt_tokens"]:
                raise ValueError("The M3ID prompt offset differs from the original tokenization")
            record.update(tokens=row["tokens"], text=row["text"], config=row["config"],
                          terminated=row["terminated"], wall_s=row["wall_s"],
                          source_tokens=source["tokens"], source_text=source["text"],
                          source_terminated=source["terminated"],
                          tokens_equal_to_original_dual=row["tokens"] == source["tokens"])
        return {"schema": "kdm_k100_InternVL_single_actual_operator_gate_v1", "passed": True,
                "completed": 8, "scientific_unique_increment": 0, "admission_duplicate_generations": 8,
                "records": self.records, "two_actual_branches_per_input": True,
                "original_prompt_seed_config_equal": True,
                "full_vocab_finite_and_registered_operator_equal": True,
                "cross_hardware_tokens_equal": all(record["tokens_equal_to_original_dual"] for record in self.records),
                "cross_hardware_exact_tokens_required_for_operator_check": False,
                "original_dual_processed_tensor_hashes": None,
                "original_dual_noise_tensor_hashes": None,
                "original_dual_tensor_hash_comparison": "unknown: original raw source does not contain these hashes",
                "gate_generation_wall_s": sum(row["wall_s"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--phase", choices=("gate", "production"), required=True)
    parser.add_argument("--missing-keys", required=True)
    parser.add_argument("--source-rows")
    parser.add_argument("--source-fixture")
    parser.add_argument("--condition-gate")
    parser.add_argument("--key-start", type=int, default=0)
    parser.add_argument("--key-stop", type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--host-registry", required=True)
    parser.add_argument("--executor-agent", required=True)
    parser.add_argument("--executor-model", default="unknown")
    parser.add_argument("--executor-effort", default="unknown")
    parser.add_argument("--plan-output")
    args = parser.parse_args()
    args.model, args.dataset, args.stage = "internvl35_8b", "food101", "formal"
    args.shard, args.n_shards, args.chunk_rows, args.continuation_receipt = 0, 1, 512, None
    registry_path = within(ROOT, args.host_registry)
    policy = json.loads(registry_path.read_text())
    host = policy["hosts"]["k100"]
    if (socket.gethostname() != host["hostname"] or str(ROOT) != host["root"]
            or host["allowed_gpus"] != [0] or host["max_workers_per_gpu"] != 1
            or host["minimum_free_mib_by_model"][args.model] != 65536):
        raise ValueError("The explicit single-card route changed its dedicated host or capacity")
    execution.REGISTRY = str(registry_path.relative_to(ROOT))
    plan = generate.load_plan(args)
    scope = restored_scope(plan)
    actual = execution.runtime_spec(ROOT, single_spec(plan["spec"]), args.model)
    environment = validate_environment(actual)
    checkpoint = execution.checkpoint_identity(actual)
    factory_path = Path(__file__).with_name("internvl_k100_single.py")
    identity = {"schema": "kdm_user_authorized_InternVL_single_hardware_mapping_v1",
        "authorization": "2026-10-03 user explicitly requested InternVL tail migration to released K100 GPU0",
        "original_registered_backend": plan["spec"], "actual_single_backend": actual,
        "hardware_changes": {"gpu_count": [2, 1], "device_map": "original 18+18 mapped wholly onto logical GPU0",
                             "max_memory": {"old": {"0": "22GiB", "1": "22GiB"}, "new": {"0": "44GiB"}}},
        "software_changed": False, "parameters_changed": False, "precision_changed": False,
        "processor_template_or_inputs_changed": False, "actual_environment": environment,
        "unchanged_checkpoint": checkpoint, "phase": args.phase, "scope": scope,
        "factory_sha256": file_hash(factory_path), "runner_sha256": file_hash(Path(__file__)),
        "registry_sha256": file_hash(registry_path),
        "executor": {"agent": args.executor_agent, "model": args.executor_model,
                     "effort": args.executor_effort, "call_id": ""}}
    source_rows = []
    if args.phase == "gate":
        tasks = list(generate.selected_tasks(plan))
        source_path = within(ROOT, args.source_rows or "")
        fixture_path = within(ROOT, args.source_fixture or "")
        source_rows = list(read_jsonl(source_path))
        fixture = json.loads(fixture_path.read_text())
        if (len(tasks) != 8 or len(source_rows) != 8
                or sorted(task["method"] for task in tasks) != ["m3id"] * 4 + ["vcd"] * 4
                or [task_id(args.model, task) for task in tasks] != [row["key"] for row in source_rows]
                or fixture["source_rows_sha256"] != file_hash(source_path)
                or fixture["gate_keys_sha256"] != file_hash(within(ROOT, args.missing_keys))):
            raise ValueError("The independent admission requires four VCD and four M3ID immutable source keys")
        identity.update(source_fixture=fixture, source_fixture_sha256=file_hash(fixture_path),
                        scientific_unique_increment=0, admission_duplicate_generations=8)
    else:
        gate_path = within(ROOT, args.condition_gate or "")
        gate = json.loads(gate_path.read_text())
        operator_path = within(ROOT, gate["operator_audit_path"])
        operator = json.loads(operator_path.read_text())
        if (not gate["passed"] or not gate["production_allowed"] or gate["completed"] != 8
                or gate["registry_sha256"] != file_hash(registry_path)
                or gate["runner_sha256"] != file_hash(Path(__file__))
                or gate["factory_sha256"] != file_hash(factory_path)
                or gate["operator_audit_sha256"] != file_hash(operator_path)
                or not operator["passed"] or not operator["original_prompt_seed_config_equal"]
                or not operator["full_vocab_finite_and_registered_operator_equal"]
                or set(plan["selected_keys"]).intersection(gate["admission_duplicate_keys"])):
            raise ValueError("Production requires the real single-hardware gate and genuinely unfinished different keys")
        identity.update(condition_gate_path=args.condition_gate, condition_gate_sha256=file_hash(gate_path))
    if args.check_plan:
        result = {"plan": plan["summary"], "hardware_identity": identity,
                  "no_GPU_execution": True, "no_generation_admission_claimed": True}
        if args.plan_output:
            write_new(within(ROOT, args.plan_output), result)
        print(json.dumps({"expected_rows": plan["summary"]["expected_generation_rows"],
                          "scope": scope, "actual_environment": environment}))
        return
    if not all(generate.NAME_PATTERN.fullmatch(value) for value in (args.run_name, args.claim_id, args.owner)):
        raise ValueError("Execution requires explicit exclusive owner and claim names")
    run = within(ROOT, args.run_root)
    run.relative_to(ROOT / "outputs/supplemental/remaining4")
    write_new(run / "commands" / (args.claim_id + ".hardware_scope.json"), identity)
    os.environ["KDM_SUPPLEMENTAL_DATASET"] = args.dataset
    original_admission = execution.validate_supplemental_runtime

    def admission(root, spec, model, cards, stage, claim_id, owner):
        if model != args.model or stage != "formal" or cards != ["0"]:
            raise ValueError("The new route admits only registered InternVL Food on K100 GPU0")
        result = original_admission(root, single_spec(spec), model, cards, stage, claim_id, owner)
        result["explicit_authorized_single_hardware_scope"] = identity
        return result

    execution.validate_supplemental_runtime = admission
    if args.phase == "production":
        generate.execute(args)
        return
    record = run / "admission" / args.claim_id
    record.mkdir(parents=True, exist_ok=False)
    try:
        admitted = admission(ROOT, plan["spec"], args.model, ["0"], args.stage, args.claim_id, args.owner)
        write_new(record / "admission.json", admitted)
        backend = generate.make_backend(admitted["runtime_spec"], "cuda:0")
        observer = MatrixObserver(backend, tasks, plan["cfg"])
        raw = record / "admission_duplicate8.jsonl"
        count = run_tasks(observer, args.model, tasks, raw,
                          {"source_provenance": plan["summary"]["source_provenance"],
                           "runtime_admission": admitted, "independent_admission_duplicate": True}, plan["cfg"])
        rows = list(read_jsonl(raw))
        if count != 8:
            raise ValueError("The real independent admission did not generate eight records")
        operator_path = record / "operator_audit.json"
        proof = observer.proof(rows, source_rows)
        write_new(operator_path, proof)
        import torch

        gate = {"schema": "kdm_K100_InternVL_single_actual_gate_receipt_v1", "passed": True,
            "completed": 8, "production_allowed": True, "scientific_unique_increment": 0,
            "admission_duplicate_keys": [row["key"] for row in rows],
            "operator_audit_path": str(operator_path.relative_to(ROOT)),
            "operator_audit_sha256": file_hash(operator_path), "registry_sha256": file_hash(registry_path),
            "factory_sha256": file_hash(factory_path), "runner_sha256": file_hash(Path(__file__)),
            "actual_hf_device_map": getattr(backend.model, "hf_device_map", None),
            "actual_placement_evidence": backend.em.actual_placement_evidence,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "finished_utc": generate.now(), "actual_identity": identity,
            "raw_path": str(raw.relative_to(ROOT)), "raw_sha256": file_hash(raw)}
        write_new(record / "condition_gate.json", gate)
        print(json.dumps({key: value for key, value in gate.items() if key != "actual_identity"}))
    except BaseException as error:
        import traceback

        write_new(record / "failed.json", {"when": generate.now(), "error": type(error).__name__ + ": " + str(error),
                   "traceback": traceback.format_exc(), "production_allowed": False,
                   "scientific_unique_increment": 0, "identity": identity})
        raise


if __name__ == "__main__":
    main()
