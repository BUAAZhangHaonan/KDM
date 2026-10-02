#!/usr/bin/env python3
"""Actual registered Food M3ID matrix/ref-off conditions on existing K100 software."""
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
from kdm.pipeline import task_id
from kdm.prompts import task_prompt
from kdm.protocol import validate_environment
from workflows.paper_core.native_audit import input_description
from workflows.supplemental.remaining11 import execution, generate
from workflows.supplemental.remaining4.native_audit import NativeAuditBackend, NativeAuditSession
from workflows.supplemental.remaining4.registered_matrix import restored_scope


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class MatrixObserver(NativeAuditBackend):
    """Keep the original M3ID score audit while accepting each registered prompt."""
    def __init__(self, backend, tasks, cfg):
        super().__init__(backend, "onevision", tasks, replace(cfg, method="m3id"))
        self.sessions = 0

    def session(self, image, prompt, reference="clean", seed=0, need_layers=False):
        index, role = divmod(self.sessions, 2)
        if index >= len(self.tasks):
            raise ValueError("Actual M3ID gate creates additional branch sessions")
        task = self.tasks[index]
        expected = task_prompt(task["sample"]["question"],
            task["reference_marker"] if role else task["marker"],
            task["reference_guided"] if role else task["guided"])
        mode = "text_only" if role else "clean"
        if prompt != expected or reference != mode or need_layers:
            raise ValueError("Actual M3ID matrix prompt, reference mode, or layers differ")
        if role and seed != stable_seed(task["sample"]["id"], self.model, 0):
            raise ValueError("Actual M3ID reference seed differs")
        session = self.backend.session(image, prompt, reference=reference,
                                       seed=seed, need_layers=False)
        self.sessions += 1
        if not role:
            self.records.append({"key": task_id(self.model, task),
                "sample_id": task["sample"]["id"],
                "condition": {key: value for key, value in task.items() if key != "sample"},
                "main_prompt": prompt, "clean_inputs": input_description(session.inputs),
                "main_prompt_token_ids": session.inputs["input_ids"].detach().cpu().tolist(),
                "offset_prompt_tokens": list(self.backend.encode(prompt)), "step_checks": []})
        else:
            if (session.text_only is not True
                    or input_description(session.inputs) != input_description(self.backend._text_inputs(prompt))):
                raise ValueError("Actual M3ID reference differs from original text-only adapter")
            self.records[index].update(reference_prompt=prompt, reference_mode=mode, seed=seed,
                reference_inputs=input_description(session.inputs),
                reference_prompt_token_ids=session.inputs["input_ids"].detach().cpu().tolist())
        return NativeAuditSession(self, session, index, reference)

    def proof(self, rows):
        if len(rows) != len(self.tasks) or self.sessions != 2 * len(rows) or self.clean_steps:
            raise ValueError("Actual two-route M3ID gate coverage is incomplete")
        for record, row, task in zip(self.records, rows, self.tasks):
            if (row["key"] != task_id(self.model, task) or record["key"] != row["key"]
                    or len(record["step_checks"]) != len(row["tokens"])
                    or len(row["trace"]) != len(row["tokens"])
                    or row["offset_prompt_tokens"] != record["offset_prompt_tokens"]
                    or row["terminated"] != (row["tokens"][-1] in self.backend.eos)):
                raise ValueError("Actual M3ID gate keys, offset, token trace, or EOS differ")
            for check, token, trace in zip(record["step_checks"], row["tokens"], row["trace"]):
                if (token != check["pipeline_argmax"] or trace["weight"] != check["weight"]
                        or trace["active"] != check["active"]
                        or check["official_logp_gap_at_pipeline_argmax"] > 1e-8):
                    raise ValueError("Actual M3ID matrix token differs from its registered score operator")
            record.update(tokens=row["tokens"], text=row["text"], config=row["config"],
                          terminated=row["terminated"], wall_s=row["wall_s"])
        return {"schema": "kdm_k100_actual_registered_m3id_condition_gate_v1",
                "model": self.model, "method": "m3id", "dataset": "food101", "passed": True,
                "completed": len(rows), "records": self.records, "noise_called": False,
                "two_actual_branches_per_input": True,
                "gate_generation_wall_s": sum(row["wall_s"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--phase", choices=("gate", "production"), required=True)
    parser.add_argument("--missing-keys", required=True)
    parser.add_argument("--key-start", type=int, default=0)
    parser.add_argument("--key-stop", type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--host-registry", required=True)
    parser.add_argument("--previous-software-gate", required=True)
    parser.add_argument("--condition-gate")
    parser.add_argument("--gate-missing-keys")
    parser.add_argument("--gate-source-runner")
    parser.add_argument("--executor-agent", required=True)
    parser.add_argument("--executor-model")
    parser.add_argument("--executor-effort")
    parser.add_argument("--plan-output")
    args = parser.parse_args()
    if bool(args.gate_missing_keys) != bool(args.gate_source_runner) or (
            args.gate_missing_keys and args.phase != "production"):
        raise ValueError("A production manifest extension requires both original gate inputs and runner")
    args.model, args.dataset, args.stage = "onevision", "food101", "formal"
    args.shard, args.n_shards, args.continuation_receipt = 0, 1, None
    args.chunk_rows = 8 if args.phase == "gate" else 512
    registry = within(ROOT, args.host_registry)
    run = within(ROOT, args.run_root)
    run.relative_to(ROOT / "outputs/supplemental/remaining4")
    previous_path = within(ROOT, args.previous_software_gate)
    previous = json.loads(previous_path.read_text())
    old_audit = within(ROOT, previous["actual_operator_audit_path"])
    software = previous["software_compatibility"]
    if (not previous["passed"] or not previous["production_allowed"] or previous["differences"]
            or previous["completed_inputs"] != 8 or software["actual_method_scope"] != "m3id"
            or software["parameters_changed"] or not software["remaining_registered_versions_unchanged"]
            or file_hash(old_audit) != previous["actual_operator_audit_sha256"]):
        raise ValueError("Existing OneVision K100 actual M3ID software evidence is absent")
    old_operator = json.loads(old_audit.read_text())
    if old_operator["model"] != "onevision" or old_operator["method"] != "m3id" or not old_operator["passed"]:
        raise ValueError("Existing software evidence belongs to another operator or model")
    policy = json.loads(registry.read_text())
    host = policy["hosts"]["k100"]
    if (socket.gethostname() != host["hostname"] or str(ROOT) != host["root"]
            or host["allowed_gpus"] != [0] or host["max_workers_per_gpu"] != 1
            or host["minimum_free_mib_by_model"]["onevision"] != 65536):
        raise ValueError("Dedicated K100 M3ID gate changes its original capacity or host")
    execution.REGISTRY = str(registry.relative_to(ROOT))
    original_spec = execution.runtime_spec

    def actual_spec(root, frozen_spec, model):
        if model != "onevision":
            raise ValueError("Existing K100 software cannot admit a different model")
        value = original_spec(root, frozen_spec, model)
        value["versions"] = copy.deepcopy(frozen_spec["versions"])
        value["versions"].update(torch=software["actual_torch"],
                                  torchvision=software["actual_torchvision"])
        return value

    execution.runtime_spec = actual_spec
    plan = generate.load_plan(args)
    scope = restored_scope(plan)
    tasks = list(generate.selected_tasks(plan))
    if not tasks or any(task["method"] != "m3id" for task in tasks):
        raise ValueError("Existing K100 runtime cannot admit VCD, IP-VCD, layers, SID, or noise")
    actual = execution.runtime_spec(ROOT, plan["spec"], args.model)
    environment = validate_environment(actual)
    checkpoint = execution.checkpoint_identity(actual)
    if args.phase == "gate" and len(tasks) != 8:
        raise ValueError("The new-condition gate must contain exactly eight real production keys")
    identity = {"existing_software_gate_path": args.previous_software_gate,
        "existing_software_gate_sha256": file_hash(previous_path),
        "actual_software": software, "actual_environment": environment,
        "unchanged_checkpoint": checkpoint, "new_condition_gate_is_separate": True,
        "phase": args.phase, "scope_counts": scope, "parameters_changed": False,
        "runner_sha256": file_hash(Path(__file__)),
        "executor": {"agent": args.executor_agent, "model": args.executor_model,
                     "effort": args.executor_effort, "call_id": ""}}
    if args.phase == "production":
        gate_path = within(ROOT, args.condition_gate or "")
        gate = json.loads(gate_path.read_text())
        operator = within(ROOT, gate["operator_audit_path"])
        gate_inputs = within(ROOT, args.gate_missing_keys or args.missing_keys)
        gate_runner = within(ROOT, args.gate_source_runner) if args.gate_source_runner else Path(__file__)
        if (not gate["passed"] or gate["completed"] != 8 or not gate["production_allowed"]
                or gate["full_missing_keys_sha256"] != file_hash(gate_inputs)
                or gate["registry_sha256"] != file_hash(registry)
                or gate["runner_sha256"] != file_hash(gate_runner)
                or gate["existing_software_gate_sha256"] != file_hash(previous_path)
                or gate["operator_audit_sha256"] != file_hash(operator)
                or set(plan["selected_keys"]).intersection(gate["completed_keys"])):
            raise ValueError("Production requires this actual new-condition eight gate and disjoint keys")
        if args.gate_missing_keys:
            current_inputs = within(ROOT, args.missing_keys)
            original_bytes = gate_inputs.read_bytes()
            current_bytes = current_inputs.read_bytes()
            original_keys = generate.read_missing_keys(gate_inputs, args.model, args.stage, args.dataset)
            current_keys = generate.read_missing_keys(current_inputs, args.model, args.stage, args.dataset)
            if (not original_bytes.endswith(b"\n") or not current_bytes.startswith(original_bytes)
                    or len(current_bytes) <= len(original_bytes)
                    or not set(gate["completed_keys"]) <= set(original_keys)
                    or set(plan["selected_keys"]).intersection(original_keys)):
                raise ValueError("The explicit extension must preserve the complete original prefix and run new keys")
            identity["explicit_registered_input_extension"] = {
                "gate_missing_keys_path": args.gate_missing_keys,
                "gate_missing_keys_sha256": file_hash(gate_inputs),
                "gate_source_runner_path": args.gate_source_runner,
                "gate_source_runner_sha256": file_hash(gate_runner),
                "current_missing_keys_path": args.missing_keys,
                "current_missing_keys_sha256": file_hash(current_inputs),
                "original_prefix_keys": len(original_keys), "current_registered_keys": len(current_keys),
                "entire_current_manifest_validated_by_original_load_plan": True,
                "original_prefix_not_rerun": True, "actual_gate_not_repeated": True,
                "generation_algorithm_changed": False}
        identity.update(condition_gate_path=args.condition_gate,
                        condition_gate_sha256=file_hash(gate_path))
    if args.check_plan:
        output = {"plan": plan["summary"], "software_and_scope": identity, "gpu_initialized": False}
        if args.plan_output:
            write_new(within(ROOT, args.plan_output), output)
        print(json.dumps({"expected_rows": len(tasks), "scope": scope, "actual_environment": environment}))
        return
    if not all(generate.NAME_PATTERN.fullmatch(value) for value in
               (args.run_name, args.claim_id, args.owner)):
        raise ValueError("Execution requires unique explicit owner and claim names")
    original_admission = execution.validate_supplemental_runtime

    def admission(*values, **kwargs):
        result = original_admission(*values, **kwargs)
        result["explicit_existing_K100_software_and_new_condition_scope"] = identity
        return result

    execution.validate_supplemental_runtime = admission
    observers = []
    original_backend = generate.make_backend
    if args.phase == "gate":
        def make_backend(spec, device):
            backend = MatrixObserver(original_backend(spec, device), tasks, plan["cfg"])
            observers.append(backend)
            return backend
        generate.make_backend = make_backend
    write_new(run / "commands" / (args.claim_id + ".actual_scope.json"), identity)
    os.environ["KDM_SUPPLEMENTAL_DATASET"] = args.dataset
    os.environ["KDM_SUPPLEMENTAL_METHOD"] = "m3id"
    generate.execute(args)
    if args.phase == "gate":
        record = run / "records/onevision/food101/formal" / args.claim_id
        complete = json.loads((record / "complete.json").read_text())
        if not complete["generation_complete"] or complete["completed"] != 8 or len(observers) != 1:
            raise ValueError("New condition gate has no complete real eight-key generation")
        rows = list(read_jsonl(within(ROOT, complete["completed_parts"][0]["raw_path"])))
        operator_path = record / "operator_audit.json"
        write_new(operator_path, observers[0].proof(rows))
        import torch
        gate = {"schema": "kdm_k100_M3ID_registered_condition_gate_receipt_v1",
            "model": args.model, "method": "m3id", "dataset": args.dataset,
            "completed": 8, "completed_keys": [row["key"] for row in rows],
            "full_missing_keys_sha256": file_hash(within(ROOT, args.missing_keys)),
            "registry_sha256": file_hash(registry), "runner_sha256": file_hash(Path(__file__)),
            "existing_software_gate_sha256": file_hash(previous_path),
            "operator_audit_path": str(operator_path.relative_to(ROOT)),
            "operator_audit_sha256": file_hash(operator_path),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "scope": scope, "actual_identity": identity, "passed": True,
            "production_allowed": True, "noise_methods_allowed": False,
            "first8_in_full_production": True}
        write_new(record / "condition_gate.json", gate)
        print(json.dumps({key: value for key, value in gate.items() if key != "actual_identity"}))


if __name__ == "__main__":
    main()
