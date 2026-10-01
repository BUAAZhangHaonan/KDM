"""OneVision VizWiz diagonal IP-M3ID with a separate actual gate for each admitted marker."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from kdm.io import file_hash, read_jsonl, within
from kdm.protocol import validate_environment
from workflows.supplemental.remaining11 import execution, generate
from workflows.supplemental.remaining4 import registered
from workflows.supplemental.remaining4.ip_m3id_compatibility import IPObserver, write

MARKERS = ("UNKNOWN", "UNCLEAR", "UNSURE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--phase", choices=("gate", "production"), required=True)
    parser.add_argument("--marker", choices=MARKERS, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--host-registry", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--software-gate")
    parser.add_argument("--plan-output")
    args = parser.parse_args()
    args.model, args.dataset, args.stage = "onevision", "vizwiz", "formal"
    args.shard, args.n_shards, args.key_start, args.key_stop = 0, 1, 0, None
    args.chunk_rows = 8 if args.phase == "gate" else 512
    args.continuation_receipt = None
    source_path = within(ROOT, args.source)
    source = json.loads(source_path.read_text())
    if (source["schema"] != "kdm_selected4_one_viz_ip_m3id_CPU_fixture_v1"
            or source["model"] != args.model or source["dataset"] != args.dataset
            or source["method"] != "instruction_m3id" or source["marker"] != args.marker
            or source["original_cuda_initialized"] or source["active_producer_overlap"] != 0):
        raise ValueError("The separate original-environment Viz CPU fixture is absent")
    for field, relative in (
            ("original_processor_source_sha256", "src/kdm/models/backbone.py"),
            ("original_text_adapter_source_sha256", "src/kdm/models/hf.py"),
            ("original_operator_source_sha256", "workflows/supplemental/remaining4/ip_m3id_compatibility.py")):
        if source[field] != file_hash(ROOT / relative):
            raise ValueError("The actual original processor/operator source changed")
    previous_path = within(ROOT, source["previous_Food_software_gate_path"])
    previous = json.loads(previous_path.read_text())
    if (file_hash(previous_path) != source["previous_Food_software_gate_sha256"]
            or not previous["passed"] or not previous["production_allowed"] or previous["differences"]
            or previous["software_compatibility"]["marker_scope"] != [args.marker]
            or previous["software_compatibility"]["dataset_scope"] != "food101"):
        raise ValueError("The prior same-marker software proof changed; it is not a Viz admission")
    first_path = within(ROOT, source["first8_keys_path"])
    tail_path = within(ROOT, source["remaining_keys_path"])
    if (file_hash(first_path) != source["first8_keys_sha256"]
            or file_hash(tail_path) != source["remaining_keys_sha256"]):
        raise ValueError("The explicit disjoint Viz key ranges changed")
    first = list(read_jsonl(first_path))
    tail = list(read_jsonl(tail_path))
    keys = [row["key"] for row in first + tail]
    if len(first) != 8 or len(tail) != 3493 or len(keys) != len(set(keys)):
        raise ValueError("The genuine gate8 and remaining3493 are not the exact 3501-key partition")
    args.missing_keys = str((first_path if args.phase == "gate" else tail_path).relative_to(ROOT))
    execution.REGISTRY = str(within(ROOT, args.host_registry).relative_to(ROOT))
    import kdm.execution
    kdm.execution.REGISTRY = execution.REGISTRY
    for description in source["original_cpu_inputs"]:
        image_path = kdm.execution.resolve_image_path(description["image_source_path"], ROOT)
        if file_hash(image_path) != description["original_image_sha256"]:
            raise ValueError("The actual original Viz image bytes differ")
    original_spec = execution.runtime_spec

    def actual_spec(root, frozen, model):
        if model != "onevision":
            raise ValueError("This software registration is OneVision only")
        actual = original_spec(root, frozen, model)
        actual["versions"] = copy.deepcopy(frozen["versions"])
        actual["versions"].update(torch="2.9.0+cu128", torchvision="0.24.0+cu128")
        return actual

    execution.runtime_spec = actual_spec
    plan = generate.load_plan(args)
    registered.retained(plan)
    tasks = list(generate.selected_tasks(plan))
    if any(task["method"] != "instruction_m3id" or task["kind"] != "instruction_preserving"
           or task["marker"] != args.marker or task["reference_marker"] != args.marker
           or task["guided"] is not True or task["reference_guided"] is not False for task in tasks):
        raise ValueError("The canonical diagonal IP-M3ID tasks differ")
    spec = execution.runtime_spec(ROOT, plan["spec"], args.model)
    environment = validate_environment(spec)
    checkpoint = execution.checkpoint_identity(spec)
    provenance = {"model": args.model, "actual_method_scope": "instruction_m3id",
        "dataset_scope": args.dataset, "marker_scope": [args.marker],
        "original_torch": plan["spec"]["versions"]["torch"], "actual_torch": spec["versions"]["torch"],
        "transformers_and_other_registered_versions_unchanged": True,
        "noise_based_methods_admitted": False,
        "original_CPU_source_sha256": file_hash(source_path),
        "previous_Food_software_gate_sha256": file_hash(previous_path),
        "Food_proof_substitutes_for_Viz_gate": False,
        "original_Viz_same_answer_comparison_available": False,
        "entrypoint_sha256": file_hash(Path(__file__)),
        "observer_source_sha256": file_hash(ROOT / "workflows/supplemental/remaining4/ip_m3id_compatibility.py"),
        "registry_sha256": file_hash(within(ROOT, args.host_registry)),
        "gate_rows_are_new_formal_keys": args.phase == "gate"}
    gate_path = None
    if args.phase == "production":
        if not args.software_gate:
            raise ValueError("Production requires this Viz marker's actual eight-input gate")
        gate_path = within(ROOT, args.software_gate)
        gate = json.loads(gate_path.read_text())
        audit_path = within(ROOT, gate["operator_audit_path"])
        audit = json.loads(audit_path.read_text())
        if (not gate["passed"] or not gate["production_allowed"] or gate["completed_inputs"] != 8
                or gate["dataset"] != args.dataset or gate["marker"] != args.marker
                or gate["method"] != "instruction_m3id"
                or gate["original_CPU_source_sha256"] != file_hash(source_path)
                or gate["software_compatibility"]["entrypoint_sha256"] != provenance["entrypoint_sha256"]
                or gate["software_compatibility"]["registry_sha256"] != provenance["registry_sha256"]
                or file_hash(audit_path) != gate["operator_audit_sha256"]
                or not audit["passed"] or audit["completed"] != 8
                or not audit["three_original_routes_per_input"] or audit["noise_called"]):
            raise ValueError("The actual same-marker Viz input/operator gate is absent")
        provenance["actual_Viz_software_gate_sha256"] = file_hash(gate_path)
    elif (len(tasks) != 8 or len({task["sample"]["id"] for task in tasks}) != 8
          or {task["sample"]["id"] for task in tasks} != {r["sample_id"] for r in source["original_cpu_inputs"]}):
        raise ValueError("The actual Viz gate requires the original CPU eight distinct inputs")
    args.retained_source_ownership = {
        "dataset": args.dataset, "method": "instruction_m3id", "marker": args.marker,
        "new_actual_formal_keys": True, "original_CPU_source_path": args.source,
        "original_CPU_source_sha256": file_hash(source_path), "active_producer_overlap": 0,
        "first8_keys_sha256": source["first8_keys_sha256"],
        "remaining_keys_sha256": source["remaining_keys_sha256"], "scientific_parameters_changed": False}
    summary = {"plan": plan["summary"], "actual_environment": environment,
               "checkpoint": checkpoint, "software_compatibility": provenance,
               "gpu_model_admission": "not_run" if args.phase == "gate" else "actual_same_Viz_marker8_passed"}
    if args.check_plan:
        import torch
        summary["cuda_initialized"] = torch.cuda.is_initialized()
        if summary["cuda_initialized"]:
            raise ValueError("The CPU plan unexpectedly initialized CUDA")
    if args.plan_output:
        write(within(ROOT, args.plan_output), summary)
    if args.check_plan:
        print(json.dumps(summary), flush=True)
        return
    original_admission = execution.validate_supplemental_runtime

    def admitted(*values, **kwargs):
        result = original_admission(*values, **kwargs)
        result["software_compatibility"] = provenance
        return result

    execution.validate_supplemental_runtime = admitted
    os.environ.update(KDM_SUPPLEMENTAL_DATASET="vizwiz", KDM_SUPPLEMENTAL_METHOD="instruction_m3id",
        KDM_VIZ_IP_MARKER=args.marker, KDM_VIZ_IP_PHASE=args.phase, KDM_VIZ_IP_SOURCE=args.source)
    if gate_path is not None:
        os.environ["KDM_VIZ_IP_GATE"] = str(gate_path.relative_to(ROOT))
    observers = []
    if args.phase == "gate":
        original_backend = generate.make_backend

        def observed_backend(*values, **kwargs):
            observer = IPObserver(original_backend(*values, **kwargs), tasks, plan["cfg"], source["original_cpu_inputs"])
            observers.append(observer)
            return observer

        generate.make_backend = observed_backend
    out = within(ROOT, args.run_root) / "admission" / args.claim_id
    try:
        generate.execute(args)
        if args.phase == "gate":
            record = within(ROOT, args.run_root) / "records/onevision/vizwiz/formal" / args.claim_id
            receipt_path = record / "formal_onevision_part_00000.complete.json"
            receipt = json.loads(receipt_path.read_text())
            raw_path = within(ROOT, receipt["raw_path"])
            if not receipt["generation_complete"] or receipt["rows"] != 8 or file_hash(raw_path) != receipt["raw_sha256"]:
                raise ValueError("The actual Viz eight-input raw is not sealed")
            rows = list(read_jsonl(raw_path))
            if ({row["key"] for row in rows} != {row["key"] for row in first}
                    or any(row["config"]["max_tokens"] != 32 or not 1 <= len(row["tokens"]) <= 32
                           or not all(math.isfinite(v) for v in row["selected_log_probabilities"]) for row in rows)):
                raise ValueError("The actual Viz keys, token budget or probabilities differ")
            audit = observers[0].proof(rows)
            import torch
            audit.update(dataset="vizwiz", marker=args.marker,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            audit_path = record / "ip_operator_audit.json"
            write(audit_path, audit)
            result = {"passed": True, "production_allowed": True, "dataset": args.dataset,
                "model": args.model, "method": "instruction_m3id", "marker": args.marker,
                "original_CPU_source_sha256": file_hash(source_path),
                "operator_audit_path": str(audit_path.relative_to(ROOT)),
                "operator_audit_sha256": file_hash(audit_path),
                "software_compatibility": provenance, "completed_inputs": len(rows),
                "wall_s_generation": audit["gate_generation_wall_s"],
                "output_tokens": sum(len(row["tokens"]) for row in rows),
                "raw_path": str(raw_path.relative_to(ROOT)), "raw_sha256": receipt["raw_sha256"],
                "complete_receipt_sha256": file_hash(receipt_path),
                "completed_utc": datetime.now(timezone.utc).isoformat()}
            write(out / "success.json", result)
            print(json.dumps(result), flush=True)
    except BaseException as error:
        write(out / "failure.json", {"error": type(error).__name__ + ": " + str(error),
            "traceback": traceback.format_exc(), "production_allowed": False,
            "automatic_retry": False, "affected_marker": args.marker,
            "created_utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
