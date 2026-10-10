#!/usr/bin/env python3
"""Register four unchanged native M3ID conditions on the original 4028 route."""
from __future__ import annotations
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import subprocess
import run


def registered_environment(root, spec):
    """Run the unchanged package/path check in this model's actual Python."""
    code = """import json,sys
from kdm.protocol import validate_environment
spec=json.load(sys.stdin)
result=validate_environment(spec)
import torch
if torch.cuda.is_initialized():
    raise ValueError('CPU environment check initialized a CUDA context')
result['cuda_initialized']=False
print(json.dumps(result))
"""
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root / "src") + ":" + str(root),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    checked = subprocess.run([spec["environment_python"], "-c", code],
        input=json.dumps(spec), cwd=root, env=env, capture_output=True, text=True, timeout=40)
    if checked.returncode:
        raise ValueError("Registered CPU environment check failed for " + spec["key"] + ": " + checked.stderr.strip())
    result = json.loads(checked.stdout)
    if (result.get("environment_python") != spec["environment_python"]
            or result.get("versions") != spec["versions"] or result.get("cuda_initialized") is not False):
        raise ValueError("Registered CPU environment evidence differs: " + spec["key"])
    return result


def prepare(sample_ids_path=None):
    from kdm.decoding import DecodeConfig
    from kdm.frozen import frozen_path, load_contract
    from kdm.io import file_hash, read_jsonl, stable_hash
    from workflows.supplemental.remaining11.execution import checkpoint_identity
    from workflows.supplemental.remaining11.generate import validate_proofs
    import native_4028_admission as admission
    root, out, workflow = run.ROOT, run.OUT, run.WORKFLOW
    if (socket.gethostname() != "4028" or os.getuid() != 1000
            or root != Path("/home/g203-4028/projects/knowledge-deficit-mitigation")):
        raise ValueError("Only the registered original 4028 project/account is selected")
    if out.exists():
        raise FileExistsError("Existing output preserved; no automatic replacement")
    manifest, frozen = load_contract(root)
    paths = ("data/current/all.jsonl", "configs/kdm/study.json", "configs/kdm/method_plan.json")
    inputs = (*paths, *(f"configs/runtime/{m}.json" for m in run.MODELS))
    for name in inputs:
        if frozen["files"].get(name) != file_hash(root / name):
            raise ValueError("Frozen input changed: " + name)
    study = run.read(root / paths[1])
    fixed = {"max_new_tokens": 32, "greedy": True, "m3id_lambda": .02, "m3id_threshold": .3}
    if any(study.get(key) != value for key, value in fixed.items()):
        raise ValueError("Registered M3ID parameters differ")
    roster_path = Path(sample_ids_path or workflow / "fixed512_sample_ids.json").resolve()
    if not roster_path.is_relative_to(root) or not roster_path.is_file():
        raise ValueError("Fixed roster must be project-contained")
    samples = run.select_samples(read_jsonl(root / paths[0]), run.read(roster_path))
    methods = run.read(root / paths[2])
    specs = {m: run.read(root / f"configs/runtime/{m}.json") for m in run.MODELS}
    original_registry = root / "configs/runtime/hosts.json"
    registry = run.read(original_registry)
    if (registry["hosts"]["4028"]["allowed_gpus"] != [0, 1, 4, 5]
            or any(registry["model_hosts"].get(m) != "4028" for m in run.MODELS)):
        raise ValueError("Original authorized cards or native model host differs")
    host = copy.deepcopy(registry["hosts"]["4028"])
    if host["gpu_uuids"] != admission.GPU_UUIDS:
        raise ValueError("Original registered 4028 GPU UUIDs differ")
    capacity = {}
    environment_checks, checkpoint_checks, inherited_proofs = {}, {}, {}
    for model, spec in specs.items():
        if (spec["gpu_count"] != 1 or spec["dtype"] != "bfloat16" or spec.get("api")
                or "m3id" not in methods[model]["vizwiz"]):
            raise ValueError("Native model support or precision differs: " + model)
        inherited_proofs[model] = validate_proofs(root, spec, model, ["m3id"], "formal", manifest, frozen)
        environment_checks[model] = registered_environment(root, spec)
        checkpoint_checks[model] = checkpoint_identity(spec)
        original_proof = spec["interface_verification"]["record"]
        proof_path = frozen_path(root, original_proof)
        proof = run.read(proof_path)
        if proof.get("passed") is not True or proof.get("completed") != 16 or proof.get("expected") != 16:
            raise ValueError("Original native 16-input capacity/interface proof is incomplete")
        peaks = proof.get("peak_allocated_bytes")
        capacity[model] = dict(original_native16_path=original_proof,
            canonical_path=str(proof_path.relative_to(root)), sha256=file_hash(proof_path),
            passed_native16=True, original_physical_gpus=proof.get("physical_gpus"),
            peak_allocated_bytes=peaks if isinstance(peaks, dict) else None,
            measured_minimum_native_m3id_memory=False,
            maximum_fixed512_native_m3id_peak_measured=False,
            admission_free_mib=admission.MINIMUM_FREE_MIB,
            admission_basis="Exclusive original 24-GiB RTX3090 route; at least 24000 MiB available. Original frozen native16 proof passed; Q25/Q35 recorded allocation peaks, Mini/Gemma did not. This is an idle-card admission budget, not a measured minimum memory requirement for all 512 inputs.")
    host.update(max_workers_per_gpu=1, minimum_free_mib_by_model={m: admission.MINIMUM_FREE_MIB for m in run.MODELS})
    prefixes = copy.deepcopy(registry["image_prefixes"])
    if prefixes.setdefault(run.VIZ_IMAGE_PREFIX, run.VIZ_IMAGE_DESTINATION) != run.VIZ_IMAGE_DESTINATION:
        raise ValueError("Original VizWiz image mapping differs")
    catalog_path = root / "outputs/records/image_content_catalog.json"
    if file_hash(catalog_path) != frozen["files"].get("outputs/records/image_content_catalog.json"):
        raise ValueError("Frozen image catalog identity differs")
    catalog = run.read(catalog_path)
    if catalog["manifest_sha256"] != file_hash(root / paths[0]):
        raise ValueError("Image catalog does not bind current all.jsonl")
    for sample in samples:
        logical = Path(sample["image_path"])
        matches = [(source, destination) for source, destination in prefixes.items()
                   if logical.is_relative_to(Path(source))]
        if len(matches) != 1:
            raise ValueError("Input image lacks a unique registered mapping")
        prefix, destination = matches[0]
        image_dir = (root / destination).resolve()
        image = (image_dir / logical.relative_to(prefix)).resolve()
        expected = catalog["images"][str(logical)]
        if (not image.is_relative_to(image_dir) or not image.is_relative_to(root) or not image.is_file()
                or image.stat().st_size != expected["size_bytes"] or file_hash(image) != expected["sha256"]):
            raise ValueError("Fixed VizWiz image byte identity differs: " + sample["id"])
    new_registry = dict(schema=1, purpose="Original 4028 core4 native M3ID on fixed512; unchanged frozen model runtimes, exclusive authorized original RTX3090 cards",
        hosts={"4028": host}, model_hosts={m: "4028" for m in run.MODELS},
        runtime_overrides={"4028": {m: dict(environment_python=specs[m]["environment_python"],
            model_path=specs[m]["kwargs"]["model_path"], source_evidence="Exact original frozen 4028 environment/checkpoint path; CPU environment, processor/config SHA and weight sizes verified during this prepare") for m in run.MODELS}},
        image_prefixes=prefixes, image_catalog=str(catalog_path.relative_to(root)),
        model_cards=admission.MODEL_CARDS, capacity_evidence=capacity)
    source_files = sorted((root / "src/kdm").rglob("*.py"))
    source_files += [root / "workflows/supplemental/remaining11/generate.py",
                     root / "workflows/supplemental/remaining11/execution.py",
                     root / "workflows/supplemental/remaining4/native_audit.py",
                     root / "workflows/paper_core/native_audit.py", *sorted(workflow.glob("*"))]
    source_hashes = {str(p.relative_to(root)): file_hash(p) for p in source_files if p.is_file()}
    input_hashes = {name: file_hash(root / name) for name in inputs}
    input_hashes.update({str(roster_path.relative_to(root)): file_hash(roster_path),
                        str(catalog_path.relative_to(root)): file_hash(catalog_path),
                        str(original_registry.relative_to(root)): file_hash(original_registry)})
    input_hashes.update({item["canonical_path"]: item["sha256"] for item in capacity.values()})
    out.mkdir(parents=True)
    for name in ("models", "logs"):
        (out / name).mkdir()
    run.write(out / "host_registry.json", new_registry)
    registration = dict(schema="kdm_native_m3id_fixed512_vizwiz_4028_core4_v1", created_utc=run.now(),
        models=list(run.MODELS), dataset="vizwiz", split="eval", rows_per_model=512, expected_rows=2048,
        task={k: v for k, v in run.native_task(samples[0]).items() if k != "sample"}, samples=samples,
        eval_id_sha256=stable_hash([s["id"] for s in samples]), frozen_specs=specs,
        base_decode_config=asdict(DecodeConfig(method="m3id")), frozen_contract_sha256=manifest["original_contract_sha256"],
        input_hashes=input_hashes, source_hashes=source_hashes,
        fixed512_binding_sha256=file_hash(roster_path), fixed_sample_ids_file=str(roster_path.relative_to(root)),
        admission_registry_sha256=file_hash(out / "host_registry.json"),
        original_host_registry_sha256=file_hash(original_registry), image_catalog_sha256=file_hash(catalog_path),
        all_vizwiz_images_exact_byte_verified=512, model_cards=admission.MODEL_CARDS,
        native_capacity_evidence=capacity, cpu_environment_checks=environment_checks,
        cpu_checkpoint_checks=checkpoint_checks, inherited_native_proofs=inherited_proofs,
        clock="Existing project native M3ID transfer: zero-based generated-token index plus tokenizer length of unguided plain task_prompt; actual tokenizer prompt length is recorded per input",
        retained_pilot_rows=8, batch_size=1, model_runtime_changed=False, new_calibration=False,
        new_reference_gt=False, guidance_matrix=False, new_vizwiz_generation=True,
        automatic_retry=False, automatic_parameter_changes=False, authors=[run.AUTHOR],
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    run.write(out / "SOURCE.json", registration)
    run.write(out / "CURRENT_STATE.json", dict(status="prepared", completed=0, expected=2048, updated_utc=run.now()))
    print(__import__("json").dumps(dict(status="prepared", models=list(run.MODELS), expected_rows=2048,
        source_sha256=file_hash(out / "SOURCE.json"), fixed512_images_verified=512,
        native_capacity_evidence=capacity)), flush=True)


if __name__ == "__main__":
    prepare()
