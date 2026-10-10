"""Exclusive admission for this fixed-512 LLaVA native M3ID K100 task.

This workflow-local registration preserves the shared admission module and its
earlier VizWiz scopes. Environment and checkpoint checks reuse that module.
The model operator and retained first-eight-input audit remain in run.py.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess

from kdm.decoding import DecodeConfig
from kdm.io import file_hash, stable_hash
from kdm.protocol import validate_environment
from workflows.supplemental.remaining11.execution import checkpoint_identity

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows/supplemental/native_m3id_viz_20261010"
OUT = ROOT / "outputs/supplemental/native_m3id_viz_20261010"
REGISTRY = "outputs/supplemental/native_m3id_viz_20261010/host_registry.json"
MODEL = "llava16_mistral"
HOST = "k100"
UUID = "GPU-4320e83b-1158-0546-73c5-b715c4dbe1ea"
MINIMUM_FREE_MIB = 32768
AUTHOR = {"agent": "/root/guided_std_audit", "role": "Codex implementation agent",
          "model_identifier": None, "reasoning_effort": None}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def host_registration(root):
    root = Path(root).resolve()
    if (root != ROOT.resolve()
            or root != Path("/home/k100/projects/knowledge-deficit-mitigation")
            or socket.gethostname() != "k100-X785-H30" or os.getuid() != 1000
            or REGISTRY != "outputs/supplemental/native_m3id_viz_20261010/host_registry.json"):
        raise ValueError("Only this registered K100 project, account and namespace are admitted")
    registry = read(root / REGISTRY)
    host = registry["hosts"].get(HOST)
    if (set(registry["hosts"]) != {HOST} or not isinstance(host, dict)
            or host["hostname"] != socket.gethostname() or Path(host["root"]) != root
            or host["allowed_gpus"] != [0] or host["max_workers_per_gpu"] != 1
            or host["gpu_uuids"] != {"0": UUID}
            or host["minimum_free_mib_by_model"] != {MODEL: MINIMUM_FREE_MIB}
            or registry["model_hosts"] != {MODEL: HOST}):
        raise ValueError("Exclusive K100 model, GPU identity or memory budget differs")
    return registry, HOST, host


def runtime_spec(root, frozen_spec, model):
    root = Path(root).resolve()
    registry, host, _details = host_registration(root)
    if model != MODEL or frozen_spec["key"] != MODEL:
        raise ValueError("Only the registered LLaVA native model is admitted")
    override = registry.get("runtime_overrides", {}).get(host, {}).get(model)
    if (not isinstance(override, dict)
            or set(override) != {"environment_python", "model_path", "source_evidence"}
            or override["environment_python"] != str(root / ".environments/native311/bin/python")
            or override["model_path"] != str(root / "cache/models/formal_v2_llava16_mistral")):
        raise ValueError("K100 environment or checkpoint differs from the completed native Food runtime")
    spec = copy.deepcopy(frozen_spec)
    spec["environment_python"] = override["environment_python"]
    spec["kwargs"]["model_path"] = override["model_path"]
    spec["processor"]["path"] = override["model_path"]
    return spec


def validate_source(root, spec):
    source = read(OUT / "SOURCE.json")
    binding = read(WORKFLOW / "fixed512_sample_ids.json")
    if (source.get("schema") != "kdm_native_m3id_fixed512_vizwiz_k100_v1"
            or source.get("models") != [MODEL] or source.get("dataset") != "vizwiz"
            or source.get("split") != "eval" or source.get("rows_per_model") != 512
            or source.get("expected_rows") != 512 or source.get("batch_size") != 1
            or source.get("retained_pilot_rows") != 8
            or source.get("new_calibration") is not False
            or source.get("new_reference_gt") is not False
            or source.get("guidance_matrix") is not False
            or source.get("automatic_retry") is not False
            or source.get("automatic_parameter_changes") is not False
            or source["frozen_specs"].get(MODEL) != spec
            or source["base_decode_config"] != asdict(DecodeConfig(method="m3id"))):
        raise ValueError("This exact fixed-512 native M3ID task or frozen runtime is not registered")
    expected_task = dict(method="m3id", kind="native_unguided", marker="NONE",
                         reference_marker="NONE", guided=False,
                         reference_guided=False, replicate=0)
    samples = source["samples"]
    ids = [sample["id"] for sample in samples]
    if (source.get("task") != expected_task or len(ids) != 512 or len(set(ids)) != 512
            or ids != binding["sample_ids"]
            or any(sample.get("dataset") != "vizwiz" or sample.get("split") != "eval"
                   for sample in samples)
            or source.get("eval_id_sha256") != stable_hash(ids)
            or source.get("fixed512_binding_sha256") != file_hash(WORKFLOW / "fixed512_sample_ids.json")
            or source.get("all_vizwiz_images_exact_byte_verified") != 512
            or source.get("image_catalog_sha256") != file_hash(root / "outputs/records/image_content_catalog.json")
            or source.get("admission_registry_sha256") != file_hash(root / REGISTRY)):
        raise ValueError("Fixed input, image-content or admission registration differs")
    for path, expected in {**source["source_hashes"], **source["input_hashes"]}.items():
        if file_hash(root / path) != expected:
            raise ValueError("Registered source or input changed: " + path)
    if str(Path(__file__).relative_to(root)) not in source["source_hashes"]:
        raise ValueError("This workflow-local admission helper lacks a registered source SHA")
    return source


def validate_supplemental_runtime(root, spec, model, cards, stage, claim_id, owner):
    root = Path(root).resolve()
    cards = [str(card) for card in cards]
    registry, host, details = host_registration(root)
    if (model != MODEL or cards != ["0"] or stage != "formal"
            or claim_id != "native_m3id_viz_20261010" or owner != "/root"
            or spec["gpu_count"] != 1 or spec["dtype"] != "bfloat16"
            or spec["versions"]["torch"] != "2.9.0+cu128"
            or spec["versions"]["transformers"] != "5.17.0"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or os.environ.get("KDM_SUPPLEMENTAL_DATASET") != "vizwiz"
            or os.environ.get("KDM_SUPPLEMENTAL_METHOD") != "m3id"
            or os.environ.get("KDM_GPU_SLOTS", "")):
        raise ValueError("Only the exclusive unchanged LLaVA fixed-512 native M3ID task is admitted")
    source = validate_source(root, spec)
    expected_lock = root / "outputs/locks/gpu_0.lock"
    if Path(os.readlink("/proc/self/fd/20")) != expected_lock:
        raise ValueError("Missing inherited FD20 lock for physical K100 GPU 0")
    observed_lock, expected_stat = os.fstat(20), expected_lock.stat()
    if ((observed_lock.st_dev, observed_lock.st_ino)
            != (expected_stat.st_dev, expected_stat.st_ino)):
        raise ValueError("Inherited GPU lock file identity differs")
    fcntl.flock(20, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lines = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    observed = {parts[0].strip(): {"uuid": parts[1].strip(), "free_mib": int(parts[2].strip())}
                for parts in (line.split(",") for line in lines)}
    if (set(observed) != {"0"} or observed["0"]["uuid"] != UUID
            or observed["0"]["free_mib"] < MINIMUM_FREE_MIB):
        raise ValueError("Registered K100 GPU UUID or unchanged model memory budget differs")
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True).splitlines()
    for line in apps:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == UUID and int(parts[1]) != os.getpid():
            raise ValueError("The registered exclusive K100 GPU is occupied by another process")
    actual = runtime_spec(root, spec, model)
    environment = validate_environment(actual)
    checkpoint = checkpoint_identity(actual)
    import kdm.execution
    kdm.execution.REGISTRY = REGISTRY
    receipt = dict(host=host, hostname=details["hostname"], project_root=str(root),
        physical_gpus=cards, gpu_observation_before_loading=observed,
        registry_path=REGISTRY, registry_sha256=file_hash(root / REGISTRY),
        admission_source_sha256=file_hash(Path(__file__)), model=model, stage=stage,
        claim_id=claim_id, owner=owner, environment=environment, checkpoint=checkpoint,
        gpu_sharing_authorized=False, gpu_worker_slots="", max_workers_per_gpu=1,
        runtime_path_evidence=registry["runtime_overrides"][host][model],
        registered_model_minimum_free_mib=MINIMUM_FREE_MIB,
        fixed512_source_sha256=file_hash(OUT / "SOURCE.json"),
        fixed512_binding_sha256=source["fixed512_binding_sha256"],
        shared_checkpoint_validator_source_sha256=file_hash(
            root / "workflows/supplemental/remaining11/execution.py"), authors=[AUTHOR])
    return {"execution": receipt, "runtime_spec": actual}
