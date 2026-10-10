"""Exclusive original-host admission for core4 native M3ID fixed VizWiz512."""
from __future__ import annotations
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
MODEL_CARDS = {"qwen25vl": "0", "qwen35_4b": "1", "minicpm26": "4", "gemma3_4b": "5"}
GPU_UUIDS = {"0": "GPU-5a0d331f-afba-237f-fde6-762a8b1ff2f3", "1": "GPU-c6e6bac7-6035-192a-1e14-0c49903fbc4e",
             "4": "GPU-8d98e481-b89b-2166-b474-c33d59e0fd3d", "5": "GPU-a56bc7bd-eeab-b2a3-2d92-9f27816ffe6d"}
MINIMUM_FREE_MIB = 24000


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def host_registration(root):
    root = Path(root).resolve()
    if (root != ROOT.resolve() or root != Path("/home/g203-4028/projects/knowledge-deficit-mitigation")
            or socket.gethostname() != "4028" or os.getuid() != 1000
            or REGISTRY != "outputs/supplemental/native_m3id_viz_20261010/host_registry.json"):
        raise ValueError("Only this original registered 4028 account/project is admitted")
    original = read(root / "configs/runtime/hosts.json")
    registry = read(root / REGISTRY)
    host = registry["hosts"].get("4028")
    if (original["hosts"]["4028"]["allowed_gpus"] != [0, 1, 4, 5]
            or original["hosts"]["4028"]["gpu_uuids"] != GPU_UUIDS
            or any(original["model_hosts"].get(m) != "4028" for m in MODEL_CARDS)
            or set(registry["hosts"]) != {"4028"} or not isinstance(host, dict)
            or host["hostname"] != "4028" or Path(host["root"]) != root
            or host["allowed_gpus"] != [0, 1, 4, 5] or host["gpu_uuids"] != GPU_UUIDS
            or host["max_workers_per_gpu"] != 1
            or host["minimum_free_mib_by_model"] != {m: MINIMUM_FREE_MIB for m in MODEL_CARDS}
            or registry["model_hosts"] != {m: "4028" for m in MODEL_CARDS}
            or registry.get("model_cards") != MODEL_CARDS):
        raise ValueError("Original card authorization or this exclusive model/card registration differs")
    return registry, "4028", host


def validate_source(root, spec, model):
    source = read(OUT / "SOURCE.json")
    binding = read(root / source["fixed_sample_ids_file"])
    if (source.get("schema") != "kdm_native_m3id_fixed512_vizwiz_4028_core4_v1"
            or source.get("models") != list(MODEL_CARDS) or source.get("model_cards") != MODEL_CARDS
            or source.get("dataset") != "vizwiz" or source.get("split") != "eval"
            or source.get("rows_per_model") != 512 or source.get("expected_rows") != 2048
            or source.get("batch_size") != 1 or source.get("retained_pilot_rows") != 8
            or source.get("new_calibration") is not False or source.get("new_reference_gt") is not False
            or source.get("guidance_matrix") is not False or source.get("automatic_retry") is not False
            or source.get("automatic_parameter_changes") is not False
            or source["frozen_specs"].get(model) != spec
            or source["base_decode_config"] != asdict(DecodeConfig(method="m3id"))):
        raise ValueError("Only this unchanged core4 fixed512 native M3ID registration is admitted")
    ids = [s["id"] for s in source["samples"]]
    if (len(ids) != 512 or len(set(ids)) != 512 or ids != binding["sample_ids"]
            or any(s.get("dataset") != "vizwiz" or s.get("split") != "eval" for s in source["samples"])
            or source.get("eval_id_sha256") != stable_hash(ids)
            or source.get("fixed512_binding_sha256") != file_hash(root / source["fixed_sample_ids_file"])
            or source.get("all_vizwiz_images_exact_byte_verified") != 512
            or source.get("image_catalog_sha256") != file_hash(root / "outputs/records/image_content_catalog.json")
            or source.get("admission_registry_sha256") != file_hash(root / REGISTRY)):
        raise ValueError("Fixed input/image/source registration differs")
    expected_task = dict(method="m3id", kind="native_unguided", marker="NONE",
                         reference_marker="NONE", guided=False, reference_guided=False, replicate=0)
    if source.get("task") != expected_task:
        raise ValueError("Task is not the registered unguided native M3ID condition")
    for name, expected in {**source["source_hashes"], **source["input_hashes"]}.items():
        if file_hash(root / name) != expected:
            raise ValueError("Registered source or input changed: " + name)
    if str(Path(__file__).relative_to(root)) not in source["source_hashes"]:
        raise ValueError("Workflow-local admission source lacks its registered SHA")
    capacity = source["native_capacity_evidence"][model]
    proof_path = root / capacity["canonical_path"]
    proof = read(proof_path)
    if (file_hash(proof_path) != capacity["sha256"] or proof.get("passed") is not True
            or proof.get("completed") != 16 or proof.get("expected") != 16
            or capacity.get("admission_free_mib") != MINIMUM_FREE_MIB):
        raise ValueError("Original native16 capacity/interface evidence changed")
    return source


def validate_supplemental_runtime(root, spec, model, cards, stage, claim_id, owner):
    root = Path(root).resolve()
    cards = [str(c) for c in cards]
    registry, host, details = host_registration(root)
    if (model not in MODEL_CARDS or cards != [MODEL_CARDS[model]] or stage != "formal"
            or claim_id != "native_m3id_viz_20261010" or owner != "/root"
            or spec["gpu_count"] != 1 or spec["dtype"] != "bfloat16"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != MODEL_CARDS[model]
            or os.environ.get("KDM_SUPPLEMENTAL_DATASET") != "vizwiz"
            or os.environ.get("KDM_SUPPLEMENTAL_METHOD") != "m3id"
            or os.environ.get("KDM_GPU_SLOTS", "")):
        raise ValueError("Only this model's fixed original physical card/native task is admitted")
    source = validate_source(root, spec, model)
    card = cards[0]
    expected_lock = root / "outputs/locks" / f"gpu_{card}.lock"
    if Path(os.readlink("/proc/self/fd/20")) != expected_lock:
        raise ValueError("Missing inherited FD20 lock for the assigned original GPU")
    actual_lock, path_lock = os.fstat(20), expected_lock.stat()
    if (actual_lock.st_dev, actual_lock.st_ino) != (path_lock.st_dev, path_lock.st_ino):
        raise ValueError("Inherited GPU lock inode differs")
    fcntl.flock(20, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lines = subprocess.check_output(["nvidia-smi", "-i", card,
        "--query-gpu=index,uuid,name,memory.total,memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()
    observed = {p[0].strip(): dict(uuid=p[1].strip(), name=p[2].strip(),
                    total_mib=int(p[3].strip()), free_mib=int(p[4].strip())) for p in (line.split(",") for line in lines)}
    if (set(observed) != {card} or observed[card]["uuid"] != GPU_UUIDS[card]
            or observed[card]["name"] != "NVIDIA GeForce RTX 3090"
            or observed[card]["total_mib"] != 24576 or observed[card]["free_mib"] < MINIMUM_FREE_MIB):
        raise ValueError("Original registered RTX3090 or exclusive idle-card capacity budget differs")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True).splitlines()
    for line in apps:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == GPU_UUIDS[card] and int(parts[1]) != os.getpid():
            raise ValueError("Assigned exclusive original GPU is occupied by another process")
    override = registry["runtime_overrides"][host][model]
    if (set(override) != {"environment_python", "model_path", "source_evidence"}
            or override["environment_python"] != spec["environment_python"]
            or override["model_path"] != spec["kwargs"]["model_path"]):
        raise ValueError("Original frozen environment or checkpoint paths were changed")
    environment = validate_environment(spec)
    checkpoint = checkpoint_identity(spec)
    import kdm.execution
    kdm.execution.REGISTRY = REGISTRY
    receipt = dict(host=host, hostname=details["hostname"], project_root=str(root),
        physical_gpus=cards, gpu_observation_before_loading=observed,
        registry_path=REGISTRY, registry_sha256=file_hash(root / REGISTRY),
        admission_source_sha256=file_hash(Path(__file__)), model=model, stage=stage,
        claim_id=claim_id, owner=owner, environment=environment, checkpoint=checkpoint,
        gpu_sharing_authorized=False, gpu_worker_slots="", max_workers_per_gpu=1,
        registered_idle_card_minimum_free_mib=MINIMUM_FREE_MIB,
        capacity_evidence=source["native_capacity_evidence"][model],
        fixed512_source_sha256=file_hash(OUT / "SOURCE.json"),
        fixed512_binding_sha256=source["fixed512_binding_sha256"],
        runtime_path_evidence=override,
        shared_checkpoint_validator_source_sha256=file_hash(root / "workflows/supplemental/remaining11/execution.py"))
    return {"execution": receipt, "runtime_spec": spec}
