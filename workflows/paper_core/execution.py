"""Core-experiment admission; frozen scientific and runtime registrations stay intact."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess

from kdm.frozen import load_contract
from kdm.io import file_hash
from kdm.protocol import validate_environment
from workflows.supplemental.remaining11.execution import checkpoint_identity
from workflows.supplemental.remaining11.generate import validate_proofs

REGISTRY = "workflows/paper_core/host_registry.json"


def registration(root: Path):
    root = Path(root).resolve()
    registry = json.loads((root / REGISTRY).read_text())
    matches = [(key, details) for key, details in registry["hosts"].items()
               if details["hostname"] == socket.gethostname() and Path(details["root"]) == root]
    if len(matches) != 1:
        raise ValueError("Core hostname/project root is not uniquely registered")
    return registry, *matches[0]


def runtime_spec(root: Path, frozen_spec: dict, model: str) -> dict:
    registry, host, _ = registration(root)
    if frozen_spec["key"] != model or model not in registry["models"]:
        raise ValueError("Core runtime model differs from the frozen registration")
    actual = copy.deepcopy(frozen_spec)
    override = registry["runtime_overrides"][host][model]
    if set(override) != {"environment_python", "model_path", "source_evidence"}:
        raise ValueError("A core runtime move may change only environment and checkpoint paths")
    actual["environment_python"] = override["environment_python"]
    actual["kwargs"]["model_path"] = override["model_path"]
    actual["processor"]["path"] = override["model_path"]
    return actual


def admit(root: Path, frozen_spec: dict, model: str, cards: list[str]) -> dict:
    root = Path(root).resolve()
    registry, host, details = registration(root)
    if len(cards) != frozen_spec["gpu_count"] or len(set(cards)) != len(cards):
        raise ValueError("Core physical GPU count differs from the registered device map")
    if not set(cards) <= {str(card) for card in details["allowed_gpus"]}:
        raise ValueError("Unauthorized core physical GPU")
    for offset, card in enumerate(sorted(cards, key=int)):
        if Path(os.readlink(f"/proc/self/fd/{20 + offset}")) != root / "outputs/locks" / f"gpu_{card}.lock":
            raise ValueError("Core worker physical GPU lock is missing")
    observed = {}
    for line in subprocess.check_output([
        "nvidia-smi", "-i", ",".join(cards), "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True).splitlines():
        index, uuid, free = [field.strip() for field in line.split(",")]
        observed[index] = {"uuid": uuid, "free_mib": int(free)}
    if set(observed) != set(cards):
        raise ValueError("Core GPU observation is incomplete")
    for card in cards:
        if observed[card]["uuid"] != details["gpu_uuids"][card]:
            raise ValueError("Core GPU UUID changed")
        if observed[card]["free_mib"] < registry["minimum_free_mib"][model]:
            raise ValueError("Core model lacks its unchanged-runtime memory budget")
    manifest, freeze = load_contract(root)
    proofs = validate_proofs(root, frozen_spec, model, ["vcd"], "formal", manifest, freeze)
    spec = runtime_spec(root, frozen_spec, model)
    environment = validate_environment(spec)
    checkpoint = checkpoint_identity(spec)
    fixed = {
        "registry_sha256": file_hash(root / REGISTRY),
        "admission_source_sha256": file_hash(Path(__file__)),
        "freeze_contract_sha256": manifest["original_contract_sha256"],
        "proofs": proofs, "model": model, "environment": environment,
        "checkpoint": checkpoint, "runtime_path_evidence": registry["runtime_overrides"][host][model],
    }
    # Only this process uses the new image-location registry; the frozen file stays intact.
    import kdm.execution
    kdm.execution.REGISTRY = REGISTRY
    return {
        "fixed_identity": fixed, "host": host, "project_root": str(root),
        "physical_gpus": cards, "gpu_observation_before_loading": observed,
        "worker_slots": os.environ.get("KDM_GPU_SLOTS", ""),
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
