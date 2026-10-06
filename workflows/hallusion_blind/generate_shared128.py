#!/usr/bin/env python3
"""Explicit K100 two-slot admission; frozen Hall128 generation stays unchanged."""
from __future__ import annotations
import copy, fcntl, importlib.metadata, json, os, socket, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from kdm.io import file_hash
from kdm.protocol import validate_environment
from workflows.supplemental.remaining11.execution import checkpoint_identity
from workflows.hallusion_blind import generate128 as frozen
from workflows.general_vqa_direct.generate import relative

ENTRY = "workflows/hallusion_blind/generate_shared128.py"
PACKAGES = ("torch", "torchvision", "transformers", "accelerate", "Pillow",
            "tokenizers", "numpy", "sentencepiece", "timm", "safetensors",
            "einops", "scipy", "requests")


def versions13():
    values = {}
    for name in PACKAGES:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def lock_identity(path):
    st = path.stat()
    return f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"


def validate_lock_records(lines, pid, main_identity, slot_identity):
    records = [line.split() for line in lines]
    def matches(identity, access):
        return any(len(row) >= 8 and row[1:4] == ["FLOCK", "ADVISORY", access]
                   and row[4] == str(pid) and row[5] == identity
                   and row[6:8] == ["0", "EOF"] for row in records)
    if not matches(main_identity, "READ") or not matches(slot_identity, "WRITE"):
        raise ValueError("Shared main lock and exclusive slot lock are both required")
    return {"shared_main": True, "exclusive_slot": True}


def check_process_locks(pid, card, slot, root=ROOT):
    main = root / f"outputs/locks/gpu_{card}.lock"
    slot_path = root / f"outputs/locks/gpu_{card}_slot_{slot}.lock"
    for fd, expected in ((20, main), (100, slot_path)):
        if Path(os.readlink(f"/proc/{pid}/fd/{fd}")) != expected:
            raise ValueError("Actual inherited GPU/slot file descriptor differs")
    proof = validate_lock_records(Path("/proc/locks").read_text().splitlines(),
                                  pid, lock_identity(main), lock_identity(slot_path))
    return {**proof, "main_path": str(main), "slot_path": str(slot_path)}


def prioritize(selected, priority):
    if len(priority) != len(set(priority)) or not set(priority) <= set(selected):
        raise ValueError("Priority keys must be unique members of the assigned shard")
    return priority + [key for key in selected if key not in set(priority)]


_original_load_plan = frozen.load_plan
def load_plan(args, root=ROOT):
    plan = _original_load_plan(args, root)
    if args.task_keys:
        assignment = json.loads(relative(root, args.task_keys).read_text())
        plan["selected"] = prioritize(plan["selected"], assignment.get("priority_keys", []))
    return plan


def shared_admit(args, plan, root=ROOT):
    registry_path = relative(root, args.registry)
    registry = json.loads(registry_path.read_text())
    matches = [(name, row) for name, row in registry["hosts"].items()
               if row["hostname"] == socket.gethostname() and Path(row["root"]) == root]
    if len(matches) != 1 or matches[0][0] != "k100":
        raise ValueError("This explicit shared entry is admitted only on K100")
    host, details = matches[0]
    sharing = registry["shared_gpu_admission"]
    slot = sharing["slots_by_model"].get(args.model)
    if (args.model not in {"internvl35_8b", "qwen35_4b"} or slot not in (0, 1)
        or args.cards != "0" or details["allowed_gpus"] != [0]
        or details.get("max_workers_per_gpu") != 2
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or os.environ.get("KDM_GPU_SLOTS") != f"0:{slot}"
        or os.environ.get("KDM_REQUEST_SLOT") != str(slot)
        or sharing["entry_path"] != ENTRY
        or sharing["entry_sha256"] != file_hash(root / ENTRY)):
        raise ValueError("Explicit physical card, two-worker registry and actual slot differ")
    lock_proof = check_process_locks(os.getpid(), "0", slot, root)
    if versions13() != sharing["original_versions13"]:
        raise ValueError("The original thirteen runtime package states differ")
    if plan["definition"]["source_sha256"] != sharing["frozen_source15"]:
        raise ValueError("The fifteen frozen generation/model sources differ")
    actual = copy.deepcopy(plan["spec"])
    gate_receipt = None
    if args.k100_intern:
        if args.model != "internvl35_8b":
            raise ValueError("The registered K100 Intern map is model-specific")
        from workflows.supplemental.remaining4.k100_intern_registered_matrix import single_spec
        declaration = registry["internvl_single_gate"]
        gate_path = relative(root, declaration["path"])
        gate = json.loads(gate_path.read_text())
        if (file_hash(gate_path) != declaration["sha256"] or not gate["passed"]
            or not gate["production_allowed"] or gate["completed"] != 8
            or file_hash(root / "workflows/supplemental/remaining4/internvl_k100_single.py") != gate["factory_sha256"]
            or file_hash(relative(root, gate["operator_audit_path"])) != gate["operator_audit_sha256"]):
            raise ValueError("Original K100 single-map admission differs")
        actual = single_spec(actual)
        gate_receipt = {**declaration, "scope": "unchanged registered single-device map"}
    elif args.model == "internvl35_8b":
        raise ValueError("Intern must retain its registered K100 single-device map")
    override = registry["runtime_overrides"][host][args.model]
    if set(override) != {"environment_python", "model_path", "source_evidence"}:
        raise ValueError("Only original checkpoint/environment paths may move")
    actual["environment_python"] = override["environment_python"]
    actual["kwargs"]["model_path"] = override["model_path"]
    actual["processor"]["path"] = override["model_path"]
    if actual["gpu_count"] != 1 or actual["dtype"] != "bfloat16":
        raise ValueError("Original single-device BF16 runtime is required")
    environment, checkpoint = validate_environment(actual), checkpoint_identity(actual)
    card, uuid, free = [v.strip() for v in subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=index,uuid,memory.free",
         "--format=csv,noheader,nounits"], text=True).strip().split(",")]
    minimum = details["minimum_free_mib_by_model"][args.model]
    if card != "0" or uuid != details["gpu_uuids"]["0"] or int(free) < minimum:
        raise ValueError("Actual UUID or free-capacity admission failed")
    process_rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
         "--format=csv,noheader,nounits"], text=True).splitlines()
    peers = []
    for line in process_rows:
        gpu, process, memory = [v.strip() for v in line.split(",")]
        pid = int(process)
        if gpu != uuid or pid == os.getpid():
            continue
        cmd = (Path("/proc") / process / "cmdline").read_bytes().split(b"\0")
        argv = [v.decode() for v in cmd if v]
        peer = next((m for m, s in sharing["slots_by_model"].items() if s != slot), None)
        if ("--model" not in argv or argv[argv.index("--model") + 1] != peer
            or "--registry" not in argv or relative(root, argv[argv.index("--registry") + 1]) != registry_path
            or not any(a.endswith(ENTRY) for a in argv)):
            raise ValueError("GPU has an unregistered shared process")
        peer_locks = check_process_locks(pid, "0", sharing["slots_by_model"][peer], root)
        stats = (Path("/proc") / process / "stat").read_text().rsplit(")", 1)[1].split()
        peers.append({"pid": pid, "start_tick": int(stats[19]), "model": peer,
                      "used_gpu_memory_mib": memory, "locks": peer_locks, "argv": argv})
    if len(peers) > 1:
        raise ValueError("Declared two-slot GPU already has excess peers")
    return actual, {"host": host, "project_root": str(root), "physical_gpus": ["0"],
        "observed_gpus": {"0": {"uuid": uuid, "free_mib": int(free)}},
        "minimum_free_mib": minimum, "runtime_spec": actual, "environment": environment,
        "checkpoint": checkpoint, "registry": str(registry_path.relative_to(root)),
        "registry_sha256": file_hash(registry_path), "runtime_path_evidence": override,
        "internvl_single_gate": gate_receipt, "shared": True, "gpu_slot": slot,
        "locks": lock_proof, "actual_peers": peers, "original_versions13": versions13(),
        "entry_path": ENTRY, "entry_sha256": file_hash(root / ENTRY),
        "sharing_authorization": sharing["authorization"],
        "ownership_policy": "shared main flock plus one exclusive registered slot",
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


if __name__ == "__main__":
    frozen.admit = shared_admit
    frozen.load_plan = load_plan
    raise SystemExit(frozen.main())
