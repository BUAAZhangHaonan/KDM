"""Scientific provenance validation helpers. No job-launching entry point."""

from __future__ import annotations

import sys

from collections import Counter

from dataclasses import asdict

import json

import math

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from kdm.decoding import DecodeConfig

from kdm.io import file_hash, read_jsonl, stable_hash, stable_seed, within

from kdm.pipeline import task_id

from kdm.prompts import task_prompt

MODELS = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b")

def source_paths(folder, model):
    paths = {"raw": folder / "raw" / f"{model}_native_vcd.jsonl",
             "raw_identity": folder / "raw" / f"{model}_native_vcd.identity.json",
             "identity": folder / f"{model}.identity.json",
             "gate": folder / f"{model}.gate_8.json", "complete": folder / f"{model}.complete.json",
             "admission_8": folder / f"{model}.admission_8.json",
             "admission_2424": folder / f"{model}.admission_2424.json"}
    replay = folder / f"{model}.replay_audit.json"
    if replay.is_file():
        paths["replay_audit"] = replay
    return paths

def validate_received(folder, model, source):
    paths = source_paths(folder, model)
    if source["model"] != model or set(paths) != set(source["files"]):
        raise ValueError("Received source model or file roles differ from the source snapshot")
    for role, path in paths.items():
        expected = source["files"][role]
        if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError("Received immutable file bytes differ: " + role)
    identity = json.loads(paths["identity"].read_text())
    gate = json.loads(paths["gate"].read_text())
    complete = json.loads(paths["complete"].read_text())
    sidecar = json.loads(paths["raw_identity"].read_text())
    cfg = asdict(DecodeConfig(method="vcd"))
    expected_identity = stable_hash(identity)
    expected_definition = {**identity, "model": model, "shard": 0, "n_shards": 1, "base_config": cfg}
    if (identity["schema"] != "kdm_p1_native_vcd_identity_v1" or identity["model"] != model
            or identity["decode_config"] != cfg or sidecar["definition"] != expected_definition
            or sidecar["identity"] != stable_hash(expected_definition)
            or complete["identity"] != expected_identity or complete["gate_identity"] != expected_identity
            or gate["identity"] != expected_identity or gate["passed"] is not True or gate["completed"] != 8):
        raise ValueError("Received native identity, configuration or real eight-input gate differs")
    registered = json.loads((ROOT / f"configs/runtime/{model}.json").read_text())
    if identity["frozen_spec"] != registered or identity["frozen_spec_sha256"] != stable_hash(registered):
        raise ValueError("Source identity differs from the frozen model registration")
    admission = json.loads(paths["admission_2424"].read_text())
    if identity["admission"] != admission["fixed_identity"]:
        raise ValueError("Source execution admission differs from the generation identity")
    if identity["source"]["eval_manifest_sha256"] != file_hash(ROOT / "data/current/all.jsonl"):
        raise ValueError("Source eval manifest differs from the current frozen manifest")
    samples = {row["id"]: row for row in read_jsonl(ROOT / "data/current/all.jsonl")
               if row["dataset"] == "food101" and row["split"] == "eval"}
    seen, keys, counts, truncated = set(), set(), Counter(), 0
    for row in read_jsonl(paths["raw"]):
        sample = row["sample"]
        task = {"sample": sample, "method": "vcd", "marker": "NONE", "reference_marker": "NONE",
                "guided": False, "reference_guided": False, "replicate": 0, "kind": "native_unguided"}
        if (row["model"] != model or sample != samples.get(sample["id"]) or sample["id"] in seen
                or row["key"] != task_id(model, task) or row["key"] in keys or row["status"] != "ok"
                or row["identity"] != sidecar["identity"] or row["config"] != cfg
                or any(row[field] != value for field, value in task.items() if field != "sample")
                or row["seed"] != stable_seed(sample["id"], model, 0)
                or row["prompt"] != task_prompt(sample["question"], guided=False)
                or row["reference_prompt"] != row["prompt"]):
            raise ValueError("Received native raw key, sample, source identity or parameters differ")
        if (not 1 <= len(row["tokens"]) <= 32
                or len(row["tokens"]) != len(row["selected_log_probabilities"])
                or len(row["tokens"]) != len(row["trace"])
                or not all(math.isfinite(value) for value in row["selected_log_probabilities"])
                or type(row["terminated"]) is not bool):
            raise ValueError("Received native response lacks finite complete token evidence")
        seen.add(sample["id"])
        keys.add(row["key"])
        counts[sample["class"]] += 1
        truncated += not row["terminated"]
    if seen != set(samples) or len(keys) != 2424 or len(counts) != 101 or set(counts.values()) != {24}:
        raise ValueError("Received native keys or complete 101 by 24 quotas differ")
    return {"model": model, "rows": 2424, "unique_keys": 2424, "class_count": 101, "per_class": 24,
            "truncated_rows": truncated, "source": source, "source_identity": sidecar["identity"],
            "generation_identity": expected_identity, "registered_backend_sha256": stable_hash(registered),
            "received_raw_path": str(paths["raw"].relative_to(ROOT)),
            "received_complete_path": str(paths["complete"].relative_to(ROOT)),
            "received_identity_path": str(paths["identity"].relative_to(ROOT)),
            "raw_sha256": source["files"]["raw"]["sha256"], "cpu_validation_passed": True,
            "gpu_runtime_performed_by_receiver": False}
