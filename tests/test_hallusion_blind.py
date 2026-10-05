"""CPU-only coverage for the blind frozen-config EOS generation boundary."""
import copy
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from kdm.decoding import DecodeConfig, Step
from kdm.io import atomic_json, file_hash, stable_hash, stable_seed
from workflows.hallusion_blind import generate as workflow


def condition(method):
    return {"model": "qwen25vl", "method": method,
            "marker": "UNCLEAR" if method.startswith("instruction_") or method == "cda_visual" else "NONE",
            "reference_marker": "UNCLEAR" if method.startswith("instruction_") or method == "cda_visual" else "NONE",
            "guided": method.startswith("instruction_") or method == "cda_visual",
            "reference_guided": False, "kind": "instruction_preserving" if method.startswith("instruction_") or method == "cda_visual" else "native_unguided",
            "replicate": 0, "config": asdict(DecodeConfig(method=method, max_tokens=None))}


@pytest.fixture
def fixture_plan(tmp_path, monkeypatch):
    for name in workflow.SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("source fixture")
    runtime = tmp_path / "configs/runtime/qwen25vl.json"
    runtime.parent.mkdir(parents=True)
    atomic_json(runtime, {"key": "qwen25vl", "dtype": "bfloat16"})
    samples = [{"id": "hallusion_" + str(i), "split": "eval", "dataset": "hallusionbench",
                "prompt": "Is the object present?", "image_paths": ["image.png"],
                "image_sha256": ["fixture"]} for i in range(951)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(s) + "\n" for s in samples))
    protocol = tmp_path / "protocol.json"
    conditions_path = tmp_path / "conditions.json"
    atomic_json(conditions_path, [condition("direct"), condition("instruction_m3id")])
    atomic_json(protocol, {"dataset_entry": {"manifest": "manifest.jsonl", "manifest_sha256": file_hash(manifest)},
                           "conditions": "conditions.json", "conditions_sha256": file_hash(conditions_path)})
    monkeypatch.setattr(workflow, "validate_native_runtime_files", lambda *args: None)
    def plan(methods, claim_id="claim", sample_ids=None):
        args = SimpleNamespace(protocol="protocol.json", model="qwen25vl", methods=methods,
                               sample_ids=sample_ids, output="outputs/hallusion_blind_20261005/runs",
                               claim_id=claim_id, owner="fixture", chunk_rows=8)
        return args, workflow.load_plan(args, root=tmp_path)
    return plan


def test_model_identity_and_task_keys_do_not_depend_on_method_shard(fixture_plan):
    _, allplan = fixture_plan(["direct", "instruction_m3id"])
    _, direct = fixture_plan(["direct"])
    _, ip = fixture_plan(["instruction_m3id"])
    assert allplan["identity"] == direct["identity"] == ip["identity"]
    assert len(allplan["tasks"]) == 1902
    assert len(direct["selected"]) == len(ip["selected"]) == 951
    assert not set(direct["selected"]) & set(ip["selected"])
    assert set(allplan["selected"]) == set(direct["selected"]) | set(ip["selected"])
    assert all(c.max_tokens is None for c in allplan["cfgs"].values())


def test_claim_rejects_overlap_but_accepts_disjoint_methods(fixture_plan):
    args, direct = fixture_plan(["direct"], "direct")
    _, keys, _ = workflow.claim(args, direct, {"fixture": True})
    assert len(keys) == 951
    args2, direct2 = fixture_plan(["direct"], "overlap")
    with pytest.raises(ValueError, match="already has an owner"):
        workflow.claim(args2, direct2, {"fixture": True})
    args3, ip = fixture_plan(["instruction_m3id"], "ip")
    _, keys3, _ = workflow.claim(args3, ip, {"fixture": True})
    assert len(keys3) == 951


class FakeSession:
    def __init__(self, prompt):
        import torch
        self.inputs = {"input_ids": torch.tensor([[ord(x) for x in prompt]]),
                       "pixel_values": torch.zeros((1, 3, 2, 2))}
        self.calls = []

    def next(self, prefix):
        self.calls.append(prefix)
        z = np.array([-8., 8.]) if len(prefix) < 40 else np.array([8., -8.])
        return Step(z, {0: np.zeros(2)}, {1: z.copy()})


class FakeBackend:
    eos = {0}
    sid_control = None

    def __init__(self):
        import torch
        self.torch = torch
        self.calls = []

    def session(self, image, prompt, **kwargs):
        value = FakeSession(prompt)
        self.calls.append((prompt, kwargs, value))
        return value

    def encode(self, text):
        return [ord(x) for x in text]

    def decode(self, tokens):
        return str(tokens)


@pytest.mark.parametrize("method", ["direct", "vcd", "dola", "deco", "sid",
                                    "instruction_vcd", "instruction_m3id", "cda_visual"])
def test_run_one_preserves_branches_and_eos_after_32(tmp_path, monkeypatch, method):
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2), (255, 255, 255)).save(image)
    c = condition(method)
    sample = {"id": "hallusion_fixture", "split": "eval", "dataset": "hallusionbench",
              "prompt": "Is the object present?", "image_paths": ["image.png"],
              "image_sha256": [file_hash(image)]}
    item = {"sample": sample, **{f: c[f] for f in ("method", "marker", "reference_marker",
            "guided", "reference_guided", "kind", "replicate")}, "condition_identity": stable_hash(c)}
    backend = FakeBackend()
    result = workflow.run_one(backend, item, DecodeConfig(**c["config"]), "qwen25vl")
    assert result["tokens"] == [1] * 40 + [0]
    assert result["terminated"] and result["config"]["max_tokens"] is None
    assert result["seed"] == stable_seed(sample["id"], "qwen25vl", 0)
    assert backend.calls[0][0] == workflow.guided_prompt(sample["prompt"], c["marker"], c["guided"])
    if method in {"vcd", "sid", "instruction_vcd", "instruction_m3id"}:
        ref = backend.calls[1]
        assert ref[0] == sample["prompt"]
        assert ref[1]["reference"] == ("text_only" if method == "instruction_m3id" else "sid" if method == "sid" else "noise")
        assert ref[1]["seed"] == result["seed"]
    if method.startswith("instruction_"):
        assert backend.calls[2][0] == sample["prompt"]
    if method == "instruction_m3id":
        assert result["offset_prompt_tokens"] == backend.encode(sample["prompt"])
        assert result["config"]["m3id_offset"] == len(backend.encode(sample["prompt"]))
    if method == "cda_visual":
        assert len(backend.calls) == 5
        assert backend.calls[1][1]["reference"] == "text_only"
        assert backend.calls[3][0] == backend.calls[4][0] == "[N/A]\nGive a concise answer."
        assert result["cda_equations"] == "ACL2025_main_text_4_6_7_no_momentum"
    if method in {"dola", "deco"}:
        assert backend.calls[0][1]["need_layers"] is True
    assert len(result["trace"]) == 41


def generated_fixture(tmp_path, monkeypatch, method):
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2), (255, 255, 255)).save(image)
    c = condition(method)
    sample = {"id": "hallusion_fixture", "split": "eval", "dataset": "hallusionbench",
              "prompt": "Is the object present?", "image_paths": ["image.png"],
              "image_sha256": [file_hash(image)]}
    item = {"sample": sample, **{f: c[f] for f in ("method", "marker", "reference_marker",
            "guided", "reference_guided", "kind", "replicate")}, "condition_identity": stable_hash(c)}
    result = workflow.run_one(FakeBackend(), item, DecodeConfig(**c["config"]), "qwen25vl")
    key = workflow.task_id("qwen25vl", item)
    plan = {"identity": "fixture_identity", "definition": {"model": "qwen25vl"},
            "datasets": {"hallusionbench": {"identity": "dataset_fixture"}},
            "tasks": {key: item}, "cfgs": {stable_hash(c): DecodeConfig(**c["config"])},
            "output": tmp_path / "model_output"}
    owner = {"identity": plan["identity"], "keys": [key], "owner": "fixture"}
    row = {"key": key, **item, "model": "qwen25vl", "identity": plan["identity"],
           "dataset_identity": "dataset_fixture", "claim_identity": stable_hash(owner),
           "wall_s": 0.1, "generation_source": "new_eos_only", **result}
    return plan, owner, row


@pytest.mark.parametrize("method", ["direct", "vcd", "dola", "deco", "sid",
                                    "instruction_vcd", "instruction_m3id", "cda_visual"])
def test_actual_full_row_passes_eos_and_branch_validation(tmp_path, monkeypatch, method):
    plan, owner, row = generated_fixture(tmp_path, monkeypatch, method)
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(row) + chr(10))
    checked = workflow.validate_rows(raw, plan, [row["key"]], stable_hash(owner), {0})
    assert checked == {"rows": 1, "truncated_rows": 0, "dataset_counts": {"hallusionbench": 1}}


@pytest.mark.parametrize("bad_field", ["tokens", "trace", "dataset_identity", "branch_prompt", "selected_log_probabilities"])
def test_validator_rejects_truncation_and_unbound_sources(tmp_path, monkeypatch, bad_field):
    plan, owner, row = generated_fixture(tmp_path, monkeypatch, "direct")
    if bad_field == "tokens":
        row["tokens"] = row["tokens"][:-1]
        row["terminated"] = False
    elif bad_field == "trace":
        row["trace"][0]["token"] = 999
    elif bad_field == "dataset_identity":
        row["dataset_identity"] = "foreign"
    elif bad_field == "branch_prompt":
        row["branch_inputs"]["main"]["prompt"] = "foreign"
    else:
        row["selected_log_probabilities"][0] = float("nan")
    raw = tmp_path / "bad.jsonl"
    raw.write_text(json.dumps(row) + chr(10))
    with pytest.raises(ValueError):
        workflow.validate_rows(raw, plan, [row["key"]], stable_hash(owner), {0})


def test_immutable_seal_receipt_and_completed_ledger_reject_tampering(tmp_path, monkeypatch):
    plan, owner, row = generated_fixture(tmp_path, monkeypatch, "direct")
    run = plan["output"] / "claims" / "fixture"
    run.mkdir(parents=True)
    workflow.write_once(run / "owner.json", owner)
    pending = run / "chunk_00000.pending.jsonl"
    pending.write_text(json.dumps(row) + chr(10))
    workflow.seal(plan, run, owner, 0, pending, [row["key"]], {0})
    assert workflow.completed_keys(plan) == {row["key"]}
    raw = run / "chunk_00000.jsonl"
    assert raw.stat().st_mode & 0o222 == 0
    raw.chmod(0o644)
    raw.write_text(raw.read_text() + chr(10))
    with pytest.raises(ValueError, match="receipt/source binding differs"):
        workflow.completed_keys(plan)
