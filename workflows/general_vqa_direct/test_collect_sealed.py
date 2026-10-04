"""CPU fixtures for source tampering, duplicate keys and row evidence."""
import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from workflows.general_vqa_direct import collect_sealed as c
from workflows.general_vqa_direct import generate as g
from workflows.general_vqa_direct.inputs import parts_for_sample


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.folder = TEST_ROOT / self._testMethodName
        self.folder.mkdir(parents=True, exist_ok=False)
        self.model = "qwen35_4b"
        sample = {"id": "scienceqa:fixture", "dataset": "scienceqa", "split": "eval",
                  "source_split": "test", "question": "Fixture question?", "prompt": "Fixture prompt",
                  "gold": "A", "options": ["one", "two"], "hint": "",
                  "image_paths": ["data/fixture.png"], "image_sha256": ["a" * 64]}
        item = g.direct_task(sample)
        key = g.task_id(self.model, item)
        definition = {"model": self.model, "dtype": "bfloat16", "batch_size": 1, "source_sha256": {"fixture": "b" * 64}}
        identity = {"identity": c.stable_hash(definition), "definition": definition}
        owner = {"claim_id": "fixture", "keys": [key], "identity": identity["identity"],
                 "source_sha256": definition["source_sha256"], "hostname": "fixture-host", "batch_size": 1,
                 "dataset_entries": {"scienceqa": {"fixture": True}},
                 "admission": {"host": "4029", "project_root": "/fixture", "runtime_spec": {"key": self.model, "dtype": "bfloat16"}}}
        cfg = g.DecodeConfig(method="direct", max_tokens=2, temperature=0, top_p=1)
        self.plan = {"identity": identity["identity"], "definition": definition, "tasks": {key: item},
                     "datasets": {"scienceqa": {"identity": "dataset-fixture", "cfg": cfg}}}
        lp = [-0.3, -0.2]
        self.row = {"key": key, **item, "model": self.model, "identity": identity["identity"],
            "claim_identity": c.stable_hash(owner), "dataset_identity": "dataset-fixture", "status": "ok",
            "text": "one", "prompt": sample["prompt"], "reference_prompt": None, "neutral_prompt": None,
            "config": asdict(cfg), "seed": g.stable_seed(sample["id"], self.model, 0),
            "tokens": [7, 2], "terminated": True, "wall_s": 0.25,
            "selected_log_probabilities": lp, "sequence_log_probability": sum(lp), "first_probability": math.exp(lp[0]),
            "trace": [{"token": token, "log_probability": prob, "sampling_log_probability": 0.0} for token, prob in zip([7, 2], lp)],
            "input_evidence": {"source_image_count": 1, "source_image_indices_in_input": [0], "image_occurrences": 1,
                "exact_manifest_prompt_sha256": c.stable_hash(sample["prompt"]), "input_token_count": 10, "input_ids_sha256": "c" * 64}}
        self.roles = {"raw": self.model + "/claims/fixture/chunk_00000.jsonl",
                      "receipt": self.model + "/claims/fixture/chunk_00000.complete.json",
                      "owner": self.model + "/claims/fixture/owner.json", "identity": self.model + "/identity.json"}
        raw = (json.dumps(self.row) + "\n").encode()
        self.receipt = {"status": "complete", "identity": identity["identity"], "claim_identity": c.stable_hash(owner),
            "keys": [key], "eos_token_ids": [2], "raw_path": "claims/fixture/chunk_00000.jsonl",
            "owner_path": "claims/fixture/owner.json", "raw_sha256": c.digest(raw), "owner_sha256": c.digest(c.json_bytes(owner)),
            "validation": {"rows": 1, "truncated_rows": 0, "dataset_counts": {"scienceqa": 1}}}
        self.files = {self.roles["raw"]: raw, self.roles["owner"]: c.json_bytes(owner),
                      self.roles["identity"]: c.json_bytes(identity), self.roles["receipt"]: c.json_bytes(self.receipt)}
        self.inv = {"host": "4029", "root": "/fixture", "hostname": "fixture-host", "files": {
            name: {"sha256": c.digest(data), "bytes": len(data)} for name, data in self.files.items()}}
        self.chunk = {"model": self.model, "roles": self.roles, "ledger_entries": [{"key": key, "dataset": "scienceqa",
            "identity": identity["identity"], "receipt": "claims/fixture/chunk_00000.complete.json",
            "receipt_sha256": c.digest(self.files[self.roles["receipt"]])}]}

    def validate_row(self, row):
        path = self.folder / "fixture.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return g.validate_rows(path, self.plan, self.receipt["keys"], self.receipt["claim_identity"], {2})

    def refresh_inventory_file(self, role):
        name = self.roles[role]
        self.inv["files"][name] = {"sha256": c.digest(self.files[name]), "bytes": len(self.files[name])}

    def test_valid_chunk(self):
        receipt, _, _, rows = c.audit_binding(self.inv, self.chunk, self.files.__getitem__)
        self.assertEqual(self.validate_row(rows[0]), receipt["validation"])
        c.audit_numbers_and_images(rows, parts_for_sample)

    def test_inventory_ignores_auxiliary_directories_and_pending(self):
        run = self.folder / c.RUN
        for name, data in self.files.items():
            c.write_new(run / name, data)
        (run / self.model / "completed_keys.jsonl").write_text(
            json.dumps(self.chunk["ledger_entries"][0]) + "\n", encoding="utf-8")
        (run / self.model / "claims/fixture/chunk_00001.pending.jsonl").write_bytes(b"incomplete")
        (run / "launches").mkdir()
        (run / "deployment").mkdir()
        with patch.dict(c.HOSTS, {"4029": {"root": str(self.folder)}}):
            inv = c.inventory(self.folder, "4029")
        self.assertEqual(len(inv["chunks"]), 1)
        self.assertEqual(set(inv["files"]), set(self.files))

    def test_raw_tampering_rejected(self):
        self.files[self.roles["raw"]] += b" "
        with self.assertRaisesRegex(ValueError, "SHA/size"):
            c.audit_binding(self.inv, self.chunk, self.files.__getitem__)

    def test_owner_tampering_with_refreshed_transport_hash_rejected(self):
        owner = c.strict_json(self.files[self.roles["owner"]])
        owner["hostname"] = "different-host"
        self.files[self.roles["owner"]] = c.json_bytes(owner)
        self.refresh_inventory_file("owner")
        with self.assertRaisesRegex(ValueError, "identity binding"):
            c.audit_binding(self.inv, self.chunk, self.files.__getitem__)

    def test_receipt_tampering_caught_by_ledger(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["completed_utc"] = "tampered"
        self.files[self.roles["receipt"]] = c.json_bytes(receipt)
        self.refresh_inventory_file("receipt")
        with self.assertRaisesRegex(ValueError, "ledger SHA"):
            c.audit_binding(self.inv, self.chunk, self.files.__getitem__)

    def test_duplicate_model_key_rejected(self):
        seen = set()
        c.add_unique_keys(seen, self.model, self.receipt["keys"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            c.add_unique_keys(seen, self.model, self.receipt["keys"])
        c.add_unique_keys(seen, "other-model", self.receipt["keys"])

    def test_wrong_eos_rejected(self):
        self.row["tokens"][-1] = 3
        with self.assertRaisesRegex(ValueError, "EOS"):
            self.validate_row(self.row)

    def test_missing_image_rejected(self):
        self.row["input_evidence"]["source_image_indices_in_input"] = []
        with self.assertRaisesRegex(ValueError, "every image"):
            self.validate_row(self.row)

    def test_multimage_repeat_order_rejected(self):
        self.row["sample"].update(dataset="mmmu", prompt="<image 2> then <image 1> then <image 2>",
                                  image_paths=["data/a.png", "data/b.png"], image_slots=[1, 2])
        self.row["input_evidence"]["source_image_indices_in_input"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "input order"):
            c.audit_numbers_and_images([self.row], parts_for_sample)

    def test_nonfinite_probability_rejected(self):
        self.row["selected_log_probabilities"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            c.audit_numbers_and_images([self.row], parts_for_sample)

    def test_immutable_copy_reuse_and_replacement_rejection(self):
        path = self.folder / "copied.jsonl"
        self.assertTrue(c.install_immutable(path, b"original\n"))
        self.assertFalse(c.install_immutable(path, b"original\n"))
        with self.assertRaisesRegex(ValueError, "refusing replacement"):
            c.install_immutable(path, b"changed\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", type=Path, required=True, help="Fresh project-local test snapshot; never /tmp")
    args = parser.parse_args()
    TEST_ROOT = args.test_root.resolve()
    if not TEST_ROOT.is_relative_to(ROOT / "outputs/general_vqa_direct/collected/snapshots"):
        raise ValueError("Tests must stay in the new collected snapshot tree")
    TEST_ROOT.mkdir(parents=True, exist_ok=False)
    unittest.main(argv=[sys.argv[0]], verbosity=2)
