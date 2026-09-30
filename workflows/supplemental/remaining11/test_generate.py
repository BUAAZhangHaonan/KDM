"""CPU regression checks against the real frozen sample and panel registries."""
from collections import Counter
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kdm.io import read_jsonl, stable_hash, stable_seed
from kdm.pipeline import experiment_tasks, probe_tasks, task_id
from workflows.supplemental.remaining11.generate import (
    MODELS, generation_key, ordered_tasks,
)


class RegisteredGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = list(read_jsonl(ROOT / "data/current/all.jsonl"))
        cls.food = [sample for sample in cls.samples if sample["dataset"] == "food101"]
        cls.methods = json.loads((ROOT / "configs/kdm/method_plan.json").read_text())
        cls.panel = json.loads((ROOT / "workflows/formal/panel.json").read_text())

    def test_real_full_food_quotas(self):
        self.assertEqual(len(self.samples), 9167)
        self.assertEqual(Counter(sample["split"] for sample in self.food), {"dev": 2424, "eval": 2424})
        counts = Counter((sample["split"], sample["class"]) for sample in self.food)
        self.assertEqual(len(counts), 202)
        self.assertEqual(set(counts.values()), {24})

    def test_published_panel_matrix_matches_shared_task_builder(self):
        for entry in self.panel["models"]:
            tasks = sum(1 for _ in experiment_tasks(self.food, entry["method_plan"]))
            self.assertEqual(tasks, entry["stage_five_eval_generation_tasks"])

    def test_remaining_models_keep_every_registered_condition(self):
        sample = next(sample for sample in self.food if sample["split"] == "eval")
        for model in MODELS:
            methods = self.methods[model]["food101"]
            original = {task_id(model, task): task for task in experiment_tasks((sample,), methods)}
            supplemental = {generation_key(model, "formal", task): task
                            for task in ordered_tasks([sample], methods, "formal", "food101")}
            self.assertEqual(supplemental, original)
            self.assertEqual(len(original), 80 if "sid" in methods else 64)
            self.assertEqual({task["marker"] for task in original.values()},
                             {"UNKNOWN", "UNCLEAR", "UNSURE", "I cannot identify it"})
            for method in ("vcd", "m3id"):
                matrix = {(task["marker"], task["reference_marker"])
                          for task in original.values() if task["method"] == method and task["kind"] == "main"}
                self.assertEqual(len(matrix), 16)
            controls = {task["method"] for task in original.values()
                        if task["kind"] == "instruction_preserving"}
            self.assertEqual(controls, {"instruction_vcd", "instruction_m3id", "cda_visual"})

    def test_independent_eval_matches_original_tasks_and_dev_is_complete(self):
        tasks = list(ordered_tasks(self.food, [], "independent", "food101"))
        self.assertEqual(len(tasks), 48480)
        self.assertEqual([task for task in tasks if task["sample"]["split"] == "eval"],
                         list(probe_tasks(self.food)))
        for split in ("dev", "eval"):
            counts = Counter(task["sample"]["id"] for task in tasks if task["sample"]["split"] == split)
            self.assertEqual(len(counts), 2424)
            self.assertEqual(set(counts.values()), {10})
        for task in tasks[:10]:
            self.assertEqual(task["replicate"], tasks.index(task))
            self.assertEqual(stable_seed(task["sample"]["id"], "gemma3_12b", task["replicate"]),
                             int(stable_hash([task["sample"]["id"], "gemma3_12b", task["replicate"]])[:16], 16) % (2**31))

    def test_candidate_original_key_and_complete_food_scope(self):
        tasks = list(ordered_tasks(self.food, [], "candidate", "food101"))
        self.assertEqual(len(tasks), 4848)
        for task in tasks:
            self.assertEqual(generation_key("gemma3_12b", "candidate", task),
                             stable_hash(["gemma3_12b", task["sample"]["id"], "closed"]))

    def test_vizwiz_reference_keeps_registered_eval_scope(self):
        vizwiz = [sample for sample in self.samples if sample["dataset"] == "vizwiz"]
        tasks = list(ordered_tasks(vizwiz, [], "independent", "vizwiz"))
        self.assertEqual(len(tasks), 35010)
        self.assertEqual({task["sample"]["split"] for task in tasks}, {"eval"})


if __name__ == "__main__":
    unittest.main()
