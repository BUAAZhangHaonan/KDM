"""Regressions for reviewed MMMU open answers without an option extraction."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from workflows.general_vqa_direct.score import correctness, load_mmmu


def main():
    evaluator = load_mmmu(ROOT / "data/general_vqa_direct_20261004/source_metadata/mmmu_eval_utils.py")
    decision = {"label": "answer_assertive", "predicted_answer": None}
    sample = {"dataset": "mmmu", "question_type": "open", "gold": "LESS"}
    assert correctness(sample, "LESS", decision, evaluator) == 1
    assert correctness(sample, "MORE", decision, evaluator) == 0
    sample["gold"] = "Yes"
    assert correctness(sample, "Yes ", decision, evaluator) == 1
    sample["gold"] = "25"
    assert correctness(sample, "25", decision, evaluator) == 1
    for label in ("invalid", "abstain"):
        assert correctness(sample, "25", {"label": label, "predicted_answer": None}, evaluator) == 0
    sample.update(question_type="multiple-choice", gold="A")
    assert correctness(sample, "Unresolved competing choices", decision, evaluator) == 0
    sample = {"dataset": "scienceqa", "gold": 0}
    assert correctness(sample, "Unresolved competing choices", decision, evaluator) == 0
    print(json.dumps({"checks": 8, "status": "passed", "evaluator": "pinned_official_mmmu"}))


if __name__ == "__main__":
    main()
