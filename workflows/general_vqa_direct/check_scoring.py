"""Finite regression checks and blind-rule agreement with actual reviewed QA."""
import argparse
import json
from pathlib import Path
from score import check_decision, correctness, explicit_binary_decision, rule_decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sample = {"dataset": "pope", "prompt": "Is there a dog in the image?", "gold": "yes"}
    for text, expected in [
        ("Yes, a dog is sitting on the grass.", "yes"),
        ("No, there is no dog. A cat is on the sofa.", "no"),
        ("Yes, but I cannot determine whether this is a dog.", None),
        ("No, actually yes. I was mistaken.", None),
        ("Yes. The animal is not a dog.", None),
        ("No. However, the correct answer is yes.", None),
        ("Yes, there appears to be a dog.", "yes"),
        ("I cannot answer from this image.", None),
    ]:
        decision = explicit_binary_decision(text)
        assert (None if decision is None else decision["predicted_answer"]) == expected, text
    negative = rule_decision(sample, "No")
    assert negative["abstain"] is False and correctness(sample, "No", negative, None) == 0
    matched = total = 0
    differences = []
    for line in args.decisions.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        total += 1
        check_decision(sample, row["answer"], row["decision"])
        proposal = rule_decision(sample, row["answer"])
        if proposal is None:
            continue
        matched += 1
        if (proposal["abstain"], proposal["predicted_answer"]) != (row["decision"]["abstain"], row["decision"]["predicted_answer"]):
            differences.append(row["qa_key"])
    result = {"regression_cases": 9, "reviewed_QA": total,
              "automatic_rule_covered": matched, "semantic_disagreements": differences,
              "status": "passed" if not differences else "failed"}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    assert not differences


if __name__ == "__main__":
    main()
