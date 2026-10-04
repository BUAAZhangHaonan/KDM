"""Reference-aware HallusionBench quality; separate from blind behavior labels.

No API calls. The source is pinned; only exact Yes/No on an audited binary item
with an ordinary reference is rule-scored. Other replies require a real Luna
quality judgment against question, FULL reply and FULL gt_answer_details.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

REVISION = "744007c232c292942c7f80eb61edb2465482da31"
SOURCE_BASE = "data/general_vqa_direct_20261004/source_pope_hallusion/hallusionbench"
MANIFEST = "data/general_vqa_direct_20261004/frozen/manifest_hallusionbench.jsonl"
SOURCE_SHA256 = {
    "HallusionBench.json": "ca6e0fb677dd56e5fb0e2700d06c58d01f8a6ae66e0f1773b2ac48c0a095d051",
    "utils.py": "535734ad70c8ea14253131ea59e4d2df6af4fae5341f044d226a8e78eaeb9c30",
    "evaluation.py": "8b7ec49f21f42f4bd399320274590cfaf470b53838862d44dc3bd530c0431c72",
}
SOURCE_URL = "https://raw.githubusercontent.com/tianyi-lab/HallusionBench/" + REVISION + "/HallusionBench.json"
QUALITY_CODE = {"incorrect": 0, "correct": 1, "unclear": 2}


def validate_review_author(author):
    if not isinstance(author, str) or not re.fullmatch(r"/root(?:/[a-z0-9_]+)*", author):
        raise ValueError("Review author must be the actual canonical agent name, not a placeholder")

# Explicit census of the pinned original visual-input questions, not prediction
# heuristics. Relative clauses containing 'which/when' are not open questions.
OPEN_ANSWER_TYPES = {
    "VS/table/1/1/1/0": "country", "VS/table/2/1/2/0": "country",
    "VS/table/1/2/1/0": "country", "VS/table/2/2/2/0": "country", "VS/table/2/2/3/0": "country",
    "VS/table/1/4/1/0": "state", "VS/table/2/4/2/0": "state", "VS/table/2/4/3/0": "state",
    "VS/table/1/5/1/0": "month", "VS/table/1/5/1/1": "month",
    "VS/table/2/5/2/0": "month", "VS/table/2/5/2/1": "month",
    "VS/table/2/5/3/0": "month", "VS/table/2/5/3/1": "month",
}
QUALITY_INSTRUCTIONS = (
    "Judge the FULL prediction against the original question and FULL reference answer. "
    "Use correct when its answer does not conflict with the reference, incorrect when it conflicts, "
    "and unclear when it provides no clear answer. For open questions, evaluate the actual country, "
    "state or month(s), not a yes/no surrogate. Read all later qualifications or corrections. "
    "If the reference says No answer or identifies inconsistent data, an appropriate statement that "
    "the answer cannot be determined can be correct; do not automatically turn such a response into "
    "unclear or incorrect merely because it declines to answer. An unsupported conflicting claim is "
    "not correct. This task judges reference-relative quality only. Behavioral abstention is judged "
    "separately without the reference and must not override this quality score."
)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_id(record):
    return "/".join(str(record[name]) for name in (
        "category", "subcategory", "visual_input", "set_id", "figure_id", "question_id"))


def official_score(record, quality_label):
    """Exact assign_correctness mapping, including the original no-visual case."""
    code = QUALITY_CODE[quality_label]
    if record["category"] == "VS" and int(record["figure_id"]) == 0:
        return int(code in (1, 2))
    return int(code == 1)


class HallusionScoring:
    def __init__(self, root):
        self.root = Path(root)
        self.base = self.root / SOURCE_BASE
        for name, expected in SOURCE_SHA256.items():
            if file_sha(self.base / name) != expected:
                raise ValueError("Pinned Hallusion source SHA differs: " + name)
        original = json.loads((self.base / "HallusionBench.json").read_text())
        self.records = {}
        for source_row, record in enumerate(original, 1):
            if int(record["visual_input"]) < 1:
                continue
            sid = "hallusionbench:" + original_id(record)
            if sid in self.records:
                raise ValueError("Duplicate original Hallusion item")
            self.records[sid] = (source_row, record)
        self.samples = [json.loads(line) for line in (self.root / MANIFEST).read_text().splitlines()]
        if len(self.samples) != 951 or len(self.records) != 951 or {s["id"] for s in self.samples} != set(self.records):
            raise ValueError("Frozen visual-input Hallusion census differs")
        for sample in self.samples:
            self.source_record(sample)
        detected = {original_id(r) for _, r in self.records.values()
                    if re.search(r"\bwhich\s+(?:country|state|month)\b", r["question"], re.I)}
        if detected != set(OPEN_ANSWER_TYPES):
            raise ValueError("The explicit 14-item open-question registry differs from original source")
        if any(r["category"] == "VS" and int(r["figure_id"]) == 0 for _, r in self.records.values()):
            raise ValueError("Unexpected no-visual assign_correctness exception in this visual-only cohort")

    def source_record(self, sample):
        if sample.get("dataset") != "hallusionbench" or sample.get("id") not in self.records:
            raise ValueError("Sample is outside the frozen Hallusion visual cohort")
        source_row, record = self.records[sample["id"]]
        if (sample.get("source_record") != record or sample.get("source_row") != source_row
                or sample.get("source_id") != original_id(record)
                or sample.get("question_source_sha256") != SOURCE_SHA256["HallusionBench.json"]
                or sample.get("question_source_uri") != SOURCE_URL
                or any(sample.get(name) != record[name] for name in (
                    "question", "gt_answer", "gt_answer_details", "visual_input", "category", "figure_id"))):
            raise ValueError("Sample original question/reference/source binding differs")
        return record

    def question_kind(self, sample):
        self.source_record(sample)
        return "open" if sample["source_id"] in OPEN_ANSWER_TYPES else "binary"

    def reference_kind(self, sample):
        reference = self.source_record(sample)["gt_answer_details"].strip().casefold()
        if reference == "no answer":
            return "no_answer"
        if reference == "no, inconsistency in table":
            return "inconsistent_reference"
        return "ordinary"

    def quality_key(self, sample, answer):
        record = self.source_record(sample)
        if not isinstance(answer, str):
            raise ValueError("Full model answer must be a string")
        return digest(["hallusion_full_reference_quality_v1", record["question"], answer, record["gt_answer_details"]])

    def literal_binary_behavior(self, sample, answer):
        """Target-blind answer behavior; ordinary 'No' is never abstention."""
        # Kind is an audited property of question syntax, not its target answer.
        if self.question_kind(sample) != "binary" or answer.strip().casefold() not in {"yes", "no"}:
            return None
        return {"label": "answer_assertive", "abstain": False, "answer_text": answer,
            "predicted_answer": answer.strip().casefold(), "evidence_span": answer,
            "decision_source": "hallusion_complete_literal_binary_behavior_v1"}

    def check_behavior_prediction(self, sample, prediction):
        """Call after generic label/evidence checks; no reference-based grading."""
        kind = self.question_kind(sample)
        if kind == "binary" and prediction not in {"yes", "no", None}:
            raise ValueError("Binary Hallusion extraction must be yes/no/None")
        if kind == "open" and prediction is not None and not isinstance(prediction, str):
            raise ValueError("Open Hallusion extraction must retain country/state/month text")

    def literal_binary_quality(self, sample, answer):
        record = self.source_record(sample)
        behavior = self.literal_binary_behavior(sample, answer)
        # A bare No is not automatically a correct answer to an unanswerable
        # comparison, even where the original binary metadata happens to be 0.
        if behavior is None or self.reference_kind(sample) != "ordinary":
            return None
        polarity = {"0": "no", "1": "yes"}.get(str(record["gt_answer"]))
        if polarity is None:
            raise ValueError("Unknown original binary polarity")
        label = "correct" if behavior["predicted_answer"] == polarity else "incorrect"
        return {"quality_key": self.quality_key(sample, answer), "quality_label": label,
            "official_correctness": QUALITY_CODE[label], "score": official_score(record, label),
            "source": "hallusion_complete_literal_binary_polarity_v1"}

    def quality_request(self, sample, answer):
        record = self.source_record(sample)
        return {"quality_key": self.quality_key(sample, answer), "dataset": "hallusionbench",
            "question": record["question"], "answer": answer, "gt_answer_details": record["gt_answer_details"],
            "question_kind": self.question_kind(sample), "reference_kind": self.reference_kind(sample),
            "expected_answer_type": OPEN_ANSWER_TYPES.get(sample["source_id"], "yes_no"),
            "source": {"sample_id": sample["id"], "source_row": sample["source_row"],
                "source_uri": SOURCE_URL, "source_sha256": SOURCE_SHA256["HallusionBench.json"],
                "source_record_sha256": digest(record)}, "memberships": []}

    def validate_quality(self, sample, answer, review):
        request = self.quality_request(sample, answer)
        for field in ("quality_key", "question", "answer", "gt_answer_details"):
            if review.get(field) != request[field]:
                raise ValueError("Hallusion quality judgment binding differs: " + field)
        label = review.get("quality_label")
        if label not in QUALITY_CODE:
            raise ValueError("Quality label must be correct/incorrect/unclear")
        for field in ("author", "model", "effort", "reason"):
            if not isinstance(review.get(field), str) or not review[field].strip():
                raise ValueError("Actual quality-review provenance/reason required: " + field)
        validate_review_author(review["author"])
        if review["model"] != "gpt-5.6-luna" or review["effort"] != "medium":
            raise ValueError("Quality review requires actual gpt-5.6-luna with medium effort")
        if "call_id" not in review or (review["call_id"] is not None and (
                not isinstance(review["call_id"], str) or not review["call_id"].strip())):
            raise ValueError("call_id must be present and either null or a nonempty actual ID")
        for field, full in (("answer_evidence", answer), ("reference_evidence", request["gt_answer_details"])):
            if not isinstance(review.get(field), str) or review[field] not in full or (full and not review[field]):
                raise ValueError("Quality evidence must be an exact original span: " + field)
        code = QUALITY_CODE[label]
        score = official_score(self.source_record(sample), label)
        if ("official_correctness" in review and review["official_correctness"] != code
                or "score" in review and review["score"] != score):
            raise ValueError("Declared quality score disagrees with the official mapping")
        return {"quality_key": request["quality_key"], "quality_label": label,
            "official_correctness": code, "score": score, "source": "luna_full_reference_quality_v1",
            "reviewer": {key: review[key] for key in ("author", "model", "effort", "call_id")}}

    def audit(self):
        def entry(sample):
            return {key: sample[key] for key in ("id", "source_id", "source_row", "question", "gt_answer", "gt_answer_details")} | {
                "answer_type": OPEN_ANSWER_TYPES.get(sample["source_id"], "yes_no"),
                "source_uri": SOURCE_URL, "source_sha256": SOURCE_SHA256["HallusionBench.json"]}
        return {"revision": REVISION, "source_sha256": SOURCE_SHA256, "frozen_manifest_sha256": file_sha(self.root / MANIFEST),
            "visual_rows": len(self.samples), "binary_rows": 937, "open_rows": 14,
            "visual_input_counts": dict(Counter(s["visual_input"] for s in self.samples)),
            "reference_kind_counts": dict(Counter(self.reference_kind(s) for s in self.samples)),
            "quality_code": QUALITY_CODE, "visual_cohort_score": "correct=1; incorrect=0; unclear=0",
            "no_visual_VS_figure0_rows": 0, "quality_instructions": QUALITY_INSTRUCTIONS,
            "open_questions": [entry(s) for s in self.samples if self.question_kind(s) == "open"],
            "special_references": [entry(s) for s in self.samples if self.reference_kind(s) != "ordinary"]}


def make_queue(scoring, raw_paths):
    pending, rules, seen = {}, [], set()
    for path in raw_paths:
        for line, text in enumerate(Path(path).read_text().splitlines(), 1):
            row = json.loads(text)
            sample, answer = row["sample"], row["text"]
            if sample["dataset"] != "hallusionbench":
                continue
            identity = (row["model"], row["key"])
            if identity in seen:
                raise ValueError("Duplicate Hallusion model/prediction key")
            seen.add(identity)
            membership = {"model": row["model"], "sample_id": sample["id"], "key": row["key"],
                          "source_raw": str(path), "source_line": line}
            rule = scoring.literal_binary_quality(sample, answer)
            if rule is not None:
                rules.append({**membership, **rule})
            else:
                request = scoring.quality_request(sample, answer)
                item = pending.setdefault(request["quality_key"], request)
                item["memberships"].append(membership)
    return list(pending.values()), rules


def self_test(scoring):
    """Synthetic review records exercise mapping only; no real labels produced."""
    by_id = {s["source_id"]: s for s in scoring.samples}
    names = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        names.append(name)
    def review(sample, answer, label):
        request = scoring.quality_request(sample, answer)
        return {**{k: request[k] for k in ("quality_key", "question", "answer", "gt_answer_details")},
            "quality_label": label, "answer_evidence": answer, "reference_evidence": request["gt_answer_details"],
            "reason": "CPU test fixture only; not an actual model judgment", "author": "/root/cpu_test_fixture",
            "model": "gpt-5.6-luna", "effort": "medium", "call_id": "CPU_TEST_FIXTURE"}
    country = by_id["VS/table/1/1/1/0"]
    check("open_gt1_wrong_country_is_not_yes", country["gt_answer"] == "1" and scoring.literal_binary_quality(country, "France") is None)
    check("wrong_country_quality_incorrect_scores_zero", scoring.validate_quality(country, "France", review(country, "France", "incorrect"))["score"] == 0)
    check("open_bare_yes_requires_quality", scoring.literal_binary_quality(country, "Yes") is None)
    scoring.check_behavior_prediction(country, "Syria")
    check("country_text_allowed_for_blind_extraction", True)
    no_answer = by_id["VS/table/2/5/3/0"]
    answer = "The table does not provide enough information to determine the month."
    check("reference_no_answer_correct_refusal_retains_one", scoring.validate_quality(no_answer, answer, review(no_answer, answer, "correct"))["score"] == 1)
    check("reference_no_answer_unclear_remains_zero", scoring.validate_quality(no_answer, answer, review(no_answer, answer, "unclear"))["score"] == 0)
    special_binary = by_id["VS/table/2/5/3/2"]
    check("reference_no_answer_binary_no_is_not_auto_correct", scoring.literal_binary_quality(special_binary, "No") is None)
    check("inconsistent_reference_binary_no_requires_review", scoring.literal_binary_quality(by_id["VS/table/2/4/2/1"], "No") is None)
    binary = next(s for s in scoring.samples if scoring.question_kind(s) == "binary" and s["gt_answer"] == "0" and scoring.reference_kind(s) == "ordinary")
    check("ordinary_binary_no_is_correct", scoring.literal_binary_quality(binary, "No")["score"] == 1)
    check("ordinary_binary_no_is_answer_not_abstention", scoring.literal_binary_behavior(binary, "No")["abstain"] is False)
    check("full_prose_yes_no_requires_quality", scoring.literal_binary_quality(binary, "No, but actually yes.") is None)
    correct_record = review(country, "Syria", "correct")
    unknown_call = copy.deepcopy(correct_record); unknown_call["call_id"] = None
    check("unknown_call_id_null_accepted", scoring.validate_quality(country, "Syria", unknown_call)["reviewer"]["call_id"] is None)
    for field, value, label in [("call_id", "", "empty_call_id_rejected"),
            ("author", "canonicalagent", "placeholder_author_rejected"),
            ("call_id", 42, "nonstring_call_id_rejected"),
            ("model", "gpt-6-luna", "different_luna_model_rejected"),
            ("effort", "high", "different_effort_rejected")]:
        invalid = copy.deepcopy(correct_record); invalid[field] = value
        try: scoring.validate_quality(country, "Syria", invalid)
        except ValueError: check(label, True)
        else: raise AssertionError(label)
    missing_call = copy.deepcopy(correct_record); del missing_call["call_id"]
    try: scoring.validate_quality(country, "Syria", missing_call)
    except ValueError: check("missing_call_id_field_rejected", True)
    else: raise AssertionError("Missing call_id field accepted")
    changed = copy.deepcopy(correct_record); changed["gt_answer_details"] = "France"
    try: scoring.validate_quality(country, "Syria", changed)
    except ValueError: check("reference_tampering_rejected", True)
    else: raise AssertionError("Reference tampering accepted")
    changed = copy.deepcopy(correct_record); changed["answer"] = "France"
    try: scoring.validate_quality(country, "Syria", changed)
    except ValueError: check("full_answer_tampering_rejected", True)
    else: raise AssertionError("Answer tampering accepted")
    other = by_id["VS/table/2/2/2/0"]; first = by_id["VS/table/1/2/1/0"]
    check("same_question_answer_different_reference_different_key", first["question"] == other["question"] and scoring.quality_key(first, "France") != scoring.quality_key(other, "France"))
    # Execute only the pinned, dependency-free mapping function. Never import
    # utils.py: its module body imports OpenAI and accesses apikey.txt.
    tree = ast.parse((scoring.base / "utils.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "assign_correctness")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "pinned_assign_correctness", "exec"), namespace)
    comparisons = 0
    for sample in scoring.samples:
        for label, code in QUALITY_CODE.items():
            original = copy.deepcopy(scoring.source_record(sample)); original["test_quality"] = code
            actual = namespace["assign_correctness"]([original], "test_quality")[0]["correct"]
            if actual != official_score(original, label): raise AssertionError("Official mapping mismatch")
            comparisons += 1
    check("official_assign_correctness_all_951_times_3", comparisons == 2853)
    return {"status": "passed", "checks": names, "count": len(names), "official_mapping_comparisons": comparisons,
            "fixture_labels_are_actual_reviews": False, "api_calls": 0, "gpu_initialized": False}


def write_outputs(path, audit, test_result=None, queue=None):
    path.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (path / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save("source_audit.json", audit)
    if test_result is not None: save("cpu_tests.json", test_result)
    for name in ("open_questions", "special_references"):
        (path / (name + ".jsonl")).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in audit[name]), encoding="utf-8")
    if queue is not None:
        pending, rules = queue
        for name, values in (("pending_quality", pending), ("rule_quality", rules)):
            (path / (name + ".jsonl")).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values), encoding="utf-8")
        save("queue_receipt.json", {"pending_unique_quality_keys": len(pending), "pending_rows": sum(len(v["memberships"]) for v in pending), "rule_rows": len(rules)})
    table = "| Original source row | ID | Type | Full reference |\n|---:|---|---|---|\n"
    table += "".join(f"| {x['source_row']} | {x['source_id']} | {x['answer_type']} | {x['gt_answer_details'].replace('|', '&#124;')} |\n" for x in audit["open_questions"])
    text = "# HallusionBench reference-aware scoring audit\n\n"
    text += "The frozen 951 visual-input questions contain 937 binary and 14 open questions (5 country, 3 state, 6 month). All 14 IDs and complete references are listed below and bound to their original one-based JSON row. The source revision is `" + REVISION + "`.\n\n"
    text += "The [official evaluator](https://github.com/tianyi-lab/HallusionBench/blob/" + REVISION + "/utils.py) judges original question, full model reply and `gt_answer_details`. Labels map to incorrect=0, correct=1, unclear=2. On these 951 items only correct earns one point. The VS figure-0 exception exists in the official implementation, but no such item occurs in this visual cohort.\n\n"
    text += "There are seven special references: four are exactly `No answer`, and three are `No, inconsistency in table` (one open state question and two binary comparisons). A quality judgment of correct remains score 1 even when the independently recorded behavior is abstention. Unclear remains score 0 here. All seven special references require full-reference review, including bare Yes/No predictions.\n\n"
    text += table + "\nOnly entire replies equal to Yes/No (case/outer whitespace ignored) on audited ordinary binary questions use the original polarity rule. All other Hallusion replies need real Luna quality review. The quality key includes original question, complete reply and complete reference; it cannot reuse a target-blind QA label. Behavior extraction remains target-blind, permits country/state/month strings on open items, and treats ordinary binary No as an answer.\n\n"
    text += "Minimal score.py integration: create HallusionScoring(ROOT) once. For Hallusion, resolve quality independently via literal_binary_quality or validate_quality, leaving missing quality pending. Use quality['score'] before any generic abstain/invalid early-return. Keep behavior review and its pending count separate; accuracy depends on quality completion and abstention rate on behavior completion. Allow arbitrary extracted text for registered open Hallusion questions via check_behavior_prediction. Existing behavior decisions may be reused only as behavior evidence, never as quality evidence.\n\n"
    text += "Quality review fields: quality_key, question, answer, gt_answer_details, quality_label, answer_evidence, reference_evidence, reason, author, model (exactly gpt-5.6-luna), effort (exactly medium), call_id. The call_id field is mandatory but may be null when the actual call ID is unavailable; otherwise it must be a nonempty actual ID string. The helper performs no API calls and does not modify predictions, manifests or GPU work.\n"
    (path / "source_audit.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True, help="Fresh audit/queue output directory")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--raw-list", type=Path, help="Previously validated immutable raw path list")
    args = parser.parse_args()
    scoring = HallusionScoring(args.root)
    tests = self_test(scoring) if args.self_test else None
    queue = None
    if args.raw_list:
        raw_paths = args.raw_list.read_text().splitlines()
        if not raw_paths or len(raw_paths) != len(set(raw_paths)):
            raise ValueError("Raw path list must be nonempty and unique")
        queue = make_queue(scoring, raw_paths)
    write_outputs(args.out, scoring.audit(), tests, queue)
    print(json.dumps({"out": str(args.out), "visual_rows": 951, "open_rows": 14,
        "test_checks": None if tests is None else tests["count"],
        "pending_quality_keys": None if queue is None else len(queue[0]),
        "rule_rows": None if queue is None else len(queue[1])}))


if __name__ == "__main__":
    main()
