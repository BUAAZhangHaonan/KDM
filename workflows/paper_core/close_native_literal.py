#!/usr/bin/env python3
"""Prepare the five actual root literal decisions and verify unchanged primary effects."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, within
from workflows.main_results.score import qah

ROOT_DECISIONS = {
    "The image shows chicken wings. ": ("chicken wings", "chicken_wings", 1, [], "完整主菜名为chicken wings。"),
    "Bread pudding with caramel sauce and whipped cream.": ("Bread pudding", "bread_pudding", 1,
        ["caramel sauce", "whipped cream"], "Bread pudding为主菜，sauce和cream为配菜。"),
    "The image shows a slice of cheesecake.": ("slice of cheesecake", "cheesecake", 0, [], "完整抽取保留份量修饰slice of。"),
    "The image shows TWG Tea macarons.": ("TWG Tea macarons", "macarons", 0, [], "完整抽取保留品牌或风味修饰TWG Tea。"),
    "Grilled pork chop with rosemary and sweet potato fries.": ("Grilled pork chop", "pork_chop", 0,
        ["rosemary", "sweet potato fries"], "完整主菜名保留烹饪修饰Grilled，后者为配菜。"),
}


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for line, raw in enumerate(stream, 1):
            yield line, json.loads(raw), hashlib.sha256(raw).hexdigest()


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(args):
    pending = within(ROOT, args.pending)
    selected = [record for _, record, _ in rows(pending)]
    if (len(selected) != 6 or len({r["qa_key"] for r in selected}) != 5
            or Counter(r["model"] for r in selected) != {"llava16_mistral": 2, "gemma3_4b": 4}
            or {r["answer"] for r in selected} != set(ROOT_DECISIONS)):
        raise ValueError("Actual literal-pending six-member scope differs")
    grouped, wanted = defaultdict(list), defaultdict(dict)
    for record in selected:
        if (record["question"] != "What specific food is shown in this image?"
                or qah(record["question"], record["answer"]) != record["qa_key"]
                or record["canonical_name_in_primary_score"] != 1 or record["abstain"] is not False
                or record["literal_extracted_name_score"] is not None):
            raise ValueError("Root literal scope or previously decided primary value differs")
        grouped[record["qa_key"]].append(record)
        wanted[record["source_path"]][record["source_line"]] = record
    actual = {}
    for name, required in wanted.items():
        with Path(name).open("rb") as stream:
            for line, raw in enumerate(stream, 1):
                if line in required:
                    actual[(name, line)] = (json.loads(raw), hashlib.sha256(raw).hexdigest())
                if line >= max(required):
                    break
    if len(actual) != 6:
        raise ValueError("A selected source line is missing")
    for record in selected:
        source, sha = actual[(record["source_path"], record["source_line"])]
        if (source["key"], source["identity"], source["model"], source["sample"]["id"],
            source["sample"]["question"], source["text"], source["seed"], source["config"], sha) != (
            record["key"], record["source_identity"], record["model"], record["sample_id"],
            record["question"], record["answer"], record["seed"], record["config"], record["raw_line_sha256"]):
            raise ValueError("Selected actual raw QA, identity, config, seed or line SHA differs")
    output = within(ROOT, args.output)
    output.mkdir(parents=True, exist_ok=False)
    decisions = []
    for key in sorted(grouped):
        members = grouped[key]
        source = members[0]
        name, canonical, literal, side, reason = ROOT_DECISIONS[source["answer"]]
        if name not in source["answer"] or any(span not in source["answer"] for span in side):
            raise ValueError("Root extraction evidence span is absent from complete response")
        decisions.append({"qa_key": key, "question": source["question"], "answer": source["answer"],
            "endorsed_primary_names": [name], "literal_full_name": name,
            "name_relations": [{"name": name, "span": name, "role": "main"}]
                + [{"name": span, "span": span, "role": "side"} for span in side],
            "canonical_override": canonical, "root_literal_score_for_verified_members": literal,
            "name_scope_ambiguous": False, "abstain": False, "needs_root": False, "root_reviewed": True,
            "root_reason": reason, "annotation_model": "gpt-6.1-sol", "annotation_effort": "max",
            "annotation_agent": "/root", "annotation_call_id": "",
            "judgment_source": "Explicit /root instruction after reading the full five QA and six-source packet",
            "prepared_by": {"agent": "/root/assets", "role": "File preparation and actual source-line verification"},
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_score_records": members, "actual_source_line_SHA_verified": True})
    with (output / "root_reviewed.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for decision in decisions:
            stream.write(json.dumps(decision, ensure_ascii=False, allow_nan=False) + "\n")
    receipt = {"root_reviewed_QA": 5, "source_memberships_verified": 6,
        "source_files_checked": len(wanted), "new_API_calls": 0, "GPU_initialized": False,
        "source_packet": str(pending.relative_to(ROOT)), "source_packet_sha256": file_hash(pending),
        "decision_file_sha256": file_hash(output / "root_reviewed.jsonl"),
        "root_author_model": "gpt-6.1-sol", "root_author_effort": "max", "prepared_by": "/root/assets"}
    save_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def verify_reuse(args):
    previous, current = within(ROOT, args.previous_score), within(ROOT, args.current_score)
    old = {record["key"]: record for _, record, _ in rows(previous / "score_rows.jsonl.gz")}
    new = {record["key"]: record for _, record, _ in rows(current / "score_rows.jsonl.gz")}
    if len(old) != 12120 or old.keys() != new.keys():
        raise ValueError("Final native member keys differ")
    fixed = ("model", "sample_id", "qa_key", "question", "answer", "target_class", "uniform_reference",
        "canonical_name_in_primary_score", "abstain", "seed", "config", "prompt", "reference_prompt",
        "source_path", "source_line", "source_identity", "raw_line_sha256", "raw_source_sha256")
    for key in old:
        if any(old[key][field] != new[key][field] for field in fixed):
            raise ValueError("Literal closure changed primary outcome or source identity")
    changed = [key for key in old if old[key]["literal_extracted_name_score"] != new[key]["literal_extracted_name_score"]]
    if (len(changed) != 6 or any(r["literal_extracted_name_score"] is None for r in new.values())
            or any(old[key]["literal_extracted_name_score"] is not None for key in changed)):
        raise ValueError("Literal closure differs from the six-source exact increment")
    for key in changed:
        name, canonical, expected_literal, _side, _reason = ROOT_DECISIONS[new[key]["answer"]]
        if (new[key]["target_class"] != canonical or new[key]["literal_extracted_name_score"] != expected_literal
                or new[key]["literal_extracted_names"] != [name]):
            raise ValueError("Full literal extraction differs from the actual root decision")
    before = pd.read_csv(previous / "all_core_conditions_same_eval.csv")
    after = pd.read_csv(current / "all_core_conditions_same_eval.csv")
    common = [column for column in before if column != "literal_pending"]
    pd.testing.assert_frame_equal(before[common], after[common], check_exact=True)
    if len(after) != 362 or not after.primary_complete.all() or not after.literal_complete.all():
        raise ValueError("Complete primary or literal condition count differs")
    if not all(after.literal_correct + after.literal_incorrect == after.n):
        raise ValueError("Full literal count denominator differs")
    output = within(ROOT, args.output)
    output.mkdir(parents=True, exist_ok=False)
    analyses, figures = within(ROOT, args.analysis), within(ROOT, args.figures)
    approved = {
        analyses: ("native_paired_statistics.json", "native_transitions_reference_stratified.csv", "comparison_identity_map.csv", "analysis_receipt.json"),
        figures: ("native_accuracy.png", "native_accuracy.pdf", "native_accuracy_difference_ci.png", "native_accuracy_difference_ci.pdf", "figure_receipt.json", "visual_acceptance.json"),
    }
    copied = []
    for folder, files in approved.items():
        target = output / ("analysis" if folder == analyses else "figures")
        target.mkdir(exist_ok=False)
        for name in files:
            source = folder / name
            shutil.copy2(source, target / name)
            if file_hash(source) != file_hash(target / name):
                raise ValueError("Copied immutable primary analysis differs")
            copied.append({"source": str(source.relative_to(ROOT)), "copied_path": str((target / name).relative_to(ROOT)),
                "sha256": file_hash(source), "reused_without_recomputing": True})
    after[after.source_namespace.eq("P1_NATIVE_VCD")].to_csv(output / "native_conditions_complete_literal.csv", index=False)
    after.to_csv(output / "all_core_conditions_complete_literal.csv", index=False)
    receipt = {"schema": "kdm_core_literal_closure_and_primary_reuse_v1", "passed": True,
        "previous_score_sha256": file_hash(previous / "score_rows.jsonl.gz"),
        "current_score_sha256": file_hash(current / "score_rows.jsonl.gz"),
        "native_rows": 12120, "literal_updated_rows": len(changed), "literal_updated_QA": 5,
        "canonical_pending": 0, "abstain_pending": 0, "literal_pending": 0,
        "primary_row_and_identity_changes": 0, "complete_conditions": 362,
        "previous_primary_table_changes": 0, "new_bootstrap_calculations": 0,
        "reused_primary_products": copied, "new_API_calls": 0, "GPU_initialized": False,
        "literal_native_counts": {model: {"correct": sum(r["literal_extracted_name_score"] == 1 for r in new.values() if r["model"] == model),
            "incorrect": sum(r["literal_extracted_name_score"] == 0 for r in new.values() if r["model"] == model)}
            for model in sorted({r["model"] for r in new.values()})}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = subs.add_parser("prepare")
    prepare_parser.add_argument("--pending", required=True)
    prepare_parser.add_argument("--output", required=True)
    verify_parser = subs.add_parser("verify-reuse")
    for flag in ("previous-score", "current-score", "analysis", "figures", "output"):
        verify_parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else verify_reuse)(args)


if __name__ == "__main__":
    main()
