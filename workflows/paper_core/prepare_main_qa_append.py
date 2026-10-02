#!/usr/bin/env python3
"""Bind explicitly accepted full-QA reviews to finite main source rows."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from kdm.io import file_hash, within
from workflows.main_results import score as shared
from workflows.paper_core.assemble_nine_main_food import output_json
from workflows.supplemental.remaining11.score import infer_qa, rows, score_target

FROZEN_SOURCES = {"core_native_direct5", "core_native_VCD5"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--decision-file", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = within(ROOT, args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest["schema"] != "kdm_nine_main_food_explicit_sources_v1":
        raise ValueError("The explicit finite main source manifest is required")
    output = within(ROOT, args.output)
    output.relative_to(ROOT / "outputs/paper_core_20261002_dev_viz")
    decisions, decision_sources, bindings = {}, [], {}
    for relative in args.decision_file:
        path = within(ROOT, relative)
        digest = file_hash(path)
        decision_sources.append({"path": str(path.relative_to(ROOT)), "sha256": digest})
        for line, decision, line_digest in rows(path):
            key = decision["qa_key"]
            if (key != shared.qah(decision["question"], decision["answer"])
                    or type(decision.get("abstain")) is not bool
                    or decision.get("needs_root", False)):
                raise ValueError("An explicitly accepted full-QA review is unresolved")
            decisions[key] = decision
            bindings[key] = {"accepted_decision_path": str(path.relative_to(ROOT)),
                             "accepted_decision_sha256": digest,
                             "accepted_decision_line": line,
                             "accepted_decision_line_sha256": line_digest}
    classes = sorted({sample["class"] for _, sample, _ in rows(ROOT / "data/current/all.jsonl")
                      if sample["dataset"] == "food101"})
    if len(classes) != 101:
        raise ValueError("The registered Food class vocabulary differs")
    patterns = shared.compile_classes(classes)
    corrections, pending, affected = [], [], set()
    counts, checked = Counter(), 0
    source_inputs = []
    for source in manifest["sources"]:
        if source["id"] in FROZEN_SOURCES:
            continue
        path = within(ROOT, source["path"])
        if file_hash(path) != source["expected_sha256"]:
            raise ValueError("An explicit immutable main score source changed")
        source_inputs.append({"id": source["id"], "path": str(path.relative_to(ROOT)),
                              "sha256": source["expected_sha256"]})
        for line, original, line_digest in rows(path):
            key = original.get("qa_key")
            decision = decisions.get(key)
            if decision is None or original.get("dataset", "food101") != "food101":
                continue
            if (original.get("question") != decision["question"]
                    or original.get("answer") != decision["answer"]):
                raise ValueError("An accepted QA hash does not bind its complete original content")
            checked += 1
            inferred = infer_qa(decision["question"], decision["answer"], patterns,
                                {}, {}, {key: decision})
            canonical, literal, reason = score_target(
                decision["answer"], original["target_class"], inferred, patterns)
            abstain = inferred["abstain"]
            if canonical is None or literal is None or type(abstain) is not bool:
                pending.append({"source": source["id"], "line": line,
                                "qa_key": key, "reason": reason})
                continue
            old = (original.get("canonical_name_in_primary_score", original.get("correct_canonical")),
                   original.get("literal_extracted_name_score", original.get("correct_literal")),
                   original.get("abstain"))
            if old == (canonical, literal, abstain):
                continue
            corrections.append({"old_score_source": str(path), "old_score_physical_line": line,
                                "old_score_line_sha256": line_digest,
                                "old_score_source_sha256": source["expected_sha256"],
                                "qa_key": key, "question": decision["question"],
                                "answer": decision["answer"], "target_class": original["target_class"],
                                "old_canonical": old[0], "old_literal": old[1], "old_abstain": old[2],
                                "new_canonical": canonical, "new_literal": literal, "new_abstain": abstain,
                                "score_reason": reason, "actual_root_decision": decision,
                                "original_source_unchanged": True, **bindings[key]})
            affected.add(key)
            counts[source["id"]] += 1
    if pending:
        raise ValueError("Accepted finite decisions still need name/behavior closure: " + str(pending))
    output.mkdir(parents=True, exist_ok=False)
    path = output / "accepted_QA_corrections.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        for value in corrections:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    manifest["accepted_overlays"] = [{"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}]
    manifest["prior_source_manifest"] = {"path": str(manifest_path.relative_to(ROOT)),
                                         "sha256": file_hash(manifest_path), "original_unchanged": True}
    manifest["actual_accepted_decision_sources"] = decision_sources
    output_json(output / "source_manifest.json", manifest)
    receipt = {"schema": "kdm_explicit_accepted_QA_overlay_preparation_v1", "passed": True,
               "checked_exact_QA_members": checked, "corrected_members": len(corrections),
               "corrected_unique_QA": len(affected), "counts_by_source": dict(counts),
               "finite_accepted_QA": len(decisions), "unresolved_accepted_decision_members": 0,
               "actual_accepted_decision_sources": decision_sources, "source_inputs": source_inputs,
               "frozen_core_and_original_score_objects_modified": 0,
               "new_semantic_judgments_generated": 0, "GPU_initialized": False,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "scorer_sha256": file_hash(ROOT / "workflows/main_results/score.py"),
               "runner_sha256": file_hash(Path(__file__)), "actual_command": [sys.executable, *sys.argv],
               "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    output_json(output / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("passed", "checked_exact_QA_members", "corrected_members",
                                             "corrected_unique_QA", "counts_by_source")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
