#!/usr/bin/env python3
"""Deduplicate finite full-QA Viz queues into mutually exclusive annotation work."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import file_hash, within
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.score import rows
from workflows.supplemental.remaining4.score_native import save_json, save_rows

ATOMIC = re.compile(r"[A-Za-z0-9]+|(?:[$£€¥₹₩]\s*)?[+-]?\d+(?:(?:[,\.]\d+)+)?(?:\s*[$£€¥₹₩%])?")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", action="append", required=True)
    parser.add_argument("--exclude-owned", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = within(ROOT, args.output)
    output.relative_to(ROOT / "outputs/supplemental/remaining4")
    excluded, sources = set(), []
    for relative in args.exclude_owned:
        path = within(ROOT, relative)
        keys = set()
        for _line, row, _sha in rows(path):
            if row["qa_key"] != frozen.qah(row["question"], row["answer"]):
                raise ValueError("An existing owner claim has different complete-QA content")
            keys.add(row["qa_key"])
        excluded.update(keys)
        sources.append({"path": relative, "sha256": file_hash(path), "unique_QA": len(keys),
                        "role": "owned_elsewhere; no final-label completion inferred"})
    union, member_keys, counts = {}, set(), Counter()
    for relative in args.queue:
        path = within(ROOT, relative)
        receipt_path = path.parent / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        sha = file_hash(path)
        if not receipt["passed"] or receipt["outputs"].get(path.name) != sha:
            raise ValueError("A finite annotation queue differs from its actually passed receipt")
        source_n = 0
        for _line, row, _sha in rows(path):
            source_n += 1
            key = frozen.qah(row["question"], row["answer"])
            if key != row["qa_key"]:
                raise ValueError("An annotation queue contains a different complete-QA binding")
            if key in excluded:
                counts["excluded_source_QA_entries_owned_elsewhere"] += 1
                continue
            qa = union.setdefault(key, {"qa_key": key, "question": row["question"],
                "answer": row["answer"], "dataset": "vizwiz", "source_memberships": [],
                "source_queues": [], "label": None, "abstain": None, "answer_text": None})
            if (qa["question"], qa["answer"]) != (row["question"], row["answer"]):
                raise ValueError("The same QA hash has different actual original content")
            qa["source_queues"].append(relative)
            for original in row["source_memberships"]:
                member = dict(original)
                member["raw_line_sha256"] = member.get("raw_line_sha256", member.get("source_line_sha256"))
                if not member["raw_line_sha256"] or not member["source_identity"]:
                    raise ValueError("An annotation membership lacks the original identity/line evidence")
                mkey = (member["model"], member["key"], member["source_identity"])
                if mkey in member_keys:
                    raise ValueError("The finite source queues repeat a previously owned raw key")
                member_keys.add(mkey)
                qa["source_memberships"].append(member)
        sources.append({"path": relative, "sha256": sha, "unique_QA_entries": source_n,
                        "receipt_sha256": file_hash(receipt_path), "role": "finite_pending_full_QA"})
    atomic, complex_rows = [], []
    for key, row in sorted(union.items()):
        subset = atomic if ATOMIC.fullmatch(row["answer"].strip()) else complex_rows
        row["owner_queue"] = "atomic" if subset is atomic else "complex"
        row["owner_index"] = len(subset)
        subset.append(row)
    if {row["qa_key"] for row in atomic} & {row["qa_key"] for row in complex_rows}:
        raise ValueError("The actual annotation owners overlap")
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "atomic_complete_QA.jsonl", atomic)
    save_rows(output / "complex_complete_QA.jsonl", complex_rows)
    receipt = {"schema": "kdm_finite_Viz_annotation_owner_queues_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "source_files": sources,
        "unique_QA": len(union), "source_memberships": len(member_keys),
        "atomic_QA": len(atomic), "complex_QA": len(complex_rows),
        "QA_owned_elsewhere": len(excluded), "counts": dict(counts),
        "atomic_memberships": sum(len(r["source_memberships"]) for r in atomic),
        "complex_memberships": sum(len(r["source_memberships"]) for r in complex_rows),
        "routing_pattern_only": ATOMIC.pattern, "new_scientific_labels_created": 0,
        "atomic_answers_defaulted_correct_or_answering": False, "full_QA_retained": True,
        "active_raw_opened": 0, "GPU_initialized": False, "API_calls": 0,
        "runner_sha256": file_hash(Path(__file__)),
        "outputs": {p.name: file_hash(p) for p in output.iterdir()}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("passed", "unique_QA", "source_memberships", "atomic_QA", "complex_QA")}))


if __name__ == "__main__":
    main()
