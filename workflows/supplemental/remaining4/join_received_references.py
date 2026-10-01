"""Append accepted Food references to finite received score cohorts, preserving labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import file_hash, stable_hash, within
from workflows.supplemental.remaining11.generate import condition_of
from workflows.supplemental.remaining11.score import rows
from workflows.supplemental.remaining4.score_native import save_json, save_rows


def checked_rows(directory, filename, receipt_name="receipt.json"):
    receipt_path = directory / receipt_name
    receipt = json.loads(receipt_path.read_text())
    path = directory / filename
    sha = file_hash(path)
    if receipt.get("passed") is not True or receipt["outputs"][filename] != sha:
        raise ValueError("The finite source differs from its passed actual receipt")
    return path, receipt, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", action="append", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = within(ROOT, args.output)
    output.relative_to(ROOT / "outputs/supplemental/remaining4")
    reference_dir = within(ROOT, args.reference_dir)
    reference_path, reference_receipt, reference_sha = checked_rows(reference_dir, "reference_G.jsonl")
    references = {(r["model"], r["sample_id"]): r for _, r, _ in rows(reference_path)}
    if (len(references) != 19392 or reference_receipt["reference_complete"] != 19392
            or reference_receipt["reference_pending"]
            or any(type(r["reference_G"]) is not bool or not r["reference_complete"] for r in references.values())):
        raise ValueError("The accepted Food reference source must have 19392 complete unique keys")
    domains = defaultdict(set)
    for _, sample, _ in rows(ROOT / "data/current/all.jsonl"):
        if sample["dataset"] == "food101":
            domains[sample["split"]].add(sample["id"])
    if set(map(len, domains.values())) != {2424}:
        raise ValueError("The original Food dev/eval domains differ")
    joined, seen, inputs, groups, counts = [], set(), [], defaultdict(list), Counter()
    changed_reference_fields = {"reference_G", "reference_complete"}
    for relative in args.score_dir:
        directory = within(ROOT, relative)
        path, receipt, sha = checked_rows(directory, "score_rows.jsonl.gz")
        cohort_n = 0
        for line, original, line_sha in rows(path):
            if original["key"] in seen:
                raise ValueError("Received cohorts repeat a generation key")
            seen.add(original["key"])
            record = dict(original)
            if record["dataset"] == "food101":
                reference = references[(record["model"], record["sample_id"])]
                if (reference["split"] != record["split"] or reference["target_class"] != record["target_class"]
                        or record["canonical_name_in_primary_score"] not in (0, 1)
                        or record["literal_extracted_name_score"] not in (0, 1)
                        or type(record["abstain"]) is not bool):
                    raise ValueError("Food score/reference identity or resolved fields differ")
                record.update(reference_G=reference["reference_G"], reference_complete=True,
                    reference_join_source_path=str(reference_path.relative_to(ROOT)),
                    reference_join_source_sha256=reference_sha, gold_rank=reference["gold_rank"],
                    correct_count=reference["correct_count"], reference_join_original_score_path=str(path.relative_to(ROOT)),
                    reference_join_original_score_line=line, reference_join_original_score_line_sha256=line_sha,
                    reference_join_original_fields={k: original.get(k) for k in changed_reference_fields})
                if any(record[k] != v for k, v in original.items() if k not in changed_reference_fields):
                    raise ValueError("Reference join changed an original score or behavior field")
                context = condition_of(record["model"], record["dataset"], record["stage"],
                    {"sample": {"split": record["split"]}, **record})
                group_key = stable_hash({"condition": context, "decode_config": record["config"]})
                groups[group_key].append((record, context))
                counts["Food_reference_joined"] += 1
            elif record["dataset"] == "vizwiz":
                if record != original:
                    raise ValueError("Viz objects must remain exactly unchanged by the Food-only join")
                counts["Viz_objects_identical"] += 1
            else:
                raise ValueError("Received score dataset is outside the selected scope")
            joined.append(record)
            cohort_n += 1
        if cohort_n != receipt["rows"]:
            raise ValueError("Actual cohort rows differ from their passed receipt")
        inputs.append({"path": str(path.relative_to(ROOT)), "sha256": sha, "rows": cohort_n,
            "receipt_sha256": file_hash(directory / "receipt.json")})
    coverage = []
    for group_id, members in sorted(groups.items()):
        records, context = [r for r, _ in members], members[0][1]
        ids = {r["sample_id"] for r in records}
        if len(ids) != len(records):
            raise ValueError("A complete condition repeats a Food input across received cohorts")
        full = ids == domains[records[0]["split"]]
        quota = Counter(r["target_class"] for r in records)
        if full and (len(quota) != 101 or set(quota.values()) != {24}):
            raise ValueError("A claimed full Food condition lacks 101 x 24 coverage")
        coverage.append({"condition_id": group_id, **context, "decode_config": records[0]["config"],
            "received_rows": len(records), "expected_full_inputs": 2424, "full_input_coverage": full,
            "received_classes": len(quota), "canonical_literal_abstain_pending": 0,
            "reference_pending": 0, "partial_cohort_used_as_main_comparison": False})
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "score_rows.jsonl.gz", joined)
    save_rows(output / "condition_coverage.jsonl", coverage)
    receipt = {"schema": "kdm_received_food_reference_join_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "rows": len(joined), "unique_keys": len(seen),
        "counts": dict(counts), "source_score_inputs": inputs, "reference_source_sha256": reference_sha,
        "reference_receipt_sha256": file_hash(reference_dir / "receipt.json"),
        "reference_label_acceptance_status": "accepted", "original_scores_and_abstentions_unchanged": True,
        "complete_received_Food_conditions": sum(r["full_input_coverage"] for r in coverage),
        "partial_denominators_promoted_to_full": False, "new_scientific_annotations": 0,
        "GPU_initialized": False, "raw_reopened": False, "runner_sha256": file_hash(Path(__file__)),
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("passed", "rows", "unique_keys", "counts", "complete_received_Food_conditions")}))


if __name__ == "__main__":
    main()
