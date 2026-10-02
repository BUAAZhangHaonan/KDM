"""Validate a newly closed finite Food cohort and bind its review append plan."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from kdm.decoding import DecodeConfig
from kdm.io import file_hash, stable_hash, stable_seed, within
from kdm.pipeline import task_id
from workflows.supplemental.remaining11.generate import condition_of
from workflows.supplemental.remaining11.score import rows
from workflows.supplemental.remaining4.score_received import finite
from workflows.paper_core.append_core_review import module_from_package

BASE = ROOT / "outputs/paper_core_20261002_dev_viz"
RUN = ROOT / "outputs/supplemental/remaining4/food53671_cpu_20261002_2140"
MANIFEST = ROOT / "outputs/supplemental/remaining4/food_formal_sealed_gap_20261002_1930/score_inputs_v3/new_parts_received_manifest.json"
AUTHORITY = BASE / "annotation/food476_final_effective_20261002_2140"
FIELDS = ("method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    require(not path.exists(), "This finite CPU receipt or append plan must be exclusive")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source(directory):
    receipt_path = directory / "receipt.json"
    receipt = load(receipt_path)
    path = directory / "score_rows.jsonl.gz"
    require(receipt["passed"] and receipt["rows"] == receipt["unique_keys"] == 53671
            and file_hash(path) == receipt["outputs"][path.name], "Finite accepted score/source coverage differs")
    return path, receipt, receipt_path


def prepare():
    output = BASE / "food53671_append_preparation_20261002_2140"
    require(not output.exists(), "The append preparation directory must be exclusive")
    manifest = load(MANIFEST)
    require(manifest["rows"] == 53671 and len(manifest["parts"]) == 106
            and manifest["local_source_sha_verified"] and manifest["active_raw_opened"] == 0,
            "This is not the explicit 106-part immutable finite source")
    score_path, score, score_receipt_path = source(RUN / "scores/v1_food476_root_closed")
    require(score["parts"] == 106 and score["source_manifest_sha256"] == file_hash(MANIFEST)
            and score["label_completion_claimed"] and score["boundary_unique_QA"] == 0
            and all(score["counts"].get(k, 0) == 0 for k in
                    ("food_canonical_pending", "food_literal_pending", "abstain_pending", "label_pending_members")),
            "The actual Food score is not entirely closed")
    joined_path, joined, joined_receipt_path = source(RUN / "reference/v1_food476_accepted_v2_join")
    require(joined["original_scores_and_abstentions_unchanged"]
            and joined["counts"]["Food_reference_joined"] == 53671
            and joined["reference_source_sha256"] == "d3feac8fef2ade5967c0ff17762e15a17aac92e244f980c01d6ed360ff8d36a0",
            "The complete accepted reference source was not joined without changing scores")
    author_receipt = load(AUTHORITY / "accepted_receipt.json")
    decision_path = AUTHORITY / "effective_decisions476.jsonl"
    require(author_receipt["passed"] and author_receipt["actual_Luna_read_written"] == 476
            and author_receipt["root_actual_full_QA_read"] == 340
            and author_receipt["final_semantic_pending"] == author_receipt["final_name_evidence_issues"] == 0
            and file_hash(decision_path) == author_receipt["final_decisions_sha256"] ==
                "9aacceba91aab3b6ce536386db58012bee3da4677fea13f946d3dc426bcd4aff",
            "The actual accepted Food476 authority differs")
    samples = {r["id"]: r for _, r, _ in rows(ROOT / "data/current/all.jsonl") if r["dataset"] == "food101"}
    parts, identities, identity_members = {}, {}, []
    for part in manifest["parts"]:
        raw = str(within(ROOT, part["received_raw_path"]).relative_to(ROOT))
        require(part["source_part_generation_complete"] and part["old292541_overlap"] == 0 and raw not in parts,
                "A received source part is incomplete or overlaps the old cohort")
        identity_path = within(ROOT, part["received_identity_path"])
        identity = load(identity_path)
        digest = file_hash(identity_path)
        require(digest == part["identity_sha256"] and stable_hash(identity["definition"]) == identity["identity"]
                and identity["definition"]["base_config"] == asdict(DecodeConfig()),
                "Full original identity or registered decode parameters differ")
        parts[raw] = part
        identities[raw] = identity
        identity_members.append({"path": str(identity_path), "sha256": digest, "identity": identity["identity"]})
    seen, source_count, models, groups = set(), Counter(), Counter(), defaultdict(list)
    preserved = {}
    for _, row, _ in rows(score_path):
        require(row["key"] not in preserved, "Scored source repeats a task key")
        preserved[row["key"]] = row
    for _, row, _ in rows(joined_path):
        require(row["key"] not in seen and row["source_path"] in parts, "Joined key repeats or has an unbound source")
        sample = samples[row["sample_id"]]
        identity = identities[row["source_path"]]
        task = {"sample": sample, **{field: row[field] for field in FIELDS}}
        require(row["key"] == task_id(row["model"], task) and row["source_identity"] == identity["identity"]
                and row["seed"] == stable_seed(sample["id"], row["model"], row["replicate"])
                and row["split"] == sample["split"] == "eval" and row["dataset"] == "food101"
                and row["target_class"] == sample["class"] and row["question"] == sample["question"]
                and row["canonical_name_in_primary_score"] in (0, 1) and row["literal_extracted_name_score"] in (0, 1)
                and type(row["abstain"]) is bool and row["reference_complete"] is True
                and type(row["reference_G"]) is bool and finite(row),
                "A task, original identity, seed, finite value, resolved label or complete reference differs")
        registered = {**identity["definition"]["base_config"], "method": row["method"]}
        if row["method"] == "instruction_m3id":
            require(isinstance(row["config"]["m3id_offset"], int) and row["config"]["m3id_offset"] > 0,
                    "The actual source-bound instruction-M3ID offset is absent")
            registered["m3id_offset"] = row["config"]["m3id_offset"]
        require(registered == row["config"], "Original actual decode config differs from the registered parameters")
        original = preserved[row["key"]]
        require(all(row[field] == value for field, value in original.items()
                    if field not in ("reference_G", "reference_complete")), "Reference join changed a scored field")
        seen.add(row["key"])
        source_count[row["source_path"]] += 1
        models[row["model"]] += 1
        context = condition_of(row["model"], row["dataset"], row["stage"], {"sample": {"split": "eval"}, **row})
        groups[stable_hash({"condition": context, "config": row["config"]})].append(row)
    require(len(seen) == 53671 and all(source_count[path] == part["rows"] for path, part in parts.items()),
            "Actual bounded per-part key coverage differs from its immutable manifest")
    output.mkdir(parents=True)
    verification = {"schema": "kdm_food53671_finite_source_score_reference_verification_v1", "passed": True,
        "rows": len(seen), "unique_keys": len(seen), "parts": len(parts), "per_model": dict(models),
        "canonical_literal_abstain_pending": 0, "reference_pending": 0, "duplicate_or_missing_keys": 0,
        "all_original_score_fields_preserved_after_reference_join": True,
        "original_identity_seed_key_config_finite_values_verified": True,
        "source_bound_dynamic_instruction_M3ID_offset_preserved": True,
        "all_reference_complete": True, "Food_condition_groups": len(groups),
        "complete_received_Food_conditions": joined["complete_received_Food_conditions"],
        "manifest_path": str(MANIFEST), "manifest_sha256": file_hash(MANIFEST),
        "score_source_sha256": score["outputs"]["score_rows.jsonl.gz"],
        "joined_source_sha256": joined["outputs"]["score_rows.jsonl.gz"],
        "score_receipt_sha256": file_hash(score_receipt_path), "join_receipt_sha256": file_hash(joined_receipt_path),
        "accepted_decisions_sha256": file_hash(decision_path), "full_identity_sidecars": len(identity_members),
        "old292541_source_read": False, "raw_reopened": False, "GPU_initialized": False,
        "new_semantic_judgments": 0, "source_sha256": file_hash(Path(__file__)),
        "completed_utc": datetime.now(timezone.utc).isoformat()}
    write(output / "finite_verification.json", verification)
    group = {"id": "extended_food53671", "kind": "scored_records", "status": "accepted", "expected_rows": 53671,
        "source_path": str(joined_path), "source_sha256": joined["outputs"]["score_rows.jsonl.gz"],
        "acceptance_receipt_path": str(joined_receipt_path), "acceptance_receipt_sha256": file_hash(joined_receipt_path),
        "acceptance_checks": {"passed": True, "rows": 53671, "unique_keys": 53671,
                              "original_scores_and_abstentions_unchanged": True, "reference_label_acceptance_status": "accepted"},
        "identity_members": identity_members, "unique_key_fields": ["key"],
        "require_decided_fields": ["canonical_name_in_primary_score", "literal_extracted_name_score", "abstain", "reference_G"],
        "exact_row_fields": {"dataset": "food101", "split": "eval", "reference_complete": True}}
    write(output / "actual_food53671_append_plan.json", {"groups": [group]})
    print(json.dumps(verification, ensure_ascii=False))


def verify_compact(package, destination):
    source_path = RUN / "reference/v1_food476_accepted_v2_join/score_rows.jsonl.gz"
    compact = package / "appended/append_food53671_v8/compact/extended_food53671"
    names = ("condition", "sample", "qa", "identity", "source", "label")
    dictionaries = {name: {row[name + "_id"]: json.loads(row["original_json"])
                          for row in pq.read_table(compact / (name + "_dictionary.parquet")).to_pylist()} for name in names}
    ledger = pq.read_table(compact / "scores.parquet").to_pylist()
    builder = module_from_package(package)
    compact_receipt = load(compact / "receipt.json")
    require(compact_receipt["original_source_sha256"] == file_hash(source_path),
            "The source-only field lookup no longer points to its immutable original score")
    count = 0
    for line, original, _ in rows(source_path):
        row = ledger[line - 1]
        require(row["source_record_line"] == line, "Compact source line differs from the original accepted score")
        payloads = {name: dictionaries[name][row[name + "_id"]] for name in names}
        restored = {key: value for name in names for key, value in payloads[name].items()
                    if key != "scalar_original_present_fields"}
        restored.update({field: row[field] for field in payloads["label"]["scalar_original_present_fields"]})
        require(restored == builder.remove_digests(original, "row", set()),
                "Actual compact scientific fields, full QA or source identity/config differ")
        fully_restored = restore_indexed_source_fields(restored, original, builder)
        require(fully_restored == original, "Indexed source-only fields do not reconstruct the complete original record")
        count += 1
    require(count == len(ledger) == 53671, "Actual reversible compact table has incomplete input coverage")
    result = {"passed": True, "actual_reconstructed_rows": count, "all_original_fields_reconstructed_equal": True,
              "all_original_fields_fully_embedded": not compact_receipt["source_only_digest_or_identifier_fields"],
              "all_non_digest_original_fields_compactly_embedded_equal": True,
              "full_original_reconstruction_requires_bound_server_score_lines": True,
              "complete_QA_and_identity_config_source_labels_equal": True, "GPU_initialized": False,
              "source_only_digest_fields_restored_from_actual_immutable_score_lines": True,
              "indexed_source_only_fields": compact_receipt["source_only_digest_or_identifier_fields"],
              "compact_path": str(compact), "compact_scores_sha256": file_hash(compact / "scores.parquet"),
              "accepted_original_score_sha256": file_hash(source_path), "runner_sha256": file_hash(Path(__file__))}
    write(destination, result)
    print(json.dumps(result))


def restore_indexed_source_fields(compact, original, builder):
    """Retrieve only the declared original extrinsic fields at the bound source line."""
    if isinstance(original, dict):
        require(isinstance(compact, dict), "An original source object changed its type")
        result = {}
        for field, value in original.items():
            if field in compact:
                result[field] = restore_indexed_source_fields(compact[field], value, builder)
            else:
                require(field in builder.RECORD_ONLY_FIELDS or field.endswith("sha256") or field == "line_sha256",
                        "A scientific field was omitted from the actual compact source")
                result[field] = value
        require(set(compact) <= set(original), "Compact source introduced an original field")
        return result
    if isinstance(original, list):
        require(isinstance(compact, list) and len(compact) == len(original), "An original source list changed")
        return [restore_indexed_source_fields(a, b, builder) for a, b in zip(compact, original)]
    require(compact == original, "An original scientific source value changed")
    return compact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-compact")
    parser.add_argument("--verification-output", default=str(
        BASE / "food53671_append_preparation_20261002_2140/actual_compact_reversible_verification.json"))
    args = parser.parse_args()
    if args.verify_compact:
        verify_compact(within(ROOT, args.verify_compact), within(ROOT, args.verification_output))
    else:
        prepare()
