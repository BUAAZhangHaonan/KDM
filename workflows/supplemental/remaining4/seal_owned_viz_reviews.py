"""Seal an actual Viz review application within its immutable member-key scope.

The mature QA scorer supplies the updated objects. This CPU-only merge selects
those objects for declared pending members and preserves every other source row.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import file_hash, within
from workflows.main_results.score import qah
from workflows.supplemental.remaining4.apply_viz_reviews import FIELDS
from workflows.supplemental.remaining4.score_native import save_json


def require(test, reason):
    if not test:
        raise ValueError(reason)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for line, raw in enumerate(stream, 1):
            yield line, json.loads(raw), raw


def rel(path):
    return str(path.relative_to(ROOT))


def pointer(path):
    return {"path": rel(path), "sha256": file_hash(path)}


def checked(directory, filename):
    receipt = load(directory / "receipt.json")
    require(receipt["passed"] and receipt["outputs"][filename] == file_hash(directory / filename),
            "An immutable passed source changed: " + rel(directory / filename))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("score-dir", "review-dir", "prepare", "source-validation", "authority", "output"):
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only application must explicitly hide GPUs")
    paths = {k: within(ROOT, getattr(args, k.replace("-", "_"))) for k in
             ("score-dir", "review-dir", "prepare", "source-validation", "authority", "output")}
    output = paths["output"]
    output.relative_to(ROOT / "outputs/supplemental/remaining4")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    original = paths["score-dir"] / "score_rows.jsonl.gz"
    candidate = paths["review-dir"] / "score_rows.jsonl.gz"
    original_receipt = checked(paths["score-dir"], original.name)
    candidate_receipt = checked(paths["review-dir"], candidate.name)
    prep_receipt = checked(paths["prepare"], "source_memberships.jsonl")
    source_validation = load(paths["source-validation"])
    require(source_validation["passed"] and source_validation["source_bound_unique_QA"] == 500
            and source_validation["source_bound_unique_generation_keys"] == 1023
            and source_validation["actual_root_full_QA_read"] == 48
            and source_validation["actual_root_ordinary_sample_read"] == 19
            and source_validation["ordinary_sample_issues"] == 0,
            "The actual provider/root source validation differs from the finite accepted scope")
    for batch in source_validation["actual_provider_sources"].values():
        for proof in batch.values():
            require(file_hash(within(ROOT, proof["path"])) == proof["sha256"], "An actual provider proof changed")
    for field in ("root_source", "root_receipt", "prepared_v3_receipt", "root_bound_source"):
        proof = source_validation[field]
        require(file_hash(within(ROOT, proof["path"])) == proof["sha256"], "An actual root/preparation source changed")
    authority_path = paths["authority"]
    authority_receipt = checked(authority_path.parent, authority_path.name)
    require(authority_receipt["closed_QA"] == authority_receipt["unique_QA"] == 500
            and authority_receipt["pending_QA"] == 0
            and authority_receipt["source_binding_validation"] == pointer(paths["source-validation"])
            and candidate_receipt["authority_sha256"] == file_hash(authority_path)
            and candidate_receipt["input_cohorts"] == [{"path": rel(original), "sha256": file_hash(original),
                "rows": original_receipt["rows"], "receipt_sha256": file_hash(paths["score-dir"] / "receipt.json")}],
            "The mature score candidate does not bind this accepted500 view and frozen score cohort")
    authority = {r["qa_key"]: r for _, r, _ in rows(authority_path)}
    require(len(authority) == 500 and all(r["annotation_complete"] and r["quality_span_resolved"]
            and r["behavior_resolved"] and r["source_annotation_provenance_valid"] for r in authority.values()),
            "An actual500 authority row remains unresolved")
    for proof_name in ("parent_full_authority", "parent_authority_receipt"):
        proof = authority_receipt[proof_name]
        require(file_hash(within(ROOT, proof["path"])) == proof["sha256"], "The full merged authority source changed")
    members_path = paths["prepare"] / "source_memberships.jsonl"
    members = {r["key"]: r for _, r, _ in rows(members_path)}
    require(len(members) == prep_receipt["source_bound_unique_keys"] == 1023
            and {m["qa_key"] for m in members.values()} == set(authority), "The accepted actual member scope changed")
    delta_source = paths["review-dir"] / "actual_review_delta.jsonl.gz"
    checked(paths["review-dir"], delta_source.name)
    delta = {r["key"]: (r, raw) for _, r, raw in rows(delta_source)}
    require(set(members) <= set(delta), "The mature review application missed an owned generation member")
    changed, unchanged, food, models, matched, seen = set(), 0, 0, Counter(), set(), set()
    extra_unchanged_semantics, old_pending, new_pending, counts = 0, Counter(), Counter(), Counter()
    unapplied_conflicts = []
    special_root = next(r for _, r, _ in rows(within(ROOT, source_validation["root_source"]["path"]))
                        if (r["batch"], r["candidate_index"]) == (0, 157))
    special_authority = authority[special_root["qa_key"]]
    require(special_authority["actual_annotation"]["answer_text"] == special_root["answer_text"]
            and special_authority["reported_source_answer_text"] == special_root["answer_text"]
            and special_authority["answer"] == special_root["answer"], "The actual partial-inability span or reply was lost")
    special_member_count = 0
    scored_path = output / "score_rows.jsonl.gz"
    delta_path = output / "actual_review_delta.jsonl.gz"
    with gzip.open(scored_path, "xb") as score_out, gzip.open(delta_path, "xb") as delta_out:
        for before, after in zip_longest(rows(original), rows(candidate)):
            require(before is not None and after is not None, "The mature candidate changed source length")
            line, old, original_bytes = before
            candidate_line, new, candidate_bytes = after
            key = old["key"]
            require(line == candidate_line and key == new["key"] and key not in seen,
                    "The mature candidate changed or repeated immutable generation keys")
            seen.add(key)
            require(all(old.get(k) == new.get(k) for k in set(old) | set(new) if k not in FIELDS),
                    "The mature QA scorer changed a frozen field")
            if key in members:
                member, d = members[key], delta[key][0]
                require(old["dataset"] == "vizwiz" and old["qa_key"] == member["qa_key"] in authority
                        and qah(old["question"], old["answer"]) == old["qa_key"]
                        and all(old.get(k) == member[k] for k in
                            ("model", "sample_id", "source_path", "source_line", "source_identity", "seed", "replicate"))
                        and old.get("raw_line_sha256", old.get("source_line_sha256"))
                            == member.get("raw_line_sha256", member.get("source_line_sha256"))
                        and d["input_score_line"] == line
                        and d["input_score_line_sha256"] == hashlib.sha256(original_bytes).hexdigest()
                        and d["previous"] == {k: old.get(k) for k in FIELDS}
                        and d["updated"] == {k: new.get(k) for k in FIELDS}
                        and type(new["abstain"]) is bool and new["answer_quality_credit"] is not None
                        and 0 <= new["answer_quality_credit"] <= 1,
                        "A scoped score member differs from its source, actual delta, or complete judgment")
                if key != new["key"] or old == new:
                    raise ValueError("The scoped actual member did not receive its accepted QA decision")
                score_out.write(candidate_bytes)
                delta_out.write(delta[key][1])
                changed.add(key)
                matched.add(old["qa_key"])
                models[old["model"]] += 1
                old_pending["behavior"] += old["abstain"] is None
                old_pending["quality"] += old["answer_quality_credit"] is None
                new_pending["behavior"] += new["abstain"] is None
                new_pending["quality"] += new["answer_quality_credit"] is None
                if old["qa_key"] == special_root["qa_key"]:
                    require(new["abstain"] is True and new["answer_quality_credit"] == 0.0
                            and old["answer"] == special_root["answer"], "The actual partial-inability score differs from root")
                    special_member_count += 1
                record = new
            else:
                score_out.write(original_bytes)
                record = old
                if old["dataset"] == "food101":
                    require(old == new, "A frozen Food object changed in the mature candidate")
                    food += 1
                elif old["dataset"] == "vizwiz":
                    unchanged += 1
                    require((old["abstain"] is None) == (new["abstain"] is None)
                            and (old["answer_quality_credit"] is None) == (new["answer_quality_credit"] is None),
                            "Out-of-scope candidate changes affect the pending population")
                    if key in delta:
                        semantic_fields = ("abstain", "semantic_label", "semantic_answer_text",
                                           "answer_quality_credit", "answer_quality_pending")
                        differences = {k: {"frozen": old.get(k), "new_actual_review": new.get(k)}
                                       for k in semantic_fields if old.get(k) != new.get(k)}
                        if differences:
                            unapplied_conflicts.append({"key": key, "qa_key": old["qa_key"],
                                "question": old["question"], "answer": old["answer"],
                                "model": old["model"], "sample_id": old["sample_id"],
                                "existing_score_source": {**pointer(original), "line": line,
                                    "line_sha256": hashlib.sha256(original_bytes).hexdigest()},
                                "new_actual_authority_source": new["decision_source"]["actual_authority_source"],
                                "semantic_differences": differences,
                                "frozen_generation_not_in_current_owned_pending_scope": True,
                                "new_review_applied_to_this_generation": False,
                                "GT_exposed": False, "new_semantic_judgment_by_writer": False})
                        else:
                            extra_unchanged_semantics += 1
                else:
                    raise ValueError("The received cohort has an unsupported dataset")
            if record["dataset"] == "vizwiz":
                counts["viz_rows"] += 1
                counts["viz_quality_pending"] += record["answer_quality_credit"] is None
                counts["viz_behavior_pending"] += record["abstain"] is None
    require(changed == set(members) and len(matched) == 500 and food == 177333
            and unchanged == 114185 and len(seen) == original_receipt["rows"] == 292541
            and extra_unchanged_semantics + len(unapplied_conflicts) == len(delta) - len(members) == 36
            and special_member_count > 0,
            "The actual scoped application differs from its finite complete-member plan")
    pending_source = paths["review-dir"] / "viz_pending_complete_QA.jsonl"
    checked(paths["review-dir"], pending_source.name)
    pending_QA, pending_members = set(), set()
    for _, row, _ in rows(pending_source):
        require(row["qa_key"] not in authority and row["qa_key"] not in pending_QA,
                "An accepted actual500 QA remains in the candidate pending population")
        pending_QA.add(row["qa_key"])
        for member in row["source_memberships"]:
            require(member["key"] not in members and member["key"] not in pending_members,
                    "The candidate pending members include an accepted or repeated key")
            pending_members.add(member["key"])
    require(len(pending_QA) == candidate_receipt["pending_unique_Viz_QA"] == 33812
            and len(pending_members) == counts["viz_quality_pending"] == counts["viz_behavior_pending"] == 49219,
            "The actual scoped pending population differs from the preserved candidate population")
    shutil.copyfile(pending_source, output / pending_source.name)
    with (output / "unapplied_historical_shared_QA_conflicts.jsonl").open("x", encoding="utf-8") as stream:
        for conflict in unapplied_conflicts:
            stream.write(json.dumps(conflict, ensure_ascii=False) + "\n")
    counts.update(food_objects_identical=food, actual_review_members_applied=len(changed))
    receipt = {"schema": "kdm_finite_actual_Viz_review_application_v1", "passed": True,
        "rows": len(seen), "unique_keys": len(seen), "counts": dict(counts),
        "matched_unique_QA": len(matched), "pending_unique_Viz_QA": len(pending_QA),
        "authority_path": rel(authority_path), "authority_sha256": file_hash(authority_path),
        "authority_receipt_sha256": file_hash(authority_path.parent / "receipt.json"),
        "input_cohorts": candidate_receipt["input_cohorts"], "mature_QA_score_candidate": pointer(candidate),
        "mature_QA_score_receipt": pointer(paths["review-dir"] / "receipt.json"),
        "immutable_member_scope": pointer(members_path), "actual_source_validation": pointer(paths["source-validation"]),
        "member_keys_only": True, "unapplied_historical_shared_QA_conflicts": len(unapplied_conflicts),
        "Food_fields_changed": 0, "raw_official_Viz_credit_changed": 0,
        "unmatched_objects_changed": 0, "new_scientific_judgments": 0, "raw_reopened": False,
        "GPU_initialized": False, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": file_hash(Path(__file__)),
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / "receipt.json", receipt)
    audit = {"schema": "kdm_actual500_owned_member_application_audit_v1", "passed": True,
        "accepted_actual_Luna_QA": 500, "actual_root_full_QA_read": 48,
        "actual_root_ordinary_sample_read": 19, "ordinary_sample_issues": 0,
        "actual_changed_generation_keys": len(changed), "changed_per_model": dict(models),
        "Food_identical_objects": food, "unmatched_Viz_identical_objects": unchanged,
        "unmatched_rows_preserve_original_JSONL_bytes": True,
        "out_of_scope_shared_QA_source_updates_excluded": extra_unchanged_semantics + len(unapplied_conflicts),
        "excluded_source_only_updates": extra_unchanged_semantics,
        "unapplied_historical_shared_QA_semantic_conflicts": len(unapplied_conflicts),
        "accepted_label_counts": dict(Counter(r["label"] for r in authority.values())),
        "root_selected_QA": sum(r["annotation_author"] == "/root" for r in authority.values()),
        "pending_unique_QA_before": original_receipt["pending_unique_Viz_QA"],
        "pending_unique_QA_after": len(pending_QA), "pending_unique_QA_reduction": 500,
        "behavior_pending_before": original_receipt["counts"]["viz_behavior_pending"],
        "behavior_pending_after": counts["viz_behavior_pending"],
        "quality_pending_before": original_receipt["counts"]["viz_quality_pending"],
        "quality_pending_after": counts["viz_quality_pending"],
        "owned_members_pending_before": dict(old_pending), "owned_members_pending_after": dict(new_pending),
        "partial_inability_case": {"batch": 0, "candidate_index": 157, "qa_key": special_root["qa_key"],
            "root_selected_span": special_root["answer_text"], "full_reply_preserved": True,
            "official_raw_values_preserved": True, "abstain": True, "answer_quality_credit": 0.0,
            "affected_members": special_member_count},
        "source_validation": pointer(paths["source-validation"]), "new_score": pointer(scored_path),
        "new_score_receipt": pointer(output / "receipt.json"), "finite_authority": pointer(authority_path),
        "elapsed_seconds": time.monotonic() - started, "runner_sha256": file_hash(Path(__file__)),
        "new_GT_judgments": 0, "writer_semantic_judgments": 0, "GPU_initialized": False}
    save_json(output / "application_audit.json", audit)
    save_json(output / "CURRENT_STATE.json", {"passed": True, "accepted_actual_QA": 500,
        "scores_applied_generation_keys": 1023, "root_full_QA_read": 48, "ordinary_sample_read": 19,
        "remaining_pending_QA_in_received_cohort": len(pending_QA),
        "new_score_dir": rel(output), "authority": pointer(authority_path),
        "audit": pointer(output / "application_audit.json")})
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
