"""Bind a finite provider batch and actual root reviews, then apply only its QA.

No labels or spans are chosen here. Explicit provider labels are serialized using
the existing codebook; root decisions remain separate higher-priority sources.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import file_hash, within
from workflows.main_results.score import qah
from workflows.supplemental.remaining4.score_native import save_json, save_rows

LABELS = {"answer_assertive", "answer_uncertain", "abstain", "invalid"}


def require(test, reason):
    if not test:
        raise ValueError(reason)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for number, raw in enumerate(stream, 1):
            require(bool(raw.strip()), "An immutable JSONL contains an empty line")
            yield number, json.loads(raw), hashlib.sha256(raw).hexdigest()


def rel(path):
    return str(path.relative_to(ROOT))


def pointer(path):
    return {"path": rel(path), "sha256": file_hash(path)}


def json_from_display(text, fields):
    result = []
    for line in text.splitlines():
        if line.lstrip().startswith("{"):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and set(value) == fields:
                result.append(value)
    return result


def event_text(event, field):
    blocks = event["payload"][field]
    require(isinstance(blocks, list), "The actual provider event is not a block list")
    require(all(isinstance(b.get("text"), str) for b in blocks), "Provider blocks lack text")
    return "\n".join(b["text"] for b in blocks)


def checked_outputs(directory, receipt):
    for filename, expected in receipt["outputs"].items():
        require(file_hash(directory / filename) == expected, "A prepared artifact changed: " + filename)


def run_module(module, arguments, log):
    command = [sys.executable, "-m", module, *arguments]
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
    require(process.returncode == 0, "A finite CPU step failed; evidence retained at " + rel(log))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepare", "provider", "assignment", "root-decisions", "root-receipt",
                 "base-config", "base-authority", "score-dir", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU execution must explicitly hide GPUs")
    paths = {name: within(ROOT, getattr(args, name.replace("-", "_"))) for name in
             ("prepare", "provider", "assignment", "root-decisions", "root-receipt",
              "base-config", "base-authority", "score-dir", "output")}
    out = paths["output"]
    out.relative_to(ROOT / "outputs/supplemental/remaining4")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    prep, provider = paths["prepare"], paths["provider"]
    prep_receipt = load(prep / "receipt.json")
    checked_outputs(prep, prep_receipt)
    require(prep_receipt["passed"] and prep_receipt["actual_Luna_model_rows"] == 500
            and prep_receipt["source_bound_unique_QA"] == 500
            and prep_receipt["source_bound_unique_keys"] == 1023,
            "The prepared current batch exceeds or differs from its finite scope")
    assignment = load(paths["assignment"])
    require(file_hash(paths["assignment"]) == prep_receipt["source_assignment_sha256"]
            and assignment["assigned_unique_QA"] == 500, "The current assignment changed")
    score_receipt = load(paths["score-dir"] / "receipt.json")
    require(score_receipt["passed"] and score_receipt["rows"] == 292541
            and file_hash(paths["score-dir"] / "receipt.json") == assignment["source_receipt_sha256"],
            "The immutable received score cohort changed")
    pending = within(ROOT, assignment["source"])
    require(pending.parent == paths["score-dir"] and file_hash(pending) == assignment["source_sha256"],
            "The original pending complete-QA queue changed")
    all_normalized = [r for _, r, _ in json_rows(prep / "normalized_schema_only.jsonl")]
    require(len(all_normalized) == 500, "Normalized provider count changed")
    normalized_by_index = {(r["batch"], r["candidate_index"]): r for r in all_normalized}
    require(len(normalized_by_index) == 500, "The normalized provider indices repeat")
    originals, members, proofs, batch_inputs = {}, {}, {}, []
    for batch in assignment["batches"]:
        b = batch["batch"]
        owner = within(ROOT, batch["owner_queue"])
        payload_path = within(ROOT, batch["payload"])
        require(file_hash(owner) == batch["owner_queue_sha256"]
                and file_hash(payload_path) == batch["payload_sha256"] and batch["rows"] == 250,
                "The immutable assigned owner/payload changed")
        owned = list(json_rows(owner))
        payload = [r for _, r, _ in json_rows(payload_path)]
        require(len(owned) == len(payload) == 250, "The assigned batch is not 250 complete QA")
        folder = provider / ("batch_" + str(b))
        receipt = load(folder / "provider_receipt.json")
        require(receipt["passed"] and receipt["batch"] == b
                and receipt["annotation_model"] == "gpt-5.6-luna"
                and receipt["annotation_effort"] == "medium"
                and receipt["full_QA_actually_displayed"] == receipt["explicit_model_rows"] == 250,
                "The actual provider identity or read/write coverage differs")
        proof_map = normalized_by_index[(b, 0)]["provider_proofs"]
        for filename, proof in proof_map.items():
            require(within(ROOT, proof["path"]) == folder / filename
                    and file_hash(folder / filename) == proof["sha256"], "A current provider proof changed")
        require(receipt["assigned_payload_sha256"] == batch["payload_sha256"]
                and receipt["final_sha256"] == file_hash(folder / "model_final.jsonl"),
                "The actual provider receipt does not bind the immutable display/final")
        displayed = [r for _, r, _ in json_rows(folder / "actually_displayed_QA_payload.jsonl")]
        read_event = load(folder / "read_tool_provider_event.json")
        read_text = event_text(read_event, "output")
        require(read_text == (folder / "read_tool_full_output.txt").read_text(encoding="utf-8")
                and json_from_display(read_text, {"i", "q", "r"}) == displayed == payload,
                "The provider did not display the exact full assigned questions and answers")
        final_event = load(folder / "model_final_provider_event.json")
        require(final_event["payload"]["role"] == "assistant"
                and final_event["payload"]["phase"] == "final_answer", "Actual final provider phase differs")
        literal = list(json_rows(folder / "model_final.jsonl"))
        require(json_from_display(event_text(final_event, "content"), {"i", "label", "span", "evidence", "root"})
                == [r for _, r, _ in literal] and len(literal) == 250,
                "The actual provider final is not the 250 saved explicit decisions")
        writer_rows = []
        for i, ((owner_line, target, owner_sha), (final_line, decision, final_sha)) in enumerate(zip(owned, literal)):
            require(target["candidate_index"] == decision["i"] == i
                    and payload[i] == {"i": i, "q": target["question"], "r": target["answer"]}
                    and qah(target["question"], target["answer"]) == target["qa_key"],
                    "Current provider decisions do not bind the assigned complete-QA index")
            n = normalized_by_index[(b, i)]
            require(n["model_declared_judgment"] == decision and n["qa_key"] == target["qa_key"]
                    and (n["question"], n["answer"]) == (target["question"], target["answer"])
                    and n["label"] == decision["label"] in LABELS
                    and n["answer_text"] == decision["span"] and n["evidence_span"] == decision["evidence"]
                    and n["abstain"] == (decision["label"] == "abstain")
                    and n["needs_root"] == bool(decision["root"])
                    and n["source_owned_queue_line_sha256"] == owner_sha
                    and n["source_model_final_line_sha256"] == final_sha,
                    "The prepared schema did more than the explicit authorized serialization")
            for filename, proof in n["provider_proofs"].items():
                require(proof == proof_map[filename], "Per-QA provider proof differs")
            require(target["qa_key"] not in originals, "Assigned complete-QA keys repeat")
            originals[target["qa_key"]] = target
            for member in target["source_memberships"]:
                require(member["key"] not in members, "Assigned generation member ownership repeats")
                members[member["key"]] = {"qa_key": target["qa_key"], **member}
            writer_rows.append({"candidate_index": i, "label": decision["label"],
                "abstain": decision["label"] == "abstain", "answer_text": decision["span"],
                "evidence_span": decision["evidence"], "needs_root": bool(decision["root"]),
                "derived_boolean_from_explicit_label": True, "abstain_field_present_in_original": False,
                "original_provider_judgment": decision, "original_provider_source": {
                    **pointer(folder / "model_final.jsonl"), "line": final_line, "line_sha256": final_sha},
                "schema_preparation_source": pointer(prep / "normalized_schema_only.jsonl")})
        input_dir = out / ("batch_" + str(b))
        input_dir.mkdir(exist_ok=False)
        declared = input_dir / "serialized_explicit_model_decisions.jsonl"
        save_rows(declared, writer_rows)
        attestation = {"actual_complete_QA_read_reported_by_model": True,
            **{k: receipt.get(k, "") for k in ("annotation_author", "annotation_model", "annotation_effort",
                "annotation_session_id", "annotation_call_id", "actual_read_tool_call_id")},
            "first": 0, "last": 249, "source_queue_sha256": file_hash(owner),
            "model_decisions_sha256": file_hash(declared), "actual_complete_QA_displayed": 250,
            "actual_provider_final_rows": 250, "derived_boolean_from_explicit_label": True,
            "original_model_boolean_field_present": False, "provider_proofs": proof_map,
            "writer_generated_semantic_labels_or_spans": False}
        attestation_path = input_dir / "actual_read_attestation.json"
        save_json(attestation_path, attestation)
        proofs[b] = {"provider_final": pointer(folder / "model_final.jsonl"), "provider_receipt": pointer(folder / "provider_receipt.json"),
                     "read_event": pointer(folder / "read_tool_provider_event.json"),
                     "final_event": pointer(folder / "model_final_provider_event.json"),
                     "assigned_owner": pointer(owner), "assigned_payload": pointer(payload_path)}
        batch_inputs.append((b, owner, declared, attestation_path))
    require(len(originals) == 500 and len(members) == 1023, "The finite current source membership scope changed")
    prep_members = {r["key"]: r for _, r, _ in json_rows(prep / "source_memberships.jsonl")}
    require(set(prep_members) == set(members), "Prepared ownership keys differ from assigned actual sources")
    for key, member in members.items():
        require(all(prep_members[key][k] == v for k, v in member.items()), "A prepared source member changed")
    found_pending = {}
    for line, record, line_sha in json_rows(pending):
        if record["qa_key"] in originals:
            target = originals[record["qa_key"]]
            require((record["question"], record["answer"], record["source_memberships"])
                    == (target["question"], target["answer"], target["source_memberships"])
                    and line == target["source_queue_line"] and line_sha == target["source_queue_line_sha256"],
                    "An assigned QA changed from its original pending queue")
            found_pending[record["qa_key"]] = True
    require(len(found_pending) == 500, "Actual current provider QA is not entirely in the original pending scope")
    root_receipt = load(paths["root-receipt"])
    root_source = list(json_rows(paths["root-decisions"]))
    require(root_receipt["passed"] and root_receipt["finite_root_QA_read"] == root_receipt["root_decisions"] == 48
            and root_receipt["ordinary_QA_sample_read"] == 19 and root_receipt["ordinary_sample_issues"] == 0
            and root_receipt["root_decisions_sha256"] == file_hash(paths["root-decisions"])
            and root_receipt["source_finite_root_sha256"] == file_hash(prep / "finite_root_review_full_QA.jsonl")
            and root_receipt["source_sample_sha256"] == file_hash(prep / "ordinary_i_mod25_QA_sample.jsonl"),
            "Actual root receipt does not bind the source-identical v3 review and ordinary sample")
    required_root = {(r["batch"], r["candidate_index"]) for _, r, _ in json_rows(prep / "finite_root_review_full_QA.jsonl")}
    require(len(required_root) == 47, "The finite difficult-QA root review scope changed")
    required_root.add((0, 157))
    roots, seen_root = [], set()
    for line, row, line_sha in root_source:
        index = row["batch"], row["candidate_index"]
        n = normalized_by_index.get(index)
        require(index in required_root and index not in seen_root and n is not None
                and (row["qa_key"], row["question"], row["answer"]) == (n["qa_key"], n["question"], n["answer"])
                and row["reviewed_original_provider_literal_sha256"] == n["source_model_final_sha256"]
                and row["annotation_author"] == "/root" and row["annotation_model"] == "gpt-6.1-sol"
                and row["annotation_effort"] == "max" and row["read_full_question_answer"] is True
                and row["target_or_official_answers_read"] is False
                and row["label"] in LABELS and type(row["abstain"]) is bool
                and row["abstain"] == (row["label"] == "abstain")
                and row["needs_root"] is False and row["root_review_required"] is False
                and (not row["answer_text"] or row["answer_text"] in row["answer"])
                and (not row["evidence_span"] or row["evidence_span"] in row["answer"]),
                "A root review differs from its actual source or remains unresolved")
        require(not row.get("annotation_call_id"), "An unknown root call ID was invented")
        if row["label"].startswith("answer_"):
            require(bool(row["answer_text"]), "An explicit root answer lacks its selected span")
        seen_root.add(index)
        roots.append({**row, "A": row["abstain"], "span_selection_mode": "actual_per_QA_model_choice",
            "source_root_decision": {**pointer(paths["root-decisions"]), "line": line, "line_sha256": line_sha},
            "source_root_receipt": pointer(paths["root-receipt"]),
            "source_actual_Luna_proofs": proofs[row["batch"]],
            "metadata_binding_writer": "/root/dev_viz_inventory", "metadata_binding_is_semantic_judgment": False})
    require(seen_root == required_root and len(roots) == 48, "The actual root reviews do not cover the finite flagged scope")
    root_bound = out / "root_actual_decisions48_bound.jsonl"
    save_rows(root_bound, roots)
    validation = {"schema": "kdm_actual_provider_and_root_binding_receipt_v1", "passed": True,
        "actual_Luna_full_QA_read": 500, "actual_Luna_explicit_decisions": 500,
        "actual_root_full_QA_read": 48, "actual_root_ordinary_sample_read": 19, "ordinary_sample_issues": 0,
        "source_bound_unique_QA": 500, "source_bound_unique_generation_keys": 1023,
        "actual_provider_sources": proofs, "root_source": pointer(paths["root-decisions"]),
        "root_receipt": pointer(paths["root-receipt"]), "prepared_v3_receipt": pointer(prep / "receipt.json"),
        "root_review_source_v2_and_v3_content_SHA_identical": True,
        "derived_boolean_from_explicit_label": True, "unknown_call_ids_invented": False,
        "writer_semantic_judgments": 0, "GT_judgments": 0, "GPU_initialized": False,
        "runner_sha256": file_hash(Path(__file__)), "root_bound_source": pointer(root_bound)}
    validation_path = out / "source_binding_validation_receipt.json"
    save_json(validation_path, validation)
    print(json.dumps({"event": "actual_source_binding_passed", "QA": 500, "root_QA": 48, "keys": 1023}), flush=True)
    persisted = []
    for b, owner, declared, attestation in batch_inputs:
        result = out / ("batch_" + str(b)) / "persisted"
        run_module("workflows.supplemental.remaining4.persist_explicit_viz_reviews",
            ["--queue", rel(owner), "--queue-sha256", file_hash(owner), "--decisions", rel(declared),
             "--decisions-sha256", file_hash(declared), "--attestation", rel(attestation),
             "--attestation-sha256", file_hash(attestation), "--first", "0", "--last", "249", "--output", rel(result)],
             out / ("persist_batch_" + str(b) + ".log"))
        require(load(result / "receipt.json")["actual_model_decisions"] == 250, "The mature writer lost actual provider QA")
        persisted.append(result)
    base_receipt = load(paths["base-authority"] / "receipt.json")
    require(base_receipt["passed"] and base_receipt["source_config_sha256"] == file_hash(paths["base-config"])
            and file_hash(paths["base-authority"] / "viz_exact_QA_authority.jsonl")
            == base_receipt["outputs"]["viz_exact_QA_authority.jsonl"], "The frozen v15 authority/config binding changed")
    config = deepcopy(load(paths["base-config"]))
    previous_targets = set()
    for entry in config["target_queues"]:
        source = within(ROOT, entry["path"])
        require(file_hash(source) == entry["sha256"], "An existing v15 target source changed")
        for _, row, _ in json_rows(source):
            require(row["qa_key"] not in previous_targets, "Existing target queue union repeats QA")
            previous_targets.add(row["qa_key"])
    require(len(previous_targets) == config["expected_unique_QA"], "The v15 finite target union changed")
    additions = [target for qa, target in originals.items() if qa not in previous_targets]
    new_targets = out / "additional_target_complete_QA.jsonl"
    save_rows(new_targets, additions)
    if additions:
        config["target_queues"].append({**pointer(new_targets), "qa_count": len(additions),
                                     "role": "current_actual_provider_complete_QA_only"})
    priority = max(entry["priority"] for entry in config["entries"]) + 10
    for result in persisted:
        config["entries"].append({**pointer(result / "decisions.jsonl"), "qa_count": 250,
                                 "priority": priority, "receipt": pointer(result / "receipt.json")})
    config["entries"].append({**pointer(root_bound), "qa_count": 48, "priority": priority + 10,
                             "receipt": pointer(validation_path)})
    config.update(expected_unique_QA=len(previous_targets) + len(additions), root_priority=priority + 10,
        parent_config_sha256=file_hash(paths["base-config"]),
        current_actual_provider_additions={"QA": 500, "new_to_target_union_QA": len(additions),
            "existing_target_QA": 500 - len(additions), "root_review_QA": 48,
            "source_validation": pointer(validation_path)})
    config_path = out / "source_config_current_actual500_root48.json"
    save_json(config_path, config)
    full_authority = out / "full_semantic_authority_v16"
    run_module("workflows.supplemental.remaining4.compose_semantic_authority",
               ["--sources", rel(config_path), "--output", rel(full_authority)], out / "compose_authority.log")
    accepted = [r for _, r, _ in json_rows(full_authority / "viz_exact_QA_authority.jsonl") if r["qa_key"] in originals]
    require(len(accepted) == 500 and all(r["annotation_complete"] and r["quality_span_resolved"]
            and r["behavior_resolved"] and r["source_annotation_provenance_valid"] for r in accepted),
            "Current actual provider/root QA still has unresolved behavior/span/provenance")
    finite = out / "accepted_actual500_authority_view"
    finite.mkdir(exist_ok=False)
    finite_path = finite / "viz_exact_QA_authority.jsonl"
    save_rows(finite_path, accepted)
    save_rows(finite / "pending_complete_QA.jsonl", [])
    view_receipt = {"schema": "kdm_actual_finite_semantic_authority_v1", "passed": True,
        "unique_QA": 500, "closed_QA": 500, "pending_QA": 0,
        "parent_full_authority": pointer(full_authority / "viz_exact_QA_authority.jsonl"),
        "parent_authority_receipt": pointer(full_authority / "receipt.json"),
        "source_binding_validation": pointer(validation_path),
        "view_is_semantic_judgment": False, "actual_root_full_QA_read": 48,
        "actual_root_ordinary_sample_read": 19, "ordinary_sample_issues": 0,
        "counts": dict(Counter(r["label"] for r in accepted)),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {p.name: file_hash(p) for p in finite.iterdir() if p.is_file()}}
    save_json(finite / "receipt.json", view_receipt)
    print(json.dumps({"event": "current_actual500_authority_accepted", "labels": view_receipt["counts"],
                      "new_target_union_QA": len(additions)}), flush=True)
    scored = out / "score_received292541_actual500_root48"
    run_module("workflows.supplemental.remaining4.apply_viz_reviews",
        ["--score-dir", rel(paths["score-dir"]), "--authority", rel(finite_path), "--output", rel(scored)],
        out / "apply_viz_reviews.log")
    owned_scored = out / "score_received292541_owned1023_actual500_root48"
    run_module("workflows.supplemental.remaining4.seal_owned_viz_reviews",
        ["--score-dir", rel(paths["score-dir"]), "--review-dir", rel(scored), "--prepare", rel(prep),
         "--source-validation", rel(validation_path), "--authority", rel(finite_path), "--output", rel(owned_scored)],
        out / "seal_owned_member_scope.log")
    audit = load(owned_scored / "application_audit.json")
    save_json(out / "application_audit.json", {**audit, "full_authority": pointer(full_authority / "viz_exact_QA_authority.jsonl"),
        "source_config": pointer(config_path), "orchestration_elapsed_seconds": time.monotonic() - started,
        "orchestration_runner_sha256": file_hash(Path(__file__))})
    save_json(out / "CURRENT_STATE.json", {"passed": True, "accepted_actual_QA": 500,
        "scores_applied_generation_keys": 1023, "root_full_QA_read": 48, "ordinary_sample_read": 19,
        "remaining_pending_QA_in_received_cohort": audit["pending_unique_QA_after"],
        "new_score_dir": rel(owned_scored), "new_full_authority_dir": rel(full_authority),
        "mature_QA_score_candidate_only": rel(scored),
        "bounded_actual_authority_view": rel(finite), "audit": pointer(out / "application_audit.json")})
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
