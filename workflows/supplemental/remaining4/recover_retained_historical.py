"""Recover only the explicitly accepted 51569 historical retained Food task keys."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kdm.decoding import DecodeConfig
from kdm.io import file_hash, stable_hash, stable_seed, within
from kdm.pipeline import task_id
from kdm.prompts import task_prompt
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.generate import condition_of
from workflows.supplemental.remaining11.score import infer_qa, load_authority, rows, score_target
from workflows.supplemental.remaining4.score_native import save_json, save_rows
from workflows.supplemental.remaining4.score_received import finite

BASE = "outputs/supplemental/remaining4"
OLD = "outputs/supplemental/remaining11/run_20260930_140337/scores/selected4_v10_root46"
CACHE = BASE + "/audit_20261001_1600/key_cache_v2_phi_sealed"
REF = BASE + "/reference_full_20261001_1600/checkpoints/v2_reviewed_role_reference"
CONE = BASE + "/native_cpu_20260930_2338/annotation/root_reviewed24_20261001/root_literal_cone1.jsonl"
EXPECTED = {"internvl35_8b": 2450, "onevision": 6823, "phi35": 16998, "qwen3vl": 25298}
TASK = ("method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")
CACHE_FIELDS = ("model", "dataset", "split", "sample_id", *TASK, "source_identity", "source_path")


def checked(path, sha):
    path = within(ROOT, str(path))
    actual = file_hash(path)
    if actual != sha:
        raise ValueError("Source bytes differ from their existing proof: " + str(path))
    return path


def raw_task(record, sample):
    return {"sample": sample, **{key: record[key] for key in TASK}}


def load_partial_sources(cache_receipt, legal, samples):
    audit_path = checked(cache_receipt["additional_source_receipt"], cache_receipt["additional_source_receipt_sha256"])
    audit = json.loads(audit_path.read_text())
    key_path = audit_path.parent / "missing_overlap_keys.jsonl"
    chosen = {r["key"]: r for _, r, _ in rows(key_path)}
    if len(chosen) != 444 or not set(chosen) <= set(legal):
        raise ValueError("The explicit sealed Phi delta differs from its 444 accepted keys")
    found, config_bank, proven = {}, {}, []
    for source in audit["sources"]:
        seal_path = checked(source["seal_path"], source["seal_sha256"])
        seal = json.loads(seal_path.read_text())
        partial = source["sealed_partial"]
        stop_path = checked(partial["source_stop_receipt_path"], partial["source_stop_receipt_sha256"])
        stop = json.loads(stop_path.read_text())
        if (seal["ownership_status"] != "held_pending_explicit_source_bound_continuation"
                or not stop["administratively_stopped"] or not stop["pidfd_exit_event_observed"]
                or seal["source_originals_modified"] or stop["scientific_parameters_changed"]
                or not partial["administratively_sealed_subset"] or not partial["generation_complete"]):
            raise ValueError("The selected source is not an immutable administratively sealed stopped prefix")
        raw_path = checked(partial["sealed_raw_path"], partial["raw_sha256"])
        identity_path = checked(partial["sealed_identity_path"], partial["identity_sha256"])
        complete_path = checked(partial["source_receipt_path"], partial["source_receipt_sha256"])
        identity = json.loads(identity_path.read_text())
        complete = json.loads(complete_path.read_text())
        if (any(partial.get(k) != v for k, v in complete.items()) or stable_hash(identity["definition"]) != identity["identity"]
                or identity["definition"]["base_config"] != asdict(DecodeConfig())):
            raise ValueError("Sealed completion, source identity or registered base config differs")
        selected_n, physical_n = 0, 0
        with raw_path.open("rb") as stream:
            for line_n, line in enumerate(stream, 1):
                if not line.endswith(b"\n"):
                    raise ValueError("A sealed raw source has an incomplete final line")
                physical_n += 1
                raw = json.loads(line)
                if raw["key"] not in chosen:
                    continue
                sample = samples.get(raw["sample"]["id"])
                key = raw["key"]
                if (key in found or raw["model"] != "phi35" or raw["identity"] != identity["identity"]
                        or sample is None or raw["sample"] != sample or raw["method"] in {"vcd", "m3id"}
                        or task_id(raw["model"], raw_task(raw, sample)) != key
                        or raw["seed"] != stable_seed(sample["id"], raw["model"], raw["replicate"])
                        or any(chosen[key].get(k) != raw.get(k) for k in TASK) or not finite(raw)):
                    raise ValueError("A selected stopped-source key, input, identity, seed or task differs")
                expected_config = {**identity["definition"]["base_config"], "method": raw["method"]}
                if raw["method"] == "instruction_m3id":
                    expected_config["m3id_offset"] = len(raw["offset_prompt_tokens"])
                if expected_config != raw["config"]:
                    raise ValueError("The actual sealed row config differs from its registered algorithm")
                provenance = {"source_path": str(raw_path.relative_to(ROOT)), "source_line": line_n,
                    "raw_line_sha256": hashlib.sha256(line).hexdigest(), "source_identity": raw["identity"],
                    "source_claim": source["claim_id"], "source_part": partial["part"],
                    "raw_source_sha256": partial["raw_sha256"], "identity_source_path": str(identity_path.relative_to(ROOT)),
                    "identity_source_sha256": partial["identity_sha256"]}
                found[key] = (raw, provenance)
                config_bank[raw["model"], stable_hash(raw["config"])] = (raw["config"], provenance)
                selected_n += 1
        if physical_n != partial["rows"] or selected_n != source["missing_overlap_rows"]:
            raise ValueError("The sealed prefix/selected key counts differ")
        proven.append({"claim": source["claim_id"], "physical_rows": physical_n, "selected_rows": selected_n,
            "raw": str(raw_path.relative_to(ROOT)), "raw_sha256": partial["raw_sha256"],
            "identity": str(identity_path.relative_to(ROOT)), "identity_sha256": partial["identity_sha256"],
            "complete_sha256": partial["source_receipt_sha256"], "seal_sha256": source["seal_sha256"],
            "stop_sha256": partial["source_stop_receipt_sha256"], "original_claim_complete": False})
    if set(found) != set(chosen):
        raise ValueError("The two stopped prefixes do not cover the exact Phi delta")
    return found, config_bank, proven, key_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-proof", action="append", default=[])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = within(ROOT, args.output)
    output.relative_to(ROOT / BASE)
    if args.verify_only:
        verify(output)
        return
    output.mkdir(parents=True, exist_ok=False)
    cache_receipt_path = ROOT / CACHE / "receipt.json"
    cache_receipt = json.loads(cache_receipt_path.read_text())
    if not cache_receipt["passed"] or cache_receipt["added_legal_subset_keys"] != 444:
        raise ValueError("The accepted historical source cache is not the audited 51569-key scope")
    key_path = checked(ROOT / CACHE / "accepted_formal_keys.jsonl", cache_receipt["key_file_sha256"])
    legal = {}
    for _, record, _ in rows(key_path):
        if record["dataset"] != "food101" or record["method"] in {"vcd", "m3id"}:
            continue
        if record["key"] in legal:
            raise ValueError("Repeated accepted historical retained task")
        legal[record["key"]] = record
    if dict(Counter(r["model"] for r in legal.values())) != EXPECTED:
        raise ValueError("The explicit historical retained population differs")
    samples = {r["id"]: r for _, r, _ in rows(ROOT / "data/current/all.jsonl") if r["dataset"] == "food101"}
    if len(samples) != 4848:
        raise ValueError("Original Food manifest is incomplete")
    classes = sorted({s["class"] for s in samples.values()})
    if len(classes) != 101:
        raise ValueError("The original 101-class vocabulary differs")
    patterns = frozen.compile_classes(classes)
    partials, config_bank, partial_proofs, partial_keys = load_partial_sources(cache_receipt, legal, samples)
    cone_path = ROOT / CONE
    cone = next(rows(cone_path))[1]
    cone_sha = file_hash(cone_path)
    if (cone_sha != "78cb0bb7a8545db04490a1118d0750bc6bc3b15217f126dfa70ccf3b84636c8f"
            or not cone["root_reviewed"] or cone["annotation_model"] != "gpt-6.1-sol"
            or cone["annotation_effort"] != "max" or cone["answer"] != "Ice cream cone"
            or cone["primary_span"] not in cone["answer"] or cone["abstain"] is not False):
        raise ValueError("The actual root literal-only authority differs")
    tokenizer_proofs, proof_inputs = {}, []
    for value in args.tokenizer_proof:
        path = within(ROOT, value)
        proof = json.loads(path.read_text())
        if (proof["schema"] != "kdm_historical_plain_tokenizer_config_proof_v1"
                or proof["GPU_initialized"] or proof["model_weights_loaded"]
                or proof["offset"] != len(proof["tokens"]) or proof["model"] not in EXPECTED
                or proof["registered_prompt_source_sha256"] != file_hash(ROOT / "src/kdm/prompts.py")
                or proof["plain_prompt"] != task_prompt(proof["question"], guided=False)):
            raise ValueError("Plain-question tokenizer proof is outside the registered reconstruction")
        tokenizer_proofs[proof["model"], proof["question"]] = proof
        proof_inputs.append({"path": str(path.relative_to(ROOT)), "sha256": file_hash(path),
            "model": proof["model"], "offset": proof["offset"], "source_identity": proof["source_identity"]})
    old_path = checked(ROOT / OLD / "score_rows.jsonl.gz", cache_receipt["score_source_sha256"])
    old_receipt = json.loads((ROOT / OLD / "execution_receipt.json").read_text())
    if (old_receipt["score_rows_sha256"] != cache_receipt["score_source_sha256"]
            or old_receipt["canonical_pending_rows"] or old_receipt["abstain_pending_rows"]):
        raise ValueError("The historical resolved score source differs from its receipt")
    old_summary_path = checked(ROOT / OLD / "summary.json", old_receipt["summary_sha256"])
    old_summary = json.loads(old_summary_path.read_text())
    records, holds, old_keys, literal_delta = [], [], set(), []
    behavior_reuse = defaultdict(set)
    counts = Counter()
    algorithm_proof = {"kind": "registered_DecodeConfig_and_run_tasks_reconstruction",
        "decoding_source_sha256": file_hash(ROOT / "src/kdm/decoding.py"),
        "pipeline_source_sha256": file_hash(ROOT / "src/kdm/pipeline.py")}
    historical_path = output / "historical_score_rows.jsonl.gz"
    with gzip.open(historical_path, "xt", encoding="utf-8", compresslevel=1) as history:
        for line_n, original, line_sha in rows(old_path):
            key = original["key"]
            if key not in legal:
                continue
            expected = legal[key]
            sample = samples.get(original["sample_id"])
            if (key in old_keys or any(original.get(k) != expected.get(k) for k in CACHE_FIELDS)
                    or sample is None or original["target_class"] != sample["class"]
                    or original["question"] != sample["question"] or original["qa_key"] != frozen.qah(original["question"], original["answer"])
                    or task_id(original["model"], raw_task(original, sample)) != key
                    or original["seed"] != stable_seed(sample["id"], original["model"], original["replicate"])
                    or original["canonical_name_in_primary_score"] not in (0, 1) or type(original["abstain"]) is not bool):
                raise ValueError("A historical score/key/input/seed or resolved label differs")
            old_keys.add(key)
            history.write(json.dumps(original, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            record = dict(original)
            record.update(historical_score_source_path=str(old_path.relative_to(ROOT)), historical_score_source_line=line_n,
                historical_score_line_sha256=line_sha, historical_score_source_sha256=cache_receipt["score_source_sha256"],
                historical_record_copy_path=str(historical_path.relative_to(ROOT)), historical_canonical_and_abstain_reused=True)
            behavior_reuse[original["qa_key"]].add(original["abstain"])
            if original["literal_extracted_name_score"] is None:
                if (original["qa_key"] != cone["qa_key"] or original["question"] != cone["question"] or original["answer"] != cone["answer"]):
                    raise ValueError("A historical literal gap is outside the four actual cone memberships")
                inferred = infer_qa(original["question"], original["answer"], patterns, {}, {}, {cone["qa_key"]: cone})
                canonical, literal, _ = score_target(original["answer"], original["target_class"], inferred, patterns)
                if canonical != original["canonical_name_in_primary_score"] or literal != 0 or inferred["abstain"] != original["abstain"]:
                    raise ValueError("The literal-only authority changes an old main label")
                record.update(literal_extracted_name_score=literal, literal_revision_source_path=str(cone_path.relative_to(ROOT)),
                    literal_revision_source_sha256=cone_sha, literal_revision_original_value=None)
                literal_delta.append({"key": key, "model": original["model"], "sample_id": original["sample_id"],
                    "qa_key": original["qa_key"], "previous_literal": None, "actual_literal": literal,
                    "canonical_and_abstain_unchanged": True, "actual_root_source": str(cone_path.relative_to(ROOT)),
                    "actual_root_sha256": cone_sha, "old_score_line_sha256": line_sha})
            config, config_source = None, None
            bank = config_bank.get((original["model"], original["config_sha256"]))
            if bank:
                config, config_source = bank[0], {"kind": "actually_sealed_same_model_config", **bank[1]}
            else:
                config = asdict(DecodeConfig(method=original["method"]))
                config_source = dict(algorithm_proof)
                if original["method"] == "instruction_m3id":
                    proof = tokenizer_proofs.get((original["model"], original["question"]))
                    if proof:
                        config["m3id_offset"] = proof["offset"]
                        config_source["actual_tokenizer_proof"] = proof
                    else:
                        config = None
            if config is None or stable_hash(config) != original["config_sha256"]:
                record.update(config=None, config_acceptance_status="held_pending_exact_source_config",
                    config_reconstruction_candidate=config, config_reconstruction_source=config_source)
                holds.append({"key": key, "model": original["model"], "sample_id": original["sample_id"],
                    "method": original["method"], "expected_config_sha256": original["config_sha256"],
                    "candidate_config_sha256": stable_hash(config) if config is not None else None})
            else:
                record.update(config=config, config_acceptance_status="matched_original_config_sha256",
                    config_reconstruction_source=config_source)
            if any(record.get(k) != v for k, v in original.items() if k != "literal_extracted_name_score"):
                raise ValueError("Historical recovery modified an original field outside the explicit literal delta")
            records.append(record)
    if len(old_keys) != 51125 or set(legal) - old_keys != set(partials) or len(literal_delta) != 4:
        raise ValueError("Historical/partial exact coverage or literal revision membership differs")
    assets = json.loads((ROOT / "outputs/supplemental/remaining11/run_20260930_140337/assets/asset_manifest.json").read_text())
    decision_paths = [within(ROOT, x) for x in old_summary["decision_files"]] + [cone_path]
    reviews, behavior, provenance, _, decisions = load_authority(ROOT / OLD,
        {"census_final_labels": assets["census_final_labels"], "historical_labels": []}, decision_paths)
    for qa_key, values in behavior_reuse.items():
        behavior[qa_key].update(values)
    boundary = {}
    for key in sorted(partials):
        raw, source = partials[key]
        sample = raw["sample"]
        inferred = infer_qa(sample["question"], raw["text"], patterns, reviews, behavior, decisions)
        if frozen.configured_marker(raw["text"], raw) and not inferred["variants"]:
            inferred.update(abstain=True, behavior_source="exact_actual_condition_marker")
        canonical, literal, reason = score_target(raw["text"], sample["class"], inferred, patterns)
        record = {**{name: raw[name] for name in TASK}, "model": raw["model"], "dataset": "food101", "split": "eval",
            "main_marker": raw["marker"], "stage": "formal", "key": key, "sample_id": sample["id"],
            "target_class": sample["class"], "question": sample["question"], "answer": raw["text"],
            "qa_key": inferred["qa_key"], "config": raw["config"], "config_sha256": stable_hash(raw["config"]),
            "config_acceptance_status": "actual_sealed_row_verified_against_identity_base_config",
            "seed": raw["seed"], "terminated": raw.get("terminated"), "prompt": raw["prompt"],
            "reference_prompt": raw.get("reference_prompt"), "neutral_prompt": raw.get("neutral_prompt"),
            "canonical_name_in_primary_score": canonical, "literal_extracted_name_score": literal,
            "abstain": inferred["abstain"], "score_reason": reason, "primary_extraction": inferred["parsed"],
            "behavior_source": inferred["behavior_source"], "decision_source": inferred["decision"],
            "behavior_history_sources": provenance.get(inferred["qa_key"], []),
            "name_history_source": inferred["review"], "new_scientific_annotation": False, **source}
        if canonical is None or literal is None or inferred["abstain"] is None:
            packet = boundary.setdefault(inferred["qa_key"], {"qa_key": inferred["qa_key"],
                "question": sample["question"], "answer": raw["text"], "primary_extraction": inferred["parsed"],
                "score_reason": reason, "behavior_source": inferred["behavior_source"], "source_memberships": []})
            packet["source_memberships"].append({name: record[name] for name in ("key", "model", "sample_id", "target_class",
                "source_path", "source_line", "source_identity", "raw_line_sha256", "canonical_name_in_primary_score",
                "literal_extracted_name_score", "abstain")})
        records.append(record)
    reference_receipt_path = ROOT / REF / "receipt.json"
    reference_receipt = json.loads(reference_receipt_path.read_text())
    reference_path = checked(ROOT / REF / "reference_G.jsonl", reference_receipt["outputs"]["reference_G.jsonl"])
    references = {(r["model"], r["sample_id"]): r for _, r, _ in rows(reference_path)}
    if not reference_receipt["passed"] or len(references) != 19392 or reference_receipt["reference_pending"]:
        raise ValueError("The accepted complete Food reference is missing")
    groups = defaultdict(list)
    domains = {s["id"] for s in samples.values() if s["split"] == "eval"}
    for record in records:
        reference = references[record["model"], record["sample_id"]]
        if (not reference["reference_complete"] or type(reference["reference_G"]) is not bool
                or reference["split"] != record["split"] or reference["target_class"] != record["target_class"]):
            raise ValueError("Score/reference model/sample/split/class differs")
        record.update(reference_G=reference["reference_G"], reference_complete=True,
            gold_rank=reference["gold_rank"], correct_count=reference["correct_count"],
            reference_join_source_path=str(reference_path.relative_to(ROOT)),
            reference_join_source_sha256=reference_receipt["outputs"]["reference_G.jsonl"])
        context = condition_of(record["model"], "food101", "formal", {"sample": {"split": "eval"}, **record})
        group_id = stable_hash({"condition": context, "decode_config_sha256": record["config_sha256"]})
        groups[group_id].append((record, context))
        counts["rows"] += 1
        counts["canonical_pending"] += record["canonical_name_in_primary_score"] is None
        counts["literal_pending"] += record["literal_extracted_name_score"] is None
        counts["abstain_pending"] += record["abstain"] is None
    coverage = []
    for cid, members in sorted(groups.items()):
        selected = [pair[0] for pair in members]
        ids = {r["sample_id"] for r in selected}
        if len(ids) != len(selected):
            raise ValueError("A historical registered condition repeats a sample key")
        quota = Counter(r["target_class"] for r in selected)
        full = ids == domains
        if full and (len(quota) != 101 or set(quota.values()) != {24}):
            raise ValueError("A full condition does not have its original 101x24 quota")
        pending = sum(r["canonical_name_in_primary_score"] is None or r["literal_extracted_name_score"] is None
            or r["abstain"] is None or r["config"] is None for r in selected)
        coverage.append({"condition_id": cid, **members[0][1], "decode_config_sha256": selected[0]["config_sha256"],
            "decode_config": selected[0]["config"], "received_rows": len(selected), "expected_full_inputs": 2424,
            "full_input_coverage": full, "received_classes": len(quota), "class_quota_verified": full,
            "canonical_literal_abstain_or_config_pending": pending, "reference_pending": 0,
            "accepted_complete_condition": full and pending == 0, "partial_cohort_used_as_main_comparison": False})
    if len(records) != 51569 or {r["key"] for r in records} != set(legal):
        raise ValueError("The final recovered historical cohort differs from its exact accepted keys")
    save_rows(output / "score_rows.jsonl.gz", records)
    save_rows(output / "partial444_score_rows.jsonl.gz", [r for r in records if r["key"] in partials])
    save_rows(output / "boundary_queue.jsonl", [boundary[k] for k in sorted(boundary)])
    save_rows(output / "config_holds.jsonl", holds)
    save_rows(output / "literal_delta.jsonl", literal_delta)
    save_rows(output / "condition_coverage.jsonl", coverage)
    per_model = {}
    for model in EXPECTED:
        selected = [r for r in records if r["model"] == model]
        per_model[model] = {"rows": len(selected), "canonical_pending": sum(r["canonical_name_in_primary_score"] is None for r in selected),
            "literal_pending": sum(r["literal_extracted_name_score"] is None for r in selected),
            "abstain_pending": sum(r["abstain"] is None for r in selected), "config_hold": sum(r["config"] is None for r in selected),
            "reference_complete": len(selected), "complete_conditions": sum(c["model"] == model and c["accepted_complete_condition"] for c in coverage)}
    receipt = {"schema": "kdm_historical_retained_food_recovery_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "rows": len(records), "unique_keys": len(legal),
        "historical_score_rows_reused": 51125, "historical_canonical_and_abstain_changed": 0,
        "historical_literal_only_updated": len(literal_delta), "new_sealed_partial_rows_scored": len(partials),
        "counts": dict(counts), "per_model": per_model, "boundary_unique_QA": len(boundary), "config_holds": len(holds),
        "reference_complete": len(records), "reference_pending": 0,
        "complete_registered_conditions": sum(c["accepted_complete_condition"] for c in coverage),
        "historical_score_source_path": str(old_path.relative_to(ROOT)), "historical_score_source_sha256": cache_receipt["score_source_sha256"],
        "accepted_key_source_path": str(key_path.relative_to(ROOT)), "accepted_key_source_sha256": cache_receipt["key_file_sha256"],
        "accepted_key_receipt_sha256": file_hash(cache_receipt_path), "selected_partial_keys_path": str(partial_keys.relative_to(ROOT)),
        "selected_partial_keys_sha256": file_hash(partial_keys), "stopped_source_proofs": partial_proofs,
        "reference_source_path": str(reference_path.relative_to(ROOT)), "reference_source_sha256": reference_receipt["outputs"]["reference_G.jsonl"],
        "reference_receipt_sha256": file_hash(reference_receipt_path), "literal_authority_path": str(cone_path.relative_to(ROOT)),
        "literal_authority_sha256": cone_sha, "tokenizer_config_proofs": proof_inputs,
        "canonical_scorer_sha256": file_hash(Path(frozen.__file__)), "inference_source_sha256": file_hash(ROOT / "workflows/supplemental/remaining11/score.py"),
        "source_sha256": file_hash(Path(__file__)), "old_raw_reopened": False, "active_raw_read": False,
        "limited_admin_sealed_raw_sources_read": len(partial_proofs), "new_scientific_annotation": 0,
        "parameters_changed": False, "GPU_initialized": False, "new_generation": 0, "new_API_calls": 0,
        "label_completion_claimed": not boundary and not holds,
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / "receipt.json", receipt)
    verify(output)
    print(json.dumps({k: receipt[k] for k in ("passed", "rows", "historical_score_rows_reused", "new_sealed_partial_rows_scored",
        "counts", "boundary_unique_QA", "config_holds", "reference_complete", "complete_registered_conditions", "per_model")}))


def verify(output):
    receipt = json.loads((output / "receipt.json").read_text())
    for name, sha in receipt["outputs"].items():
        checked(output / name, sha)
    records = {r["key"]: r for _, r, _ in rows(output / "score_rows.jsonl.gz")}
    if len(records) != 51569:
        raise ValueError("Recovered key coverage differs at actual output verification")
    literal_keys = {r["key"] for _, r, _ in rows(output / "literal_delta.jsonl")}
    preserved = 0
    for _, historical, _ in rows(output / "historical_score_rows.jsonl.gz"):
        result = records[historical["key"]]
        for field, value in historical.items():
            if field == "literal_extracted_name_score" and historical["key"] in literal_keys:
                if value is not None or result[field] != 0:
                    raise ValueError("Actual literal revision differs from its root-defined delta")
            elif result.get(field) != value:
                raise ValueError("Historical output lost or modified an original score field")
        preserved += 1
    if preserved != 51125 or len(literal_keys) != 4:
        raise ValueError("Historical preservation coverage differs")
    config_verified = sum(r.get("config") is not None and stable_hash(r["config"]) == r["config_sha256"] for r in records.values())
    if config_verified + receipt["config_holds"] != len(records):
        raise ValueError("Actual reconstructed config does not match its original hash")
    verification = {"schema": "kdm_historical_retained_food_output_verification_v1", "passed": True,
        "rows": len(records), "historical_objects_preserved": preserved, "canonical_and_abstain_original_field_difference": 0,
        "literal_actual_delta_objects": len(literal_keys), "config_sha256_matched": config_verified,
        "config_held": receipt["config_holds"], "all_reference_complete": all(r["reference_complete"] for r in records.values()),
        "boundary_unique_QA": receipt["boundary_unique_QA"], "complete_conditions": receipt["complete_registered_conditions"],
        "partial_denominator_promoted": False, "outputs_source_sha256": receipt["outputs"]["score_rows.jsonl.gz"],
        "new_GPU_experiments": 0, "new_semantic_annotation": 0}
    verification_path = output / "verification.json"
    if verification_path.exists():
        if json.loads(verification_path.read_text()) != verification:
            raise ValueError("A repeated finite verification differs from its saved actual result")
    else:
        save_json(verification_path, verification)


if __name__ == "__main__":
    main()
