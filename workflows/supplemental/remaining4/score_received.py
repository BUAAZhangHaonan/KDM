#!/usr/bin/env python3
"""Score explicit immutable received parts with the registered scoring functions."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import file_hash, within
from kdm.models.official_vqa_normalizer import VQAEval
from kdm.scoring import lexical_label, vqa_score
from workflows.main_results import score as frozen
from workflows.supplemental.remaining11.score import infer_qa, load_authority, rows, score_target
from workflows.supplemental.remaining4.native import MODELS
from workflows.supplemental.remaining4.score_native import save_json, save_rows

BASE = "outputs/supplemental/remaining4"
FIELDS = ("method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")


def load_viz_authority(paths):
    authority = {}
    for path in paths:
        source_sha = file_hash(path)
        receipt_path = path.parent / "receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("schema") in {
                "kdm_finite_viz_exact_QA_authority_v1",
                "kdm_actual_finite_semantic_authority_v1",
            }:
                if not receipt.get("passed") or receipt["outputs"].get(path.name) != source_sha:
                    raise ValueError("The finite Viz authority differs from its passed source receipt")
                if receipt["schema"] == "kdm_actual_finite_semantic_authority_v1" and (
                    receipt["pending_QA"] != 0
                    or receipt["closed_QA"] != receipt["unique_QA"]
                ):
                    raise ValueError("The actual annotation authority is not fully closed")
        for line, row, line_sha in rows(path):
            key = frozen.qah(row["question"], row["answer"])
            if row["qa_key"] != key:
                raise ValueError("A Viz authority QA differs from its complete original content")
            # The authority adapter retains unresolved QAs in this same file.
            # They remain in the finite queue; no judgment is inferred from them.
            if row.get("annotation_complete") is False:
                continue
            if type(row.get("abstain")) is not bool:
                raise ValueError("A Viz authority must contain actual complete QA and a resolved behavior")
            text = row.get("answer_text")
            if text and (not isinstance(text, str) or text not in row["answer"]):
                raise ValueError("A reused Viz answer must be an exact continuous original answer span")
            previous = authority.get(key)
            if previous and (previous["abstain"] != row["abstain"]
                    or previous.get("answer_text") != text or previous.get("label") != row.get("label")):
                raise ValueError("Conflicting exact-QA Viz authorities need explicit adjudication")
            authority[key] = {**row, "decision_source_path": str(path.relative_to(ROOT)),
                "decision_source_sha256": source_sha, "decision_source_line": line,
                "decision_line_sha256": line_sha}
    return authority


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return False
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decision-file", action="append", default=[])
    parser.add_argument("--viz-authority-file", action="append", default=[])
    args = parser.parse_args()
    manifest_path, output = (within(ROOT, value) for value in (args.manifest, args.output))
    output.relative_to(ROOT / BASE)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (manifest["schema"] != "kdm_selected4_registered_received_manifest_v1"
            or not manifest["local_source_sha_verified"] or manifest["active_raw_opened"] != 0
            or not manifest["original_sources_preserved"]):
        raise ValueError("The source must be an actually received immutable-part manifest")
    expected = int(manifest["rows"])
    if expected != sum(int(part["rows"]) for part in manifest["parts"]):
        raise ValueError("Manifest rows differ from the explicit completed parts")
    checked = {}
    for part in manifest["parts"]:
        if part["model"] not in MODELS or not part["source_part_generation_complete"]:
            raise ValueError("Part model or generation status lies outside selected-four scope")
        for entry in part["files"]:
            path = within(ROOT, entry["received_path"])
            if path not in checked:
                checked[path] = file_hash(path)
            if checked[path] != entry["sha256"]:
                raise ValueError("Received immutable source member differs: " + str(path))
    asset_path = ROOT / "outputs/supplemental/remaining11/run_20260930_140337/assets/asset_manifest.json"
    assets = json.loads(asset_path.read_text())
    authority = ROOT / "outputs/supplemental/remaining11/run_20260930_140337/scores/selected4_v10_root46"
    summary_path = authority / "summary.json"
    summary = json.loads(summary_path.read_text())
    decision_paths = [within(ROOT, p) for p in summary.get("decision_files", [])]
    decision_paths += [within(ROOT, p) for p in args.decision_file]
    reviews, behavior, _, _, decisions = load_authority(
        authority, {"census_final_labels": assets["census_final_labels"], "historical_labels": []}, decision_paths)
    viz_paths = [within(ROOT, p) for p in args.viz_authority_file]
    viz_authority = load_viz_authority(viz_paths)
    samples = {record["id"]: record for _, record, _ in rows(ROOT / "data/current/all.jsonl")}
    patterns = frozen.compile_classes(sorted({s["class"] for s in samples.values() if s["dataset"] == "food101"}))
    normalizer = VQAEval(None, None)
    normalize = lambda text: normalizer.processDigitArticle(normalizer.processPunctuation(text))
    scored, boundary, seen, counts = [], {}, set(), Counter()
    for part in manifest["parts"]:
        raw_path = within(ROOT, part["received_raw_path"])
        identity_path = within(ROOT, part["received_identity_path"])
        identity = json.loads(identity_path.read_text())["identity"]
        part_rows = 0
        for line, row, line_sha in rows(raw_path):
            sample = row["sample"]
            current = samples.get(sample["id"])
            if (row["key"] in seen or row["identity"] != identity or row["model"] != part["model"]
                    or current is None or any(sample.get(f) != current.get(f)
                        for f in ("id", "dataset", "split", "question", "image_path", "class"))
                    or sample["dataset"] != part["dataset"] or not finite(row)):
                raise ValueError("Actual part key, input, source identity or finite values differ")
            if row["method"] in {"vcd", "m3id"} and (row["guided"] or row["reference_guided"]):
                raise ValueError("New plain guided VCD/M3ID generation is outside the current scope")
            seen.add(row["key"])
            part_rows += 1
            inferred = infer_qa(sample["question"], row["text"], patterns, reviews, behavior, decisions)
            decision = inferred["decision"]
            if sample["dataset"] == "vizwiz":
                if inferred["qa_key"] in viz_authority:
                    decision = {**viz_authority[inferred["qa_key"]],
                        "previous_inferred_decision_source": decision}
                    inferred = {**inferred, "decision": decision, "abstain": decision["abstain"],
                        "behavior_source": "actual_final_Viz_exact_QA_authority"}
                elif decision is None:
                    registered_label = lexical_label(row["text"])
                    if registered_label is not None:
                        decision = {"qa_key": inferred["qa_key"], "question": sample["question"],
                            "answer": row["text"], "label": registered_label,
                            "abstain": registered_label == "abstain", "answer_text": "",
                            "evidence_span": row["text"], "annotation_source": "existing_registered_exact_phrase_rule",
                            "rule_path": "src/kdm/scoring.py",
                            "rule_sha256": file_hash(ROOT / "src/kdm/scoring.py")}
                        inferred = {**inferred, "decision": decision,
                            "abstain": decision["abstain"], "behavior_source": decision["annotation_source"]}
                    else:
                        inferred = {**inferred, "abstain": None,
                            "behavior_source": "Viz_requires_actual_exact_QA_annotation"}
                if decision and (decision.get("root_review_required") or decision.get("needs_root")):
                    inferred = {**inferred, "abstain": None,
                        "behavior_source": "actual_Viz_annotation_pending_root_adjudication"}
            record = {"model": row["model"], "dataset": sample["dataset"], "split": sample["split"],
                **{field: row[field] for field in FIELDS}, "main_marker": row["marker"], "stage": part["stage"],
                "key": row["key"], "sample_id": sample["id"], "question": sample["question"], "answer": row["text"],
                "source_path": str(raw_path.relative_to(ROOT)), "source_line": line, "raw_line_sha256": line_sha,
                "source_identity": row["identity"], "source_claim": part["claim_id"], "source_part": part["part"],
                "seed": row.get("seed"), "terminated": row.get("terminated"), "config": row.get("config"),
                "qa_key": inferred["qa_key"], "abstain": inferred["abstain"],
                "behavior_source": inferred["behavior_source"], "decision_source": decision,
                "reference_G": None, "reference_complete": False}
            pending = inferred["abstain"] is None or bool(decision and
                (decision.get("root_review_required") or decision.get("needs_root")))
            if sample["dataset"] == "food101":
                canonical, literal, reason = score_target(row["text"], sample["class"], inferred, patterns)
                record.update(target_class=sample["class"], canonical_name_in_primary_score=canonical,
                    literal_extracted_name_score=literal, score_reason=reason, primary_extraction=inferred["parsed"])
                pending = pending or canonical is None or literal is None
                counts["food_rows"] += 1
                counts["food_canonical_pending"] += canonical is None
                counts["food_literal_pending"] += literal is None
            elif sample["dataset"] == "vizwiz":
                refs = sample["official_answers"]
                if len(refs) != 10 or sample.get("annotated_answerable") not in (0, 1, False, True):
                    raise ValueError("The frozen VizWiz official ten answers or answerability are incomplete")
                raw_credit = float(vqa_score(row["text"], refs, normalize))
                answer_text = decision.get("answer_text") if decision else None
                label = decision.get("label") if decision else None
                if decision and decision.get("answer_text_span") and not answer_text:
                    answer_text = decision["answer_text_span"]
                if not pending and (inferred["abstain"] is True or label == "invalid"):
                    quality = 0.0
                elif not pending and answer_text and answer_text in row["text"]:
                    quality = float(vqa_score(answer_text, refs, normalize))
                else:
                    quality = None
                    pending = True
                record.update(official_answers=refs, annotated_answerable=sample["annotated_answerable"],
                    official_consensus_raw_credit=raw_credit, answer_quality_credit=quality,
                    semantic_answer_text=answer_text, semantic_label=label,
                    answer_quality_source=decision if decision else None,
                    answer_quality_pending=quality is None,
                    vqa_score_scope="original official leave-one-annotator-out continuous consensus")
                counts["viz_rows"] += 1
                counts["viz_quality_pending"] += quality is None
            else:
                raise ValueError("The received extraction contains an unregistered dataset")
            counts["abstain_pending"] += inferred["abstain"] is None
            counts["label_pending_members"] += pending
            if pending:
                member = {k: record[k] for k in ("model", "key", "sample_id", "source_path", "source_line", "raw_line_sha256", "source_identity")}
                qa = boundary.setdefault(record["qa_key"], {"qa_key": record["qa_key"], "question": record["question"],
                    "answer": record["answer"], "source_memberships": [],
                    "needs_behavior": record["abstain"] is None,
                    "needs_answer_span": sample["dataset"] == "vizwiz" and record["answer_quality_pending"],
                    "needs_root_acceptance": bool(decision and
                        (decision.get("root_review_required") or decision.get("needs_root")))})
                qa["source_memberships"].append(member)
            scored.append(record)
        if part_rows != int(part["rows"]):
            raise ValueError("A received completed part has a different actual row count")
    if len(scored) != expected:
        raise ValueError("The actual scored count differs from the manifest")
    output.mkdir(parents=True, exist_ok=False)
    save_rows(output / "score_rows.jsonl.gz", scored)
    save_rows(output / "boundary_queue.jsonl", list(boundary.values()))
    receipt = {"schema": "kdm_selected4_received_part_cpu_score_v1", "passed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "rows": expected, "unique_keys": len(seen),
        "parts": len(manifest["parts"]), "counts": dict(counts), "boundary_unique_QA": len(boundary),
        "source_manifest": str(manifest_path.relative_to(ROOT)), "source_manifest_sha256": file_hash(manifest_path),
        "source_member_files_verified": len(checked), "authority_summary_sha256": file_hash(summary_path),
        "decision_files": [{"path": str(p.relative_to(ROOT)), "sha256": file_hash(p)} for p in decision_paths],
        "viz_authority_files": [{"path": str(p.relative_to(ROOT)), "sha256": file_hash(p)} for p in viz_paths],
        "reused_closed_viz_authority_unique_QA": len(viz_authority),
        "Viz_main_quality_requires_actual_annotated_answer_span": True,
        "full_raw_official_credit_used_as_annotated_short_answer": False,
        "unannotated_Viz_behavior_defaulted_to_answer_or_abstention": False,
        "runner_sha256": file_hash(Path(__file__)), "GPU_initialized": False, "new_generation": 0, "new_API_calls": 0,
        "reference_join_complete": False, "label_completion_claimed": not counts["label_pending_members"],
        "official_unanswerable_consensus_preserved": True,
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("passed", "rows", "parts", "counts", "boundary_unique_QA")}))


if __name__ == "__main__":
    main()
