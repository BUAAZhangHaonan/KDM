"""Source-bound Food-101 scoring and references for the supplemental models."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from kdm.io import file_hash, stable_hash, atomic_json
from workflows.main_results import score as shared
from workflows.supplemental.remaining11.generate import MODELS

COND = ("model", "dataset", "split", "method", "kind", "marker",
        "reference_marker", "guided", "reference_guided", "replicate")
UNCERTAIN = re.compile(r"\b(?:unknown|unclear|unsure|uncertain|unable|cannot|can't|"
                       r"can not|don't know|do not know|not sure|not certain|"
                       r"not able|hard to|difficult to|insufficient|unidentifiable|"
                       r"unrecognizable|not possible|no food)\b", re.I)
REFUSAL = re.compile(r"\b(?:cannot|can't|can not|unable to|not able to)\s+"
                     r"(?:reliably\s+|accurately\s+|clearly\s+)?"
                     r"(?:identify|determine|tell|recognize|discern|answer)\b", re.I)


def rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for number, raw in enumerate(stream, 1):
            if raw.strip():
                yield number, json.loads(raw), hashlib.sha256(raw).hexdigest()


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def source_record(entry, line, row, line_sha):
    return {"source_path": entry["path"], "source_line": line,
            "source_identity": row.get("identity"), "raw_line_sha256": line_sha,
            "source_container": entry.get("source_container"),
            "source_member": entry.get("source_member"),
            "source_cohort": entry.get("cohort", "registered_native"),
            "original_key": row.get("original_key", row["key"]),
            "original_source_identity": row.get("original_identity", row.get("identity"))}


def selected_sources(run, manifest, models=MODELS, received_manifests=()):
    selected = [dict(entry) for entry in manifest["sources"]
                if entry["model"] in models and
                entry["status"] in {"selected_complete", "selected_partial"}]
    for entry in selected:
        entry["keys"] = {row["key"] for _, row, _ in rows(entry["usable_keys_file"])}
        if len(entry["keys"]) != entry["selected_rows"]:
            raise ValueError("Selected source key count differs: " + entry["path"])
    default_extra = run / "remote_completed/received_manifest.json"
    extras = ([default_extra] if default_extra.exists() else []) + list(received_manifests)
    for extra in extras:
        received = json.loads(extra.read_text())
        for item in received["parts"]:
            if item["model"] not in models:
                continue
            entry = dict(item)
            entry["path"] = str(ROOT / item["received_raw_path"])
            entry["keys"] = None
            entry["receipt_path"] = str(ROOT / item["received_receipt_path"])
            receipt = json.loads(Path(entry["receipt_path"]).read_text())
            if receipt["raw_sha256"] != file_hash(entry["path"]):
                raise ValueError("Received completed part bytes differ from receipt")
            selected.append(entry)
    for path in sorted((run / "records").glob("*/*/*/*/*.complete.json")):
        receipt = json.loads(path.read_text())
        if receipt["model"] not in models:
            continue
        raw = ROOT / receipt["raw_path"]
        if receipt["generation_complete"] is not True or file_hash(raw) != receipt["raw_sha256"]:
            raise ValueError("Local completed part differs from its verified receipt")
        selected.append({"path": str(raw), "model": receipt["model"],
                         "stage": receipt["stage"], "keys": None,
                         "receipt_path": str(path), "cohort": "registered_native"})
    for path in sorted((run / "sealed_failed").glob("*/*/*.complete.json")):
        receipt = json.loads(path.read_text())
        if receipt["model"] not in models:
            continue
        raw = ROOT / receipt["raw_path"]
        if not receipt.get("part_validation_complete") or file_hash(raw) != receipt["raw_sha256"]:
            raise ValueError("Sealed successful prefix differs from its verified receipt")
        selected.append({"path": str(raw), "model": receipt["model"],
                         "stage": receipt["stage"], "keys": None,
                         "receipt_path": str(path), "cohort": "registered_native_failed_claim_prefix"})
    paths = [entry["path"] for entry in selected]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate selected source path")
    return selected


def load_authority(run, manifest, decision_files):
    reviews = {}
    authority_path = ROOT / "outputs/annotations/main_results/reviewed_answers.jsonl"
    source_manifest = ROOT / "outputs/annotations/main_results/source_manifest.json"
    source = json.loads(source_manifest.read_text())
    if file_hash(authority_path) != source["reviewed_answers_sha256"]:
        raise ValueError("Frozen five-model exact-QA authority bytes differ")
    for line, row, _ in rows(authority_path):
        qkey = shared.qah(row["question"], row["answer"])
        if qkey in reviews:
            raise ValueError("Duplicate exact-QA authority")
        reviews[qkey] = {"question": row["question"], "answer": row["answer"],
                         "variants": row["variants"], "authority": str(authority_path),
                         "authority_line": line}
    behavior = defaultdict(set)
    behavior_provenance = defaultdict(list)
    census = Path(manifest["census_final_labels"]["path"])
    for line, row, _ in rows(census):
        if row["dataset"] != "food101":
            continue
        q, answer = row["question"], row["text"]
        qkey = shared.qah(q, answer)
        label = row.get("label", row.get("behavior_label"))
        if label not in {"answer_assertive", "answer_uncertain", "abstain", "invalid"}:
            raise ValueError("Final closed census contains an undecided behavior label")
        behavior[qkey].add(label == "abstain")
        item = {"path": str(census), "line": line, "raw_identity": row.get("raw_identity"),
                "raw_key": row["key"], "label": label,
                "label_source": row.get("label_source"),
                "original_author_model": row.get("judge_response_model"),
                "original_call_id": row.get("judge_response_id")}
        if len(behavior_provenance[qkey]) < 3:
            behavior_provenance[qkey].append(item)
    historical = {}
    for label_source in manifest["historical_labels"]:
        for line, row, _ in rows(label_source["path"]):
            key = (row["model"], row["key"], row["source_identity"])
            if key in historical:
                raise ValueError("Duplicate source-bound historical label")
            historical[key] = {"path": label_source["path"], "line": line, "label": row}
    decisions = {}
    for path in decision_files:
        for line, row, _ in rows(path):
            qkey = shared.qah(row["question"], row["answer"])
            if row["qa_key"] != qkey:
                raise ValueError("Review QA key differs from its complete question/answer")
            if type(row.get("abstain")) is not bool:
                raise ValueError("A final review must resolve abstention")
            if row.get("annotation_model") == "gpt-5.6-luna" and row.get("annotation_effort") != "medium":
                raise ValueError("Luna annotation effort differs from the requested effort")
            decisions[qkey] = {**row, "decision_source_path": str(path),
                               "decision_source_line": line}
    return reviews, behavior, behavior_provenance, historical, decisions


def endorsed_review(variants):
    roots = [v for v in variants if v.get("_root_overlay")
             or v.get("_source") == "root_independent_gap_review"]
    followups = [v for v in variants if str(v.get("_source", "")).startswith("followup_")]
    return roots or followups or variants


def infer_qa(question, answer, patterns, reviews, behavior, decisions):
    """Extract names and decide behavior without reading the target category."""
    qkey = shared.qah(question, answer)
    parsed = shared.parse_target_blind_primary(answer, patterns)
    review = reviews.get(qkey)
    decision = decisions.get(qkey)
    variants = endorsed_review(review["variants"]) if review else []
    if decision:
        variant = dict(decision)
        relevant = [r["name"] for r in shared.relations(variant)
                    if shared.role(r.get("role")) in {"main", "coequal", "competing"}]
        variant["endorsed_primary_names"] = list(dict.fromkeys(shared.names_of(variant) + relevant))
        variants = [variant]
    normalized_variants = []
    for original in variants:
        variant = dict(original)
        if isinstance(variant.get("name_relations"), dict):
            variant["name_relations"] = [
                {"name": name, "role": relation}
                for name, relation in variant["name_relations"].items()
            ]
        if variant.get("multiple_primary") is True:
            override = variant.get("canonical_override")
            if override not in (None, "multiple_primary"):
                raise ValueError("Explicit multiple-primary review conflicts with canonical override: " + qkey)
            variant["canonical_override"] = "multiple_primary"
        normalized_variants.append(variant)
    variants = normalized_variants
    values = {v["abstain"] for v in variants if type(v.get("abstain")) is bool}
    binary_reason = None
    if values:
        abstain = next(iter(values)) if len(values) == 1 else None
        binary_reason = "exact_QA_review" if len(values) == 1 else "exact_QA_behavior_conflict"
    elif qkey in behavior:
        values = behavior[qkey]
        abstain = next(iter(values)) if len(values) == 1 else None
        binary_reason = "final_census_exact_QA_behavior" if len(values) == 1 else "census_exact_QA_behavior_conflict"
    else:
        abstain = None
    all_classes = shared.classes_for_name(answer, patterns)
    normalized = shared.fmt(answer)
    if binary_reason is None:
        if normalized in {"unknown", "unclear", "unsure", "i don't know", "i do not know"}:
            abstain, binary_reason = True, "standalone_registered_refusal"
        elif REFUSAL.search(answer) and not all_classes and not re.search(r"\b(?:but|however)\b", answer, re.I):
            abstain, binary_reason = True, "explicit_cannot_identify_without_answer"
        elif UNCERTAIN.search(answer):
            abstain, binary_reason = None, "behavior_boundary"
        else:
            abstain, binary_reason = False, "normal_answer_or_generated_fragment"
    name_boundary = False
    if not variants and parsed["status"] == "unique_class":
        remainder = parsed["primary_name"]
        for cls, pattern in patterns:
            remainder = pattern.sub(" ", remainder)
        if re.search(r"\b(?:and|or|either|not|rather|instead)\b", remainder, re.I):
            name_boundary = True
    return {"qa_key": qkey, "parsed": parsed, "variants": variants,
            "abstain": abstain, "behavior_source": binary_reason,
            "all_canonical_candidates": all_classes, "name_boundary": name_boundary,
            "review": review, "decision": decision,
            "annotation_model": decision.get("annotation_model") if decision else None,
            "annotation_effort": decision.get("annotation_effort") if decision else None,
            "annotation_call_id": decision.get("annotation_call_id", "") if decision else ""}


def score_target(answer, target, inferred, patterns):
    if inferred["abstain"] is True:
        return 0, 0, "explicit_abstention"
    if inferred["decision"] and inferred["decision"].get("needs_root"):
        return None, None, "semantic_annotation_requested_root_review"
    if inferred["variants"]:
        outcomes = []
        for original in inferred["variants"]:
            variant = dict(original)
            variant["abstain"] = inferred["abstain"]
            state = shared.extract(variant, patterns)
            override = variant.get("canonical_override")
            if not state["names"] and state["ambiguous"] is False:
                outcomes.append((0, 0, "reviewed_no_endorsed_primary_food"))
            elif override in {"explicit_outside_101", "multiple_primary"}:
                outcomes.append((0, 0, override))
            elif override:
                name = variant.get("literal_full_name")
                literal = int(shared.fmt(name) in {shared.fmt(target), shared.fmt(target.replace("_", " "))}) if name else 0
                outcomes.append((int(override == target), variant.get("literal_score_override", literal), "root_exact_QA_primary_class"))
            else:
                outcomes.append(shared.score_variant(answer, target, variant, patterns))
        canonical = {value[0] for value in outcomes}
        literal = {value[1] for value in outcomes}
        return (next(iter(canonical)) if len(canonical) == 1 else None,
                next(iter(literal)) if len(literal) == 1 else None,
                "exact_QA_review" if len(canonical) == 1 else "exact_QA_name_conflict")
    parsed = inferred["parsed"]
    if not inferred["all_canonical_candidates"]:
        return 0, 0, "target_blind_no_101_name_in_complete_response"
    if inferred["name_boundary"]:
        return None, None, "primary_or_coequal_name_boundary"
    if parsed["status"] == "unique_class":
        canonical = int(parsed["canonical_candidates"] == [target])
        name = re.sub(r"^(?:a|an|the)\s+", "", parsed["primary_name"], flags=re.I)
        literal = int(shared.fmt(name) in {shared.fmt(target), shared.fmt(target.replace("_", " "))})
        return canonical, literal, "shared_target_blind_primary"
    if parsed["status"] in {"multiple_classes", "explicit_outside_101"}:
        return 0, 0, parsed["status"]
    if shared.target_absent(answer, target):
        return 0, 0, "target_absent_from_complete_response_after_extraction"
    return None, None, "primary_name_unresolved"


def condition(row):
    sample = row["sample"]
    return {"model": row["model"], "dataset": sample["dataset"], "split": sample["split"],
            **{name: row[name] for name in COND[3:]}}


def score(run, output, decision_files, models=MODELS, received_manifests=()):
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = run / "assets/asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    samples = {row["id"]: row for _, row, _ in rows(ROOT / "data/current/all.jsonl")
               if row["dataset"] == "food101"}
    classes = sorted({sample["class"] for sample in samples.values()})
    if len(samples) != 4848 or len(classes) != 101:
        raise ValueError("Original complete Food-101 manifest is required")
    patterns = shared.compile_classes(classes)
    reviews, behavior, behavior_provenance, historical, decisions = load_authority(run, manifest, decision_files)
    sources = selected_sources(run, manifest, models, received_manifests)
    qa_cache, seen, ranks, attempts, pending = {}, set(), {}, defaultdict(dict), {}
    counts, stats, source_counts = defaultdict(Counter), Counter(), []
    score_path = output / "score_rows.jsonl.gz"
    with gzip.open(score_path, "xt", encoding="utf-8", newline="\n", compresslevel=1) as destination:
        for entry in sources:
            chosen = 0
            for line, row, line_sha in rows(entry["path"]):
                if entry["keys"] is not None and row["key"] not in entry["keys"]:
                    continue
                sample = row["sample"]
                if row["model"] != entry["model"] or sample != samples.get(sample["id"]):
                    raise ValueError("Chosen source model/sample differs from registered original")
                if row.get("status", "ok") != "ok":
                    raise ValueError("Chosen source contains a failed row")
                identity = (row["model"], entry["stage"], row["key"])
                if identity in seen:
                    raise ValueError("Duplicate chosen supplemental task: " + str(identity))
                seen.add(identity)
                chosen += 1
                provenance = source_record(entry, line, row, line_sha)
                if entry["stage"] == "candidate":
                    key = row["model"], sample["id"]
                    if key in ranks:
                        raise ValueError("Duplicate candidate reference")
                    rank = row["gold_rank"]
                    if not 1 <= rank <= 101:
                        raise ValueError("Invalid registered candidate rank")
                    ranks[key] = {"gold_rank": rank, **provenance}
                    stats["candidate_rows"] += 1
                    continue
                question, answer = sample["question"], row["text"]
                qkey = shared.qah(question, answer)
                if qkey not in qa_cache:
                    qa_cache[qkey] = infer_qa(question, answer, patterns, reviews, behavior, decisions)
                inferred = dict(qa_cache[qkey])
                if shared.configured_marker(answer, row) and not inferred["variants"]:
                    inferred.update(abstain=True, behavior_source="exact_actual_condition_marker")
                label = historical.get((row["model"], row["key"], row.get("identity")))
                if label and label["label"]["text"] != answer:
                    raise ValueError("Source-bound historical answer text differs")
                if (inferred["abstain"] is None and label
                        and label["label"]["screening_label"] == "abstain"):
                    inferred.update(abstain=True, behavior_source="source_bound_historical_abstention")
                canonical, literal, reason = score_target(answer, sample["class"], inferred, patterns)
                cond = condition(row)
                states = [shared.extract(v, patterns) for v in inferred["variants"]]
                names = sorted({name for state in states for name in state["literal"]})
                if not states and inferred["parsed"]["primary_name"]:
                    names = [inferred["parsed"]["primary_name"]]
                record = {**cond, "main_marker": row["marker"], "key": row["key"],
                          "sample_id": sample["id"], "target_class": sample["class"],
                          "qa_key": qkey, "stage": entry["stage"], **provenance,
                          "question": question, "answer": answer, "seed": row.get("seed"),
                          "terminated": row.get("terminated"),
                          "canonical_name_in_primary_score": canonical,
                          "literal_extracted_name_score": literal,
                          "literal_extracted_names": names,
                          "canonical_class_candidates": (sorted({c for state in states for c in state["classes"]})
                                                         if states else inferred["parsed"]["canonical_candidates"]),
                          "abstain": inferred["abstain"], "score_reason": reason,
                          "behavior_source": inferred["behavior_source"],
                          "behavior_history_sources": behavior_provenance.get(qkey, []),
                          "name_history_source": inferred["review"]["authority"] if inferred["review"] else None,
                          "historical_preliminary": label,
                          "annotation_model": inferred["annotation_model"],
                          "annotation_effort": inferred["annotation_effort"],
                          "annotation_call_id": inferred["annotation_call_id"],
                          "annotation_session_id": inferred["decision"].get("annotation_session_id") if inferred["decision"] else None,
                          "annotation_source_path": inferred["decision"].get("decision_source_path") if inferred["decision"] else None,
                          "annotation_source_line": inferred["decision"].get("decision_source_line") if inferred["decision"] else None,
                          "annotation_needs_root": inferred["decision"].get("needs_root", False) if inferred["decision"] else False,
                          "main_prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                          "reference_prompt_sha256": hashlib.sha256(row["reference_prompt"].encode()).hexdigest() if row.get("reference_prompt") is not None else None,
                          "config_sha256": stable_hash(row["config"])}
                destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                stats[entry["stage"] + "_rows"] += 1
                stats["canonical_pending_rows"] += canonical is None
                stats["abstain_pending_rows"] += inferred["abstain"] is None
                stats["exact_QA_name_reused_rows"] += inferred["review"] is not None
                stats["exact_QA_behavior_reused_rows"] += inferred["behavior_source"] == "final_census_exact_QA_behavior"
                if (stats["formal_rows"] + stats["independent_rows"]) % 10000 == 0:
                    atomic_json(output / "progress.json", {
                        "rows": stats["formal_rows"] + stats["independent_rows"],
                        "source_path": entry["path"], "source_line": line,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "scoring"})
                if canonical is None or inferred["abstain"] is None:
                    group = pending.setdefault(qkey, {"qa_key": qkey, "question": question,
                               "answer": answer, "reason": [], "memberships": []})
                    group["reason"] = sorted(set(group["reason"] + [reason, inferred["behavior_source"]]))
                    group["memberships"].append({"model": row["model"], "key": row["key"],
                        "stage": entry["stage"], "split": sample["split"], "sample_id": sample["id"],
                        "target_class": sample["class"], **provenance})
                if entry["stage"] == "independent":
                    if row["kind"] != "independent_attempt" or row.get("attempt") is not True:
                        raise ValueError("Reference source is not an independent attempt")
                    key = row["model"], sample["id"]
                    replicate = row["replicate"]
                    if replicate not in range(10) or replicate in attempts[key]:
                        raise ValueError("Duplicate or invalid independent replicate")
                    attempts[key][replicate] = {"replicate": replicate, "key": row["key"],
                        "canonical_name_in_primary_score": canonical, "abstain": inferred["abstain"],
                        "terminated": row.get("terminated"), "seed": row["seed"], **provenance}
                else:
                    counter = counts[tuple(cond[name] for name in COND)]
                    counter["n"] += 1
                    counter["canonical_correct"] += canonical == 1
                    counter["canonical_unknown"] += canonical is None
                    counter["literal_correct"] += literal == 1
                    counter["literal_unknown"] += literal is None
                    counter["abstain_true"] += inferred["abstain"] is True
                    counter["abstain_unknown"] += inferred["abstain"] is None
            if entry["keys"] is not None and chosen != len(entry["keys"]):
                raise ValueError("Chosen source is missing its selected keys")
            source_counts.append({"model": entry["model"], "stage": entry["stage"],
                                  "source_path": entry["path"], "selected_rows": chosen})
    with (output / "boundary_queue.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for qkey in sorted(pending):
            stream.write(json.dumps(pending[qkey], ensure_ascii=False, separators=(",", ":")) + "\n")
    references = []
    for model in models:
        for sample_id, sample in sorted(samples.items()):
            key = model, sample_id
            rank = ranks.get(key)
            observed = attempts[key]
            complete = set(observed) == set(range(10))
            resolved = complete and all(item["canonical_name_in_primary_score"] is not None
                                        for item in observed.values())
            correct = sum(item["canonical_name_in_primary_score"] == 1 for item in observed.values())
            gt = ((rank["gold_rank"] > 1 and correct == 0) if rank and resolved else None)
            references.append({"model": model, "dataset": "food101", "sample_id": sample_id,
                "split": sample["split"], "target_class": sample["class"],
                "gold_rank": rank["gold_rank"] if rank else None,
                "rank_source": rank, "attempts": [observed[r] for r in sorted(observed)],
                "attempt_count": len(observed), "correct_count": correct if resolved else None,
                "reference_G": gt, "reference_complete": rank is not None and resolved,
                "missing_replicates": sorted(set(range(10)) - set(observed)),
                "historical_accepted_reference": None})
    with (output / "reference_G.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for record in references:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    fields = list(COND) + ["main_marker", "n", "canonical_correct", "canonical_unknown",
                          "literal_correct", "literal_unknown", "abstain_true", "abstain_unknown"]
    with (output / "condition_counts.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, counter in sorted(counts.items(), key=lambda pair: tuple(map(str, pair[0]))):
            item = dict(zip(COND, key))
            writer.writerow({**item, "main_marker": item["marker"],
                             **{field: counter[field] for field in fields[len(COND) + 1:]}})
    summary = {"schema": "kdm_remaining11_main_score_v3", "updated_utc": datetime.now(timezone.utc).isoformat(),
               "models": list(models),
               "rows": stats["formal_rows"] + stats["independent_rows"], **stats,
               "unique_scored_QA": len(qa_cache), "pending_boundary_questions": len(pending),
               "formal_condition_cells_observed": len(counts),
               "reference_expected_rows": len(references),
               "reference_complete_rows": sum(row["reference_complete"] for row in references),
               "reference_pending_rows": sum(not row["reference_complete"] for row in references),
               "reference_rule": "gold_rank>1 and all ten current-primary correct attempts=0",
               "source_manifest": str(manifest_path), "source_manifest_sha256": file_hash(manifest_path),
               "additional_received_manifests": [{"path": str(path), "sha256": file_hash(path)}
                                                for path in received_manifests],
               "selected_sources": source_counts, "decision_files": [str(path) for path in decision_files],
               "rule_authorship": "program rules using the frozen main-results scoring functions",
               "scorer_sha256": file_hash(Path(__file__)),
               "human_authorship": "only explicitly supplied semantic review records retain their actual author/model",
               "score_rows_path": str(score_path), "complete": False}
    write_json(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "selected_sources"}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="run_20260930_140337")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--decision-file", action="append", default=[], type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--received-manifest", action="append", default=[], type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.output_name):
        raise ValueError("Invalid exclusive score checkpoint name")
    run = ROOT / "outputs/supplemental/remaining11" / args.run
    decisions = [path if path.is_absolute() else ROOT / path for path in args.decision_file]
    if len(args.models) != len(set(args.models)):
        raise ValueError("Duplicate scoring model selection")
    received = [path if path.is_absolute() else ROOT / path for path in args.received_manifest]
    score(run, run / "scores" / args.output_name, decisions, args.models, received)


if __name__ == "__main__":
    main()
