#!/usr/bin/env python3
"""Zero-API preliminary screening of completed formal shards only.

Example (after the shard has a verified complete.json):
  venv/bin/python workflows/acceleration_v4/formal_postprocess.py \
    --shard outputs/records/acceleration_v4/RUN/MODEL/shard_000_of_001 \
    --out outputs/annotations/acceleration_v4/RUN_SCREEN
Repeat --shard to combine completed disjoint shards. --check-only writes nothing.
No live-file snapshots, paid API, semantic extraction, GPU work or final GT claims.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash, within
from kdm.protocol import validate_freeze
import formal_runner as formal

RULE = "workflows/quick_match_v1/match.py"
spec = importlib.util.spec_from_file_location("kdm_quick_match_reused_v1", ROOT / RULE)
quick = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quick)


def condition(row):
    return (row["model"], formal.stage_of(row), row["kind"], row["method"],
            row["marker"], row["reference_marker"], row["reference_guided"])


def counts_record(counts):
    n = sum(counts.values())
    correct, wrong = counts["correct"], counts["incorrect"]
    abstain, unresolved = counts["abstain"], counts["needs_confirmation"]
    if correct + wrong + abstain + unresolved != n:
        raise ValueError("Unexpected Food preliminary label")
    return {"rows": n, "matched": correct + wrong + abstain,
            "matched_answer_correct": correct, "matched_answer_incorrect": wrong,
            "exact_abstention": abstain, "unresolved": unresolved}


def validate_completed_shards(record_paths):
    freeze = validate_freeze(ROOT)
    samples = [s for s in read_jsonl(ROOT / "data/current/all.jsonl") if s["dataset"] == "food101"]
    if len(samples) != 4848 or Counter(s["split"] for s in samples) != {"dev": 2424, "eval": 2424}:
        raise ValueError("Original Food coverage changed")
    plans = json.loads((ROOT / "configs/kdm/method_plan.json").read_text())
    accepted, shard_keys = [], set()
    counts_by_model = {}
    for record_path in record_paths:
        path = within(ROOT, record_path)
        if not path.is_relative_to(ROOT / "outputs/records/acceleration_v4"):
            raise ValueError("Expected a formal acceleration_v4 record directory")
        required = [path / x for x in ("admission.json", "progress.json", "complete.json")]
        if not all(p.is_file() for p in required):
            raise ValueError("Only completed formal shards may be screened: " + str(path))
        admission, progress, complete = [json.loads(p.read_text()) for p in required]
        model = admission["model"]
        if model not in formal.FIVE or admission.get("schema") != "kdm_acceleration_v4_frozen_formal_generation":
            raise ValueError("Not a registered first-panel formal shard")
        if progress["status"] != "generation_complete" or complete.get("generation_complete") is not True:
            raise ValueError("Live/failed/incomplete shard refused")
        if complete["admission_sha256"] != file_hash(required[0]):
            raise ValueError("Completion does not bind to admission")
        expected_sources = {
            "panel_sha256": ROOT / formal.PANEL,
            "amendment_sha256": ROOT / formal.AMENDMENT,
            "manifest_sha256": ROOT / "data/current/all.jsonl",
            "method_plan_sha256": ROOT / "configs/kdm/method_plan.json",
            "backend_spec_sha256": ROOT / f"configs/runtime/{model}.json",
            "freeze_receipt_sha256": ROOT / "outputs/records/preregistration_freeze.json",
            "runner_sha256": ROOT / "workflows/acceleration_v4/formal_runner.py",
        }
        if any(admission[k] != file_hash(p) for k, p in expected_sources.items()) or admission["source_blobs"] != freeze["source_blobs"]:
            raise ValueError("Bound formal source/spec/panel changed")
        shard, n_shards = progress["shard"], progress["n_shards"]
        pair = (model, shard)
        if pair in shard_keys or model in counts_by_model and counts_by_model[model] != n_shards:
            raise ValueError("Duplicate shard or inconsistent model partition")
        shard_keys.add(pair)
        counts_by_model[model] = n_shards
        if complete["model"] != model or complete["shard"] != shard or complete["n_shards"] != n_shards:
            raise ValueError("Completion partition mismatch")
        raw = within(ROOT, progress["output"])
        if not raw.is_relative_to(ROOT / "outputs/raw/acceleration_v4"):
            raise ValueError("Unexpected raw output location")
        errors = Path(str(raw) + ".errors.jsonl")
        if errors.exists() and errors.stat().st_size:
            raise ValueError("Completed shard contains failures; retain for investigation")
        before = raw.stat()
        if complete["raw_sha256"] != file_hash(raw) or complete["sidecar_sha256"] != file_hash(raw.with_suffix(".identity.json")):
            raise ValueError("Completed raw file or sidecar changed")
        proof = formal.verify_output(raw, model, samples, plans[model]["food101"], admission, shard, n_shards)
        if proof["rows"] != complete["rows"] or proof["rows"] != admission["task_plan"]["expected_generation_rows"]:
            raise ValueError("Completion has incomplete generation coverage")
        after = raw.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Input changed during completed-shard verification")
        accepted.append({
            "record_directory": str(path.relative_to(ROOT)), "raw_path": str(raw.relative_to(ROOT)),
            "model": model, "shard": shard, "n_shards": n_shards, "rows": proof["rows"],
            "raw_sha256": proof["raw_sha256"], "sidecar_sha256": proof["sidecar_sha256"],
            "admission_sha256": file_hash(required[0]), "complete_sha256": file_hash(required[2]),
            "stage_counts": proof["stage_counts"],
        })
    if not accepted:
        raise ValueError("At least one completed shard required")
    model_coverage = {
        model: {"included_shards": sorted(s for m, s in shard_keys if m == model), "n_shards": count,
                "full_model_generation_present": {s for m, s in shard_keys if m == model} == set(range(count))}
        for model, count in counts_by_model.items()
    }
    return accepted, model_coverage


def markdown_summary(report):
    lines = [
        "# Food formal generation: preliminary character matching", "",
        "Only completed and provenance-verified input shards are included. These are full-answer lexical matches, not final semantic labels or abstention GT. Unmatched text remains unresolved. No API request or human review is claimed.", "",
        f"Screened rows: {report['rows']}. Included complete model outputs: {sum(x['full_model_generation_present'] for x in report['model_coverage'].values())}/5. Full panel present: {report['full_panel_generation_present']}.", "",
        "## Stage totals", "",
        "| Model | Stage | Rows | Matched correct | Matched incorrect | Exact abstention | Unresolved |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["stages"]:
        lines.append("| " + " | ".join(str(row[k]) for k in (
            "model", "stage", "rows", "matched_answer_correct", "matched_answer_incorrect", "exact_abstention", "unresolved")) + " |")
    lines += ["", "## UNKNOWN main-condition preliminary table", "",
              "Every row below uses UNKNOWN for both main and reference where applicable. A subset of completed shards is explicitly marked by model_coverage in summary.json; it is not a whole-model result.", "",
              "| Model | Method | Rows | Matched correct | Matched incorrect | Exact abstention | Unresolved |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["conditions"]:
        if row["stage"] == "unknown_main":
            lines.append("| " + " | ".join(str(row[k]) for k in (
                "model", "method", "rows", "matched_answer_correct", "matched_answer_incorrect", "exact_abstention", "unresolved")) + " |")
    lines += ["", "All prompt-matrix conditions are retained in summary.json. No unresolved item was counted as an incorrect answer. Generation completion does not establish semantic judging, should-abstain GT, mechanism completion or research completion.", ""]
    return "\n".join(lines)


def screen(accepted, coverage, out):
    out = within(ROOT, out)
    if not out.is_relative_to(ROOT / "outputs/annotations/acceleration_v4"):
        raise ValueError("Fresh screening output must be inside outputs/annotations/acceleration_v4")
    out.mkdir(parents=True, exist_ok=False)
    matcher = quick.Matcher()
    group_counts, stage_counts = defaultdict(Counter), defaultdict(Counter)
    seen, total = set(), Counter()
    sources = []
    partial = out / "labels.partial.jsonl"
    try:
        with partial.open("x", encoding="utf-8") as sink:
            for source in accepted:
                path = ROOT / source["raw_path"]
                before = path.stat()
                digest = hashlib.sha256()
                rows = 0
                with path.open("rb") as stream:
                    for line_number, line in enumerate(stream, 1):
                        row = json.loads(line)
                        pair = (row["model"], row["key"])
                        if pair in seen:
                            raise ValueError("Duplicate source response across included shards")
                        seen.add(pair)
                        result = matcher.classify(row)
                        if result["final_gt"] or result["human_reviewed"]:
                            raise ValueError("Preliminary matcher cannot claim final/human labels")
                        group = condition(row)
                        group_counts[group][result["screening_label"]] += 1
                        stage_counts[group[:2]][result["screening_label"]] += 1
                        total[result["screening_label"]] += 1
                        derived = {
                            "schema": "formal_v4_quick_match_v1", "model": row["model"], "key": row["key"],
                            "sample_id": row["sample"]["id"], "dataset": "food101", "split": row["sample"]["split"],
                            "stage": group[1], **{k: row[k] for k in formal.FIELDS},
                            "source_identity": row["identity"], "source_path": source["raw_path"],
                            "source_line": line_number, "source_row_sha256": hashlib.sha256(line).hexdigest(), **result,
                        }
                        sink.write(json.dumps(derived, ensure_ascii=False, allow_nan=False) + "\n")
                        digest.update(line)
                        rows += 1
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or digest.hexdigest() != source["raw_sha256"] or rows != source["rows"]:
                    raise ValueError("Completed source changed during screening")
                sources.append(source)
        conditions = []
        for key, counts in sorted(group_counts.items()):
            conditions.append({**dict(zip(("model", "stage", "kind", "method", "marker", "reference_marker", "reference_guided"), key)),
                               **counts_record(counts)})
        stages = [{"model": model, "stage": stage, **counts_record(counts)}
                  for (model, stage), counts in sorted(stage_counts.items())]
        report = {
            "schema": "formal_v4_completed_shards_preliminary_screening", "automatic_preliminary_only": True,
            "api_requests": 0, "gpu_used": False, "human_reviewed": False, "final_gt": False,
            "semantic_labeling_complete": False, "research_complete": False,
            **counts_record(total), "counts": dict(total), "conditions": conditions, "stages": stages,
            "model_coverage": coverage, "full_panel_generation_present":
                set(coverage) == formal.FIVE and all(x["full_model_generation_present"] for x in coverage.values()),
            "sources": sources,
            "rule_files_sha256": {p: file_hash(ROOT / p) for p in (
                RULE, "configs/kdm/food_aliases.json", "src/kdm/scoring.py",
                "src/kdm/models/official_vqa_normalizer.py", "workflows/acceleration_v4/formal_postprocess.py")},
            "limits": "Full-string exact alias/abstention matching only. Unresolved is neither wrong nor correct. Counts describe included verified completed shards. No independent-answer/closed-rank joint GT or human review claim.",
        }
        partial.rename(out / "labels.jsonl")
        atomic_json(out / "summary.json", report)
        (out / "SUMMARY.md").write_text(markdown_summary(report), encoding="utf-8")
        print(json.dumps({"out": str(out), **counts_record(total), "api_requests": 0, "final_gt": False}))
    except BaseException as exc:
        atomic_json(out / "failure.json", {"error": type(exc).__name__ + ": " + str(exc),
                    "partial_outputs_not_accepted": True, "final_gt": False, "api_requests": 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True,
                        help="Completed formal record directory; repeat for disjoint shards")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and args.out is None:
        parser.error("--out required unless --check-only")
    accepted, coverage = validate_completed_shards(args.shard)
    if args.check_only:
        print(json.dumps({"accepted": accepted, "model_coverage": coverage, "gpu_used": False,
                          "api_requests": 0, "screening_performed": False}, ensure_ascii=False, indent=2))
    else:
        screen(accepted, coverage, args.out)


if __name__ == "__main__":
    main()

