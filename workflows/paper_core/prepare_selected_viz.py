"""Write VizWiz task keys from a complete, frozen Food dev selection."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.io import atomic_json, file_hash, within
from kdm.pipeline import task_id
from kdm.prompts import MARKERS
from workflows.paper_core.dev_viz import DEV_METHODS, MODELS, planned_tasks, read_missing, roster

FIELDS = ("method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")


def require(test, reason):
    if not test:
        raise ValueError(reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--selected-configs", required=True)
    parser.add_argument("--inventory", default="outputs/paper_core_20261002_dev_viz/inventory_20261002_1700/CURRENT_STATE.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Planning must explicitly hide GPUs")
    selected_path = within(ROOT, args.selected_configs)
    analysis_path = selected_path.parent / "analysis_receipt.json"
    choices = json.loads(selected_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    require(analysis["passed"] and analysis["dev_only_selection"]
            and args.model in analysis["models"]
            and analysis["selected_configs"] == 4 * len(analysis["models"])
            and analysis["dev_rows"] == analysis["expected_dev_rows"] == 6464 * len(analysis["models"]),
            "VizWiz requires a passed complete Food-dev-only selection")
    require(len(choices) == analysis["selected_configs"], "The saved selection count changed")
    chosen = [row for row in choices if row["model"] == args.model]
    require(len(chosen) == 4 and {row["method"] for row in chosen} == set(DEV_METHODS),
            "Each selected model requires four distinct methods")
    for row in chosen:
        require(row["selection"] == "dev_selected" and row["selection_split"] == "dev"
                and row["n"] == 404 and row["marker"] in MARKERS
                and bool(row["selection_frozen_at_utc"]), "A selected configuration is incomplete")
    score_sources = {row["source_scores"]: row["source_scores_sha256"] for row in chosen}
    require(all(file_hash(within(ROOT, path)) == sha for path, sha in score_sources.items()),
            "The original dev scores bound to a frozen choice changed")
    inventory_path = within(ROOT, args.inventory)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    viz = inventory["fixed_vizwiz512"]
    require(inventory["schema"] == "core_five_dev_viz_registered_asset_inventory_v1"
            and viz["existing_registered_method_raw_complete"] == 0
            and (viz["n"], viz["official_unanswerable"], viz["official_answerable"], viz["verified_images"])
            == (512, 166, 346, 512), "The registered VizWiz zero-match inventory differs")
    roster_path, samples = roster("viz512")
    require({sample["id"] for sample in samples} and len(samples) == 512, "The fixed roster differs")
    tasks = planned_tasks(args.model, "viz512", args.selected_configs)
    keys = [task_id(args.model, task) for task in tasks]
    require(len(keys) == len(set(keys)) and len(keys) in (2560, 3072), "The bounded VizWiz task scope changed")
    counts = Counter((task["method"], task["kind"], task["marker"]) for task in tasks)
    require(set(counts.values()) == {512}, "All methods must share the complete fixed roster")
    out = within(ROOT, args.out)
    out.relative_to(ROOT / "outputs/paper_core_20261002_dev_viz")
    out.mkdir(parents=True, exist_ok=False)
    missing_path = out / (args.model + "_viz512_missing.jsonl")
    with missing_path.open("x", encoding="utf-8") as stream:
        for key, task in zip(keys, tasks):
            record = {"key": key, "model": args.model, "sample_id": task["sample"]["id"],
                      **{field: task[field] for field in FIELDS}}
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    checked = read_missing(str(missing_path.relative_to(ROOT)), args.model, tasks)
    require(len(checked) == len(tasks), "The saved task-key manifest differs from the actual frozen plan")
    receipt = dict(passed=True, model=args.model, stage="viz512", expected_inputs=512,
        official_unanswerable=166, official_answerable=346, planned_missing_generation_keys=len(keys),
        reused_registered_method_generations=0, new_generations_completed=0, GPU_initialized=False,
        selection_source=str(selected_path.relative_to(ROOT)), selection_source_sha256=file_hash(selected_path),
        selection_analysis_source=str(analysis_path.relative_to(ROOT)), selection_analysis_sha256=file_hash(analysis_path),
        selected_configuration_records=chosen, dev_score_sources=score_sources,
        registered_zero_match_inventory=str(inventory_path.relative_to(ROOT)), inventory_sha256=file_hash(inventory_path),
        zero_match_inventory_checked_utc=inventory["checked_utc"],
        roster_source=str(roster_path.relative_to(ROOT)), roster_sha256=file_hash(roster_path),
        configurations=[dict(method=method, kind=kind, marker=marker, missing_keys=n)
                        for (method, kind, marker), n in counts.items()],
        missing_manifest=str(missing_path.relative_to(ROOT)), missing_manifest_sha256=file_hash(missing_path),
        created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=file_hash(Path(__file__)))
    atomic_json(out / "plan_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
