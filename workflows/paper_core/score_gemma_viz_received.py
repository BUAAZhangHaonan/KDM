"""Score the three received Gemma Viz cohorts with explicit accepted reviews."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from kdm.io import file_hash, within

BASE = "outputs/paper_core_20261002_dev_viz"
INPUTS = (
    ("first4", "received_viz_gemma4_20261003_0150", 2048),
    ("matched1", "received_viz_gemma_matched1_20261003_0225", 512),
    ("cda1", "joined_viz_gemma_cda512_20261003_0240", 512),
)
FINAL27 = BASE + "/annotation/gemma_pilot27_schema_adapter_20261003_0210_v3/accepted_root27_viz_decisions.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    authority_dir = within(ROOT, args.authority_root)
    authority = json.loads((authority_dir / "receipt.json").read_text(encoding="utf-8-sig"))
    if authority["passed"] is not True:
        raise ValueError("The supplied real review authority is not accepted")
    files = []
    for record in authority["outputs"]:
        path = authority_dir / Path(record["output"].replace("\\", "/")).name
        if file_hash(path) != record["output_sha256"]:
            raise ValueError("The received actual review changed source bytes")
        files.append(str(path.relative_to(ROOT)))
    if file_hash(ROOT / FINAL27) != "5390e6dc075000697571a190445bca4fdd99ec58026464162dc015507f7d0afb":
        raise ValueError("The final source-bound root27 adapter differs")
    previous_path = ROOT / BASE / "scoring_viz_gemma_first4_root27_20261003_0155/receipt.json"
    previous = json.loads(previous_path.read_text())
    if previous["passed"] is not True or previous["rows"] != 2048 or previous["reference_join_missing"] != 0:
        raise ValueError("The original accepted Gemma source cohort differs")
    base_command = list(previous["actual_command"])
    obsolete = BASE + "/annotation/gemma_pilot27_received_20261003_0150/gemma_viz_pilot27_actual_root_20261003_0105/root_decisions27.jsonl"
    old_index = base_command.index(obsolete)
    if base_command[old_index - 1] != "--decision-file":
        raise ValueError("The original root27 command source differs")
    base_command[old_index] = FINAL27
    for path in files:
        base_command.extend(("--decision-file", path))
    output = within(ROOT, args.out)
    output.resolve().relative_to(ROOT / BASE)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for name, input_name, expected_rows in INPUTS:
        command = list(base_command)
        command[command.index("--input-dir") + 1] = BASE + "/" + input_name
        target = output / name
        command[command.index("--out") + 1] = str(target.relative_to(ROOT))
        started = datetime.now(timezone.utc).isoformat()
        subprocess.run(command, cwd=ROOT, check=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        receipt_path = target / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if (receipt["passed"] is not True or receipt["rows"] != expected_rows
                or receipt["raw_complete_conditions"] != expected_rows // 512
                or receipt["reference_join_missing"] != 0):
            raise ValueError("The actually scored source cohort has a coverage or reference error")
        results.append({"name": name, "rows": expected_rows,
                        "conditions": receipt["conditions"], "receipt": str(receipt_path.relative_to(ROOT)),
                        "receipt_sha256": file_hash(receipt_path), "pending_QA": receipt["pending_QA"],
                        "quality_pending_rows": receipt["quality_pending_rows"],
                        "abstain_pending_rows": receipt["abstain_pending_rows"],
                        "primary_complete_conditions": receipt["primary_complete_conditions"],
                        "actual_command": command, "started_utc": started,
                        "original_rule_executor_template": receipt["rule_executor"]})
    proof = {"passed": True, "actual_scored_rows": 3072, "cohorts": results,
             "completed_utc": datetime.now(timezone.utc).isoformat(),
             "source_authority_receipt": str((authority_dir / "receipt.json").relative_to(ROOT)),
             "source_authority_receipt_sha256": file_hash(authority_dir / "receipt.json"),
             "authority_QA": authority["accepted_QA"], "source_entry_sha256": file_hash(Path(__file__)),
             "actual_executor": {"agent": "/root", "model": "gpt-6.1-sol", "effort": "max", "call_id": ""},
             "old_source_or_scores_modified": False, "new_semantic_judgments_generated": 0,
             "GPU_initialized": False, "final_all_labels_complete_claimed": False}
    (output / "round_receipt.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"passed": True, "rows": 3072,
                      "pending_by_cohort": {row["name"]: row["pending_QA"] for row in results}}))


if __name__ == "__main__":
    main()
