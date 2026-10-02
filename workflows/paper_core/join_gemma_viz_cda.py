"""Join the two sealed Gemma Viz CDA pieces without changing their rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from kdm.io import file_hash, stable_hash, within
from workflows.paper_core.dev_viz import audit_rows, condition, planned_tasks
from workflows.paper_core.score_dev_viz import audit_raw

PARTS = (
    (260, "25f5784729f01f855e3fbfdba24b344fd922cb6dccca04c187af69a2e2f8f508",
     "787c3c1761778bb3b3c4cd94425c3bb157acd458730fdb342d20a00fc629b961"),
    (252, "c681e3aa720febe6fb54ad9598cecfbb325bb4ee0a410a8a9b4c076ba0e39d4b",
     "61137866a26940a0b62fc6b2aced6dd4b4147e264f1f4ddccc956ceb6f49cec6"),
)
NAME = "cda_visual_3_7c68de7eeef18e7b.jsonl"
SIDECAR_SHA = "7cb27758cd74901db910d1ac63cea3f435f57b26e255bcd910ca7b60c46eaa17"
SELECTED_SHA = "0d59f129f2c02b002fddfaa21af5c916d11cc958c8ab69a33d13e4c248ecebf2"


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-root", required=True)
    parser.add_argument("--selected-configs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    inputs = within(ROOT, args.part_root)
    selected = within(ROOT, args.selected_configs)
    output = within(ROOT, args.out)
    output.resolve().relative_to(ROOT / "outputs/paper_core_20261002_dev_viz")
    require(file_hash(selected) == SELECTED_SHA, "The frozen Gemma dev selection differs")
    tasks = [task for task in planned_tasks("gemma3_4b", "viz512", args.selected_configs)
             if condition(task) == "7c68de7eeef18e7b"]
    require(len(tasks) == 512, "The original CDA plan does not contain 512 inputs")
    marker = tasks[0]["marker"]
    payloads, sources, links = [], [], []
    identity_bytes = None
    for index, (expected_n, raw_sha, receipt_sha) in enumerate(PARTS):
        raw = inputs / f"part{index}" / NAME
        identity_path = raw.with_suffix(".identity.json")
        receipt_path = raw.with_suffix(".complete.json")
        require(file_hash(raw) == raw_sha and file_hash(receipt_path) == receipt_sha
                and file_hash(identity_path) == SIDECAR_SHA,
                "A sealed original part changed bytes")
        sidecar = json.loads(identity_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        definition = sidecar["definition"]
        core = {key: value for key, value in definition.items()
                if key not in {"shard", "n_shards", "base_config"}}
        require(sidecar["identity"] == stable_hash(definition)
                and receipt["identity"] == stable_hash(core)
                and definition["model"] == "gemma3_4b" and definition["stage"] == "viz512"
                and definition["selected_configs_sha256"] == SELECTED_SHA
                and receipt["passed"] is True and receipt["rows"] == expected_n
                and receipt["raw_sha256"] == raw_sha
                and receipt["condition"] == "7c68de7eeef18e7b"
                and receipt["method"] == "cda_visual" and receipt["marker"] == marker,
                "The original part identity or condition differs")
        current_identity = identity_path.read_bytes()
        require(identity_bytes is None or identity_bytes == current_identity,
                "The two original sidecars differ")
        identity_bytes = current_identity
        lines = raw.read_bytes().splitlines(keepends=True)
        require(len(lines) == expected_n and all(line.endswith(b"\n") for line in lines),
                "A sealed original JSONL line is incomplete")
        for line_n, line in enumerate(lines, 1):
            row = json.loads(line)
            audit_raw(row, row["sample"], "gemma3_4b")
            require(row["identity"] == sidecar["identity"], "A row ledger identity differs")
            links.append({"key": row["key"], "source_host": "6403",
                          "original_source_path": receipt["raw"], "source_part": index,
                          "source_line": line_n, "source_line_sha256": hashlib.sha256(line).hexdigest(),
                          "raw_sha256": raw_sha, "complete_sha256": receipt_sha,
                          "identity_sha256": SIDECAR_SHA})
        payloads.append(b"".join(lines))
        sources.append({"part": index, "rows": expected_n, "copied_raw": str(raw),
                        "original_source_host": "6403", "original_complete_receipt": receipt,
                        "copied_raw_sha256": raw_sha, "copied_complete_sha256": receipt_sha,
                        "copied_identity_sha256": SIDECAR_SHA})
    output.mkdir(parents=True, exist_ok=False)
    raw_dir = output / "gemma3_4b/raw"
    raw_dir.mkdir(parents=True)
    merged = raw_dir / NAME
    merged.write_bytes(b"".join(payloads))
    accepted_rows = audit_rows("gemma3_4b", tasks, merged)
    require(len({row["key"] for row in accepted_rows}) == len(links) == 512,
            "The actual joined task keys are incomplete or duplicated")
    shutil.copyfile(inputs / "part0" / NAME.replace(".jsonl", ".identity.json"),
                    merged.with_suffix(".identity.json"))
    completed = datetime.now(timezone.utc).isoformat()
    receipt = {"passed": True, "identity": sources[0]["original_complete_receipt"]["identity"],
               "condition": "7c68de7eeef18e7b", "method": "cda_visual", "marker": marker,
               "rows": 512, "required": 512, "raw": str(merged.relative_to(ROOT)),
               "raw_sha256": file_hash(merged), "completed_at_utc": completed,
               "generation_wall_s": sum(row["wall_s"] for row in accepted_rows),
               "scope": "composite_complete_condition", "original_source_parts": sources,
               "new_generations": 0, "scientific_row_bytes_changed": False,
               "actual_executor": {"agent": "/root", "model": "gpt-6.1-sol", "effort": "max", "call_id": ""}}
    merged.with_suffix(".complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (output / "original_row_sources.jsonl").open("x") as stream:
        for link in links:
            stream.write(json.dumps(link) + "\n")
    (output / "JOIN_RECEIPT.json").write_text(json.dumps({
        **receipt, "source_join_entry_sha256": file_hash(Path(__file__)),
        "original_row_sources_sha256": file_hash(output / "original_row_sources.jsonl"),
        "actual_command": [sys.executable, *sys.argv], "GPU_initialized": False}, indent=2) + "\n")
    print(json.dumps({"passed": True, "rows": 512, "source_parts": [260, 252],
                      "raw_sha256": receipt["raw_sha256"], "new_generations": 0}))


if __name__ == "__main__":
    main()
