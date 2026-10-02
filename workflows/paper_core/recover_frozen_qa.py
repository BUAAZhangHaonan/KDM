#!/usr/bin/env python3
"""Recover existing full QA through the frozen dictionary's exact raw pointers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kdm.io import file_hash, within
from workflows.main_results.score import qah
from workflows.paper_core.assemble_nine_main_food import output_json


def load_recovered(directory):
    directory = within(ROOT, directory)
    receipt = json.loads((directory / "receipt.json").read_text())
    source = ROOT / "outputs/paper_20260929/qa.parquet"
    if (not receipt["passed"] or receipt["frozen_QA_sha256"] != file_hash(source)
            or file_hash(directory / "recovered_qa.parquet") != receipt["outputs"]["recovered_qa.parquet"]):
        raise ValueError("The exact recovered QA or original dictionary changed")
    data = pd.read_parquet(directory / "recovered_qa.parquet")
    if len(data) != 44782 or data[["qa_key", "question", "answer"]].isna().any().any():
        raise ValueError("The recovered full QA dictionary is incomplete")
    return data, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = within(ROOT, args.output)
    output.relative_to(ROOT / "outputs/paper_core_20261002_dev_viz")
    source = ROOT / "outputs/paper_20260929"
    path = source / "qa.parquet"
    frame = pd.read_parquet(path)
    if len(frame) != 44782 or frame.qa_id.duplicated().any() or frame.qa_key.duplicated().any():
        raise ValueError("The frozen finite QA inventory differs")
    missing = frame.question.isna() | frame.answer.isna()
    pending = frame[missing].copy()
    if pending[["raw_source_file_id", "raw_source_line"]].isna().any().any():
        raise ValueError("A missing full QA has no original raw pointer")
    sources = pd.read_csv(source / "sources.csv").set_index("source_file_id")

    def recover(group):
        sid, needed = group
        metadata = sources.loc[int(sid)]
        raw = within(ROOT, metadata.project_relative_path)
        if raw != Path(metadata.real_path) or not raw.is_file() or raw.stat().st_size != int(metadata.bytes):
            raise ValueError("An original frozen raw file differs from its source inventory")
        before = raw.stat()
        wanted = {int(row.raw_source_line): row for row in needed.itertuples()}
        if len(wanted) != len(needed):
            raise ValueError("Frozen QA pointers repeat an original raw line")
        stop = max(wanted)
        restored, bindings = [], []
        with gzip.open(raw, "rb") as stream:
            for number, line in enumerate(stream, 1):
                if number in wanted:
                    old = wanted[number]
                    if not line.endswith(b"\n"):
                        raise ValueError("The original pointed raw line is incomplete")
                    row = json.loads(line)
                    question, answer = row["sample"]["question"], row["text"]
                    if row["model"] != old.raw_model or qah(question, answer) != old.qa_key:
                        raise ValueError("An original full QA differs from its frozen hash or model")
                    restored.append((old.Index, question, answer))
                    bindings.append({"qa_id": old.qa_id, "qa_key": old.qa_key,
                                     "raw_source_file_id": int(sid), "raw_source_line": number,
                                     "raw_path": str(raw.relative_to(ROOT)),
                                     "raw_line_sha256": hashlib.sha256(line).hexdigest()})
                if number >= stop:
                    break
        after = raw.stat()
        if len(restored) != len(needed) or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("A frozen pointer is absent or the original file changed during reading")
        proof = {"source_file_id": int(sid), "path": str(raw.relative_to(ROOT)),
                 "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                 "resolved_QA": len(restored), "highest_requested_line": stop}
        return restored, bindings, proof

    groups = list(pending.groupby("raw_source_file_id"))
    all_bindings, raw_proofs = [], []
    with ThreadPoolExecutor(max_workers=min(5, len(groups))) as executor:
        for restored, bindings, proof in executor.map(recover, groups):
            for index, question, answer in restored:
                frame.loc[index, ["question", "answer"]] = [question, answer]
            all_bindings.extend(bindings); raw_proofs.append(proof)
    if frame[["question", "answer"]].isna().any().any():
        raise ValueError("The finite full QA restoration is unresolved")
    if any(qah(row.question, row.answer) != row.qa_key for row in frame.itertuples()):
        raise ValueError("A restored or originally present QA differs from its frozen hash")
    original = pd.read_parquet(path)
    pd.testing.assert_frame_equal(original.drop(columns=["question", "answer"]),
                                  frame.drop(columns=["question", "answer"]))
    pd.testing.assert_frame_equal(original.loc[~missing], frame.loc[~missing])
    output.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(output / "recovered_qa.parquet", index=False)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(output / "recovered_qa.parquet"))
    pd.DataFrame(all_bindings).to_parquet(output / "recovered_QA_source_bindings.parquet", index=False)
    output_json(output / "receipt.json", {
        "schema": "kdm_exact_frozen_QA_raw_pointer_recovery_v1", "passed": True,
        "rows": len(frame), "previously_missing_full_QA": int(missing.sum()),
        "frozen_QA_path": str(path.relative_to(ROOT)), "frozen_QA_sha256": file_hash(path),
        "source_inventory_sha256": file_hash(source / "sources.csv"), "original_raw_files": raw_proofs,
        "full_original_QA_hashes_all_match": True, "original_metadata_and_present_QA_preserved": True,
        "new_generation": 0, "new_scoring_or_annotation": 0, "frozen_objects_modified": 0,
        "GPU_initialized": False, "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": file_hash(Path(__file__)), "actual_command": [sys.executable, *sys.argv],
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"passed": True, "rows": len(frame), "existing_QA_recovered": int(missing.sum()),
                      "original_raw_files": len(raw_proofs), "new_scoring_or_annotation": 0}))


if __name__ == "__main__":
    main()
