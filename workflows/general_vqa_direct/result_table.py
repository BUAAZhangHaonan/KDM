"""Report all 36 frozen cells; publish metrics only for complete resolved cells."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, values):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/general_vqa_direct_20261004/frozen/protocol.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    expected = {}
    for name, entry in protocol["datasets"].items():
        raw = (ROOT / entry["manifest"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["manifest_sha256"]
        ids = {json.loads(line)["id"] for line in raw.splitlines() if line.strip()}
        assert len(ids) == entry["expected_rows"]
        expected[name] = ids
    groups = defaultdict(list)
    seen = set()
    for row in read_rows(args.scores):
        assert row["model"] in protocol["models"] and row["sample_id"] in expected[row["dataset"]]
        key = (row["model"], row["dataset"], row["sample_id"])
        assert key not in seen, "Duplicate model/dataset/sample"
        seen.add(key)
        if row["score"] is not None:
            assert row["score"] in (0, 1)
        assert row["abstain"] is None or type(row["abstain"]) is bool
        groups[(row["model"], row["dataset"])].append(row)
    cells, gaps = [], []
    for model in protocol["models"]:
        for dataset in protocol["datasets"]:
            rows = groups[(model, dataset)]
            missing = expected[dataset] - {r["sample_id"] for r in rows}
            unresolved = sum(r["score"] is None or r["abstain"] is None for r in rows)
            scored = [r for r in rows if r["score"] is not None and r["abstain"] is not None]
            counts = Counter(r["label"] for r in scored)
            complete = not missing and unresolved == 0
            n = len(expected[dataset])
            correct = sum(r["score"] for r in scored)
            abstentions = sum(r["abstain"] for r in scored)
            cells.append({"model": model, "dataset": dataset, "expected": n,
                "generated": len(rows), "scored": len(scored), "pending_semantic": unresolved,
                "pending_behavior": sum(r["abstain"] is None for r in rows),
                "pending_quality": sum(r["score"] is None for r in rows),
                "missing_generation": len(missing), "correct": correct,
                "correct_answer": sum(r["score"] == 1 and not r["abstain"] for r in scored),
                "correct_abstention": sum(r["score"] == 1 and r["abstain"] for r in scored),
                "wrong_answer": sum(r["score"] == 0 and not r["abstain"] and r["label"] != "invalid" for r in scored),
                "abstentions": abstentions, "invalid": counts["invalid"],
                "truncated": sum(not r["terminated"] for r in rows),
                "accuracy": correct / n if complete else None,
                "abstention_rate": abstentions / n if complete else None,
                "status": "complete" if complete else "in_progress"})
            gaps.extend({"model": model, "dataset": dataset, "sample_id": sample_id} for sample_id in sorted(missing))
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "metrics_all_36.csv", cells)
    if gaps:
        write_csv(args.out / "missing_generation.csv", gaps)
    source_sha = hashlib.sha256(args.scores.read_bytes()).hexdigest()
    summary = {"updated_utc": datetime.now(timezone.utc).isoformat(), "expected": sum(c["expected"] for c in cells),
        "generated": sum(c["generated"] for c in cells), "scored": sum(c["scored"] for c in cells),
        "pending_semantic": sum(c["pending_semantic"] for c in cells),
        "missing_generation": len(gaps), "accepted_cells": sum(c["status"] == "complete" for c in cells),
        "total_cells": len(cells), "source_scores": str(args.scores), "source_scores_sha256": source_sha}
    (args.out / "CURRENT_STATE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
