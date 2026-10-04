"""Make a bounded, non-overlapping actual semantic-review assignment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--max-rows", type=int, default=150)
    parser.add_argument("--input-token-estimate", type=int, default=16000)
    args = parser.parse_args()
    claimed = set()
    for path in args.annotations.glob("*/pending_review.jsonl"):
        claimed.update(row["qa_key"] for row in load(path))
    for path in args.annotations.glob("*/decisions.jsonl"):
        claimed.update(row["qa_key"] for row in load(path))
    selected, estimate = [], 0
    for row in load(args.pending):
        if row["qa_key"] in claimed:
            continue
        item = {name: row[name] for name in ("qa_key", "dataset", "question", "answer", "options")}
        item["membership_count"] = len(row["memberships"])
        # Conservative character estimate plus question/key/schema overhead.
        cost = (len(json.dumps(item, ensure_ascii=False)) + 2) // 3 + 50
        if selected and (len(selected) >= args.max_rows or estimate + cost > args.input_token_estimate):
            break
        selected.append(item)
        estimate += cost
    if not selected:
        print(json.dumps({"assigned": 0, "remaining_unclaimed": 0}))
        return
    out = args.annotations / args.batch_id
    out.mkdir(parents=True, exist_ok=False)
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode("utf-8")
    (out / "pending_review.jsonl").write_bytes(raw)
    record = {"batch_id": args.batch_id, "owner": args.owner,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "assigned": len(selected), "estimated_input_tokens": estimate,
              "source_pending": str(args.pending), "input_sha256": hashlib.sha256(raw).hexdigest(),
              "annotation_model": "gpt-5.6-luna", "effort": "medium",
              "status": "assigned_not_annotated", "keys": [row["qa_key"] for row in selected]}
    (out / "assignment.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items() if key != "keys"}))


if __name__ == "__main__":
    main()
