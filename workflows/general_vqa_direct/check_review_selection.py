"""Regressions for full-batch corrections and concurrent review dispatch."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from score import check_selected_batch, has_review_completion, write_rows


def main():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for field in ("qa_key", "quality_key"):
            batch = root / field
            batch.mkdir()
            write_rows(batch / "pending_review.jsonl", [{field: "one"}, {field: "two"}])
            selected = batch / "selected.jsonl"
            for rejected in ([{field: "one"}], [{field: "one"}, {field: "one"}],
                             [{field: "one"}, {field: "other"}]):
                write_rows(selected, rejected)
                try:
                    check_selected_batch(selected, field)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Partial, duplicate or foreign selection was accepted")
            write_rows(selected, [{field: "two"}, {field: "one"}])
            check_selected_batch(selected, field)
            assert not has_review_completion(selected)
            for name in ("ACTIVE.json", "write_receipt.json", "quality_receipt.json",
                         "receipt.json", "root_validation_receipt.json"):
                receipt = batch / name
                receipt.write_text('{}')
                assert has_review_completion(selected)
                receipt.unlink()
        annotations = root / "annotations"
        live = annotations / "live"
        live.mkdir(parents=True)
        write_rows(live / "pending_review.jsonl", [{"qa_key": "reserved"}])
        (live / "decisions.jsonl").write_text('{"unfinished":')
        assert not has_review_completion(live / "decisions.jsonl")
        pending = root / "pending.jsonl"
        record = {"dataset": "pope", "question": "Is there a dog?", "answer": "No",
                  "options": [], "memberships": [{}]}
        write_rows(pending, [dict(record, qa_key="reserved"), dict(record, qa_key="new")])
        subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_review.py")),
            "--pending", str(pending), "--annotations", str(annotations),
            "--batch-id", "next", "--owner", "/root/test"], check=True, capture_output=True)
        assigned = [json.loads(line) for line in (annotations / "next/pending_review.jsonl").read_text().splitlines()]
        assert [row["qa_key"] for row in assigned] == ["new"]
    print("passed: complete selected batches, completion evidence, and immutable live-claim exclusion")


if __name__ == "__main__":
    main()
