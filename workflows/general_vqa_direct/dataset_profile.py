"""Inspect frozen question/answer schemas without loading a model."""
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
rows = [json.loads(line) for line in (ROOT / "data/general_vqa_direct_20261004/frozen/manifest_hallusionbench.jsonl").read_text().splitlines()]
print(json.dumps({"n": len(rows), "gold_type_counts": dict(Counter(type(r["gold"]).__name__ for r in rows)),
                  "gold_values": dict(Counter(str(r["gold"]) for r in rows))}))
for row in rows:
    if re.search(r"\b(?:which|what|who|how|where|when|name)\b", row["question"], re.I):
        print(json.dumps({k: row.get(k) for k in ("id", "question", "gold", "gt_answer", "gt_answer_details", "sample_note", "source_row")}, ensure_ascii=False))
