"""Derive a compact SID receipt; keep the full original evidence unchanged."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(source, target):
    source, target = Path(source), Path(target)
    raw = source.read_bytes()
    result = json.loads(raw)
    result["detail_evidence"] = {
        "path": str(source), "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "scope": "Original complete proof remains on disk. This derived receipt omits repeated selected and blocked index arrays; it does not rerun or alter comparisons."
    }
    for visit in result.get("evidence", []):
        for side in ("actual", "official"):
            events = visit.pop(side, None)
            if events is None:
                continue
            visit[side + "_event_count"] = len(events)
            visit[side + "_layers"] = sorted({event["layer"] for event in events})
            for event in events:
                for name in ("selected", "mask_last_row_blocked"):
                    if name in event:
                        event[name + "_count"] = len(event.pop(name))
            visit[side + "_events_without_index_arrays"] = events
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if target.exists():
        if target.read_text() != payload:
            raise ValueError("Existing receipt differs; choose a new output path")
    else:
        with target.open("x") as stream:
            stream.write(payload)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()
    summarize(args.source, args.target)
