"""Assign only verified never-claimed or sealed-and-released Direct keys."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from workflows.general_vqa_direct import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", default="outputs/general_vqa_direct/run_20261004")
    parser.add_argument("--out", default="outputs/general_vqa_direct/assignments")
    parser.add_argument("--supersede-assignment", action="append", default=[],
                        help="Prior root receipt superseded only for keys actually released after its creation")
    args = parser.parse_args()
    options = SimpleNamespace(batch_size=1, claim_id="assignment", owner="/root",
        model=args.model, protocol="data/general_vqa_direct_20261004/frozen/protocol.json",
        datasets=args.datasets, sample_ids=None, output=args.source)
    plan = generate.load_plan(options)
    complete = generate.completed_keys(plan)
    occupied, sources, released_at = set(), {}, {}
    for path in (plan["output"] / "claims").glob("*/owner.json"):
        owner = json.loads(path.read_text())
        sources[str(path.relative_to(ROOT))] = generate.file_hash(path)
        released = set()
        release_path = path.parent / "released.json"
        if release_path.exists():
            release = json.loads(release_path.read_text())
            assert release["claim_identity"] == generate.stable_hash(owner)
            assert release["reason"] == "stop_after_completed_chunk"
            released = set(release["keys"])
            assert released <= set(owner["keys"])
            proc = Path(f"/proc/{owner['pid']}/stat")
            assert not proc.exists() or int(proc.read_text().rsplit(")", 1)[1].split()[19]) != owner["start_tick"], "Source is still running"
            for key in released:
                released_at[key] = max(released_at.get(key, ""), release["released_utc"])
            sources[str(release_path.relative_to(ROOT))] = generate.file_hash(release_path)
        occupied.update(set(owner["keys"]) - released - complete)
    selected = [key for key in plan["selected"] if key not in complete and key not in occupied]
    assert selected, "No unclaimed or released keys"
    folder = generate.relative(ROOT, args.out)
    folder.mkdir(parents=True, exist_ok=True)
    allowed = {generate.relative(ROOT, name).resolve() for name in args.supersede_assignment}
    assert all(path.parent == folder.resolve() and path.is_file() for path in allowed)
    superseded = []
    for path in folder.glob("*.assignment.json"):
        old = json.loads(path.read_text())
        if old["model"] == args.model:
            overlap = set(selected) & set(old["keys"])
            if overlap:
                assert path.resolve() in allowed, "Assignment overlaps a previous root dispatch"
                assert all(key in released_at and datetime.fromisoformat(released_at[key]) > datetime.fromisoformat(old["created_utc"]) for key in overlap), "Supersession lacks a later actual source release"
                superseded.append({"receipt": str(path.relative_to(ROOT)), "sha256": generate.file_hash(path), "released_overlap_rows": len(overlap)})
    assert {str(path.relative_to(ROOT)) for path in allowed} == {item["receipt"] for item in superseded}, "Unused supersession request"
    target = folder / f"{args.model}_{args.name}.ids.jsonl"
    receipt = folder / f"{args.model}_{args.name}.assignment.json"
    assert not target.exists() and not receipt.exists()
    target.write_text("".join(json.dumps({"id": plan["tasks"][key]["sample"]["id"]}) + "\n" for key in selected))
    record = {"model": args.model, "datasets": args.datasets, "owner": "/root",
        "created_utc": datetime.now(timezone.utc).isoformat(), "keys": selected,
        "rows": len(selected), "source_output": args.source, "source_identity": plan["identity"],
        "source_bindings": sources, "ids": str(target.relative_to(ROOT)), "ids_sha256": generate.file_hash(target),
        "intersection_generated": 0, "intersection_unreleased_claim": 0,
        "intersection_active_prior_assignments": 0, "superseded_released_assignments": superseded}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("keys", "source_bindings")}))


if __name__ == "__main__":
    main()
