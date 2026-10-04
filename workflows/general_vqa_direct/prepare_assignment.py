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


def partition_keys(keys, count):
    if count < 1 or count > len(keys) or len(keys) != len(set(keys)):
        raise ValueError("Partitions require distinct keys and a nonempty shard for each worker")
    shards = [keys[index::count] for index in range(count)]
    flat = [key for shard in shards for key in shard]
    assert len(flat) == len(set(flat)) == len(keys) and set(flat) == set(keys)
    return shards


def claim_state(plan, complete):
    occupied, sources, released_at = set(), {}, {}
    for path in (plan["output"] / "claims").glob("*/owner.json"):
        owner = json.loads(path.read_text())
        assert owner["identity"] == plan["identity"], "Source claim identity differs"
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
    ledger = plan["output"] / "completed_keys.jsonl"
    if ledger.exists():
        sources[str(ledger.relative_to(ROOT))] = generate.file_hash(ledger)
    return occupied, sources, released_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", default="outputs/general_vqa_direct/run_20261004")
    parser.add_argument("--out", default="outputs/general_vqa_direct/assignments")
    parser.add_argument("--native-source", action="store_true",
                        help="Read the gated native engine and exclude its original engine's completed or claimed keys")
    parser.add_argument("--reference-output", default="outputs/general_vqa_direct/run_20261004")
    parser.add_argument("--partitions", type=int, default=1,
                        help="Partition all verified remaining keys into balanced disjoint worker assignments")
    parser.add_argument("--supersede-assignment", action="append", default=[],
                        help="Prior root receipt superseded only for keys actually released after its creation")
    args = parser.parse_args()
    options = SimpleNamespace(batch_size=1, claim_id="assignment", owner="/root",
        model=args.model, protocol="data/general_vqa_direct_20261004/frozen/protocol.json",
        datasets=args.datasets, sample_ids=None, output=args.source)
    if args.native_source:
        from workflows.general_vqa_direct import native_fast
        options.reference_output = args.reference_output
        plan, original = native_fast.plans(options)
        plans = [plan, original]
    else:
        plan = generate.load_plan(options)
        plans = [plan]
    complete, occupied, sources, released_at = set(), set(), {}, {}
    for source in plans:
        done = generate.completed_keys(source)
        used, bindings, releases = claim_state(source, done)
        complete.update(done)
        occupied.update(used)
        sources.update(bindings)
        for key, stamp in releases.items():
            released_at[key] = max(released_at.get(key, ""), stamp)
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
    shards = partition_keys(selected, args.partitions)
    targets = [(folder / f"{args.model}_{args.name}{'' if len(shards) == 1 else '_part'+str(i+1)}.ids.jsonl",
                folder / f"{args.model}_{args.name}{'' if len(shards) == 1 else '_part'+str(i+1)}.assignment.json")
               for i in range(len(shards))]
    assert all(not p.exists() for pair in targets for p in pair)
    for index, (keys, (target, receipt)) in enumerate(zip(shards, targets)):
        target.write_text("".join(json.dumps({"id": plan["tasks"][key]["sample"]["id"]}) + "\n" for key in keys))
        record = {"model": args.model, "datasets": args.datasets, "owner": "/root",
            "created_utc": datetime.now(timezone.utc).isoformat(), "keys": keys,
            "rows": len(keys), "source_output": args.source, "source_identity": plan["identity"],
            "source_bindings": sources, "ids": str(target.relative_to(ROOT)), "ids_sha256": generate.file_hash(target),
            "intersection_generated": 0, "intersection_unreleased_claim": 0,
            "intersection_active_prior_assignments": 0, "superseded_released_assignments": superseded,
            "partition_index": index + 1, "partitions": len(shards),
            "partition_union_rows": len(selected), "partition_union_sha256": generate.stable_hash(sorted(selected)),
            "partition_overlap": 0, "partition_missing": 0,
            "source_identities": [source["identity"] for source in plans]}
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k not in ("keys", "source_bindings")}))


if __name__ == "__main__":
    main()
