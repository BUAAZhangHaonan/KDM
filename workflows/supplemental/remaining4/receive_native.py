#!/usr/bin/env python3
"""Copy a finite explicit native snapshot and preserve its immutable source hashes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, action="append", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--central-relative", required=True)
    parser.add_argument("--exclude-received-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    target = args.directory.resolve()
    target.mkdir(parents=True, exist_ok=False)
    central_relative = PurePosixPath(args.central_relative)
    if central_relative.is_absolute() or ".." in central_relative.parts:
        raise ValueError("Central received directory must stay project-relative")
    source_parts, snapshot_records = [], []
    for path in args.snapshot:
        original = json.loads(path.read_text(encoding="utf-8"))
        if original["schema"] != "kdm_native_immutable_parts_snapshot_v1":
            raise ValueError("An explicit immutable native producer snapshot is required")
        source_parts.extend(original["parts"])
        snapshot_records.append({"path": str(path.resolve()), "sha256": digest(path),
            "captured_utc": original["captured_utc"], "source_host": original["host"]})
    aliases = {"owner.json": "owner", "keys.jsonl": "keys", "admission.json": "admission", "operator_audit.json": "operator_audit"}
    prior_parts, prior_sources = {}, []
    for path in args.exclude_received_manifest:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["schema"] != "kdm_remaining4_native_received_manifest_v1" or not manifest["source_files_sha_verified"]:
            raise ValueError("Previous immutable received manifest lacks source-byte verification")
        for item in manifest["parts"]:
            key = item["source_host"], item["claim_id"], item["part"]
            if key in prior_parts:
                raise ValueError("Repeated previous received part identity")
            prior_parts[key] = item
        prior_sources.append({"path": str(path.resolve()), "sha256": digest(path), "rows": manifest["received_rows"]})
    selected_parts, excluded_parts, selected_keys = [], [], set()
    for part in source_parts:
        key = part["source_host"], part["claim_id"], part["part"]
        if key in selected_keys:
            raise ValueError("Input snapshots repeat a native source part")
        selected_keys.add(key)
        if key in prior_parts:
            prior = prior_parts[key]
            if any(prior[field] != part[field] for field in ("model", "method", "rows")):
                raise ValueError("Previously received part declaration changed")
            for entry in part["files"]:
                kind = aliases.get(entry["kind"], entry["kind"])
                if (kind not in prior["files"] or prior["files"][kind]["sha256"] != entry["sha256"]
                        or prior["files"][kind]["remote_path"] != entry["path"]):
                    raise ValueError("Previously received immutable source changed")
            excluded_parts.append({"source_host": part["source_host"], "claim_id": part["claim_id"],
                                   "part": part["part"], "rows": part["rows"],
                                   "raw_sha256": prior["files"]["raw"]["sha256"]})
        else:
            selected_parts.append(part)
    source_parts = selected_parts
    if not source_parts:
        raise ValueError("Snapshots have no new sealed native parts")
    groups = {}
    for part in source_parts:
        if part["source_host"] not in {"d4030", "6403"} or not part["generation_complete"]:
            raise ValueError("Unknown source host or incomplete producer part")
        key = part["source_host"], part["claim_id"]
        folder = target / part["claim_id"]
        group = groups.setdefault(key, {"folder": folder, "files": {}})
        for entry in part["files"]:
            remote = PurePosixPath(entry["path"])
            remote.relative_to(PurePosixPath(part["source_root"]))
            if not remote.is_absolute() or ".." in remote.parts:
                raise ValueError("Source member leaves its explicit project root")
            previous = group["files"].get(remote.name)
            if previous and previous != entry:
                raise ValueError("Received source basenames collide inside an exclusive claim")
            group["files"][remote.name] = entry

    def receive(key_and_group):
        (host, claim), group = key_and_group
        folder = group["folder"]
        folder.mkdir(exist_ok=False)
        sources = [host + ":" + entry["path"] for entry in group["files"].values()]
        subprocess.run(["scp", "-o", "BatchMode=yes", *sources, str(folder)], check=True)
        total = 0
        for name, entry in group["files"].items():
            path = folder / name
            if digest(path) != entry["sha256"]:
                raise ValueError("Actual received source SHA differs: " + str(path))
            total += path.stat().st_size
        return {"source_host": host, "claim_id": claim, "files": len(group["files"]), "bytes": total}

    with ThreadPoolExecutor(max_workers=3) as pool:
        transported = list(pool.map(receive, groups.items()))
    parts = []
    for original in source_parts:
        files = {}
        for entry in original["files"]:
            kind = aliases.get(entry["kind"], entry["kind"])
            if kind in files:
                raise ValueError("A source part repeats a proof kind")
            relative = central_relative / original["claim_id"] / PurePosixPath(entry["path"]).name
            files[kind] = {"remote_path": entry["path"], "received_path": str(relative), "sha256": entry["sha256"]}
        parts.append({**{field: original[field] for field in ("model", "method", "source_host", "source_root", "claim_id", "part", "rows")},
            "files": files, "original_relative_raw": original["source_relative_raw"],
            "original_relative_receipt": original["source_relative_receipt"], "source_snapshot_finished_utc": original["finished_utc"]})
    manifest = {"schema": "kdm_remaining4_native_received_manifest_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "parts": parts, "snapshots": snapshot_records, "transported_claims": transported,
        "received_rows": sum(part["rows"] for part in parts), "source_files_sha_verified": True,
        "active_raw_files_read": False, "claims_changed": False, "GPU_queries": 0,
        "receiver_sha256": digest(Path(__file__)), "CPU_row_validation": "pending_central_score_entry"}
    manifest.update(previous_received_manifests=prior_sources, excluded_previous_parts=excluded_parts,
                    previous_raw_files_read=False, previous_raw_files_transported=False)
    with (target / "received_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"received_manifest": str(target / "received_manifest.json"),
        "parts": len(parts), "rows": manifest["received_rows"], "transported": transported}, indent=2), flush=True)


if __name__ == "__main__":
    main()
