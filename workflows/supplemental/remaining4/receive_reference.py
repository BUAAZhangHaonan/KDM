#!/usr/bin/env python3
"""Receive only new immutable independent-reference parts from an explicit snapshot."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import subprocess

from receive_native import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--central-relative", required=True)
    parser.add_argument("--exclude-received-manifest", action="append", default=[], type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if (snapshot["schema"] != "kdm_reference_immutable_parts_snapshot_v1"
            or snapshot["model"] != "internvl35_8b" or snapshot["dataset"] != "food101"
            or snapshot["stage"] != "independent" or snapshot["source_host"] != "d4030"
            or snapshot["active_raw_files_read"] or snapshot["registered_expected_rows"] != 48480):
        raise ValueError("An explicit registered InternVL Food independent sealed snapshot is required")
    central = PurePosixPath(args.central_relative)
    if central.is_absolute() or ".." in central.parts:
        raise ValueError("Received directory must stay project-relative")
    previous, previous_sources = {}, []
    for path in args.exclude_received_manifest:
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior["schema"] != "kdm_remaining4_reference_received_manifest_v1" or not prior["source_files_sha_verified"]:
            raise ValueError("Previous reference received manifest lacks actual byte verification")
        for item in prior["parts"]:
            key = item["source_host"], item["claim_id"], item["part"]
            if key in previous:
                raise ValueError("Previous reference manifests repeat a part")
            previous[key] = item
        previous_sources.append({"path": str(path.resolve()), "sha256": digest(path), "rows": prior["received_rows"]})
    aliases = {"owner.json": "owner", "keys.jsonl": "keys", "admission.json": "admission"}
    selected, excluded, seen = [], [], set()
    for item in snapshot["parts"]:
        key = item["source_host"], item["claim_id"], item["part"]
        if key in seen or not item["generation_complete"]:
            raise ValueError("Snapshot repeats a part or includes unsealed raw")
        seen.add(key)
        if key in previous:
            prior = previous[key]
            if any(item[field] != prior[field] for field in ("model", "dataset", "stage", "rows")):
                raise ValueError("Previously received reference source declaration changed")
            for file in item["files"]:
                old = prior["files"][aliases.get(file["kind"], file["kind"])]
                if (old["remote_path"], old["sha256"]) != (file["path"], file["sha256"]):
                    raise ValueError("Previously received immutable reference member changed")
            excluded.append({"source_host": key[0], "claim_id": key[1], "part": key[2], "rows": item["rows"]})
        else:
            selected.append(item)
    if len(seen) != snapshot["parts_count"] or sum(part["rows"] for part in snapshot["parts"]) != snapshot["immutable_rows"] or not selected:
        raise ValueError("Snapshot sealed coverage differs or has no new reference parts")
    target = args.directory.resolve()
    target.mkdir(parents=True, exist_ok=False)
    groups = {}
    for item in selected:
        if (item["model"] != "internvl35_8b" or item["stage"] != "independent" or item["dataset"] != "food101"
                or item["source_host"] != "d4030" or item["source_root"] != snapshot["source_root"]):
            raise ValueError("Part leaves the explicitly authorized reference source")
        group = groups.setdefault(item["claim_id"], {})
        for file in item["files"]:
            remote = PurePosixPath(file["path"])
            remote.relative_to(PurePosixPath(item["source_root"]))
            if not remote.is_absolute() or ".." in remote.parts:
                raise ValueError("A source member leaves its explicit project root")
            if remote.name in group and group[remote.name] != file:
                raise ValueError("Source file basenames collide within a reference claim")
            group[remote.name] = file

    def transport(item):
        claim, files = item
        folder = target / claim
        folder.mkdir(exist_ok=False)
        subprocess.run(["scp", "-o", "BatchMode=yes", *["d4030:" + entry["path"] for entry in files.values()], str(folder)], check=True)
        for name, entry in files.items():
            if digest(folder / name) != entry["sha256"]:
                raise ValueError("Actual received reference bytes differ: " + str(folder / name))
        return {"claim_id": claim, "source_host": "d4030", "files": len(files),
                "bytes": sum((folder / name).stat().st_size for name in files)}

    with ThreadPoolExecutor(max_workers=2) as executor:
        transported = list(executor.map(transport, groups.items()))
    parts = []
    for item in selected:
        files = {}
        for file in item["files"]:
            kind = aliases.get(file["kind"], file["kind"])
            if kind in files:
                raise ValueError("A source part repeats a proof kind")
            files[kind] = {"remote_path": file["path"], "sha256": file["sha256"],
                          "received_path": str(central / item["claim_id"] / PurePosixPath(file["path"]).name)}
        if set(files) != {"raw", "identity", "complete", "owner", "keys", "admission"}:
            raise ValueError("Reference part proof members differ")
        parts.append({**{field: item[field] for field in ("model", "dataset", "stage", "source_host", "source_root", "claim_id", "part", "rows")},
            "files": files, "original_relative_raw": str(PurePosixPath(item["raw_path"]).relative_to(PurePosixPath(item["source_root"]))),
            "original_relative_receipt": str(PurePosixPath(item["receipt_path"]).relative_to(PurePosixPath(item["source_root"]))),
            "source_snapshot_finished_utc": item["finished_utc"], "source_identity": item["identity"]})
    manifest = {"schema": "kdm_remaining4_reference_received_manifest_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "parts": parts, "received_rows": sum(item["rows"] for item in parts), "source_files_sha_verified": True,
        "snapshot_path": str(args.snapshot.resolve()), "snapshot_sha256": digest(args.snapshot),
        "snapshot_captured_utc": snapshot["captured_utc"], "transported_claims": transported,
        "previous_received_manifests": previous_sources, "excluded_previous_parts": excluded,
        "receiver_sha256": digest(Path(__file__)), "active_raw_files_read": False, "claims_changed": False,
        "previous_raw_files_read": False, "GPU_queries": 0, "CPU_row_validation": "pending_finite_reference_score"}
    with (target / "received_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"parts": len(parts), "rows": manifest["received_rows"], "transported": transported}, indent=2), flush=True)


if __name__ == "__main__":
    main()
