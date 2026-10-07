#!/usr/bin/env python3
"""Read-only byte verification of the sealed cost artifacts after transport."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workflow = Path(__file__).resolve().parent
    output = (args.output or workflow.parents[1] / "outputs" / "inference_cost_a100_100_20261007").resolve()
    manifest_path = output / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["passed"] and manifest["actual_measured_replies"] == 2700
    assert manifest["independent_warmup_replies"] == 216
    paths = set()
    for item in manifest["files"]:
        path = (output / item["path"]).resolve()
        path.relative_to(output)
        assert item["path"] not in paths
        paths.add(item["path"])
        assert path.is_file() and path.stat().st_size == item["bytes"], item["path"]
        assert sha(path) == item["sha256"], item["path"]
    for name, expected in manifest["workflow_sources_sha256"].items():
        assert sha(workflow / name) == expected, name
    print(json.dumps(dict(passed=True, output=str(output), files=len(paths),
        measured=2700, warmups=216, workflow_sources=len(manifest["workflow_sources_sha256"]),
        manifest_sha256=sha(manifest_path))), flush=True)


if __name__ == "__main__":
    main()
