"""Copy explicitly indexed original rows; preserve identity and complete receipts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from kdm.io import file_hash, stable_hash, within

FIELDS = ("model", "method", "kind", "marker", "reference_marker", "guided", "reference_guided", "replicate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path, bindings_path, output = (within(ROOT, p) for p in (args.manifest, args.bindings, args.output))
    output.relative_to(ROOT / "outputs/supplemental/remaining4")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    indexed = defaultdict(dict)
    seen = set()
    with gzip.open(bindings_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["source_host"] != args.source_host:
                continue
            if row["key"] in seen or row["source_line"] in indexed[row["source_raw"]]:
                raise ValueError("The explicit original source key index repeats a key or line")
            seen.add(row["key"])
            indexed[row["source_raw"]][row["source_line"]] = row
    parts = [p for p in manifest["parts"] if p["source_host"] == args.source_host]
    if not parts or set(indexed) != {p["source_raw"] for p in parts}:
        raise ValueError("The finite original part index differs from the receive manifest")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), output / "source_code_snapshot.py")
    records, original_members, total = [], [], 0
    for number, part in enumerate(parts):
        original_raw = within(ROOT, str(Path(part["source_raw"]).relative_to(ROOT)))
        original_identity = within(ROOT, str(Path(part["source_identity_sidecar"]).relative_to(ROOT)))
        original_receipt = within(ROOT, str(Path(part["source_part_receipt"]).relative_to(ROOT)))
        receipt = json.loads(original_receipt.read_text(encoding="utf-8-sig"))
        identity = json.loads(original_identity.read_text(encoding="utf-8-sig"))
        actual_raw_sha = file_hash(original_raw)
        if (receipt["generation_complete"] is not True or receipt["model"] != part["model"]
                or actual_raw_sha != receipt["raw_sha256"]
                or receipt["raw_sha256"] != part["raw_sha256_from_receipt"]):
            raise ValueError("An original immutable raw part differs from its generation receipt")
        if (file_hash(original_identity) != receipt["identity_sha256"]
                or receipt["identity_sha256"] != part["identity_sha256_from_receipt"]
                or identity["identity"] != stable_hash(identity["definition"])
                or identity["definition"]["model"] != part["model"]):
            raise ValueError("The original model identity or registered source fingerprint differs")
        requested = indexed[part["source_raw"]]
        if (sorted(requested) != sorted(part["source_lines"])
                or len(part["source_lines"]) != len(set(part["source_lines"]))
                or len(requested) != part["selected_key_count"]):
            raise ValueError("The selected original line ownership differs")
        view = output / ("part_" + str(number).zfill(4) + ".jsonl.gz")
        copied_identity = view.with_suffix(".identity.json")
        copied_receipt = output / ("part_" + str(number).zfill(4) + ".original_complete.json")
        shutil.copyfile(original_identity, copied_identity)
        shutil.copyfile(original_receipt, copied_receipt)
        count, visited = 0, 0
        with original_raw.open("rb") as source, gzip.open(view, "xb", compresslevel=1) as target:
            for source_line, raw_line in enumerate(source, 1):
                visited += 1
                if not raw_line.endswith(b"\n"):
                    raise ValueError("A purported original complete part contains an incomplete line")
                if source_line not in requested:
                    continue
                actual = json.loads(raw_line)
                json.dumps(actual, allow_nan=False)
                binding = requested[source_line]
                if (actual["key"] != binding["key"] or actual["identity"] != binding["row_identity"]
                        or actual["identity"] != identity["identity"] or actual["status"] != "ok"
                        or actual["sample"]["id"] != binding["sample_id"]
                        or actual["seed"] != binding["seed"]
                        or any(actual[field] != binding[field] for field in FIELDS)):
                    raise ValueError("A selected original row differs from its exact registered source binding")
                target.write(raw_line)
                count += 1
                original_members.append({"key": actual["key"], "received_path": str(view.relative_to(ROOT)),
                    "received_line": count, "source_host": args.source_host,
                    "source_original_raw": str(original_raw), "source_original_line": source_line,
                    "raw_line_sha256": hashlib.sha256(raw_line).hexdigest(),
                    "source_original_raw_sha256": receipt["raw_sha256"],
                    "source_original_receipt": str(original_receipt),
                    "source_original_receipt_sha256": file_hash(copied_receipt),
                    "source_identity_sha256": receipt["identity_sha256"]})
        if visited != receipt["rows"] or count != len(requested):
            raise ValueError("Original complete count or actual selected-view coverage differs")
        total += count
        records.append({"model": part["model"], "dataset": "food101", "stage": "formal",
            "claim_id": part["source_claim"], "part": receipt["part"], "rows": count,
            "original_complete_part_rows": visited, "derived_selected_view": True,
            "source_part_generation_complete": True, "received_raw_path": str(view.relative_to(ROOT)),
            "received_identity_path": str(copied_identity.relative_to(ROOT)),
            "files": [{"received_path": str(p.relative_to(ROOT)), "sha256": file_hash(p)}
                      for p in (view, copied_identity, copied_receipt)]})
    if total != len(seen):
        raise ValueError("The actually written original selected view loses or repeats task keys")
    with gzip.open(output / "original_row_bindings.jsonl.gz", "xt", encoding="utf-8", compresslevel=1) as stream:
        for row in original_members:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    proof = {"schema": "kdm_selected4_registered_received_manifest_v1", "rows": total, "parts": records,
        "passed": True, "local_source_sha_verified": True, "active_raw_opened": 0,
        "original_sources_preserved": True, "selection_only": True, "new_generation": 0,
        "source_host": args.source_host, "actual_hostname": socket.gethostname(),
        "input_manifest": str(manifest_path), "input_manifest_sha256": file_hash(manifest_path),
        "input_bindings": str(bindings_path), "input_bindings_sha256": file_hash(bindings_path),
        "original_row_bindings_sha256": file_hash(output / "original_row_bindings.jsonl.gz"),
        "source_entry_sha256": file_hash(Path(__file__)), "actual_command": [sys.executable, *sys.argv],
        "completed_utc": datetime.now(timezone.utc).isoformat(), "GPU_initialized": False}
    (output / "manifest.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "actual_original_rows_copied": total,
                      "parts": len(records), "new_generation": 0}))


if __name__ == "__main__":
    main()
