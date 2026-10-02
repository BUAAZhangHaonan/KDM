"""Receive completed registered parts from an explicit producer inventory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from kdm.io import file_hash, within
from workflows.supplemental.remaining4 import collect_source_key_views as collector

FIELDS = collector.FIELDS
SCHEMAS = {"kdm_immutable_registered_food_main_and_matrix_source_manifest_v1",
           "kdm_actual_sealed_registered_source_manifest_v1",
           "kdm_actual_completed_registered_sources_delta_v1"}
METHODS = {"instruction_vcd", "instruction_m3id", "cda_visual"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--registered-mechanism-only", action="store_true")
    args = parser.parse_args()
    schemas = {"kdm_registered_matrix_immutable_delta_v1"} if args.registered_mechanism_only else SCHEMAS
    methods = {"vcd", "m3id"} if args.registered_mechanism_only else METHODS
    output = within(ROOT, args.output)
    output.relative_to(ROOT / "outputs/supplemental/remaining4")
    plan_root = output.with_name(output.name + "_input_proof")
    plan_root.mkdir(parents=True, exist_ok=False)
    parts, bindings, inputs, seen_paths, seen_keys = [], [], [], set(), set()
    for relative in args.manifest:
        path = within(ROOT, relative)
        inventory = json.loads(path.read_text(encoding="utf-8-sig"))
        if inventory["schema"] not in schemas:
            raise ValueError("An explicit immutable producer inventory is required")
        inputs.append({"path": str(path), "sha256": file_hash(path)})
        for source in inventory["sources"]:
            if source["host"] != args.source_host:
                continue
            if not set(source["methods"]).issubset(methods):
                continue
            if source["immutable"] is not True or Path(source["root"]) != ROOT:
                raise ValueError("The declared host/root or immutability differs")
            raw, sidecar, receipt_path = [
                within(ROOT, source[name])
                for name in ("raw_path", "identity_path", "receipt_path")
            ]
            if raw in seen_paths:
                raise ValueError("Producer inventories repeat a completed part")
            seen_paths.add(raw)
            for name, member in (("raw", raw), ("identity", sidecar), ("receipt", receipt_path)):
                if file_hash(member) != source[name + "_sha256"]:
                    raise ValueError("An immutable producer member changed")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
            index = source.get("sealed_receipt_part_index")
            if index is None:
                completion = receipt
            else:
                if (source.get("source_proof_kind") != "actual_sealed_partial_prefix"
                        or receipt["schema"] != "kdm_selected4_registered_sealed_prefix_v1"
                        or receipt["whole_claim_complete"] is not False):
                    raise ValueError("A partial prefix lacks an actual sealing receipt")
                entry = receipt["parts"][index]
                if within(ROOT, entry["path"]) != raw:
                    raise ValueError("The sealed part pointer differs from the source")
                completion = entry["completed_prefix_validation"]
            if (completion["generation_complete"] is not True
                    or completion["rows"] != source["count"]
                    or completion["raw_sha256"] != source["raw_sha256"]
                    or completion["identity_sha256"] != source["identity_sha256"]):
                raise ValueError("The actual part completion count differs")
            lines = []
            with raw.open("rb") as stream:
                for number, line in enumerate(stream, 1):
                    if not line.endswith(b"\n"):
                        raise ValueError("A completed producer part has a partial line")
                    row = json.loads(line)
                    if row["key"] in seen_keys or row["method"] not in methods:
                        raise ValueError("A part repeats a key or contains a method outside the explicit receive scope")
                    seen_keys.add(row["key"])
                    lines.append(number)
                    bindings.append({
                        **{name: row[name] for name in FIELDS},
                        "source_host": args.source_host, "source_raw": str(raw),
                        "source_line": number, "key": row["key"],
                        "row_identity": row["identity"], "seed": row["seed"],
                        "sample_id": row["sample"]["id"],
                    })
            if len(lines) != source["count"]:
                raise ValueError("The actual immutable part has a different row count")
            parts.append({
                "source_host": args.source_host, "source_raw": str(raw),
                "source_identity_sidecar": str(sidecar),
                "source_part_receipt": str(receipt_path), "model": source["model"],
                "source_claim": source["claim_id"],
                "sealed_receipt_part_index": index,
                "raw_sha256_from_receipt": source["raw_sha256"],
                "identity_sha256_from_receipt": source["identity_sha256"],
                "source_lines": lines, "selected_key_count": len(lines),
            })
    if not parts:
        raise ValueError("This host has no completed registered main parts in the inventory")
    manifest = plan_root / "selected_parts.json"
    manifest.write_text(json.dumps({
        "schema": "kdm_explicit_completed_main_receive_plan_v1", "parts": parts,
        "original_producer_inventories": inputs,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    binding_path = plan_root / "selected_original_bindings.jsonl.gz"
    with gzip.open(binding_path, "xt", encoding="utf-8", compresslevel=1) as stream:
        for binding in bindings:
            stream.write(json.dumps(binding, allow_nan=False) + "\n")
    old_argv = sys.argv
    try:
        sys.argv = [collector.__file__, "--manifest", str(manifest), "--bindings",
                    str(binding_path), "--source-host", args.source_host,
                    "--output", str(output)]
        collector.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
