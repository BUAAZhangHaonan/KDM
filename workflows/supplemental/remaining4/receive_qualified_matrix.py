"""Receive only existing matrix rows approved by the finite scientific-tuple audit."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from kdm.io import file_hash, within
from workflows.supplemental.remaining4 import collect_source_key_views as collector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--source-host", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit, output = (within(ROOT, value) for value in (args.audit_dir, args.output))
    output.relative_to(ROOT / "outputs/supplemental/remaining4")
    summary_path = audit / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary["schema"] != "kdm_matrix_scientific_tuple_intersection_v1"
            or summary["host"] != args.source_host or summary["unresolved_reasons"]
            or not all(summary[field] for field in ("full_decode_config_verified",
                "seed_prompt_scientific_sample_verified", "checkpoint_processor_and_adapter_verified"))):
        raise ValueError("The actual matrix audit is not qualified or belongs to another host")
    inventory = json.loads((audit / "original_actual_part_source_manifest.json").read_text())
    if inventory["schema"] != "kdm_original_registered_formal_completed_parts_v1" or inventory["originals_modified"]:
        raise ValueError("The original finite part source inventory differs")
    parts = {entry["raw_path"]: entry for entry in inventory["sources"]}
    requested = defaultdict(dict)
    original_keys, scientific_keys = set(), set()
    tuple_path = audit / "reusable_scientific_tuple_bindings.jsonl.gz"
    with gzip.open(tuple_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key, science = row["original_key"], row["scientific_tuple_sha256"]
            if (row["source_host"] != args.source_host or key in original_keys or science in scientific_keys
                    or row["source_line"] in requested[row["source_raw"]]):
                raise ValueError("Qualified matrix rows repeat original keys, science tuples or source lines")
            original_keys.add(key); scientific_keys.add(science)
            requested[row["source_raw"]][row["source_line"]] = row
    if len(original_keys) != summary["unique_reusable_bindings"]:
        raise ValueError("Qualified tuple counts differ from their actual audit")
    proof = output.with_name(output.name + "_input_proof")
    proof.mkdir(parents=True, exist_ok=False)
    selected_parts, bindings = [], []
    for raw_path, needed in requested.items():
        entry = parts[raw_path]
        if (entry["host"] != args.source_host or Path(entry["root"]) != ROOT
                or not entry["verified_actual_completed_part"] or entry["scientific_identity_and_decode_errors"]):
            raise ValueError("A selected old matrix source lacks its original completed-part proof")
        raw, sidecar, complete = [within(ROOT, entry[field])
                                  for field in ("raw_path", "identity_path", "receipt_path")]
        for name, path in (("raw", raw), ("identity", sidecar), ("receipt", complete)):
            if file_hash(path) != entry[name + "_sha256"]:
                raise ValueError("An audited immutable matrix source changed")
        identity = json.loads(sidecar.read_text())["identity"]
        with raw.open("rb") as stream:
            for number, line in enumerate(stream, 1):
                if number not in needed:
                    continue
                approved, row = needed[number], json.loads(line)
                condition = approved["condition"]
                if (not line.endswith(b"\n") or row["key"] != approved["original_key"]
                        or row["identity"] != identity or row["seed"] != condition["seed"]
                        or row["config"] != condition["config"]
                        or row["sample"]["id"] != condition["sample_id"]
                        or any(row[field] != condition[field] for field in collector.FIELDS)
                        or row["prompt"] != condition["prompt"]
                        or row["reference_prompt"] != condition["reference_prompt"]):
                    raise ValueError("An original matrix row differs from its qualified full scientific tuple")
                bindings.append({**{name: row[name] for name in collector.FIELDS},
                    "source_host": args.source_host, "source_raw": str(raw), "source_line": number,
                    "key": row["key"], "row_identity": row["identity"], "seed": row["seed"],
                    "sample_id": row["sample"]["id"]})
        selected_parts.append({"source_host": args.source_host, "source_raw": str(raw),
            "source_identity_sidecar": str(sidecar), "source_part_receipt": str(complete),
            "model": entry["model"], "source_claim": entry["claim_id"],
            "raw_sha256_from_receipt": entry["raw_sha256"],
            "identity_sha256_from_receipt": entry["identity_sha256"],
            "source_lines": sorted(needed), "selected_key_count": len(needed)})
    if len(bindings) != len(original_keys):
        raise ValueError("The actual qualified source line coverage differs")
    manifest = proof / "selected_parts.json"
    manifest.write_text(json.dumps({"schema": "kdm_qualified_existing_matrix_receive_plan_v1",
        "parts": selected_parts, "audit_source": str(audit), "audit_summary_sha256": file_hash(summary_path),
        "qualified_tuples_sha256": file_hash(tuple_path), "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    path = proof / "selected_original_bindings.jsonl.gz"
    with gzip.open(path, "xt", encoding="utf-8") as stream:
        for binding in bindings:
            stream.write(json.dumps(binding, allow_nan=False) + "\n")
    original_argv = sys.argv
    try:
        sys.argv = [collector.__file__, "--manifest", str(manifest), "--bindings", str(path),
                    "--source-host", args.source_host, "--output", str(output)]
        collector.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
