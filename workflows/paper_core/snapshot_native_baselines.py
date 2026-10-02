#!/usr/bin/env python3
"""Seal complete native baseline JSONL records for concurrent CPU scoring."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from kdm.decoding import DecodeConfig
from kdm.io import file_hash, read_jsonl, stable_hash, stable_seed, within
from kdm.pipeline import task_id
from kdm.prompts import task_prompt

MODELS = {"qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26",
          "gemma3_4b", "internvl35_8b", "onevision", "phi35", "qwen3vl"}


def snapshot(source, output, limit):
    output.relative_to(ROOT / "outputs/paper_core_20261002_dev_viz")
    if not source.is_dir():
        raise FileNotFoundError("The explicit native source directory is missing")
    samples = {s["id"]: s for s in read_jsonl(ROOT / "data/current/all.jsonl")
               if s["dataset"] == "food101" and s["split"] == "eval"}
    gates = sorted(source.glob("*.pilot.json"))
    if not gates:
        raise FileNotFoundError("No actual completed native eight-input pilot exists")
    output.mkdir(parents=True, exist_ok=False)
    parts = []
    all_keys = set()
    for gate_path in gates:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate["passed"] is not True or gate["completed"] != 8:
            raise ValueError("An actual native pilot is incomplete")
        stem = gate_path.name.removesuffix(".pilot.json")
        raw = source / "raw" / (stem + ".jsonl")
        sidecar_path = raw.with_suffix(".identity.json")
        claim_path = source / (stem + ".claim.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        definition = sidecar["definition"]
        inner = {k: v for k, v in definition.items()
                 if k not in {"shard", "n_shards", "base_config"}}
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        model, method = gate["model"], gate["method"]
        if (model not in MODELS or method not in {"dola", "deco", "sid"}
                or stable_hash(definition) != sidecar["identity"]
                or sidecar["identity"] != gate["ledger_identity"]
                or stable_hash(inner) != gate["identity"]
                or claim["identity"] != gate["identity"]
                or claim["expected"] != gate["expected_full_part"]):
            raise ValueError("Native pilot, owned part and ledger identity differ")
        cfg = asdict(DecodeConfig(method=method))
        if inner["config"] != cfg or definition["base_config"] != cfg:
            raise ValueError("The native decode configuration differs")
        dest = output / (stem + ".jsonl")
        pilot_digest = hashlib.sha256()
        count, keys, ids = 0, [], []
        with raw.open("rb") as stream, dest.open("xb") as sealed:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                row = json.loads(line)
                sample = row["sample"]
                task = {"sample": sample, "method": method, "marker": "NONE",
                        "reference_marker": "NONE", "guided": False,
                        "reference_guided": False, "replicate": 0,
                        "kind": "native_unguided"}
                if (sample != samples.get(sample["id"]) or row["model"] != model
                        or row["key"] != task_id(model, task)
                        or row["key"] in all_keys or sample["id"] not in claim["sample_ids"]
                        or row["identity"] != sidecar["identity"]
                        or any(row[k] != v for k, v in task.items() if k != "sample")
                        or row["config"] != cfg or row["status"] != "ok"
                        or row["seed"] != stable_seed(sample["id"], model, 0)
                        or row["prompt"] != task_prompt(sample["question"], guided=False)
                        or type(row["terminated"]) is not bool
                        or not 1 <= len(row["tokens"]) <= 32
                        or len(row["tokens"]) != len(row["selected_log_probabilities"])
                        or not all(math.isfinite(v) for v in row["selected_log_probabilities"])
                        or not math.isfinite(row["first_probability"])):
                    raise ValueError("A closed native record differs from its actual owned part")
                all_keys.add(row["key"])
                keys.append(row["key"])
                ids.append(sample["id"])
                if count < 8:
                    pilot_digest.update(line)
                sealed.write(line)
                count += 1
                if count == limit:
                    break
        if count < 8 or pilot_digest.hexdigest() != gate["raw_sha256"]:
            raise ValueError("The sealed first eight records differ from the actual pilot SHA")
        copied = {}
        for label, path in (("identity", sidecar_path), ("pilot", gate_path),
                            ("claim", claim_path)):
            dst = output / (stem + "." + label + ".json")
            with dst.open("xb") as stream:
                stream.write(path.read_bytes())
            copied[label] = {"path": str(dst.relative_to(ROOT)), "sha256": file_hash(dst),
                             "source_path": str(path)}
        parts.append({"model": model, "method": method, "rows": count,
                      "expected_owned_part": claim["expected"], "source_hostname": socket.gethostname(),
                      "source_raw_path": str(raw), "source_live_at_snapshot": True,
                      "sealed_raw_path": str(dest.relative_to(ROOT)), "raw_sha256": file_hash(dest),
                      "ledger_identity": sidecar["identity"], "runner_identity": gate["identity"],
                      "keys_sha256": stable_hash(keys), "sample_ids_sha256": stable_hash(ids),
                      "source_part_complete_claimed": False, "evidence": copied})
    receipt = {"schema": "kdm_sealed_native_baseline_snapshot_v1", "passed": True,
               "snapshot_utc": datetime.now(timezone.utc).isoformat(), "parts": parts,
               "rows": sum(p["rows"] for p in parts), "unique_keys": len(all_keys),
               "maximum_rows_per_part": limit, "whole_conditions_complete_claimed": False,
               "GPU_initialized": False, "runner_sha256": file_hash(Path(__file__))}
    with (output / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return {k: v for k, v in receipt.items() if k != "parts"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=512)
    args = parser.parse_args()
    if not 8 <= args.limit <= 2424:
        raise ValueError("A snapshot must include eight to 2424 complete records per part")
    print(json.dumps(snapshot(within(ROOT, args.source), within(ROOT, args.output), args.limit)))


if __name__ == "__main__":
    main()
