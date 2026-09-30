"""Replay saved native prefixes to resolve an operator-audit discrepancy; generate no answers."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kdm.decoding import DecodeConfig, distribution
from kdm.execution import resolve_image_path
from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash
from kdm.pipeline import make_backend, sessions
from kdm.probability import log_normalize
from execution import admit, runtime_spec
from native_audit import NativeAuditBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpus", required=True)
    args = parser.parse_args()
    folder = ROOT / "outputs/paper_core_20260930" / args.run_id
    raw_path = folder / "raw" / f"{args.model}_native_vcd.jsonl"
    rows = list(read_jsonl(raw_path))
    if len(rows) != 8:
        raise ValueError("Audit replay requires the saved eight inputs")
    identity = json.loads((folder / f"{args.model}.identity.json").read_text())
    old_hash = file_hash(raw_path)
    frozen_spec = json.loads((ROOT / f"configs/runtime/{args.model}.json").read_text())
    receipt = admit(ROOT, frozen_spec, args.model, args.physical_gpus.split(","))
    if receipt["fixed_identity"] != identity["admission"]:
        raise ValueError("Native audit runtime differs from the saved generation")
    backend = NativeAuditBackend(make_backend(runtime_spec(ROOT, frozen_spec, args.model), "cuda:0"),
                                 args.model, [row["sample"] for row in rows])
    issues = []
    for row in rows:
        cfg = DecodeConfig(**row["config"])
        if asdict(cfg) != identity["decode_config"]:
            raise ValueError("Native saved generation configuration differs")
        with Image.open(resolve_image_path(row["sample"]["image_path"], ROOT)) as source:
            image = source.convert("RGB")
        main_view, ref_view, _, _, _ = sessions(backend, image, row, cfg, row["seed"])
        for position, token in enumerate(row["tokens"]):
            prefix = tuple(row["tokens"][:position])
            clean, noisy = main_view.next(prefix), ref_view.next(prefix)
            project, _ = distribution(clean, noisy, cfg, position)
            c, r = np.asarray(clean.logits, dtype=np.float64), np.asarray(noisy.logits, dtype=np.float64)
            keep = c >= c.max() + np.log(cfg.beta)
            official = log_normalize(np.where(keep, (1 + cfg.alpha) * c - cfg.alpha * r, -np.inf))
            support_equal = np.array_equal(np.isfinite(project), keep)
            error = float(np.max(np.abs(project[keep] - official[keep])))
            observed_gap = float(project.max() - project[token])
            official_gap = float(official.max() - official[token])
            check = backend.records[-1]["step_checks"][-1]
            check.update(observed_token=token, pipeline_distribution_argmax=int(np.argmax(project)),
                         pipeline_operator_error=error, observed_logp_gap=observed_gap,
                         official_observed_logp_gap=official_gap, support_equal=support_equal)
            if not support_equal or error > 1e-8 or observed_gap > 1e-8 or official_gap > 1e-8:
                issues.append({"sample_id": row["sample"]["id"], **check})
        backend.records[-1].update(tokens=row["tokens"], text=row["text"],
                                   terminated=row["terminated"], truncated=not row["terminated"],
                                   eos_token_ids=sorted(backend.eos), wall_s=row["wall_s"], config=row["config"])
    passed = not issues and file_hash(raw_path) == old_hash
    proof = {"schema": "kdm_native_existing_prefix_audit_v1", "model": args.model,
             "identity": stable_hash(identity), "required": 8, "completed": 8,
             "passed": passed, "records": backend.records, "issues": issues,
             "raw_sha256_before_and_after": old_hash,
             "new_free_generations": 0, "measurement": "saved real prefixes, same registered inputs and seed",
             "earlier_failure": "raw-score argmax check rejected a saved generated token; retain source and resolve by actual pipeline operator with numerical tolerance",
             "audit_source_sha256": file_hash(Path(__file__)),
             "gate_generation_wall_s": sum(row["wall_s"] for row in rows),
             "admission": receipt, "written_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(folder / f"{args.model}.replay_audit.json", proof)
    if not passed:
        raise ValueError("Native saved-prefix operator discrepancy remains unresolved")
    atomic_json(folder / f"{args.model}.gate_8.json", proof)
    print(json.dumps({"model": args.model, "passed": passed, "rows": 8,
                      "new_generations": 0, "gate_generation_wall_s": proof["gate_generation_wall_s"],
                      "raw_sha256": old_hash}, indent=2))


if __name__ == "__main__":
    main()
