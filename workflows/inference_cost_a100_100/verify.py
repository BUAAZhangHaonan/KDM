#!/usr/bin/env python3
"""CPU-only verification of the cost roster, condition identity and raw accounting."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).parent)]
from kdm.decoding import DecodeConfig
from kdm.io import stable_seed
from kdm.prompts import task_prompt
from run import OUT, MODELS, LABELS, EXPECTED_UUIDS, read, write, sha, jsonl, task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    if args.snapshot:
        paths = ("run.py", "worker.sh", "verify.py")
        values = {}
        for name in paths:
            source = Path(__file__).with_name(name)
            destination = OUT / "workflow_snapshot" / name
            destination.parent.mkdir(exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(source.read_bytes())
            values[name] = sha(source)
        write(OUT / "workflow_source.json", dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=values))
    frozen = read(OUT / "SOURCE.json")
    measured_ids = [s["id"] for s in frozen["samples"]]
    warmup_ids = [s["id"] for s in frozen["warmup_samples"]]
    assert len(measured_ids) == len(set(measured_ids)) == 100
    assert len(warmup_ids) == len(set(warmup_ids)) == 8
    assert not set(measured_ids) & set(warmup_ids)
    assert frozen["cost_sampling_seed"] == 20260929
    assert frozen["expected_measurements"] == 2700 and frozen["batch_size"] == 1
    for relative, expected in frozen["source_hashes"].items():
        assert sha(ROOT / relative) == expected, relative
    if (OUT / "workflow_source.json").exists():
        for name, expected in read(OUT / "workflow_source.json")["source_sha256"].items():
            assert sha(OUT / "workflow_snapshot" / name) == expected
            assert sha(Path(__file__).with_name(name)) == expected
    counts = []
    total = 0
    warmups = 0
    summaries = {}
    for model in MODELS:
        folder = OUT / "models" / model
        phases = {"measurement": jsonl(folder / "measurements.jsonl"), "warmup": jsonl(folder / "warmup.jsonl")}
        if not args.partial:
            assert len(phases["measurement"]) == 300, model
            assert len(phases["warmup"]) == 24, model
            receipt = read(folder / "complete.json")
            assert receipt["passed"] and receipt["measurements_sha256"] == sha(folder / "measurements.jsonl")
            assert not (folder / "failure.json").exists()
        for phase, rows in phases.items():
            samples = frozen["samples"] if phase == "measurement" else frozen["warmup_samples"]
            pairs = set()
            devices = set()
            for row in rows:
                sample = samples[row["ordinal"] - 1]
                label = row["cost_condition"]
                key = (label, row["sample_id"])
                assert key not in pairs
                pairs.add(key)
                expected_task = task(model, sample, label, frozen["selections"])
                expected_cfg = asdict(DecodeConfig(method=expected_task["method"]))
                assert row["task"] == {k:v for k,v in expected_task.items() if k != "sample"}
                assert row["config"] == expected_cfg
                assert row["seed"] == stable_seed(sample["id"], model, 0)
                assert row["sample_id"] == sample["id"] and row["target_class"] == sample["class"]
                assert row["model"] == model and row["phase"] == phase
                assert row["dataset"] == "food101" and row["split"] == "eval"
                assert row["batch_size"] == 1 and row["status"] == "ok"
                assert math.isfinite(row["seconds"]) and row["seconds"] > 0
                assert row["output_tokens"] == len(row["tokens"]) and 0 < len(row["tokens"]) <= 32
                assert len(row["selected_log_probabilities"]) == len(row["tokens"])
                assert all(math.isfinite(v) for v in row["selected_log_probabilities"])
                assert row["termination_status"] == ("eos" if row["terminated"] else "max_tokens")
                assert row["gpu_count"] == (2 if model == "internvl35_8b" else 1)
                assert row["gpu_uuids"] == {c: EXPECTED_UUIDS[c] for c in row["physical_gpus"]}
                assert row["gpu_seconds"] == row["seconds"] * row["gpu_count"]
                assert row["model_dtype"] == ("float16" if model == "onevision" else "bfloat16")
                assert row["environment_python"] == frozen["specs"][model]["actual_spec"]["environment_python"]
                devices.add(tuple(row["physical_gpus"]))
                prompts = row["branch_prompts"]
                guided_prompt = task_prompt(sample["question"], expected_task["marker"], True)
                plain = task_prompt(sample["question"], guided=False)
                if label == "cda_visual":
                    null_prompt = task_prompt("[N/A]", guided=False)
                    assert prompts == dict(abstention=guided_prompt, prior=plain, context=plain, null_prior=null_prompt, null_context=null_prompt)
                    assert row["branch_count"] == 5 and row["cda_equations"] == "ACL2025_main_text_4_6_7_no_momentum"
                elif label == "guided_vcd":
                    assert prompts == dict(main=guided_prompt, reference=guided_prompt) and row["branch_count"] == 2
                else:
                    assert prompts == dict(main=guided_prompt, reference=plain, neutral=plain) and row["branch_count"] == 3
            assert len(devices) <= 1, model
            for label in LABELS:
                group = [r for r in rows if r["cost_condition"] == label]
                if not args.partial:
                    assert {r["sample_id"] for r in group} == {s["id"] for s in samples}
                if phase == "measurement" and group:
                    summaries[(model, label)] = dict(n=len(group), mean=statistics.mean(r["seconds"] for r in group),
                        std=statistics.stdev(r["seconds"] for r in group) if len(group) > 1 else None)
        total += len(phases["measurement"])
        warmups += len(phases["warmup"])
        counts.append(dict(model=model, measured=len(phases["measurement"]), warmups=len(phases["warmup"])))
    if not args.partial:
        assert total == 2700 and warmups == 216 and len(summaries) == 27
        with (OUT / "condition_summary.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                observed = summaries[(row["model"], row["condition"])]
                assert int(row["n"]) == 100 and int(row["std_ddof"]) == 1
                assert math.isclose(float(row["mean_seconds"]), observed["mean"], rel_tol=1e-12)
                assert math.isclose(float(row["sample_std_seconds"]), observed["std"], rel_tol=1e-12)
        with (OUT / "mean_ratios.csv").open(newline="") as handle:
            ratios = list(csv.DictReader(handle))
        assert len(ratios) == 9
        for row in ratios:
            means = {l:summaries[(row["model"], l)]["mean"] for l in LABELS}
            for field, a, b in (("ip_vcd_over_guided_vcd", "ip_vcd", "guided_vcd"),
                                ("cda_over_guided_vcd", "cda_visual", "guided_vcd"),
                                ("cda_over_ip_vcd", "cda_visual", "ip_vcd")):
                assert math.isclose(float(row[field]), means[a] / means[b], rel_tol=1e-12)
    result = dict(passed=True, scope="partial_rows" if args.partial else "complete_2700_measurements", measured=total, warmups=warmups,
        counts=counts, same_roster_verified=True, frozen_prompts_and_configs_verified=True, source_identity_verified=True,
        real_token_termination_and_gpu_identity_verified=True, no_semantic_labels_read=True, std_ddof=1,
        checked_utc=datetime.now(timezone.utc).isoformat())
    write(OUT / ("partial_verification.json" if args.partial else "acceptance.json"), result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
