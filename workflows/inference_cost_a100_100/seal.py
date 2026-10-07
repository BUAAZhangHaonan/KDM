#!/usr/bin/env python3
"""Seal completed cost artifacts and keep loading/warmup/allocation time separate."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from run import ROOT, OUT, MODELS, now, read, write, sha, jsonl, csv_file


def main():
    status = read(OUT / "RUN_STATUS.json")
    acceptance = read(OUT / "acceptance.json")
    assert status["passed"] and status["measured"] == 2700
    assert acceptance["passed"] and acceptance["scope"] == "complete_2700_measurements"
    commands = jsonl(OUT / "commands.jsonl")
    assert len(commands) == 9 and {r["model"] for r in commands} == set(MODELS)
    source = read(OUT / "SOURCE.json")
    rows = []
    for model in MODELS:
        folder = OUT / "models" / model
        complete = read(folder / "complete.json")
        assert complete["passed"] and complete["measured"] == 300 and complete["warmups"] == 24
        count = source["specs"][model]["original_spec"]["gpu_count"]
        rows.append(dict(model=model, gpu_count=count, measured_replies=300, warmup_replies=24,
            measured_wall_seconds=complete["measured_seconds"], warmup_wall_seconds=complete["warmup_seconds"],
            model_loading_wall_seconds=complete["model_load_seconds"],
            worker_admission_to_completion_wall_seconds=complete["worker_wall_seconds"],
            measured_allocated_gpu_seconds=complete["measured_seconds"] * count,
            warmup_allocated_gpu_seconds=complete["warmup_seconds"] * count,
            model_loading_allocated_gpu_seconds=complete["model_load_seconds"] * count,
            worker_admission_to_completion_allocated_gpu_seconds=complete["worker_wall_seconds"] * count))
    csv_file(OUT / "allocation_time_accounting.csv", rows)
    totals = {field:sum(r[field] for r in rows) for field in (
        "measured_allocated_gpu_seconds", "warmup_allocated_gpu_seconds", "model_loading_allocated_gpu_seconds",
        "worker_admission_to_completion_allocated_gpu_seconds")}
    write(OUT / "allocation_time_accounting.json", dict(totals=totals,
        allocation_definition="Measured wall seconds multiplied by allocated GPU count; not CUDA-kernel active time",
        worker_interval_definition="Existing worker timer from admission to receipt creation; excludes imports before admission and queue waits",
        means_and_std_exclude_loading_and_warmup=True, created_utc=now()))
    (OUT / "METHOD_NOTES.md").write_text(
        "# Cost interpretation\n\n"
        "The condition summaries use exactly 100 measured inputs per model and method. Sample standard deviations use ddof=1 across inputs. Ratios are ratios of the condition mean wall-clock times. Generated token counts include a generated EOS token. All 32-token-limited replies remain in the summaries. The generated-token count and EOS/token-limit state accompany each duration.\n\n"
        "Timing includes image reading/RGB conversion, prompt/session preparation, generation and cleanup, with CUDA synchronization on every allocated GPU at both boundaries. Input-content identity verification precedes timing. Each input is reread inside the timed interval under the host's normal filesystem cache. Model loading, independent warmup8, result serialization and scheduling waits do not enter condition means.\n\n"
        "The cost subset uses the new independently registered seed 20260929. The original Food/Viz sampling and original generation/noise seed rules remain separate. Nine models share the exact frozen 100 IDs. Per-model method comparisons retain the same precision, checkpoint, environment and GPU allocation. InternVL uses its original two-card layout; other models use one card. OneVision retains FP16 and the remaining models retain BF16.\n\n"
        "allocation_time_accounting.csv separately reports measured intervals, loading, warmup and worker admission-to-completion time. Allocated GPU seconds multiply wall time by assigned GPU count. These are not GPU kernel-active seconds or whole-queue occupancy. Differences in generated length and implementation work remain part of the reported end-to-end cost.\n",
        encoding="utf-8")
    code = {p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.is_file() and p.suffix in {".py", ".sh", ".md"}}
    items = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            items.append(dict(path=str(path.relative_to(OUT)), bytes=path.stat().st_size, sha256=sha(path)))
    write(OUT / "artifact_manifest.json", dict(schema="kdm_inference_cost_a100_100_sealed_artifacts_v1", passed=True,
        expected_measured_replies=2700, actual_measured_replies=2700, independent_warmup_replies=216,
        exact_models=list(MODELS), one_dispatch_per_model_verified=True, files=items, workflow_sources_sha256=code,
        manifest_excludes_itself=True, created_utc=now()))
    print(json.dumps(dict(passed=True, measured=2700, warmups=216, files=len(items), allocation_time_totals=totals)), flush=True)


if __name__ == "__main__":
    main()
