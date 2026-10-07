#!/usr/bin/env python3
"""Bounded A100 wall-clock cost measurement through the existing KDM operators."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
OUT = ROOT / "outputs/inference_cost_a100_100_20261007"
MODELS = ("qwen25vl", "qwen35_4b", "llava16_mistral", "minicpm26", "gemma3_4b",
          "internvl35_8b", "onevision", "phi35", "qwen3vl")
LABELS = ("guided_vcd", "ip_vcd", "cda_visual")
SELECTION = "outputs/paper_core_20261002_dev_viz/ip_dev_full21008_closed_selected_20261003_2226/selected_configs.json"
COST_SEED = 20260929
EXPECTED_UUIDS = {"0": "GPU-6f5dc226-6850-9f93-d4b7-b6f2d618b402",
                  "1": "GPU-5c45e961-7442-eb90-9b8c-295a1cf14995"}
ALGORITHMS = ("src/kdm/pipeline.py", "src/kdm/decoding.py", "src/kdm/cda.py",
              "src/kdm/prompts.py", "src/kdm/probability.py", "src/kdm/models/hf.py",
              "src/kdm/models/backbone.py", "src/kdm/models/remote.py",
              "src/kdm/models/internvl_dual.py", "src/kdm/models/internvl_preprocessing.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    part.replace(path)


def append(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def csv_file(path, rows, fields=None):
    if not rows and fields is None:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def task(model, sample, label, selected):
    marker = selected[model]["marker"] if label != "cda_visual" else "UNKNOWN"
    return dict(sample=sample, method="vcd" if label == "guided_vcd" else (
                "instruction_vcd" if label == "ip_vcd" else "cda_visual"),
                kind="main" if label == "guided_vcd" else "instruction_preserving",
                marker=marker, reference_marker=marker, guided=True,
                reference_guided=label == "guided_vcd", replicate=0)


def prepare():
    import numpy as np
    from kdm.decoding import DecodeConfig
    from kdm.frozen import load_contract
    from kdm.io import read_jsonl
    if OUT.exists():
        raise FileExistsError("The exclusive cost run directory already exists")
    if socket.gethostname() != "oem-PowerEdge-T640":
        raise ValueError("Cost measurement is authorized only on 6403")
    manifest, freeze = load_contract(ROOT)
    paths = ("configs/kdm/study.json", "configs/kdm/method_plan.json", "data/current/all.jsonl")
    source_hashes = {p: sha(ROOT / p) for p in (*paths, *ALGORITHMS, SELECTION)}
    for path in paths:
        if source_hashes[path] != freeze["files"][path]:
            raise ValueError("Frozen input differs: " + path)
    study = read(ROOT / paths[0])
    if study["max_new_tokens"] != 32 or not study["greedy"] or study["alpha"] != 1 or study["beta"] != .1:
        raise ValueError("The registered Food decoding configuration differs")
    samples = sorted((s for s in read_jsonl(ROOT / paths[2]) if s["dataset"] == "food101" and s["split"] == "eval"), key=lambda s: s["id"])
    if len(samples) != 2424 or len({s["id"] for s in samples}) != 2424 or set(Counter(s["class"] for s in samples).values()) != {24}:
        raise ValueError("Food eval2424 identity differs")
    rng = np.random.RandomState(COST_SEED)
    positions = rng.choice(len(samples), size=100, replace=False).tolist()
    selected_samples = [samples[i] for i in positions]
    remaining = [i for i in range(len(samples)) if i not in set(positions)]
    warm_positions = rng.choice(remaining, size=8, replace=False).tolist()
    warm_samples = [samples[i] for i in warm_positions]
    selections = {r["model"]: r for r in read(ROOT / SELECTION) if r["method"] == "instruction_vcd"}
    if set(selections) != set(MODELS):
        raise ValueError("The frozen IP-VCD selections do not cover the requested nine models")
    registries = [read(ROOT / p) for p in ("workflows/paper_core/host_registry_a100.json",
                   "workflows/paper_core/host_registry.json", "workflows/supplemental/remaining4/6403_host_registry.json",
                   "workflows/general_vqa_direct/host_registry_q35_6403.json")]
    overrides = {}
    for registry in registries:
        overrides.update(registry["runtime_overrides"].get("6403", {}))
    if set(overrides) != set(MODELS):
        raise ValueError("Missing unchanged registered A100 runtime paths")
    specs = {}
    for model in MODELS:
        path = f"configs/runtime/{model}.json"
        if sha(ROOT / path) != freeze["files"][path]:
            raise ValueError("Frozen model spec changed: " + model)
        spec = read(ROOT / path)
        if spec["gpu_count"] != (2 if model == "internvl35_8b" else 1):
            raise ValueError("The registered GPU count differs")
        if spec["dtype"] != ("float16" if model == "onevision" else "bfloat16"):
            raise ValueError("The registered precision differs")
        actual = copy.deepcopy(spec)
        actual["environment_python"] = overrides[model]["environment_python"]
        actual["kwargs"]["model_path"] = overrides[model]["model_path"]
        actual["processor"]["path"] = overrides[model]["model_path"]
        specs[model] = {"original_spec": spec, "actual_spec": actual, "original_spec_sha256": sha(ROOT / path)}
    OUT.mkdir(parents=True)
    host_registry = dict(schema=1, purpose="Authorized Food eval100 inference cost; unchanged registered operators and runtime paths",
        hosts={"6403": dict(hostname=socket.gethostname(), root=str(ROOT), allowed_gpus=[0, 1],
               max_workers_per_gpu=1, gpu_uuids=EXPECTED_UUIDS,
               minimum_free_mib_by_model={model: 22528 if model == "internvl35_8b" else 32768 for model in MODELS})},
        model_hosts={model: "6403" for model in MODELS}, runtime_overrides={"6403": overrides},
        image_prefixes={"/home/g203-4028/projects/knowledge-deficit-mitigation/data/images": "data/images"},
        image_catalog="outputs/records/image_content_catalog.json")
    write(OUT / "host_registry.json", host_registry)
    frozen = dict(schema="kdm_inference_cost_a100_100_v1", created_utc=now(), models=list(MODELS), labels=list(LABELS),
        expected_measurements=2700, samples=selected_samples, warmup_samples=warm_samples,
        cost_sampling_seed=COST_SEED,
        cost_sampling_algorithm="NumPy RandomState(20260929).choice on sorted Food eval2424 sample IDs, 100 without replacement; then 8 independent IDs from remaining indices",
        seed_scope="Independent registration for this cost sample; not the original Food/Viz sample seed or per-input generation seed",
        numpy_sampling_version=np.__version__, roster_positions=positions, warmup_positions=warm_positions,
        selections=selections, specs=specs, frozen_contract_sha256=manifest["original_contract_sha256"],
        source_hashes=source_hashes, base_decode_config=asdict(DecodeConfig()), batch_size=1,
        timing="After model loading; CUDA synchronize before starting, image read/RGB conversion/prompt/session construction/generation/session and image cleanup, CUDA synchronize before stopping; excludes result serialization and model loading",
        warmup_reason="Eight independent inputs per model through all three conditions to initialize kernels/caches; warmup never enters the 100-input summaries",
        condition_order="Rotate the three conditions by sample ordinal; all conditions use each common ID once",
        cda_operating_point=dict(method="cda_visual", kind="instruction_preserving", marker="UNKNOWN", reference_marker="UNKNOWN", guided=True, reference_guided=False,
             source="workflows/paper_core/assemble_nine_main_food.py requested_conditions; actual Food main-comparison operating point", equations="ACL2025_main_text_4_6_7_no_momentum", null_convention="same_prefix_text_placeholder_and_uniform_image", branches=5),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        automatic_retry=False, automatic_parameter_changes=False, performance_scoring=False,
        executor="/root/cost_a100_100")
    write(OUT / "SOURCE.json", frozen)
    for name, values in (("eval100", selected_samples), ("warmup8", warm_samples)):
        csv_file(OUT / (name + ".csv"), [dict(ordinal=i + 1, sample_id=s["id"], target_class=s["class"], split=s["split"], image_path=s["image_path"]) for i, s in enumerate(values)])
    for model in MODELS:
        write(OUT / "registration" / (model + ".json"), dict(model=model, runtime=specs[model], conditions=[dict(label=l, branch_count=(2, 3, 5)[i], task={k:v for k,v in task(model, selected_samples[0], l, selections).items() if k != "sample"}, decode_config=asdict(DecodeConfig(method=task(model, selected_samples[0], l, selections)["method"]))) for i,l in enumerate(LABELS)]))
    write(OUT / "RUN_STATUS.json", dict(status="prepared", expected_measurements=2700, measured=0, created_utc=now()))
    write(OUT / "CURRENT_STATE.json", dict(status="prepared", expected_measurements=2700, measured=0, created_utc=now()))
    print(json.dumps(dict(status="prepared", expected_measurements=2700, eval_ids=100, independent_warmup_ids=8, source_sha256=sha(OUT / "SOURCE.json"))), flush=True)


def one_input(backend, model, sample, label, frozen):
    """The same branches and operators as pipeline.run_tasks, without Ledger I/O."""
    from PIL import Image
    from kdm.cda import generate_cda
    from kdm.decoding import DecodeConfig, generate
    from kdm.execution import resolve_image_path
    from kdm.io import stable_seed
    from kdm.pipeline import sessions
    from kdm.prompts import task_prompt
    t = task(model, sample, label, frozen["selections"])
    cfg = DecodeConfig(method=t["method"])
    seed = stable_seed(sample["id"], model, 0)
    image = None
    main = reference = neutral = prior = context = null_prior = null_context = null_image = None
    try:
        with Image.open(resolve_image_path(sample["image_path"], ROOT)) as opened:
            image = opened.convert("RGB")
        main, reference, neutral, prompt, rprompt = sessions(backend, image, t, cfg, seed)
        if label == "cda_visual":
            plain = task_prompt(sample["question"], guided=False)
            prior = backend.session(image, plain, reference="text_only")
            context = backend.session(image, plain)
            null_prompt = task_prompt("[N/A]", guided=False)
            null_prior = backend.session(image, null_prompt, reference="text_only")
            null_image = Image.new("RGB", image.size, (127, 127, 127))
            null_context = backend.session(null_image, null_prompt)
            result = generate_cda(prior, context, main, null_prior, null_context, cfg, backend.eos, backend.decode, seed)
            branch_prompts = dict(abstention=prompt, prior=plain, context=plain, null_prior=null_prompt, null_context=null_prompt)
        else:
            result = generate(main, reference, cfg, backend.eos, backend.decode, seed, neutral)
            branch_prompts = dict(main=prompt, reference=rprompt)
            if neutral is not None:
                branch_prompts["neutral"] = task_prompt(sample["question"], guided=False)
        if result["status"] != "ok" or not 0 < len(result["tokens"]) <= 32 or result["terminated"] != (result["tokens"][-1] in backend.eos):
            raise ValueError("Actual generation identity or termination is invalid")
        return dict(model=model, sample_id=sample["id"], target_class=sample["class"], dataset="food101", split="eval",
            cost_condition=label, task={k:v for k,v in t.items() if k != "sample"}, config=asdict(cfg), seed=seed,
            branch_prompts=branch_prompts, branch_count=2 if label == "guided_vcd" else (3 if label == "ip_vcd" else 5),
            output_tokens=len(result["tokens"]), termination_status="eos" if result["terminated"] else "max_tokens", **result)
    finally:
        del main, reference, neutral, prior, context, null_prior, null_context
        if null_image is not None:
            null_image.close()
        if image is not None:
            image.close()


def worker(model, cards):
    import torch
    from kdm.execution import resolve_image_path
    from kdm.frozen import load_contract
    from kdm.pipeline import make_backend
    from workflows.supplemental.remaining11.generate import validate_proofs
    from workflows.supplemental.remaining11 import execution
    frozen = read(OUT / "SOURCE.json")
    folder = OUT / "models" / model
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "claim.json").open("x", encoding="utf-8") as handle:
        json.dump(dict(model=model, cards=cards, pid=os.getpid(), started_utc=now(), argv=sys.argv, owner="/root/cost_a100_100"), handle)
    started = time.perf_counter()
    try:
        for path, expected in frozen["source_hashes"].items():
            if sha(ROOT / path) != expected:
                raise ValueError("Frozen cost source changed: " + path)
        spec = frozen["specs"][model]["original_spec"]
        manifest, freeze = load_contract(ROOT)
        proofs = validate_proofs(ROOT, spec, model, ["vcd", "instruction_vcd", "cda_visual"], "formal", manifest, freeze)
        execution.REGISTRY = str((OUT / "host_registry.json").relative_to(ROOT))
        os.environ["KDM_SUPPLEMENTAL_DATASET"] = "food101"
        admission = execution.validate_supplemental_runtime(ROOT, spec, model, cards, "formal", "inference_cost_a100_100_20261007", "/root/cost_a100_100")
        actual = admission["runtime_spec"]
        if actual != frozen["specs"][model]["actual_spec"]:
            raise ValueError("Cost runtime differs from its path-only registration")
        for sample in (*frozen["warmup_samples"], *frozen["samples"]):
            resolve_image_path(sample["image_path"], ROOT)
        write(folder / "admission.json", dict(admission=admission, proofs=proofs, source_sha256=sha(OUT / "SOURCE.json"), observed_utc=now()))
        write(folder / "CURRENT_STATE.json", dict(status="loading", model=model, cards=cards, pid=os.getpid(), updated_utc=now()))
        load_start = time.perf_counter()
        backend = make_backend(actual, "cuda:0")
        for logical in range(len(cards)):
            torch.cuda.synchronize(logical)
        load_s = time.perf_counter() - load_start
        placement = Counter(str(p.device) for p in backend.model.parameters())
        dtype_counts = Counter(str(p.dtype) for p in backend.model.parameters() if p.is_floating_point())
        expected_dtype = "torch.float16" if model == "onevision" else "torch.bfloat16"
        if set(dtype_counts) != {expected_dtype} or any("cuda" not in d for d in placement):
            raise ValueError("Loaded model precision/placement differs from the requested registration")
        write(folder / "loaded.json", dict(model_load_seconds=load_s, parameter_devices=dict(placement), floating_parameter_dtypes=dict(dtype_counts), eos_token_ids=sorted(backend.eos), model_device_map=getattr(backend.model, "hf_device_map", None), loaded_utc=now()))
        print(json.dumps(dict(event="loaded", model=model, cards=cards, load_seconds=load_s)), flush=True)
        measured = []
        warmup = []
        for phase, samples, path in (("warmup", frozen["warmup_samples"], folder / "warmup.jsonl"), ("measurement", frozen["samples"], folder / "measurements.jsonl")):
            for ordinal, sample in enumerate(samples):
                for label in LABELS[ordinal % 3:] + LABELS[:ordinal % 3]:
                    for logical in range(len(cards)):
                        torch.cuda.synchronize(logical)
                    before = time.perf_counter()
                    row = one_input(backend, model, sample, label, frozen)
                    for logical in range(len(cards)):
                        torch.cuda.synchronize(logical)
                    seconds = time.perf_counter() - before
                    row.update(seconds=seconds, phase=phase, ordinal=ordinal + 1, gpu_count=len(cards), physical_gpus=cards,
                        gpu_uuids={c: EXPECTED_UUIDS[c] for c in cards}, gpu_seconds=seconds * len(cards), batch_size=1,
                        environment_python=sys.executable, model_dtype=actual["dtype"], measured_utc=now())
                    if not math.isfinite(seconds) or seconds <= 0:
                        raise ValueError("Invalid measured wall clock")
                    append(path, row)
                    (measured if phase == "measurement" else warmup).append(row)
                if phase == "measurement" and ordinal == 3:
                    means = {label: statistics.mean(r["seconds"] for r in measured if r["cost_condition"] == label) for label in LABELS}
                    estimate = sum(means.values()) * (100 - 4)
                    write(folder / "pilot4.json", dict(model=model, actual_measurements=12, counted_in_eval100=True, seconds_per_input_by_condition=means,
                        estimated_remaining_model_wall_seconds=estimate, estimated_remaining_gpu_seconds=estimate * len(cards), model_load_seconds=load_s, completed_utc=now()))
                    print(json.dumps(dict(event="pilot4", model=model, means=means, estimated_remaining_wall_seconds=estimate)), flush=True)
                write(folder / "CURRENT_STATE.json", dict(status="running", phase=phase, model=model, pid=os.getpid(), cards=cards,
                    completed_measurements=len(measured), completed_warmups=len(warmup), completed_input_ordinal=ordinal + 1, updated_utc=now()))
        if len(measured) != 300 or len(warmup) != 24:
            raise ValueError("Missing cost measurements or independent warmups")
        for label in LABELS:
            rows = [r for r in measured if r["cost_condition"] == label]
            if len(rows) != 100 or {r["sample_id"] for r in rows} != {s["id"] for s in frozen["samples"]}:
                raise ValueError("The shared 100-input roster differs")
        write(folder / "complete.json", dict(passed=True, model=model, measured=300, warmups=24, measurements_sha256=sha(folder / "measurements.jsonl"),
            model_load_seconds=load_s, measured_seconds=sum(r["seconds"] for r in measured), warmup_seconds=sum(r["seconds"] for r in warmup),
            worker_wall_seconds=time.perf_counter() - started, completed_utc=now()))
        write(folder / "CURRENT_STATE.json", dict(status="complete", model=model, measured=300, warmups=24, updated_utc=now()))
        print(json.dumps(dict(event="complete", model=model, measured=300)), flush=True)
    except BaseException as error:
        evidence = dict(status="failed", model=model, error=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter() - started, automatic_retry=False, parameters_changed=False, failed_utc=now())
        if hasattr(error, "cda_diagnostics"):
            evidence["cda_diagnostics"] = error.cda_diagnostics
        write(folder / "failure.json", evidence)
        write(folder / "CURRENT_STATE.json", evidence)
        raise


def summarize():
    frozen = read(OUT / "SOURCE.json")
    summaries = []
    ratios = []
    details = []
    failures = []
    complete = []
    for model in MODELS:
        folder = OUT / "models" / model
        rows = jsonl(folder / "measurements.jsonl")
        if (folder / "failure.json").exists():
            failures.append(read(folder / "failure.json"))
        if (folder / "complete.json").exists():
            receipt = read(folder / "complete.json")
            if not receipt["passed"] or receipt["measurements_sha256"] != sha(folder / "measurements.jsonl"):
                raise ValueError("Completed raw cost identity differs")
            complete.append(model)
        means = {}
        for label in LABELS:
            group = [r for r in rows if r["cost_condition"] == label]
            if not group:
                continue
            if len({r["sample_id"] for r in group}) != len(group):
                raise ValueError("Duplicated measured input")
            seconds = [r["seconds"] for r in group]
            mean = statistics.mean(seconds)
            means[label] = mean
            summaries.append(dict(model=model, condition=label, n=len(group), mean_seconds=mean,
                sample_std_seconds=statistics.stdev(seconds) if len(seconds) > 1 else None, std_ddof=1,
                mean_output_tokens=statistics.mean(r["output_tokens"] for r in group),
                eos_count=sum(r["terminated"] for r in group), max_tokens_count=sum(not r["terminated"] for r in group),
                gpu_count=group[0]["gpu_count"], physical_gpus=",".join(group[0]["physical_gpus"]), dtype=group[0]["model_dtype"],
                complete_100=len(group) == 100, input_roster_sha256=sha(OUT / "eval100.csv")))
            details.extend(dict(model=r["model"], condition=r["cost_condition"], sample_id=r["sample_id"], ordinal=r["ordinal"],
                seconds=r["seconds"], output_tokens=r["output_tokens"], termination_status=r["termination_status"],
                gpu_count=r["gpu_count"], physical_gpus=",".join(r["physical_gpus"]), gpu_uuids=";".join(r["gpu_uuids"].values()),
                gpu_seconds=r["gpu_seconds"], dtype=r["model_dtype"], seed=r["seed"], batch_size=r["batch_size"]) for r in group)
        if set(means) == set(LABELS):
            ratios.append(dict(model=model, ip_vcd_over_guided_vcd=means["ip_vcd"] / means["guided_vcd"],
                cda_over_guided_vcd=means["cda_visual"] / means["guided_vcd"], cda_over_ip_vcd=means["cda_visual"] / means["ip_vcd"],
                ratio_definition="ratio of condition mean wall seconds", complete_100=model in complete))
    csv_file(OUT / "condition_summary.csv", summaries)
    csv_file(OUT / "mean_ratios.csv", ratios)
    csv_file(OUT / "per_input.csv", details)
    passed = len(details) == 2700 and set(complete) == set(MODELS) and not failures
    status = dict(status="complete" if passed else ("failed_or_partial" if failures else "running"), passed=passed,
        expected_measurements=2700, measured=len(details), complete_models=complete, failed_models=[r["model"] for r in failures],
        summary_conditions=len(summaries), input_ids=100, std_ddof=1, warmups_excluded=True,
        no_model_loading_in_measurements=True, no_parameter_fallback=True, performance_scoring=False, updated_utc=now())
    write(OUT / "RUN_STATUS.json", status)
    write(OUT / "CURRENT_STATE.json", status)
    report = ["# A100 inference cost on the common Food eval100 subset", "", frozen["timing"], "",
        "All methods use batch 1, greedy decoding, temperature 0, top_p 1, and 32 new tokens maximum. BF16 is retained for eight models; OneVision retains FP16. InternVL retains its original explicit two-card device map. Independent warmup8 inputs and model loading are excluded from the 100-input means and sample standard deviations (ddof=1).", "",
        "Cost sampling seed 20260929 is registered independently for this measurement. All three conditions of all nine models use the same frozen eval100 IDs. Per-input generation/noise seeds still use the existing stable_seed(sample_id, model, 0).", "",
        "| Model | Condition | n | Mean seconds | Sample std seconds | Mean output tokens | GPU count |", "|---|---|---:|---:|---:|---:|---:|"]
    report.extend(f"| {r['model']} | {r['condition']} | {r['n']} | {r['mean_seconds']:.6f} | {r['sample_std_seconds']:.6f} | {r['mean_output_tokens']:.2f} | {r['gpu_count']} |" for r in summaries if r["sample_std_seconds"] is not None)
    report.extend(["", "Ratios are ratios of the measured condition means; generated lengths and termination statuses are retained in per_input.csv and measurements.jsonl. These measurements describe end-to-end single-input cost on these registered implementations and this 100-input subset.", "", f"Actual measurements: {len(details)}/2700. Completed models: {', '.join(complete)}. Failed models: {', '.join(status['failed_models']) or 'none'}. "])
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return status


def supervise():
    frozen = read(OUT / "SOURCE.json")
    (OUT / "logs").mkdir(exist_ok=True)
    with (OUT / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (OUT / "supervisor_claim.json").open("x") as handle:
            json.dump(dict(pid=os.getpid(), starttick=Path("/proc/self/stat").read_text().split()[21], started_utc=now(), argv=sys.argv), handle)
        # Dual InternVL first; then independent single-card workers, one per card.
        batches = [(('internvl35_8b', ['0', '1']),), (('qwen3vl', ['0']), ('gemma3_4b', ['1'])),
                   (('onevision', ['0']), ('llava16_mistral', ['1'])),
                   (('minicpm26', ['0']), ('phi35', ['1'])), (('qwen25vl', ['0']), ('qwen35_4b', ['1']))]
        for batch in batches:
            processes = []
            for model, cards in batch:
                python = frozen["specs"][model]["actual_spec"]["environment_python"]
                command = ["bash", str(Path(__file__).with_name("worker.sh")), str(ROOT), ",".join(cards), python,
                    str(Path(__file__)), "worker", "--model", model, "--cards", ",".join(cards)]
                handle = (OUT / "logs" / (model + ".log")).open("x")
                process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                append(OUT / "commands.jsonl", dict(model=model, cards=cards, command=command, pid=process.pid, started_utc=now()))
                processes.append((model, process, handle))
                print(json.dumps(dict(event="launched", model=model, cards=cards, pid=process.pid)), flush=True)
            pending = list(processes)
            while pending:
                changed = False
                for item in pending[:]:
                    model, process, handle = item
                    result = process.poll()
                    if result is not None:
                        handle.close()
                        append(OUT / "exits.jsonl", dict(model=model, exit_code=result, completed_utc=now()))
                        print(json.dumps(dict(event="worker_exit", model=model, exit_code=result)), flush=True)
                        pending.remove(item)
                        changed = True
                if changed:
                    summarize()
                if pending:
                    time.sleep(5)
        final = summarize()
        write(OUT / "supervisor_complete.json", final)
        print(json.dumps(final), flush=True)
        if not final["passed"]:
            raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "worker", "supervise", "summarize"))
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--cards")
    a = p.parse_args()
    if a.mode == "prepare":
        prepare()
    elif a.mode == "worker":
        worker(a.model, a.cards.split(","))
    elif a.mode == "supervise":
        supervise()
    else:
        print(json.dumps(summarize()), flush=True)


if __name__ == "__main__":
    main()
