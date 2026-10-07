#!/usr/bin/env python3
"""Finish the fixed nine-model cost queue as each owned A100 worker exits."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import run
from run import ROOT, OUT, MODELS, now, read, write, append, sha


def complete_lines(path):
    path = Path(path)
    if not path.exists():
        return []
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        payload = payload.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in payload else b""
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


run.jsonl = complete_lines  # Reader only; the model timing implementation stays unchanged.


def worker_alive(claim):
    proc = Path("/proc") / str(claim["pid"])
    if not proc.exists():
        return False
    stat = (proc / "stat").read_text()
    if stat.split()[2] == "Z":
        return False
    command = (proc / "cmdline").read_bytes().split(b"\0")
    args = [value.decode() for value in command if value]
    if "worker" not in args or "--model" not in args or args[args.index("--model") + 1] != claim["model"]:
        raise ValueError("Owned worker PID identity differs")
    if not any(value.endswith("workflows/inference_cost_a100_100/run.py") for value in args):
        raise ValueError("PID is outside this bounded cost workflow")
    return True


def observe():
    active = {}
    for model in MODELS:
        path = OUT / "models" / model / "claim.json"
        if path.exists():
            claim = read(path)
            if worker_alive(claim):
                active[model] = claim
    return active


def replace_old_supervisor():
    old = read(OUT / "supervisor_claim.json")
    proc = Path("/proc") / str(old["pid"])
    if not proc.exists():
        raise ValueError("Expected original supervisor is absent; no implicit replacement")
    assert (proc / "stat").read_text().split()[21] == old["starttick"]
    args = [value.decode() for value in (proc / "cmdline").read_bytes().split(b"\0") if value]
    assert args[-1] == "supervise" and any(value.endswith("workflows/inference_cost_a100_100/run.py") for value in args)
    active = observe()
    transition = dict(old_supervisor=old, old_supervisor_command=args, active_worker_claims_before=active,
        action="Replace only the owned CPU scheduling process; retain all existing model workers and locks",
        worker_termination_requested=False, scientific_source_changed=False, measurement_runner_changed=False,
        model_retry_requested=False, started_utc=now())
    write(OUT / "scheduler_transition.json", transition)
    os.kill(old["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while proc.exists() and (proc / "stat").read_text().split()[2] != "Z":
        if time.monotonic() >= deadline:
            raise TimeoutError("Original supervisor did not stop; do not dispatch")
        time.sleep(.1)
    after = observe()
    assert {m:c["pid"] for m,c in active.items()} == {m:c["pid"] for m,c in after.items()}
    transition.update(old_supervisor_stopped_utc=now(), active_worker_claims_after=after,
        active_worker_pid_continuity_verified=True)
    write(OUT / "scheduler_transition.json", transition)
    print(json.dumps(dict(event="scheduler_replaced", old_supervisor_pid=old["pid"], retained_worker_pids={m:c["pid"] for m,c in after.items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-supervisor", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    if args.snapshot:
        print(json.dumps(dict(active=observe(), attempted=[r["model"] for r in complete_lines(OUT / "commands.jsonl")], scope="fixed nine models only")), flush=True)
        return
    if args.replace_supervisor:
        replace_old_supervisor()
    frozen = read(OUT / "SOURCE.json")
    with (OUT / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (OUT / "refill_supervisor_claim.json").open("x") as handle:
            json.dump(dict(pid=os.getpid(), starttick=Path("/proc/self/stat").read_text().split()[21], started_utc=now()), handle)
        write(OUT / "refill_source.json", dict(source_sha256=sha(Path(__file__)), fixed_queue=list(MODELS),
            measurement_runner_sha256=sha(Path(__file__).with_name("run.py")),
            aggregation_reads_only_complete_newline_terminated_records=True, automatic_retry=False, created_utc=now()))
        commands = complete_lines(OUT / "commands.jsonl")
        assert len({r["model"] for r in commands}) == len(commands), "A model was dispatched twice"
        attempted = {r["model"] for r in commands}
        queue = [m for m in ("internvl35_8b", "qwen3vl", "gemma3_4b", "onevision", "llava16_mistral", "phi35", "minicpm26", "qwen25vl", "qwen35_4b") if m not in attempted]
        terminal = {r["model"] for r in complete_lines(OUT / "exits.jsonl")}
        jobs = {}
        while True:
            for model, (process, handle, cards) in list(jobs.items()):
                result = process.poll()
                if result is not None:
                    handle.close()
                    append(OUT / "exits.jsonl", dict(model=model, exit_code=result, scheduler="refill", completed_utc=now()))
                    terminal.add(model)
                    del jobs[model]
                    print(json.dumps(dict(event="worker_exit", model=model, exit_code=result)), flush=True)
                    run.summarize()
            active = observe()
            for model in attempted - terminal - set(active) - set(jobs):
                folder = OUT / "models" / model
                receipt = read(folder / "complete.json") if (folder / "complete.json").exists() else None
                failure = read(folder / "failure.json") if (folder / "failure.json").exists() else None
                append(OUT / "exits.jsonl", dict(model=model, exit_code=None, adopted_process=True,
                    actual_complete_receipt_passed=bool(receipt and receipt.get("passed")), failure_evidence_present=bool(failure), completed_utc=now()))
                terminal.add(model)
                print(json.dumps(dict(event="adopted_worker_finished", model=model, passed=bool(receipt and receipt.get("passed")))), flush=True)
                run.summarize()
            occupied = {c for claim in active.values() for c in claim["cards"]}
            occupied.update(c for _, _, cards in jobs.values() for c in cards)
            free = [c for c in ["0", "1"] if c not in occupied]
            while queue and free:
                model = queue[0]
                count = frozen["specs"][model]["original_spec"]["gpu_count"]
                if count > len(free):
                    break
                cards = free[:count]
                free = free[count:]
                queue.pop(0)
                assert model not in attempted and not (OUT / "models" / model / "claim.json").exists()
                python = frozen["specs"][model]["actual_spec"]["environment_python"]
                command = ["bash", str(Path(__file__).with_name("worker.sh")), str(ROOT), ",".join(cards), python,
                    str(Path(__file__).with_name("run.py")), "worker", "--model", model, "--cards", ",".join(cards)]
                handle = (OUT / "logs" / (model + ".log")).open("x")
                process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                jobs[model] = (process, handle, cards)
                attempted.add(model)
                append(OUT / "commands.jsonl", dict(model=model, cards=cards, command=command, pid=process.pid, scheduler="refill", started_utc=now()))
                print(json.dumps(dict(event="launched", model=model, cards=cards, pid=process.pid)), flush=True)
            if not queue and not active and not jobs:
                break
            time.sleep(5)
        assert attempted == set(MODELS)
        final = run.summarize()
        if not final["passed"]:
            final["status"] = "failed_or_partial"
            write(OUT / "RUN_STATUS.json", final)
            write(OUT / "CURRENT_STATE.json", final)
        write(OUT / "supervisor_complete.json", final)
        print(json.dumps(final), flush=True)
        if not final["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
