#!/usr/bin/env python3
"""Food-only dispatch; shared single-GPU queue, fixed dual-GPU models, no retries."""
import argparse,fcntl,json,os,queue,subprocess,sys,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import admission
from kdm.io import atomic_json,file_hash
from kdm.execution import local_host
from runner import now, AMENDMENT
SINGLES=["qwen35_9b","qwen35_4b","qwen25vl","gemma3_4b","llava15_7b","onevision","llava16_mistral","minicpm26","minicpm45","phi35","llava16_vicuna"]
PAIRS=["gemma3_12b","llava15_13b","internvl35_8b"]
def execute(host,name):
    local_host(ROOT,host)
    run=ROOT/"outputs/records/probes_v2"/name
    run.mkdir(parents=True,exist_ok=False)
    lock=(ROOT/"outputs/locks/food_closed_v3.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    keys=SINGLES+PAIRS if host=="4028" else ["qwen3vl","glm46v"]
    report={"host":host,"scheduler_pid":os.getpid(),"started_utc":now(),"host_complete":False,
        "scope":"Food101 closed only; independent generation paused by user","expected_per_model":4848,
        "amendment_sha256":file_hash(ROOT/AMENDMENT),
        "workflow_sha256":{p.name:file_hash(p) for p in Path(__file__).parent.iterdir() if p.is_file()},
        "jobs":{key:{"status":"not_started"} for key in keys}}
    mutex=threading.Lock()
    def update(key,**fields):
        with mutex:
            report["jobs"][key].update(fields);atomic_json(run/"status.json",report)
    def job(key,cards):
        spec=json.loads((ROOT/f"configs/runtime/{key}.json").read_text())
        assert spec["gpu_count"]==len(cards)
        command=["bash",str(ROOT/"workflows/food_closed_v3/worker.sh"),str(ROOT),",".join(map(str,cards)),spec["environment_python"],"-u",str(ROOT/"workflows/food_closed_v3/runner.py"),"--model",key,"--run-dir",str(run.relative_to(ROOT))]
        try:
            with (run/f"{key}.log").open("x") as log:
                update(key,status="starting",physical_gpus=cards,command=command)
                p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                update(key,status="running",worker_pid=p.pid,started_utc=now())
                code=p.wait()
                update(key,exit_code=code)
                if code:raise RuntimeError(f"Exit {code}; no automatic retry")
                receipt=json.loads((run/f"{key}_complete.json").read_text())
                if not receipt.get("complete") or receipt.get("closed")!=4848:raise ValueError("Incomplete coverage")
                update(key,status="complete",finished_utc=now())
        except Exception as e:
            update(key,status="failed",error=f"{type(e).__name__}: {e}",finished_utc=now())
    atomic_json(run/"status.json",report)
    if host=="4028":
        q=queue.Queue()
        for key in SINGLES:q.put(key)
        def consume(card):
            while True:
                try:key=q.get_nowait()
                except queue.Empty:return
                job(key,[card]);q.task_done()
        def pairs():
            for key in PAIRS:job(key,[4,5])
            with ThreadPoolExecutor(max_workers=2) as p:
                list(p.map(consume,[4,5]))
        with ThreadPoolExecutor(max_workers=3) as p:
            fs=[p.submit(consume,0),p.submit(consume,1),p.submit(pairs)]
            for f in fs:f.result()
    else:
        with ThreadPoolExecutor(max_workers=2) as p:
            fs=[p.submit(job,"qwen3vl",[1]),p.submit(job,"glm46v",[0])]
            for f in fs:f.result()
    report["host_complete"]=all(j["status"]=="complete" for j in report["jobs"].values())
    report["finished_utc"]=now();atomic_json(run/"status.json",report)
    return 0 if report["host_complete"] else 1
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--host",required=True,choices=["4028","6403"]);p.add_argument("--name",required=True)
    a=p.parse_args();raise SystemExit(execute(a.host,a.name))
