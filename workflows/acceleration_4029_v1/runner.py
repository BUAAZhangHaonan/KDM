#!/usr/bin/env python3
"""Versioned4029 execution of the unchanged qwen35_4b formal task matrix."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"workflows/acceleration_v4"))
import formal_runner as formal
sys.path.insert(0,str(Path(__file__).parent))
import admission


def check_four_shards():
 samples=[s for s in formal.read_jsonl(ROOT/"data/current/all.jsonl") if s["dataset"]=="food101"]
 methods=json.loads((ROOT/"configs/kdm/method_plan.json").read_text())["qwen35_4b"]["food101"]
 expected={formal.task_id("qwen35_4b",t) for t in formal.experiment_tasks(samples,methods=methods)}
 parts=[];reports=[]
 for shard in range(4):
  keys=set();sample_ids=set()
  for task in formal.ordered_tasks(samples,methods,shard,4):
   if int(formal.stable_hash(task["sample"]["id"])[:8],16)%4!=shard:
    raise ValueError("Frozen second shard filter would drop a task")
   keys.add(formal.task_id("qwen35_4b",task));sample_ids.add(task["sample"]["id"])
  parts.append(keys);reports.append({"shard":shard,"gpu":shard+4,"questions":len(sample_ids),"tasks":len(keys)})
 if set.union(*parts)!=expected or sum(map(len,parts))!=len(expected) or len(expected)!=155136:
  raise ValueError("Four shards are not an exact disjoint original task partition")
 return {"model":"qwen35_4b","full_tasks":len(expected),"shards":reports,
         "double_filter_idempotent":True,"disjoint_complete":True,"gpu_started":False}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--model",choices=["qwen35_4b"],default="qwen35_4b")
 p.add_argument("--shard",type=int,choices=range(4))
 p.add_argument("--n-shards",type=int,choices=[4],default=4)
 p.add_argument("--run-name")
 p.add_argument("--check-plan",action="store_true")
 p.add_argument("--check-four-shards",action="store_true")
 args=p.parse_args()
 if args.check_four_shards:
  print(json.dumps(check_four_shards(),indent=2));return
 if args.shard is None:p.error("--shard0..3 required")
 if args.check_plan:
  *_,summary=formal.load_plan(args.model,args.shard,4)
  print(json.dumps({"migration_definition":admission.DEFINITION,"cpu_plan":summary,"gpu_started":False},indent=2));return
 if not args.run_name:p.error("--run-name required for actual execution")
 admission.check_host(ROOT,os.environ.get("CUDA_VISIBLE_DEVICES","").split(","))
 # Separate data-partition lock guards against accidental duplicate launch names.
 lock_path=ROOT/"outputs/locks"/f"4029_qwen35_4b_shard_{args.shard}_of_4.lock"
 lock=lock_path.open("a")
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 admission.install(args.shard)
 formal.validate_runtime=admission.validate_runtime
 formal.make_backend=admission.make_backend
 formal.execute(args)


if __name__=="__main__":main()

