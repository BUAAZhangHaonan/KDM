#!/usr/bin/env python3
"""Screen one completed formal stage while later stages may still be running.

Only an exact stage counter AND exact planned task-key coverage are admitted.
The consumed immutable byte prefix is hashed twice; later append-only growth is
allowed. This never marks the model, remaining matrix, semantic labels or GT done.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(Path(__file__).parent))
from kdm.io import atomic_json,file_hash,read_jsonl,stable_hash,within
from kdm.protocol import validate_freeze
import formal_runner as formal
import formal_postprocess as post


def check_stage_counter(progress,stage,expected):
 observed=progress.get("stage_counts",{}).get(stage,0)
 declared=progress.get("stage_expected",{}).get(stage)
 if expected<=0 or declared!=expected or observed!=expected:
  raise ValueError(f"Stage not exactly complete: {stage}: observed={observed}, expected={expected}, declared={declared}")
 # A failure after this stage does not invalidate a fully checked earlier prefix.
 return True


def prefix_hash(path,nbytes):
 digest=hashlib.sha256()
 with path.open("rb") as f:
  remaining=nbytes
  while remaining:
   chunk=f.read(min(1024*1024,remaining))
   if not chunk:raise ValueError("Original completed-stage prefix shortened")
   digest.update(chunk);remaining-=len(chunk)
 return digest.hexdigest()


def scan_stage(raw_path,model,stage,expected,sample_map,identity,on_row=None):
 """Read only through the final expected row; future append bytes are irrelevant."""
 seen=set();digest=hashlib.sha256();nbytes=0;last_line=0
 with raw_path.open("rb") as stream:
  for line_number,line in enumerate(stream,1):
   if not line.endswith(b"\n"):raise ValueError("Incomplete row before completed stage ended")
   digest.update(line);nbytes+=len(line);last_line=line_number
   row=json.loads(line)
   if formal.stage_of(row)!=stage:continue
   key=row["key"];task={k:row[k] for k in formal.FIELDS};task["sample"]=row["sample"]
   if key not in expected or key in seen or formal.task_id(model,task)!=key:
    raise ValueError("Unexpected/duplicate/content-mismatched completed-stage task")
   if row["identity"]!=identity or row["model"]!=model or row["sample"]!=sample_map.get(row["sample"]["id"]):
    raise ValueError("Completed-stage identity/sample mismatch")
   if row["status"]!="ok" or not isinstance(row["terminated"],bool) or not 1<=len(row["tokens"])<=32:
    raise ValueError("Missing successful token/termination evidence")
   if len(row["selected_log_probabilities"])!=len(row["tokens"]) or not all(math.isfinite(x) for x in row["selected_log_probabilities"]):
    raise ValueError("Invalid stage token probabilities")
   seen.add(key)
   if on_row:on_row(row,line_number,line)
   if seen==expected:break
 if seen!=expected:raise ValueError(f"Stage record coverage incomplete: {len(seen)}/{len(expected)}")
 original=digest.hexdigest()
 if prefix_hash(raw_path,nbytes)!=original:raise ValueError("Completed-stage prefix changed during scan")
 return {"rows":len(seen),"source_prefix_bytes":nbytes,"source_prefix_sha256":original,
         "source_last_line":last_line,"later_file_growth_allowed":True}


def admit(record_path,stage,samples,freeze):
 record=within(ROOT,record_path)
 if not record.is_relative_to(ROOT/"outputs/records/acceleration_v4"):raise ValueError("Invalid formal record directory")
 admission_path=record/"admission.json";progress_path=record/"progress.json"
 admission=json.loads(admission_path.read_text());progress=json.loads(progress_path.read_text())
 if admission.get("schema")!="kdm_acceleration_v4_frozen_formal_generation" or admission["model"] not in formal.FIVE:
  raise ValueError("Not the registered formal first panel")
 model=admission["model"];shard=progress["shard"];n_shards=progress["n_shards"]
 paths={"runner_sha256":ROOT/"workflows/acceleration_v4/formal_runner.py",
        "panel_sha256":ROOT/formal.PANEL,"amendment_sha256":ROOT/formal.AMENDMENT,
        "manifest_sha256":ROOT/"data/current/all.jsonl",
        "method_plan_sha256":ROOT/"configs/kdm/method_plan.json",
        "backend_spec_sha256":ROOT/f"configs/runtime/{model}.json",
        "freeze_receipt_sha256":ROOT/"outputs/records/preregistration_freeze.json"}
 if any(admission[k]!=file_hash(p) for k,p in paths.items()) or admission["source_blobs"]!=freeze["source_blobs"]:
  raise ValueError("Bound original formal sources changed")
 methods=json.loads((ROOT/"configs/kdm/method_plan.json").read_text())[model]["food101"]
 expected={formal.task_id(model,t) for t in formal.ordered_tasks(samples,methods,shard,n_shards) if formal.stage_of(t)==stage}
 check_stage_counter(progress,stage,len(expected))
 if admission["task_plan"]["stage_expected"][stage]!=len(expected):raise ValueError("Admission stage plan changed")
 raw=within(ROOT,progress["output"])
 if not raw.is_relative_to(ROOT/"outputs/raw/acceleration_v4"):raise ValueError("Invalid original formal raw path")
 side_path=raw.with_suffix(".identity.json");side=json.loads(side_path.read_text())
 definition={**admission,"model":model,"shard":shard,"n_shards":n_shards,"base_config":formal.asdict(formal.CONFIG)}
 if side["definition"]!=definition or side["identity"]!=stable_hash(definition):raise ValueError("Original formal sidecar changed")
 return {"record_directory":str(record.relative_to(ROOT)),"source_path":str(raw.relative_to(ROOT)),
         "model":model,"shard":shard,"n_shards":n_shards,"stage":stage,
         "sidecar_sha256":file_hash(side_path),"source_identity":side["identity"],
         "admission_sha256":file_hash(admission_path),"stage_counter_observed":len(expected),
         "full_model_generation_claimed":False},expected


def stage_summary_markdown(report):
 lines=["# Completed-stage preliminary Food matching","",
        "Each included shard has exact stage coverage. Later stages may still be running. This is zero-API whole-answer character matching, not final semantic labeling or abstention GT.","",
        f"Stage: {report['completed_stage']}. Rows: {report['rows']}. Full five-model stage present: {report['full_panel_stage_present']}.","",
        "| Model | Method | Kind | Rows | Matched correct | Matched incorrect | Exact abstention | Unresolved |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
 for row in report["conditions"]:
  lines.append("| "+" | ".join(str(row[k]) for k in ("model","method","kind","rows","matched_answer_correct","matched_answer_incorrect","exact_abstention","unresolved"))+" |")
 lines+=["","Unresolved answers were not counted as wrong. Input model/shard coverage is explicit in summary.json. The remaining prompt matrix, full model, semantic annotation, and final GT are not declared complete.",""]
 return "\n".join(lines)


def run(args):
 freeze=validate_freeze(ROOT)
 samples=[s for s in read_jsonl(ROOT/"data/current/all.jsonl") if s["dataset"]=="food101"]
 if Counter(s["split"] for s in samples)!={"dev":2424,"eval":2424}:raise ValueError("Food split changed")
 sample_map={s["id"]:s for s in samples}
 admitted=[];partitions={};seen_partitions=set()
 for record in args.shard:
  source,expected=admit(record,args.completed_stage,samples,freeze)
  pair=(source["model"],source["shard"])
  if pair in seen_partitions or source["model"] in partitions and partitions[source["model"]]!=source["n_shards"]:
   raise ValueError("Duplicate/inconsistent stage partition")
  seen_partitions.add(pair);partitions[source["model"]]=source["n_shards"];admitted.append((source,expected))
 coverage={model:{"included_shards":sorted(s for m,s in seen_partitions if m==model),"n_shards":n,
                 "full_model_stage_present":{s for m,s in seen_partitions if m==model}==set(range(n)),
                 "full_model_generation_claimed":False} for model,n in partitions.items()}
 if args.check_only:
  proofs=[{**source,**scan_stage(ROOT/source["source_path"],source["model"],args.completed_stage,expected,sample_map,source["source_identity"])} for source,expected in admitted]
  print(json.dumps({"stage_verified":args.completed_stage,"sources":proofs,"coverage":coverage,
                    "gpu_used":False,"api_requests":0,"screening_performed":False,"full_model_generation_claimed":False}));return
 out=within(ROOT,args.out)
 if not out.is_relative_to(ROOT/"outputs/annotations/acceleration_v4"):raise ValueError("Screening output must remain under acceleration_v4 annotations")
 out.mkdir(parents=True,exist_ok=False)
 partial=out/"labels.partial.jsonl";matcher=post.quick.Matcher()
 groups=defaultdict(Counter);total=Counter();seen=set();proofs=[]
 try:
  with partial.open("x",encoding="utf-8") as sink:
   for source,expected in admitted:
    def on_row(row,line_number,line):
     pair=(row["model"],row["key"])
     if pair in seen:raise ValueError("Duplicate stage response across inputs")
     seen.add(pair);result=matcher.classify(row);group=post.condition(row)
     groups[group][result["screening_label"]]+=1;total[result["screening_label"]]+=1
     value={"schema":"formal_v4_completed_stage_quick_match_v1","model":row["model"],"key":row["key"],
            "sample_id":row["sample"]["id"],"dataset":"food101","split":row["sample"]["split"],
            **{k:row[k] for k in formal.FIELDS},"stage":args.completed_stage,
            "source_identity":row["identity"],"source_path":source["source_path"],"source_line":line_number,
            "source_row_sha256":hashlib.sha256(line).hexdigest(),**result}
     sink.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")
    proof=scan_stage(ROOT/source["source_path"],source["model"],args.completed_stage,expected,sample_map,source["source_identity"],on_row)
    proofs.append({**source,**proof})
  conditions=[{**dict(zip(("model","stage","kind","method","marker","reference_marker","reference_guided"),key)),**post.counts_record(c)} for key,c in sorted(groups.items())]
  report={"schema":"formal_v4_completed_stage_screening","completed_stage":args.completed_stage,
          "stage_complete_for_input_shards":True,"full_model_generation_claimed":False,"final_gt":False,
          "semantic_labeling_complete":False,"human_reviewed":False,"gpu_used":False,"api_requests":0,
          **post.counts_record(total),"conditions":conditions,"model_stage_coverage":coverage,"sources":proofs,
          "full_panel_stage_present":set(coverage)==formal.FIVE and all(x["full_model_stage_present"] for x in coverage.values()),
          "rules_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in (Path(__file__),ROOT/"workflows/quick_match_v1/match.py",ROOT/"configs/kdm/food_aliases.json",ROOT/"src/kdm/scoring.py")},
          "limits":"Only the exact completed stage of included shards. Future appends excluded. Character matching is preliminary; unresolved is neither correct nor incorrect."}
  partial.rename(out/"labels.jsonl");atomic_json(out/"summary.json",report)
  (out/"SUMMARY.md").write_text(stage_summary_markdown(report),encoding="utf-8")
  print(json.dumps({"out":str(out),"completed_stage":args.completed_stage,**post.counts_record(total),"full_model_generation_claimed":False,"final_gt":False}))
 except BaseException as e:
  atomic_json(out/"failure.json",{"error":type(e).__name__+": "+str(e),"partial_outputs_not_accepted":True,"final_gt":False});raise


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--shard",type=Path,action="append",required=True)
 p.add_argument("--completed-stage",choices=("unknown_main","unknown_controls"),required=True)
 p.add_argument("--out",type=Path);p.add_argument("--check-only",action="store_true")
 a=p.parse_args()
 if not a.check_only and a.out is None:p.error("--out required unless --check-only")
 run(a)

if __name__=="__main__":main()

