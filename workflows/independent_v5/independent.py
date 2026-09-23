#!/usr/bin/env python3
"""Gated vLLM independent Food attempts; CPU plan is the default.

This file is a prepared entry point, NOT a production-ready validation receipt.
Five models only, all4848 Food questions x10 attempts/model. A new vLLM cohort
must not be merged with old NumPy-sampled independent answers.

Before --execute:
1. Finish and verify this model's entire vLLM closed-score cohort (4848 rows).
2. Produce a native independent-prompt reference on the original fixed16 images:
   schema=kdm_native_independent_reference_v1; model, backend, source_blobs,
   manifest_sha256, eos_token_ids from the loaded native backend, and rows with
   sample, prompt, unexpanded_prompt_ids, expanded_prompt_ids, processor_summary,
   greedy_tokens (including EOS), greedy_selected_log_probabilities.
   Use task_prompt(question,guided=False,attempt=True), not the closed prompt.
3. On the same images in the actual vLLM engine, compare expanded tokens and
   multimodal processor tensors, greedy tokens/EOS and matched-prefix selected
   raw log probabilities. Record numerical differences, cache tokens, throughput,
   and peak memory. The required check receipt schema is documented by
   validate_independent_proof below; it binds this implementation, reference,
   checkpoint and versions. No receipt may assert exact random-sampler parity:
   NumPy and vLLM seeded random draws are distinct backend cohorts.
4. Check10 simultaneous same-prefix requests have distinct stable seeds,
   valid native-EOS handling,32-token bounds and finite selected raw logprobs.
   A real positive cache-reuse measurement and throughput are mandatory.
5. Do NOT enqueue until this evidence passes. This script never creates it.

Example CPU plan:
 python vllm_independent.py --model qwen25vl
Example CPU preflight (only once all real receipts exist):
 python vllm_independent.py --model qwen25vl --closed-record RECORD_DIR
   --native-reference REF_JSON --verification CHECK_JSON --cpu-preflight
Production additionally requires --execute --run FRESH_NAME through the approved
food_closed_v3/worker.sh using the vLLM environment on6403 GPU0 or1.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from collections import Counter
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(Path(__file__).parent))
from kdm.io import Ledger,atomic_json,file_hash,stable_hash,stable_seed,read_jsonl,within
from kdm.pipeline import task_id
from kdm.decoding import DecodeConfig
from kdm.prompts import task_prompt
from kdm.execution import resolve_image_path
sys.path.insert(0,str(ROOT/"workflows/acceleration_v4"))
import vllm_closed_v3 as closed

FIVE={"qwen25vl","qwen35_4b","llava16_mistral","minicpm26","gemma3_4b"}
assert set(closed.PATHS)==FIVE
CONFIG=DecodeConfig(temperature=1.0,top_p=1.0)


def food_samples():
 samples=[s for s in read_jsonl(ROOT/"data/current/all.jsonl") if s["dataset"]=="food101"]
 if len(samples)!=4848 or len({s["id"] for s in samples})!=4848 or Counter(s["split"] for s in samples)!={"dev":2424,"eval":2424}:
  raise ValueError("Original full Food manifest required")
 return samples


def task(sample,replicate):
 return {"sample":sample,"method":"direct","marker":"UNKNOWN","reference_marker":"UNKNOWN",
         "guided":False,"reference_guided":False,"attempt":True,"replicate":replicate,"kind":"independent_attempt"}


def parameters(model,sample_id,replicate,eos):
 if not eos or any(type(x) is not int or x<0 for x in eos):raise ValueError("Native EOS IDs required")
 return {"n":1,"temperature":1.0,"top_p":1.0,"top_k":-1,"min_p":0.0,
         "max_tokens":32,"min_tokens":0,"presence_penalty":0.0,"frequency_penalty":0.0,
         "repetition_penalty":1.0,"seed":stable_seed(sample_id,model,replicate),
         "stop_token_ids":sorted(set(eos)),"ignore_eos":True,"logprobs":0,
         "detokenize":False,"stop":None}


def engine_base_ids(model,base,expanded,bos):
 # LLaVA HF expanded inputs add BOS although template.encode(add_special_tokens=False) does not.
 if model=="llava16_mistral" and expanded and expanded[0]==bos and (not base or base[0]!=bos):
  return [bos]+list(base)
 return list(base)


def image_ranges(token_types):
 values=list(token_types);ranges=[];start=None
 for index,value in enumerate(values+[0]):
  if value==1 and start is None:start=index
  if value!=1 and start is not None:ranges.append((start,index));start=None
 return set(ranges)


def check_gemma_attention_receipt(path,expected_ranges,physical_gpu):
 receipt=json.loads(Path(path).read_text())
 module=ROOT/"workflows/acceleration_v4/gemma_audit_worker.py"
 if (receipt.get("backend")!="TRITON_ATTN" or receipt.get("nonempty_real_ranges") is not True
     or receipt.get("scope")!="real_metadata_builder_output_not_kernel_execution_proof"
     or receipt.get("num_actual_tokens",0)<=0
     or receipt.get("physical_gpu")!=physical_gpu or receipt.get("module_sha256")!=file_hash(module)):
  raise ValueError("Missing or mismatched real Gemma Triton metadata audit")
 requested={(int(a),int(b)) for spans in receipt["ranges"].values() for a,b in spans if b>a}
 tensor={(int(a),int(b)) for spans in receipt["tensor_ranges"] for a,b in spans if b>a}
 if not expected_ranges or requested!=expected_ranges or not requested.issubset(tensor):
  raise ValueError("Gemma real engine image ranges differ from native token_type_ids")
 return {"receipt_sha256":file_hash(path),"expected_native_ranges":[list(x) for x in sorted(expected_ranges)],
         "metadata_ranges_checked":True,"kernel_execution_claimed":False}


class Generator(closed.Scorer):
 def prepare(self,sample):
  # Exact v3 chat/processor branches; only the task prompt changes.
  from PIL import Image
  with Image.open(resolve_image_path(sample["image_path"],ROOT)) as source:im=source.convert("RGB")
  prompt=task_prompt(sample["question"],guided=False,attempt=True)
  if self.mt=="minicpmv":
   text=self.proc.tokenizer.apply_chat_template([{"role":"user","content":"(<image>./</image>)\n"+prompt}],tokenize=False,add_generation_prompt=True)
   inp=self.proc(text=[text],images=[[im]],return_tensors="pt")
   mask=inp["attention_mask"];pos=mask.long().cumsum(-1)-1;inp["position_ids"]=pos.masked_fill(mask==0,0)
  else:
   msgs=[{"role":"user","content":[{"type":"image","image":im},{"type":"text","text":prompt}]}]
   kw={"add_generation_prompt":True}
   if self.mt in ("qwen3_5","qwen3_vl","glm4v"):kw["enable_thinking"]=False
   text=self.proc.apply_chat_template(msgs,tokenize=False,**kw)
   inp=self.proc.apply_chat_template(msgs,tokenize=True,return_dict=True,return_tensors="pt",**kw)
  return im,prompt,self.proc.tokenizer.encode(text,add_special_tokens=False),inp

 def load(self):
  from vllm import LLM
  extra={}
  if self.mt=="gemma3":
   if not getattr(self,"audit_path",None):raise ValueError("Fresh real Gemma attention-audit path required")
   if self.audit_path.exists():raise ValueError("Gemma attention receipt must be fresh")
   os.environ["KDM_ATTN_AUDIT_PATH"]=str(self.audit_path)
   os.environ["PYTHONPATH"]=str(ROOT)+os.pathsep+os.environ.get("PYTHONPATH","")
   extra={"attention_config":{"backend":"TRITON_ATTN"},"worker_cls":"workflows.acceleration_v4.gemma_audit_worker.GemmaAuditWorker"}
  self.llm=LLM(model=self.path,dtype="bfloat16",trust_remote_code=True,tensor_parallel_size=1,
   gpu_memory_utilization=0.80,max_model_len=32768,max_num_seqs=128,max_num_batched_tokens=8192,
   enable_prefix_caching=True,enforce_eager=True,limit_mm_per_prompt={"image":1,"video":0},
   logprobs_mode="raw_logprobs",max_logprobs=128,seed=0,disable_log_stats=False,
   skip_mm_profiling=True,generation_config="vllm",**extra)

 def generate_ten(self,sample,eos):
  from vllm import SamplingParams
  im,prompt,base,inp=self.prepare(sample);expanded=inp["input_ids"][0].tolist()
  base=engine_base_ids(self.model,base,expanded,self.proc.tokenizer.bos_token_id)
  requests=[{"prompt_token_ids":list(base),"multi_modal_data":{"image":im},
             "multi_modal_uuids":{"image":sample["id"]}} for _ in range(10)]
  params=[SamplingParams(**parameters(self.model,sample["id"],i,eos)) for i in range(10)]
  started=time.perf_counter()
  # Complete one genuine replicate, then reuse its prompt cache for the other nine.
  results=self.llm.generate(requests[:1],params[:1],use_tqdm=False)
  results+=self.llm.generate(requests[1:],params[1:],use_tqdm=False)
  elapsed=time.perf_counter()-started
  if len(results)!=10:raise ValueError("Engine omitted independent attempts")
  if self.mt=="gemma3" and not getattr(self,"attention_audit",None):
   types=inp.get("token_type_ids")
   if types is None:raise ValueError("Native Gemma token_type_ids required")
   self.attention_audit=check_gemma_attention_receipt(self.audit_path,image_ranges(types[0].tolist()),os.environ["CUDA_VISIBLE_DEVICES"])
   atomic_json(self.audit_path.with_name("gemma_attention_range_check.json"),self.attention_audit)
  rows=[]
  for replicate,result in enumerate(results):
   if result.prompt_token_ids!=expanded or len(result.outputs)!=1:raise ValueError("Engine prompt expansion/request multiplicity changed")
   completion=result.outputs[0];tokens=list(completion.token_ids)
   if not 1<=len(tokens)<=32 or any(tok in eos for tok in tokens[:-1]):raise ValueError("Native EOS/token budget mismatch")
   if completion.logprobs is None or len(completion.logprobs)!=len(tokens):raise ValueError("Missing actual token log probabilities")
   lp=[float(values[tok].logprob) for tok,values in zip(tokens,completion.logprobs)]
   if not all(math.isfinite(x) for x in lp):raise ValueError("Nonfinite selected token log probability")
   terminated=tokens[-1] in eos
   if not terminated and len(tokens)!=32:raise ValueError("Unexpected non-native early stop")
   rows.append({**task(sample,replicate),"status":"ok","model":self.model,"prompt":prompt,
     "reference_prompt":None,"neutral_prompt":None,"config":asdict(CONFIG),
     "seed":stable_seed(sample["id"],self.model,replicate),"text":self.proc.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False),
     "tokens":tokens,"selected_log_probabilities":lp,"sequence_log_probability":float(sum(lp)),
     "first_probability":math.exp(lp[0]),"terminated":terminated,
     "trace":[{"token":tok,"log_probability":value,"sampling_log_probability":value,"weight":0.0,"active":False,"layer":None} for tok,value in zip(tokens,lp)],
     "wall_s":elapsed,"timing_scope":"wall time of one genuine attempt plus nine parallel attempts sharing its prompt cache; do not sum as per-attempt compute time",
     "engine_finish_reason":completion.finish_reason,"engine_stop_reason":completion.stop_reason,
     "num_cached_tokens":getattr(result,"num_cached_tokens",0) or 0,
     "requested_prompt_tokens":len(result.prompt_token_ids),"sampling_backend":"vllm_separate_cohort_not_numpy_draw_equivalence"})
  return rows


def validate_closed(record_dir,model,samples):
 record=within(ROOT,record_dir);complete_path=record/"complete.json"
 if not complete_path.is_file():raise ValueError("Closed cohort is not complete; do not enqueue independent work")
 done=json.loads(complete_path.read_text())
 if done.get("complete") is not True or done.get("closed")!=4848:raise ValueError("Whole4848 closed cohort completion required")
 ident=done["identity"]
 if ident["model"]!=model or ident["schema"]!="kdm_vllm_tree_closed_v4" or ident["manifest_sha256"]!=file_hash(ROOT/"data/current/all.jsonl"):
  raise ValueError("Wrong closed model/cohort/manifest")
 progress=json.loads((record/"progress.json").read_text());raw=within(ROOT,progress["output"])
 if progress["status"]!="complete" or file_hash(raw)!=done["output_sha256"]:raise ValueError("Closed results incomplete or changed")
 side=json.loads(raw.with_suffix(".identity.json").read_text())
 if side["definition"]!=ident or side["identity"]!=stable_hash(ident):raise ValueError("Closed identity mismatch")
 bench_path=record/"benchmark.json";bench=json.loads(bench_path.read_text())
 if file_hash(bench_path)!=ident["benchmark_sha256"] or not bench.get("input_parity") or not bench.get("finite_101_class_scores") or not bench.get("top1_parity_on_reference") or not math.isfinite(bench["speedup"]) or bench["speedup"]<=0:
  raise ValueError("Closed input/score/performance benchmark gate missing")
 expected={s["id"]:s for s in samples};seen=set();names=set(json.loads((ROOT/"configs/kdm/food_aliases.json").read_text()))
 for row in read_jsonl(raw):
  sid=row["sample"]["id"];scores=row["candidate_scores"]
  if sid in seen or row["sample"]!=expected.get(sid) or row["identity"]!=side["identity"]:raise ValueError("Closed row coverage or identity mismatch")
  if len(scores)!=101 or {x["label"] for x in scores}!=names or any(not math.isfinite(x[k]) for x in scores for k in ("sum_logp","mean_logp")):
   raise ValueError("Missing/nonfinite closed class scores")
  seen.add(sid)
 if seen!=set(expected):raise ValueError("Closed cohort omitted questions")
 return {"record_directory":str(record.relative_to(ROOT)),"complete_sha256":file_hash(complete_path),"raw_sha256":done["output_sha256"],"identity":ident}


def validate_independent_proof(model,native_path,check_path,sc):
 """Only accept concrete native16/actual-engine evidence produced before launch."""
 native_path=within(ROOT,native_path);check_path=within(ROOT,check_path)
 native=json.loads(native_path.read_text());check=json.loads(check_path.read_text())
 if native.get("schema")!="kdm_native_independent_reference_v1" or native["model"]!=model:
  raise ValueError("Independent-prompt native reference missing; closed reference is insufficient")
 if check.get("schema")!="kdm_vllm_independent_reference_check_v1" or check["model"]!=model or check.get("passed") is not True:
  raise ValueError("Real independent engine verification missing")
 if check["native_reference_sha256"]!=file_hash(native_path) or check["implementation_sha256"]!=file_hash(Path(__file__)) or check["closed_preparer_sha256"]!=file_hash(Path(closed.__file__)):
  raise ValueError("Independent verification source/reference changed")
 versions={p:version(p) for p in ("vllm","torch","transformers")}
 if check["versions"]!=versions:raise ValueError("Verified engine versions differ")
 if native["manifest_sha256"]!=file_hash(ROOT/"data/current/all.jsonl"):raise ValueError("Reference manifest mismatch")
 fixed={s["id"] for s in read_jsonl(ROOT/"data/current/interface16.jsonl")}
 rows=native["rows"];checks=check["rows"]
 if len(rows)!=16 or {x["sample"]["id"] for x in rows}!=fixed or len(checks)!=16 or {x["sample_id"] for x in checks}!=fixed:
  raise ValueError("All original native16 samples required")
 eos=native["eos_token_ids"]
 parameters(model,"cpu-eos-check",0,eos)
 if sorted(check["eos_token_ids"])!=sorted(eos):raise ValueError("Verified native EOS differ")
 for row in rows:
  _,prompt,base,inp=sc.prepare(row["sample"])
  if prompt!=row["prompt"] or base!=row["unexpanded_prompt_ids"] or inp["input_ids"][0].tolist()!=row["expanded_prompt_ids"]:
   raise ValueError("Independent template/input token mismatch")
  if any(k not in inp or closed.summarize(inp[k])!=closed.core_summary(value) for k,value in row["processor_summary"].items()):
   raise ValueError("Native processor tensors changed")
 by_id={x["sample"]["id"]:x for x in rows}
 for row in checks:
  native_row=by_id[row["sample_id"]]
  if row["engine_greedy_tokens"]!=native_row["greedy_tokens"]:raise ValueError("Greedy token/EOS mismatch not admitted")
  probs=row["engine_greedy_selected_log_probabilities"];old=native_row["greedy_selected_log_probabilities"]
  if len(probs)!=len(old) or not probs or not all(math.isfinite(x) for x in probs+old):raise ValueError("Missing finite native/engine logprob comparison")
  error=max(abs(x-y) for x,y in zip(probs,old))
  if not math.isclose(error,row["max_abs_selected_logprob_error"],abs_tol=1e-12):raise ValueError("Logprob difference report inconsistent")
  if row.get("engine_expanded_prompt_ids_equal") is not True or not row.get("engine_processor_tensor_parity") or not all(v is True for v in row["engine_processor_tensor_parity"].values()):
   raise ValueError("Actual vLLM multimodal processor input parity missing")
  if row.get("ten_distinct_stable_seeds") is not True or row.get("native_eos_handling") is not True or row.get("finite_selected_logprobs") is not True:
   raise ValueError("Ten-request sampling/EOS evidence missing")
  if row["num_cached_tokens"]<=0 or row["ten_request_wall_s"]<=0 or not math.isfinite(row["ten_request_wall_s"]):raise ValueError("Real cache/throughput benchmark missing")
 if check.get("sampling_cohort")!="vllm_separate_from_numpy" or check.get("numerical_difference_review")!="accepted_for_separate_cohort":
  raise ValueError("Numerical changes need an explicit separate-cohort review record")
 if model=="gemma3_4b":
  audit=check.get("gemma_attention_audit")
  if not audit or audit.get("metadata_ranges_checked") is not True or audit.get("kernel_execution_claimed") is not False:
   raise ValueError("Independent benchmark lacks real Gemma metadata range audit")
  if audit.get("module_sha256")!=file_hash(ROOT/"workflows/acceleration_v4/gemma_audit_worker.py"):
   raise ValueError("Gemma attention-audit module changed since independent benchmark")
 source_spec=native["backend"]
 for item in source_spec["weights"]:
  if (Path(sc.path)/item["filename"]).stat().st_size!=item["size_bytes"]:raise ValueError("Checkpoint weight size mismatch")
 if file_hash(Path(sc.path)/"config.json")!=source_spec["model_config_sha256"]:raise ValueError("Checkpoint config changed")
 for name,digest in source_spec["processor"]["files"].items():
  if file_hash(Path(sc.path)/name)!=digest:raise ValueError("Processor source file changed")
 return native,{"native_reference_sha256":file_hash(native_path),"verification_sha256":file_hash(check_path),
                "versions":versions,"eos_token_ids":eos,"source_spec_sha256":stable_hash(source_spec)}


def worker_admission():
 sys.path.insert(0,str(ROOT/"workflows/food_closed_v3"))
 import admission
 import kdm.execution as execution
 cards=os.environ.get("CUDA_VISIBLE_DEVICES","").split(",")
 execution.local_host(ROOT,"6403")
 if len(cards)!=1 or cards[0] not in {"0","1"}:raise ValueError("6403 one authorized physical GPU required")
 owned=Path(os.readlink("/proc/self/fd/20"))
 if owned!=ROOT/"outputs/locks"/f"gpu_{cards[0]}.lock":raise ValueError("Approved worker GPU lock required")
 execution.validate_host(ROOT,cards)
 return {"host":"6403","physical_gpus":cards,"worker_sha256":file_hash(ROOT/"workflows/food_closed_v3/worker.sh"),"admission_sha256":file_hash(Path(admission.__file__))}


def execute(args,samples,sc,closed_proof,native,proof):
 resource=worker_admission()
 if not args.run or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}",args.run):raise ValueError("Fresh run name required")
 run=ROOT/"outputs/records/acceleration_v4"/args.run/args.model
 output=ROOT/"outputs/raw/acceleration_v4"/args.run/args.model/"independent.jsonl"
 if run.exists() or output.parent.exists():raise ValueError("Fresh run/output required; no automatic retry/resume")
 run.mkdir(parents=True,exist_ok=False)
 identity={"schema":"kdm_vllm_independent_food_v1","model":args.model,"checkpoint_path":sc.path,
  "source_blobs":native["source_blobs"],"manifest_sha256":file_hash(ROOT/"data/current/all.jsonl"),
  "panel_sha256":file_hash(ROOT/"workflows/acceleration_v4/panel.json"),
  "amendment_sha256":file_hash(ROOT/"docs/current/PROTOCOL_AMENDMENT_20260923_ACCELERATION.md"),
  "implementation_sha256":file_hash(Path(__file__)),"closed_preparer_sha256":file_hash(Path(closed.__file__)),
  "closed_completion":closed_proof,"independent_verification":proof,"execution":resource,
  "base_config":asdict(CONFIG),"sampling_seed_rule":"stable_seed(sample_id,model,replicate)",
  "sampling_backend":"vllm_separate_cohort_not_numpy_draw_equivalence","generation_config":"vllm",
  "dtype":"bfloat16","repeats":10,"splits":["dev","eval"],"eos_token_ids":native["eos_token_ids"],
  "prefix_cache":"same image UUID and exact prompt tokens across10 separately seeded n=1 requests; first genuine attempt then nine parallel requests",
  "model_engine_overrides":{"llava_bos":"prepend only when native expanded BOS is absent from template tokens",
     "gemma_attention_backend":"TRITON_ATTN" if sc.mt=="gemma3" else None,
     "gemma_audit_worker_sha256":file_hash(ROOT/"workflows/acceleration_v4/gemma_audit_worker.py") if sc.mt=="gemma3" else None}}
 ledger=Ledger(output,identity)
 progress={"model":args.model,"pid":os.getpid(),"status":"loading","independent":0,"expected":48480,"output":str(output.relative_to(ROOT)),"final_gt":False,"automatic_labels_complete":False}
 atomic_json(run/"identity.json",{"identity":ledger.identity,"definition":identity})
 atomic_json(run/"progress.json",progress)
 try:
  if sc.mt=="gemma3":sc.audit_path=run/"gemma_attention_ranges.json"
  if not hasattr(sc,"llm"):sc.load()
  progress["status"]="running"
  for sample in samples:
   rows=sc.generate_ten(sample,native["eos_token_ids"])
   for row in rows:
    key=task_id(args.model,task(sample,row["replicate"]));ledger.add(key,{**row,"key":key})
   progress.update(independent=len(ledger.keys),last_sample_id=sample["id"],updated_unix=time.time())
   atomic_json(run/"progress.json",progress)
  expected={task_id(args.model,task(s,i)) for s in samples for i in range(10)}
  if ledger.keys!=expected:raise ValueError("Incomplete sample/replicate coverage")
  atomic_json(run/"complete.json",{"generation_complete":True,"independent":48480,"identity":ledger.identity,
   "output_sha256":file_hash(output),"final_gt":False,"automatic_labels_complete":False})
  progress["status"]="generation_complete";atomic_json(run/"progress.json",progress)
 except BaseException as e:
  import traceback
  atomic_json(run/"error.json",{"error":type(e).__name__+": "+str(e),"traceback":traceback.format_exc(),"no_retry_performed":True})
  progress.update(status="failed",updated_unix=time.time());atomic_json(run/"progress.json",progress);raise


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model",required=True,choices=sorted(FIVE))
 p.add_argument("--closed-record");p.add_argument("--native-reference");p.add_argument("--verification")
 p.add_argument("--cpu-preflight",action="store_true");p.add_argument("--execute",action="store_true");p.add_argument("--run")
 a=p.parse_args();samples=food_samples()
 if not a.cpu_preflight and not a.execute:
  print(json.dumps({"model":a.model,"food_questions":4848,"independent_answers":48480,"panel_answers":242400,
   "gpu_loaded":False,"ready_to_enqueue":False,"required_real_evidence":["full4848_closed_completion","independent_prompt_native16_reference_and_native_EOS","actual_engine_multimodal_processor_parity","greedy_token_EOS_and_raw_logprob_comparison","ten_seed_native_EOS_cache_and_throughput_benchmark"],
   "sampling":parameters(a.model,samples[0]["id"],0,[0]),"sampling_eos_in_plan":"placeholder only; production requires loaded-native reference EOS","implementation_sha256":file_hash(Path(__file__))},indent=2));return
 if not all([a.closed_record,a.native_reference,a.verification]):p.error("All three real evidence paths are required before preflight/execution")
 closed_proof=validate_closed(a.closed_record,a.model,samples)
 sc=Generator(a.model)
 native,proof=validate_independent_proof(a.model,a.native_reference,a.verification,sc)
 if not a.execute:
  print(json.dumps({"cpu_preflight_passed":True,"gpu_loaded":False,"production_not_started":True,"proof":proof}));return
 execute(a,samples,sc,closed_proof,native,proof)

if __name__=="__main__":main()

