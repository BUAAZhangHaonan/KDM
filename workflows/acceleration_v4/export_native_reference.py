#!/usr/bin/env python3
"""Four-sample native HF reference; frozen methods and inputs are unchanged."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"workflows/food_closed_v3"))
from runner import MemoBackend, load_inputs
from kdm.pipeline import make_backend, closed_rank
from kdm.execution import resolve_image_path
from kdm.io import atomic_json, file_hash
from kdm.protocol import validate_freeze, validate_runtime
from PIL import Image
INDICES=[0,1616,3232,4847]
OUT=ROOT/"outputs/records/acceleration_v4/native_reference"
def now():return datetime.now(timezone.utc).isoformat()
def tensor_summary(v):
    import torch
    if torch.is_tensor(v):
        x=v.detach().cpu().contiguous()
        d={"shape":list(x.shape),"dtype":str(x.dtype),"sha256":hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}
        if x.numel()<=2048 and not x.is_floating_point():d["values"]=x.tolist()
        if x.numel() and x.is_floating_point():
            y=x.float();d.update(min=float(y.min()),max=float(y.max()),mean=float(y.mean()))
        return d
    if isinstance(v,dict):return {str(k):tensor_summary(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [tensor_summary(x) for x in v]
    return v if v is None or isinstance(v,(str,int,float,bool)) else {"type":type(v).__name__}
def unexpanded(backend,image,prompt):
    em=backend.em
    if em.mt=="minicpmv":
        body="(<image>./</image>)"+chr(10)+prompt
        text=em.proc.tokenizer.apply_chat_template([{"role":"user","content":body}],tokenize=False,add_generation_prompt=True)
    else:
        msgs=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}]
        kw={"tokenize":False,"add_generation_prompt":True}
        if em.mt in ("qwen3_5","qwen3_vl","glm4v"):kw["enable_thinking"]=False
        text=em.proc.apply_chat_template(msgs,**kw)
    return text,backend.tokenizer.encode(text,add_special_tokens=False)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",required=True);a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/(a.model+".json");progress=OUT/(a.model+".progress.json")
    if dest.exists() or progress.exists():raise RuntimeError("Fresh reference required; no overwrite or retry")
    p={"model":a.model,"pid":os.getpid(),"status":"loading","indices":INDICES,"completed":0,"started_utc":now(),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")}
    atomic_json(progress,p)
    try:
        spec_path=ROOT/"configs/runtime"/(a.model+".json");spec=json.loads(spec_path.read_text())
        freeze=validate_freeze(ROOT)
        execution=validate_runtime(ROOT,spec,a.model,os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        samples,names=load_inputs();food=[s for s in samples if s["dataset"]=="food101"]
        backend=make_backend(spec,"cuda:0")
        candidate_ids={n:backend.encode(n.replace("_"," ")) for n in names}
        result={"schema":"kdm_native_reference_v4","model":a.model,"indices":INDICES,"backend":spec,"backend_spec_sha256":file_hash(spec_path),"manifest_sha256":file_hash(ROOT/"data/current/all.jsonl"),"source_blobs":freeze["source_blobs"],"execution":execution,"candidate_token_ids":candidate_ids,"workflow_sha256":file_hash(Path(__file__)),"rows":[]}
        p["status"]="running";atomic_json(progress,p)
        for idx in INDICES:
            sample=food[idx]
            with Image.open(resolve_image_path(sample["image_path"],ROOT)) as im:image=im.convert("RGB")
            memo=MemoBackend(backend);start=time.perf_counter()
            row=closed_rank(memo,image,sample["question"],names,sample["class"])
            inputs=memo.last_session.session.inputs
            ids=inputs.get("input_ids")
            input_ids=ids.detach().cpu().tolist() if hasattr(ids,"detach") else None
            text,unexpanded_ids=unexpanded(backend,image,row["prompt"])
            result["rows"].append({"food_index":idx,"reference":{"sample":sample,"wall_s":time.perf_counter()-start,**row},"unexpanded_prompt_text":text,"unexpanded_prompt_ids":unexpanded_ids,"expanded_prompt_ids":input_ids[0] if input_ids else None,"candidates_tokens":candidate_ids,"processor_summary":tensor_summary(inputs),"memo_hits":memo.last_session.hits,"memo_misses":memo.last_session.misses})
            del memo
            atomic_json(OUT/(a.model+".partial.json"),result)
            p.update(completed=len(result["rows"]),last_sample_id=sample["id"],last_update_utc=now());atomic_json(progress,p)
        result.update(status="complete",finished_utc=now())
        atomic_json(dest,result);p.update(status="complete",finished_utc=now());atomic_json(progress,p)
    except BaseException as e:
        err={"model":a.model,"error":type(e).__name__+": "+str(e),"traceback":traceback.format_exc(),"at_utc":now(),"completed":p["completed"]}
        atomic_json(OUT/(a.model+".error.json"),err);p.update(status="failed",error=err["error"],last_update_utc=now());atomic_json(progress,p);raise
if __name__=="__main__":main()
