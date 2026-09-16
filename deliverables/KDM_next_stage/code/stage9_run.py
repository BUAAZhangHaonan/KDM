"""Stage-9 visual evidence and fixed-reference reachability experiments.

Uses the existing S6Model implementation without rewriting published decoders.
Only q9b and llava16 are in the new inference matrix. Other models are retained
in the full stage-6 paired analysis. No full vocabulary arrays are persisted.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time,sys
from pathlib import Path
import numpy as np
from PIL import Image
from stage9_common import (ROOT,ALLOWED_GPUS,initialize,inside,read_jsonl,
                          load_records,write_json,stable_key)
from stage9_geometry import Interval,target_interval,confidence_terms

MODEL_PATHS={'q9b':'/home/g203-4028/Models/Qwen3.5-9B',
             'llava16':'/home/g203-4028/Models/llava-v1.6-mistral-7b-hf'}


def image_condition(image:Image.Image,condition:str)->Image.Image:
    image=image.convert('RGB')
    if condition=='original':return image.copy()
    if condition not in ('short64','short32'):raise ValueError(condition)
    short=int(condition[5:]);w,h=image.size
    # Prevent an accidental upsample masquerading as a degradation.
    if min(w,h)<short:raise ValueError('Input image is smaller than the frozen target resolution.')
    ratio=short/min(w,h);size=(max(1,round(w*ratio)),max(1,round(h*ratio)))
    return image.resize(size,Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.BICUBIC)


def noise_seed(file:str)->int:
    # Exactly the stage-6 seed definition, shared across evidence conditions.
    return int(hashlib.sha256(f'{file}:vcd'.encode()).hexdigest(),16)%(2**31)


def clean_record(record:dict)->dict:
    return {k:v for k,v in record.items() if k!='first_probs'}


def decode(em,inputs,prior,method:str,seed:int,t0:int):
    if method=='direct':return clean_record(em.decode_single(inputs,12,method='direct'))
    if method=='vcd':
        return clean_record(em.decode_two_branch(inputs,em.noised_inputs(inputs,seed),'vcd',12))
    if method=='m3id':
        return clean_record(em.decode_two_branch(inputs,prior,'m3id',12,t0_sched=t0))
    raise ValueError(method)


def target_paths(tokenizer,class_name:str,eos:int)->list[tuple[str,list[int]]]:
    phrase=class_name.replace('_',' ')
    texts=[phrase,phrase[:1].upper()+phrase[1:]]
    paths=[];seen=set()
    for text in texts:
        ids=list(tokenizer.encode(text,add_special_tokens=False))+[eos]
        if tuple(ids) in seen:continue
        seen.add(tuple(ids));paths.append((text,ids))
    return paths


def probe_sequence(em,inputs,reference,ids:list[int],validate=True)->dict:
    """Teacher forcing determines prefix-specific affine inequalities exactly.
    Validation uses the SAME fixed reference and a label-assisted alpha witness;
    it is a diagnostic, not a deployable decoder or a reported method baseline.
    """
    import torch
    if len(ids)>12:return dict(status='exceeds_frozen_generation_budget',length=len(ids))
    interval=Interval();steps=[]
    with torch.inference_mode():
        a=em._fwd(inputs=inputs);b=em._fwd(inputs=reference)
        for pos,token in enumerate(ids):
            z=a.logits[0,-1].float().detach().cpu().numpy().astype(np.float64)
            r=b.logits[0,-1].float().detach().cpu().numpy().astype(np.float64)
            one=target_interval(z,r,token,.1)
            interval=interval.intersect(one)
            # Record the original winner too; no correctness enters this identity.
            winner=int(z.argmax());terms=confidence_terms(z,r,winner,1.0,.1)
            if terms.get('identity_error',0)>1e-8:raise AssertionError('Log-probability identity failed.')
            steps.append(dict(position=pos,target_token=token,original_winner=winner,
                              target_interval=one.to_dict(),winner_confidence=terms))
            if interval.empty:break
            if pos+1<len(ids):
                a=em._fwd(tok=int(token),pkv=a.past_key_values)
                b=em._fwd(tok=int(token),pkv=b.past_key_values)
        witness=interval.witness(4.0)
        validation=None
        if validate and witness is not None:
            out=em.decode_two_branch(inputs,reference,'vcd',12,alpha_override=witness)
            got=list(out['tokens'])
            validation=(got==ids)
            if not validation:
                raise AssertionError(f'Analytic witness mismatch: alpha={witness}, target={ids}, actual={got}')
    return dict(status='ok',target_tokens=ids,interval=interval.to_dict(),
                feasible_default=interval.contains(1.0),feasible_up_to_two=not interval.intersect(Interval(0,2,True,True)).empty,
                feasible_any_nonnegative=not interval.empty,
                validation_alpha=witness,validation_passed=validation,steps=steps)


def run(root:Path,model:str,em,prompt:str,score_fn,shard=0,shards=1,
        conditions=('original','short64','short32'),methods=('direct','vcd','m3id'),
        use_legacy=True,limit=0):
    if model not in MODEL_PATHS:raise ValueError(model)
    if not 0<=shard<shards:raise ValueError('Invalid shard.')
    selected=[r for r in read_jsonl(root/'data/stage9/manifest.jsonl') if r['model']==model]
    selected.sort(key=lambda r:r['file'])
    selected=[r for i,r in enumerate(selected) if i%shards==shard]
    if limit:selected=selected[:limit]
    if not selected:raise ValueError('No selected images.')
    all_classes=sorted({r['class'] for r in read_jsonl(root/'data/samples_manifest.jsonl')})
    outdir=inside(root,root/'outputs/raw/stage9');outdir.mkdir(parents=True,exist_ok=True)
    path=outdir/f'{model}_evidence_{shard}.jsonl'
    gpath=outdir/f'{model}_geometry_{shard}.jsonl'
    done={r['key'] for r in read_jsonl(path)} if path.exists() else set()
    gdone={r['key'] for r in read_jsonl(gpath)} if gpath.exists() else set()
    legacy={}
    if use_legacy:
        lp=root/'outputs/raw'/f'{model}_s6_eval_naming.jsonl'
        for r in load_records(lp).values():legacy[r['file'],r['method']]=r
    consistency_checks=0
    if use_legacy:
        for sample in selected[:2]:
            old=legacy.get((sample['file'],'direct'))
            if old is None:raise KeyError('Missing direct record for model compatibility check.')
            image=Image.open(sample['image_path']).convert('RGB')
            check=em.decode_single(em.build(image,prompt),12,method='direct')
            if list(check['tokens'])!=list(old['tokens']):
                raise RuntimeError(f"Direct token mismatch on {sample['file']}; preserve source and stop.")
            consistency_checks+=1
    t0=len(em.tokenizer(prompt,add_special_tokens=False)['input_ids'])
    eos=em.tokenizer.eos_token_id
    if eos not in em.eos:raise ValueError('Tokenizer termination token is not in the existing decoder stop set.')
    started=time.time();counts={'new_generations':0,'reused_generations':0,'paths':0,'witness_checks':0,'direct_consistency_checks':consistency_checks}
    def append(p,r):
        with p.open('a',encoding='utf-8') as f:
            f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
    for sample in selected:
        original=Image.open(sample['image_path']).convert('RGB')
        for condition in conditions:
            image=image_condition(original,condition)
            inputs=em.build(image,prompt)
            prior=em.build_text_only(prompt) if 'm3id' in methods else None
            for method in methods:
                key=f"{model}:{condition}:{sample['file']}:{method}"
                if key in done:continue
                if condition=='original' and use_legacy:
                    if (sample['file'],method) not in legacy:raise KeyError(f'Missing legacy record {key}')
                    raw=clean_record(legacy[sample['file'],method]);counts['reused_generations']+=1
                    if raw.get('prompt')!=prompt:raise ValueError('Frozen prompt differs from source record.')
                    origin='stage6_cache'
                else:
                    raw=decode(em,inputs,prior,method,noise_seed(sample['file']),t0)
                    raw['outcome']=score_fn(raw['text'],sample['class'],all_classes)
                    counts['new_generations']+=1;origin='stage9_inference'
                rec={**raw,'key':key,'model':model,'condition':condition,'method':method,
                     'class':sample['class'],'file':sample['file'],'stratum':sample['stratum'],
                     'task':'naming','prompt':prompt,'source':origin,'abstained':False}
                append(path,rec);done.add(key)
            if sample.get('geometry') and condition in ('original','short32'):
                ref=em.noised_inputs(inputs,noise_seed(sample['file']))
                for name,ids in target_paths(em.tokenizer,sample['class'],eos):
                    key=f"{model}:{condition}:{sample['file']}:{','.join(map(str,ids))}"
                    if key in gdone:continue
                    result=probe_sequence(em,inputs,ref,ids)
                    result.update(key=key,model=model,condition=condition,file=sample['file'],
                                  stratum=sample['stratum'],class_name=sample['class'],
                                  target_name=name,target_scope='exact_token_sequence_with_specified_termination')
                    append(gpath,result);gdone.add(key);counts['paths']+=1
                    counts['witness_checks']+=result.get('validation_passed') is True
    counts.update(seconds=time.time()-started,selected=len(selected),model=model,shard=shard,shards=shards)
    write_json(root/'outputs/tables/stage9'/f'{model}_run_{shard}.json',counts)
    return counts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--model',choices=tuple(MODEL_PATHS),required=True)
    ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--shards',type=int,default=1);ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args()
    if a.gpu not in ALLOWED_GPUS:raise SystemExit('Only physical GPUs 0,1,4,5 are authorized.')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    root=initialize(a.root);sys.path.insert(0,str(root/'code'))
    from stage6_engine import S6Model
    import prompts,scoring,torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for real inference.')
    em=S6Model(MODEL_PATHS[a.model],'cuda:0')
    result=run(root,a.model,em,prompts.STYLES['style1'][-1],scoring.score_naming,
               a.shard,a.shards,limit=a.limit)
    result['physical_gpu']=a.gpu
    result['peak_allocated_MiB']=torch.cuda.max_memory_allocated()/1024**2
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
