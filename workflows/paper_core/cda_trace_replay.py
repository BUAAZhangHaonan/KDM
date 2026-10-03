#!/usr/bin/env python3
"""Measure five real CDA branches along a bounded frozen response prefix."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(Path(__file__).parent)]
from kdm.cda import cda_calibration,entropy
from kdm.execution import resolve_image_path
from kdm.io import atomic_json,file_hash,read_jsonl,stable_seed,within
from kdm.pipeline import make_backend,sessions,json_safe
from kdm.decoding import DecodeConfig
from kdm.probability import log_normalize
from kdm.prompts import task_prompt
from workflows.paper_core.author_baseline_pilot import load_inputs,admit,MODELS

def now():return datetime.now(timezone.utc).isoformat()

def tensor_evidence(value):
    import torch
    if isinstance(value,torch.Tensor):
        actual=value.detach().contiguous().cpu()
        result={'shape':list(actual.shape),'dtype':str(actual.dtype),
                'sha256':hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest()}
        if not actual.is_floating_point() and actual.numel()<=100000:
            result['values']=actual.tolist()
        return result
    if isinstance(value,dict):return {str(k):tensor_evidence(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [tensor_evidence(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'Unrecorded input type: {type(value)}')

def execute(a):
    _,spec,proof=load_inputs(a.model,'cda_visual')
    history=within(ROOT,a.history)
    rows=list(read_jsonl(history))
    assert len(rows)==12 and len({r['sample']['id'] for r in rows})==12
    assert all(r['model']==a.model and r['method']=='cda_visual' and r['marker']=='UNKNOWN' for r in rows)
    gate=json.loads(within(ROOT,a.cleanup_gate).read_text())
    assert gate['passed'] and gate['protected_scientific_source_changed'] is False
    identity={'model':a.model,'scope':'bounded_current_teacher_forcing_of_frozen_CDA_paths',
              'historical_paths':len(rows),'history_sha256':file_hash(history),'proof':proof,
              'runner_sha256':file_hash(Path(__file__)),'cda_sha256':file_hash(ROOT/'src/kdm/cda.py'),
              'precision':spec['dtype'],'max_GPU_seconds':a.max_gpu_seconds}
    if a.plan:return identity
    cards=a.physical_gpus.split(',')
    actual,admitted=admit(spec,a.model,'cda_visual',a.registry,cards,a.run_id,a.owner,a.admission_kind)
    out=within(ROOT,a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(out/'identity.json',identity)
    atomic_json(out/'claim.json',{'pid':os.getpid(),'starttick':Path('/proc/self/stat').read_text().split()[21],
               'owner':a.owner,'admission':admitted,'started_utc':now()})
    from PIL import Image
    started=time.perf_counter();backend=make_backend(actual,'cuda:0')
    load_s=time.perf_counter()-started
    completed=[];events=[]
    for original in rows:
        if (time.perf_counter()-started)*len(cards)>=a.max_gpu_seconds:break
        sample=original['sample'];sid=sample['id'];seed=stable_seed(sid,a.model,original['replicate'])
        assert seed==original['seed']
        imagepath=resolve_image_path(sample['image_path'],ROOT)
        with Image.open(imagepath) as im:image=im.convert('RGB')
        cfg=DecodeConfig(**original['config'])
        main,unused_ref,unused_neutral,prompt,_=sessions(backend,image,original,cfg,seed)
        assert unused_ref is None and unused_neutral is None
        plain=task_prompt(sample['question'],guided=False)
        null=task_prompt('[N/A]',guided=False)
        branches=[backend.session(image,plain,reference='text_only'),backend.session(image,plain),main,
                  backend.session(image,null,reference='text_only'),
                  backend.session(Image.new('RGB',image.size,(127,127,127)),null)]
        names=['prior','context','abstention','null_prior','null_context']
        evidence={name:tensor_evidence(session.inputs) for name,session in zip(names,branches)}
        stem=hashlib.sha256(sid.encode()).hexdigest()[:20]
        atomic_json(out/f'{stem}.inputs.json',{'sample_id':sid,'seed':seed,'image_path':str(imagepath),
                    'image_sha256':file_hash(imagepath),'image_size':list(image.size),
                    'prompts':dict(zip(names,[plain,plain,prompt,null,null])),
                    'actual_processed_inputs':evidence,'admission':admitted,
                    'prefix_policy':'HFSession monotonic cache; same historical token prefix for five independent sessions',
                    'historical_source':original['historical_source'],'old_trace':original.get('trace'),
                    'historical_config':original['config'],
                    'frozen_decision':original['frozen_decision'],'historical_tokens':original['tokens']})
        steps=[];mixtures=[];mixture_logps=[];path_events=[]
        for pos,tok in enumerate(original['tokens']):
            prefix=tuple(original['tokens'][:pos])
            logits=[s.next(prefix).logits for s in branches]
            assert all(np.isfinite(z).all() for z in logits)
            cal=cda_calibration(logits[0],logits[1],logits[3],logits[4])
            weights=np.array([cal['wp'],cal['wc'],cal['wa']],float)
            mixture=sum(w*z for w,z in zip(weights,logits[:3]))
            lp=log_normalize(mixture)
            assert np.isfinite(mixture).all() and np.isfinite(lp).all()
            # Save all five original arrays without conversion or vocabulary truncation.
            states=np.stack(logits)
            steps.append(states)
            mixtures.append(mixture);mixture_logps.append(lp)
            oldtrace=original.get('trace',[])
            event={'model':a.model,'sample_id':sid,'position':pos,'prefix_tokens':list(prefix),
                   'historical_token':tok,'current_argmax':int(np.argmax(lp)),
                   'historical_argmax_agrees':int(np.argmax(lp))==tok,
                   'all_five_entropies':dict(zip(names,map(entropy,logits))),**cal,
                   'mixture_logp_observed':float(lp[tok]),'mixture_argmax_logp':float(lp.max()),
                   'weights_sum_residual':float(abs(weights.sum()-1)),
                   'frozen_behavior':original['frozen_decision']['state'],
                   'frozen_reference':original['frozen_decision']['uniform_reference'],
                   'historical_weight_max_abs_difference':float(np.max(np.abs(weights-np.array(oldtrace[pos]['weights'])))) if pos<len(oldtrace) and 'weights' in oldtrace[pos] else None}
            path_events.append(event)
        arraypath=out/f'{stem}.branches.npz'
        np.savez_compressed(arraypath,logits=np.stack(steps),mixture_logits=np.stack(mixtures),
                            mixture_logprob=np.stack(mixture_logps),tokens=np.array(original['tokens'],dtype=np.int64))
        # Independent recomputation from the written, full-vocabulary artifact.
        with np.load(arraypath,allow_pickle=False) as archive:
            loaded={name:archive[name] for name in archive.files}
        max_residual=0.
        for pos,z in enumerate(loaded['logits']):
            e={name:entropy(v) for name,v in zip(names,z)}
            rp=max(e['prior']-e['null_prior'],0)/e['null_prior']
            rc=max(e['context']-e['null_context'],0)/e['null_context']
            wp,wc=(rp*rp/(rp+rc),rc*rc/(rp+rc)) if rp+rc>0 else (0.,0.)
            w=np.array([wp,wc,1-wp-wc])
            mixed=w[0]*z[0]+w[1]*z[1]+w[2]*z[2]
            saved=path_events[pos]
            residual=max(abs(rp-saved['rp']),abs(rc-saved['rc']),abs(wp-saved['wp']),abs(wc-saved['wc']),
                         float(np.max(np.abs(mixed-loaded['mixture_logits'][pos]))),
                         float(np.max(np.abs(log_normalize(mixed)-loaded['mixture_logprob'][pos]))),
                         abs(float(log_normalize(mixed)[original['tokens'][pos]])-saved['mixture_logp_observed']))
            max_residual=max(max_residual,residual)
        assert max_residual<=1e-10
        for event in path_events:
            event.update(branches_file=str(arraypath),eq4_6_7_saved_array_recompute_max_residual=max_residual)
            with (out/'events.jsonl').open('a') as f:f.write(json.dumps(json_safe(event),allow_nan=False)+'\n')
        completed.append({'sample_id':sid,'tokens':len(path_events),'branches_file':str(arraypath),
                          'branches_sha256':file_hash(arraypath),'input_evidence_sha256':file_hash(out/f'{stem}.inputs.json'),
                          'eq4_6_7_residual':max_residual})
        events.extend(path_events)
        atomic_json(out/'progress.json',{'completed':len(completed),'GPU_seconds':(time.perf_counter()-started)*len(cards)})
        del branches,main,logits,steps,states,loaded,mixtures,mixture_logps
    used=(time.perf_counter()-started)*len(cards)
    result={'passed':True,'complete':len(completed)==len(rows),'model':a.model,'actual_paths':len(completed),
            'actual_positions':len(events),'historical_token_agreement':sum(e['historical_argmax_agrees'] for e in events),
            'historical_token_disagreement':sum(not e['historical_argmax_agrees'] for e in events),
            'model_load_seconds':load_s,'actual_GPU_seconds':used,'budget_GPU_seconds':a.max_gpu_seconds,
            'remaining_paths':len(rows)-len(completed),'paths':completed,'completed_utc':now(),
            'scope':'current replay of selected frozen prefixes; original generation and scoring unchanged',
            'events_sha256':file_hash(out/'events.jsonl') if events else None}
    atomic_json(out/'complete.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--history',required=True);p.add_argument('--output',required=True)
    p.add_argument('--registry',required=True);p.add_argument('--physical-gpus',required=True)
    p.add_argument('--admission-kind',choices=('core','supplemental'),required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--cleanup-gate',required=True)
    p.add_argument('--max-gpu-seconds',type=float,default=800)
    p.add_argument('--owner',default='/root/cleanup_verified');p.add_argument('--plan',action='store_true')
    print(json.dumps(execute(p.parse_args()),ensure_ascii=False,allow_nan=False))
