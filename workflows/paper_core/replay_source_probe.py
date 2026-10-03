#!/usr/bin/env python3
"""Record two actual session schedules at the nine historical mismatch positions."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(Path(__file__).parent)]
from kdm.io import atomic_json,file_hash,read_jsonl,within,stable_seed
from kdm.execution import resolve_image_path
from kdm.pipeline import make_backend,json_safe
from kdm.decoding import DecodeConfig
from kdm.prompts import task_prompt
from kdm.probability import contrast,instruction_preserving
from workflows.paper_core.author_baseline_pilot import load_inputs,admit
from workflows.paper_core.cda_trace_replay import tensor_evidence,now

def finite(value):return float(value) if np.isfinite(value) else None

def execute(a):
    _,spec,proof=load_inputs(a.model,'vcd')
    source=within(ROOT,a.history);paths=list(read_jsonl(source))
    assert len(paths)=={'llava16_mistral':2,'minicpm26':3,'qwen35_4b':1}[a.model]
    assert all(row['model']==a.model for row in paths)
    gate=json.loads(within(ROOT,a.cleanup_gate).read_text())
    assert gate['passed'] and gate['protected_scientific_source_changed'] is False
    identity={'model':a.model,'history_sha256':file_hash(source),'proof':proof,
              'scope':'current controlled schedule/hardware measurement; historical unknown fields remain unknown',
              'actual_hardware_role':a.hardware_role,'runner_sha256':file_hash(Path(__file__)),
              'pipeline_sha256':file_hash(ROOT/'src/kdm/pipeline.py'),
              'HFSession_sha256':file_hash(ROOT/'src/kdm/models/hf.py'),
              'max_gpu_seconds':a.max_gpu_seconds}
    if a.plan:return identity
    cards=a.physical_gpus.split(',')
    actual,admitted=admit(spec,a.model,'vcd',a.registry,cards,a.run_id,a.owner,'core')
    out=within(ROOT,a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    atomic_json(out/'identity.json',identity)
    atomic_json(out/'claim.json',{'pid':os.getpid(),'starttick':Path('/proc/self/stat').read_text().split()[21],
                                'owner':a.owner,'admission':admitted,'started_utc':now()})
    from PIL import Image
    started=time.perf_counter();backend=make_backend(actual,'cuda:0');load_s=time.perf_counter()-started
    results=[];completed=[]
    for original in paths:
        sid=original['sample']['id'];seed=original['seed']
        assert seed==stable_seed(sid,a.model,original['replicate'])
        question=original['sample']['question']
        plain=task_prompt(question,guided=False)
        gprompt=task_prompt(question,original['marker'],True)
        hprompt=task_prompt(question,original['reference_marker'],True)
        cfg=DecodeConfig(**original['config'])
        imagepath=resolve_image_path(original['sample']['image_path'],ROOT)
        with Image.open(imagepath) as im:image=im.convert('RGB')
        for schedule in ('historical','four_view'):
            if (time.perf_counter()-started)*len(cards)>=a.max_gpu_seconds:break
            order=('g','h') if original['replay_kind']=='guided' else ('g','r','c')
            if schedule=='four_view':order=('c','r','g','h')
            definitions={'g':(gprompt,'clean'),'h':(hprompt,'noise'),'c':(plain,'clean'),'r':(plain,'noise')}
            branches={name:backend.session(image,definitions[name][0],reference=definitions[name][1],seed=seed) for name in order}
            stem=hashlib.sha256((sid+schedule).encode()).hexdigest()[:20]
            atomic_json(out/f'{stem}.inputs.json',{'sample_id':sid,'schedule':schedule,'creation_order':list(order),
                    'forward_order':list(order),'prompts':{name:definitions[name][0] for name in order},
                    'processed_inputs':{name:tensor_evidence(branches[name].inputs) for name in order},
                    'seed':seed,'noise_timestep':500,'image_path':str(imagepath),'image_sha256':file_hash(imagepath),
                    'historical_source':original['historical_source'],'historical_config':original['config'],
                    'admission':admitted,'cache_policy':'separate HFSession caches advanced through the same saved tokens',
                    'historical_reply':original['text'],'historical_tokens':original['tokens']})
            saved={};events=[]
            for pos in range(max(original['target_positions'])+1):
                prefix=tuple(original['tokens'][:pos])
                logits={name:branches[name].next(prefix).logits for name in order}
                assert all(np.isfinite(value).all() for value in logits.values())
                if pos not in original['target_positions']:continue
                if original['replay_kind']=='guided':score,support=contrast(logits['g'],logits['h'],cfg.alpha,cfg.beta)
                else:score,support=instruction_preserving(logits['g'],logits['c'],logits['r'],cfg.alpha,cfg.beta)
                tok=original['tokens'][pos];argmax=int(np.argmax(score))
                audit=next(row for row in original['previous_replay_audit'] if int(row['position'])==pos)
                event={'model':a.model,'sample_id':sid,'position':pos,'prefix_tokens':list(prefix),
                       'method':original['method'],'replay_kind':original['replay_kind'],'schedule':schedule,
                       'hardware_role':a.hardware_role,'historical_token':tok,'current_argmax':argmax,
                       'current_matches_historical':argmax==tok,'historical_token_text':backend.decode([tok]),
                       'current_argmax_text':backend.decode([argmax]),'historical_in_current_support':bool(support[tok]),
                       'current_support_count':int(support.sum()),'current_historical_logp':finite(score[tok]),
                       'current_argmax_logp':finite(score[argmax]),'current_gap':finite(score[argmax]-score[tok]),
                       'old_replay_argmax':int(audit['replay_argmax']),'old_replay_gap':float(audit['gap']),
                       'historical_source':original['historical_source']}
                events.append(event)
                for name,value in logits.items():saved[f'{pos}_{name}']=value
                saved[f'{pos}_score']=score;saved[f'{pos}_support']=support
            arraypath=out/f'{stem}.distributions.npz';np.savez_compressed(arraypath,**saved)
            with np.load(arraypath,allow_pickle=False) as loaded:
                assert all(np.array_equal(value,loaded[name]) for name,value in saved.items())
            for event in events:
                event['distribution_file']=str(arraypath)
                with (out/'events.jsonl').open('a') as f:f.write(json.dumps(json_safe(event),allow_nan=False)+'\n')
            results.extend(events);completed.append({'sample_id':sid,'schedule':schedule,'positions':len(events),
                    'distribution_sha256':file_hash(arraypath),'input_evidence_sha256':file_hash(out/f'{stem}.inputs.json')})
            del branches,logits,saved
    receipt={'passed':True,'complete':len(completed)==len(paths)*2,'model':a.model,
             'hardware_role':a.hardware_role,'actual_paths':len(completed),'actual_positions':len(results),
             'historical_token_agreements':sum(row['current_matches_historical'] for row in results),
             'model_load_seconds':load_s,'actual_GPU_seconds':(time.perf_counter()-started)*len(cards),
             'budget_GPU_seconds':a.max_gpu_seconds,'paths':completed,'completed_utc':now(),
             'historical_unknown_inputs_recovered':False,'events_sha256':file_hash(out/'events.jsonl') if results else None}
    atomic_json(out/'complete.json',receipt)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('llava16_mistral','minicpm26','qwen35_4b'),required=True)
    p.add_argument('--history',required=True);p.add_argument('--output',required=True);p.add_argument('--registry',required=True)
    p.add_argument('--physical-gpus',required=True);p.add_argument('--run-id',required=True);p.add_argument('--cleanup-gate',required=True)
    p.add_argument('--hardware-role',choices=('original_3090','previous_replay_hardware'),required=True)
    p.add_argument('--max-gpu-seconds',type=float,default=600);p.add_argument('--owner',default='/root/cleanup_verified')
    p.add_argument('--plan',action='store_true')
    print(json.dumps(execute(p.parse_args()),ensure_ascii=False,allow_nan=False))
