#!/usr/bin/env python3
"""Bounded real-input DoLa/SID author-operator comparison.

Free generation uses the active decoder; the old formula below is only a
same-prefix diagnostic tied to ede009c7. Old full answers stay immutable.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
import csv,json,os,sys,time,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(Path(__file__).parent)]
from kdm.decoding import DecodeConfig,distribution,generate
from kdm.execution import resolve_image_path
from kdm.frozen import load_contract
from kdm.io import atomic_json,file_hash,read_jsonl,stable_hash,stable_seed,within
from kdm.pipeline import make_backend,sessions,task_id,json_safe
from kdm.probability import log_normalize,contrast
from workflows.supplemental.remaining11.generate import validate_proofs

MODELS=('qwen25vl','qwen35_4b','llava16_mistral','minicpm26','gemma3_4b',
        'internvl35_8b','onevision','phi35','qwen3vl')
SID_MODELS=('qwen25vl','llava16_mistral','internvl35_8b','onevision','phi35','qwen3vl')
REVISION='author_greedy_core_20261004'


def now():return datetime.now(timezone.utc).isoformat()


def load_inputs(model,method):
    if model not in MODELS or method not in ('dola','sid','cda_visual','vcd'):
        raise ValueError('Unregistered closeout model/method')
    if method=='sid' and model not in SID_MODELS:raise ValueError('SID not admitted for this model')
    manifest,frozen=load_contract(ROOT)
    paths=('data/current/all.jsonl','configs/kdm/study.json','configs/kdm/method_plan.json',
           f'configs/runtime/{model}.json')
    hashes={p:file_hash(ROOT/p) for p in paths}
    if any(frozen['files'].get(p)!=v for p,v in hashes.items()):
        raise ValueError('Frozen input/runtime changed')
    spec=json.loads((ROOT/paths[-1]).read_text())
    proof=validate_proofs(ROOT,spec,model,[method],'formal',manifest,frozen)
    samples=[s for s in read_jsonl(ROOT/paths[0]) if s['dataset']=='food101' and s['split']=='eval']
    assert len(samples)==2424 and len({s['id'] for s in samples})==2424
    assert set(Counter(s['class'] for s in samples).values())=={24}
    listing=ROOT/'workflows/paper_core/attribution_representative101.csv'
    with listing.open() as f:representatives=list(csv.DictReader(f))
    by_id={s['id']:s for s in samples}
    selected=[by_id[r['sample_id']] for r in representatives]
    assert len(selected)==101 and len({s['class'] for s in selected})==101
    return selected,spec,{'frozen_input_sha256':hashes,'verified_method_proofs':proof,
                         'representative_list_sha256':file_hash(listing),
                         'original_contract_sha256':manifest['original_contract_sha256']}


def admit(spec,model,method,registry,cards,run_id,owner,admission_kind):
    """Existing exact runtime/lock/UUID/checkpoint admission; no fallbacks."""
    if admission_kind=='core':
        from native_baselines import admission
        return admission(spec,model,cards,registry)
    from workflows.supplemental.remaining11 import execution
    execution.REGISTRY=registry
    os.environ['KDM_SUPPLEMENTAL_DATASET']='food101'
    os.environ['KDM_SUPPLEMENTAL_METHOD']=method
    value=execution.validate_supplemental_runtime(ROOT,spec,model,cards,'formal',run_id,owner)
    return value['runtime_spec'],value['execution']


def historical_operator(main,reference,method):
    """Audit-only ede009c7 formula; never used to generate a new response."""
    p=log_normalize(main.logits)
    if method=='sid':return contrast(p,reference.logits,1.,.1)[0],None
    candidates={i:log_normalize(z) for i,z in main.early_raw.items()}
    def js(q):
        mid=np.logaddexp(p,q)-np.log(2.)
        return float(.5*(np.exp(p)*(p-mid)+np.exp(q)*(q-mid)).sum())
    layer=max(sorted(candidates),key=lambda i:js(candidates[i]))
    score=p-candidates[layer];score[p<p.max()+np.log(.1)]=-np.inf
    return log_normalize(score),layer


def task(sample,method):
    return {'sample':sample,'method':method,'kind':'native_unguided_author_core',
            'marker':'NONE','reference_marker':'NONE','guided':False,'reference_guided':False,
            'replicate':0,'implementation_revision':REVISION}


def load_history(path,model,method,selected):
    rows=list(read_jsonl(path));needed={s['id'] for s in selected}
    found={}
    for row in rows:
        if row['model']!=model or row['method']!=method:raise ValueError('History identity mismatch')
        sid=row['sample']['id']
        if sid in needed:
            if sid in found:raise ValueError('Duplicate historical sample')
            if row.get('guided') or row.get('reference_guided') or not row.get('tokens'):
                raise ValueError('Historical baseline is not unguided actual generation')
            if not row.get('historical_source'):raise ValueError('History requires original path/line provenance')
            found[sid]=row
    if set(found)!=needed:raise ValueError('Historical full answers missing for paired subset')
    return found


def execute(args):
    samples,spec,provenance=load_inputs(args.model,args.method)
    samples=samples[:args.count]
    cfg=DecodeConfig(method=args.method,alpha=.5 if args.method=='sid' else 1.)
    old=load_history(within(ROOT,args.history),args.model,args.method,samples)
    identity={'model':args.model,'method':args.method,'implementation_revision':REVISION,
              'config':asdict(cfg),'provenance':provenance,'sample_ids':[s['id'] for s in samples],
              'history_sha256':file_hash(within(ROOT,args.history)),
              'decoder_sha256':file_hash(ROOT/'src/kdm/decoding.py'),
              'runner_sha256':file_hash(Path(__file__)),
              'study_settings_retained':{'dola_candidate_layers':'registered_all_early_layers',
                  'repetition_penalty':1.,'sid_keep_visual_tokens':100,
                  'dtype':spec['kwargs'].get('dtype',spec.get('dtype'))},
              'official_commits':{'dola':'805230e57e63ca561cb759994681b122ff6f81f0',
                                  'sid':'127dd412fa6b61ab1c9babf6979ec4da98002438'}}
    if args.plan:return {'identity':identity,'actual_gpu_generation':False}
    gate=json.loads(within(ROOT,args.cleanup_gate).read_text())
    if not gate.get('passed') or gate.get('protected_scientific_source_changed') is not False:
        raise ValueError('Root cleanup receipt has not passed')
    actual,admitted=admit(spec,args.model,args.method,args.registry,args.physical_gpus.split(','),
                          args.run_id,args.owner,args.admission_kind)
    out=within(ROOT,args.output)
    out.mkdir(parents=True,exist_ok=True)
    claim=out/'claim.json'
    with claim.open('x') as f:
        json.dump({'identity':identity,'admission':admitted,'pid':os.getpid(),
                   'starttick':Path('/proc/self/stat').read_text().split()[21],
                   'owner':args.owner,'started_utc':now()},f)
    atomic_json(out/'identity.json',identity)
    from PIL import Image
    started=time.perf_counter();rows=[];prefix_rows=[]
    raw=out/'new_predictions.jsonl';prefix_path=out/'same_prefix.jsonl'
    if args.reuse_pilot:
        if args.count!=101:raise ValueError('Only the 101 subset may reuse its pilot')
        source=within(ROOT,args.reuse_pilot)
        source_identity=json.loads((source/'identity.json').read_text())
        receipt=json.loads((source/'complete.json').read_text())
        if (not receipt['passed'] or receipt['completed']!=8
                or source_identity['model']!=args.model or source_identity['method']!=args.method
                or source_identity['decoder_sha256']!=identity['decoder_sha256']
                or source_identity['config']!=identity['config']
                or source_identity['sample_ids']!=identity['sample_ids'][:8]
                or file_hash(source/'new_predictions.jsonl')!=receipt['raw_sha256']
                or file_hash(source/'same_prefix.jsonl')!=receipt['same_prefix_sha256']):
            raise ValueError('Pilot reuse identity or actual output differs')
        rows=list(read_jsonl(source/'new_predictions.jsonl'))
        prefix_rows=list(read_jsonl(source/'same_prefix.jsonl'))
        raw.write_bytes((source/'new_predictions.jsonl').read_bytes())
        prefix_path.write_bytes((source/'same_prefix.jsonl').read_bytes())
        atomic_json(out/'pilot_reuse.json',{'source':str(source),'receipt':receipt,
                    'reused_sample_ids':[r['sample']['id'] for r in rows]})
    completed={r['sample']['id'] for r in rows}
    backend=make_backend(actual,'cuda:0');load_s=time.perf_counter()-started
    for sample in samples:
        if sample['id'] in completed:continue
        start=time.perf_counter();seed=stable_seed(sample['id'],args.model,0)
        with Image.open(resolve_image_path(sample['image_path'],ROOT)) as im:image=im.convert('RGB')
        t=task(sample,args.method)
        main,ref,neutral,prompt,rprompt=sessions(backend,image,t,cfg,seed)
        result=generate(main,ref,cfg,backend.eos,backend.decode,seed,neutral)
        historical=old[sample['id']]
        row={**t,'model':args.model,'key':task_id(args.model,t),'config':asdict(cfg),'seed':seed,
             'prompt':prompt,'reference_prompt':rprompt if ref else None,**result,
             'generation_wall_s':time.perf_counter()-start,'decoder_sha256':identity['decoder_sha256'],
             'historical_source':historical['historical_source'],'old_text':historical['text'],
             'old_tokens':historical['tokens'],'tokens_equal':result['tokens']==historical['tokens'],
             'text_equal':result['text']==historical['text'],'score_status':'pending_exact_QA_or_semantic'}
        with raw.open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        rows.append(row)
        # Fresh sessions avoid nonmonotonic cache state from free generation.
        if args.method=='sid' and getattr(backend,'sid_control',None):
            backend.sid_control.close();backend.sid_control=None
        del main,ref,neutral
        main,ref,neutral,_,_=sessions(backend,image,t,cfg,seed)
        for pos in range(min(3,len(historical['tokens']))):
            prefix=tuple(historical['tokens'][:pos]);m=main.next(prefix);r=ref.next(prefix) if ref else None
            new_dist,meta=distribution(m,r,cfg,pos)
            old_dist,old_layer=historical_operator(m,r,args.method)
            old_token=int(np.argmax(old_dist));new_token=int(np.argmax(new_dist))
            rec={'model':args.model,'method':args.method,'sample_id':sample['id'],
                 'prefix_tokens':list(prefix),'position':pos,'historical_actual_token':historical['tokens'][pos],
                 'historical_formula_argmax_on_current_forward':old_token,'new_argmax':new_token,
                 'old_layer':old_layer,'new_layer':meta.get('layer'),'argmax_changed':old_token!=new_token,
                 'old_support':int(np.isfinite(old_dist).sum()),'new_support':int(np.isfinite(new_dist).sum()),
                 'historical_source':historical['historical_source'],
                 'scope':'same current forward; not proof of identical historical hardware forward'}
            with prefix_path.open('a') as f:f.write(json.dumps(json_safe(rec),allow_nan=False)+'\n')
            prefix_rows.append(rec)
        if args.method=='sid' and getattr(backend,'sid_control',None):
            backend.sid_control.close();backend.sid_control=None
        del main,ref,neutral
    expected={s['id'] for s in samples}
    if len(rows)!=len(expected) or {r['sample']['id'] for r in rows}!=expected:
        raise ValueError('Actual output is duplicated or missing representative inputs')
    if len({r['key'] for r in rows})!=len(rows):raise ValueError('Duplicate generated task keys')
    for row in rows:
        if (row['config']!=asdict(cfg) or row['seed']!=stable_seed(row['sample']['id'],args.model,0)
                or row['key']!=task_id(args.model,task(row['sample'],args.method))
                or row['status']!='ok' or not 0<len(row['tokens'])<=32
                or row['terminated']!=(row['tokens'][-1] in backend.eos)
                or len(row['tokens'])!=len(row['selected_log_probabilities'])
                or not all(math.isfinite(v) for v in row['selected_log_probabilities'])
                or not math.isfinite(row['first_probability'])):
            raise ValueError('Actual identity/seed/config/token/probability validation failed')
    receipt={'passed':True,'model':args.model,'method':args.method,'completed':len(rows),
             'model_load_wall_s':load_s,'total_wall_s':time.perf_counter()-started,
             'generation_wall_s':sum(r['generation_wall_s'] for r in rows),
             'changed_text':sum(not r['text_equal'] for r in rows),
             'changed_tokens':sum(not r['tokens_equal'] for r in rows),
             'same_prefix_positions':len(prefix_rows),'raw_sha256':file_hash(raw),
             'same_prefix_sha256':file_hash(prefix_path),'completed_utc':now(),
             'score_status':'pending','actual_gpu_generation':True,
             'scope':'fixed representative subset; not full-eval equivalence'}
    atomic_json(out/'complete.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--method',choices=('dola','sid'),required=True)
    p.add_argument('--count',type=int,choices=(8,101),default=8)
    p.add_argument('--history',required=True);p.add_argument('--registry',required=True)
    p.add_argument('--admission-kind',choices=('core','supplemental'),required=True)
    p.add_argument('--physical-gpus',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--output',required=True);p.add_argument('--cleanup-gate',required=True)
    p.add_argument('--owner',default='/root/baseline_fix_pilot');p.add_argument('--plan',action='store_true')
    p.add_argument('--reuse-pilot')
    print(json.dumps(execute(p.parse_args()),ensure_ascii=False,indent=2,allow_nan=False))
