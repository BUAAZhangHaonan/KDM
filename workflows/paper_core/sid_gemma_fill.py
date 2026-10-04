"""New native SID admission and missing Food eval on unchanged Gemma3 runtime."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(Path(__file__).parent)]
from kdm.io import atomic_json, file_hash, read_jsonl, stable_seed, stable_hash
from kdm.execution import resolve_image_path
from kdm.pipeline import make_backend, sessions, task_id
from kdm.decoding import DecodeConfig, generate
from author_baseline_pilot import load_inputs
from native_baselines import admission

def now(): return datetime.now(timezone.utc).isoformat()

def task(s):
    return dict(sample=s,method='sid',kind='native_unguided_author_core',marker='NONE',
                reference_marker='NONE',guided=False,reference_guided=False,replicate=0,
                implementation_revision='gemma_native_mask_sid_20261004')

def run(a):
    representatives,spec,provenance=load_inputs('gemma3_4b','vcd')
    # The inherited proof covers checkpoint/native-forward identity only.
    # SID receives the actual, separate attention admission below.
    actual,admitted=admission(spec,'gemma3_4b',a.physical_gpus.split(','),a.registry)
    all_samples=[s for s in read_jsonl(ROOT/'data/current/all.jsonl')
                 if s['dataset']=='food101' and s['split']=='eval']
    pilot_ids=[s['id'] for s in representatives[:8]]
    cfg=DecodeConfig(method='sid',alpha=.5)
    if a.pilot:
        samples=representatives[:8]
    else:
        prior=ROOT/a.admission
        receipt=json.loads((prior/'complete.json').read_text())
        identity=json.loads((prior/'identity.json').read_text())
        if (not receipt['passed'] or receipt['completed']!=8 or
            identity['sid_adapter_sha256']!=file_hash(ROOT/'src/kdm/models/sid.py') or
            file_hash(prior/'new_predictions.jsonl')!=receipt['raw_sha256']):
            raise ValueError('The new SID 8-input admission differs or is incomplete')
        if a.sample_list:
            selected=json.loads((ROOT/a.sample_list).read_text())
            if selected['schema']!='gemma_sid_remaining_v1' or selected['model']!='gemma3_4b':
                raise ValueError('Invalid source-bound remaining-key manifest')
            ids=selected['sample_ids']
            if len(ids)!=len(set(ids)) or set(ids)&set(selected['excluded_completed_ids']):
                raise ValueError('Remaining-key manifest contains duplicate or completed inputs')
            for source in selected['seal_receipts']:
                path=ROOT/source['path']
                if file_hash(path)!=source['sha256'] or not json.loads(path.read_text())['passed']:
                    raise ValueError('Remaining-key source seal changed')
            available={s['id']:s for s in all_samples}
            samples=[available[sid] for sid in ids]
            if set(ids)&set(pilot_ids):raise ValueError('Already completed pilot requested')
        else:
            samples=[s for s in all_samples if s['id'] not in pilot_ids and
                     int(stable_hash(s['id'])[:8],16)%a.parts==a.part]
    assert len(all_samples)==2424 and len({s['id'] for s in all_samples})==2424
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False)
    identity={'model':'gemma3_4b','method':'sid','config':asdict(cfg),
              'sample_ids':[s['id'] for s in samples], 'pilot':a.pilot,
              'source_inputs':provenance,'runtime_admission':admitted,
              'sid_adapter_sha256':file_hash(ROOT/'src/kdm/models/sid.py'),
              'runner_sha256':file_hash(Path(__file__)),
              'official_sid_commit':'127dd412fa6b61ab1c9babf6979ec4da98002438',
              'new_sid_mechanism_admission':'actual 8-input attention and native-mask checks',
              'mask_adaptation':'preserve native sliding/full and bidirectional-image mask; block unselected image columns',
              'sid_fixed':{'aggregation_block_1based':2,'rank':100,'alpha':.5,'greedy_support':'full_vocab'},
              'partition':{'part':a.part,'parts':a.parts,'excluded_pilot_ids':[] if a.pilot else pilot_ids}}
    if a.sample_list:
        identity['partition']['remaining_manifest']={'path':a.sample_list,'sha256':file_hash(ROOT/a.sample_list)}
    atomic_json(out/'identity.json',identity)
    atomic_json(out/'claim.json',dict(pid=os.getpid(),starttick=Path('/proc/self/stat').read_text().split()[21],
                                     owner='/root/sid_gemma_qwen35',started_utc=now(),identity_sha256=file_hash(out/'identity.json')))
    started=time.perf_counter();backend=make_backend(actual,'cuda:0');load_s=time.perf_counter()-started
    rows=[];audit_rows=[];torch=backend.torch
    from PIL import Image
    for index,s in enumerate(samples):
        started_one=time.perf_counter()
        with Image.open(resolve_image_path(s['image_path'],ROOT)) as im:image=im.convert('RGB')
        seed=stable_seed(s['id'],'gemma3_4b',0);t=task(s)
        main,ref,neutral,prompt,rprompt=sessions(backend,image,t,cfg,seed)
        input_ids=ref.inputs['input_ids'];control=ref.control
        checks=[]
        if a.pilot:
            if not torch.equal(main.inputs['input_ids'],input_ids):raise ValueError('Branch prompt tokens differ')
            if not torch.equal(main.inputs['pixel_values'],ref.inputs['pixel_values']):raise ValueError('SID must use identical pixels')
            def audit(event):
                attention=event['attention'];mask=event['mask'];original=event['original_mask']
                if original is None:raise ValueError('Gemma native mask was not materialized')
                if not torch.isfinite(attention).all():raise ValueError('Attention is not finite')
                # Literal official selection: mean heads, final query, bottom-rank 100.
                oracle=attention.mean(dim=0)[-1,event['start']:event['start']+event['length']].topk(100,largest=False).indices.sort().values
                if not torch.equal(oracle,event['selected']):raise ValueError('Official attention selection mismatch')
                blocked=torch.zeros(mask.shape[-1],device=mask.device,dtype=torch.bool)
                blocked[event['start']:event['start']+event['length']]=True
                blocked[event['start']+oracle]=False
                if not torch.isneginf(mask[...,blocked]).all():raise ValueError('Unselected image keys leaked')
                if not torch.equal(mask[...,~blocked],original[...,~blocked]):raise ValueError('Native mask was altered outside dropped visuals')
                checks.append({'layer':event['layer'],'query_length':event['query_length'],
                               'kv_length':event['kv_length'],'visual_start':event['start'],
                               'visual_length':event['length'],'selected':oracle.cpu().tolist(),
                               'attention_finite':True,'official_selection_equal':True,
                               'native_mask_retained':True,'blocked_count':int(blocked.sum()),
                               'native_mask_sha256':hashlib.sha256(original.detach().float().cpu().numpy().tobytes()).hexdigest()})
            control.audit=audit
        before=None
        if a.pilot and index==0:before=main.next(()).logits.copy()
        result=generate(main,ref,cfg,backend.eos,backend.decode,seed,neutral)
        if a.pilot:
            if len(checks)!=32*len(result['tokens']):raise ValueError('Not every downstream layer/token was audited')
            if getattr(backend,'_sid_active_control',None):raise ValueError('SID control leaked')
            if any(layer._forward_pre_hooks for layer in control.layers):raise ValueError('SID hooks leaked')
            if before is not None:
                clean=backend.session(image,prompt).next(()).logits
                if not np.array_equal(clean,before):raise ValueError('Clean branch changed after SID')
        if result['status']!='ok' or not 0<len(result['tokens'])<=32:raise ValueError('Invalid generation completion')
        if not np.isfinite(result['selected_log_probabilities']).all():raise ValueError('Nonfinite selected probabilities')
        row={**t,'model':'gemma3_4b','key':task_id('gemma3_4b',t),'seed':seed,'prompt':prompt,
             'reference_prompt':rprompt,'config':asdict(cfg),**result,
             'generation_wall_s':time.perf_counter()-started_one,
             'prompt_token_ids':input_ids.cpu().tolist(),'visual_start':control.start,
             'visual_length':control.length,'sid_adapter_sha256':identity['sid_adapter_sha256'],
             'score_status':'pending_exact_QA_or_semantic'}
        with (out/'new_predictions.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        if a.pilot:
            audit_row={'sample_id':s['id'],'steps':len(result['tokens']),'checks':checks}
            with (out/'attention_audit.jsonl').open('a') as f:f.write(json.dumps(audit_row)+'\n')
        rows.append(row)
        mean=float(np.mean([r['generation_wall_s'] for r in rows]))
        progress={'completed':len(rows),'expected':len(samples),'remaining':len(samples)-len(rows),
                  'mean_sample_s':mean,'eta_s':mean*(len(samples)-len(rows)),
                  'updated_utc':now(),'actual_gpu_generation':True}
        atomic_json(out/'progress.json',progress)
        if a.pilot or len(rows)%100==0:print(json.dumps(progress),flush=True)
        del main,ref,neutral
    receipt={'passed':True,'completed':len(rows),'expected':len(samples),
             'raw_sha256':file_hash(out/'new_predictions.jsonl'),
             'unique_keys':len({r['key'] for r in rows}),
             'mean_sample_s':float(np.mean([r['generation_wall_s'] for r in rows])),
             'elapsed_s':time.perf_counter()-started,'load_s':load_s,'completed_utc':now(),
             'pilot':a.pilot,'score_status':'pending','full_eval_source_denominator':2424}
    if a.pilot:receipt['attention_audit_sha256']=file_hash(out/'attention_audit.jsonl')
    atomic_json(out/'complete.json',receipt)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--physical-gpus',required=True);p.add_argument('--registry',default='workflows/paper_core/host_registry.json')
    p.add_argument('--pilot',action='store_true');p.add_argument('--admission')
    p.add_argument('--parts',type=int,default=1);p.add_argument('--part',type=int,default=0)
    p.add_argument('--sample-list')
    a=p.parse_args()
    if not a.pilot and not a.admission:p.error('Production requires its actual 8-input admission')
    if not 0<=a.part<a.parts:p.error('Invalid exclusive partition')
    run(a)
