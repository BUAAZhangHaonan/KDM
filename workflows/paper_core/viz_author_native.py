"""Run original DoLa/DeCo on fixed Viz512, sealing eight then 64-input pieces."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import argparse,json,math,os,socket,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(Path(__file__).parent)]
from kdm.io import atomic_json,file_hash,read_jsonl,stable_hash,within
from kdm.pipeline import make_backend,run_tasks,task_id
from kdm.decoding import DecodeConfig
from kdm.prompts import task_prompt
from kdm.execution import resolve_image_path
from kdm.frozen import load_contract
from workflows.supplemental.remaining11.generate import validate_proofs
from author_baseline_pilot import load_inputs
from native_baselines import admission
from workflows.paper_core.dev_viz import roster

MODELS=('qwen25vl','qwen35_4b','llava16_mistral','minicpm26','gemma3_4b','internvl35_8b','onevision','phi35','qwen3vl')
CORE=set(MODELS[:5])
REVISION='native_DoLa_DeCo_Viz512_20261004'
K100_GATE='outputs/supplemental/remaining4/intern_k100_single_admission_launch2_20261003_1355/admission/k100_intern_single_actual_duplicate8_launch2_20261003_1355/condition_gate.json'
K100_GATE_SHA='1ebd535c708a202be45f951bc9057e903b34263c77c126e271a2eb758a754336'
def now():return datetime.now(timezone.utc).isoformat()

def native_task(sample,method):
    return dict(sample=sample,method=method,kind='native_unguided_author_core',marker='NONE',
        reference_marker='NONE',guided=False,reference_guided=False,replicate=0,
        implementation_revision=REVISION)

def assigned_tasks(args,samples):
    all_tasks={task_id(args.model,native_task(s,m)):native_task(s,m)
               for m in ('dola','deco') for s in samples}
    missing=None;source=None
    if args.missing_keys:
        source=json.loads(within(ROOT,args.source_seal).read_text())
        assert source['passed'] and source['producer_exited'] and source['model']==args.model
        path=within(ROOT,args.missing_keys)
        assert file_hash(path)==source['remaining_keys_sha256']
        rows=list(read_jsonl(path));missing={r['key'] for r in rows}
        assert len(rows)==len(missing)==source['remaining'] and missing<=all_tasks.keys()
        assert not missing & set(source['completed_keys'])
        for row in rows:
            task=all_tasks[row['key']]
            assert row['model']==args.model and row['method']==task['method'] and row['sample_id']==task['sample']['id']
    result={method:[] for method in args.methods}
    for method in args.methods:
        for sample in samples:
            task=native_task(sample,method);key=task_id(args.model,task)
            if missing is not None and key not in missing:continue
            partition_key=[args.partition_salt,key] if args.partition_salt else key
            if int(stable_hash(partition_key)[:8],16)%args.parts!=args.part:continue
            result[method].append(task)
    return result,source

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--methods',nargs='+',choices=('dola','deco'),default=['dola','deco'])
    p.add_argument('--output',required=True);p.add_argument('--physical-gpus',required=True)
    p.add_argument('--registry',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--owner',default='/root');p.add_argument('--plan',action='store_true')
    p.add_argument('--supplemental-location',action='store_true')
    p.add_argument('--k100-intern',action='store_true')
    p.add_argument('--missing-keys');p.add_argument('--source-seal')
    p.add_argument('--parts',type=int,default=1);p.add_argument('--part',type=int,default=0)
    p.add_argument('--partition-salt',default='')
    args=p.parse_args()
    if not 0<=args.part<args.parts:p.error('Invalid mutually exclusive partition')
    if bool(args.missing_keys)!=bool(args.source_seal):p.error('A remaining-key list requires its exact sealed-source receipt')
    if args.parts>1 and not args.missing_keys:p.error('Partition only actual sealed remaining keys')
    proofs={}
    for method in args.methods:
        _,spec,proofs[method]=load_inputs(args.model,'vcd' if method=='deco' else method)
        if method=='deco':
            manifest,frozen=load_contract(ROOT)
            proofs[method]['deco_projection_proof']=validate_proofs(ROOT,spec,args.model,['deco'],'formal',manifest,frozen)
    listing,samples=roster('viz512');assert len(samples)==512
    assignments,source_seal=assigned_tasks(args,samples)
    total=sum(len(tasks) for tasks in assignments.values())
    assert total>0
    if args.plan:
        print(json.dumps(dict(model=args.model,methods=args.methods,missing=total,
          keys=[task_id(args.model,t) for ts in assignments.values() for t in ts],
          roster_sha256=file_hash(listing),dtype=spec['dtype'],proofs=proofs)));return
    cards=args.physical_gpus.split(',')
    if args.model in CORE and not args.supplemental_location:
        actual,admitted=admission(spec,args.model,cards,args.registry)
    else:
        from workflows.supplemental.remaining11 import execution
        execution.REGISTRY=args.registry
        os.environ['KDM_SUPPLEMENTAL_DATASET']='vizwiz'
        os.environ['KDM_SUPPLEMENTAL_METHOD']=args.methods[0]
        if args.k100_intern:
            from workflows.supplemental.remaining4.k100_intern_registered_matrix import single_spec
            path=ROOT/K100_GATE;gate=json.loads(path.read_text())
            assert args.model=='internvl35_8b' and cards==['0']
            assert file_hash(path)==K100_GATE_SHA and gate['passed'] and gate['production_allowed'] and gate['completed']==8
            assert file_hash(ROOT/'workflows/supplemental/remaining4/internvl_k100_single.py')==gate['factory_sha256']
            assert file_hash(ROOT/gate['operator_audit_path'])==gate['operator_audit_sha256']
            a=execution.validate_supplemental_runtime(ROOT,single_spec(spec),args.model,cards,'independent',args.run_id,args.owner)
            a['execution']['reused_hardware_gate']=dict(path=K100_GATE,sha256=K100_GATE_SHA,
                scope='existing single-map checkpoint/software/load-capacity admission only',
                DoLa_DeCo_operator_admitted_by_prior_gate=False,
                native_projection_proof=proofs,new_actual_layer_audit_required=True)
        else:
            a=execution.validate_supplemental_runtime(ROOT,spec,args.model,cards,'formal',args.run_id,args.owner)
        actual,admitted=a['runtime_spec'],a['execution']
    assert actual['dtype']==spec['dtype']
    out=within(ROOT,args.output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'claim.json',dict(pid=os.getpid(),starttick=Path('/proc/self/stat').read_text().split()[21],
        host=socket.gethostname(),physical_gpus=cards,model=args.model,methods=args.methods,
        expected=total,owner=args.owner,started_utc=now(),
        planned_keys=[task_id(args.model,t) for ts in assignments.values() for t in ts],
        source_seal=args.source_seal,source_seal_sha256=file_hash(within(ROOT,args.source_seal)) if args.source_seal else None,
        partition=dict(parts=args.parts,part=args.part,salt=args.partition_salt)))
    started=time.perf_counter();completed=0
    try:
        backend=make_backend(actual,'cuda:0');load_s=time.perf_counter()-started
        for method in args.methods:
            method_tasks=assignments[method]
            if not method_tasks:continue
            cfg=DecodeConfig(method=method)
            assert cfg.max_tokens==32 and cfg.temperature==0 and cfg.top_p==1
            ranges=[(0,min(8,len(method_tasks)))]+[(i,min(i+64,len(method_tasks))) for i in range(8,len(method_tasks),64)]
            method_s=0.;method_n=0
            for start,stop in ranges:
                tasks=method_tasks[start:stop]
                selected=[t['sample'] for t in tasks]
                part=out/f'{method}_{start:03d}_{stop:03d}';part.mkdir(exist_ok=False)
                identity=dict(model=args.model,method=method,revision=REVISION,runtime_spec=actual,
                    runtime_admission=admitted,config=asdict(cfg),sample_ids=[s['id'] for s in selected],
                    roster_sha256=file_hash(listing),proof=proofs[method],
                    decoder_sha256=file_hash(ROOT/'src/kdm/decoding.py'),runner_sha256=file_hash(Path(__file__)),
                    pilot=start==0,pilot_in_full_denominator=True,full_expected_n=512,
                    source_seal_sha256=file_hash(within(ROOT,args.source_seal)) if args.source_seal else None,
                    assigned_method_n=len(method_tasks),partition=dict(parts=args.parts,part=args.part,salt=args.partition_salt))
                atomic_json(part/'identity.json',identity)
                if start==0:
                    from PIL import Image
                    with Image.open(resolve_image_path(selected[0]['image_path'],ROOT)) as im: image=im.convert('RGB')
                    session=backend.session(image,task_prompt(selected[0]['question'],guided=False),need_layers=True)
                    step=session.next(())
                    assert step.early_raw and all(np.isfinite(v).all() and v.shape==step.logits.shape for v in step.early_raw.values())
                    atomic_json(part/'layer_audit.json',dict(sample_id=selected[0]['id'],
                        layer_indices=list(step.early_raw),vocabulary_size=len(step.logits),
                        all_layer_projections_finite=True,actual_need_layers=True,
                        decoder_sha256=identity['decoder_sha256'],configured_operator=asdict(cfg)))
                    del session,step
                assert all(t['guided'] is False and t['reference_guided'] is False
                    and t['marker']==t['reference_marker']=='NONE' for t in tasks)
                raw=part/'new_predictions.jsonl'
                run_tasks(backend,args.model,tasks,raw,identity,cfg)
                rows=list(read_jsonl(raw));assert len(rows)==len(selected)==len({r['key'] for r in rows})
                assert {r['sample']['id'] for r in rows}==set(identity['sample_ids'])
                for row in rows:
                    assert row['status']=='ok' and 0<len(row['tokens'])<=32
                    assert len(row['tokens'])==len(row['selected_log_probabilities'])
                    assert all(math.isfinite(v) for v in row['selected_log_probabilities'])
                    assert row['config']==asdict(cfg) and row['prompt']==task_prompt(row['sample']['question'],guided=False)
                seconds=sum(r['wall_s'] for r in rows);method_s+=seconds;method_n+=len(rows);completed+=len(rows)
                atomic_json(part/'complete.json',dict(passed=True,model=args.model,method=method,
                    completed=len(rows),expected=len(rows),raw_sha256=file_hash(raw),
                    mean_sample_s=seconds/len(rows),pilot=start==0,completed_utc=now(),
                    identity_sha256=file_hash(part/'identity.json'),full_eval_source_denominator=512))
                progress=dict(model=args.model,method=method,method_completed=method_n,method_expected=len(method_tasks),
                    completed=completed,expected=total,mean_sample_s=method_s/method_n,
                    method_eta_s=(len(method_tasks)-method_n)*method_s/method_n,updated_utc=now(),load_s=load_s)
                atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        atomic_json(out/'complete.json',dict(passed=True,completed=completed,expected=total,
            elapsed_s=time.perf_counter()-started,completed_utc=now(),sealed_parts=True))
    except BaseException as error:
        atomic_json(out/'failure.json',dict(error=repr(error),completed=completed,at_utc=now(),automatic_retry=False))
        raise
if __name__=='__main__':main()
