#!/usr/bin/env python3
"""Budget-released IP-only dev404 missing keys through the unchanged native chain."""
import argparse
from collections import defaultdict
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows.paper_core.ip_dev_pilot import CORE,EXTRA,METHODS,tasks_for,audit,write
from workflows.paper_core.dev_viz import roster,condition,now
from workflows.paper_core.native_audit import input_description
from workflows.paper_core import execution as core
from workflows.supplemental.remaining11 import execution as supplemental
from workflows.supplemental.remaining11.generate import validate_proofs
from kdm.frozen import load_contract
from kdm.decoding import DecodeConfig
from kdm.io import file_hash,read_jsonl,within,stable_hash
from kdm.pipeline import make_backend,run_tasks,task_id
from kdm.prompts import MARKERS

class BudgetInputs:
    def __init__(self,backend,started,gpus,cap,path):
        self.backend,self.started,self.gpus,self.cap,self.path=backend,started,gpus,cap,path
        self.current=None;self.capture=False
    def __getattr__(self,name):return getattr(self.backend,name)
    def session(self,image,prompt,reference='clean',seed=0,need_layers=False):
        if (time.perf_counter()-self.started)*self.gpus>=self.cap:
            raise RuntimeError('Allocated actual GPU-time budget exhausted; preserve partial and stop without retry')
        session=self.backend.session(image,prompt,reference=reference,seed=seed,need_layers=need_layers)
        if self.capture:
            with self.path.open('a')as stream:
                stream.write(json.dumps(dict(sample_id=self.current['sample']['id'],condition=condition(self.current),
                  prompt=prompt,reference=reference,seed=seed,inputs=input_description(session.inputs)),allow_nan=False)+'\n')
        return session

def k100_scope(args,spec,budget):
    if args.model!='onevision' or args.methods!=['instruction_m3id']:
        raise ValueError('K100 OneVision scope excludes all noise methods')
    pilot=within(ROOT,args.k100_dev_pilot_root)
    receipt=json.loads((pilot/'pilot_receipt.json').read_text())
    identity=json.loads((pilot/'identity.json').read_text())
    plan=identity['plan']
    matching=[s for s in budget['sources']if s['source_receipt']==str((pilot/'pilot_receipt.json').relative_to(ROOT))]
    if len(matching)!=1 or not receipt['passed'] or receipt['actual_rows']!=24 or file_hash(pilot/'pilot_receipt.json')!=matching[0]['source_receipt_sha256']:
        raise ValueError('The exact completed dev24 gate must be reused')
    if set(args.markers)!=set(plan['markers']) or plan['methods']!=args.methods or plan['registry_sha256']!=file_hash(within(ROOT,args.host_registry)):
        raise ValueError('Independent successful dev marker/runtime scope differs')
    for marker,source in plan['inherited_marker_software_gates'].items():
        path=within(ROOT,source['path']);gate=json.loads(path.read_text());scope=gate['software_compatibility']
        if file_hash(path)!=source['sha256'] or not gate['passed'] or not gate['production_allowed'] or gate['differences'] or scope['actual_method_scope']!='instruction_m3id' or scope['dataset_scope']!='food101' or scope['marker_scope']!=[marker]:
            raise ValueError('Original same-marker Food software gate differs')
        if file_hash(within(ROOT,gate['operator_audit_path']))!=gate['operator_audit_sha256']:
            raise ValueError('Original software operator proof differs')
    for c in receipt['conditions']:
        raw=within(ROOT,c['raw']);operator=raw.with_suffix('.operator.json')
        if file_hash(raw)!=c['raw_sha256'] or file_hash(operator)!=c['operator_audit_sha256']:
            raise ValueError('The completed original-input dev8 operator gate changed')
        actual=json.loads(operator.read_text())
        if not actual['passed'] or not actual['three_original_routes_per_input'] or actual['noise_called']:
            raise ValueError('The original three-route IP operator failed')
    original=supplemental.runtime_spec
    def runtime(root,frozen,model):
        actual=original(root,frozen,model);actual['versions']=copy.deepcopy(frozen['versions'])
        actual['versions'].update(torch='2.9.0+cu128',torchvision='0.24.0+cu128');return actual
    supplemental.runtime_spec=runtime
    return dict(exact_reused_dev24_receipt_sha256=file_hash(pilot/'pilot_receipt.json'),
      inherited_marker_software_gates=plan['inherited_marker_software_gates'],original_cpu_fixture_sha256=plan['original_CPU_fixture_sha256'],
      new_dev_original_GPU_same_answer_comparison_available=False,scientific_parameters_changed=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('plan','full'),required=True)
    p.add_argument('--model',choices=CORE+EXTRA,required=True)
    p.add_argument('--methods',nargs='+',choices=METHODS,required=True)
    p.add_argument('--markers',nargs='+',choices=MARKERS,default=list(MARKERS))
    for key in ('run-id','claim-id','owner','host-registry','missing-keys','budget-release','budget-release-sha256'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--allocated-gpu-seconds',type=float,required=True)
    p.add_argument('--physical-gpus',required=True)
    p.add_argument('--k100-dev-pilot-root')
    args=p.parse_args()
    import re
    if not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',v)for v in (args.run_id,args.claim_id,args.owner)):
        raise ValueError('Invalid exclusive claim identity')
    budget_path=within(ROOT,args.budget_release);budget=json.loads(budget_path.read_text())
    if file_hash(budget_path)!=args.budget_release_sha256 or not budget['released'] or budget['estimated_total_GPU_hours']>budget['authorized_total_GPU_hours_cap'] or budget['authorized_total_GPU_hours_cap']!=24 or budget['actual_unique_pilot_rows']!=416:
        raise ValueError('Actual nine-model pilot and authorized 24 GPU-hour release required')
    if args.model in CORE and args.methods!=['instruction_m3id']:
        raise ValueError('Completed core IP-VCD must not be rerun')
    path,samples=roster('dev404');expected={task_id(args.model,t):t for t in tasks_for(samples,args.methods,args.markers)}
    missing=list(read_jsonl(within(ROOT,args.missing_keys)));keys=[r['key']for r in missing]
    if not keys or len(keys)!=len(set(keys)) or not set(keys)<=expected.keys() or set(keys).intersection(budget['reused_pilot_keys']):
        raise ValueError('Manifest has duplicate, foreign or already completed pilot keys')
    tasks=[expected[k]for k in keys]
    if any(r.get('model',args.model)!=args.model or r.get('sample_id',t['sample']['id'])!=t['sample']['id']for r,t in zip(missing,tasks)):
        raise ValueError('Explicit missing-key scientific identity differs')
    cards=args.physical_gpus.split(',')
    if not 0<args.allocated_gpu_seconds<=24*3600-budget['actual_pilot_gpu_seconds_including_load']:
        raise ValueError('Invalid per-claim allocation')
    spec=json.loads((ROOT/f'configs/runtime/{args.model}.json').read_text());manifest,freeze=load_contract(ROOT)
    proofs=validate_proofs(ROOT,spec,args.model,args.methods,'formal',manifest,freeze)
    source=None
    if args.model in EXTRA:
        from workflows.supplemental.remaining4.native import source_inputs
        _,original,_,source=source_inputs(args.model,'m3id','food101')
        if spec!=original:raise ValueError('Original native frozen runtime differs')
    kscope=k100_scope(args,spec,budget)if args.k100_dev_pilot_root else None
    base=ROOT/'outputs/paper_core_20261002_dev_viz'/args.run_id/args.claim_id
    plan=dict(schema='kdm_IP_only_dev404_budgeted_missing_claim_v1',model=args.model,claim_id=args.claim_id,owner=args.owner,
       methods=args.methods,markers=args.markers,stage='dev404',expected_rows=len(keys),sample_roster_sha256=file_hash(path),
       missing_keys_path=args.missing_keys,missing_keys_sha256=file_hash(within(ROOT,args.missing_keys)),
       original_spec=spec,proofs=proofs,frozen_native_source=source,registry=args.host_registry,
       registry_sha256=file_hash(within(ROOT,args.host_registry)),runner_sha256=file_hash(Path(__file__)),
       inherited_budget_release_path=args.budget_release,inherited_budget_release_sha256=args.budget_release_sha256,
       allocated_actual_gpu_seconds_including_load=args.allocated_gpu_seconds,
       inherited_K100_IP_scope=kscope,reused_pilot_key_overlap=0,scientific_parameters_changed=False)
    if args.mode=='plan':
        write(base/'cpu_plan.json',plan);print(json.dumps({'model':args.model,'expected_rows':len(keys),'pilot_overlap':0,'allocated_GPU_seconds':args.allocated_gpu_seconds}));return
    if any(int(v.split(':')[1])>=2 for v in os.environ.get('KDM_GPU_SLOTS','').split(',')if v):
        raise ValueError('Current A100 authorization is limited to two worker slots')
    base.mkdir(parents=True,exist_ok=True);lock=(base/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(base/'claim.json',plan);started=time.perf_counter()
    try:
        os.environ['KDM_SUPPLEMENTAL_DATASET']='food101'
        if args.model in EXTRA:
            supplemental.REGISTRY=args.host_registry
            admission=supplemental.validate_supplemental_runtime(ROOT,spec,args.model,cards,'formal',args.claim_id,args.owner)
            actual_spec=admission['runtime_spec']
        else:
            core.REGISTRY=args.host_registry;admission=core.admit(ROOT,spec,args.model,cards)
            actual_spec=core.runtime_spec(ROOT,spec,args.model)
        identity=dict(plan=plan,actual_admission=admission,runtime_spec=actual_spec,pid=os.getpid(),claimed_at_utc=now(),
                      executor={'agent':'/root/tail_scheduler','model':'unknown','effort':'unknown'})
        write(base/'identity.json',identity);backend=make_backend(actual_spec,'cuda:0');load_wall=time.perf_counter()-started
        backend=BudgetInputs(backend,started,len(cards),args.allocated_gpu_seconds,base/'first8_actual_inputs.jsonl')
        groups=defaultdict(list)
        for task in tasks:groups[condition(task)].append(task)
        completed=[];written=0
        for cid,group in groups.items():
            for offset in range(0,len(group),128):
                part=group[offset:offset+128];raw=base/'sealed_parts'/f'{part[0]["method"]}_{cid}_{offset:05d}.jsonl'
                if raw.exists():raise FileExistsError('No implicit repeat or partial recovery')
                before=time.perf_counter()
                # The eight new production boundaries are recorded without altering the native sessions.
                capture=min(len(part),max(0,8-written))
                for task in part[:capture]:
                    backend.current=task;backend.capture=True
                    run_tasks(backend,args.model,[task],raw,identity,DecodeConfig());written+=1
                backend.capture=False
                if capture<len(part):
                    run_tasks(backend,args.model,part[capture:],raw,identity,DecodeConfig());written+=len(part)-capture
                rows=audit(args.model,part,raw)
                detail=dict(passed=True,model=args.model,method=part[0]['method'],marker=part[0]['marker'],rows=len(rows),
                    raw=str(raw.relative_to(ROOT)),raw_sha256=file_hash(raw),completed_keys=[r['key']for r in rows],
                    generation_wall_s=sum(r['wall_s']for r in rows),output_tokens=sum(len(r['tokens'])for r in rows),
                    execution_wall_s=time.perf_counter()-before,completed_at_utc=now(),scope='budgeted_dev404')
                write(raw.with_suffix('.complete.json'),detail);completed.append(detail)
        elapsed=time.perf_counter()-started
        if sum(c['rows']for c in completed)!=len(keys):raise ValueError('Actual completion count differs')
        write(base/'complete_receipt.json',dict(passed=True,actual_rows=len(keys),conditions=completed,
              model_load_wall_s=load_wall,elapsed_s=elapsed,gpu_count=len(cards),gpu_seconds_including_load=elapsed*len(cards),
              allocated_gpu_seconds=args.allocated_gpu_seconds,claimed_identity_sha256=stable_hash(identity),completed_utc=now()))
        print(json.dumps({'passed':True,'model':args.model,'actual_rows':len(keys),'elapsed_s':elapsed}),flush=True)
    except BaseException as error:
        write(base/'failure.json',dict(error=type(error).__name__+': '+str(error),traceback=traceback.format_exc(),
          actual_GPU_seconds_including_load=(time.perf_counter()-started)*len(cards),automatic_retry=False,parameters_changed=False,failed_utc=now()))
        raise
if __name__=='__main__':main()
