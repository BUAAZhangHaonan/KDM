#!/usr/bin/env python3
"""Only the authorized missing main Viz512 conditions, through the original runtime."""
import argparse,json,os,pathlib,sys,time,traceback
from collections import defaultdict
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows.paper_core.dev_viz import roster,condition,audit_rows,now,PilotInputs
from workflows.paper_core.ip_dev_pilot import CORE,EXTRA,audit,write
from workflows.paper_core.ip_dev_full import BudgetInputs
from workflows.paper_core import execution as core
from workflows.supplemental.remaining11 import execution as supplemental
from workflows.supplemental.remaining11.generate import validate_proofs
from kdm.frozen import load_contract
from kdm.decoding import DecodeConfig
from kdm.pipeline import make_backend,run_tasks,task_id
from kdm.io import file_hash,read_jsonl,within

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['plan','pilot','full']);p.add_argument('--model',choices=CORE+EXTRA,required=True)
 for k in ['run-id','claim-id','owner','host-registry','authority','authority-sha256','missing-keys','physical-gpus']:p.add_argument('--'+k,required=True)
 p.add_argument('--allocated-gpu-seconds',type=float,required=True);p.add_argument('--budget-release');a=p.parse_args()
 authpath=within(ROOT,a.authority);auth=json.loads(authpath.read_text());assert file_hash(authpath)==a.authority_sha256
 assert auth['schema']=='kdm_nine_model_selected_viz512_main_gap_authority_v1'and auth['passed']and auth['all52_new_IP_dev404_conditions_closed']and auth['authorized_main_configurations']==21
 assert len(auth['conditions'])==21 and len({(x['model'],x['method'])for x in auth['conditions']})==21
 for s in auth['source_bindings']:assert file_hash(within(ROOT,s['path']))==s['sha256']
 specs=[x for x in auth['conditions']if x['model']==a.model];allowed={'instruction_m3id'}if a.model in CORE else{'instruction_m3id','instruction_vcd','vcd','cda_visual'}
 assert {x['method']for x in specs}==allowed
 rosterpath,samples=roster('viz512');assert file_hash(rosterpath)==auth['roster_sha256']
 tasks=[]
 for c in specs:
  assert c['method']in allowed and c['replicate']==0 and c['rows']==512
  if c['method']=='vcd':assert(c['kind'],c['marker'],c['reference_marker'],c['guided'],c['reference_guided'])==('native_unguided','NONE','NONE',False,False)
  else:
   assert c['kind']=='instruction_preserving'and c['reference_marker']==c['marker']and c['guided']is True and c['reference_guided']is False
   if c['method']=='cda_visual':assert c['marker']=='UNKNOWN'and c['selection']=='original_single_registered_CDA'
   else:assert c['selection']=='dev_selected'and c['dev_n']==404 and c['selection_frozen_at_utc']
  for sample in samples:tasks.append(dict(sample=sample,**{k:c[k]for k in ['method','kind','marker','reference_marker','guided','reference_guided','replicate']}))
 expected={task_id(a.model,t):t for t in tasks};missingpath=within(ROOT,a.missing_keys);missing=list(read_jsonl(missingpath));keys=[x['key']for x in missing]
 assert keys and len(keys)==len(set(keys))and set(keys)<=expected.keys()and not set(keys)&set(auth['exact_reused_keys'])
 chosen=[expected[k]for k in keys];base=ROOT/'outputs/paper_core_20261002_dev_viz'/a.run_id/a.claim_id
 groups=defaultdict(list)
 for t in chosen:groups[condition(t)].append(t)
 pilot={c:ts[:min(8,len(ts))]for c,ts in groups.items()}
 spec=json.loads((ROOT/f'configs/runtime/{a.model}.json').read_text());manifest,freeze=load_contract(ROOT);proofs=validate_proofs(ROOT,spec,a.model,list(allowed),'formal',manifest,freeze)
 native_source=None
 if a.model in EXTRA:
  from workflows.supplemental.remaining4.native import source_inputs
  _,original,_,native_source=source_inputs(a.model,'vcd','vizwiz');assert spec==original
 plan=dict(schema='kdm_selected_main_Viz512_missing_claim_v1',model=a.model,claim_id=a.claim_id,owner=a.owner,stage='viz512',registered_configurations=specs,authority=a.authority,authority_sha256=a.authority_sha256,missing_keys=a.missing_keys,missing_keys_sha256=file_hash(missingpath),expected_rows=len(keys),pilot_rows=sum(len(ts)for ts in pilot.values()),roster_sha256=file_hash(rosterpath),proofs=proofs,native_original_source=native_source,scientific_parameters_changed=False,allocated_actual_gpu_seconds_including_load=a.allocated_gpu_seconds)
 if a.mode=='plan':write(base/'cpu_plan.json',plan);print(json.dumps({'model':a.model,'missing_rows':len(keys),'pilot_rows':plan['pilot_rows'],'GPU_initialized':False}));return
 import fcntl
 base.mkdir(parents=True,exist_ok=True);lock=(base/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if a.mode=='pilot':assert not(base/'pilot_receipt.json').exists()and not(base/'identity.json').exists()
 else:
  gate=json.loads((base/'pilot_receipt.json').read_text());release=json.loads(within(ROOT,a.budget_release).read_text())
  assert gate['passed']and gate['authority_sha256']==a.authority_sha256 and gate['missing_keys_sha256']==file_hash(missingpath)
  assert release['released']and release['stage']=='viz512'and release['pilot_receipt_sha256']==file_hash(base/'pilot_receipt.json')and release['authority_sha256']==a.authority_sha256
  assert release['allocated_GPU_seconds_including_reused_pilot']==a.allocated_gpu_seconds and release['actual_pilot_measured']and release['current_authorized_scope']=='registered21_main_Viz512_missing_keys_only'and release['estimated_GPU_hours']*3600<=a.allocated_gpu_seconds
 assert a.allocated_gpu_seconds>0
 if any(int(v.split(':')[1])>=2 for v in os.environ.get('KDM_GPU_SLOTS','').split(',')if v):raise ValueError('At most two authorized A100 slots')
 current_cap=a.allocated_gpu_seconds-(gate['GPU_seconds_including_load']if a.mode=='full'else 0);assert current_cap>0
 started=time.perf_counter();cards=a.physical_gpus.split(',')
 try:
  if a.model in EXTRA:
   supplemental.REGISTRY=a.host_registry;os.environ['KDM_SUPPLEMENTAL_DATASET']='vizwiz';admitted=supplemental.validate_supplemental_runtime(ROOT,spec,a.model,cards,'formal',a.claim_id,a.owner);actual=admitted['runtime_spec']
  else:core.REGISTRY=a.host_registry;admitted=core.admit(ROOT,spec,a.model,cards);actual=core.runtime_spec(ROOT,spec,a.model)
  identity=dict(plan=plan,actual_admission=admitted,runtime_spec=actual,pid=os.getpid(),claimed_at_utc=now(),stage='viz512')
  if a.mode=='full':
   old=json.loads((base/'identity.json').read_text());assert old['runtime_spec']==actual
   assert {k:v for k,v in old['plan'].items()if k!='allocated_actual_gpu_seconds_including_load'}=={k:v for k,v in plan.items()if k!='allocated_actual_gpu_seconds_including_load'}
   write(base/'full_identity.json',identity)
  else:write(base/'identity.json',identity);write(base/'claim.json',plan)
  backend=make_backend(actual,'cuda:0');load=time.perf_counter()-started;backend=BudgetInputs(backend,started,len(cards),current_cap,base/'pilot_inputs.jsonl');done=[]
  excluded={k for d in gate['conditions']for k in d['completed_keys']}if a.mode=='full'else set()
  for cid,ts in groups.items():
   selected=pilot[cid]if a.mode=='pilot'else[t for t in ts if task_id(a.model,t)not in excluded]
   for offset in range(0,len(selected),128):
    part=selected[offset:offset+128];raw=base/'sealed_parts'/f'{a.mode}_{part[0]["method"]}_{cid}_{offset:05d}.jsonl';assert not raw.exists()
    for t in part:
     backend.current=t;backend.capture=a.mode=='pilot';run_tasks(backend,a.model,[t],raw,identity,DecodeConfig())
    rows=audit(a.model,part,raw)if part[0]['method'].startswith('instruction_')else audit_rows(a.model,part,raw)
    detail=dict(passed=True,model=a.model,method=part[0]['method'],marker=part[0]['marker'],rows=len(rows),raw=str(raw.relative_to(ROOT)),raw_sha256=file_hash(raw),completed_keys=[x['key']for x in rows],generation_wall_s=sum(x['wall_s']for x in rows),output_tokens=sum(len(x['tokens'])for x in rows),completed_at_utc=now(),scope='selected_main_viz512')
    write(raw.with_suffix('.complete.json'),detail);done.append(detail)
  if a.mode=='full':assert excluded|{k for d in done for k in d['completed_keys']}==set(keys)
  elapsed=time.perf_counter()-started
  receipt=dict(passed=True,model=a.model,stage='viz512',actual_rows=sum(x['rows']for x in done),conditions=done,model_load_wall_s=load,elapsed_s=elapsed,gpu_count=len(cards),GPU_seconds_including_load=elapsed*len(cards),authority_sha256=a.authority_sha256,missing_keys_sha256=file_hash(missingpath),completed_utc=now(),scientific_parameters_changed=False)
  write(base/('pilot_receipt.json'if a.mode=='pilot'else'complete_receipt.json'),receipt);print(json.dumps({'passed':True,'model':a.model,'actual_rows':receipt['actual_rows'],'GPU_seconds':receipt['GPU_seconds_including_load']}))
 except BaseException as e:
  write(base/(a.mode+'_failure.json'),dict(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc(),actual_GPU_seconds_including_load=(time.perf_counter()-started)*len(cards),automatic_retry=False,parameters_changed=False));raise
if __name__=='__main__':main()
