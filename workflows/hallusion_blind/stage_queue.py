"""Bounded successor jobs after a registered source worker exits; no retries or GPU polling."""
import argparse,json,os,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from kdm.io import atomic_json
from workflows.general_vqa_direct.generate import now
def alive(owner):
 path=Path('/proc')/str(owner['pid'])/'stat'
 if not path.exists():return False
 fields=path.read_text().rsplit(')',1)[1].split()
 return fields[0]!='Z' and int(fields[19])==owner['start_tick']
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
 plan_path=ROOT/a.plan;plan=json.loads(plan_path.read_text());out=plan_path.with_suffix('.status.json')
 ownerpath=ROOT/plan['wait_owner'];owner=json.loads(ownerpath.read_text())
 if owner['hostname']!=socket.gethostname():raise ValueError('Successor must wait on its actual source host')
 atomic_json(out,{'status':'waiting_on_registered_worker','pid':os.getpid(),'source_pid':owner['pid'],
    'source_start_tick':owner['start_tick'],'created_utc':now(),'jobs':plan['jobs']})
 while alive(owner):time.sleep(5)
 if (ownerpath.parent/'released.json').exists():
  atomic_json(out,{'status':'source_handoff_released_successors_held','updated_utc':now(),
      'source_pid':owner['pid'],'reason':'root must bind successor tasks to the new target owner'})
  return 0
 failures=[]
 for i,job in enumerate(plan['jobs']):
  registry=json.loads((ROOT/job['registry']).read_text());host=job['host'];model=job['model']
  spec=json.loads((ROOT/f'configs/runtime/{model}.json').read_text())
  python=registry.get('runtime_overrides',{}).get(host,{}).get(model,{}).get('environment_python',spec['environment_python'])
  claim_id=job['claim_id']
  launches=ROOT/'outputs/hallusion_blind_20261005/launches';launches.mkdir(exist_ok=True)
  receipt=launches/(claim_id+'.launch.json');log=launches/(claim_id+'.log')
  if receipt.exists() or log.exists():raise ValueError('No automatic duplicate/retry of successor '+claim_id)
  cmd=['bash',str(ROOT/'workflows/supplemental/remaining4/worker_registered.sh'),str(ROOT),
       job['cards'],python,str(ROOT/job['registry']),'workflows/hallusion_blind/generate.py','--execute',
       '--model',model,'--methods',*job['methods'],'--registry',job['registry'],'--cards',job['cards'],
       '--claim-id',claim_id,'--owner','/root','--chunk-rows','8']
  if job.get('output'):cmd+=['--output',job['output']]
  if job.get('sample_ids'):cmd+=['--sample-ids',job['sample_ids']]
  if job.get('k100_intern'):cmd+=['--k100-intern']
  with log.open('x') as stream:
   proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  atomic_json(receipt,{'status':'dispatched_not_yet_accepted','model':model,'host':host,'physical_gpus':job['cards'],
       'pid':proc.pid,'started_utc':now(),'command':cmd,'log':str(log.relative_to(ROOT)),'claim_id':claim_id,
       'queue_plan':a.plan,'source_exit_evidence':{'pid':owner['pid'],'start_tick':owner['start_tick'],'active':False}})
  atomic_json(out,{'status':'running_successor','job':i,'claim_id':claim_id,'pid':proc.pid,'updated_utc':now()})
  code=proc.wait()
  if code:
   failures.append({'job':i,'claim_id':claim_id,'exit_code':code})
   atomic_json(out,{'status':'successor_failed_no_retry_continue_independent_jobs','failures':failures,
       'remaining_jobs':plan['jobs'][i+1:],'updated_utc':now()})
 atomic_json(out,{'status':'all_successors_finished_with_failures' if failures else 'all_successors_finished',
     'updated_utc':now(),'jobs':len(plan['jobs']),'failures':failures})
 return 1 if failures else 0
if __name__=='__main__':raise SystemExit(main())
