"""Explicit host/card claims; methods retain a loaded frozen checkpoint."""
import argparse,json,socket,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from workflows.general_vqa_direct.launch import JOBS
from workflows.general_vqa_direct.generate import now,write_once
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--models',nargs='+',required=True);p.add_argument('--phase',required=True)
 p.add_argument('--methods',nargs='+',required=True);p.add_argument('--host');p.add_argument('--cards')
 p.add_argument('--sample-ids');p.add_argument('--registry',default='workflows/general_vqa_direct/host_registry.json')
 a=p.parse_args();reg=ROOT/a.registry;r=json.loads(reg.read_text())
 for model in a.models:
  host,cards=JOBS[model]
  if a.host:host,cards=a.host,a.cards
  row=r['hosts'][host]
  if row['hostname']!=socket.gethostname() or Path(row['root'])!=ROOT:raise ValueError('Explicit current host required')
  spec=json.loads((ROOT/f'configs/runtime/{model}.json').read_text())
  python=r.get('runtime_overrides',{}).get(host,{}).get(model,{}).get('environment_python',spec['environment_python'])
  identity=f'{model}_{host}_{a.phase}'
  cmd=['bash',str(ROOT/'workflows/supplemental/remaining4/worker_registered.sh'),str(ROOT),cards,python,str(reg),
       'workflows/hallusion_blind/generate.py','--execute','--model',model,'--methods',*a.methods,
       '--registry',a.registry,'--cards',cards,'--claim-id',identity,'--owner','/root','--chunk-rows','8']
  if host=='k100' and model=='internvl35_8b':cmd.append('--k100-intern')
  if a.sample_ids:cmd+=['--sample-ids',a.sample_ids]
  folder=ROOT/'outputs/hallusion_blind_20261005/launches';folder.mkdir(parents=True,exist_ok=True)
  log=folder/(identity+'.log');receipt=folder/(identity+'.launch.json')
  if log.exists() or receipt.exists():raise ValueError('Do not repeat a launch identity')
  with log.open('x') as out:
   process=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  record={'status':'dispatched_not_yet_accepted','model':model,'host':host,'physical_gpus':cards,
          'pid':process.pid,'started_utc':now(),'command':cmd,'log':str(log.relative_to(ROOT)),'claim_id':identity}
  write_once(receipt,record);print(json.dumps(record),flush=True)
if __name__=='__main__':main()
