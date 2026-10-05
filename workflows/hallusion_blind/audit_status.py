"""Verify new immutable chunks, deduplicate exact transported copies, persist compact state."""
import argparse,json,sys,subprocess
from pathlib import Path
from argparse import Namespace
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from kdm.io import atomic_json,file_hash,stable_hash,read_jsonl
from workflows.hallusion_blind.generate import load_plan,validate_rows,now
def brief_failure(path):
 x=json.loads(path.read_text())
 return {k:x.get(k) for k in ['failed_utc','key','claim_identity','error','automatic_retry','automatic_parameter_change']} | {'source':str(path.relative_to(ROOT))}
def main():
 b=ROOT/'outputs/hallusion_blind_20261005'
 protocol=json.loads((b/'registration/protocol.json').read_text())
 conditions=json.loads((b/'registration/conditions.json').read_text())
 plans={}
 for model in protocol['models']:
  args=Namespace(model=model,methods=[c['method'] for c in conditions if c['model']==model],
   protocol=str((b/'registration/protocol.json').relative_to(ROOT)),output=str((b/'runs').relative_to(ROOT)),sample_ids=None)
  plans[model]=load_plan(args)
 roots=[p for p in [b/'runs',b/'runs_continuation'] if p.exists()]
 if (b/'collected').exists():roots.extend(p/n for p in (b/'collected').iterdir() for n in ['runs','runs_continuation'] if (p/n).exists())
 seen={};rawlist=[];counts=defaultdict(int);newcounts=defaultdict(int);replicas=0
 for folder in roots:
  for model in folder.iterdir():
   if model.name not in plans:continue
   plan=plans[model.name]
   for receipt_path in sorted(model.glob('claims/*/chunk_*.complete.json')):
    receipt=json.loads(receipt_path.read_text());raw=model/receipt['raw_path'];ownerpath=model/receipt['owner_path']
    owner=json.loads(ownerpath.read_text());sha=file_hash(raw)
    if (receipt['identity']!=plan['identity'] or sha!=receipt['raw_sha256']
       or stable_hash(owner)!=receipt['claim_identity'] or file_hash(ownerpath)!=receipt['owner_sha256']):
     raise ValueError('Immutable receipt/owner/raw/source binding differs '+str(receipt_path))
    checked=validate_rows(raw,plan,receipt['keys'],receipt['claim_identity'],set(receipt['eos_token_ids']))
    if checked!=receipt['validation']:raise ValueError('Receipt validation differs')
    repeated=[k for k in receipt['keys'] if k in seen]
    if repeated:
     if len(repeated)!=len(receipt['keys']) or any(seen[k]!=sha for k in repeated):
      raise ValueError('Non-identical duplicate model-condition-input output')
     replicas+=1;continue
    rawlist.append(str(raw))
    for row in read_jsonl(raw):
     key=row['key'];seen[key]=sha
     counts[(model.name,row['method'])]+=1
     if row['generation_source']=='new_eos_only':newcounts[(model.name,row['method'])]+=1
 (b/'validated_raw_list.txt').write_text('\n'.join(rawlist)+'\n')
 workers=[]
 for p in (b/'collected').glob('*/status.json'):
  for w in json.loads(p.read_text())['workers']:
   if 'failed' in w:w['failed']={k:w['failed'].get(k) for k in ['failed_utc','key','claim_identity','error']}
   workers.append(w)
 for m in protocol['models']:
  for ownerfile in [f for folder in [b/'runs',b/'runs_continuation'] for f in (folder/m).glob('claims/*/owner.json')]:
   owner=json.loads(ownerfile.read_text())
   if owner['admission'].get('host')!='4028':continue
   claim=ownerfile.parent;proc=Path('/proc')/str(owner['pid']);alive=False
   if (proc/'stat').exists():
    stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
    alive=int(stat[19])==owner['start_tick'] and stat[0]!='Z'
   row={'model':m,'claim_id':owner['claim_id'],'pid':owner['pid'],'start_tick':owner['start_tick'],
        'process_running':alive,'physical_gpus':owner['admission']['physical_gpus'],'status':'running' if alive else 'exited',
        'sealed_rows':sum(json.loads(f.read_text())['validation']['rows'] for f in claim.glob('chunk_*.complete.json')),
        'first8':[json.loads(f.read_text()) for f in claim.glob('first8_*.json')]}
   for name in ['complete.json','failed.json','released.json']:
    if (claim/name).exists():
     row[name[:-5]]=brief_failure(claim/name) if name=='failed.json' else json.loads((claim/name).read_text())
     if name=='failed.json':row['status']='failed'
   workers.append(row)
 latest=None
 for p in sorted((b/'scoring').glob('snapshot_*/receipt.json')):
  latest={'path':str(p.relative_to(ROOT)),**json.loads(p.read_text())}
 state={'updated_utc':now(),'dataset':'hallusionbench','blind_questions':951,
   'hall_development_split':False,'registered_upper_bound':{'conditions':70,'answers':66570},
   'cpu_admitted_upper_bound':{'conditions':68,'answers':64668,'sid_runtime_admission':'pending'},
   'token_budget':None,'termination':'natural_EOS_only','excluded_server':'d4030',
   'verified_reused_Direct':sum(n for (m,x),n in counts.items() if x=='direct')-sum(n for (m,x),n in newcounts.items() if x=='direct'),
   'new_EOS_validated':sum(newcounts.values()),'total_EOS_validated':len(seen),
   'remaining_to_cpu_admitted_upper_bound':64668-len(seen),'verified_raw_files':len(rawlist),
   'exact_transport_replica_chunks_skipped':replicas,'scoring':latest,
   'conditions':[{'model':m,'method':x,'completed':n,'expected':951} for (m,x),n in sorted(counts.items())],
   'workers':workers,'commits':subprocess.check_output(['git','log','-4','--format=%h'],cwd=ROOT,text=True).splitlines(),'git_pushed':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==subprocess.check_output(['git','rev-parse','refs/remotes/origin/master'],cwd=ROOT,text=True).strip(),'generation_failures':[brief_failure(p) for folder in roots for p in folder.glob('*/claims/*/failed.json')]}
 atomic_json(b/'CURRENT_STATE.json',state)
 text='# HallusionBench 951盲测运行状态\n\n'
 text+=f"更新UTC：{state['updated_utc']}。完整EOS {len(seen)} / 候选64668；精确Direct复用{state['verified_reused_Direct']}；新生成验收{state['new_EOS_validated']}。\n\n"
 text+='九模型Food单工作点。未在Hall划dev/选措辞。无输出token上限；架构/显存失败保留真实错误，不能算完成。4030不连接。\n\n'
 text+='| 模型 | 方法 | 验收完整回复 | 最终分母 |\n|---|---|---:|---:|\n'
 text+=''.join(f'| {m} | {x} | {n} | 951 |\n' for (m,x),n in sorted(counts.items()))
 text+='\nSID：MiniCPM/Qwen3.5沿原架构范围不适用；Qwen2.5的30题和Qwen3-VL的59题不足100视觉词元，两面板排除，不过滤题目。其余5面板先CPU前检、再实际准入。\n'
 (b/'RUN_STATUS.md').write_text(text)
 print(json.dumps({k:v for k,v in state.items() if k not in {'conditions','workers'}},indent=2))
if __name__=='__main__':main()
