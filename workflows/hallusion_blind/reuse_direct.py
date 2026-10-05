"""Bind verified natural-EOS Direct sources to the new blind panel without inference."""
import argparse,json,sys
from pathlib import Path
from argparse import Namespace
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from kdm.io import read_jsonl,file_hash,stable_hash
from workflows.hallusion_blind.generate import load_plan,claim,seal,write_once,now
def main():
 base=ROOT/'outputs/hallusion_blind_20261005/registration'
 receipt=json.loads((base/'direct_reuse_identity_validation.json').read_text())
 source=ROOT/receipt['source']
 if file_hash(source)!=receipt['source_sha256'] or not receipt['source_owner_identity_checkpoint_precision_and_raw_receipt_validated']:
  raise ValueError('Reuse source identity gate failed')
 rows=list(read_jsonl(source));by_model={}
 for item in rows:by_model.setdefault(item['row']['model'],[]).append(item)
 for model,items in by_model.items():
  assignments=base/f'direct_reuse_ids_{model}.jsonl'
  if assignments.exists():raise ValueError('Inspect existing reuse instead of repeating')
  assignments.write_text(''.join(json.dumps({'id':x['row']['sample']['id']})+'\n' for x in items))
  a=Namespace(model=model,methods=['direct'],protocol=str((base/'protocol.json').relative_to(ROOT)),
    output='outputs/hallusion_blind_20261005/runs',sample_ids=str(assignments.relative_to(ROOT)),
    claim_id=f'{model}_verified_legacy_eos001',owner='/root',chunk_rows=16)
  plan=load_plan(a)
  admission={'host':'cpu_verified_reuse','source_receipt':str((base/'direct_reuse_identity_validation.json').relative_to(ROOT)),
    'source_receipt_sha256':file_hash(base/'direct_reuse_identity_validation.json'),'new_gpu_generations':0}
  run,keys,owner=claim(a,plan,admission)
  if run is None:raise ValueError('Unexpected existing imported source')
  mapped={x['row']['sample']['id']:x for x in items}
  pending=run/'chunk_00000.pending.jsonl';eos=set()
  with pending.open('x') as out:
   for key in keys:
    task=plan['tasks'][key];bound=mapped[task['sample']['id']];old=bound['row']
    if old['sample']!=task['sample'] or not old['terminated'] or old['prompt']!=task['sample']['prompt']:
     raise ValueError('Full sample/natural EOS/prompt differs')
    eos.add(old['tokens'][-1])
    row={**old,**task,'key':key,'identity':plan['identity'],'claim_identity':stable_hash(owner),
      'dataset_identity':plan['datasets']['hallusionbench']['identity'],'config':asdict(plan['cfgs'][task['condition_identity']]),
      'generation_source':'exact_reused_natural_eos','reused_source':{k:v for k,v in bound.items() if k not in {'row','source_score_row'}},
      'original_generation_fields':{k:old.get(k) for k in ['key','identity','claim_identity','dataset_identity','config','kind','engine']},
      'source_score_row':bound['source_score_row']}
    out.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
  seal(plan,run,owner,0,pending,keys,eos)
  write_once(run/'complete.json',{'status':'complete','completed_rows':len(keys),'new_gpu_generations':0,
       'identity':plan['identity'],'claim_identity':stable_hash(owner),'completed_utc':now()})
  print(json.dumps({'model':model,'verified_reused':len(keys),'new_gpu_generations':0}),flush=True)
if __name__=='__main__':main()
