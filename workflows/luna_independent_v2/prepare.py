"""Verify completed independent cohorts, prepare blind exact-deduplicated Luna queue."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from workflows.independent_v5.postprocess import inspect_attempt,checked_closed,Matcher
from kdm.io import stable_hash
from workflows.luna_semantic_v1.finalize import collect
OUT=ROOT/'outputs/annotations/luna_independent_v2/completed3_20260923_v1'
OLD=ROOT/'outputs/annotations/luna_semantic_v1/unresolved_9047_20260923_v1'
SOURCES=[('panel5_independent_food_v5','qwen25vl'),('panel5_independent_food_v5','minicpm26'),('panel5_independent_food_v5_spawn1','llava16_mistral')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.open()]
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def jsonl(p,rs):
 with p.open('x') as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 latest=load(OLD/'latest.json');assert latest['active_export']=='validated_v3'
 oldmeta,oldqueue,oldresults,oldfiles,errors=collect()
 assert sha(OLD/'validated_v3/summary.json')==latest['summary_sha256']
 oldbyhash={oldqueue[i]['group_sha256']:(oldqueue[i],r) for i,r in oldresults.items()}
 samples={s['id']:s for s in rows(ROOT/'data/current/all.jsonl') if s['dataset']=='food101'}
 assert len(samples)==4848 and Counter(s['split'] for s in samples.values())=={'dev':2424,'eval':2424}
 groups={};mapping=[];sources=[];files={};matcher=Matcher()
 def bind(p):files[str(p.relative_to(ROOT))]=sha(p)
 for run,model in SOURCES:
  directory=ROOT/'outputs/annotations/independent_v5'/run/model;summary=load(directory/'summary.json')
  assert summary['generated_answers']==summary['screened_answers']==48480
  for n,h in summary['outputs_sha256'].items():assert sha(directory/n)==h;bind(directory/n)
  bind(directory/'summary.json');labels=rows(directory/'labels.jsonl');assert len(labels)==48480
  source=summary['source'];raw=ROOT/source['raw'];record=ROOT/source['record'];sidepath=raw.with_suffix('.identity.json');side=load(sidepath);done=load(record/'complete.json')
  assert sha(raw)==source['raw_sha256']==done['output_sha256'];assert sha(sidepath)==source['sidecar_sha256'];assert sha(record/'complete.json')==source['complete_sha256']
  assert side['identity']==stable_hash(side['definition'])==source['identity']==done['identity']
  assert done['generation_complete'] and done['independent']==48480
  for p in [raw,sidepath,record/'complete.json',record/'identity.json']:bind(p)
  for p,h in summary['rules_sha256'].items():assert sha(ROOT/p)==h;bind(ROOT/p)
  ranks,proof=checked_closed(ROOT/summary['closed_source']['record'],model,samples,matcher);assert proof==summary['closed_source']
  for p in [ROOT/proof['raw'],(ROOT/proof['raw']).with_suffix('.identity.json'),ROOT/proof['record']/'completed_verification_20260923.json',ROOT/proof['record']/'complete.json']:bind(p)
  seen=set();per={sid:set() for sid in samples};seeds={sid:set() for sid in samples};unresolved=0
  with raw.open('rb') as f:
   for number,line in enumerate(f,1):
    assert line.endswith(b'\n');r=json.loads(line);l=labels[number-1];sid,rep=inspect_attempt(r,model,samples,side['identity'],side['definition']['eos_token_ids'])
    assert r['key'] not in seen and rep not in per[sid] and r['seed'] not in seeds[sid];seen.add(r['key']);per[sid].add(rep);seeds[sid].add(r['seed'])
    assert l['source_line']==number and l['source_row_sha256']==hashlib.sha256(line).hexdigest() and l['key']==r['key'] and l['text']==r['text'] and l['source_identity']==side['identity']
    if l['screening_label']!='unresolved':continue
    unresolved+=1;content={'question':r['sample']['question'],'answer':r['text']};g=digest(content)
    if g in groups:assert groups[g]==content
    groups[g]=content;mapping.append({'group_sha256':g,'source_label_file':str((directory/'labels.jsonl').relative_to(ROOT)),'source_raw_file':str(raw.relative_to(ROOT)),'source_label':l})
  assert len(seen)==48480 and all(v==set(range(10)) for v in per.values()) and unresolved==summary['counts']['unresolved']
  sources.append({'run':run,'model':model,'annotation_directory':str(directory.relative_to(ROOT)),'generated':48480,'unresolved':unresolved,'questions':4848,'coverage_verified':True,'source':source,'closed_source':proof})
 pending=[];reused=[];held=[];allgroups=[]
 for g,content in groups.items():
  if g in oldbyhash:
   q,r=oldbyhash[g];assert content=={k:q[k] for k in ['question','answer']}
   obj={'group_sha256':g,**content,'old_group_id':r['id'],'old_export':str((OLD/'validated_v3').relative_to(ROOT)),'result':r}
   if r.get('status')=='unresolved':held.append(obj);state='old_unresolved_preserved'
   else:reused.append(obj);state='reused_validated_v3'
   ident=None
  else:
   ident=len(pending)+1;pending.append({'id':ident,'group_sha256':g,**content});state='pending_luna'
  allgroups.append({'group_sha256':g,'pending_id':ident,'state':state})
 byhash={g['group_sha256']:g for g in allgroups}
 for m in mapping:m.update(pending_id=byhash[m['group_sha256']]['pending_id'],state=byhash[m['group_sha256']]['state'])
 for n,rs in [('blind_queue.jsonl',pending),('source_mapping.jsonl',mapping),('reused_groups.jsonl',reused),('held_unresolved_groups.jsonl',held),('group_index.jsonl',allgroups)]:jsonl(OUT/n,rs)
 (OUT/'batches').mkdir();(OUT/'results').mkdir()
 for start in range(0,len(pending),100):jsonl(OUT/'batches'/f'input_{start//100+1:03d}.jsonl',pending[start:start+100])
 rubric=(OLD/'RUBRIC.txt').read_text();(OUT/'RUBRIC.txt').write_text(rubric)
 for rel in ['latest.json','validated_v3/identity.json','validated_v3/summary.json','validated_v3/semantic_labels.jsonl','blind_queue.jsonl','manifest.json']:bind(OLD/rel)
 for rel in oldfiles:bind(OLD/rel)
 bind(ROOT/'data/current/all.jsonl');bind(Path(__file__))
 manifest={'schema':'kdm_luna_independent_queue_v2','model':'gpt-6-luna','reasoning_effort':'medium','source_rows':sum(s['generated'] for s in sources),'source_unresolved_rows':len(mapping),'unique_groups':len(groups),'new_pending_groups':len(pending),'reused_valid_groups':len(reused),'preserved_old_unresolved_groups':len(held),'mapping_states':dict(Counter(m['state'] for m in mapping)),'sources':sources,'input_files_sha256':files,'queue_files_sha256':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()},'human_reviewed':False,'final_gt':False,'api_requests':0,'postprocess_repeated':False}
 write(OUT/'manifest.json',manifest)
 print(json.dumps({k:manifest[k] for k in ['source_rows','source_unresolved_rows','unique_groups','new_pending_groups','reused_valid_groups','preserved_old_unresolved_groups','mapping_states']}))
if __name__=='__main__':main()
