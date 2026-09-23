"""Map actual Luna judgments to census gaps and separately resolve two formal scores."""
import json,hashlib,sys
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'workflows/deepseek_annotation_v2')]
from annotate import parse_label,digest
from score import score_run
from kdm.scoring import food_correct
BASE=ROOT/'outputs/annotations/luna_census_remaining_v1/remaining108_plus2_20260924'
PARENT=ROOT/'outputs/annotations/deepseek_v2/census/merged_after_retry151_v1'
OLD=ROOT/'outputs/annotations/luna_semantic_v1/unresolved_9047_20260923_v1/validated_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.open()]
def write(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def jsonl(p,rs):
 with p.open('x') as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 m=load(BASE/'manifest.json')
 for n,k in [('blind_input.jsonl','input_sha256'),('source_mapping.jsonl','mapping_sha256'),('RUBRIC.txt','rubric_sha256')]:assert sha(BASE/n)==m[k]
 assert sha(PARENT/'labels.jsonl')==m['parent_labels_sha256'] and sha(PARENT/'errors.jsonl')==m['errors_sha256']
 qs={r['id']:r for r in rows(BASE/'blind_input.jsonl')};rs=rows(BASE/'results.jsonl');assert len(rs)==110 and {r['id'] for r in rs}==set(qs)
 judgments={r['id']:r for r in rs};resultfiles={'results.jsonl':sha(BASE/'results.jsonl')}
 for p in sorted(BASE.glob('corrections_v*.jsonl'),key=lambda p:int(p.stem.split('_v')[-1])):
  resultfiles[p.name]=sha(p)
  for r in rows(p):assert r['id'] in judgments;judgments[r['id']]=r
 for i,r in judgments.items():
  assert r['group_sha256']==qs[i]['group_sha256']
  if r.get('status')=='unresolved':assert r.get('reason') and 'annotation' not in r
  else:parse_label(json.dumps(r['annotation'],ensure_ascii=False),qs[i]['answer'])
 execution=load(BASE/'execution.json');assert execution['model']=='gpt-6-luna' and execution['reasoning_effort']=='medium' and execution['human_reviewed'] is False and execution['final_gt'] is False
 assert execution['input_sha256']==sha(BASE/'blind_input.jsonl') and execution['results_sha256']==sha(BASE/'results.jsonl')
 for n,h in resultfiles.items():
  if n!='results.jsonl':assert execution['correction_sha256'][n]==h
 assert len(execution['reviewed_ids'])==110 and set(execution['reviewed_ids'])==set(qs) and execution['subagent']
 mapping=rows(BASE/'source_mapping.jsonl');assert len(mapping)==110
 parentid=load(PARENT/'identity.json')
 definition={**parentid['definition'],'schema':'kdm_census_deepseek_plus_luna108_v1','parent_identity':parentid['identity'],'parent_labels_sha256':m['parent_labels_sha256'],'luna_manifest_sha256':sha(BASE/'manifest.json'),'luna_results':resultfiles,'execution_sha256':sha(BASE/'execution.json'),'finalizer_sha256':sha(Path(__file__)),'human_reviewed':False,'api_requests':0}
 identity=digest(definition);out=BASE/'census_merged_v1';out.mkdir(exist_ok=False)
 write(out/'identity.json',{'identity':identity,'definition':definition});(out/'queue.sources.json').write_bytes((PARENT/'queue.sources.json').read_bytes())
 delta=[];unresolved=[];fragment=[]
 for item in mapping:
  i=item['id'];r=judgments[i];src=item['source'];assert qs[i]['answer']==src['text']
  if item['kind']=='formal_fragment':fragment.append((item,r));continue
  if r.get('status')=='unresolved':unresolved.append({'key':src['key'],'reason':r['reason'],'source_id':i});continue
  delta.append({**src,**r['annotation'],'identity':identity,'label_source':'blinded_gpt6_luna_medium_subagent','subagent_input_id':i,'human_reviewed':False,'final_gt':False})
 assert len(delta)+len(unresolved)==108 and len(fragment)==2
 seen=set()
 with (out/'labels.jsonl').open('x') as f:
  for r in rows(PARENT/'labels.jsonl'):
   assert r['identity']==parentid['identity'] and r['key'] not in seen;seen.add(r['key']);r={**r,'source_annotation_identity':r['identity'],'identity':identity};f.write(json.dumps(r,ensure_ascii=False)+'\n')
  for r in delta:assert r['key'] not in seen;seen.add(r['key']);f.write(json.dumps(r,ensure_ascii=False)+'\n')
 assert len(seen)==293236+len(delta)
 original_queue={r['key'] for r in rows(ROOT/'outputs/annotations/deepseek_v2/census/queue.jsonl')}
 assert seen|{r['key'] for r in unresolved}==original_queue and len(original_queue)==293344
 jsonl(out/'labels_delta.jsonl',delta);jsonl(out/'unresolved.jsonl',unresolved)
 report=score_run(ROOT,out)
 receipt={'expected':293344,'previous_validated':293236,'new_luna_validated':len(delta),'validated_labels':len(seen),'unresolved':len(unresolved),'all_keys_accounted_for':True,'automatic_behavior_annotation_complete':not unresolved,'human_reviewed':False,'api_requests':0}
 write(out/'merge_receipt.json',receipt)
 # Never mutate the earlier v3 files: active independent queues bind their hashes.
 oldsum=load(OLD/'summary.json')
 for n,h in oldsum['output_sha256'].items():assert sha(OLD/n)==h
 aliases=load(ROOT/'configs/kdm/food_aliases.json');changes={}
 for item,r in fragment:
  src=item['source'];rawpath=ROOT/src['source_path']
  with rawpath.open('rb') as f:
   for n,line in enumerate(f,1):
    if n==src['source_line']:raw=json.loads(line);assert hashlib.sha256(line).hexdigest()==src['source_row_sha256'];break
   else:raise ValueError('Original formal row missing')
  assert raw['key']==src['key'] and raw['text']==src['text'] and raw['identity']==src['source_identity']
  assert qs[item['id']]['question']==raw['sample']['question']
  assert food_correct(src['text'],raw['sample']['class'],aliases)==0
  new={**src,'source_annotation_identity':src['annotation_identity'],'annotation_identity':identity,'score':0.0,'correctness_resolved':True,'correctness_resolution':'No correct answer present in the original fragment; zero credit under frozen scoring and explicit user clarification 2026-09-24','regenerated':False}
  if r.get('status')=='unresolved':new.update(status='unresolved',reason=r['reason'])
  else:
   a=r['annotation'];assert not a['label'].startswith('answer_') or food_correct(a['answer_text'],raw['sample']['class'],aliases)==0
   new.pop('reason',None);new.update(status='annotated',**a)
  changes[src['key']]=new
 formal=BASE/'formal_fragments_resolved_v1';formal.mkdir(exist_ok=False)
 sem=[changes.get(r['key'],r) for r in rows(OLD/'semantic_labels.jsonl')];merged=[];counts=Counter();groups=defaultdict(Counter);fields=['model','stage','kind','method','guided','reference_guided','marker','reference_marker']
 for r in rows(OLD/'merged_stage_labels.jsonl'):
  if r['key'] in changes:
   x=changes[r['key']];r={**r,'derived_label':x.get('label','unresolved'),'derived_score':0.0,'annotation_mode':'luna_fragment_resolution','semantic_annotation_identity':identity,'correctness_resolved':True}
  merged.append(r)
  for c in [counts,groups[tuple(r[k] for k in fields)]]:
   c['rows']+=1;c[r['derived_label']]+=1;c['correct_score_sum']+=r['derived_score'] or 0;c['correctness_unresolved']+=r['derived_score'] is None
 def metric(c):
  n=c['rows'];return {**dict(c),'accuracy_lower':c['correct_score_sum']/n,'accuracy_upper':(c['correct_score_sum']+c['correctness_unresolved'])/n,'abstention_lower':c['abstain']/n,'abstention_upper':(c['abstain']+c['unresolved'])/n}
 jsonl(formal/'labels_delta.jsonl',list(changes.values()));jsonl(formal/'semantic_labels.jsonl',sem);jsonl(formal/'merged_stage_labels.jsonl',merged)
 fsum={**oldsum,'schema':'kdm_formal_fragment_correctness_resolution_v1','annotation_identity':identity,'parent_summary_sha256':sha(OLD/'summary.json'),'expanded_labels':dict(Counter(r.get('label','unresolved') for r in sem)),'semantic_rows_annotated':sum(r['status']=='annotated' for r in sem),'semantic_rows_unresolved':sum(r['status']=='unresolved' for r in sem),'full_stage_counts':metric(counts),'conditions':[{**dict(zip(fields,k)),**metric(v)} for k,v in sorted(groups.items())],'resolved_zero_credit_fragments':2,'regenerated_answers':0,'human_reviewed':False,'final_gt':False,'limits':'Original source replies remain unchanged. Behavior uncertainty is separate from correctness uncertainty; both specified fragments receive zero credit. Old v3 is preserved for existing dependent evidence.'}
 fsum['semantic_judgments']=dict(Counter({r['group_id']:r.get('label','unresolved') for r in sem}.values()))
 fsum['output_sha256']={n:sha(formal/n) for n in ['semantic_labels.jsonl','merged_stage_labels.jsonl','labels_delta.jsonl']};write(formal/'summary.json',fsum)
 write(BASE/'completion.json',{'census':receipt,'formal_fragments':{'zero_credit':2,'behavior_unresolved':sum(x['status']=='unresolved' for x in changes.values())},'census_output':str(out.relative_to(ROOT)),'formal_output':str(formal.relative_to(ROOT)),'human_reviewed':False,'final_gt':False,'api_requests':0})
 print(json.dumps(load(BASE/'completion.json'),ensure_ascii=False))
if __name__=='__main__':main()
