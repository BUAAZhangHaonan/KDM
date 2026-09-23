"""Validate and map authored Luna judgments; no semantic classification or API calls."""
import argparse,json,hashlib,sys
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from workflows.deepseek_annotation_v2.annotate import parse_label
from workflows.independent_v5.postprocess import question_summary
from kdm.scoring import food_correct
BASE=ROOT/'outputs/annotations/luna_independent_v2/completed3_20260923_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.open()]
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def jsonl(p,rs):
 with p.open('x') as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def validate(r,q):
 assert r['id']==q['id'] and r['group_sha256']==q['group_sha256']
 if r.get('status')=='unresolved':assert isinstance(r.get('reason'),str) and r['reason'].strip() and 'annotation' not in r
 else:parse_label(json.dumps(r['annotation'],ensure_ascii=False),q['answer'])
def collect(batch=None):
 meta=load(BASE/'manifest.json');queue={r['id']:r for r in rows(BASE/'blind_queue.jsonl')};results={};proof={}
 for rel,h in meta['queue_files_sha256'].items():assert sha(BASE/rel)==h,rel
 inputs=sorted((BASE/'batches').glob('input_*.jsonl'))
 if batch is not None:inputs=[BASE/'batches'/f'input_{batch:03d}.jsonl']
 for p in inputs:
  qs=rows(p);expected={q['id'] for q in qs};result=BASE/'results'/p.name.replace('input_','batch_');data=rows(result)
  assert len(data)==len(expected) and {r['id'] for r in data}==expected
  proof[str(result.relative_to(BASE))]=sha(result)
  for r in data:assert r['id'] not in results;validate(r,queue[r['id']]);results[r['id']]=r
 for p in sorted((BASE/'results').glob('corrections_v*.jsonl'),key=lambda x:int(x.stem.rsplit('_v',1)[1])):
  proof[str(p.relative_to(BASE))]=sha(p);seen=set()
  for r in rows(p):
   if batch is not None and r['id'] not in results:continue
   assert r['id'] in results and r['id'] not in seen and r.get('reason');seen.add(r['id']);validate(r,queue[r['id']]);results[r['id']]=r
 if batch is None:assert len(results)==len(queue)==meta['new_pending_groups']
 return meta,queue,results,proof

def run(args):
 meta,queue,results,proof=collect(args.batch)
 if args.batch is not None:
  print(json.dumps({'batch':args.batch,'groups':len(results),'schema_spans_valid':True,'semantic_quality_not_established_by_validator':True}));return
 for rel,h in meta['input_files_sha256'].items():assert sha(ROOT/rel)==h,rel
 pilot=load(BASE/'pilot_review.json');assert pilot['model']=='gpt-6-luna' and pilot['reasoning_effort']=='medium' and pilot['short_answer_quality_passed'] is True
 assert pilot['human_reviewed'] is False and pilot['final_gt'] is False
 assert set(pilot['reviewed_ids'])=={q['id'] for q in rows(BASE/'batches/input_001.jsonl')}
 for rel,h in pilot['reviewed_files_sha256'].items():assert sha(BASE/rel)==h,rel
 assert pilot['reviewed_files_sha256']['batches/input_001.jsonl']==sha(BASE/'batches/input_001.jsonl')
 # Bind explicit subagent execution receipts to every original batch result.
 execution=load(BASE/'batch_execution.json');assert execution['model']=='gpt-6-luna' and execution['reasoning_effort']=='medium'
 assert execution['human_reviewed'] is False and execution['final_gt'] is False
 executed={}
 for b in execution['batches']:
  assert b['subagent'] and b['result_file'] not in executed
  assert sha(BASE/b['input_file'])==b['input_sha256'] and sha(BASE/b['result_file'])==b['result_sha256'];executed[b['result_file']]=b
 assert set(executed)=={p for p in proof if p.startswith('results/batch_')}
 groups={q['group_sha256']:r for i,r in results.items() for q in [queue[i]]}
 for name in ['reused_groups.jsonl','held_unresolved_groups.jsonl']:
  for x in rows(BASE/name):assert x['group_sha256'] not in groups;groups[x['group_sha256']]=x['result']
 assert len(groups)==meta['unique_groups']
 by_source={}
 for m in rows(BASE/'source_mapping.jsonl'):
  l=m['source_label'];k=(l['model'],l['key']);assert k not in by_source;by_source[k]=m
 samples={r['id']:r for r in rows(ROOT/'data/current/all.jsonl') if r['dataset']=='food101'};aliases=load(ROOT/'configs/kdm/food_aliases.json')
 definition={'schema':'kdm_luna_independent_semantic_v2','manifest_sha256':sha(BASE/'manifest.json'),'result_files_sha256':proof,'pilot_sha256':sha(BASE/'pilot_review.json'),'execution_sha256':sha(BASE/'batch_execution.json'),'finalizer_sha256':sha(Path(__file__)),'model':'gpt-6-luna','reasoning_effort':'medium','human_reviewed':False,'final_gt':False,'api_requests':0}
 ident=digest(definition);merged=[];questions=[];counts={};semantic=[]
 for s in meta['sources']:
  d=ROOT/s['annotation_directory'];attempts=defaultdict(dict);c=Counter();existing={q['sample_id']:q for q in rows(d/'questions.jsonl')}
  for l in rows(d/'labels.jsonl'):
   m=by_source.get((l['model'],l['key']));out=dict(l)
   if m is None:out.update(annotation_mode='original_free_screening',semantic_identity=None)
   else:
    r=groups[m['group_sha256']];out.update(annotation_mode=m['state'],semantic_identity=ident,semantic_group_sha256=m['group_sha256'])
    if r.get('status')=='unresolved':out.update(screening_label='unresolved',preliminary_score=None,semantic_status='unresolved',semantic_reason=r['reason'])
    else:
     a=r['annotation'];parse_label(json.dumps(a),l['text']);score=float(food_correct(a['answer_text'],samples[l['sample_id']]['class'],aliases)) if a['label'].startswith('answer_') else 0.0
     state=('correct' if score else 'incorrect') if a['label'].startswith('answer_') else a['label']
     out.update(screening_label=state,preliminary_score=score,semantic_status='annotated',behavior_label=a['label'],evidence_span=a['evidence_span'],answer_text=a['answer_text'])
    semantic.append(out)
   c[out['screening_label']]+=1;merged.append(out);assert l['replicate'] not in attempts[l['sample_id']];attempts[l['sample_id']][l['replicate']]=out
  assert sum(c.values())==48480 and len(attempts)==4848
  counts[s['model']]=dict(c)
  for sid,a in attempts.items():
   q=question_summary(s['model'],samples[sid],a,existing[sid]['gold_rank']);q.update(schema='kdm_independent_food_semantic_probe_v2',evidence_state=q['evidence_state'].replace('free_screening','automatic_semantic_and_free_screening'),semantic_identity=ident);questions.append(q)
 summary={'identity':ident,'generated_answers':len(merged),'semantic_source_rows':len(semantic),'new_luna_groups':len(results),'reused_groups':meta['reused_valid_groups'],'counts_by_model':counts,'questions':len(questions),'question_states':dict(Counter(q['evidence_state'] for q in questions)),'automatic_preliminary_only':True,'human_reviewed':False,'final_gt':False,'api_requests':0,'limits':'Exact reused behavior judgments are rescored against each target sample with frozen Food scoring. Unresolved stays unknown. This automatic conjunction is not human-reviewed or final GT.'}
 if args.check_only:print(json.dumps(summary));return
 out=BASE/args.export;assert out.parent==BASE;out.mkdir(exist_ok=False)
 write(out/'identity.json',{'identity':ident,'definition':definition});jsonl(out/'semantic_labels.jsonl',semantic);jsonl(out/'merged_independent_labels.jsonl',merged);jsonl(out/'questions.jsonl',questions)
 summary['outputs_sha256']={n:sha(out/n) for n in ['semantic_labels.jsonl','merged_independent_labels.jsonl','questions.jsonl']};write(out/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--batch',type=int);p.add_argument('--check-only',action='store_true');p.add_argument('--export',default='validated_v1');run(p.parse_args())
