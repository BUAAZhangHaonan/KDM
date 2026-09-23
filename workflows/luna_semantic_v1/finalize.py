"""Validate Luna's explicitly authored blinded judgments and map to exact source rows."""
import argparse,hashlib,json,math,sys
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/annotations/luna_semantic_v1/unresolved_9047_20260923_v1'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'workflows/deepseek_annotation_v2')]
from annotate import parse_label
from kdm.scoring import food_correct

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(j):return hashlib.sha256(json.dumps(j,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def read(p):return [json.loads(s) for s in p.open()]
def write(p,obj):
 with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def jsonl(p,rows):
 with p.open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def collect():
 meta=json.loads((BASE/'manifest.json').read_text())
 assert sha(BASE/'blind_queue.jsonl')==meta['queue_sha256'] and sha(BASE/'source_mapping.jsonl')==meta['mapping_sha256']
 assert sha(BASE/'RUBRIC.txt')==meta['rubric_sha256']
 queue=read(BASE/'blind_queue.jsonl');byid={r['id']:r for r in queue};results={};files={};original_errors=[]
 inputs=list(sorted((BASE/'batches').glob('input_*.jsonl')))
 for p in inputs:
  expected={r['id'] for r in read(p)};result=BASE/'results'/p.name.replace('input_','batch_')
  assert result.is_file(),f'Missing batch {result.name}'
  batch=read(result);assert len(batch)==len(expected) and {r['id'] for r in batch}==expected,f'Batch coverage differs: {p.name}'
  files[str(result.relative_to(BASE))]=sha(result)
  for r in batch:
   i=r['id'];assert i not in results and r['group_sha256']==byid[i]['group_sha256']
   if r.get('status')=='unresolved':
    assert isinstance(r.get('reason'),str) and r['reason'].strip() and 'annotation' not in r
   else:
    assert isinstance(r.get('annotation'),dict)
    try:parse_label(json.dumps(r['annotation'],ensure_ascii=False),byid[i]['answer'])
    except ValueError as exc:original_errors.append({'id':i,'file':str(result.relative_to(BASE)),'error':str(exc)})
   results[i]=r
 assert len(results)==len(queue)==meta['exact_question_answer_groups']
 for correction_path in sorted((BASE/'results').glob('corrections_v*.jsonl'),key=lambda p:int(p.stem.rsplit('_v',1)[1])):
  corrections=read(correction_path);changed=set()
  files[str(correction_path.relative_to(BASE))]=sha(correction_path)
  for r in corrections:
   i=r['id'];assert i in results and i not in changed and r['group_sha256']==byid[i]['group_sha256']
   assert isinstance(r.get('reason'),str) and r['reason'].strip()
   if r.get('status')=='unresolved':assert 'annotation' not in r
   else:parse_label(json.dumps(r['annotation'],ensure_ascii=False),byid[i]['answer'])
   changed.add(i);results[i]=r
 for i,r in results.items():
  if r.get('status')!='unresolved':parse_label(json.dumps(r['annotation'],ensure_ascii=False),byid[i]['answer'])
 return meta,byid,results,files,original_errors

def run(check_only=False,export_name="validated_v3"):
 assert export_name in {"validated_v3"},"Original provisional v1 remains preserved"
 meta,queue,results,files,original_errors=collect()
 review_receipts={};reviewed=set()
 for name,expected in [('second_pass_review_001_007.json',set(range(1,701))),('second_pass_review_008_019.json',set(range(701,1869)))]:
  receipt=BASE/name;review=json.loads(receipt.read_text())
  assert review['model']=='gpt-6-luna' and review['reasoning_effort']=='medium'
  assert review['human_reviewed'] is False and review['final_gt'] is False
  ids=review['reviewed_ids'];assert len(ids)==len(expected) and set(ids)==expected and not reviewed.intersection(ids)
  reviewed.update(ids)
  for field in ['input_sha256','original_batch_sha256','correction_sha256']:
   for path,expected_sha in review[field].items():assert sha(BASE/path)==expected_sha,'Second pass evidence changed'
  review_receipts[name]=sha(receipt)
 assert reviewed==set(queue)
 minimal_path=BASE/'minimal_span_review_47.json';minimal=json.loads(minimal_path.read_text())
 expected_minimal={r['id'] for r in read(BASE/'minimal_span_candidate_review.jsonl')}
 assert len(expected_minimal)==47 and len(minimal['reviewed_ids'])==47 and set(minimal['reviewed_ids'])==expected_minimal
 assert minimal['model']=='gpt-6-luna' and minimal['reasoning_effort']=='medium' and minimal['human_reviewed'] is False
 review_receipts['minimal_span_review_47.json']=sha(minimal_path)
 review_receipts['minimal_span_candidate_review.jsonl']=sha(BASE/'minimal_span_candidate_review.jsonl')
 mapping=read(BASE/'source_mapping.jsonl');assert len(mapping)==9047
 aliases=json.loads((ROOT/'configs/kdm/food_aliases.json').read_text())
 source_labels=[];raws={};source_proofs=[]
 for source in meta['sources']:
  p=ROOT/source['screening_directory'];assert sha(p/'labels.jsonl')==source['labels_sha256'] and sha(p/'summary.json')==source['summary_sha256']
  original_summary=json.loads((p/'summary.json').read_text())
  for rule_path,expected_sha in original_summary['rules_sha256'].items():assert sha(ROOT/rule_path)==expected_sha,'Original scoring/screening rule changed'
  labels=read(p/'labels.jsonl');source_labels+=labels;byline={r['source_line']:r for r in labels}
  proof=source['source_prefix'];raw=ROOT/proof['source_path'];h=hashlib.sha256();nbytes=0
  with raw.open('rb') as f:
   for number,line in enumerate(f,1):
    if nbytes>=proof['source_prefix_bytes']:break
    h.update(line);nbytes+=len(line)
    if number not in byline:continue
    item=byline[number];r=json.loads(line);pair=(item['model'],item['key']);assert pair not in raws
    assert line.endswith(b'\n') and hashlib.sha256(line).hexdigest()==item['source_row_sha256']
    assert r['key']==item['key'] and r['identity']==item['source_identity'] and r['sample']['id']==item['sample_id'];raws[pair]=r
  assert nbytes==proof['source_prefix_bytes'] and h.hexdigest()==proof['source_prefix_sha256']
  source_proofs.append(proof)
 assert len(source_labels)==len(raws)==26664
 definition={'schema':'kdm_luna_semantic_9047_v1','model':'gpt-6-luna','reasoning_effort':'medium','subagent':'/root/luna_semantic_9047','second_pass_subagents':['/root/luna_semantic_9047','/root/luna_span_008_013'],'second_pass_receipts_sha256':review_receipts,'queue_manifest_sha256':sha(BASE/'manifest.json'),'result_files_sha256':files,'original_batch_validation_errors_preserved':original_errors,'effective_annotations_validated':True,'rubric_sha256':meta['rubric_sha256'],'source_mapping_sha256':meta['mapping_sha256'],'scoring_sha256':sha(ROOT/'src/kdm/scoring.py'),'aliases_sha256':sha(ROOT/'configs/kdm/food_aliases.json'),'finalizer_sha256':sha(Path(__file__)),'human_reviewed':False,'final_gt':False,'separate_api_calls':0}
 identity=digest(definition);annotations=[];by_pair={}
 for m in mapping:
  l=m['source_label'];pair=(l['model'],l['key']);raw=raws[pair];q=queue[m['id']];r=results[m['id']]
  assert q['question']==raw['sample']['question'] and q['answer']==raw['text'] and q['group_sha256']==m['group_sha256']
  assert pair not in by_pair and l['screening_label']=='needs_confirmation'
  item={'schema':'kdm_luna_semantic_row_v1','annotation_identity':identity,'group_id':m['id'],'group_sha256':m['group_sha256'],'model':l['model'],'key':l['key'],'sample_id':l['sample_id'],'text':raw['text'],'source_identity':l['source_identity'],'source_row_sha256':l['source_row_sha256'],'source_path':l['source_path'],'source_line':l['source_line'],'human_reviewed':False,'final_gt':False}
  if r.get('status')=='unresolved':item.update(status='unresolved',reason=r['reason'],score=None)
  else:
   a=r['annotation'];score=float(food_correct(a['answer_text'],raw['sample']['class'],aliases)) if a['label'].startswith('answer_') else 0.0
   item.update(status='annotated',**a,score=score)
  annotations.append(item);by_pair[pair]=item
 merged=[];counts=Counter();groups=defaultdict(Counter)
 fields=['model','stage','kind','method','guided','reference_guided','marker','reference_marker']
 for label in source_labels:
  pair=(label['model'],label['key']);semantic=by_pair.get(pair)
  if semantic:
   state=semantic.get('label','unresolved');score=semantic['score'];mode='luna_semantic'
  else:
   state={'abstain':'abstain','correct':'answer_assertive','incorrect':'answer_assertive'}[label['screening_label']];score=label['preliminary_score'];mode='original_exact_match'
  item={**label,'derived_label':state,'derived_score':score,'annotation_mode':mode,'semantic_annotation_identity':identity if semantic else None};merged.append(item)
  group=tuple(label[k] for k in fields)
  for c in [counts,groups[group]]:
   c['rows']+=1;c[state]+=1;c['correct_score_sum']+=score if score is not None else 0
 def metrics(c):
  d=dict(c);n=c['rows'];u=c['unresolved'];d.update(accuracy_lower=c['correct_score_sum']/n,accuracy_upper=(c['correct_score_sum']+u)/n,abstention_lower=c['abstain']/n,abstention_upper=(c['abstain']+u)/n);return d
 summary={'schema':'kdm_luna_semantic_9047_summary_v1','annotation_identity':identity,'source_unresolved_rows':9047,'unique_question_answer_groups':len(queue),'semantic_judgments':dict(Counter(r.get('annotation',{}).get('label','unresolved') for r in results.values())),'expanded_labels':dict(Counter(r.get('label','unresolved') for r in annotations)),'semantic_rows_annotated':sum(r['status']=='annotated' for r in annotations),'semantic_rows_unresolved':sum(r['status']=='unresolved' for r in annotations),'full_stage_rows':26664,'full_stage_counts':metrics(counts),'conditions':[{**dict(zip(fields,k)),**metrics(v)} for k,v in sorted(groups.items())],'source_proofs':source_proofs,'human_reviewed':False,'final_gt':False,'separate_api_calls':0,'limits':'Only the 9047 unresolved rows received blinded Luna behavior judgments. Other rows retain exact-match screening. Correctness follows the frozen Food alias rule applied to extracted answer spans; the judge did not receive ground truth. Bounds cover remaining unresolved judgments, not semantic judge errors. No human review or justified-abstention GT claimed.'}
 if check_only:print(json.dumps({'valid':True,'source_rows':9047,'groups':len(queue),'annotated':summary['semantic_rows_annotated'],'unresolved':summary['semantic_rows_unresolved']}));return
 out=BASE/export_name;out.mkdir(exist_ok=False)
 write(out/'identity.json',{'identity':identity,'definition':definition});jsonl(out/'semantic_labels.jsonl',annotations);jsonl(out/'merged_stage_labels.jsonl',merged)
 summary['output_sha256']={name:sha(out/name) for name in ['semantic_labels.jsonl','merged_stage_labels.jsonl']};write(out/'summary.json',summary)
 lines=['# Luna automatic semantic annotation','',f"Requested source rows: 9047. Exact blinded groups: {len(queue)}.",f"Annotated source rows: {summary['semantic_rows_annotated']}. Remaining unresolved: {summary['semantic_rows_unresolved']}.",'', '| Behavior | Source rows |','|---|---:|']
 lines += [f'| {k} | {v} |' for k,v in sorted(summary['expanded_labels'].items())]
 lines += ['', 'Judge: GPT-6 Luna, medium reasoning, Codex subagent. Original judgments and versioned corrections retained. Full input coverage and exact evidence spans checked. Only these 9047 rows were semantically judged; other merged rows retain original exact-match screening.', '', 'These are automatic answer-behavior labels, not human review and not a determination that a model should abstain. Frozen Food correctness scoring is applied separately to extracted spans. No separate API request was made.']
 (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'out':str(out),'annotated':summary['semantic_rows_annotated'],'unresolved':summary['semantic_rows_unresolved'],'counts':summary['expanded_labels']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args();run(a.check_only)
