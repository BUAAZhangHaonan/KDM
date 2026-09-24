import contextlib,io,runpy,json,hashlib,csv
from pathlib import Path
from collections import Counter,defaultdict
with contextlib.redirect_stdout(io.StringIO()):d=runpy.run_path('outputs/analysis/three_model_joint_gt_20260924_v1/compute.py')
globals().update({k:v for k,v in d.items() if not k.startswith('__')})
from workflows.deepseek_annotation_v2.annotate import parse_label
R=Path('outputs/annotations/luna_independent_residual_v1/remaining192_20260924')
e=json.loads((R/'execution.json').read_text());assert e['model']=='gpt-6-luna' and e['reasoning_effort']=='medium' and e['input_sha256']==sha(R/'blind.jsonl') and e['result_sha256']==sha(R/'results.jsonl')
bs=rows(R/'blind.jsonl'); rr=rows(R/'results.jsonl'); B={r['id']:r for r in bs}; D={r['id']:r for r in rr}
assert len(B)==len(D)==len(rr)==188 and set(B)==set(D)
for cp in sorted(R.glob('corrections_v*.jsonl'),key=lambda p:int(p.stem.rsplit('_v',1)[1])):
 for r in rows(cp):
  assert r['id'] in D and r.get('reason');D[r['id']]=dict(D[r['id']],**r)
for k,r in D.items():
 q=B[k];assert r['group_sha256']==q['group_sha256']
 if r['status']=='annotated':parse_label(json.dumps(r['annotation'],ensure_ascii=False),q['answer'])
 else:assert r['status']=='unresolved' and r.get('reason')
mapping=rows(R/'mapping.jsonl');assert len(mapping)==192
updates={}
for m in mapping:
 l=m['source'];r=D[m['group_id']];q=B[m['group_id']];assert q['answer']==l['text'];u=dict(l)
 if r['status']=='annotated':
  a=r['annotation'];score=float(food_correct(a['answer_text'],samples[l['sample_id']]['class'],aliases)) if a['label'].startswith('answer_') else 0.0
  u.update(screening_label=('correct' if score else 'incorrect') if a['label'].startswith('answer_') else a['label'],preliminary_score=score,behavior_label=a['label'],answer_text=a['answer_text'],evidence_span=a['evidence_span'],semantic_status='annotated',residual_source=str(R/'results.jsonl'),residual_group_id=r['id'])
 else:u.update(residual_reason=r['reason'])
 updates[(l['model'],l['key'])]=u
ls=rows(BASE/'merged_independent_labels.jsonl');attempts=defaultdict(list)
for l in ls:
 u=updates.get((l['model'],l['key']),l);attempts[(l['model'],l['sample_id'])].append(u)
newqs=[]; changes=[]
for q in qs:
 aa=attempts[(q['model'],q['sample_id'])];assert len(aa)==10 and len({a['replicate'] for a in aa})==10
 c=Counter(a['screening_label'] for a in aa)
 gt=False if c['correct'] or q['gold_rank']==1 else None if c['unresolved'] else True
 state='independent_correct_answer_observed' if c['correct'] else 'closed_rank_one_no_joint_support' if q['gold_rank']==1 else 'unresolved_independent_answers' if c['unresolved'] else 'joint_deficit_supported_by_automatic_residual_review'
 n=dict(q,counts={k:c[k] for k in ['correct','incorrect','abstain','invalid','unresolved']},preliminary_behavioral_warrant=gt,evidence_state=state,residual_review_source=str(R/'results.jsonl'))
 newqs.append(n)
 if gt!=q['preliminary_behavioral_warrant']:changes.append({'model':q['model'],'sample_id':q['sample_id'],'old_gt':q['preliminary_behavioral_warrant'],'new_gt':gt})
N={(q['model'],q['sample_id']):q for q in newqs};newjoined=[]
for x in joined:
 q=N[(x['model'],x['sample_id'])];y=dict(x,gt=q['preliminary_behavioral_warrant'],gt_state=q['evidence_state'],counts=q['counts']);newjoined.append(y)
newstats=stats(newjoined)
conflicts=[r for r in newjoined if r['gt'] is True and r['correct']]
result={'gt_definition':report['gt_definition'],'rows':newstats,'question_states':dict(Counter(q['evidence_state'] for q in newqs)),'residual_groups':188,'residual_rows':192,'residual_label_counts':dict(Counter(u['screening_label'] for u in updates.values())),'question_changes':changes,'conflict_unique_questions':len({(r['model'],r['sample_id']) for r in conflicts}),'baseline_sources_sha256':report['sources_sha256'],'new_sources_sha256':{str(p):sha(p) for p in [R/'blind.jsonl',R/'mapping.jsonl',R/'results.jsonl',R/'execution.json',*R.glob('corrections_v*.jsonl'),*R.glob('correction_execution_v*.json')]},'human_reviewed':False,'user_requested_operational_gt':True,'scientific_limit':'Fixed ten stochastic answers and candidate rank are operational reference, not proof of inability. Native census and vLLM independent backend differences retained.'}
p=OUT/'resolved_report.json';assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for name,data in [('resolved_questions.jsonl',newqs),('resolved_joined.jsonl',newjoined),('resolved_conflicts.jsonl',conflicts),('residual192_labels.jsonl',list(updates.values())),('merged_independent_labels.jsonl',[u for aa in attempts.values() for u in aa])]:
 with (OUT/name).open('x') as f:
  for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
with (OUT/'unknown81_resolution.tsv').open('x') as f:
 w=csv.writer(f,delimiter='\t');w.writerow(['model','sample_id','split','gold_rank','old_unresolved_attempts','new_unresolved_attempts','new_GT_should_abstain','previously_unresolved_texts','unguided_text','unguided_correct','guided_text','guided_correct'])
 for q in qs:
  if q['preliminary_behavioral_warrant'] is not None:continue
  key=(q['model'],q['sample_id']);n=N[key];c={r['guided']:r for r in newjoined if (r['model'],r['sample_id'])==key}
  txt=[m['source']['text'] for m in mapping if (m['source']['model'],m['source']['sample_id'])==key]
  w.writerow([*key,q['split'],q['gold_rank'],q['counts']['unresolved'],n['counts']['unresolved'],n['preliminary_behavioral_warrant'],json.dumps(txt,ensure_ascii=False),c[False]['text'],c[False]['correct'],c[True]['text'],c[True]['correct']])
print(json.dumps({k:v for k,v in result.items() if k in ['question_states','residual_label_counts','conflict_unique_questions']},ensure_ascii=False))
for r in newstats:
 if r['split'] in ['all','eval']:print(json.dumps(r))
