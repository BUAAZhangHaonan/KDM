import sys,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
sys.path[:0]=['.','src']
from kdm.scoring import food_correct
from kdm.io import stable_hash
OUT=Path('outputs/analysis/three_model_joint_gt_20260924_v1')
BASE=Path('outputs/annotations/luna_independent_v2/completed3_20260923_v1/validated_v1')
CENSUS=Path('outputs/annotations/luna_census_remaining_v1/remaining108_plus2_20260924/census_merged_v1/labels.jsonl')
def rows(p):return [json.loads(x) for x in p.open()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(CENSUS)==json.loads((CENSUS.parent/'metrics.json').read_text())['labels_sha256']
qs=rows(BASE/'questions.jsonl'); assert len(qs)==14544
summary=json.loads((BASE/'summary.json').read_text())
for f,h in summary['outputs_sha256'].items():assert sha(BASE/f)==h
models=sorted({q['model'] for q in qs});Q={(q['model'],q['sample_id']):q for q in qs}; assert len(Q)==len(qs)
samples={x['id']:x for x in rows(Path('data/current/all.jsonl')) if x['dataset']=='food101'}
aliases=json.loads(Path('configs/kdm/food_aliases.json').read_text())
labels={}
for r in rows(CENSUS):
 if r['model'] in models and r['dataset']=='food101':
  k=(r['model'],r['sample_id'],r['guided']);assert k not in labels;labels[k]=r
assert len(labels)==29088
sources={str(CENSUS):sha(CENSUS),str(BASE/'questions.jsonl'):sha(BASE/'questions.jsonl'),str(BASE/'merged_independent_labels.jsonl'):sha(BASE/'merged_independent_labels.jsonl')}
for m in models:
 p=Path(f'outputs/raw/current/{m}/census.jsonl');sources[str(p)]=sha(p)
 for raw in rows(p):
  if raw['sample']['dataset']!='food101':continue
  l=labels[(m,raw['sample']['id'],raw['guided'])]
  assert l['text']==raw['text'] and l['raw_record_sha256']==stable_hash(raw)
  assert raw['sample']==samples[raw['sample']['id']]
joined=[]
for (m,sid),q in Q.items():
 for guided in [False,True]:
  l=labels[(m,sid,guided)];correct=l['label'].startswith('answer_') and food_correct(l['answer_text'],samples[sid]['class'],aliases)
  joined.append(dict(model=m,sample_id=sid,split=q['split'],guided=guided,gt=q['preliminary_behavioral_warrant'],gt_state=q['evidence_state'],counts=q['counts'],gold_rank=q['gold_rank'],label=l['label'],correct=bool(correct),text=l['text'],answer_text=l['answer_text'],census_key=l['key']))
def stats(data):
 result=[]
 for m in models+['ALL']:
  for split in ['all','dev','eval']:
   for guided in [False,True]:
    r=[x for x in data if (m=='ALL' or x['model']==m) and (split=='all' or x['split']==split) and x['guided']==guided];a=[x for x in r if x['label']=='abstain'];c=Counter('unknown' if x['gt'] is None else 'right' if x['gt'] else 'wrong' for x in a)
    result.append(dict(model=m,split=split,guided=guided,n=len(r),gt_positive=sum(x['gt'] is True for x in r),gt_unknown=sum(x['gt'] is None for x in r),actual_abstain=len(a),abstain_right=c['right'],abstain_wrong=c['wrong'],abstain_unknown=c['unknown'],precision_resolved=c['right']/(c['right']+c['wrong']) if c['right']+c['wrong'] else None,precision_lower=c['right']/len(a) if a else None,precision_upper=(c['right']+c['unknown'])/len(a) if a else None,gt_positive_but_correct=sum(x['gt'] is True and x['correct'] for x in r),gt_unknown_but_correct=sum(x['gt'] is None and x['correct'] for x in r)))
 return result
report={'gt_definition':'no correct in10 independent answers AND gold candidate rank>1; unknown retained','scope':'three models Food101; all/dev/eval separated','sources_sha256':sources,'rows':stats(joined),'conflict_unique_questions':sum(any(labels[(m,sid,g)]['label'].startswith('answer_') and food_correct(labels[(m,sid,g)]['answer_text'],samples[sid]['class'],aliases) for g in [False,True]) for (m,sid),q in Q.items() if q['preliminary_behavioral_warrant'] is True),'human_reviewed':False,'gt_use':'user-requested operational reference; original immutable preliminary flags retained'}
(OUT/'baseline_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
for name,data in [('baseline_joined.jsonl',joined),('unknown81_questions.jsonl',[q for q in qs if q['preliminary_behavioral_warrant'] is None]),('gt_positive_correct_conflicts.jsonl',[x for x in joined if x['gt'] is True and x['correct']])]:
 with (OUT/name).open('w') as f:
  for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
for r in report['rows']:
 if r['split'] in ['all','eval']:print(json.dumps(r))
print('CONFLICT_UNIQUE',report['conflict_unique_questions'])
