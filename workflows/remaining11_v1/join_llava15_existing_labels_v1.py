"""Bind completed LLaVA15 scores to existing exact free labels, without relabeling."""
from pathlib import Path
import sys,json,math,hashlib,collections
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'src')]
from workflows.independent_v5 import postprocess as b
from kdm.io import read_jsonl,file_hash,stable_hash
model='llava15_7b'
cr=R/'outputs/records/acceleration_v4/remaining11_llava15_7b_closed_20260924_spawn_v2'/model
ir=R/'outputs/records/acceleration_v4/extra11_independent_food_llava15_7b_v1'/model
lab=R/'outputs/annotations/remaining11_v1/llava15_independent_free_20260924_v1'
out=R/'outputs/annotations/remaining11_v1/llava15_joint_existing_free_20260924_v1'
samples={s['id']:s for s in read_jsonl(R/'data/current/all.jsonl') if s['dataset']=='food101'}
assert len(samples)==4848 and collections.Counter(s['split'] for s in samples.values())=={'dev':2424,'eval':2424}
cd=b.load(cr/'complete.json');cp=b.load(cr/'progress.json');raw=R/cp['output'];cs=b.load(raw.with_suffix('.identity.json'));definition=cs['definition']
assert cd['complete'] and cd['closed']==4848 and cp['status']=='complete'
assert cd['identity']==definition and cs['identity']==stable_hash(definition) and definition['model']==model
assert definition['manifest_sha256']==file_hash(R/'data/current/all.jsonl') and cd['output_sha256']==file_hash(raw)
matcher=b.Matcher();ranks={}
for row in read_jsonl(raw):
 sid=row['sample']['id'];assert sid not in ranks and row['sample']==samples[sid] and row['model']==model and row['identity']==cs['identity'] and row['status']=='ok'
 assert row['key']==stable_hash([model,sid,'closed'])
 ranks[sid]=matcher.classify(row)['gold_rank']
 for s in row['candidate_scores']:
  assert type(s['n_tokens']) is int and s['n_tokens']>0 and math.isfinite(s['sum_logp']) and math.isclose(s['sum_logp']/s['n_tokens'],s['mean_logp'],rel_tol=1e-8,abs_tol=1e-8)
assert set(ranks)==set(samples)
idone=b.load(ir/'complete.json');ip=b.load(ir/'progress.json');indraw=R/ip['output'];side=b.load(indraw.with_suffix('.identity.json'));idef=side['definition']
assert idone['generation_complete'] and idone['independent']==48480 and ip['status']=='generation_complete'
assert side['identity']==stable_hash(idef)==idone['identity'] and idef['closed_completion']['status']=='pending_separate_closed_cohort'
assert idef['base_config']==b.CONFIG and idef['manifest_sha256']==file_hash(R/'data/current/all.jsonl')
ls=b.load(lab/'summary.json');assert ls['raw_sha256']==idone['output_sha256']==file_hash(indraw) and ls['label_sha256']==file_hash(lab/'labels.jsonl')
assert ls['rule_sha256']==file_hash(R/'workflows/independent_v5/postprocess.py')
labels=list(read_jsonl(lab/'labels.jsonl'));assert len(labels)==48480
attempts={sid:{} for sid in samples};seen=set();counts=collections.Counter()
with indraw.open('rb') as f:
 for n,content in enumerate(f,1):
  row=json.loads(content);sid,rep=b.inspect_attempt(row,model,samples,side['identity'],idef['eos_token_ids']);v=labels[n-1]
  assert v['source_line']==n and v['source_row_sha256']==hashlib.sha256(content).hexdigest() and v['key']==row['key'] and v['source_identity']==side['identity'] and v['text']==row['text'] and v['sample_id']==sid and v['replicate']==rep and v['seed']==row['seed']
  assert row['key'] not in seen and rep not in attempts[sid];seen.add(row['key']);attempts[sid][rep]=v;counts[v['screening_label']]+=1
assert len(seen)==48480 and all(set(v)==set(range(10)) for v in attempts.values())
questions=[b.question_summary(model,s,attempts[sid],ranks[sid]) for sid,s in samples.items()]
out.mkdir(exist_ok=False)
with (out/'questions.jsonl').open('x') as f:
 for q in questions:f.write(json.dumps(q,ensure_ascii=False,allow_nan=False)+'\n')
summary={'model':model,'questions':4848,'answers':48480,'closed_scores':4848,'counts':dict(counts),'question_states':dict(collections.Counter(q['evidence_state'] for q in questions)),'labels_reused_exactly':str(lab.relative_to(R)),'labels_sha256':file_hash(lab/'labels.jsonl'),'closed_raw_sha256':file_hash(raw),'closed_identity':cs['identity'],'independent_raw_sha256':file_hash(indraw),'independent_identity':side['identity'],'entry_sha256':file_hash(__file__),'questions_sha256':file_hash(out/'questions.jsonl'),'late_binding':True,'source_sidecars_modified':False,'automatic_preliminary_only':True,'final_gt':False,'human_reviewed':False,'api_requests':0,'gpu_used':False}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
