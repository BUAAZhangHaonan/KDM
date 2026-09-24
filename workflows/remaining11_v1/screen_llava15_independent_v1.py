"""Free labels only for a verified completed cohort; no pending class-score GT."""
from pathlib import Path
import sys,json,hashlib,collections
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'src')]
from workflows.independent_v5 import postprocess as b
from kdm.io import file_hash,stable_hash,read_jsonl
D=R/'outputs/records/acceleration_v4/extra11_independent_food_llava15_7b_v1/llava15_7b';done=b.load(D/'complete.json');progress=b.load(D/'progress.json');raw=R/progress['output'];side=b.load(raw.with_suffix('.identity.json'));definition=side['definition'];proof=b.load(R/'outputs/records/independent_k100_llava15_v1/completed_verification.json')
assert done['generation_complete'] and done['independent']==48480 and progress['status']=='generation_complete'
assert side['identity']==done['identity']==stable_hash(definition) and definition['model']=='llava15_7b'
assert definition['manifest_sha256']==file_hash(R/'data/current/all.jsonl') and definition['base_config']==b.CONFIG
assert definition['closed_completion']['status']=='pending_separate_closed_cohort'
assert done['output_sha256']==proof['raw_sha256']==file_hash(raw)
out=R/'outputs/annotations/remaining11_v1/llava15_independent_free_20260924_v1';out.mkdir(exist_ok=False)
samples={s['id']:s for s in read_jsonl(R/'data/current/all.jsonl') if s['dataset']=='food101'};matcher=b.Matcher();seen=set();reps=collections.defaultdict(set);counts=collections.Counter()
with raw.open('rb') as f,(out/'labels.partial.jsonl').open('x') as sink:
 for line,content in enumerate(f,1):
  assert content.endswith(b'\n');row=json.loads(content);sid,rep=b.inspect_attempt(row,'llava15_7b',samples,side['identity'],definition['eos_token_ids']);assert row['key'] not in seen and rep not in reps[sid];seen.add(row['key']);reps[sid].add(rep)
  label=b.label_attempt(matcher,row);counts[label['screening_label']]+=1
  value=dict(schema='kdm_independent_food_free_label_v1',model='llava15_7b',sample_id=sid,split=samples[sid]['split'],replicate=rep,seed=row['seed'],key=row['key'],text=row['text'],source_identity=side['identity'],source_line=line,source_row_sha256=hashlib.sha256(content).hexdigest(),**label)
  sink.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
assert len(seen)==48480 and len(reps)==4848 and all(v==set(range(10)) for v in reps.values()) and file_hash(raw)==done['output_sha256']
(out/'labels.partial.jsonl').rename(out/'labels.jsonl')
summary={'model':'llava15_7b','generated_answers':48480,'screened_answers':48480,'counts':dict(counts),'question_gt_generated':False,'reason':'Candidate scoring still pending; no joint GT generated','final_gt':False,'human_reviewed':False,'raw_sha256':file_hash(raw),'source_verification_sha256':file_hash(R/'outputs/records/independent_k100_llava15_v1/completed_verification.json'),'entry_sha256':file_hash(__file__),'label_sha256':file_hash(out/'labels.jsonl'),'rule_sha256':file_hash(R/'workflows/independent_v5/postprocess.py')};(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
