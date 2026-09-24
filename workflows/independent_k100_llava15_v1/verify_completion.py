from pathlib import Path
import sys,json,hashlib,collections,time
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'src')]
from workflows.independent_v5 import postprocess as b
from kdm.io import read_jsonl,file_hash,stable_hash
D=R/'outputs/records/acceleration_v4/extra11_independent_food_llava15_7b_v1/llava15_7b';done=b.load(D/'complete.json');progress=b.load(D/'progress.json');raw=R/progress['output'];sidepath=raw.with_suffix('.identity.json');side=b.load(sidepath);definition=side['definition']
assert done['generation_complete'] and done['independent']==48480 and progress['status']=='generation_complete'
assert side['identity']==done['identity']==stable_hash(definition)
assert definition['model']=='llava15_7b' and definition['base_config']==b.CONFIG and definition['manifest_sha256']==file_hash(R/'data/current/all.jsonl')
assert definition['closed_completion']['status']=='pending_separate_closed_cohort'
samples={s['id']:s for s in read_jsonl(R/'data/current/all.jsonl') if s['dataset']=='food101'};assert len(samples)==4848
seen=set();reps=collections.defaultdict(set);seeds=collections.defaultdict(set)
for row in read_jsonl(raw):
 sid,rep=b.inspect_attempt(row,'llava15_7b',samples,side['identity'],definition['eos_token_ids'])
 assert row['key'] not in seen and rep not in reps[sid] and row['seed'] not in seeds[sid]
 seen.add(row['key']);reps[sid].add(rep);seeds[sid].add(row['seed'])
assert len(seen)==48480 and set(reps)==set(samples) and all(v==set(range(10)) for v in reps.values())
assert file_hash(raw)==done['output_sha256']
B=R/'outputs/records/independent_k100_llava15_v1';files=[*B.rglob('*'),*D.rglob('*'),raw,sidepath,* (R/'outputs/records/remaining11_llava15_v1/independent_native16').rglob('*')]
files=[p for p in files if p.is_file() and p.name!='completed_verification.json']
receipt={'verified':True,'model':'llava15_7b','rows':48480,'questions':4848,'replicates_per_question':10,'raw_sha256':file_hash(raw),'sidecar_sha256':file_hash(sidepath),'source_identity':side['identity'],'native_eos_seed_config_prompt_logprob_verified':True,'final_gt':False,'candidate_pending':True,'source_files':{str(p.relative_to(R)):file_hash(p) for p in files},'finished_unix':time.time()}
(B/'completed_verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'}));print('SOURCE_FILES',len(receipt['source_files']))
