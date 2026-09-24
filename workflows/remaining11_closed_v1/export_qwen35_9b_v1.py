import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from workflows.acceleration_v4 import vllm_closed_v6 as core
from kdm.io import read_jsonl,file_hash,atomic_json
from importlib.metadata import version
model='qwen35_9b';specpath=ROOT/f'configs/runtime/{model}.json';spec=json.loads(specpath.read_text());actual={k:version(k) for k in spec['versions']};assert actual==spec['versions']
core.PATHS[model]=spec['kwargs']['model_path'];sc=core.Scorer(model)
src=ROOT/f'outputs/raw/probes_v2/food_closed_20260923_v3/{model}/closed.jsonl';identity=json.loads(src.with_name('closed.identity.json').read_text());assert identity['definition']['backend_spec_sha256']==file_hash(specpath)
old=list(read_jsonl(src));indices=[i*(len(old)-1)//15 for i in range(16)];rows=[]
for i in indices:
 r=old[i];assert r['identity']==identity['identity'] and len(r['candidate_scores'])==101
 assert [s['label'] for s in r['candidate_scores']]==sc.names
 assert [s['n_tokens'] for s in r['candidate_scores']]==list(map(len,sc.tokens))
 _,prompt,base,inp=sc.prepare(r['sample']);assert prompt==r['prompt']
 rows.append({'reference':r,'unexpanded_prompt_ids':base,'expanded_prompt_ids':inp['input_ids'][0].tolist(),'processor_summary':core.summarize(dict(inp)),'candidates_tokens':sc.tokens})
p=ROOT/'outputs/records/remaining11_closed_v1/qwen35_9b_v1/native_closed16_reference.json'
assert not p.exists()
atomic_json(p,{'model':model,'backend':spec,'backend_spec_sha256':file_hash(specpath),'source_blobs':identity['definition']['source_blobs'],'source_raw_sha256':file_hash(src),'source_identity_sha256':file_hash(src.with_name('closed.identity.json')),'source_identity':identity,'native_cpu_export_versions':actual,'rows':rows,'selection_rule':'16 evenly spaced row indices over existing1731 TF scores; engineering input and score reference','indices':indices})
print('NATIVE_CPU16_COMPLETE',indices,flush=True)
