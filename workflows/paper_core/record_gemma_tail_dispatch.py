"""Record the four assigned Gemma continuations from the one sealed source."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from kdm.io import atomic_json,file_hash,read_jsonl
B=Path('outputs/paper_core_20261002_dev_viz/viz_native_baselines_20261004_1645')
source=B/'gemma_native_dola_deco_512'
seal=json.loads((ROOT/source/'sealed_for_handoff.json').read_text())
missing=list(read_jsonl(ROOT/source/'sealed_remaining_keys.jsonl'))
expected={row['key'] for row in missing};completed=set(seal['completed_keys'])
locations=[('4028-root','0','gemma_tail4_4028g0'),('4028-root','5','gemma_tail4_4028g5'),
           ('4029','2','gemma_tail4_4029g2_registryfix'),('6403','1','gemma_tail4_6403g1_registryfix')]
all_keys=set();assignments=[]
for part,(host,gpu,name) in enumerate(locations):
    path=ROOT/B/f'gemma_tail_plan_{part}.json';plan=json.loads(path.read_text());keys=set(plan['keys'])
    assert len(keys)==plan['missing'] and not keys&all_keys and not keys&completed
    assert plan['dtype']=='bfloat16'
    counts={method:sum(row['key'] in keys and row['method']==method for row in missing) for method in ('dola','deco')}
    assignments.append(dict(part=part,parts=4,host=host,gpu=gpu,output=str(B/name),expected=len(keys),
       method_counts=counts,plan=str(path.relative_to(ROOT)),plan_sha256=file_hash(path),planned_keys=plan['keys']))
    all_keys|=keys
assert all_keys==expected and len(all_keys)+len(completed)==1024
receipt=dict(passed=True,source=str(source),source_seal_sha256=file_hash(ROOT/source/'sealed_for_handoff.json'),
    completed=len(completed),remaining=len(expected),zero_completed_intersection=True,
    pairwise_disjoint=True,remaining_union_complete=True,assignments=assignments,
    source_stopped_once=True,method_and_generation_parameters_changed=False,
    zero_output_failed_attempts=['gemma_tail4_4029g2','gemma_tail4_6403g1'],
    location_registry_correction='model_hosts scalar registered host plus explicit cross-host runtime overrides')
atomic_json(ROOT/B/'gemma_tail4_dispatch.json',receipt)
print(json.dumps({**{k:v for k,v in receipt.items() if k!='assignments'},
    'assignments':[{k:v for k,v in item.items() if k!='planned_keys'} for item in assignments]}))
