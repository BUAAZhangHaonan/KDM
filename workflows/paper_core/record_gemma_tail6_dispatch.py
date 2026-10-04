"""Validate the six current Gemma owners against the original1024 task keys."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from kdm.io import atomic_json,file_hash,read_jsonl
B=Path('outputs/paper_core_20261002_dev_viz/viz_native_baselines_20261004_1645')
old=json.loads((ROOT/B/'gemma_tail4_dispatch.json').read_text())
base=json.loads((ROOT/old['source']/'sealed_for_handoff.json').read_text())
original=set(base['completed_keys'])|{r['key'] for r in read_jsonl(ROOT/base['remaining_keys'])}
covered=set(base['completed_keys']);completed=len(covered);assigned=[];seals=[]
for source_card,new_card in [(0,1),(5,4)]:
    source=B/f'gemma_tail4_4028g{source_card}'
    seal=json.loads((ROOT/source/'sealed_for_handoff.json').read_text())
    keys=set(seal['completed_keys']);assert not keys&covered
    covered|=keys;completed+=len(keys);seals.append(str(source/'sealed_for_handoff.json'))
    remaining={r['key'] for r in read_jsonl(ROOT/seal['remaining_keys'])}
    local=set()
    for part,gpu in [(0,source_card),(1,new_card)]:
        plan_path=B/f'gemma_wave2_g{source_card}_part{part}_plan.json'
        plan=json.loads((ROOT/plan_path).read_text());keys=set(plan['keys'])
        assert len(keys)==plan['missing'] and keys and not keys&(covered|local)
        local|=keys
        assigned.append(dict(host='4028-root',gpu=str(gpu),output=str(B/f'gemma_tail6_4028g{gpu}'),
            source_seal=str(source/'sealed_for_handoff.json'),source_seal_sha256=file_hash(ROOT/source/'sealed_for_handoff.json'),
            parts=2,part=part,partition_salt=f'gemma_wave2_4028g{source_card}',expected=len(keys),
            plan=str(plan_path),plan_sha256=file_hash(ROOT/plan_path),planned_keys=plan['keys']))
    assert local==remaining;covered|=local
for item in old['assignments'][2:]:
    keys=set(item['planned_keys']);assert not keys&covered;covered|=keys;assigned.append(item)
assert covered==original and len(covered)==1024
result=dict(passed=True,full_expected=1024,completed_in_sealed_sources=completed,
    currently_assigned=sum(x['expected'] for x in assigned),original_source=old['source'],sealed_parent_sources=seals,
    pairwise_disjoint=True,full_key_union_complete=True,assignments=assigned,
    partition_salt_purpose='avoid nested modulus selecting an empty child; task identity and all scientific settings unchanged')
atomic_json(ROOT/B/'gemma_tail6_dispatch.json',result)
print(json.dumps({**{k:v for k,v in result.items() if k!='assignments'},'assignments':[{k:v for k,v in x.items() if k!='planned_keys'} for x in assigned]}))
