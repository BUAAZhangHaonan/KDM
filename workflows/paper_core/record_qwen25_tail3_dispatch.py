"""Record three true remaining-key owners after Qwen2.5's two source seals."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from kdm.io import atomic_json,file_hash,read_jsonl
B=Path('outputs/paper_core_20261002_dev_viz/viz_native_baselines_20261004_1645')
sources=['qwen25_dola_deco','qwen25_A100_remaining_native'];covered=set();completed=0
for name in sources:
    seal=json.loads((ROOT/B/name/'sealed_for_handoff.json').read_text());keys=set(seal['completed_keys'])
    assert not keys&covered;covered|=keys;completed+=len(keys)
missing={r['key'] for r in read_jsonl(ROOT/B/sources[-1]/'sealed_remaining_keys.jsonl')}
expected=covered|missing;assert len(expected)==1024 and not covered&missing
assigned=[];new=set()
for part,host,gpu,name in [(0,'4028-root','1','qwen25_tail3_4028g1'),
                         (1,'4028-root','4','qwen25_tail3_4028g4'),
                         (2,'6403','0','qwen25_tail3_A100slot1')]:
    path=B/f'qwen25_tail3_part{part}_plan.json';plan=json.loads((ROOT/path).read_text());keys=set(plan['keys'])
    assert keys and len(keys)==plan['missing'] and not keys&(covered|new);new|=keys
    assigned.append(dict(part=part,parts=3,host=host,gpu=gpu,output=str(B/name),expected=len(keys),
        plan=str(path),plan_sha256=file_hash(ROOT/path),planned_keys=plan['keys']))
assert new==missing and covered|new==expected
result=dict(passed=True,model='qwen25vl',full_expected=1024,completed_in_sealed_sources=completed,
    actual_remaining=len(missing),source_seals=[str(B/n/'sealed_for_handoff.json') for n in sources],
    pairwise_disjoint=True,zero_completed_intersection=True,full_key_union_complete=True,assignments=assigned)
atomic_json(ROOT/B/'qwen25_tail3_dispatch.json',result)
print(json.dumps({**{k:v for k,v in result.items() if k!='assignments'},
    'assignments':[{k:v for k,v in x.items() if k!='planned_keys'} for x in assigned]}))
