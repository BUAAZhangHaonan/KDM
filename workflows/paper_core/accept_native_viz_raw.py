"""Accept complete native Viz512 raw unions, separately from semantic scoring."""
import argparse,csv,json,math,sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from kdm.io import atomic_json,file_hash,read_jsonl,stable_seed
from kdm.pipeline import task_id
from kdm.decoding import DecodeConfig
from kdm.prompts import task_prompt
from workflows.paper_core.dev_viz import roster
from workflows.paper_core.viz_author_native import native_task,now
B=Path('outputs/paper_core_20261002_dev_viz/viz_native_baselines_20261004_1645')
SOURCES={
 'phi35':(['phi_native_dola_deco_512'],[]),
 'llava16_mistral':(['mistral_native_dola_deco_512'],[]),
 'internvl35_8b':(['intern_native_dola_deco_512'],[]),
 'gemma3_4b':([f'gemma_tail6_4028g{x}' for x in (0,1,4,5)]+['gemma_tail4_4029g2_registryfix','gemma_tail4_6403g1_registryfix'],
    ['gemma_native_dola_deco_512','gemma_tail4_4028g0','gemma_tail4_4028g5']),
 'qwen25vl':(['qwen25_tail3_4028g1','qwen25_tail3_4028g4','qwen25_tail3_A100slot1'],
    ['qwen25_dola_deco','qwen25_A100_remaining_native']),
}
p=argparse.ArgumentParser();p.add_argument('--model',choices=SOURCES,required=True);a=p.parse_args()
listing,samples=roster('viz512');expected={task_id(a.model,native_task(s,m)):native_task(s,m) for m in ('dola','deco') for s in samples}
assert len(expected)==1024
parts=[];authorities=[]
regular,sealed=SOURCES[a.model]
for name in regular:
    folder=ROOT/B/name;authority=folder/'complete.json';complete=json.loads(authority.read_text())
    assert complete['passed'] and complete['completed']==complete['expected']
    count=0
    for path in sorted(folder.glob('*/complete.json')):
        c=json.loads(path.read_text());assert c['passed'] and c['completed']==c['expected']
        parts.append((path.parent/'new_predictions.jsonl',c['raw_sha256'],c['completed'],str(path.relative_to(ROOT))))
        count+=c['completed']
    assert count==complete['completed'];authorities.append(str(authority.relative_to(ROOT)))
for name in sealed:
    authority=ROOT/B/name/'sealed_for_handoff.json';seal=json.loads(authority.read_text())
    assert seal['passed'] and seal['producer_exited'] and seal['zero_completed_remaining_intersection'] and seal['original_key_union_complete']
    for item in seal['source_parts']:
        parts.append((ROOT/item['raw'],item['raw_sha256'],item['completed'],str(authority.relative_to(ROOT))))
    authorities.append(str(authority.relative_to(ROOT)))
seen=set();index=[];counts={'dola':0,'deco':0}
for raw,digest,n,authority in parts:
    assert file_hash(raw)==digest
    rows=list(read_jsonl(raw));assert len(rows)==n
    identity=json.loads((raw.parent/'identity.json').read_text());ledger=json.loads(raw.with_suffix('.identity.json').read_text())
    assert identity['model']==a.model and identity['roster_sha256']==file_hash(listing)
    for line,row in enumerate(rows,1):
        key=row['key'];assert key not in seen and key in expected;seen.add(key);task=expected[key]
        assert row['sample']==task['sample'] and row['identity']==ledger['identity'] and row['status']=='ok'
        assert all(row[k]==task[k] for k in ('method','kind','marker','reference_marker','guided','reference_guided','replicate','implementation_revision'))
        assert row['config']==asdict(DecodeConfig(method=row['method'])) and row['seed']==stable_seed(row['sample']['id'],a.model,0)
        assert row['prompt']==task_prompt(row['sample']['question'],guided=False)
        assert 0<len(row['tokens'])<=32 and len(row['tokens'])==len(row['selected_log_probabilities'])
        assert math.isfinite(row['first_probability']) and all(math.isfinite(x) for x in row['selected_log_probabilities'])
        counts[row['method']]+=1
        index.append(dict(model=a.model,method=row['method'],sample_id=row['sample']['id'],key=key,
            raw=str(raw.relative_to(ROOT)),line=line,raw_sha256=digest,identity=ledger['identity'],acceptance_source=authority))
assert seen==set(expected) and counts=={'dola':512,'deco':512}
out=ROOT/B/'accepted_raw1024';out.mkdir(exist_ok=True)
csv_path=out/f'{a.model}_native_source_index.csv'
with csv_path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
receipt=dict(model=a.model,raw_complete=True,completed=1024,expected=1024,method_counts=counts,
    zero_duplicate=True,zero_missing=True,all_original_unguided=True,finite_probabilities=True,
    frozen_seeds_prompts_configs_verified=True,source_index=str(csv_path.relative_to(ROOT)),
    source_index_sha256=file_hash(csv_path),source_authorities=authorities,score_acceptance=False,accepted_utc=now())
atomic_json(out/f'{a.model}_native_acceptance.json',receipt);print(json.dumps(receipt))
