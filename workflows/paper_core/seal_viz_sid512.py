"""Verify and index the disjoint 8+504 SID outputs, without changing scores."""
import argparse,csv,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(Path(__file__).parent)]
from kdm.io import atomic_json,file_hash,read_jsonl,within
from workflows.paper_core.dev_viz import roster

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--pilot',required=True)
p.add_argument('--production',required=True);p.add_argument('--output',required=True);a=p.parse_args()
listing,samples=roster('viz512');expected={s['id']:s for s in samples}
index=[];rows=[];identities=[];receipts=[]
for name,n in ((a.pilot,8),(a.production,504)):
    folder=within(ROOT,name);gate=json.loads((folder/'complete.json').read_text())
    identity=json.loads((folder/'identity.json').read_text());claim=json.loads((folder/'claim.json').read_text())
    raw=folder/'new_predictions.jsonl';items=list(read_jsonl(raw))
    assert gate['passed'] and gate['completed']==gate['expected']==n==len(items)
    assert gate['model']==a.model and file_hash(raw)==gate['raw_sha256']
    assert identity['model']==a.model and identity['fixed_operator']==dict(aggregation_block_1based=2,rank=100,alpha=.5,greedy_support='full_vocab')
    assert identity['signature']['roster_sha256']==file_hash(listing)
    assert [r['sample']['id'] for r in items]==identity['sample_ids']
    for line,row in enumerate(items,1):
        assert row['sample']==expected[row['sample']['id']] and row['config']==identity['config']
        assert row['guided'] is False and row['reference_guided'] is False
        assert row['marker']==row['reference_marker']=='NONE' and row['method']=='sid'
        index.append(dict(model=a.model,sample_id=row['sample']['id'],key=row['key'],
            source_path=str(raw.relative_to(ROOT)),source_line=line,original_host=claim['host'],
            identity_sha256=file_hash(folder/'identity.json'),raw_sha256=gate['raw_sha256']))
    rows.extend(items);identities.append(identity);receipts.append(gate)
assert identities[0]['signature']==identities[1]['signature']
assert len(rows)==len({r['key'] for r in rows})==len({r['sample']['id'] for r in rows})==512
assert {r['sample']['id'] for r in rows}==set(expected)
assert Counter(int(r['sample']['annotated_answerable']) for r in rows)=={0:166,1:346}
out=within(ROOT,a.output);out.mkdir(parents=True,exist_ok=True)
csvpath=out/f'{a.model}_sid512_source_index.csv'
assert not csvpath.exists()
with csvpath.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=index[0]);writer.writeheader();writer.writerows(index)
result=dict(model=a.model,method='sid',implementation='native_unguided_author_core',n=512,
    zero_duplicate_keys=True,zero_missing_ids=True,pilot_production_overlap=0,
    official_unanswerable=166,official_answerable=346,raw_complete=True,
    semantic_score_acceptance=False,scoring_owner='root CPU scoring queue',
    signature=identities[0]['signature'],source_index=str(csvpath.relative_to(ROOT)),
    source_index_sha256=file_hash(csvpath),raw_receipts=receipts)
atomic_json(out/f'{a.model}_sid512_acceptance.json',result)
print(json.dumps(dict(model=a.model,raw_complete=True,n=512,zero_overlap=True,score_acceptance=False)),flush=True)
