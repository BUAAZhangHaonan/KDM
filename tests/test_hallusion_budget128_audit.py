"""CPU fixtures verify migrated namespaces and replica/ownership boundaries."""
import json,shutil
from pathlib import Path
import pytest
from kdm.io import file_hash,stable_hash
from workflows.hallusion_blind import audit128 as audit

def fixture_plan():
    condition={'model':'fixture','method':'direct'}
    return {'fixture':{'identity':'fixture_identity',
        'definition':{'source_sha256':{'fixture_source':'fixed_sha'}},
        'tasks':{k:{} for k in ('key_a','key_b','key_c')}}},[condition]

def chunk(base,namespace,keys=('key_a',),claim='fixture_claim',host=None,text='fixture'):
    modelbase=(base/'collected'/host if host else base)/namespace/'fixture'
    run=modelbase/'claims'/claim;run.mkdir(parents=True)
    owner={'identity':'fixture_identity','source_sha256':{'fixture_source':'fixed_sha'},
           'keys':list(keys),'claim_id':claim}
    ownerpath=run/'owner.json';ownerpath.write_text(json.dumps(owner))
    cid=stable_hash({'model':'fixture','method':'direct'})
    rows=[{'key':key,'condition_identity':cid,'truncated':False,'terminated':True,
           'generation_source':'new_budget128','text':text} for key in keys]
    raw=run/'chunk_00000.jsonl';raw.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    receipt={'status':'complete','identity':'fixture_identity','keys':list(keys),
        'raw_path':str(raw.relative_to(modelbase)),'raw_sha256':file_hash(raw),
        'owner_path':str(ownerpath.relative_to(modelbase)),'owner_sha256':file_hash(ownerpath),
        'claim_identity':stable_hash(owner),'eos_token_ids':[0],
        'validation':{'rows':len(rows),'truncated_rows':0,'dataset_counts':{'hallusionbench':len(rows)}}}
    path=run/'chunk_00000.complete.json';path.write_text(json.dumps(receipt))
    return modelbase,path

def fake_validator(calls):
    def validate(path,plan,keys,claim_identity,eos):
        calls.append(path)
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert [row['key'] for row in rows]==keys
        return {'rows':len(rows),'truncated_rows':0,'dataset_counts':{'hallusionbench':len(rows)}}
    return validate

def test_explicit_migration_namespaces_include_collected_hosts_only(tmp_path):
    for namespace in audit.OUTPUT_NAMESPACES:
        (tmp_path/namespace).mkdir()
        (tmp_path/'collected'/'6403'/namespace).mkdir(parents=True)
    (tmp_path/'runs_unregistered_exploration').mkdir()
    (tmp_path/'collected'/'6403'/'reused_unregistered').mkdir()
    roots=audit.run_roots(tmp_path)
    assert len(roots)==8
    assert {p.name for p in roots}==set(audit.OUTPUT_NAMESPACES)
    assert len(audit.run_roots(tmp_path,namespaces=('runs_recovered_a100',)))==2

def test_actual_migrated_chunks_and_exact_transport_copies_count_once(tmp_path,monkeypatch):
    plans,eligible=fixture_plan();calls=[]
    chunk(tmp_path,'runs_recovered_a100',keys=('key_a',))
    original,_=chunk(tmp_path,'runs_recovered_3090',keys=('key_b',))
    destination=tmp_path/'collected'/'4028'/'runs_recovered_3090'/'fixture'
    shutil.copytree(original,destination)
    monkeypatch.setattr(audit,'validate_rows',fake_validator(calls))
    rawlist,seen,counts,replicas=audit.verified_chunks(tmp_path,plans,eligible)
    assert set(seen)=={'key_a','key_b'} and len(rawlist)==2 and replicas==1
    assert sum(c['generated'] for c in counts.values())==2
    assert len(calls)==3  # A transported replica still passes the original validator.

@pytest.mark.parametrize('duplicate_keys,text',[(('key_a',),'different'),(('key_a','key_b'),'fixture')])
def test_changed_or_partial_key_overlap_never_counts_as_replica(tmp_path,monkeypatch,duplicate_keys,text):
    plans,eligible=fixture_plan()
    chunk(tmp_path,'runs',keys=('key_a',),claim='source_claim')
    chunk(tmp_path,'runs_recovered_a100',keys=duplicate_keys,claim='target_claim',text=text)
    monkeypatch.setattr(audit,'validate_rows',fake_validator([]))
    with pytest.raises(ValueError,match='Non-identical duplicate'):
        audit.verified_chunks(tmp_path,plans,eligible)

def test_sealed_output_must_belong_to_real_recorded_claim(tmp_path,monkeypatch):
    plans,eligible=fixture_plan();_,receiptpath=chunk(tmp_path,'runs_recovered_3090')
    ownerpath=receiptpath.parent/'owner.json';owner=json.loads(ownerpath.read_text())
    owner['keys']=['key_b'];ownerpath.write_text(json.dumps(owner))
    receipt=json.loads(receiptpath.read_text())
    receipt.update(owner_sha256=file_hash(ownerpath),claim_identity=stable_hash(owner))
    receiptpath.write_text(json.dumps(receipt))
    monkeypatch.setattr(audit,'validate_rows',fake_validator([]))
    with pytest.raises(ValueError,match='recorded model/claim ownership'):
        audit.verified_chunks(tmp_path,plans,eligible)

def test_receipt_paths_cannot_point_outside_the_claim(tmp_path,monkeypatch):
    plans,eligible=fixture_plan();_,receiptpath=chunk(tmp_path,'runs_recovered_a100')
    receipt=json.loads(receiptpath.read_text());receipt['raw_path']='../../some_other_output.jsonl'
    receiptpath.write_text(json.dumps(receipt))
    monkeypatch.setattr(audit,'validate_rows',fake_validator([]))
    with pytest.raises(ValueError,match='leave their actual claim directory'):
        audit.verified_chunks(tmp_path,plans,eligible)
