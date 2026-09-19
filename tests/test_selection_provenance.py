"""Selection provenance consumes real helper chains over explicitly synthetic CPU ledgers."""
import importlib.util
import json
from pathlib import Path
import shutil
import pytest
from kdm.io import file_hash,read_jsonl
from kdm.pipeline import select_models
from kdm.provenance import validate_census_inputs
from kdm.selection_provenance import validate_selection
from test_census_provenance import census,rewrite_definition,rewrite_row


@pytest.fixture
def selected(census):
    c=census;paths=[c.write('a'),c.write('b')]
    rows=[row for path in paths for row in read_jsonl(path)]
    annotations={row['key']:{'text':row['text'],'label':'abstain','evidence':'UNKNOWN','answer_text':''} for row in rows}
    selection=select_models(paths,annotations,[sample['id'] for sample in c.samples])
    path=c.root/'selection.json';path.write_text(json.dumps(selection))
    receipt=validate_census_inputs(c.root,paths,c.manifest,c.freeze,require_complete_panel=True)
    receipt.update(operation='select',output_sha256=file_hash(path),annotations_sha256='a'*64)
    sidecar=path.with_suffix('.sources.json');sidecar.write_text(json.dumps(receipt))
    return c,path,sidecar,paths,selection


def test_selection_reuses_complete_census_chain_without_mutating_selection(selected):
    c,path,sidecar,paths,selection=selected;before=path.read_bytes()
    result=validate_selection(c.root,path,c.manifest,c.freeze)
    assert result['selection']==selection and path.read_bytes()==before
    assert result['provenance']['census']['complete_panel'] is True
    assert result['provenance']['selection_sha256']==file_hash(path)
    assert result['provenance']['source_receipt_sha256']==file_hash(sidecar)
    assert result['provenance']['recorded_annotations_sha256']=='a'*64


@pytest.mark.parametrize('mutation',[
    'missing_receipt','changed_selection','wrong_operation','partial_flag','mock_flag','missing_annotation_identity',
    'missing_source_sidecar','changed_raw_identity','changed_raw_tokens','old_source','wrong_freeze',
    'source_sha','sidecar_sha','incomplete_sources','missing_model_decision','wrong_denominator',
    'wrong_selection_rule','wrong_row_sources','duplicate_condition'])
def test_selection_rejects_detached_changed_or_incomplete_evidence(selected,mutation):
    c,path,sidecar,paths,selection=selected;receipt=json.loads(sidecar.read_text())
    if mutation=='missing_receipt':sidecar.unlink()
    elif mutation=='changed_selection':path.write_text(path.read_text()+' ')
    elif mutation=='wrong_operation':receipt['operation']='annotation-queue'
    elif mutation=='partial_flag':receipt['complete_panel']=False
    elif mutation=='mock_flag':receipt['formal_evidence']=False
    elif mutation=='missing_annotation_identity':receipt.pop('annotations_sha256')
    elif mutation=='missing_source_sidecar':paths[0].with_suffix('.identity.json').unlink()
    elif mutation=='changed_raw_identity':rewrite_row(paths[0],lambda row:row.update(identity='different'))
    elif mutation=='changed_raw_tokens':rewrite_row(paths[0],lambda row:row.update(tokens=[10,11]))
    elif mutation=='old_source':rewrite_definition(paths[0],lambda definition:definition.update(source_blobs=['old-source']))
    elif mutation=='wrong_freeze':receipt['freeze_receipt_sha256']='0'*64
    elif mutation=='source_sha':receipt['sources'][0]['sha256']='0'*64
    elif mutation=='sidecar_sha':receipt['sources'][0]['sidecar_sha256']='0'*64
    elif mutation=='incomplete_sources':receipt['sources'].pop()
    else:
        if mutation=='missing_model_decision':selection.pop()
        elif mutation=='wrong_denominator':selection[0]['n']-=1
        elif mutation=='wrong_selection_rule':selection[0]['selected']=False
        elif mutation=='wrong_row_sources':selection[0]['selection_sources']=[str(paths[1])]
        elif mutation=='duplicate_condition':selection.append(selection[0])
        path.write_text(json.dumps(selection));receipt['output_sha256']=file_hash(path)
    if mutation!='missing_receipt':sidecar.write_text(json.dumps(receipt))
    with pytest.raises((ValueError,FileNotFoundError)):
        validate_selection(c.root,path,c.manifest,c.freeze)


def test_copy_to_another_registered_project_root_retains_original_selection_sources(selected):
    c,path,sidecar,paths,selection=selected
    copied=c.root/'remote-copy';files=[p for p in c.root.rglob('*') if p.is_file()]
    for source in files:
        target=copied/source.relative_to(c.root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    registry={'schema':1,'hosts':{
        '4028':{'root':str(c.root),'hostname':'original','allowed_gpus':[0],'gpu_uuids':{'0':'unit'}},
        '6403':{'root':str(copied),'hostname':'copy','allowed_gpus':[1],'gpu_uuids':{'1':'unit'}}},
        'model_hosts':{'a':'4028','b':'6403'},'image_prefixes':{}}
    (copied/'configs/runtime/hosts.json').write_text(json.dumps(registry))
    # This test isolates path portability; host receipt validation is separately covered
    # by census provenance tests. Its synthetic freeze predates the fixture registry.
    result=validate_selection(copied,'selection.json','data/current/all.jsonl',c.freeze)
    assert result['selection']==selection
    assert (copied/'selection.json').read_bytes()==path.read_bytes()
    assert result['provenance']['census']['sources'][0]['path']==str(paths[0].relative_to(c.root))


def test_prepare_refuses_changed_selection_before_generating_manifests(selected,monkeypatch):
    c,path,sidecar,paths,selection=selected
    source=Path(__file__).parents[1]/'scripts/prepare_selected_manifests.py'
    spec=importlib.util.spec_from_file_location('prepare_selection_gate',source)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setattr(mod,'validate_freeze',lambda root:c.freeze)
    monkeypatch.setattr(mod,'write_manifest',lambda *a,**kw:pytest.fail('Invalid selection wrote a manifest'))
    monkeypatch.setattr('sys.argv',['prepare','--root',str(c.root),'--manifest',str(c.manifest),
        '--selection',str(path),'--method-plan','never-opened.json','--out-dir','must-not-write'])
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='original select source receipt'):mod.main()
    assert not (c.root/'must-not-write').exists()
