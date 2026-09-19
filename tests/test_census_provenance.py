from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from kdm.decoding import DecodeConfig
from kdm.io import file_hash,stable_hash,stable_seed,read_jsonl
from kdm.pipeline import census_tasks,task_id,select_models
from kdm.prompts import task_prompt
from kdm.provenance import validate_census_inputs,census_input_paths
from kdm.protocol import validate_census_collection


@pytest.fixture
def census(tmp_path):
    samples=[{'id':f's{i}','dataset':'food101','split':'eval','question':'Name the food.'} for i in range(8)]
    root=tmp_path;files={};specs={}
    def save(relative,value,jsonl=False):
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in value) if jsonl else json.dumps(value))
        files[relative]=file_hash(path);return path
    manifest=save('data/current/all.jsonl',samples,True)
    save('configs/kdm/models.json',[{'key':'a'},{'key':'b'}])
    for model in ['a','b']:
        specs[model]={'key':model,'factory':'kdm.models.hf:HFBackend','kwargs':{'model_path':'unused'}}
        save(f'configs/runtime/{model}.json',specs[model])
    frozen={'status':'frozen','source_blobs':['100644 fixed-source 0\tsrc/kdm/pipeline.py'],'files':dict(files)}
    freeze_path=root/'outputs/records/preregistration_freeze.json';freeze_path.parent.mkdir(parents=True);freeze_path.write_text(json.dumps(frozen))
    cfg=asdict(DecodeConfig())
    def write(model='a',name=None,shard=0,n_shards=1,choose=None):
        path=root/(name or f'outputs/raw/{model}.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
        definition={'schema':'kdm_current_v2','model':model,'backend':specs[model],'backend_spec_sha256':files[f'configs/runtime/{model}.json'],
            'source_blobs':frozen['source_blobs'],'freeze_receipt_sha256':file_hash(freeze_path),'manifest_sha256':file_hash(manifest),
            'base_config':cfg,'shard':shard,'n_shards':n_shards}
        identity=stable_hash(definition);rows=[]
        for task in census_tasks(samples):
            if int(stable_hash(task['sample']['id'])[:8],16)%n_shards!=shard:continue
            row={**task,'model':model,'key':task_id(model,task),'identity':identity,'status':'ok','text':'UNKNOWN','tokens':[1],
                'terminated':True,'config':cfg,'seed':stable_seed(task['sample']['id'],model,0),
                'prompt':task_prompt(task['sample']['question'],task['marker'],task['guided']),'reference_prompt':None,'neutral_prompt':None}
            if choose is None or choose(row):rows.append(row)
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        path.with_suffix('.identity.json').write_text(json.dumps({'definition':definition,'identity':identity}))
        return path
    return SimpleNamespace(root=root,manifest=manifest,freeze=frozen,specs=specs,files=files,samples=samples,write=write,freeze_path=freeze_path)


def check(c,paths,complete=False,**kwargs):
    return validate_census_inputs(c.root,paths,c.manifest,c.freeze,complete,**kwargs)


def rewrite_definition(path,edit):
    sidecar=path.with_suffix('.identity.json');meta=json.loads(sidecar.read_text());edit(meta['definition']);meta['identity']=stable_hash(meta['definition']);sidecar.write_text(json.dumps(meta))
    rows=list(read_jsonl(path))
    for row in rows:row['identity']=meta['identity']
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def rewrite_row(path,edit):
    rows=list(read_jsonl(path));edit(rows[0]);path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_legal_shards_and_selection_preserve_denominator(census):
    c=census;paths=[c.write(name=f'outputs/raw/a_part{i}.jsonl',shard=i,n_shards=2) for i in [0,1]]
    result=check(c,paths);assert result['models']['a']['record_count']==16 and len(result['models']['a']['paths'])==2
    assert len(set(result['models']['a']['identities']))==2
    validate_census_collection(paths,c.samples,['a'])
    rows=[r for path in paths for r in read_jsonl(path)]
    annotations={r['key']:{'text':r['text'],'label':'abstain','evidence':'UNKNOWN','answer_text':''} for r in rows}
    selection=select_models(paths,annotations,[s['id'] for s in c.samples])
    assert len(selection)==1 and selection[0]['n']==8 and selection[0]['n_abstain']==8 and selection[0]['selected'] is True
    assert selection[0]['selection_sources']==[str(path) for path in paths]
    with pytest.raises(ValueError,match='duplicate'):check(c,paths+[paths[0]])
    with pytest.raises(ValueError,match='duplicate'):validate_census_collection(paths+[paths[0]],c.samples,['a'])
    with pytest.raises(ValueError,match='duplicated'):select_models(paths+[paths[0]],annotations,[s['id'] for s in c.samples])


def test_same_identity_fragments_and_merged_file(census):
    c=census;a=c.write(name='outputs/raw/first.jsonl',choose=lambda r:r['guided']);b=c.write(name='outputs/raw/second.jsonl',choose=lambda r:not r['guided'])
    result=check(c,[a,b]);assert len(set(result['models']['a']['identities']))==1
    merged=c.root/'outputs/raw/merged.jsonl';merged.write_text(a.read_text()+b.read_text());merged.with_suffix('.identity.json').write_text(a.with_suffix('.identity.json').read_text())
    assert check(c,[merged])['models']['a']['record_count']==16
    with pytest.raises(ValueError,match='Incomplete'):check(c,[a])


def test_panel_subset_requires_each_model_complete(census):
    c=census;a=c.write();assert check(c,[a])['complete_panel'] is False
    with pytest.raises(ValueError,match='Missing fixed candidate'):check(c,[a],complete=True)
    b=c.write('b');assert check(c,[a,b],complete=True)['complete_panel'] is True


@pytest.mark.parametrize('mutation,error',[
    ('no_sidecar','sidecar'),('row_identity','record identity'),('digest','digest'),('source','source blobs'),
    ('spec','backend'),('spec_sha','backend'),('manifest','manifest'),('freeze','freeze receipt'),
    ('base_config','base configuration'),('row_config','row configuration'),('seed','seed'),('prompt','prompt'),
    ('shard_owner','different shard'),('shard_invalid','shard declaration')])
def test_rejects_stale_mixed_or_wrong_identity(census,mutation,error):
    c=census;path=c.write()
    if mutation=='no_sidecar':path.with_suffix('.identity.json').unlink()
    elif mutation=='row_identity':rewrite_row(path,lambda r:r.update(identity='different-run'))
    elif mutation=='digest':
        meta=json.loads(path.with_suffix('.identity.json').read_text());meta['identity']='bad';path.with_suffix('.identity.json').write_text(json.dumps(meta))
    elif mutation=='source':rewrite_definition(path,lambda d:d.update(source_blobs=['old-source']))
    elif mutation=='spec':rewrite_definition(path,lambda d:d['backend'].update(kwargs={'model_path':'other'}))
    elif mutation=='spec_sha':rewrite_definition(path,lambda d:d.update(backend_spec_sha256='old-spec'))
    elif mutation=='manifest':rewrite_definition(path,lambda d:d.update(manifest_sha256='old-manifest'))
    elif mutation=='freeze':rewrite_definition(path,lambda d:d.update(freeze_receipt_sha256='old-freeze'))
    elif mutation=='base_config':rewrite_definition(path,lambda d:d['base_config'].update(temperature=9))
    elif mutation=='row_config':rewrite_row(path,lambda r:r['config'].update(max_tokens=999))
    elif mutation=='seed':rewrite_row(path,lambda r:r.update(seed=-1))
    elif mutation=='prompt':rewrite_row(path,lambda r:r.update(prompt='different prompt'))
    elif mutation=='shard_owner':rewrite_definition(path,lambda d:d.update(n_shards=2,shard=0))
    elif mutation=='shard_invalid':rewrite_definition(path,lambda d:d.update(n_shards=2,shard=True))
    with pytest.raises(ValueError,match=error):check(c,[path])


def test_different_shard_definitions_and_cross_identity_cat_rejected(census):
    c=census;a=c.write(name='outputs/raw/a0.jsonl',shard=0,n_shards=2);b=c.write(name='outputs/raw/a1.jsonl',shard=1,n_shards=2)
    merged=c.root/'outputs/raw/mixed.jsonl';merged.write_text(a.read_text()+b.read_text());merged.with_suffix('.identity.json').write_text(a.with_suffix('.identity.json').read_text())
    with pytest.raises(ValueError,match='record identity'):check(c,[merged])
    rewrite_definition(b,lambda d:d.update(unregistered_run_variant='other'))
    with pytest.raises(ValueError,match='Mixed census run'):check(c,[a,b])


def test_mock_requires_explicit_scope_and_actual_mock_factory(census):
    c=census;c.specs['a'].update(purpose='CPU_TEST_ONLY',factory='kdm.models.mock:MockBackend')
    spec=c.root/'configs/runtime/a.json';spec.write_text(json.dumps(c.specs['a']));c.files['configs/runtime/a.json']=file_hash(spec);c.freeze['files']['configs/runtime/a.json']=file_hash(spec);c.freeze_path.write_text(json.dumps(c.freeze))
    path=c.write(name='outputs/verification/mock.jsonl');rewrite_definition(path,lambda d:d.update(source_blobs={'software_fixture':True,'formal_evidence':False}))
    with pytest.raises(ValueError,match='Mock census'):check(c,[path])
    assert check(c,[path],allow_mock=True)['formal_evidence'] is False
    formal=c.write(name='outputs/raw/mock.jsonl')
    with pytest.raises(ValueError,match='Mock census'):check(c,[formal],allow_mock=True)


@pytest.mark.parametrize('command',['annotation-queue','select','analyze'])
def test_cli_rejects_bad_census_before_annotation_or_analysis(census,monkeypatch,command):
    from kdm.cli import main
    import kdm.protocol
    c=census;path=c.write();rewrite_row(path,lambda r:r.update(identity='stale'))
    monkeypatch.setattr(kdm.protocol,'validate_freeze',lambda root:c.freeze)
    args=['--root',str(c.root),command,'--out','outputs/verification/result.json']
    if command=='select':args+=['--census',str(path),'--manifest',str(c.manifest),'--annotations','missing_annotations.jsonl']
    else:args+=['--records',str(path)]
    if command=='analyze':args+=['--annotations','missing_annotations.jsonl','--aliases','missing_aliases.json']
    with pytest.raises(ValueError,match='record identity'):main(args)
    assert not (c.root/'outputs/verification/result.json').exists()


def test_annotation_cli_accepts_complete_model_subset_and_writes_receipt(census,monkeypatch):
    from kdm.cli import main
    import kdm.protocol
    c=census;path=c.write();monkeypatch.setattr(kdm.protocol,'validate_freeze',lambda root:c.freeze)
    main(['--root',str(c.root),'annotation-queue','--records',str(path),'--out','outputs/verification/queue.jsonl'])
    out=c.root/'outputs/verification/queue.jsonl';receipt=json.loads(out.with_suffix('.sources.json').read_text())
    assert receipt['output_sha256']==file_hash(out) and receipt['complete_panel'] is False and receipt['sources'][0]['sha256']==file_hash(path)
    assert all('model' not in row and 'identity' not in row for row in read_jsonl(out))


def test_census_detector_rejects_mixed_stages_in_one_ledger(census):
    path=census.write();rewrite_row(path,lambda r:r.update(kind='main'))
    with pytest.raises(ValueError,match='other stages'):census_input_paths([path])


def test_select_cli_aggregates_legal_sources_and_records_label_identity(census,monkeypatch):
    from kdm.cli import main
    import kdm.protocol,kdm.human_review
    c=census;paths=[c.write(name=f'outputs/raw/a{i}.jsonl',shard=i,n_shards=2) for i in [0,1]]+[c.write('b')]
    annotations={r['key']:{'text':r['text'],'label':'abstain','evidence':'UNKNOWN','answer_text':''} for path in paths for r in read_jsonl(path)}
    annotation_path=c.root/'annotations.jsonl';annotation_path.write_text('fixture with independently tested human-review gate')
    monkeypatch.setattr(kdm.protocol,'validate_freeze',lambda root:c.freeze)
    monkeypatch.setattr(kdm.human_review,'validate_human_review',lambda *a:annotations)
    main(['--root',str(c.root),'select','--census',*[str(path.relative_to(c.root)) for path in paths],
          '--manifest','data/current/all.jsonl','--annotations',str(annotation_path),'--out','outputs/verification/selection.json'])
    out=c.root/'outputs/verification/selection.json';result=json.loads(out.read_text());receipt=json.loads(out.with_suffix('.sources.json').read_text())
    assert len(result)==2 and all(r['n']==8 and r['n_abstain']==8 and r['selected'] for r in result)
    assert receipt['complete_panel'] is True and receipt['annotations_sha256']==file_hash(annotation_path)
    assert len(receipt['models']['a']['paths'])==2


def test_cli_resolves_validated_raw_paths_under_root(census,monkeypatch):
    from kdm.cli import main
    import kdm.protocol
    c=census;path=c.write();monkeypatch.setattr(kdm.protocol,'validate_freeze',lambda root:c.freeze)
    main(['--root',str(c.root),'annotation-queue','--records',str(path.relative_to(c.root)),
          '--out','outputs/verification/queue.jsonl'])
    assert len(list(read_jsonl(c.root/'outputs/verification/queue.jsonl')))==16


@pytest.mark.parametrize('mutation',['valid_remote','missing_receipt','wrong_host','wrong_card','changed_resolver'])
def test_consumption_checks_recorded_execution_host_not_consumers_host(census,mutation,monkeypatch):
    import kdm.execution as execution
    c=census
    registry={'schema':1,'hosts':{'6403':{'hostname':'remote','root':'/remote/project','allowed_gpus':[1],'gpu_uuids':{'1':'remote-uuid'}}},
        'model_hosts':{'a':'6403','b':'6403'},'image_prefixes':{'/logical':'data/images'},'image_catalog':'catalog.json'}
    def save(relative,value):
        p=c.root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
        c.freeze['files'][relative]=file_hash(p)
    save('configs/runtime/hosts.json',registry);save('catalog.json',{'synthetic':True})
    save('src/kdm/execution.py',{'synthetic':'resolver'})
    c.freeze_path.write_text(json.dumps(c.freeze));path=c.write()
    receipt={'host':'6403','hostname':'remote','project_root':'/remote/project','physical_gpus':['1'],'gpu_uuids':{'1':'remote-uuid'},
        'registry_sha256':file_hash(c.root/'configs/runtime/hosts.json'),'resolver_sha256':file_hash(c.root/'src/kdm/execution.py'),
        'image_catalog_sha256':file_hash(c.root/'catalog.json')}
    if mutation=='wrong_host':receipt['host']='4028'
    elif mutation=='wrong_card':receipt['physical_gpus']=['0']
    if mutation!='missing_receipt':rewrite_definition(path,lambda d:d.update(execution=receipt))
    if mutation=='changed_resolver':(c.root/'src/kdm/execution.py').write_text('changed')
    monkeypatch.setattr(execution.socket,'gethostname',lambda:'central-consumer')
    monkeypatch.setattr(execution.subprocess,'check_output',lambda *a,**kw:pytest.fail('Consumer queried GPUs'))
    if mutation=='valid_remote':assert check(c,[path])['models']['a']['formal_evidence']
    else:
        with pytest.raises(ValueError):check(c,[path])
