import importlib.util,json
from pathlib import Path
import pytest


@pytest.mark.parametrize('command,extra,expected',[
    ('run',['--mode','experiment','--manifest','unused','--methods','vcd,m3id,dola,deco'],('vcd','m3id','dola','deco')),
    ('mechanism',['--records','records','--methods','m3id,deco'],('m3id','deco')),
    ('replay',['--records','records'],('dola','sid')),
])
def test_formal_entry_method_proof_precedes_backend(tmp_path,monkeypatch,command,extra,expected):
    import kdm.cli as cli
    import kdm.protocol as protocol
    import kdm.pipeline as pipeline
    spec=tmp_path/'spec.json';spec.write_text('{}')
    records=tmp_path/'records';records.write_text(''.join(json.dumps({'method':m})+'\n' for m in ['direct','dola','sid','dola']))
    extra=[str(records) if x=='records' else x for x in extra]
    if command=='run':
        from kdm.io import file_hash
        (tmp_path/'data/current').mkdir(parents=True);(tmp_path/'configs/kdm').mkdir(parents=True);(tmp_path/'outputs/records').mkdir(parents=True)
        manifest=tmp_path/'data/current/all.jsonl';manifest.write_text(json.dumps({'id':'s','dataset':'fixture','split':'eval'})+'\n')
        (tmp_path/'configs/kdm/models.json').write_text(json.dumps([{'key':'m'}]))
        plan=tmp_path/'plan.json';plan.write_text(json.dumps({'m':{'fixture':list(expected)}}))
        (tmp_path/'outputs/records/preregistration_freeze.json').write_text(json.dumps({'status':'frozen','files':{'plan.json':file_hash(plan),'data/current/all.jsonl':file_hash(manifest)}}))
        extra=[str(manifest) if x=='unused' else x for x in extra]+['--method-plan',str(plan)]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(protocol,'validate_runtime',lambda *args:None)
    if command!='run':
        (tmp_path/'outputs/records').mkdir(parents=True)
        (tmp_path/'outputs/records/preregistration_freeze.json').write_text(json.dumps({'files':{},'source_blobs':[]}))
    import kdm.task_provenance as provenance
    monkeypatch.setattr(provenance,'validate_task_inputs',lambda *args,**kwargs:{'sources':[]})
    monkeypatch.setattr(provenance,'validate_measurement_methods',lambda *args:None)
    def frozen(root):
        from kdm.io import file_hash
        value=json.loads((root/'outputs/records/preregistration_freeze.json').read_text())
        if 'plan.json' in value['files']:value['files']['configs/kdm/method_plan.json']=value['files']['plan.json']
        value['files']['configs/runtime/m.json']=file_hash(spec);value['source_blobs']=[]
        return value
    monkeypatch.setattr(protocol,'validate_freeze',frozen)
    def gate(root,spec,methods):
        assert methods==expected
        raise ValueError('missing method proof')
    monkeypatch.setattr(protocol,'validate_method_runtime',gate,raising=False)
    monkeypatch.setattr(pipeline,'make_backend',lambda *args:pytest.fail('backend was loaded before proof'))
    with pytest.raises(ValueError,match='missing method proof'):
        cli.main(['--root',str(tmp_path),command,'--model-spec',str(spec),'--model','m','--gpu','0','--out','formal.jsonl',*extra])


@pytest.mark.parametrize('mode,mock',[('probe',False),('experiment',True)])
def test_non_method_runs_do_not_require_method_proof(tmp_path,monkeypatch,mode,mock):
    import kdm.cli as cli
    import kdm.protocol as protocol
    import kdm.pipeline as pipeline
    spec=tmp_path/'spec.json';spec.write_text(json.dumps({'purpose':'CPU_TEST_ONLY'} if mock else {}))
    from kdm.io import file_hash
    (tmp_path/'data/current').mkdir(parents=True);(tmp_path/'outputs/records').mkdir(parents=True)
    manifest=tmp_path/'data/current/all.jsonl';manifest.write_text(json.dumps({'id':'s','dataset':'fixture','split':'eval'})+'\n')
    (tmp_path/'outputs/records/preregistration_freeze.json').write_text('{}')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(protocol,'validate_runtime',lambda *args:None)
    monkeypatch.setattr(protocol,'validate_freeze',lambda *args:{'source_blobs':['frozen'],'files':{'configs/runtime/m.json':file_hash(spec)}})
    monkeypatch.setattr(protocol,'code_identity',lambda *args:{})
    monkeypatch.setattr(protocol,'validate_method_runtime',lambda *args:pytest.fail('unexpected method gate'),raising=False)
    def backend(*args):raise RuntimeError('backend reached without method gate')
    monkeypatch.setattr(pipeline,'make_backend',backend)
    with pytest.raises(RuntimeError,match='backend reached'):
        cli.main(['--root',str(tmp_path),'run','--mode',mode,'--manifest',str(manifest),'--model-spec',str(spec),'--model','m','--gpu','0','--out','outputs/verification/mock.jsonl' if mock else 'formal.jsonl'])


def test_complete_response_requires_method_proof_and_final_human_labels(tmp_path,monkeypatch):
    import kdm.protocol as protocol
    import kdm.human_review as human
    spec=importlib.util.spec_from_file_location('complete_entry',Path(__file__).parents[1]/'scripts/complete_response_audit.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    config=tmp_path/'spec.json';config.write_text('{}');events=[]
    from kdm.io import file_hash
    (tmp_path/'outputs/records').mkdir(parents=True)
    (tmp_path/'outputs/records/preregistration_freeze.json').write_text(json.dumps({'files':{'configs/runtime/m.json':file_hash(config)}}))
    import kdm.task_provenance as provenance
    monkeypatch.setattr(provenance,'validate_task_inputs',lambda *args,**kwargs:{'sources':[]})
    monkeypatch.setattr(provenance,'validate_measurement_methods',lambda *args:None)
    monkeypatch.setattr(mod,'execution_identity',lambda *args:{})
    monkeypatch.setattr(protocol,'validate_method_runtime',lambda root,spec,methods:events.append(methods),raising=False)
    def reviewed(path,records):
        assert path=='annotations' and records==[str(tmp_path/'raw-records')]
        events.append('human_review');raise ValueError('human review incomplete')
    monkeypatch.setattr(human,'validate_human_review',reviewed)
    monkeypatch.setattr(mod,'validate_annotations',lambda *args:pytest.fail('formal used provisional annotations'))
    monkeypatch.setattr(mod,'make_backend',lambda *args:pytest.fail('backend loaded before human review'))
    monkeypatch.setattr('sys.argv',['complete','--root',str(tmp_path),'--records','raw-records','--manifest','unused',
        '--annotations','annotations','--model-spec',str(config),'--model','m','--gpu','0','--out','formal.jsonl','--methods','dola,instruction_vcd'])
    with pytest.raises(ValueError,match='human review incomplete'):mod.main()
    assert events==[('dola','instruction_vcd'),'human_review']


def test_formal_census_without_freeze_cannot_reach_backend(tmp_path,monkeypatch):
    import kdm.cli as cli
    import kdm.protocol as protocol
    import kdm.pipeline as pipeline
    spec=tmp_path/'spec.json';spec.write_text('{}')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(protocol,'validate_runtime',lambda *args:None)
    monkeypatch.setattr(pipeline,'make_backend',lambda *args:pytest.fail('Unfrozen census loaded model'))
    with pytest.raises(FileNotFoundError,match='preregistration_freeze'):
        cli.main(['--root',str(tmp_path),'run','--mode','census','--manifest','unused','--model-spec',str(spec),'--model','m','--gpu','0','--out','formal.jsonl'])


@pytest.mark.parametrize('changed_manifest',[False,True])
def test_formal_census_matches_frozen_full_manifest_before_model_load(tmp_path,monkeypatch,changed_manifest):
    import kdm.cli as cli
    import kdm.protocol as protocol
    import kdm.pipeline as pipeline
    from kdm.io import file_hash
    spec=tmp_path/'spec.json';spec.write_text('{}');manifest=tmp_path/'manifest.jsonl';manifest.write_text('frozen content')
    expected=file_hash(manifest)
    if changed_manifest:manifest.write_text('subset or different version')
    (tmp_path/'outputs/records').mkdir(parents=True)
    (tmp_path/'outputs/records/preregistration_freeze.json').write_text('synthetic receipt content')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setattr(protocol,'validate_runtime',lambda *args:None)
    monkeypatch.setattr(protocol,'validate_freeze',lambda root:{'source_blobs':['frozen-source'],'files':{'data/current/all.jsonl':expected,'configs/runtime/m.json':file_hash(spec)}})
    def backend(*args):raise RuntimeError('backend reached after frozen full manifest check')
    monkeypatch.setattr(pipeline,'make_backend',backend)
    with pytest.raises(ValueError if changed_manifest else RuntimeError) as exc:
        cli.main(['--root',str(tmp_path),'run','--mode','census','--manifest',str(manifest),'--model-spec',str(spec),'--model','m','--gpu','0','--out','formal.jsonl'])
    assert ('Census manifest differs' if changed_manifest else 'backend reached') in str(exc.value)
