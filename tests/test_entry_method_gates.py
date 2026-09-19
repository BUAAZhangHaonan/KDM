import importlib.util,json
from pathlib import Path
import pytest


@pytest.mark.parametrize('command,extra,expected',[
    ('run',['--mode','experiment','--manifest','unused','--methods','vcd,dola'],('vcd','dola')),
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
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(protocol,'validate_runtime',lambda *args:None)
    def gate(root,spec,methods):
        assert methods==expected
        raise ValueError('missing method proof')
    monkeypatch.setattr(protocol,'validate_method_runtime',gate,raising=False)
    monkeypatch.setattr(pipeline,'make_backend',lambda *args:pytest.fail('backend was loaded before proof'))
    with pytest.raises(ValueError,match='missing method proof'):
        cli.main(['--root',str(tmp_path),command,'--model-spec',str(spec),'--model','m','--gpu','0','--out','formal.jsonl',*extra])


@pytest.mark.parametrize('mode,mock',[('census',False),('probe',False),('experiment',True)])
def test_non_method_runs_do_not_require_method_proof(tmp_path,monkeypatch,mode,mock):
    import kdm.cli as cli
    import kdm.protocol as protocol
    import kdm.pipeline as pipeline
    spec=tmp_path/'spec.json';spec.write_text(json.dumps({'purpose':'CPU_TEST_ONLY'} if mock else {}))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(protocol,'validate_runtime',lambda *args:None)
    monkeypatch.setattr(protocol,'code_identity',lambda *args:{})
    monkeypatch.setattr(protocol,'validate_method_runtime',lambda *args:pytest.fail('unexpected method gate'),raising=False)
    def backend(*args):raise RuntimeError('backend reached without method gate')
    monkeypatch.setattr(pipeline,'make_backend',backend)
    with pytest.raises(RuntimeError,match='backend reached'):
        cli.main(['--root',str(tmp_path),'run','--mode',mode,'--manifest','unused','--model-spec',str(spec),'--model','m','--gpu','0','--out','outputs/verification/mock.jsonl' if mock else 'formal.jsonl'])


def test_complete_response_requires_method_proof_and_final_human_labels(tmp_path,monkeypatch):
    import kdm.protocol as protocol
    import kdm.human_review as human
    spec=importlib.util.spec_from_file_location('complete_entry',Path(__file__).parents[1]/'scripts/complete_response_audit.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    config=tmp_path/'spec.json';config.write_text('{}');events=[]
    monkeypatch.setattr(mod,'execution_identity',lambda *args:{})
    monkeypatch.setattr(protocol,'validate_method_runtime',lambda root,spec,methods:events.append(methods),raising=False)
    def reviewed(path,records):
        assert path=='annotations' and records==['raw-records']
        events.append('human_review');raise ValueError('human review incomplete')
    monkeypatch.setattr(human,'validate_human_review',reviewed)
    monkeypatch.setattr(mod,'validate_annotations',lambda *args:pytest.fail('formal used provisional annotations'))
    monkeypatch.setattr(mod,'make_backend',lambda *args:pytest.fail('backend loaded before human review'))
    monkeypatch.setattr('sys.argv',['complete','--root',str(tmp_path),'--records','raw-records','--manifest','unused',
        '--annotations','annotations','--model-spec',str(config),'--model','m','--gpu','0','--out','formal.jsonl','--methods','dola,instruction_vcd'])
    with pytest.raises(ValueError,match='human review incomplete'):mod.main()
    assert events==[('dola','instruction_vcd'),'human_review']
