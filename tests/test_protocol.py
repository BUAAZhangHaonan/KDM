import json
import pytest
from kdm.protocol import validate_census_collection, selected_samples
from kdm.pipeline import census_tasks, task_id


def test_selection_refuses_missing_unguided_and_missing_candidate(tmp_path):
    samples=[{"id":"s", "dataset":"food101", "split":"eval"}]
    path=tmp_path/"census.jsonl"
    rows=[dict(t, model="m", key=task_id("m",t), status="ok", tokens=[1], terminated=True)
          for t in census_tasks(samples)]
    def write(items):
        path.write_text("".join(json.dumps(r)+"\n" for r in items))
    write(rows)
    validate_census_collection([path],samples,["m"])
    with pytest.raises(ValueError, match="Missing fixed candidate"):
        validate_census_collection([path],samples,["m","other"])
    write(rows[:1])
    with pytest.raises(ValueError, match="Incomplete guided or unguided"):
        validate_census_collection([path],samples,["m"])
    write(rows+[rows[0]])
    with pytest.raises(ValueError, match="duplicate census task"):
        validate_census_collection([path],samples,["m"])


def test_selected_manifest_preserves_entire_dataset_and_split():
    samples=[{"id":str(i),"dataset":"food101" if i<2 else "vizwiz","split":"dev" if i==0 else "eval"} for i in range(3)]
    selection=[{"model":"m","dataset":"food101","n":2,"n_abstain":1,"selected":True},
               {"model":"m","dataset":"vizwiz","n":1,"n_abstain":0,"selected":False}]
    assert selected_samples(samples,selection,"m")==samples[:2]
    selection[0]["n"]=1
    with pytest.raises(ValueError, match="denominator"):
        selected_samples(samples,selection,"m")


def test_runtime_rejects_unresolved_and_unlocked_model(tmp_path):
    from kdm.protocol import validate_runtime
    with pytest.raises(ValueError, match="unresolved"):
        validate_runtime(tmp_path,{"key":"m","availability":"missing"},"m",["0"])
    spec={"key":"m","availability":"resolved","gpu_count":2}
    with pytest.raises(ValueError, match="GPU count"):
        validate_runtime(tmp_path,spec,"m",["0"])
    spec["gpu_count"]=1
    with pytest.raises(ValueError, match="[Ww]orker"):
        validate_runtime(tmp_path,spec,"m",["0"])


def test_runtime_refuses_incomplete_or_changed_adapter_identity(tmp_path,monkeypatch):
    import os,sys
    from kdm.io import file_hash
    from kdm.protocol import validate_runtime
    model='minicpm26'
    (tmp_path/'data/current').mkdir(parents=True)
    manifest=tmp_path/'data/current/interface16.jsonl';manifest.write_text('frozen interface manifest')
    sources=tmp_path/'src/kdm/models';sources.mkdir(parents=True)
    for name in ['hf.py','backbone.py','sid.py','remote.py']:(sources/name).write_text(name)
    weights=tmp_path/'weights';weights.mkdir();(weights/'w.bin').write_bytes(b'1234')
    spec={'key':model,'availability':'resolved','gpu_count':1,'environment_python':sys.executable,
          'versions':{},'kwargs':{'model_path':str(weights)},'weights':[{'filename':'w.bin','size_bytes':4}],
          'interface_verification':{'status':'passed','record':'proof.json'}}
    proof={'passed':True,'completed':16,'expected':16,'spec':spec,
           'manifest_sha256':file_hash(manifest),'runtime_adapter_sha256':{}}
    original=os.readlink
    monkeypatch.setattr(os,'readlink',lambda path,*a,**kw:str(tmp_path/'outputs/locks/gpu_0.lock') if str(path)=='/proc/self/fd/20' else original(path,*a,**kw))
    def save(): (tmp_path/'proof.json').write_text(json.dumps(proof))
    save()
    with pytest.raises(ValueError,match='omits required'):validate_runtime(tmp_path,spec,model,['0'])
    proof['runtime_adapter_sha256']={p.name:file_hash(p) for p in sources.iterdir()};save()
    validate_runtime(tmp_path,spec,model,['0'])
    (sources/'remote.py').write_text('different runtime')
    with pytest.raises(ValueError,match='Adapter changed'):validate_runtime(tmp_path,spec,model,['0'])


def test_method_gate_requires_actual_projection_and_sid_reference(tmp_path):
    from kdm.protocol import validate_method_runtime
    from kdm.io import file_hash
    spec={'key':'llava15_7b','interface_verification':{'record':'native.json'},'mechanism_validation':{'sid_reference':'not_verified'}}
    path=tmp_path/'native.json';path.write_text('{}')
    validate_method_runtime(tmp_path,spec,['vcd','m3id'])
    with pytest.raises(ValueError,match='layer projection'):validate_method_runtime(tmp_path,spec,['dola'])
    path.write_text(json.dumps({'layer_projection_check':{'status':'passed','head_final_error':0.,'raw_projection_error':0.,'norm_projection_error':0.,'raw_layers':32,'normalized_layers':[20,21]}}))
    validate_method_runtime(tmp_path,spec,['dola','deco'])
    with pytest.raises(ValueError,match='SID official'):validate_method_runtime(tmp_path,spec,['sid'])
    sources=tmp_path/'src/kdm/models';sources.mkdir(parents=True)
    for name in ['hf.py','backbone.py','sid.py']:(sources/name).write_text(name)
    spec['mechanism_validation']['sid_reference']={'status':'passed','record':'sid.json'}
    proof={'passed':True,'reference_commit':'127dd412fa6b61ab1c9babf6979ec4da98002438','spec':spec,
           'checks':dict.fromkeys(['official_selection','official_reference_logits','causal_mask_preserved','interleaved_sessions','nonmonotonic_prefix'],True),
           'runtime_adapter_sha256':{p.name:file_hash(p) for p in sources.iterdir()}}
    def save(): (tmp_path/'sid.json').write_text(json.dumps(proof))
    save();validate_method_runtime(tmp_path,spec,['sid'])
    proof['checks']['interleaved_sessions']=False;save()
    with pytest.raises(ValueError,match='session-isolation'):validate_method_runtime(tmp_path,spec,['sid'])
    proof['checks']['interleaved_sessions']=True;save();(sources/'sid.py').write_text('changed')
    with pytest.raises(ValueError,match='implementation changed'):validate_method_runtime(tmp_path,spec,['sid'])
