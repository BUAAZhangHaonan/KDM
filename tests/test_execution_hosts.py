"""Synthetic CPU fixtures only: no GPU process, real checkpoint or remote write."""
import json
import os
from pathlib import Path
import sys
import pytest
from kdm import execution as e
from kdm.io import file_hash


@pytest.fixture
def host(tmp_path,monkeypatch):
    def write(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(value if isinstance(value,str) else json.dumps(value));return p
    registry={'schema':1,'hosts':{
        '4028':{'root':'/central-not-present','hostname':'central','allowed_gpus':[0],'gpu_uuids':{'0':'central-uuid'}},
        '6403':{'root':str(tmp_path),'hostname':'remote','allowed_gpus':[1],'gpu_uuids':{'1':'remote-uuid'}}},
        'model_hosts':{'qwen3vl':'6403','local_model':'4028'},
        'image_prefixes':{'/logical/food':'data/images','/logical/viz':'cache/images'},
        'image_catalog':'outputs/records/image_content_catalog.json'}
    write('configs/runtime/hosts.json',registry)
    write('src/kdm/execution.py','unit-test resolver identity')
    manifest=write('data/current/all.jsonl','{"id":"a","image_path":"/logical/food/a.jpg"}\n')
    target=write('data/images/a.jpg','original image bytes')
    write('outputs/records/image_content_catalog.json',{'schema':1,'manifest_sha256':file_hash(manifest),
        'images':{'/logical/food/a.jpg':{'sha256':file_hash(target),'size_bytes':target.stat().st_size}}})
    monkeypatch.setattr(e.socket,'gethostname',lambda:'remote')
    monkeypatch.setattr(e.subprocess,'check_output',lambda *a,**kw:'1, remote-uuid\n')
    return tmp_path,registry,write


def test_host_binding_cards_uuid_and_model_assignment(host,monkeypatch):
    root,registry,write=host
    assert e.validate_host(root,['1'],'qwen3vl')[0]=='6403'
    for cards in (['0'],['1','1'],[]):
        with pytest.raises(ValueError,match='GPU'):e.validate_host(root,cards,'qwen3vl')
    with pytest.raises(ValueError,match='different execution host'):e.validate_host(root,['1'],'local_model')
    with pytest.raises(ValueError,match='hostname/project'):e.local_host(root,'4028')
    monkeypatch.setattr(e.subprocess,'check_output',lambda *a,**kw:'1, wrong-uuid\n')
    with pytest.raises(ValueError,match='UUID'):e.validate_host(root,['1'],'qwen3vl')
    monkeypatch.setattr(e.socket,'gethostname',lambda:'other')
    with pytest.raises(ValueError,match='hostname/project'):e.local_host(root)


def test_mapping_preserves_logical_sample_and_checks_bytes_once_until_file_changes(host,monkeypatch):
    root,_,write=host
    logical='/logical/food/a.jpg';sample={'id':'a','image_path':logical};before=json.dumps(sample)
    calls=[];original=e.file_hash
    def hashed(path):calls.append(str(path));return original(path)
    monkeypatch.setattr(e,'file_hash',hashed)
    target=e.resolve_image_path(logical,root)
    assert target==root/'data/images/a.jpg' and json.dumps(sample)==before
    e.resolve_image_path(logical,root)
    assert calls.count(str(target))==1
    target.write_text('altered image bytes!')
    with pytest.raises(ValueError,match='content differs'):e.resolve_image_path(logical,root)


@pytest.mark.parametrize('logical',['/logical/foodish/a.jpg','/logical/food/../a.jpg','/unregistered/a.jpg'])
def test_mapping_rejects_unregistered_or_traversing_paths(host,logical):
    with pytest.raises(ValueError,match='logical source'):e.resolve_image_path(logical,host[0])


def test_mapping_rejects_symlink_escape(host):
    root,_,write=host
    target=root/'data/images/a.jpg';target.unlink()
    elsewhere=write('outside.jpg','original image bytes');target.symlink_to(elsewhere)
    with pytest.raises(ValueError,match='escapes'):e.resolve_image_path('/logical/food/a.jpg',root)


def test_local_preflight_requires_locks_env_weights_but_not_finished_native_proof(host,monkeypatch):
    from kdm.protocol import validate_execution_runtime
    root,_,write=host
    weight=write('weights/one.bin','weight')
    spec={'key':'qwen3vl','availability':'resolved','gpu_count':1,'environment_python':sys.executable,
        'versions':{},'kwargs':{'model_path':str(weight.parent)},'weights':[{'filename':'one.bin','size_bytes':6}],
        'interface_verification':{'status':'pending'}}
    with pytest.raises(ValueError,match='[Ww]orker'):validate_execution_runtime(root,spec,'qwen3vl',['1'])
    old=os.readlink
    monkeypatch.setattr(os,'readlink',lambda p,*a,**kw:str(root/'outputs/locks/gpu_1.lock') if str(p)=='/proc/self/fd/20' else old(p,*a,**kw))
    receipt=validate_execution_runtime(root,spec,'qwen3vl',['1'])
    assert receipt['host']=='6403'
    weight.unlink()
    with pytest.raises(ValueError,match='registered model weight'):validate_execution_runtime(root,spec,'qwen3vl',['1'])
    spec['environment_python']='/wrong/python'
    with pytest.raises(ValueError,match='environment Python'):validate_execution_runtime(root,spec,'qwen3vl',['1'])


def test_portable_execution_receipt_never_checks_consumer_host_gpu_or_weights(host,monkeypatch):
    root,_,write=host
    receipt=e.execution_receipt(root,['1'],'qwen3vl')
    monkeypatch.setattr(e.socket,'gethostname',lambda:'unrelated-consumer')
    monkeypatch.setattr(e.subprocess,'check_output',lambda *a,**kw:pytest.fail('Portable receipt queried a GPU'))
    e.validate_execution_receipt(root,receipt,'qwen3vl',1)
    with pytest.raises(ValueError):e.validate_execution_receipt(root,receipt,'local_model',1)
    receipt['physical_gpus']=['0']
    with pytest.raises(ValueError,match='unauthorized'):e.validate_execution_receipt(root,receipt,'qwen3vl',1)


def test_native_proof_is_portable_but_local_weight_gate_still_refuses_missing_weights(host,monkeypatch):
    from kdm.protocol import validate_native_runtime_files,validate_local_checkpoint
    root,_,write=host
    manifest=write('data/current/interface16.jsonl','fixed-interface')
    for name in ('hf.py','backbone.py'):write('src/kdm/models/'+name,'fixed-'+name)
    spec={'key':'qwen3vl','gpu_count':1,'kwargs':{'model_path':'/remote/weights/not-on-consumer'},
        'weights':[{'filename':'part.safetensors','size_bytes':100}],
        'interface_verification':{'status':'passed','record':'proof.json'}}
    proof={'passed':True,'completed':16,'expected':16,'spec':spec,'manifest_sha256':file_hash(manifest),
        'execution':e.execution_receipt(root,['1'],'qwen3vl'),
        'runtime_adapter_sha256':{n:file_hash(root/'src/kdm/models'/n) for n in ('hf.py','backbone.py')}}
    write('proof.json',proof)
    monkeypatch.setattr(e.socket,'gethostname',lambda:'consumer')
    validate_native_runtime_files(root,spec,'qwen3vl')
    with pytest.raises(ValueError,match='Missing or incomplete'):validate_local_checkpoint(spec)


@pytest.mark.parametrize('mutation',['none','missing_stage','old_environment','wrong_stage','incomplete_visits','duplicate_branch','wrong_maximum','wrong_count_record','changed_script','wrong_host'])
def test_remote_resource_gate_requires_all_matching_current_proofs(host,mutation):
    from kdm.protocol import validate_resource_runtime
    root,_,write=host
    fields={'key':'qwen3vl','gpu_count':1,'environment_python':'unit-test-python','kwargs':{'model_path':'/remote/weights'},'versions':{'transformers':'unit-test'}}
    spec={**fields,'visual_count_verification':{'status':'passed','record':'counts.json'},
        'resource_verification':{'status':'passed','records':{stage:stage+'.json' for stage in ('layer','instruction_vcd','cda')}}}
    sample={'id':'vizwiz:largest','dataset':'vizwiz','split':'eval','image_path':'/logical/viz/largest.jpg'}
    write('data/current/all.jsonl',json.dumps(sample)+'\n')
    for path in ('verification/check_full_visual_counts.py','verification/max_input_resource_check.py','src/kdm/models/hf.py','src/kdm/models/backbone.py'):write(path,'test-source '+path)
    write('headers.json',{'rows':[]})
    counts={'processor_execution':{'host':'6403','hostname':'remote','environment_python':spec['environment_python'],'versions':spec['versions']},'spec':spec,'manifest_sha256':file_hash(root/'data/current/all.jsonl'),'total_images':9167,
        'boundary_checks_passed':True,'groups':[{'dataset':'vizwiz','maximum':1024}],
        'boundary_checks':[{'id':sample['id'],'formula_tokens':1024,'equal':True}],
        'script_sha256':file_hash(root/'verification/check_full_visual_counts.py'),
        'header_record':'headers.json','header_record_sha256':file_hash(root/'headers.json')}
    write('counts.json',counts)
    branches={'layer':['main_need_layers'],'instruction_vcd':['main','noise_reference','neutral'],
        'cda':['prior_text','context_image','abstention_image','null_prior_text','null_context_image']}
    for stage,names in branches.items():
        proof={'passed':True,'phase':'complete','stage':stage,'spec':spec,'sample':sample,
            'script_sha256':file_hash(root/'verification/max_input_resource_check.py'),
            'manifest_sha256':file_hash(root/'data/current/all.jsonl'),'count_record':'counts.json','count_record_sha256':file_hash(root/'counts.json'),
            'expected_visual_tokens':1024,'execution':e.execution_receipt(root,['1'],'qwen3vl'),
            'runtime_adapter_sha256':{n:file_hash(root/'src/kdm/models'/n) for n in ('hf.py','backbone.py')},
            'visits':[{'prefix_length':i,'branch':branch,'logits_finite':True,'all_layer_logits_finite':True} for i in (0,1,2) for branch in names]}
        write(stage+'.json',proof)
    path=root/'cda.json';proof=json.loads(path.read_text())
    if mutation=='none':assert 'counts.json' in validate_resource_runtime(root,spec);return
    if mutation=='missing_stage':spec['resource_verification']['records'].pop('layer')
    elif mutation=='old_environment':proof['spec']['versions']['transformers']='wrong'
    elif mutation=='wrong_stage':proof['stage']='layer'
    elif mutation=='incomplete_visits':proof['visits'].pop()
    elif mutation=='duplicate_branch':proof['visits'][1]=proof['visits'][0]
    elif mutation=='wrong_maximum':proof['expected_visual_tokens']=100
    elif mutation=='wrong_count_record':proof['count_record']='elsewhere.json'
    elif mutation=='changed_script':write('verification/max_input_resource_check.py','different')
    elif mutation=='wrong_host':proof['execution']['host']='4028'
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):validate_resource_runtime(root,spec)


def test_processor_count_preflight_is_cpu_only_and_checks_environment(host,monkeypatch):
    from kdm.protocol import validate_processor_runtime
    root,_,write=host
    weight=write('weights/part','w')
    spec={'key':'qwen3vl','environment_python':sys.executable,'versions':{},'kwargs':{'model_path':str(weight.parent)},'weights':[{'filename':'part','size_bytes':1}]}
    monkeypatch.setattr(e.subprocess,'check_output',lambda *a,**kw:pytest.fail('CPU count preflight queried GPU'))
    assert validate_processor_runtime(root,spec)['host']=='6403'
    spec['environment_python']='/different/python'
    with pytest.raises(ValueError,match='environment Python'):validate_processor_runtime(root,spec)
