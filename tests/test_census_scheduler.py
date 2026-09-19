"""CPU process doubles only; all fixture files remain in pytest's project cache."""
import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import pytest
from kdm.io import file_hash


@pytest.fixture
def scheduler(tmp_path,monkeypatch):
    source=Path(__file__).parents[1]/'scripts/run_census_panel.py'
    spec=importlib.util.spec_from_file_location('census_scheduler',source)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setattr(mod,'ROOT',tmp_path);monkeypatch.setattr(mod,'PROJECT_ROOT',tmp_path)
    import kdm.protocol as protocol
    monkeypatch.setattr(protocol,'code_identity',lambda root:['synthetic-committed-source'])
    def write(name,text):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);return path
    keys=[key for lane in mod.LANES.values() for key in lane['models']]
    write('configs/kdm/models.json',json.dumps([{'key':key} for key in keys]))
    write('data/current/all.jsonl',''.join(json.dumps({'id':str(i),'dataset':'food101' if i<4848 else 'vizwiz'})+'\n' for i in range(9167)))
    write('data/current/interface16.jsonl','synthetic interface identity')
    write('configs/kdm/method_plan.json',json.dumps({key:{d:['vcd','m3id','dola','deco'] for d in ('food101','vizwiz')} for key in keys}))
    write('proof.json',json.dumps({'passed':True,'completed':16,'expected':16,'layer_projection_check':{'status':'passed','head_final_error':0.,'raw_projection_error':0.,'norm_projection_error':0.,'raw_layers':32,'normalized_layers':[20]}}))
    adapter_names=['hf.py','backbone.py','sid.py','remote.py','internvl_preprocessing.py']
    for name in adapter_names:write('src/kdm/models/'+name,'synthetic '+name)
    for lane in mod.LANES.values():
        for key in lane['models']:
            runtime={'key':key,'availability':'resolved','gpu_count':len(lane['cards']),
                'environment_python':'SYNTHETIC_NO_EXECUTION','weights':[],
                'interface_verification':{'status':'passed','record':key+'_proof.json'}}
            write('configs/runtime/'+key+'.json',json.dumps(runtime))
            proof=json.loads((tmp_path/'proof.json').read_text());proof.update(spec=runtime,
                manifest_sha256=file_hash(tmp_path/'data/current/interface16.jsonl'),
                runtime_adapter_sha256={name:file_hash(tmp_path/'src/kdm/models'/name) for name in adapter_names})
            write(key+'_proof.json',json.dumps(proof))
    names=['docs/current/PREREGISTER.md','configs/kdm/food_aliases.json',
        'outputs/records/protocol_user_decisions_20260919.json','scripts/run_census_panel.py',
        'scripts/worker.sh','scripts/verify_complete.py','configs/runtime/semantic_judge.json']
    for name in names:write(name,'synthetic fixture')
    names+=['data/current/all.jsonl','data/current/interface16.jsonl','configs/kdm/models.json','configs/kdm/method_plan.json']+[key+'_proof.json' for key in keys]+[f'configs/runtime/{key}.json' for key in keys]
    freeze={'status':'frozen','source_blobs':['synthetic-committed-source'],'files':{name:file_hash(tmp_path/name) for name in names}}
    write('outputs/records/preregistration_freeze.json',json.dumps(freeze))
    (tmp_path/'outputs/locks').mkdir()
    return mod,tmp_path,freeze


def test_success_exact_panel_and_pair_stage_order(scheduler,monkeypatch):
    mod,root,_=scheduler;events=[];guard=threading.Lock()
    class Process:
        pid=999
        def __init__(self,command,**kwargs):
            self.key=command[command.index('--model')+1]
            with guard:events.append(('launch',self.key,command[3]))
        def wait(self):
            with guard:events.append(('finish',self.key))
            return 0
    def verify(command,**kwargs):
        key=command[command.index('--model')+1]
        with guard:events.append(('verified',key))
        assert '--mode' in command and command[command.index('--mode')+1]=='census'
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(mod.subprocess,'Popen',Process);monkeypatch.setattr(mod.subprocess,'run',verify)
    assert mod.execute('outputs/records/synthetic-run')==0
    launched=[row[1] for row in events if row[0]=='launch']
    verified=[row[1] for row in events if row[0]=='verified']
    assert len(launched)==len(set(launched))==16 and set(verified)==set(launched)
    last_pair=max(i for i,row in enumerate(events) if row[0]=='verified' and row[1] in mod.LANES['pair45']['models'])
    singles=mod.LANES['gpu4_after_pairs']['models']+mod.LANES['gpu5_after_pairs']['models']
    assert all(i>last_pair for i,row in enumerate(events) if row[0]=='launch' and row[1] in singles)
    report=json.loads((root/'outputs/records/synthetic-run/status.json').read_text())
    assert report['complete'] and all(row['status']=='complete' for row in report['jobs'].values())
    assert report['schedule']['expected_panel_responses']==293344


@pytest.mark.parametrize('failure_stage',['generation','verification'])
def test_failure_stops_new_work_without_retry_or_killing_running_jobs(scheduler,monkeypatch,failure_stage):
    mod,root,_=scheduler;started=[];waited=[];verified=[];guard=threading.Lock()
    three_started=threading.Event();failure_reported=threading.Event()
    real_atomic=mod.atomic_json
    def atomic(path,data):
        real_atomic(path,data)
        if data.get('jobs',{}).get('glm46v',{}).get('status')=='failed':failure_reported.set()
    monkeypatch.setattr(mod,'atomic_json',atomic)
    class Process:
        pid=999
        def __init__(self,command,**kwargs):
            self.key=command[command.index('--model')+1]
            with guard:
                started.append(self.key)
                if len(started)==3:three_started.set()
        def wait(self):
            if self.key=='glm46v':assert three_started.wait(5)
            else:assert failure_reported.wait(5)
            with guard:waited.append(self.key)
            return 9 if self.key=='glm46v' and failure_stage=='generation' else 0
    def verify(command,**kwargs):
        key=command[command.index('--model')+1];verified.append(key)
        return SimpleNamespace(returncode=8 if key=='glm46v' and failure_stage=='verification' else 0)
    monkeypatch.setattr(mod.subprocess,'Popen',Process);monkeypatch.setattr(mod.subprocess,'run',verify)
    assert mod.execute('outputs/records/synthetic-failure')==1
    assert set(started)=={'glm46v','qwen35_9b','gemma3_12b'} and len(started)==3
    assert set(waited)==set(started)
    report=json.loads((root/'outputs/records/synthetic-failure/status.json').read_text())
    assert not report['complete'] and report['jobs']['glm46v']['status']=='failed'
    assert sum(row['status']=='not_started' for row in report['jobs'].values())==13
    if failure_stage=='generation':assert 'glm46v' not in verified


@pytest.mark.parametrize('mutation',['missing','unfrozen','changed_script','changed_spec','omitted_scheduler','omitted_worker','omitted_verifier','omitted_judge','omitted_method_plan','omitted_native_proof','source_change','self_reference'])
def test_freeze_rejection_never_launches(scheduler,monkeypatch,mutation):
    mod,root,freeze=scheduler;path=root/'outputs/records/preregistration_freeze.json'
    if mutation=='missing':path.unlink()
    elif mutation=='unfrozen':freeze['status']='draft'
    elif mutation=='source_change':freeze['source_blobs']=['different-active-source']
    elif mutation=='self_reference':freeze['files']['outputs/records/preregistration_freeze.json']='irrelevant'
    elif mutation=='changed_script':(root/'scripts/worker.sh').write_text('changed')
    elif mutation=='changed_spec':
        spec_path=root/'configs/runtime/glm46v.json';data=json.loads(spec_path.read_text());data['dtype']='changed';spec_path.write_text(json.dumps(data))
    else:
        omitted={'omitted_scheduler':'scripts/run_census_panel.py','omitted_worker':'scripts/worker.sh',
            'omitted_verifier':'scripts/verify_complete.py','omitted_judge':'configs/runtime/semantic_judge.json',
            'omitted_method_plan':'configs/kdm/method_plan.json','omitted_native_proof':'glm46v_proof.json'}[mutation]
        del freeze['files'][omitted]
    if mutation!='missing':path.write_text(json.dumps(freeze))
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**kw:pytest.fail('Invalid freeze launched a process'))
    with pytest.raises((ValueError,FileNotFoundError)):mod.execute('outputs/records/rejected')
    assert not (root/'outputs/records/rejected').exists()


@pytest.mark.parametrize('mutation',['duplicate_panel','missing_lane_model','wrong_gpu_count'])
def test_schedule_rejects_incomplete_duplicate_or_misallocated_models(scheduler,monkeypatch,mutation):
    mod,root,_=scheduler
    if mutation=='duplicate_panel':
        path=root/'configs/kdm/models.json';panel=json.loads(path.read_text());panel.append(panel[0]);path.write_text(json.dumps(panel))
    elif mutation=='missing_lane_model':
        lanes=json.loads(json.dumps(mod.LANES));lanes['gpu1']['models'].pop();monkeypatch.setattr(mod,'LANES',lanes)
    else:
        path=root/'configs/runtime/internvl35_8b.json';data=json.loads(path.read_text());data['gpu_count']=1;path.write_text(json.dumps(data))
    with pytest.raises(ValueError):mod.plan()


@pytest.mark.parametrize('mutation',['missing_condition','missing_baseline','unproved_sid'])
def test_frozen_plan_structure_and_method_proofs_are_real_gates(scheduler,monkeypatch,mutation):
    mod,root,freeze=scheduler;plan_path=root/'configs/kdm/method_plan.json';plan=json.loads(plan_path.read_text())
    if mutation=='missing_condition':del plan['glm46v']['vizwiz']
    elif mutation=='missing_baseline':plan['glm46v']['food101'].remove('deco')
    else:plan['glm46v']['food101'].append('sid')
    plan_path.write_text(json.dumps(plan));freeze['files']['configs/kdm/method_plan.json']=file_hash(plan_path)
    (root/'outputs/records/preregistration_freeze.json').write_text(json.dumps(freeze))
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**kw:pytest.fail('Unready method plan launched a process'))
    with pytest.raises(ValueError):mod.execute('outputs/records/rejected-plan')


def test_unrelated_files_do_not_change_frozen_active_source_identity(scheduler):
    mod,root,_=scheduler
    (root/'unrelated-note.md').write_text('unrelated document fixture')
    assert mod.validate_freeze(root)['source_blobs']==['synthetic-committed-source']


@pytest.mark.parametrize('mutation',['omitted_sid_proof','failed_sid_checks','valid_sid'])
def test_freeze_requires_independent_frozen_sid_proof(scheduler,mutation):
    mod,root,freeze=scheduler
    plan_path=root/'configs/kdm/method_plan.json';plan=json.loads(plan_path.read_text());plan['glm46v']['food101'].append('sid');plan_path.write_text(json.dumps(plan))
    runtime_path=root/'configs/runtime/glm46v.json';runtime=json.loads(runtime_path.read_text())
    runtime['mechanism_validation']={'sid_reference':{'status':'passed','record':'sid_proof.json'}};runtime_path.write_text(json.dumps(runtime))
    proof={'passed':True,'reference_commit':'127dd412fa6b61ab1c9babf6979ec4da98002438','spec':runtime,
        'checks':dict.fromkeys(['official_selection','official_reference_logits','causal_mask_preserved','interleaved_sessions','nonmonotonic_prefix'],True),
        'runtime_adapter_sha256':{name:file_hash(root/'src/kdm/models'/name) for name in ('hf.py','backbone.py','sid.py')}}
    if mutation=='failed_sid_checks':proof['checks']['interleaved_sessions']=False
    (root/'sid_proof.json').write_text(json.dumps(proof))
    for relative in ('configs/kdm/method_plan.json','configs/runtime/glm46v.json'):freeze['files'][relative]=file_hash(root/relative)
    if mutation!='omitted_sid_proof':freeze['files']['sid_proof.json']=file_hash(root/'sid_proof.json')
    (root/'outputs/records/preregistration_freeze.json').write_text(json.dumps(freeze))
    if mutation=='valid_sid':assert mod.validate_freeze(root)['status']=='frozen'
    else:
        with pytest.raises(ValueError):mod.validate_freeze(root)


def test_native_evidence_must_match_frozen_runtime_before_dispatch(scheduler):
    mod,root,freeze=scheduler
    path=root/'configs/runtime/gemma3_4b.json';spec=json.loads(path.read_text());spec['dtype']='changed';path.write_text(json.dumps(spec))
    freeze['files']['configs/runtime/gemma3_4b.json']=file_hash(path)
    (root/'outputs/records/preregistration_freeze.json').write_text(json.dumps(freeze))
    with pytest.raises(ValueError,match='changed after interface'):mod.validate_freeze(root)
