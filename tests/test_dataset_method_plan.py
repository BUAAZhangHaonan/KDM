import importlib.util,json
from pathlib import Path
import pytest
from kdm.reports import validate_report_coverage,validated_method_plan
from kdm.pipeline import experiment_tasks,probe_tasks,task_id
from kdm.io import file_hash
BASE=['vcd','m3id','dola','deco']


def rows(samples,methods,model='m'):
    return [{**t,'model':model,'key':task_id(model,t),'status':'ok'} for t in [*experiment_tasks(samples,methods),*probe_tasks(samples)]]


def dataset_fixture():
    food=[{'id':'f'+str(i),'dataset':'food101','class':'c'+str(i),'split':'eval' if i==0 else 'dev'} for i in range(101)]
    viz=[{'id':'v','dataset':'vizwiz','split':'eval'}]
    selection=[{'model':m,'dataset':d,'selected':m=='m'} for m in ('m','not_selected') for d in ('food101','vizwiz')]
    plan={m:{'food101':BASE+['sid'],'vizwiz':BASE} for m in ('m','not_selected')}
    closed=[{'model':'m','sample':food[0],'candidate_scores':[{'label':s['class'],'sum_logp':-1.,'mean_logp':-1.,'n_tokens':1} for s in food],
        'gold_rank':1,'ranking_rule':'mean_log_probability','target':'c0'}]
    return food,viz,selection,plan,closed


def test_dataset_plan_requires_full_sid_only_in_its_declared_condition():
    food,viz,selection,plan,closed=dataset_fixture()
    records=rows(food,BASE+['sid'])+rows(viz,BASE)
    result=validate_report_coverage(records,food+viz,selection,closed,plan)
    assert result['selected_conditions']==2 and 'not_selected' in result['method_plan']
    with pytest.raises(ValueError,match='Missing frozen report task'):
        validate_report_coverage([r for r in records if r['method']!='sid'],food+viz,selection,closed,plan)
    extra=[r for r in rows(viz,BASE+['sid']) if r['method']=='sid']
    with pytest.raises(ValueError,match='Unexpected or unselected'):
        validate_report_coverage(records+extra,food+viz,selection,closed,plan)
    # The excluded SID condition still requires every question for all four methods and probes.
    with pytest.raises(ValueError,match='Missing frozen report task'):
        validate_report_coverage([r for r in records if r['sample']['dataset']!='vizwiz'],food+viz,selection,closed,plan)


@pytest.mark.parametrize('plan',[{'m':BASE},{'m':{'food101':BASE}}, {'m':{'food101':BASE,'vizwiz':['vcd','sid']}}])
def test_old_partial_or_missing_baseline_plans_are_rejected(plan):
    with pytest.raises(ValueError):validated_method_plan(plan,[('m','food101'),('m','vizwiz')])


def test_preparation_preserves_all_selected_samples_splits_and_plan_arguments(tmp_path,monkeypatch):
    source=Path(__file__).parents[1]/'scripts/prepare_selected_manifests.py'
    spec=importlib.util.spec_from_file_location('prepare_conditions',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    image=tmp_path/'outputs/verification/image.png';image.parent.mkdir(parents=True);image.write_bytes(b'existence fixture')
    samples=[{'id':d+split,'dataset':d,'split':split,'image_path':str(image)} for d in ('food101','vizwiz') for split in ('dev','eval')]
    manifest=tmp_path/'all.jsonl';manifest.write_text(''.join(json.dumps(s)+'\n' for s in samples))
    selection=[{'model':m,'dataset':d,'n':2,'n_abstain':1 if m=='m' else 0,'selected':m=='m'} for m in ('m','n') for d in ('food101','vizwiz')]
    selected=tmp_path/'selection.json';selected.write_text(json.dumps(selection))
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({m:{'food101':BASE+['sid'],'vizwiz':BASE} for m in ('m','n')}))
    (tmp_path/'configs/runtime').mkdir(parents=True);(tmp_path/'configs/runtime/m.json').write_text(json.dumps({'key':'m'}))
    (tmp_path/'outputs/records').mkdir(parents=True)
    (tmp_path/'outputs/records/preregistration_freeze.json').write_text(json.dumps({'status':'frozen','files':{'plan.json':file_hash(plan)}}))
    checked=[];monkeypatch.setattr(mod,'validate_method_runtime',lambda root,spec,methods:checked.append(methods))
    monkeypatch.setattr('sys.argv',['prepare','--root',str(tmp_path),'--manifest',str(manifest),'--selection',str(selected),'--method-plan',str(plan),'--out-dir','prepared'])
    mod.main()
    receipt=json.loads((tmp_path/'prepared/execution_plan.json').read_text())
    assert len(receipt['conditions'])==2 and receipt['method_plan_sha256']==file_hash(plan)
    assert receipt['method_plan']=='plan.json'
    assert checked==[tuple(BASE+['sid']),tuple(BASE)] and not (tmp_path/'prepared/n').exists()
    for item in receipt['conditions']:
        saved=[json.loads(line) for line in (tmp_path/item['manifest']).read_text().splitlines()]
        assert saved==[s for s in samples if s['dataset']==item['dataset']]
        assert item['samples']==2 and item['eval_samples']==1 and '--method-plan' in item['experiment_arguments']
        assert item['experiment_arguments'][item['experiment_arguments'].index('--method-plan')+1]=='plan.json'
    # An unchanged manifest/plan is reusable; no different previous artifact is overwritten.
    mod.main()
    (tmp_path/'prepared/m/vizwiz.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='overwrite'):mod.main()


@pytest.mark.parametrize('mutation',['none','missing_plan','wrong_methods','mixed_dataset','missing_sample','changed_plan'])
def test_formal_experiment_task_plan_gate_precedes_backend(tmp_path,monkeypatch,mutation):
    import kdm.cli as cli
    import kdm.protocol as protocol
    import kdm.pipeline as pipeline
    (tmp_path/'data/current').mkdir(parents=True);(tmp_path/'configs/kdm').mkdir(parents=True);(tmp_path/'outputs/records').mkdir(parents=True)
    original=[{'id':d+str(i),'dataset':d,'split':'dev' if i==0 else 'eval'} for d in ('food101','vizwiz') for i in (0,1)]
    all_path=tmp_path/'data/current/all.jsonl';all_path.write_text(''.join(json.dumps(s)+'\n' for s in original))
    selected=[s for s in original if s['dataset']=='vizwiz']
    if mutation=='mixed_dataset':selected=original
    if mutation=='missing_sample':selected=selected[1:]
    manifest=tmp_path/'selected.jsonl';manifest.write_text(''.join(json.dumps(s)+'\n' for s in selected))
    (tmp_path/'configs/kdm/models.json').write_text(json.dumps([{'key':'m'}]))
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'m':{'food101':BASE+['sid'],'vizwiz':BASE}}))
    freeze={'status':'frozen','files':{'plan.json':file_hash(plan),'data/current/all.jsonl':file_hash(all_path)}}
    (tmp_path/'outputs/records/preregistration_freeze.json').write_text(json.dumps(freeze))
    if mutation=='changed_plan':plan.write_text(plan.read_text()+' ')
    spec=tmp_path/'spec.json';spec.write_text('{}')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setattr(protocol,'validate_runtime',lambda *args:None)
    monkeypatch.setattr(protocol,'code_identity',lambda *args:{})
    def frozen(root):
        value=json.loads((root/'outputs/records/preregistration_freeze.json').read_text())
        value['files']['configs/kdm/method_plan.json']=value['files']['plan.json'];value['source_blobs']=[]
        return value
    monkeypatch.setattr(protocol,'validate_freeze',frozen)
    def method_proof(root,spec,methods):assert list(methods)==BASE
    monkeypatch.setattr(protocol,'validate_method_runtime',method_proof)
    def backend(*args):raise RuntimeError('backend admission reached')
    monkeypatch.setattr(pipeline,'make_backend',backend)
    args=['--root',str(tmp_path),'run','--mode','experiment','--manifest',str(manifest),'--model-spec',str(spec),'--model','m','--gpu','0','--out','out.jsonl',
        '--methods',','.join(BASE+(['sid'] if mutation=='wrong_methods' else []))]
    if mutation!='missing_plan':args+=['--method-plan',str(plan)]
    with pytest.raises(RuntimeError if mutation=='none' else ValueError) as exc:cli.main(args)
    if mutation=='none':assert 'backend admission reached' in str(exc.value)


def test_selected_plan_and_manifest_are_identical_across_hosts_and_reusable_after_sync(tmp_path,monkeypatch):
    import shutil
    import kdm.execution as execution
    from kdm.io import stable_hash
    source=Path(__file__).parents[1]/'scripts/prepare_selected_manifests.py'
    module=importlib.util.spec_from_file_location('prepare_portable_conditions',source)
    mod=importlib.util.module_from_spec(module);module.loader.exec_module(mod)
    central=tmp_path/'central';remote=tmp_path/'remote'
    logical='/unmounted-original-host/images/original.jpg'
    assert not Path(logical).exists()
    samples=[{'id':dataset+split,'dataset':dataset,'split':split,'image_path':logical,'question':'q'}
             for dataset in ('food101','vizwiz') for split in ('dev','eval')]
    original_rows=json.loads(json.dumps(samples))
    original_tasks=[task_id('m',t) for t in experiment_tasks(samples)]
    selection=[{'model':'m','dataset':d,'n':2,'n_abstain':1,'selected':True} for d in ('food101','vizwiz')]
    registry={'schema':1,'hosts':{
        '4028':{'root':str(central),'hostname':'central','allowed_gpus':[0],'gpu_uuids':{'0':'unused'}},
        '6403':{'root':str(remote),'hostname':'remote','allowed_gpus':[1],'gpu_uuids':{'1':'unused'}}},
        'model_hosts':{'m':'6403'},'image_prefixes':{'/unmounted-original-host/images':'data/images'},
        'image_catalog':'outputs/records/image_content_catalog.json'}
    def write(root,path,payload):
        target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(payload if isinstance(payload,str) else json.dumps(payload));return target
    for root in (central,remote):
        manifest=write(root,'data/current/all.jsonl',''.join(json.dumps(s)+'\n' for s in samples))
        image=write(root,'data/images/original.jpg','exact original bytes')
        write(root,'selection.json',selection)
        plan=write(root,'configs/kdm/method_plan.json',{'m':{'food101':BASE,'vizwiz':BASE}})
        write(root,'configs/runtime/m.json',{'key':'m'})
        write(root,'configs/runtime/hosts.json',registry)
        write(root,'outputs/records/preregistration_freeze.json',{'status':'frozen','files':{'configs/kdm/method_plan.json':file_hash(plan)}})
        write(root,'outputs/records/image_content_catalog.json',{'schema':1,'manifest_sha256':file_hash(manifest),
            'images':{logical:{'sha256':file_hash(image),'size_bytes':image.stat().st_size}}})
    monkeypatch.setattr(mod,'validate_method_runtime',lambda *args:None)
    def prepare(root,name):
        monkeypatch.setattr(execution.socket,'gethostname',lambda:name)
        monkeypatch.setattr('sys.argv',['prepare','--root',str(root),'--manifest','data/current/all.jsonl',
            '--selection','selection.json','--method-plan','configs/kdm/method_plan.json','--out-dir','prepared'])
        mod.main()
    # CWD is neither project root: relative CLI inputs resolve against explicit --root.
    monkeypatch.chdir(tmp_path)
    prepare(central,'central')
    shutil.copytree(central/'prepared',remote/'prepared')
    central_receipt=(central/'prepared/execution_plan.json').read_bytes()
    prepare(remote,'remote')
    assert (remote/'prepared/execution_plan.json').read_bytes()==central_receipt
    # Recreate the selected manifests on the remote host from unchanged original rows.
    for dataset in ('food101','vizwiz'):(remote/f'prepared/m/{dataset}.jsonl').unlink()
    prepare(remote,'remote')
    assert (remote/'prepared/execution_plan.json').read_bytes()==central_receipt
    receipt=json.loads(central_receipt)
    assert receipt['method_plan']=='configs/kdm/method_plan.json'
    saved=[]
    for condition in receipt['conditions']:
        args=condition['experiment_arguments']
        for option in ('--manifest','--model-spec','--method-plan'):
            value=args[args.index(option)+1]
            assert not Path(value).is_absolute() and (remote/value).is_file()
        relative=condition['manifest']
        assert (central/relative).read_bytes()==(remote/relative).read_bytes()
        saved.extend(json.loads(line) for line in (remote/relative).read_text().splitlines())
    assert saved==original_rows
    assert [task_id('m',t) for t in experiment_tasks(saved)]==original_tasks
    # A changed mapped image cannot generate a new selected manifest.
    (remote/'data/images/original.jpg').write_text('different image bytes')
    (remote/'prepared/m/food101.jsonl').unlink()
    with pytest.raises(ValueError,match='content differs'):prepare(remote,'remote')
    assert not (remote/'prepared/m/food101.jsonl').exists()
