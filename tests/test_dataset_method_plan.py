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
    image=tmp_path/'image.png';image.write_bytes(b'existence fixture')
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
    assert checked==[tuple(BASE+['sid']),tuple(BASE)] and not (tmp_path/'prepared/n').exists()
    for item in receipt['conditions']:
        saved=[json.loads(line) for line in (tmp_path/item['manifest']).read_text().splitlines()]
        assert saved==[s for s in samples if s['dataset']==item['dataset']]
        assert item['samples']==2 and item['eval_samples']==1 and '--method-plan' in item['experiment_arguments']
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
    def method_proof(root,spec,methods):assert list(methods)==BASE
    monkeypatch.setattr(protocol,'validate_method_runtime',method_proof)
    def backend(*args):raise RuntimeError('backend admission reached')
    monkeypatch.setattr(pipeline,'make_backend',backend)
    args=['--root',str(tmp_path),'run','--mode','experiment','--manifest',str(manifest),'--model-spec',str(spec),'--model','m','--gpu','0','--out','out.jsonl',
        '--methods',','.join(BASE+(['sid'] if mutation=='wrong_methods' else []))]
    if mutation!='missing_plan':args+=['--method-plan',str(plan)]
    with pytest.raises(RuntimeError if mutation=='none' else ValueError) as exc:cli.main(args)
    if mutation=='none':assert 'backend admission reached' in str(exc.value)
