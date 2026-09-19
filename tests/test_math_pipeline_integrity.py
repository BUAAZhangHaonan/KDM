"""Regression evidence for frozen coverage and exact-prefix probability contracts."""
import importlib.util
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pytest
from PIL import Image
from kdm.decoding import DecodeConfig,Step,generate,replay
from kdm.pipeline import finite_response_audit,experiment_tasks,census_tasks,task_id
from kdm.models.mock import MockBackend,MockSession
from kdm.analysis import evaluate,probe_summary,annotated_rows
from kdm.reports import behavioral_support,evidence_indices
from kdm.scoring import label_response
from kdm.prompts import MARKERS


def script(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parents[1]/'scripts'/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.mark.parametrize('method',['instruction_vcd','instruction_m3id'])
def test_three_condition_replay_equals_generation(method):
    b=MockBackend();cfg=DecodeConfig(method=method,m3id_offset=9,m3id_threshold=.9)
    def sessions():return b.session(None,'g'),b.session(None,'r','noise'),b.session(None,'c')
    g,r,c=sessions();out=generate(g,r,cfg,b.eos,b.decode,neutral_main=c)
    g,r,c=sessions();rows=replay(g,r,cfg,out['tokens'],neutral_main=c)
    assert g.calls==r.calls==c.calls
    assert [row['modified_logp'] for row in rows]==out['selected_log_probabilities']
    for row in rows:
        reconstructed=row['base_logp']+row['weight']*(row['neutral_clean_logp']-row['reference_logp'])-row['log_normalizer']
        assert reconstructed==pytest.approx(row['modified_logp'],abs=1e-12)
        assert row['prefix']==out['tokens'][:row['step']]


@pytest.mark.parametrize('method',['vcd','m3id','dola','deco','instruction_vcd','instruction_m3id'])
def test_finite_audit_scores_all_eos_for_terminated_donor(method):
    b=MockBackend();b.eos={1,3}
    # Marker encodings may not contain an EOS in their interior for this fixture.
    b.encode=lambda text:[0] if text in MARKERS else [2]
    result=finite_response_audit(b,Image.new('RGB',(8,8)),'q','UNKNOWN','UNCLEAR',
        [{'kind':'answer','text':'milk','tokens':[2,3]}],DecodeConfig(method=method))
    assert {tuple(r['tokens']) for r in result['responses'] if r['kind']=='answer'}=={(2,1),(2,3)}
    for row in result['responses']:
        assert row['modified_logp']==sum(s['modified_logp'] for s in row['steps'])


def test_sid_has_full_reference_prompt_matrix():
    tasks=list(experiment_tasks([{'id':'x','split':'eval'}],methods=('sid',)))
    assert len([t for t in tasks if t['method']=='sid'])==16


def test_completion_rejects_matching_key_with_failed_or_wrong_task():
    verify=script('verify_complete').verify
    sample={'id':'x','split':'eval'};tasks=list(census_tasks([sample]))
    records=[{**t,'model':'m','key':task_id('m',t),'status':'ok','text':'x','tokens':[1],'terminated':True} for t in tasks]
    assert verify([sample],records,'m','census')['complete']
    records[0]['status']='error'
    assert not verify([sample],records,'m','census')['complete']
    records[0]['status']='ok';records[0]['guided']=False
    assert not verify([sample],records,'m','census')['complete']


def test_donor_pool_requires_every_prompt_and_termination():
    donor_pool=script('complete_response_audit').donor_pool
    rows=[{'model':'m','sample':{'id':'x'},'method':'direct','guided':True,'kind':'main',
           'marker':marker,'text':'UNKNOWN','key':marker,'tokens':[0,3],'terminated':True,'status':'ok'} for marker in MARKERS]
    assert len(donor_pool(rows,'m',{})[0]['x'])==4
    with pytest.raises(ValueError,match='four'):donor_pool(rows[:-1],'m',{})
    with pytest.raises(ValueError,match='Duplicate'):donor_pool(rows+[rows[0]],'m',{})
    rows[0]['terminated']=False
    pool,_=donor_pool(rows,'m',{})
    assert script('complete_response_audit').donor_measurement_status(pool['x'])=='incomplete_direct_donor_pool'
    assert pool['x'][0]['tokens']==[0,3] and pool['x'][0]['terminated'] is False


def test_method_cannot_shrink_denominator():
    rows=[]
    for sid in ('a','b'):
        row={'sample':{'id':sid,'split':'eval','dataset':'food101','cluster':'c'},'model':'m',
             'method':'direct','guided':True,'marker':'UNKNOWN','reference_marker':'UNKNOWN',
             'reference_guided':True,'kind':'main','text':'UNKNOWN','label':'abstain','correct':0.}
        rows.append(row)
        if sid=='a':rows.append({**row,'method':'vcd'})
    with pytest.raises(ValueError,match='entire'):evaluate(rows,20)


def test_missing_and_duplicate_evidence_are_errors():
    row={'model':'m','sample':{'id':'x','dataset':'food101'}}
    with pytest.raises(ValueError,match='probe'):behavioral_support(row,{}, {})
    with pytest.raises(ValueError,match='rank'):behavioral_support(row,{('m','x'):{'mean_correctness':0}}, {})
    p={'model':'m','sample_id':'x','mean_correctness':0}
    with pytest.raises(ValueError,match='Duplicate'):evidence_indices([p,p],[])


def test_partial_vqa_credit_is_not_zero_success():
    row={'model':'m','sample':{'id':'x','dataset':'vizwiz','annotated_answerable':1}}
    assert not behavioral_support(row,{('m','x'):{'mean_correctness':.03}}, {})
    assert behavioral_support(row,{('m','x'):{'mean_correctness':0}}, {})


def test_annotation_revision_overrides_fast_phrase():
    assert label_response('UNKNOWN',{'k':{'text':'UNKNOWN','label':'invalid'}},'k')=='invalid'


def test_scoring_requires_endorsed_answer_span(tmp_path):
    path=tmp_path/'rows.jsonl';row={'key':'x','text':'Perhaps milk','sample':{'dataset':'food101','class':'milk'}}
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='answer span'):
        annotated_rows([path],{'x':{'text':row['text'],'label':'answer_uncertain'}},{'milk':['milk']})


def test_replicate_identity_is_exact():
    rows=[{'kind':'independent_attempt','model':'m','sample':{'id':'x'},'replicate':i,
           'correct':0,'label':'abstain','text':'UNKNOWN'} for i in range(1,11)]
    with pytest.raises(ValueError,match='Incomplete'):probe_summary(rows)


@pytest.mark.parametrize('method',['dola','deco','sid'])
def test_mechanism_records_actual_reference(method):
    from kdm.mechanism import measure_path
    b=MockBackend();result=measure_path(b,None,'q',[0,3],method)
    assert len(result['steps'])==2
    for row in result['steps']:
        assert row['prefix']==[0,3][:row['step']]
        assert len(row['observed_logp'])==4
        if method in {'dola','deco'}:
            assert 'selected_guided_layer' in row
            assert row['base_logp']+row['reference_weight']*row['observed_logp']['guided_reference']-row['log_normalizer']==pytest.approx(row['modified_logp'])
    group=result['steps'][0]['initial_token_group']
    if group['status']=='defined':assert abs(group['closure_error'])<1e-11


def test_report_gate_requires_whole_frozen_task_universe():
    from kdm.reports import validate_report_coverage
    from kdm.pipeline import probe_tasks
    samples=[{'id':sid,'dataset':'fixture','split':'eval'} for sid in ('a','b')]
    selected=[{'model':'m','dataset':'fixture','selected':True}]
    tasks=list(experiment_tasks(samples))+list(probe_tasks(samples))
    rows=[{**task,'model':'m','key':task_id('m',task),'status':'ok'} for task in tasks]
    assert validate_report_coverage(rows,samples,selected,[],baseline_plan(selected))['selected_conditions']==1
    with pytest.raises(ValueError,match='Missing frozen'):
        validate_report_coverage([r for r in rows if r['sample']['id']=='a'],samples,selected,[],baseline_plan(selected))
    with pytest.raises(ValueError,match='Missing frozen'):
        validate_report_coverage([r for r in rows if r['method']!='deco'],samples,selected,[],baseline_plan(selected))


def test_vqa_scoring_cannot_use_generic_normalization():
    from kdm.scoring import vqa_score
    with pytest.raises(ValueError,match='Official'):vqa_score('yes',['yes']*10)



def baseline_plan(selection):
    plan={}
    for row in selection:plan.setdefault(row['model'],{})[row['dataset']]=['vcd','m3id','dola','deco']
    return plan


def report_records(samples,model='m',methods=('vcd','m3id','dola','deco')):
    from kdm.pipeline import probe_tasks
    tasks=list(experiment_tasks(samples,methods))+list(probe_tasks(samples))
    return [{**task,'model':model,'key':task_id(model,task),'status':'ok'} for task in tasks]


def test_report_gate_rejects_extra_unselected_and_forged_model():
    from kdm.reports import validate_report_coverage
    samples=[{'id':'x','dataset':'fixture','split':'eval'}]
    selection=[{'model':'m','dataset':'fixture','selected':True}]
    rows=report_records(samples)
    for extra in (report_records(samples,'unselected')[0],{**rows[0],'key':'unexpected'}):
        with pytest.raises(ValueError,match='Unexpected or unselected'):
            validate_report_coverage(rows+[extra],samples,selection,[],baseline_plan(selection))
    rows[0]['model']='forged'
    with pytest.raises(ValueError,match='disagrees'):validate_report_coverage(rows,samples,selection,[],baseline_plan(selection))


def test_report_gate_compares_all_101_frozen_class_names():
    from kdm.reports import validate_report_coverage
    samples=[{'id':str(i),'dataset':'food101','class':'class_'+str(i),
              'split':'eval' if i==0 else 'dev'} for i in range(101)]
    selection=[{'model':'m','dataset':'food101','selected':True}]
    rows=report_records(samples)
    scores=[{'label':sample['class'],'sum_logp':-1.,'mean_logp':-1.,'n_tokens':1} for sample in samples]
    closed=[{'model':'m','sample':samples[0],'candidate_scores':scores,'gold_rank':1,
             'ranking_rule':'mean_log_probability','target':'class_0'}]
    validate_report_coverage(rows,samples,selection,closed,baseline_plan(selection))
    closed[0]['candidate_scores'][-1]['label']='forged_class'
    with pytest.raises(ValueError,match='frozen 101'):validate_report_coverage(rows,samples,selection,closed,baseline_plan(selection))


def test_report_method_plan_is_model_specific_and_fixed():
    from kdm.reports import validate_report_coverage
    samples=[{'id':'x','dataset':'fixture','split':'eval'}]
    selection=[{'model':model,'dataset':'fixture','selected':True} for model in ('m','n')]
    baselines=['vcd','m3id','dola','deco'];plan={'m':{'fixture':baselines+['sid']},'n':{'fixture':baselines}}
    rows=report_records(samples,'m',plan['m']['fixture'])+report_records(samples,'n',plan['n']['fixture'])
    out=validate_report_coverage(rows,samples,selection,[],method_plan=plan)
    assert 'sid' in out['method_plan']['m']['fixture'] and 'sid' not in out['method_plan']['n']['fixture']
    with pytest.raises(ValueError,match='Unexpected or unselected'):
        validate_report_coverage(rows,samples,selection,[],baseline_plan(selection))
    with pytest.raises(ValueError,match='all four'):
        validate_report_coverage(rows,samples,selection,[],method_plan={'m':{'fixture':['vcd','sid']}})
    with pytest.raises(ValueError,match='exactly cover'):
        validate_report_coverage(rows,samples,selection,[],method_plan={'other':{'fixture':baselines}})


def test_report_rejects_extra_closed_record():
    from kdm.reports import validate_report_coverage
    samples=[{'id':'x','dataset':'fixture','split':'eval'}]
    selection=[{'model':'m','dataset':'fixture','selected':True}]
    with pytest.raises(ValueError,match='Closed-set records do not match'):
        validate_report_coverage(report_records(samples),samples,selection,[{'model':'extra','sample':samples[0]}],baseline_plan(selection))



@pytest.mark.parametrize('remote',[False,True])
def test_incomplete_donor_audit_records_undefined_without_model_generation(tmp_path,monkeypatch,remote):
    mod=script('complete_response_audit')
    image=tmp_path/'image.png';Image.new('RGB',(8,8)).save(image)
    logical='/unmounted-original-host/image.png' if remote else str(image)
    if remote:assert not Path(logical).exists()
    mapped=[]
    def resolve(path,root):
        assert path==logical and root==tmp_path
        mapped.append(path);return image
    monkeypatch.setattr(mod,'resolve_image_path',resolve)
    sample={'id':'x','split':'eval','dataset':'fixture','question':'q','image_path':logical}
    tasks=[t for t in experiment_tasks([sample]) if t['method']=='direct']
    rows=[{**t,'model':'m','key':task_id('m',t),'status':'ok','text':'UNKNOWN',
           'tokens':[0,3],'terminated':True,'seed':0} for t in tasks]
    rows[0]['tokens']=[0]*32;rows[0]['terminated']=False
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(json.dumps(sample)+'\n')
    records=tmp_path/'records.jsonl';records.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    annotations=tmp_path/'annotations.jsonl';annotations.write_text(''.join(json.dumps({
        'key':r['key'],'text':r['text'],'label':'abstain','evidence':r['text'],'answer_text':''})+'\n' for r in rows))
    spec=tmp_path/'spec.json';spec.write_text(json.dumps({'purpose':'CPU_TEST_ONLY'}))
    out=tmp_path/'outputs/verification/audit.jsonl'
    monkeypatch.setattr(mod,'make_backend',lambda *args:pytest.fail('Undefined pool must not load a backend'))
    monkeypatch.setattr('sys.argv',['audit','--root',str(tmp_path),'--manifest',str(manifest),'--records',str(records),
        '--annotations',str(annotations),'--model-spec',str(spec),'--model','m','--gpu','0','--methods','vcd','--out',str(out)])
    mod.main()
    result=[json.loads(line) for line in out.read_text().splitlines()]
    assert len(result)==16
    assert mapped==[logical] and json.loads(manifest.read_text())['image_path']==logical
    assert all(r['measurement_status']=='incomplete_direct_donor_pool' and r['finite_response_identity'] is None for r in result)
    assert all(len(r['donor_pool'])==4 and r['donor_pool'][0]['tokens']==[0]*32 and not r['donor_pool'][0]['terminated'] for r in result)


def test_complete_audit_execution_admission_before_backend(tmp_path,monkeypatch):
    mod=script('complete_response_audit');import kdm.protocol as protocol
    called=[]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(protocol,'validate_runtime',lambda *args:called.append('runtime'))
    monkeypatch.setattr(protocol,'code_identity',lambda root:called.append('source') or ['blob'])
    assert mod.execution_identity(tmp_path,{},'m',tmp_path/'outputs/formal/audit.jsonl','0')==['blob']
    assert called==['runtime','source']
    with pytest.raises(ValueError,match='verification'):
        mod.execution_identity(tmp_path,{'purpose':'CPU_TEST_ONLY'},'m',tmp_path/'outputs/formal/audit.jsonl','0')


def test_complete_audit_rejects_missing_or_changed_input_identity(tmp_path):
    mod=script('complete_response_audit')
    path=tmp_path/'records.jsonl';path.write_text(json.dumps({'key':'x','identity':'changed'})+'\n')
    with pytest.raises(ValueError,match='sidecar'):mod.validate_input_ledgers([path],'spec',True)
    from kdm.io import stable_hash
    definition={'backend_spec_sha256':'spec'}
    path.with_suffix('.identity.json').write_text(json.dumps({'definition':definition,'identity':stable_hash(definition)}))
    with pytest.raises(ValueError,match='record identity'):mod.validate_input_ledgers([path],'spec',True)
    with pytest.raises(ValueError,match='backend spec'):mod.validate_input_ledgers([path],'other',True)


def test_invalid_donor_is_preserved_and_truncation_is_explicit():
    mod=script('complete_response_audit')
    rows=[{'model':'m','sample':{'id':'x'},'method':'direct','guided':True,'kind':'main','status':'ok',
           'marker':marker,'text':'','key':marker,'tokens':[3],'terminated':True} for marker in MARKERS]
    pools,_=mod.donor_pool(rows,'m',{})
    assert len(pools['x'])==4 and all(row['label']=='invalid' for row in pools['x'])
    assert mod.donor_measurement_status(pools['x'])=='no_concrete_donor'
    pools['x'][0]['terminated']=False
    assert mod.donor_measurement_status(pools['x'])=='incomplete_direct_donor_pool'



def test_annotation_queue_rejects_duplicate_and_changed_identity(tmp_path):
    from kdm.annotation import build_queue
    source=tmp_path/'source.jsonl';row={'key':'x','text':'UNKNOWN','sample':{'question':'q'}}
    source.write_text(json.dumps(row)+'\n');out=tmp_path/'queue.jsonl'
    assert build_queue([source],out)==1 and build_queue([source],out)==1
    with pytest.raises(ValueError,match='Duplicate'):build_queue([source,source],tmp_path/'dupe.jsonl')
    row['text']='UNCLEAR';source.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='differs'):build_queue([source],out)
    assert json.loads(out.read_text())['text']=='UNKNOWN'


def test_annotation_cannot_endorse_answer_and_abstain(tmp_path):
    from kdm.annotation import validate_annotations
    row={'key':'x','text':'UNKNOWN','label':'abstain','evidence':'UNKNOWN','answer_text':'UNKNOWN'}
    path=tmp_path/'ann.jsonl';path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='cannot endorse'):validate_annotations(path)
    mod=script('annotate_responses')
    with pytest.raises(ValueError,match='cannot endorse'):
        mod.parse_label(json.dumps({'label':'abstain','evidence_span':'UNKNOWN','answer_text':'UNKNOWN'}),'UNKNOWN')
    with pytest.raises(ValueError,match='JSON object'):mod.parse_label('[]','UNKNOWN')


def test_semantic_judge_is_blinded_and_keeps_raw_reply(tmp_path,monkeypatch):
    mod=script('annotate_responses')
    queue=tmp_path/'queue.jsonl';queue.write_text(json.dumps({'key':'private-model-method-key','text':'Maybe milk',
        'question':'What is shown?','label':None,'source':'requires_semantic_review'})+'\n')
    seen={}
    class Response:
        status_code=200
        def raise_for_status(self):pass
        def json(self):return {'model':'independent-judge','id':'reply-id','kdm_judge_receipt_sha256':'receipt-sha','choices':[{'finish_reason':'stop','message':{'content':json.dumps({
            'label':'answer_uncertain','evidence_span':'Maybe','answer_text':'milk'})}}]}
    class Session:
        trust_env=True
        def post(self,url,**kwargs):
            seen.update(kwargs);seen['trust_env']=self.trust_env;return Response()
    monkeypatch.setattr(mod.requests,'Session',Session)
    out=tmp_path/'out.jsonl'
    judge_spec=tmp_path/'judge.json';judge_spec.write_text('{}')
    monkeypatch.setattr(mod,'validate_judge_identity',lambda *args:{'receipt':{},'receipt_sha256':'receipt-sha'})
    monkeypatch.setattr('sys.argv',['judge','--root',str(tmp_path),'--queue',str(queue),'--endpoint','http://localhost:8000/v1',
        '--judge-model','independent-judge','--judge-spec',str(judge_spec),'--out',str(out)])
    mod.main()
    request=json.loads(seen['json']['messages'][1]['content'])
    assert set(request)=={'task','question','answer','instructions'}
    assert 'private-model-method-key' not in json.dumps(seen['json'])
    assert seen['trust_env'] is False and seen['allow_redirects'] is False
    row=json.loads(out.read_text());assert row['judge_response_id']=='reply-id' and row['judge_raw_content']
