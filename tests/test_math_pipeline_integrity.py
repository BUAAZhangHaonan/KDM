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
           'marker':marker,'text':'UNKNOWN','key':marker,'tokens':[0,3],'terminated':True} for marker in MARKERS]
    assert len(donor_pool(rows,'m',{})[0]['x'])==4
    with pytest.raises(ValueError,match='four'):donor_pool(rows[:-1],'m',{})
    with pytest.raises(ValueError,match='Duplicate'):donor_pool(rows+[rows[0]],'m',{})
    rows[0]['terminated']=False
    with pytest.raises(ValueError,match='Truncated'):donor_pool(rows,'m',{})


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
    assert validate_report_coverage(rows,samples,selected,[])['selected_conditions']==1
    with pytest.raises(ValueError,match='Missing frozen'):
        validate_report_coverage([r for r in rows if r['sample']['id']=='a'],samples,selected,[])
    with pytest.raises(ValueError,match='Missing frozen'):
        validate_report_coverage([r for r in rows if r['method']!='deco'],samples,selected,[])


def test_vqa_scoring_cannot_use_generic_normalization():
    from kdm.scoring import vqa_score
    with pytest.raises(ValueError,match='Official'):vqa_score('yes',['yes']*10)



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
            validate_report_coverage(rows+[extra],samples,selection,[])
    rows[0]['model']='forged'
    with pytest.raises(ValueError,match='disagrees'):validate_report_coverage(rows,samples,selection,[])


def test_report_gate_compares_all_101_frozen_class_names():
    from kdm.reports import validate_report_coverage
    samples=[{'id':str(i),'dataset':'food101','class':'class_'+str(i),
              'split':'eval' if i==0 else 'dev'} for i in range(101)]
    selection=[{'model':'m','dataset':'food101','selected':True}]
    rows=report_records(samples)
    scores=[{'label':sample['class'],'sum_logp':-1.,'mean_logp':-1.,'n_tokens':1} for sample in samples]
    closed=[{'model':'m','sample':samples[0],'candidate_scores':scores,'gold_rank':1,
             'ranking_rule':'mean_log_probability','target':'class_0'}]
    validate_report_coverage(rows,samples,selection,closed)
    closed[0]['candidate_scores'][-1]['label']='forged_class'
    with pytest.raises(ValueError,match='frozen 101'):validate_report_coverage(rows,samples,selection,closed)


def test_report_method_plan_is_model_specific_and_fixed():
    from kdm.reports import validate_report_coverage
    samples=[{'id':'x','dataset':'fixture','split':'eval'}]
    selection=[{'model':model,'dataset':'fixture','selected':True} for model in ('m','n')]
    baselines=['vcd','m3id','dola','deco'];plan={'m':baselines+['sid'],'n':baselines}
    rows=report_records(samples,'m',plan['m'])+report_records(samples,'n',plan['n'])
    out=validate_report_coverage(rows,samples,selection,[],method_plan=plan)
    assert 'sid' in out['method_plan']['m'] and 'sid' not in out['method_plan']['n']
    with pytest.raises(ValueError,match='Unexpected or unselected'):
        validate_report_coverage(rows,samples,selection,[])
    with pytest.raises(ValueError,match='all four'):
        validate_report_coverage(rows,samples,selection,[],method_plan={'m':['vcd','sid']})
    with pytest.raises(ValueError,match='unselected model'):
        validate_report_coverage(rows,samples,selection,[],method_plan={'other':baselines})


def test_report_rejects_extra_closed_record():
    from kdm.reports import validate_report_coverage
    samples=[{'id':'x','dataset':'fixture','split':'eval'}]
    selection=[{'model':'m','dataset':'fixture','selected':True}]
    with pytest.raises(ValueError,match='Closed-set records do not match'):
        validate_report_coverage(report_records(samples),samples,selection,[{'model':'extra','sample':samples[0]}])
