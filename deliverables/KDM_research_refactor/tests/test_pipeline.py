import json,os
from dataclasses import replace
from pathlib import Path
import numpy as np,pytest
from PIL import Image
from kdm.models.mock import MockBackend
from kdm.pipeline import census_tasks,experiment_tasks,probe_tasks,run_tasks,select_models,closed_rank,finite_response_audit,json_safe
from kdm.decoding import DecodeConfig
from kdm.analysis import evaluate,probe_summary,copy_abstention_comparator
from kdm.io import read_jsonl,atomic_json
from kdm.mechanism import measure_path


def samples(tmp_path):
    img=tmp_path/'image.png';Image.new('RGB',(8,8)).save(img)
    return [{'id':str(i),'cluster':str(i),'dataset':'fixture','split':'eval','question':'q','gold':['milk'],'image_path':str(img)} for i in range(3)]


def test_end_to_end_mock_and_resume(tmp_path):
    s=samples(tmp_path);b=MockBackend();out=tmp_path/'r.jsonl'
    n=run_tasks(b,'mock',census_tasks(s),out,{'mode':'census'},DecodeConfig(max_tokens=3));assert n==6
    assert run_tasks(b,'mock',census_tasks(s),out,{'mode':'census'},DecodeConfig(max_tokens=3))==6
    sel=select_models([out],{},[r['id'] for r in s]);assert sel[0]['n_abstain']==3
    e=tmp_path/'exp.jsonl';n=run_tasks(b,'mock',experiment_tasks(s,markers=('UNKNOWN',)),e,{},DecodeConfig(max_tokens=3));assert n==30
    rows=list(read_jsonl(e));assert {'instruction_vcd','instruction_m3id','cda_visual'}<={r['method'] for r in rows}


def test_shards_complete(tmp_path):
    s=samples(tmp_path);keys=[]
    for shard in range(2):
        out=tmp_path/f'{shard}.jsonl';run_tasks(MockBackend(),'mock',census_tasks(s),out,{},DecodeConfig(max_tokens=2),shard,2)
        if out.exists():keys.extend(r['key'] for r in read_jsonl(out))
    assert len(keys)==len(set(keys))==6


def test_shared_prefix_measurement():
    b=MockBackend();image=Image.new('RGB',(8,8));result=measure_path(b,image,'q',[0,3])
    assert result['first_divergence']==0
    assert all(abs(r['pair_identity_error'])<1e-12 for r in result['steps'] if 'pair_identity_error' in r)
    json.dumps(json_safe(result),allow_nan=False)


def test_closed_and_finite_response():
    b=MockBackend();image=Image.new('RGB',(8,8))
    r=closed_rank(b,image,'q',['UNKNOWN','milk'],'UNKNOWN');assert r['gold_rank']==1
    result=finite_response_audit(b,image,'q','UNKNOWN','UNCLEAR',['milk'],DecodeConfig())
    assert result['measurement_scope']=='conditional_on_listed_complete_token_sequences'
    json.dumps(json_safe(result),allow_nan=False)


def test_behavior_statistics_and_invalid(tmp_path):
    samples_=samples(tmp_path);rows=[]
    for i,s in enumerate(samples_):
        common={'model':'m','sample':s,'marker':'UNKNOWN','reference_marker':'UNKNOWN','reference_guided':True,'kind':'main','guided':True}
        rows.extend([{**common,'method':'direct','text':'UNKNOWN','label':'abstain','correct':0.},
                     {**common,'method':'vcd','text':'milk' if i<2 else '', 'label':'answer_assertive' if i<2 else 'invalid','correct':float(i==0)}])
    r=evaluate(rows,bootstrap=30)[0]
    assert r['n_lost']==2 and r['abstention_to_invalid']==1
    assert r['answer_rate']==pytest.approx(2/3)
    assert r['answered_accuracy']==.5


def test_probe_unanswerable_not_unknown_knowledge():
    rows=[{'kind':'independent_attempt','model':'m','sample':{'id':'x','annotated_answerable':0},'replicate':i,'correct':0.,'label':'answer_assertive','text':'milk'} for i in range(10)]
    assert probe_summary(rows)[0]['mean_correctness'] is None
    with pytest.raises(ValueError):probe_summary(rows[:-1])


def test_copy_baseline():
    d={'a':{'label':'abstain','text':'UNKNOWN'},'b':{'label':'answer_assertive','text':'x'}}
    m={'a':{'label':'answer_assertive','text':'y'},'b':{'label':'answer_assertive','text':'z'}}
    assert copy_abstention_comparator(d,m)['a']['text']=='UNKNOWN'
    assert copy_abstention_comparator(d,m)['b']['text']=='z'

def test_selection_is_model_and_dataset_specific(tmp_path):
    rows=[]
    for i,dataset in enumerate(['food101','vizwiz']):
        s={'id':str(i),'dataset':dataset}
        rows.append({'key':str(i),'model':'m','kind':'census','marker':'UNKNOWN','sample':s,'text':'UNKNOWN' if i==0 else 'milk'})
    f=tmp_path/'census.jsonl';f.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    ann={'1':{'text':'milk','label':'answer_assertive'}}
    result=select_models([f],ann,['0','1'])
    assert [(r['dataset'],r['selected']) for r in result]==[('food101',True),('vizwiz',False)]
