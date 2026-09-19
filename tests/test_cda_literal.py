import math
from pathlib import Path
import numpy as np
import pytest
from kdm.cda import cda_calibration,cda_weights,generate_cda,CDAUndefinedError
from kdm.decoding import DecodeConfig,Step


def shannon(probabilities):
    return -sum(p*math.log(p) for p in probabilities if p)


def test_equations_six_seven_keep_signed_residual():
    p=[.5,.5];c=[.6,.4];np_=[.99,.01];nc=[.98,.02]
    hp,hc,hnp,hnc=map(shannon,(p,c,np_,nc))
    rp=(hp-hnp)/hnp;rc=(hc-hnc)/hnc
    wp=rp*rp/(rp+rc);wc=rc*rc/(rp+rc);wa=1-wp-wc
    actual=cda_calibration(*map(np.log,(p,c,np_,nc)))
    assert np.allclose([actual[k] for k in ('rp','rc','wp','wc','wa')],[rp,rc,wp,wc,wa])
    assert actual['negative_wa'] and actual['wa']<0 and not actual['zero_sum_extension']
    assert np.allclose(cda_weights(*map(np.log,(p,c,np_,nc))),[wp,wc,wa])
    assert abs(wp+wc+wa-1)<1e-12
    reverse=cda_calibration(*map(np.log,(np_,nc,p,c)))
    assert [reverse[k] for k in ('rp','rc','wp','wc','wa')]==[0,0,0,0,1]
    assert reverse['zero_sum_extension']


class Session:
    def __init__(self,prob):self.logits=np.log(prob);self.calls=[]
    def next(self,prefix):self.calls.append(prefix);return Step(self.logits)


def test_five_sessions_share_prefix_and_each_step_records_actual_negative_weight():
    probabilities=([.5,.5],[.6,.4],[.8,.2],[.99,.01],[.98,.02])
    sessions=[Session(p) for p in probabilities]
    cfg=DecodeConfig(max_tokens=2)
    out=generate_cda(*sessions,cfg,set(),lambda tokens:str(tokens))
    weights=cda_weights(*(np.log(probabilities[i]) for i in (0,1,3,4)))
    scores=sum(w*np.log(p) for w,p in zip(weights,probabilities[:3]))
    logp=scores-np.logaddexp.reduce(scores)
    assert out['negative_wa_steps']==2 and out['zero_sum_extension_steps']==0
    assert not out['terminated'] and len(out['tokens'])==2
    assert all(session.calls==[(),tuple(out['tokens'][:1])] for session in sessions)
    for row in out['trace']:
        assert row['negative_wa'] and row['wa']<0
        assert np.isclose(row['log_probability'],logp[row['token']])
        assert np.allclose(row['weights'],[row['wp'],row['wc'],row['wa']])


def test_zero_null_entropy_is_undefined_and_keeps_prefix_diagnostics():
    logits=np.array([0.,-np.inf])
    with pytest.raises(CDAUndefinedError,match='nonpositive_null_entropy') as err:
        cda_calibration([0.,0.],[0.,0.],logits,[0.,0.])
    assert err.value.cda_diagnostics['h_null_prior']==0
    class Null:
        def next(self,prefix):return Step(logits)
    with pytest.raises(CDAUndefinedError) as err:
        generate_cda(Session([.5,.5]),Session([.5,.5]),Session([.5,.5]),Null(),Session([.5,.5]),DecodeConfig(),set(),str)
    assert err.value.cda_diagnostics['prefix_tokens']==[] and err.value.cda_diagnostics['completed_trace']==[]


def test_undefined_cda_calibration_is_preserved_in_task_error_log(tmp_path,monkeypatch):
    import json
    import kdm.cda as cda
    from PIL import Image
    from kdm.pipeline import run_tasks,experiment_tasks
    from kdm.models.mock import MockBackend
    image=tmp_path/'image.png';Image.new('RGB',(8,8)).save(image)
    sample={'id':'synthetic','split':'eval','question':'q','image_path':str(image)}
    tasks=[t for t in experiment_tasks([sample]) if t['method']=='cda_visual']
    assert tasks
    def undefined(*args,**kwargs):
        raise CDAUndefinedError('nonpositive_null_entropy',{'h_null_prior':0.,'prefix_tokens':[1],'completed_trace':[]})
    monkeypatch.setattr(cda,'generate_cda',undefined);out=tmp_path/'generation.jsonl'
    with pytest.raises(CDAUndefinedError):run_tasks(MockBackend(),'fixture',tasks,out,{},DecodeConfig())
    error=json.loads(Path(str(out)+'.errors.jsonl').read_text())
    assert error['cda_diagnostics']['status']=='undefined'
    assert error['cda_diagnostics']['prefix_tokens']==[1]
    assert not out.exists()
