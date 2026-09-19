import numpy as np
import pytest
from kdm.models.mock import MockBackend,MockSession
from kdm.decoding import *
from kdm.cda import cda_weights,generate_cda
from kdm.probability import log_normalize

@pytest.mark.parametrize('method',['direct','vcd','m3id','dola','deco','sid','icd','instruction_vcd','instruction_m3id'])
def test_all_decoders(method):
    b=MockBackend();m=b.session(None,'x');r=b.session(None,'x','noise');neutral=b.session(None,'x')
    cfg=DecodeConfig(method=method,m3id_threshold=.9,m3id_offset=10)
    out=generate(m,r,cfg,b.eos,b.decode,17,neutral)
    assert out['status']=='ok' and out['terminated']
    assert len(out['tokens'])==len(out['trace'])
    assert all(0<=np.exp(x)<=1 for x in out['selected_log_probabilities'])

def test_m3id_gate():
    step=Step(np.log([.9,.1]));r=Step(np.log([.1,.9]))
    cfg=DecodeConfig(method='m3id',m3id_offset=10)
    out,meta=distribution(step,r,cfg,0)
    assert not meta['active'];assert np.allclose(out,step.logits)

def test_cda_weights():
    p=np.array([1.,2.,0.]);c=np.array([2.,0.,1.]);null=np.zeros(3)
    w=cda_weights(p,c,null,null);assert np.allclose(w,[0,0,1])  # Lower input entropy gives zero r under the literal equation (6).
    assert np.allclose(cda_weights(null,null,null,null),[0,0,1])

def test_cda_generation():
    b=MockBackend();sessions=[b.session(None,'x') for _ in range(5)]
    out=generate_cda(*sessions,DecodeConfig(),b.eos,b.decode);assert out['terminated']

def test_sampling_reproducibility_and_logp():
    lp=np.log([.6,.3,.1]);cfg=DecodeConfig(temperature=.7,top_p=.8)
    x=[draw_token(lp,cfg,np.random.default_rng(i)) for i in range(20)]
    y=[draw_token(lp,cfg,np.random.default_rng(i)) for i in range(20)]
    assert x==y;assert np.isclose(np.exp(sampling_distribution(lp,cfg)).sum(),1)

def test_replay_same_prefix():
    b=MockBackend();m=b.session(None,'x');r=b.session(None,'x','noise');cfg=DecodeConfig(method='vcd')
    rows=replay(m,r,cfg,[2,3],{'group':[0,1]})
    assert m.calls==r.calls==[(),(2,)] and 'log_normalizer' in rows[0]

@pytest.mark.parametrize('method',['vcd','dola','deco'])
def test_numpy_against_torch_local_operator(method):
    import torch
    rng=np.random.default_rng(34);z=rng.normal(size=17);q=rng.normal(size=17)
    raw={i:rng.normal(size=17) for i in range(4)};norm={i:rng.normal(size=17) for i in range(2,5)}
    step=Step(z,raw,norm);cfg=DecodeConfig(method=method,deco_topk=7)
    out,meta=distribution(step,Step(q),cfg,0)
    tz=torch.tensor(z);tq=torch.tensor(q)
    if method=='vcd':
        expected=2*tz-tq;expected[tz<tz.max()+np.log(.1)]=-torch.inf
    elif method=='dola':
        expected=tz-torch.log_softmax(torch.tensor(raw[meta['layer']]),-1)
        expected[tz<tz.max()+np.log(.1)]=-torch.inf
    else:
        expected=tz+meta['weight']*torch.tensor(norm[meta['layer']])
        prob=torch.softmax(tz,-1);v,ids=torch.topk(prob,7);cut=min(7,int(torch.searchsorted(v.cumsum(0),torch.tensor(.9)))+1)
        keep=torch.zeros(17,dtype=torch.bool);keep[ids[:cut]]=True;expected[~keep]=-torch.inf
    assert np.allclose(out,torch.log_softmax(expected,-1).numpy())

@pytest.mark.parametrize('method',['dola','deco'])
def test_layer_replay_uses_actual_reference_and_support(method):
    from kdm.models.mock import MockBackend
    from PIL import Image
    from kdm.decoding import DecodeConfig,replay
    b=MockBackend();session=b.session(Image.new('RGB',(8,8)),'question',need_layers=True)
    result=replay(session,None,DecodeConfig(method=method),[0,3],{'event':[0,1]})
    assert all('reference_weight' in row and 'log_normalizer' in row for row in result)
    for row in result:
        if row['event']['status']=='defined':assert abs(row['event']['closure_error'])<1e-11
