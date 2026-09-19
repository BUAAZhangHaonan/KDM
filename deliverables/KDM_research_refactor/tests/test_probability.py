import numpy as np
import pytest
from kdm.probability import *

@pytest.mark.parametrize('alpha',[0.,.1,.5,1.,2.])
@pytest.mark.parametrize('beta',[0.,.1,.5])
def test_exact_group_decomposition(alpha,beta):
    rng=np.random.default_rng(90)
    for _ in range(100):
        p=rng.dirichlet(np.ones(12)*3);q=rng.dirichlet(np.ones(12)*3);e=np.arange(12)<5
        r=event_decomposition(np.log(p),np.log(q),e,alpha,beta)
        if r['status']=='defined':assert abs(r['closure_error'])<1e-10

@pytest.mark.parametrize('alpha',[.2,1.,2.])
def test_derivative(alpha):
    p=np.log([.3,.1,.2,.4]);q=np.log([.1,.4,.2,.3]);a=np.array([1,1,0,0],bool);h=1e-5
    plus=np.exp(contrast(p,q,alpha+h,0)[0])[a].sum()
    minus=np.exp(contrast(p,q,alpha-h,0)[0])[a].sum()
    assert abs((plus-minus)/(2*h)-event_derivative(p,q,a,alpha,0))<1e-8

@pytest.mark.parametrize('bad',[[0,0],[np.nan,1],[-1,2]])
def test_bad_probabilities(bad):
    with pytest.raises(ValueError):log_probs(bad)

def test_instruction_pairwise_effect():
    rng=np.random.default_rng(83)
    for _ in range(100):
        pg,qg,pn,qn=rng.normal(size=(4,20));a=1.3
        output,_=instruction_preserving(pg,pn,qn,a,0)
        original,_=contrast(pg,qg,a,0)
        correction=instruction_interaction(pg,qg,pn,qn)
        delta=(output-original)-(output-original)[0]
        assert np.max(np.abs(delta-a*(correction-correction[0])))<1e-12
        neutral,_=contrast(pn,qn,a,0)
        assert np.max(np.abs((output-neutral)-(output-neutral)[0] - ((pg-pn)-(pg-pn)[0])))<1e-12

def test_instruction_common_effect_recovers_vcd():
    pg=np.array([2.,1.,0.]);pn=np.array([1.,0.,0.]);qn=np.array([0.,1.,2.]);qg=qn+(pg-pn)
    x,_=instruction_preserving(pg,pn,qn,1,.1);y,_=contrast(pg,qg,1,.1)
    assert np.allclose(x,y)

def test_three_branch_limits():
    g=np.array([1.,2.,3.]);n=np.array([2.,4.,1.])
    x,_=instruction_preserving(g,n,n,1,0);assert np.allclose(x,log_normalize(g))
    x,_=instruction_preserving(n,n,g,1,0);y,_=contrast(n,g,1,0);assert np.allclose(x,y)

def test_zero_strength_retains_guided_distribution_on_support():
    g=log_probs([.65,.34,.01]);c=log_probs([.1,.4,.5]);r=log_probs([.4,.2,.4])
    out,keep=instruction_preserving(g,c,r,0.,.1)
    assert keep.tolist()==[True,True,False]
    assert np.allclose(np.exp(out[keep]),np.exp(g[keep])/np.exp(g[keep]).sum())

def test_complete_sequence_identity():
    p=np.log([.10,.05,.15]);m=np.log([.001,.10,.05]);e=np.array([1,1,0],bool)
    assert abs(complete_response_shift(p,m,e)['closure_error'])<1e-12

def test_same_group_mass_different_internal_allocation():
    p=np.log([.35,.05,.3,.3]);a=np.array([1,1,0,0],bool)
    q1=np.log([.55,.05,.2,.2]);q2=np.log([.05,.55,.2,.2])
    x=event_decomposition(p,q1,a,1,0);y=event_decomposition(p,q2,a,1,0)
    assert np.isclose(x['reference_event_mass'],y['reference_event_mass'])
    assert x['event_mass_after']<.4<y['event_mass_after']

def test_invalid_support():
    with pytest.raises(ValueError):contrast([0,0],[0,-np.inf],1,0)
    assert event_decomposition(np.log([.99,.01]),np.log([.5,.5]),[False,True],1,.1)['status']=='event_excluded'

@pytest.mark.parametrize('weights',[(2,-1),(1,-1),(1,.2),(1.3,-.3),(1,0),(0,1)])
def test_general_operator_family(weights):
    from kdm.probability import general_group_decomposition
    rng=np.random.default_rng(789)
    for _ in range(50):
        p,q=rng.normal(size=(2,11));event=np.arange(11)<4
        r=general_group_decomposition(p,q,event,*weights,beta=0)
        assert abs(r['closure_error'])<1e-11

@pytest.mark.parametrize('strength',[0.,.5,1.,2.])
def test_variational_identity(strength):
    from kdm.probability import tilt_objective,log_normalize
    rng=np.random.default_rng(982)
    base=log_normalize(rng.normal(size=31));advantage=rng.normal(size=31)
    optimum=log_normalize(base+strength*advantage)
    optimum_value=tilt_objective(np.exp(optimum),base,advantage,strength)
    for _ in range(50):
        policy=rng.dirichlet(np.ones(31))
        gap=optimum_value-tilt_objective(policy,base,advantage,strength)
        kl=np.dot(policy,np.log(policy)-optimum)
        assert gap>=-1e-12
        assert np.isclose(gap,kl,atol=3e-12,rtol=0)

def test_general_operator_explicit_candidate_support():
    p=np.log([.4,.3,.2,.1]);q=np.log([.1,.2,.3,.4]);e=np.array([1,0,1,0],bool)
    result=general_group_decomposition(p,q,e,1.,.6,support=[True,False,True,True])
    assert abs(result['closure_error'])<1e-12
