"""CPU algebra checks against fixed author code expressions, no model mocks.

Constructed distributions validate operators only, not experimental effects.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kdm.decoding import DecodeConfig,Step,distribution,_author_dola_divergence
from kdm.probability import log_normalize


def run():
    rng=np.random.default_rng(20261004)
    checked=0;max_error=0.
    for n in (8,127,1024):
        for _ in range(32):
            z=rng.normal(0,4,size=n)
            raw={i:rng.normal(0,4,size=n) for i in range(4)}
            final=torch.tensor(z,dtype=torch.float64)
            logits=torch.stack([torch.tensor(raw[i],dtype=torch.float64) for i in sorted(raw)])
            p=F.softmax(final,dim=-1).expand_as(logits)
            q=F.softmax(logits,dim=-1);mid=.5*(p+q)
            # Literal author DoLa F.kl_div(input_log_probability, mixture).
            d=.5*(F.kl_div(F.log_softmax(final,dim=-1).expand_as(logits),mid,reduction='none').mean(-1)
                      +F.kl_div(F.log_softmax(logits,dim=-1),mid,reduction='none').mean(-1))
            expected_layer=int(d.argmax())
            out,meta=distribution(Step(z,early_raw=raw),None,DecodeConfig(method='dola'),0)
            assert meta['layer']==expected_layer
            expected=F.log_softmax(final,-1)-F.log_softmax(logits[expected_layer],-1)
            keep=F.log_softmax(final,-1)>=F.log_softmax(final,-1).max()+np.log(.1)
            expected[~keep]=-torch.inf;expected=F.log_softmax(expected,-1).numpy()
            assert np.array_equal(np.isfinite(out),np.isfinite(expected))
            error=float(np.max(np.abs(out[np.isfinite(out)]-expected[np.isfinite(expected)])))
            max_error=max(max_error,error);assert error<1e-11
            for layer in raw:
                actual=_author_dola_divergence(log_normalize(z),log_normalize(raw[layer]))
                assert abs(actual-float(d[layer]))<1e-12
            # Official greedy SID ignores its previously computed beta support.
            r=rng.normal(0,4,size=n)
            sid,meta=distribution(Step(z),Step(r),DecodeConfig(method='sid',alpha=.5),0)
            expected_sid=F.log_softmax(1.5*final-.5*torch.tensor(r),-1).numpy()
            assert np.allclose(sid,expected_sid,atol=1e-11)
            assert meta['n_retained']==n
            # VCD still uses its registered clean support and alpha=1.
            vcd,_=distribution(Step(z),Step(r),DecodeConfig(method='vcd'),0)
            expected_vcd=2*final-torch.tensor(r)
            expected_vcd[~keep]=-torch.inf
            assert np.allclose(vcd,F.log_softmax(expected_vcd,-1).numpy(),atol=1e-11)
            checked+=1
    # A known support-sensitive SID case must follow author greedy.
    z=Step(np.array([0.,-3.]));r=Step(np.array([0.,-10.]))
    sid,_=distribution(z,r,DecodeConfig(method='sid',alpha=.5),0)
    assert int(sid.argmax())==1
    # Author sampling still masks; only the greedy branch changed.
    sid_sample,_=distribution(z,r,DecodeConfig(method='sid',alpha=.5,temperature=1.),0)
    assert np.isneginf(sid_sample[1])
    return {'passed':True,'constructed_cases':checked,'dola_max_logp_error':max_error,
            'sid_greedy_unmasked':True,'sid_sampling_mask_retained':True,
            'vcd_registered_operator_unchanged':True,'actual_model_inference':False,
            'scope':'CPU operator expressions; not answer-level impact or runtime admission'}


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
