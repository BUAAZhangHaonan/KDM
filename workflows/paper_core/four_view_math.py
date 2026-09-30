"""Exact same-prefix score attribution for VCD and instruction-preserving VCD.

All inputs are vectors from the SAME generated prefix. The caller is responsible
for image/prompt identity and semantic labels. Natural logarithms are used.
"""
from __future__ import annotations
import numpy as np


def log_probs(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=1 or not np.isfinite(x).all():
        raise ValueError('Provide a finite one-dimensional full-vocabulary logit vector.')
    t=x-x.max()
    return t-np.log(np.exp(t).sum())


def analyze(g, h, c, r, token_a: int, token_b: int, alpha: float=1., beta: float=.1) -> dict:
    """Compare candidates a and b; positive margins favor candidate a.

    g/h: guided clean/reference; c/r: unguided clean/reference.
    Candidate a is an observed response-branch token, with its semantic role
    supplied separately by the calling measurement procedure.
    """
    if not np.isfinite([alpha,beta]).all() or alpha<0 or not 0<=beta<=1:
        raise ValueError('Invalid contrast parameters.')
    g,h,c,r=map(log_probs,(g,h,c,r))
    if not g.shape==h.shape==c.shape==r.shape:raise ValueError('Vocabularies differ.')
    a,b=int(token_a),int(token_b)
    if a==b or min(a,b)<0 or max(a,b)>=len(g):raise ValueError('Invalid candidate pair.')
    dg,dr=g-c,h-r
    native=(1+alpha)*c-alpha*r
    guided=(1+alpha)*g-alpha*h
    ip=g+alpha*(c-r)
    interaction=alpha*(dg-dr)
    sg=np.ones(len(g),bool) if beta==0 else g>=g.max()+np.log(beta)
    sc=np.ones(len(c),bool) if beta==0 else c>=c.max()+np.log(beta)
    margin=lambda x:float(x[a]-x[b])
    masked=lambda x,k:np.where(k,x,-np.inf)
    output={
        'candidate_a':a,'candidate_b':b,'alpha':alpha,'beta':beta,
        'logp_a':{name:float(x[a]) for name,x in zip(['g','h','c','r'],[g,h,c,r])},
        'logp_b':{name:float(x[b]) for name,x in zip(['g','h','c','r'],[g,h,c,r])},
        'base_pair_margins':{name:margin(x) for name,x in zip(['g','h','c','r'],[g,h,c,r])},
        'clean_guide_margin_change':margin(dg),'reference_guide_margin_change':margin(dr),
        'unguided_visual_margin_change':alpha*margin(c-r),
        'native_pair_margin':margin(native),'guided_pair_margin':margin(guided),
        'ip_pair_margin':margin(ip),'interaction_pair_margin':margin(interaction),
        'ip_minus_guided_margin':margin(ip)-margin(guided),
        'pair_in_guided_support':bool(sg[a] and sg[b]),'pair_in_native_support':bool(sc[a] and sc[b]),
        'guided_support_a':bool(sg[a]),'guided_support_b':bool(sg[b]),
        'native_support_a':bool(sc[a]),'native_support_b':bool(sc[b]),
        'guided_support_size':int(sg.sum()),'native_support_size':int(sc.sum()),
        'argmax_guided':int(np.argmax(masked(guided,sg))),
        'argmax_ip':int(np.argmax(masked(ip,sg))),
        'argmax_native':int(np.argmax(masked(native,sc))),
        'argmax_native_on_guided_support':int(np.argmax(masked(native,sg))),
        'decomposition_closure':margin(guided)-margin(native)-margin(dg)-margin(interaction),
        'ip_difference_closure':margin(ip)-margin(guided)+margin(interaction),
        'interpolation':[],
    }
    # Score-level intervention; the two endpoints exactly reproduce the methods
    # at the declared prefix on the SAME guided plausibility support.
    for weight in [0.,.5,1.]:
        score=ip+weight*interaction
        v=score[sg];normalizer=float(v.max()+np.log(np.exp(v-v.max()).sum()))
        output['interpolation'].append({'interaction_weight':weight,'pair_margin':margin(score),
                                       'argmax':int(np.argmax(masked(score,sg))),
                                       'log_normalizer':normalizer})
    return output
