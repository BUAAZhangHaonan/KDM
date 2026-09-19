"""CDA equations 4/6/7, with an explicitly declared visual-context mapping."""
from __future__ import annotations
import numpy as np
from .probability import log_normalize
from .decoding import draw_token


def entropy(logits):
    lp=log_normalize(logits);p=np.exp(lp);nz=p>0
    return float(-np.dot(p[nz],lp[nz]))


def cda_weights(prior,context,null_prior,null_context):
    hp,hc,hnp,hnc=map(entropy,(prior,context,null_prior,null_context))
    if hnp<=0 or hnc<=0: raise ValueError('CDA null entropy must be positive')
    rp=max(hnp-hp,0)/hnp;rc=max(hnc-hc,0)/hnc
    total=rp+rc
    # Continuous extension of r_i^2/(r_p+r_c) at the origin.
    wp,wc=(rp*rp/total,rc*rc/total) if total>0 else (0.,0.)
    return np.array([wp,wc,1-wp-wc],float)


def generate_cda(prior,context,abstention,null_prior,null_context,cfg,eos,decode,seed=0):
    tokens=[];trace=[];logps=[];rng=np.random.default_rng(seed)
    for _ in range(cfg.max_tokens):
        states=[s.next(tuple(tokens)).logits for s in (prior,context,abstention,null_prior,null_context)]
        w=cda_weights(states[0],states[1],states[3],states[4])
        logits=sum(weight*z for weight,z in zip(w,states[:3]));lp=log_normalize(logits)
        tok=draw_token(lp,cfg,rng);tokens.append(tok);logps.append(float(lp[tok]))
        trace.append({'token':tok,'weights':w.tolist(),'log_probability':float(lp[tok])})
        if tok in eos:break
    return {'status':'ok','text':decode(tokens),'tokens':tokens,'selected_log_probabilities':logps,
            'sequence_log_probability':float(sum(logps)),'first_probability':float(np.exp(logps[0])),
            'trace':trace,'terminated':tokens[-1] in eos,'implementation':'CDA_visual_context_adaptation',
            'null_convention':'same_prefix_text_placeholder_and_uniform_image'}
