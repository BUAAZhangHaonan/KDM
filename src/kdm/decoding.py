"""Shared autoregressive decoder and matched-prefix replay."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
from .probability import log_normalize, contrast, pair_margin

@dataclass
class Step:
    logits: np.ndarray
    early_raw: dict[int,np.ndarray]=field(default_factory=dict)
    early_normalized: dict[int,np.ndarray]=field(default_factory=dict)

class Session(Protocol):
    def next(self,prefix:tuple[int,...])->Step: ...

@dataclass(frozen=True)
class DecodeConfig:
    method:str="direct"
    alpha:float=1.0
    beta:float=.1
    m3id_lambda:float=.02
    m3id_threshold:float=.3
    m3id_offset:int=0
    deco_alpha:float=.6
    deco_topk:int=20
    deco_topp:float=.9
    max_tokens:int=32
    temperature:float=0.0
    top_p:float=1.0


def _js(lp,lq):
    p,q=np.exp(lp),np.exp(lq);mid=(p+q)/2
    a=p>0;b=q>0
    return float(.5*(np.sum(p[a]*(lp[a]-np.log(mid[a])))+np.sum(q[b]*(lq[b]-np.log(mid[b])))))


def distribution(main:Step,reference:Step|None,cfg:DecodeConfig,t:int):
    p=log_normalize(main.logits);meta={"weight":0.,"layer":None,"active":False}
    if cfg.method=="direct": return p,meta
    if cfg.method in {"vcd","icd","sid"}:
        if reference is None: raise ValueError("Reference session required")
        out,keep=contrast(p,reference.logits,cfg.alpha,cfg.beta)
        meta.update(weight=cfg.alpha,active=cfg.alpha>0,n_retained=int(keep.sum()))
        return out,meta
    if cfg.method=="m3id":
        if reference is None: raise ValueError("M3ID needs its text-only reference")
        active=float(np.exp(p.max()))<cfg.m3id_threshold
        weight=float(np.expm1(cfg.m3id_lambda*(t+cfg.m3id_offset))) if active else 0.
        out,_=contrast(p,reference.logits,weight,0)
        meta.update(weight=weight,active=active)
        return out,meta
    if cfg.method=="dola":
        if not main.early_raw: raise ValueError("Raw premature-layer logits required")
        # Matches the project's declared JSD selection; all candidates are recorded.
        lps={i:log_normalize(z) for i,z in main.early_raw.items()}
        layer=max(sorted(lps),key=lambda i:_js(p,lps[i]))
        keep=p>=p.max()+np.log(cfg.beta)
        score=p-lps[layer];score[~keep]=-np.inf
        meta.update(weight=1.,active=True,layer=layer)
        return log_normalize(score),meta
    if cfg.method=="deco":
        if not main.early_normalized: raise ValueError("Normalized selected-layer logits required")
        indices=np.argsort(-p,kind='stable')[:cfg.deco_topk]
        cutoff=min(len(indices),int(np.searchsorted(np.cumsum(np.exp(p[indices])),cfg.deco_topp))+1)
        candidates=indices[:cutoff]
        best=(-np.inf,None)
        for i,z in sorted(main.early_normalized.items()):
            value=float(np.exp(log_normalize(z)[candidates]).max())
            if value>best[0]: best=(value,i)
        weight=cfg.deco_alpha*best[0]
        score=np.asarray(main.logits,float)+weight*np.asarray(main.early_normalized[best[1]],float)
        keep=np.zeros(len(p),bool);keep[candidates]=True;score[~keep]=-np.inf
        meta.update(weight=weight,active=True,layer=best[1])
        return log_normalize(score),meta
    raise ValueError(f"Unknown decoder: {cfg.method}")


def step_distribution(main,reference,cfg,t,neutral=None):
    """One implementation for free generation and exact-prefix replay."""
    if cfg.method not in {'instruction_vcd','instruction_m3id'}:
        return distribution(main,reference,cfg,t)
    if neutral is None or reference is None:raise ValueError('Three sessions required')
    from .probability import instruction_preserving
    weight=cfg.alpha;active=weight>0
    if cfg.method=='instruction_m3id':
        active=bool(np.exp(log_normalize(neutral.logits).max())<cfg.m3id_threshold)
        weight=float(np.expm1(cfg.m3id_lambda*(t+cfg.m3id_offset))) if active else 0.
    out,keep=instruction_preserving(main.logits,neutral.logits,reference.logits,weight,
                                    0. if cfg.method=='instruction_m3id' else cfg.beta)
    return out,{'weight':weight,'active':active,'layer':None,'n_retained':int(keep.sum())}


def sampling_distribution(logp,cfg):
    if cfg.temperature==0:
        out=np.full_like(logp,-np.inf);out[int(np.argmax(logp))]=0.;return out
    if cfg.temperature<0 or not 0<cfg.top_p<=1: raise ValueError("Invalid sampling settings")
    lp=log_normalize(logp/cfg.temperature)
    ids=np.argsort(-lp,kind='stable');cum=np.cumsum(np.exp(lp[ids]))
    count=min(len(ids),int(np.searchsorted(cum,cfg.top_p))+1)
    kept=ids[:count];out=np.full_like(lp,-np.inf);out[kept]=log_normalize(lp[kept])
    return out


def draw_token(logp,cfg,rng):
    if cfg.temperature==0: return int(np.argmax(logp))
    lp=sampling_distribution(logp,cfg)
    return int(rng.choice(len(lp),p=np.exp(lp)))


def generate(main,reference,cfg,eos,decode,seed=0,neutral_main=None):
    if cfg.max_tokens<=0: raise ValueError("Positive token budget required")
    rng=np.random.default_rng(seed);tokens=[];lp_selected=[];trace=[]
    for t in range(cfg.max_tokens):
        m=main.next(tuple(tokens));r=reference.next(tuple(tokens)) if reference else None
        neutral=neutral_main.next(tuple(tokens)) if neutral_main else None
        out,meta=step_distribution(m,r,cfg,t,neutral)
        tok=draw_token(out,cfg,rng)
        tokens.append(tok);lp_selected.append(float(out[tok]))
        sampling_lp=sampling_distribution(out,cfg)
        trace.append({**meta,"token":tok,"log_probability":float(out[tok]),
                      "sampling_log_probability":float(sampling_lp[tok])})
        if tok in eos: break
    return {"text":decode(tokens),"tokens":tokens,"selected_log_probabilities":lp_selected,
            "sequence_log_probability":float(sum(lp_selected)),
            "first_probability":float(np.exp(lp_selected[0])),
            "trace":trace,"terminated":bool(tokens[-1] in eos),
            "status":"ok"}


def replay(main,reference,cfg,tokens,token_groups=None,neutral_main=None):
    """Teacher-force an actual sequence, retaining every local normalizer.
    No assumption of an equivalence between sequence and one-step contrast.
    """
    from .probability import event_decomposition
    rows=[];prefix=[]
    for t,tok in enumerate(tokens):
        m=main.next(tuple(prefix));r=reference.next(tuple(prefix)) if reference else None
        neutral=neutral_main.next(tuple(prefix)) if neutral_main else None
        p=log_normalize(m.logits);out,meta=step_distribution(m,r,cfg,t,neutral)
        rec={"step":t,"prefix":list(prefix),"token":int(tok),"base_logp":float(p[tok]),
             "modified_logp":float(out[tok]),**meta}
        if r is not None:
            q=log_normalize(r.logits);rec['reference_logp']=float(q[tok])
            if cfg.method in {"vcd","icd","sid","m3id"}:
                w=meta['weight'];beta=0 if cfg.method=='m3id' else cfg.beta
                keep=np.isfinite(out)
                from scipy.special import logsumexp
                rec['log_normalizer']=float(logsumexp(((1+w)*p-w*q)[keep]))
                rec['in_support']=bool(keep[tok])
                if token_groups:
                    for name,ids in token_groups.items():
                        e=np.zeros(len(p),bool);e[list(ids)]=True
                        if e.any() and not e.all():
                            rec[name]=event_decomposition(p,q,e,w,beta)
                competitor=int(np.argmax(out))
                if competitor!=tok:
                    rec['pair']=pair_margin(p,q,int(tok),competitor,w)
                    rec['competitor']=competitor
        if cfg.method in {'instruction_vcd','instruction_m3id'}:
            from scipy.special import logsumexp
            q=log_normalize(r.logits);c=log_normalize(neutral.logits);keep=np.isfinite(out)
            w=meta['weight']
            rec.update(neutral_clean_logp=float(c[tok]),reference_logp=float(q[tok]),
                       log_normalizer=float(logsumexp((p+w*(c-q))[keep])),in_support=bool(keep[tok]))
        if cfg.method in {'dola','deco'}:
            from .probability import general_group_decomposition
            from scipy.special import logsumexp
            mu=-1. if cfg.method=='dola' else meta['weight']
            source=m.early_raw if cfg.method=='dola' else m.early_normalized
            q=log_normalize(source[meta['layer']]);keep=np.isfinite(out)
            rec.update(reference_logp=float(q[tok]),condition_weight=1.,reference_weight=mu,
                       log_normalizer=float(logsumexp((p+mu*q)[keep])),in_support=bool(keep[tok]))
            if token_groups:
                for name,ids in token_groups.items():
                    e=np.zeros(len(p),bool);e[list(ids)]=True
                    if e.any() and not e.all():
                        rec[name]=general_group_decomposition(p,q,e,1.,mu,support=keep)
        rows.append(rec);prefix.append(int(tok))
    return rows
