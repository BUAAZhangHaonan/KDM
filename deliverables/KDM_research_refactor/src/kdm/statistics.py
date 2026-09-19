"""Paired estimates with explicit denominators and cluster resampling."""
from __future__ import annotations
import numpy as np


def mean_ci(values,clusters,n_boot=2000,seed=0):
    v=np.asarray(values,float); c=np.asarray(clusters)
    if len(v)!=len(c) or not len(v) or not np.isfinite(v).all():
        raise ValueError("Finite nonempty paired vectors required")
    levels=np.unique(c); sums=np.array([v[c==k].sum() for k in levels])
    ns=np.array([(c==k).sum() for k in levels]);rng=np.random.default_rng(seed)
    if len(levels)<2: return {"mean":float(v.mean()),"lo":None,"hi":None,"n":len(v),"clusters":1}
    estimates=np.empty(n_boot)
    for b in range(n_boot):
        idx=rng.integers(len(levels),size=len(levels))
        estimates[b]=sums[idx].sum()/ns[idx].sum()
    return {"mean":float(v.mean()),"lo":float(np.quantile(estimates,.025)),
            "hi":float(np.quantile(estimates,.975)),"n":len(v),"clusters":len(levels)}


def ratio_ci(numerator,denominator,clusters,n_boot=2000,seed=0):
    x=np.asarray(numerator,float);d=np.asarray(denominator,float);c=np.asarray(clusters)
    if not np.isfinite(x).all() or not np.isfinite(d).all() or not(len(x)==len(d)==len(c)) or (x<0).any() or (d<0).any() or (x>d).any():
        raise ValueError("Invalid ratio event indicators")
    if d.sum()==0: return {"value":None,"lo":None,"hi":None,"denominator":0,"valid_bootstrap":0}
    levels=np.unique(c);xs=np.array([x[c==k].sum() for k in levels]);ds=np.array([d[c==k].sum() for k in levels])
    rng=np.random.default_rng(seed);values=[]
    for _ in range(n_boot):
        idx=rng.integers(len(levels),size=len(levels));den=ds[idx].sum()
        if den>0: values.append(xs[idx].sum()/den)
    return {"value":float(x.sum()/d.sum()),"lo":float(np.quantile(values,.025)) if len(levels)>1 else None,
            "hi":float(np.quantile(values,.975)) if len(levels)>1 else None,
            "denominator":int(d.sum()),"valid_bootstrap":len(values)}


def paired_behavior(before,after):
    if set(before)!=set(after): raise ValueError("Pair set mismatch")
    ids=sorted(before);rows=[]
    for k in ids:
        b,m=before[k],after[k]
        if b['sample_id']!=m['sample_id']: raise ValueError("Sample identity mismatch")
        ab=b['label']=='abstain'; am=m['label']=='abstain'
        answered=m['label'] in {'answer_uncertain','answer_assertive'}
        cb=float(b['correct']);cm=float(m['correct'])
        rows.append({"sample_id":k,"cluster":b['cluster'],"baseline_abstain":int(ab),
          "retained_abstain":int(ab and am),"lost_abstain":int(ab and answered),"abstain_to_invalid":int(ab and m["label"]=="invalid"),
          "lost_to_correct":float(ab and answered)*cm,
          "lost_to_error":float(ab and answered)*(1-cm),
          "accuracy_change":cm-cb,"baseline_correct":cb,"method_correct":cm})
    return rows


def risk_at_coverage(correct,score,coverage=.8):
    y=np.asarray(correct,float);s=np.asarray(score,float)
    if not np.isfinite(y).all() or (y<0).any() or (y>1).any() or y.shape!=s.shape or not len(y) or not 0<coverage<=1 or not np.isfinite(s).all():
        raise ValueError("Invalid coverage inputs")
    k=max(1,int(np.floor(coverage*len(y))))
    cutoff=np.sort(s)[-k]
    above=s>cutoff;ties=s==cutoff
    risk=(np.sum(1-y[above])+(k-above.sum())*np.mean(1-y[ties]))/k
    return {"risk":float(risk),"coverage":k/len(y),"n_selected":k}
