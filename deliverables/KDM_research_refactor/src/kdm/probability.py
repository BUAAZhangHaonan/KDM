"""Exact finite-support calculations; no model semantics are inferred here."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp


def log_normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or np.isnan(x).any() or np.isposinf(x).any():
        raise ValueError("Expected one-dimensional finite/-inf scores")
    z = logsumexp(x)
    if not np.isfinite(z):
        raise ValueError("Empty probability support")
    return x - z


def log_probs(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Invalid probabilities")
    with np.errstate(divide="ignore"):
        return log_normalize(np.log(p))


def contrast(log_p, log_q, alpha=1.0, beta=0.1):
    """VCD-form local distribution and support. Natural logarithms throughout."""
    if alpha < 0 or not 0 <= beta <= 1:
        raise ValueError("alpha must be nonnegative; beta must be in [0,1]")
    p, q = log_normalize(log_p), log_normalize(log_q)
    if p.shape != q.shape:
        raise ValueError("Different vocabularies")
    keep = np.isfinite(p) if beta == 0 else p >= p.max() + np.log(beta)
    if alpha > 0 and not np.isfinite(q[keep]).all():
        raise ValueError("Reference must be positive on retained support")
    scores = p.copy()
    if alpha:
        scores[keep] = (1 + alpha) * p[keep] - alpha * q[keep]
    scores[~keep] = -np.inf
    return log_normalize(scores), keep


def renyi(log_p, log_q, order):
    if order <= 0:
        raise ValueError("Positive Renyi order required")
    p, q = log_normalize(log_p), log_normalize(log_q)
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("This implementation requires positive conditional probabilities")
    if np.isclose(order, 1.0, atol=1e-12, rtol=0):
        return float(np.dot(np.exp(p), p-q))
    return float(logsumexp(order*p + (1-order)*q) / (order-1))


def event_log_odds(log_p, event):
    p = log_normalize(log_p)
    event = np.asarray(event, bool)
    if event.shape != p.shape or not event.any() or event.all():
        raise ValueError("Event must be a nontrivial partition")
    return float(logsumexp(p[event]) - logsumexp(p[~event]))


def event_decomposition(log_p, log_q, event, alpha=1.0, beta=0.1):
    """Exact mask + group-odds + within-group Renyi decomposition at one prefix.

    A group denotes a supplied partition of next-token candidates. It is not
    automatically the event of a complete natural-language abstention.
    """
    p, q = log_normalize(log_p), log_normalize(log_q)
    event = np.asarray(event, bool)
    if event.shape != p.shape:
        raise ValueError("Invalid event mask")
    out, keep = contrast(p, q, alpha, beta)
    ae, ce = event & keep, ~event & keep
    if not ae.any() or not ce.any():
        return {"status":"event_excluded" if not ae.any() else "complement_excluded",
                "event_mass_after":float(np.exp(logsumexp(out[event]))),
                "n_retained":int(keep.sum())}
    p_s, q_s = log_normalize(p[keep]), log_normalize(q[keep])
    e = event[keep]
    base = event_log_odds(p_s, e)
    ref = event_log_odds(q_s, e)
    da = renyi(p_s[e], q_s[e], 1+alpha)
    dc = renyi(p_s[~e], q_s[~e], 1+alpha)
    mask_change = base - event_log_odds(p, event)
    semantic_change = alpha * (base-ref)
    lexical_change = alpha * (da-dc)
    actual_change = event_log_odds(out, event)-event_log_odds(p,event)
    return {"status":"defined", "support_change":mask_change,
            "reference_group_change":semantic_change,
            "within_group_change":lexical_change, "actual_change":actual_change,
            "closure_error":actual_change-mask_change-semantic_change-lexical_change,
            "event_mass_before":float(np.exp(logsumexp(p[event]))),
            "event_mass_after":float(np.exp(logsumexp(out[event]))),
            "reference_event_mass":float(np.exp(logsumexp(q[event]))),
            "n_retained":int(keep.sum())}


def event_derivative(log_p, log_q, event, alpha=1.0, beta=0.1):
    """Derivative with a fixed plausibility support and fixed reference."""
    p, q = log_normalize(log_p), log_normalize(log_q)
    out, keep = contrast(p,q,alpha,beta)
    e = np.asarray(event,bool)
    weights = np.exp(out[keep]); score = p[keep]-q[keep]
    indicators=e[keep].astype(float)
    return float(np.dot(weights, indicators*score)
                 - np.dot(weights, indicators)*np.dot(weights,score))


def complete_response_shift(log_base, log_modified, event):
    """Exact identity on a declared finite set of disjoint complete responses.

    log_modified must come from actual autoregressive normalized step scores.
    A product of unnormalized contrast scores is not accepted as that quantity.
    The result is conditional on the declared response set.
    """
    p, m = log_normalize(log_base), log_normalize(log_modified)
    e = np.asarray(event,bool)
    if e.shape != p.shape or not e.any() or e.all():
        raise ValueError("Invalid response partition")
    if not np.isfinite(p).all():
        raise ValueError("Base response probabilities must be positive")
    delta=m-p
    term_a=float(logsumexp(log_normalize(p[e])+delta[e]))
    term_b=float(logsumexp(log_normalize(p[~e])+delta[~e]))
    observed=event_log_odds(m,e)-event_log_odds(p,e)
    if not np.isfinite(observed):
        return {'status':'zero_group_mass','observed':observed,'event_term':term_a,'answer_term':term_b,'closure_error':None}
    return {"observed":observed,"event_term":term_a,"answer_term":term_b,
            "closure_error":observed-term_a+term_b}


def pair_margin(log_p,log_q,abstention_id,answer_id,alpha):
    p,q=log_normalize(log_p),log_normalize(log_q)
    if abstention_id==answer_id:
        raise ValueError("Pair tokens must be distinct")
    direct=float(p[abstention_id]-p[answer_id])
    reference=float(q[abstention_id]-q[answer_id])
    return {"direct_margin":direct,"reference_margin":reference,
            "modified_margin":(1+alpha)*direct-alpha*reference,
            "reference_excess":reference-direct}


def instruction_preserving(log_guided,log_clear_neutral,log_reference_neutral,alpha=1.,beta=.1):
    """Guided base + instruction-independent visual log-likelihood ratio.

    Fixed-prefix pairwise odds preserve the guided-vs-neutral instruction effect
    on shared retained candidates. A guarantee of correct final answers is not
    supplied by this identity. The three-distribution product belongs to the
    product-of-experts family; the defining contribution is the choice of views.
    """
    if alpha < 0 or not 0 <= beta <= 1: raise ValueError("Invalid contrast settings")
    g,c,r = map(log_normalize,(log_guided,log_clear_neutral,log_reference_neutral))
    if not g.shape==c.shape==r.shape: raise ValueError("Different vocabularies")
    keep=np.isfinite(g) if beta==0 else g>=g.max()+np.log(beta)
    if not np.isfinite(c[keep]).all() or not np.isfinite(r[keep]).all():
        raise ValueError("Neutral distributions must be positive on support")
    result=g.copy();result[keep]+=alpha*(c[keep]-r[keep]);result[~keep]=-np.inf
    return log_normalize(result),keep


def instruction_interaction(log_pg,log_qg,log_pn,log_qn):
    """Difference of instruction responses between reference and clear views.
    Scores are represented as log probabilities; only differences between tokens
    enter an odds claim. This calculation uses four observed conditions.
    """
    pg,qg,pn,qn=map(log_normalize,(log_pg,log_qg,log_pn,log_qn))
    return (qg-qn)-(pg-pn)


def general_group_decomposition(log_p,log_q,event,condition_weight=2.,reference_weight=-1.,beta=.1,support=None):
    """Exact group decomposition for arbitrary two-logit coefficients.

    Includes VCD, DoLa and additive layer-score correction after their actual
    coefficients and reference probabilities have been specified.
    """
    p,q=log_normalize(log_p),log_normalize(log_q);e=np.asarray(event,bool)
    if p.shape!=q.shape or e.shape!=p.shape or not e.any() or e.all():raise ValueError('Invalid distribution partition')
    if not 0<=beta<=1:raise ValueError('Invalid support parameter')
    keep=np.isfinite(p) if beta==0 else p>=p.max()+np.log(beta)
    if support is not None:
        keep=np.asarray(support,bool)
        if keep.shape!=p.shape or not keep.any():raise ValueError('Invalid explicit support')
    if not np.isfinite(p[keep]).all() or not np.isfinite(q[keep]).all():raise ValueError('Positive distributions on support required')
    if not (e&keep).any() or not ((~e)&keep).any():return {'status':'group_excluded'}
    a,b=float(condition_weight),float(reference_weight)
    if not np.isfinite([a,b]).all():raise ValueError('Finite coefficients required')
    score=np.full_like(p,-np.inf);score[keep]=a*p[keep]+b*q[keep];m=log_normalize(score)
    ps,qs=log_normalize(p[keep]),log_normalize(q[keep]);es=e[keep]
    def inner(mask):return float(logsumexp(a*log_normalize(ps[mask])+b*log_normalize(qs[mask])))
    support=event_log_odds(ps,es)-event_log_odds(p,e)
    between=(a-1)*event_log_odds(ps,es)+b*event_log_odds(qs,es)
    inside=inner(es)-inner(~es);actual=event_log_odds(m,e)-event_log_odds(p,e)
    return {'status':'defined','support_change':support,'between_group_change':between,
            'within_group_change':inside,'actual_change':actual,
            'closure_error':actual-support-between-inside}


def tilt_objective(policy, log_base, advantage, strength=1.):
    """Expected reference advantage minus KL to a positive base distribution.

    All vectors are restricted to the same finite retained support. The unique
    maximizer is proportional to base * exp(strength * advantage).
    """
    x=np.asarray(policy,dtype=np.float64);base=log_normalize(log_base)
    d=np.asarray(advantage,dtype=np.float64)
    if x.shape!=base.shape or d.shape!=base.shape or not np.isfinite(x).all() or (x<0).any() or not np.isclose(x.sum(),1.,atol=1e-12,rtol=0):
        raise ValueError('A normalized policy on the supplied support is required')
    if not np.isfinite(base).all() or not np.isfinite(d).all() or not np.isfinite(strength) or strength<0:
        raise ValueError('Positive base, finite advantages and nonnegative strength required')
    positive=x>0
    kl=float(np.dot(x[positive],np.log(x[positive])-base[positive]))
    return float(strength*np.dot(x,d)-kl)
