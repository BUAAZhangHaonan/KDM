"""CDA equations 4/6/7, with an explicitly declared visual-context mapping."""
from __future__ import annotations
import numpy as np
from .probability import log_normalize
from .decoding import draw_token


def entropy(logits):
    lp=log_normalize(logits);p=np.exp(lp);nz=p>0
    return float(-np.dot(p[nz],lp[nz]))


class CDAUndefinedError(ValueError):
    """An undefined calibration is an error record, never a substitute mixture."""
    def __init__(self,reason,details):
        super().__init__('CDA calibration undefined: '+reason)
        self.cda_diagnostics={'status':'undefined','reason':reason,**details}


def cda_calibration(prior,context,null_prior,null_context):
    hp,hc,hnp,hnc=map(entropy,(prior,context,null_prior,null_context))
    entropies={'h_prior':hp,'h_context':hc,'h_null_prior':hnp,'h_null_context':hnc}
    if hnp<=0 or hnc<=0:raise CDAUndefinedError('nonpositive_null_entropy',entropies)
    # ACL 2025 main-text equation (6), as explicitly selected by the user.
    rp=max(hp-hnp,0)/hnp;rc=max(hc-hnc,0)/hnc
    total=rp+rc
    # Equation (7) has a removable 0/0 at the origin; preserve its continuous extension.
    wp,wc=(rp*rp/total,rc*rc/total) if total>0 else (0.,0.)
    wa=1-wp-wc
    if not np.isfinite([rp,rc,wp,wc,wa]).all():
        raise CDAUndefinedError('nonfinite_calibration_arithmetic',entropies)
    return {**entropies,'rp':rp,'rc':rc,'wp':wp,'wc':wc,'wa':wa,
            'negative_wa':bool(wa<0),'zero_sum_extension':bool(total==0),
            'calibration_status':'zero_sum_continuous_extension' if total==0 else 'defined'}


def cda_weights(prior,context,null_prior,null_context):
    state=cda_calibration(prior,context,null_prior,null_context)
    return np.array([state['wp'],state['wc'],state['wa']],float)


def generate_cda(prior,context,abstention,null_prior,null_context,cfg,eos,decode,seed=0):
    tokens=[];trace=[];logps=[];rng=np.random.default_rng(seed)
    for _ in range(cfg.max_tokens):
        states=[s.next(tuple(tokens)).logits for s in (prior,context,abstention,null_prior,null_context)]
        try:calibration=cda_calibration(states[0],states[1],states[3],states[4])
        except CDAUndefinedError as exc:
            exc.cda_diagnostics.update(step=len(tokens),prefix_tokens=list(tokens),completed_trace=list(trace))
            raise
        w=np.array([calibration['wp'],calibration['wc'],calibration['wa']],float)
        logits=sum(weight*z for weight,z in zip(w,states[:3]));lp=log_normalize(logits)
        tok=draw_token(lp,cfg,rng);tokens.append(tok);logps.append(float(lp[tok]))
        trace.append({'token':tok,'weights':w.tolist(),**calibration,'log_probability':float(lp[tok])})
        if tok in eos:break
    return {'status':'ok','text':decode(tokens),'tokens':tokens,'selected_log_probabilities':logps,
            'sequence_log_probability':float(sum(logps)),'first_probability':float(np.exp(logps[0])),
            'trace':trace,'terminated':tokens[-1] in eos,'implementation':'CDA_visual_context_adaptation',
            'null_convention':'same_prefix_text_placeholder_and_uniform_image',
            'cda_equations':'ACL2025_main_text_4_6_7_no_momentum',
            'negative_wa_steps':sum(row['negative_wa'] for row in trace),
            'zero_sum_extension_steps':sum(row['zero_sum_extension'] for row in trace)}
