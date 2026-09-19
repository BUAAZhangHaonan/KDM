"""Controlled four-condition measurements along an observed direct-answer path."""
from __future__ import annotations
import numpy as np
from .prompts import task_prompt, MARKERS
from .decoding import DecodeConfig, distribution, Step
from .probability import log_normalize, event_decomposition, instruction_preserving, instruction_interaction


def measure_path(backend,image,question,tokens,method='vcd',marker='UNKNOWN',reference_marker='UNKNOWN',seed=0):
    if method in {'dola','deco'}:
        return measure_layer_path(backend,image,question,tokens,method,marker,reference_marker)
    if method not in {'vcd','m3id','sid'}:raise ValueError('Unsupported shared-prefix method')
    neutral=task_prompt(question,guided=False)
    guided=task_prompt(question,marker,True)
    reference_guided=task_prompt(question,reference_marker,True)
    mode={'vcd':'noise','m3id':'text_only','sid':'sid'}[method]
    g=backend.session(image,guided);qg=backend.session(image,reference_guided,reference=mode,seed=seed)
    c=backend.session(image,neutral);r=backend.session(image,neutral,reference=mode,seed=seed)
    offset=len(backend.encode(guided))
    cfg=DecodeConfig(method=method,m3id_offset=offset)
    # This deliberately records token-prefix groups separately from full-response semantics.
    first_ids=sorted({backend.encode(text)[0] for text in MARKERS if backend.encode(text)})
    rows=[];first_divergence=None
    for t,observed in enumerate(tokens):
        prefix=tuple(tokens[:t]);gs=g.next(prefix);qgs=qg.next(prefix);cs=c.next(prefix);rs=r.next(prefix)
        pg,pq,pc,pr=[log_normalize(v.logits) for v in (gs,qgs,cs,rs)]
        old,meta=distribution(gs,qgs,cfg,t);w=meta['weight'];beta=0. if method=='m3id' else cfg.beta
        modified,_=instruction_preserving(pg,pc,pr,w,beta)
        actual_weight=w
        if method=='m3id':
            active=np.exp(pc.max())<cfg.m3id_threshold
            actual_weight=float(np.expm1(cfg.m3id_lambda*(t+len(backend.encode(neutral))))) if active else 0.
        actual_modified,_=instruction_preserving(pg,pc,pr,actual_weight,beta)
        competitor=int(np.argmax(old));direct_argmax=int(np.argmax(pg))
        if first_divergence is None and competitor!=int(observed):first_divergence=t
        j=int(observed);k=competitor
        interaction=instruction_interaction(pg,pq,pc,pr)
        keep=np.isfinite(old)&np.isfinite(modified)
        record={'step':t,'prefix':list(prefix),'observed_token':j,'direct_argmax':direct_argmax,'method_argmax':k,
                'weight':w,'gate_active':meta['active'],
                'actual_instruction_method_weight':actual_weight,
                'actual_instruction_method_argmax':int(np.argmax(actual_modified)),
                'identity_comparison':'matched_original_weight_for_four_condition_identity',
                'observed_logp':{'guided_clean':float(pg[j]),'guided_reference':float(pq[j]),
                                 'neutral_clean':float(pc[j]),'neutral_reference':float(pr[j])},
                'guided_clean_advantage':float(pg[j]-pg[k]),
                'guided_reference_advantage':float(pq[j]-pq[k]),
                'contrast_advantage':float(old[j]-old[k]),
                'instruction_interaction_advantage':float(interaction[j]-interaction[k]),
                'common_pair_retained':bool(keep[j] and keep[k])}
        if keep[j] and keep[k]:
            record['pair_identity_error']=float((modified[j]-modified[k])-(old[j]-old[k])-w*(interaction[j]-interaction[k]))
        if t==0:
            event=np.zeros(len(pg),bool);event[first_ids]=True
            record['marker_initial_token_ids']=first_ids
            record['initial_token_group']=event_decomposition(pg,pq,event,w,beta)
            record['group_scope']='initial_tokens_of_declared_markers_not_complete_semantic_responses'
        rows.append(record)
    if method=='sid' and getattr(backend,'sid_control',None):
        backend.sid_control.close();backend.sid_control=None
    return {'observed_tokens':list(tokens),'first_divergence':first_divergence,'steps':rows,
            'method':method,'marker':marker,'reference_marker':reference_marker,
            'matched_prefix':'original_direct_response',
            'abstention_definition':'whole_response_annotation_in_separate_ledger'}


def measure_layer_path(backend,image,question,tokens,method,marker,reference_marker):
    """Actual selected-layer references, without asserting a VCD-form identity."""
    if reference_marker!=marker:raise ValueError('Layer reference has no independent prompt')
    guided=backend.session(image,task_prompt(question,marker,True),need_layers=True)
    neutral=backend.session(image,task_prompt(question,guided=False),need_layers=True)
    cfg=DecodeConfig(method=method);rows=[];first=None
    first_ids=sorted({backend.encode(text)[0] for text in MARKERS if backend.encode(text)})
    from .probability import general_group_decomposition
    from scipy.special import logsumexp
    for t,observed in enumerate(tokens):
        prefix=tuple(tokens[:t]);gs=guided.next(prefix);cs=neutral.next(prefix)
        out,meta=distribution(gs,None,cfg,t);nout,nmeta=distribution(cs,None,cfg,t)
        pg,pc=log_normalize(gs.logits),log_normalize(cs.logits)
        gsource=gs.early_raw if method=='dola' else gs.early_normalized
        csource=cs.early_raw if method=='dola' else cs.early_normalized
        qg=log_normalize(gsource[meta['layer']]);r=log_normalize(csource[nmeta['layer']])
        mu=-1. if method=='dola' else meta['weight'];keep=np.isfinite(out)
        j=int(observed);k=int(np.argmax(out))
        if first is None and j!=k:first=t
        row={'step':t,'prefix':list(prefix),'observed_token':j,'direct_argmax':int(np.argmax(pg)),
             'method_argmax':k,'neutral_method_argmax':int(np.argmax(nout)),
             'selected_guided_layer':meta['layer'],'selected_neutral_layer':nmeta['layer'],
             'condition_weight':1.,'reference_weight':mu,'n_retained':int(keep.sum()),
             'observed_logp':{'guided_clean':float(pg[j]),'guided_reference':float(qg[j]),
                              'neutral_clean':float(pc[j]),'neutral_reference':float(r[j])},
             'base_logp':float(pg[j]),'modified_logp':float(out[j]),
             'log_normalizer':float(logsumexp((pg+mu*qg)[keep])),
             'guided_clean_advantage':float(pg[j]-pg[k]),
             'guided_reference_advantage':float(qg[j]-qg[k]),
             'contrast_advantage':float(out[j]-out[k]),'common_pair_retained':bool(keep[j] and keep[k]),
             'reference_definition':'actual_selected_layer_within_each_prompt_condition'}
        if t==0:
            event=np.zeros(len(pg),bool);event[first_ids]=True
            if event.any() and not event.all():
                row['initial_token_group']=general_group_decomposition(pg,qg,event,1.,mu,support=keep)
            row['marker_initial_token_ids']=first_ids
            row['group_scope']='initial_tokens_of_declared_markers_not_complete_semantic_responses'
        rows.append(row)
    return {'observed_tokens':list(tokens),'first_divergence':first,'steps':rows,
            'method':method,'marker':marker,'reference_marker':reference_marker,
            'matched_prefix':'original_direct_response',
            'abstention_definition':'whole_response_annotation_in_separate_ledger'}
