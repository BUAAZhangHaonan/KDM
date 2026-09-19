"""Single-question figures with explicit data provenance on every canvas."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .probability import log_probs,contrast,event_decomposition,instruction_preserving,log_normalize


def canvas(title,subtitle,size=(7.1,4.2)):
    fig,ax=plt.subplots(figsize=size)
    fig.subplots_adjust(left=.17,right=.96,bottom=.23,top=.79)
    fig.text(.08,.94,title,fontsize=14,weight='bold')
    fig.text(.08,.865,subtitle,fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=9);ax.set_axisbelow(True)
    return fig,ax


def save(fig,stem,provenance):
    stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
    fig.text(.08,.035,provenance,fontsize=8)
    files=[]
    for suffix in ('png','pdf'):
        path=stem.with_suffix('.'+suffix)
        fig.savefig(path,dpi=220,metadata={'Creator':'KDM visualization'} if suffix=='pdf' else None)
        files.append(str(path))
    plt.close(fig);return files


def plot_reported_fates(source,stem):
    d=json.load(open(source));counts=[d['lost_outcomes'][k] for k in ('wrong_specific_food','generic_food','unrelated_object','correct_specific_food')]+[d['retained']]
    if sum(counts)!=d['baseline_abstentions']:raise ValueError('Fate counts do not equal original abstentions')
    labels=['Other incorrect food','Broad food description','Other object','Correct food name','Retained abstention']
    fig,ax=canvas('Where the original abstentions went','LLaVA-v1.6-7B | Food-101 | 72 baseline abstentions',size=(7.1,4.6))
    fig.subplots_adjust(left=.32)
    y=np.arange(5);ax.barh(y,counts,height=.6);ax.set_yticks(y,labels);ax.invert_yaxis()
    for i,n in enumerate(counts):ax.text(n+.4,i,str(n),va='center',fontsize=10)
    ax.set_xlim(0,max(counts)*1.14);ax.set_xlabel('Number of original abstentions',fontsize=10)
    return save(fig,stem,'USER-REPORTED COUNTS | 66 replacements; no new model inference in this package')


def plot_lexical_boundary(stem):
    p=np.array([.35,.05,.3,.3]);qs=[np.array([.55,.05,.2,.2]),np.array([.05,.55,.2,.2])]
    group=np.array([1,1,0,0],bool)
    result=[event_decomposition(log_probs(p),log_probs(q),group,1,0) for q in qs]
    vals=[r['event_mass_after'] for r in result]
    fig,ax=canvas('The same group probability can yield opposite effects','Both references assign 60% to abstention; only the word allocation changes')
    ax.bar([0,1],np.array(vals)*100,width=.47)
    ax.axhline(40,linestyle='--',linewidth=1.4,label='Direct abstention probability: 40%')
    for i,v in enumerate(vals):ax.text(i,v*100+2,f'{100*v:.1f}%',ha='center',fontsize=12,weight='bold')
    ax.set_xticks([0,1],['Reference: UNKNOWN 55%\nUNCLEAR 5%','Reference: UNKNOWN 5%\nUNCLEAR 55%'])
    ax.set_ylim(0,95);ax.set_ylabel('Abstention group probability (%)',fontsize=10);ax.legend(loc='upper left',fontsize=8,frameon=False)
    return save(fig,stem,'MATHEMATICAL EXAMPLE | Four outcomes; exact VCD-form transformation, alpha = 1')


def plot_decomposition(stem):
    p=log_probs([.32,.02,.36,.2,.1]);q=log_probs([.5,.04,.26,.15,.05]);group=np.array([1,1,0,0,0],bool)
    d=event_decomposition(p,q,group,1,.1)
    values=[d['support_change'],d['reference_group_change'],d['within_group_change']]
    starts=[0,values[0],sum(values[:2])];fig,ax=canvas('An exact decomposition of the probability change','Support, between-group reference preference, and within-group allocation')
    for i,(start,v) in enumerate(zip(starts,values)):
        ax.bar(i,v,bottom=start,width=.56)
        ax.text(i,start+v+(.045 if v>=0 else -.08),f'{v:+.3f}',ha='center',fontsize=10)
        if i<2:ax.plot([i+.28,i+.72],[start+v,start+v],linestyle=':',linewidth=1)
    ax.scatter([3],[sum(values)],s=75,marker='D');ax.text(3,sum(values)-.11,f'{sum(values):+.3f}',ha='center',fontsize=10)
    ax.axhline(0,linewidth=.8);ax.set_xticks(range(4),['Candidate\nrestriction','Reference\ngroup preference','Within-group\nallocation','Total change'])
    ax.set_ylabel('Change in log odds',fontsize=10);ax.margins(y=.3)
    return save(fig,stem,'MATHEMATICAL EXAMPLE | Natural logarithms; all terms computed from the same distributions')


def plot_instruction_invariance(stem):
    rng=np.random.default_rng(519);x=[];old=[];new=[]
    for _ in range(100):
        g,qg,c,r=rng.normal(size=(4,9))
        neutral,_=contrast(c,r,1,0);ordinary,_=contrast(g,qg,1,0);proposed,_=instruction_preserving(g,c,r,1,0)
        x.append((g[0]-g[1])-(c[0]-c[1]))
        old.append((ordinary[0]-ordinary[1])-(neutral[0]-neutral[1]))
        new.append((proposed[0]-proposed[1])-(neutral[0]-neutral[1]))
    fig,ax=canvas('Preserving the instruction-induced preference','Observed score differences in independently generated finite distributions')
    ax.scatter(x,old,s=18,marker='x',alpha=.5,label='Ordinary guided contrast')
    ax.scatter(x,new,s=18,marker='o',alpha=.7,label='Instruction-preserving combination')
    lim=7;ax.plot([-lim,lim],[-lim,lim],linestyle='--',linewidth=1)
    ax.set_xlim(-lim,lim);ylim=max(10.,max(abs(v) for v in old+new)*1.1);ax.set_ylim(-ylim,ylim)
    ax.set_xlabel('Instruction effect in direct pairwise log odds',fontsize=9)
    ax.set_ylabel('Instruction effect after contrast',fontsize=9)
    ax.legend(fontsize=8,frameon=False,loc='upper left')
    return save(fig,stem,'MATHEMATICAL EXAMPLE | 100 independent four-condition score arrays; no VLM measurements')


def plot_semantic_matrix(rows,stem,source_note):
    markers=['UNKNOWN','UNCLEAR','UNSURE','I cannot identify it'];matrix=np.full((4,4),np.nan);seen=set()
    for r in rows:
        i=markers.index(r['marker']);j=markers.index(r['reference_marker'])
        if (i,j) in seen:raise ValueError('Duplicate matrix cell')
        seen.add((i,j));matrix[i,j]=np.nan if r['retention'] is None else r['retention']
    if len(seen)!=16:raise ValueError('All sixteen response conditions are required')
    fig,ax=canvas('Abstention retention across response expressions','Rows: clear-image instruction | Columns: reference instruction',size=(7.1,5.5))
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    short=['UNKNOWN','UNCLEAR','UNSURE','I cannot\nidentify it']
    ax.set_xticks(range(4),short);ax.set_yticks(range(4),short)
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:.2f}' if np.isfinite(matrix[i,j]) else 'N/A',ha='center',va='center',fontsize=11)
    fig.colorbar(im,ax=ax,label='Retained original abstentions',fraction=.04,pad=.04)
    return save(fig,stem,source_note)


def plot_method_tradeoff(rows,stem,source_note):
    fig,ax=canvas('Correct answers and retained abstentions','Both coordinates are required to interpret the behavior change')
    for r in rows:
        ax.scatter(r['accuracy']*100,r['retention']*100,s=65)
        ax.annotate(r['method'],(r['accuracy']*100,r['retention']*100),xytext=(6,5),textcoords='offset points',fontsize=9)
    ax.set_xlabel('Correct answers / all questions (%)',fontsize=10)
    ax.set_ylabel('Retained appropriate abstentions (%)',fontsize=10)
    ax.set_ylim(0,108);ax.margins(x=.4)
    return save(fig,stem,source_note)
