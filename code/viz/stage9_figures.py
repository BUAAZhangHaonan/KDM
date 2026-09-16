"""Seven independently executable figures. Each figure has ONE axes.

Real aggregate charts and illustrative/fixture charts are stored separately.
No displayed coordinate is manufactured to replace a missing experimental result.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from stage9_geometry import target_interval

MODEL_NAMES={'q4b':'Qwen3.5-4B','q9b':'Qwen3.5-9B','llava16':'LLaVA-v1.6-7B','internvl4b':'InternVL3.5-4B'}
METHOD_NAMES={'direct':'Direct','vcd':'VCD','m3id':'M3ID','dola':'DoLa','deco':'DeCo'}
COLORS={'positive':'#2E827C','negative':'#C35B4E','ink':'#243344','muted':'#687585',
        'blue':'#4263A2','line':'#D9E0E5','light':'#F3F6F8'}
M_COLORS={'direct':'#243344','vcd':'#4263A2','m3id':'#8B63A8','dola':'#BB7A33','deco':'#2E827C'}
MARKERS={'vcd':'o','m3id':'s','dola':'D','deco':'^'}


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':12,
        'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':240,'figure.facecolor':'white'})


def save(fig,path:Path,source:str):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.text(.01,.015,source,fontsize=6.8,color=COLORS['muted'],ha='left',va='bottom')
    fig.savefig(path.with_suffix('.png'),dpi=240,bbox_inches='tight',pad_inches=.10)
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.10)
    plt.close(fig)


def error_exchange(data:Path,out:Path):
    t=pd.read_csv(data);t=t[t.stratum=='high_acc'].copy()
    fig,ax=plt.subplots(figsize=(7.2,6.2))
    y=np.arange(len(t));correct=100*t.corrected_n/t.n;wrong=100*t.induced_n/t.n
    ax.barh(y,correct,height=.57,color=COLORS['positive'],label='Incorrect to correct')
    ax.barh(y,-wrong,height=.57,color=COLORS['negative'],label='Correct to incorrect')
    ax.scatter(correct-wrong,y,s=27,marker='D',color=COLORS['ink'],zorder=4,label='Net accuracy change')
    ax.axvline(0,color=COLORS['ink'],linewidth=.7)
    ax.set_yticks(y,[f'{MODEL_NAMES[r.model]}  |  {METHOD_NAMES[r.method]}' for r in t.itertuples()])
    ax.invert_yaxis();ax.set_xlim(-23,8);ax.set_xlabel('Percentage of all evaluated images (percentage points)')
    for j in (3.5,7.5,11.5):ax.axhline(j,color=COLORS['line'],linewidth=.8)
    for j,v in enumerate(correct-wrong):
        ax.annotate(f'{v:+.1f}',(v,j),xytext=(5 if v>=0 else -5,0),textcoords='offset points',
                    va='center',ha='left' if v>=0 else 'right',fontsize=7.2,color=COLORS['ink'])
    fig.text(.29,.970,'Corrections and induced errors must be counted together',fontsize=11.5,ha='left',va='top')
    ax.text(0,1.02,'High-accuracy category subset  |  600 images per model',transform=ax.transAxes,fontsize=9,color=COLORS['muted'])
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.065),ncol=3,frameon=False,fontsize=7.5,handlelength=1.2)
    fig.subplots_adjust(left=.29,right=.98,bottom=.12,top=.80)
    save(fig,out,'REAL DATA  |  KDM e5eda213  |  stage6_core.csv; counts verified against end-point accuracy')


def confidence_tradeoff(data:Path,out:Path):
    t=pd.read_csv(data);fig,ax=plt.subplots(figsize=(6.9,4.5))
    for st,color in [('low_acc',COLORS['negative']),('high_acc',COLORS['positive'])]:
        for m in MARKERS:
            q=t[(t.stratum==st)&(t.method==m)]
            ax.scatter(100*(q.acc_method-q.acc_direct),100*q.dconf_unchanged,s=48,
                       marker=MARKERS[m],facecolor=color,edgecolor='white',linewidth=.55,zorder=3)
    ax.axvline(0,color=COLORS['ink'],linewidth=.8);ax.axhline(0,color=COLORS['line'],linewidth=.8)
    ax.set_xlim(-19,8);ax.set_ylim(-1.5,42)
    ax.set_xlabel('Accuracy change (percentage points)')
    ax.set_ylabel('Confidence change on unchanged answers\n(percentage points)')
    ax.set_title('Higher confidence does not require higher accuracy',loc='left',pad=20)
    row=t[(t.model=='internvl4b')&(t.method=='dola')&(t.stratum=='high_acc')].iloc[0]
    ax.annotate('InternVL3.5-4B / DoLa',xy=(100*(row.acc_method-row.acc_direct),100*row.dconf_unchanged),
        xytext=(-17,30),arrowprops={'arrowstyle':'-','lw':.7,'color':COLORS['muted']},fontsize=8)
    handles=[Line2D([],[],marker='o',linestyle='',color=COLORS['negative'],label='Low-accuracy subset'),
             Line2D([],[],marker='o',linestyle='',color=COLORS['positive'],label='High-accuracy subset')]
    l=ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.42,1.015),ncol=2,frameon=False,fontsize=7.6);ax.add_artist(l)
    ax.legend(handles=[Line2D([],[],marker=v,linestyle='',color=COLORS['ink'],label=METHOD_NAMES[k]) for k,v in MARKERS.items()],
              loc='lower left',ncol=4,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.14,bottom=.17,right=.98,top=.88)
    save(fig,out,'REAL DATA  |  32 model-subset-method combinations  |  Positive confidence means a positive subset mean')


def offset_reversal(data:Path,out:Path):
    t=pd.read_csv(data);fig,ax=plt.subplots(figsize=(6.9,4.5))
    for r in t.itertuples():
        col=COLORS['negative'] if r.stratum=='low_acc' else COLORS['positive']
        ls='-' if r.model=='q4b' else '--'
        ax.plot([0,1,2],100*np.array([r.direct,r.intervention,r.offset_removed]),color=col,
                linestyle=ls,marker='o',markersize=5,linewidth=1.7,
                label=f"{MODEL_NAMES[r.model]} / {METHOD_NAMES[r.method]} / {'low' if r.stratum=='low_acc' else 'high'}")
    ax.set_xticks([0,1,2],['Direct decoding','Decoding intervention','Offset removed'])
    ax.set_ylabel('Expected calibration error (percentage points)');ax.set_ylim(0,91)
    ax.set_title('Undoing a confidence offset can also undo a calibration gain',loc='left',pad=15)
    ax.legend(frameon=False,fontsize=7.5,loc='upper left')
    fig.subplots_adjust(left=.12,bottom=.16,right=.98,top=.87)
    save(fig,out,'REAL DATA  |  Four selected stage6_correction.csv rows  |  Diagnostic score transform, not a new method')


def reachable_intervals(out:Path):
    cases=[('An aligned reference preserves the wrong winner',np.array([2.,1.,-2.]),np.zeros(3),1),
           ('A relative advantage crosses the original margin',np.array([2.,1.,-2.]),np.array([2.,-1.,0.]),1),
           ('An excluded target remains unreachable',np.array([3.,2.,-1.]),np.array([3.,2.,-12.]),2)]
    fig,ax=plt.subplots(figsize=(7.2,3.9));meta=[]
    for i,(name,z,r,target) in enumerate(cases):
        result=target_interval(z,r,target)
        ax.broken_barh([(0,4)],(i-.23,.46),facecolors=COLORS['light'])
        if not result.empty:
            lo=max(0,result.lower);hi=min(4,result.upper)
            ax.broken_barh([(lo,hi-lo)],(i-.23,.46),facecolors=COLORS['positive'])
            ax.text(hi-.08,i,'Target selected',va='center',ha='right',color='white',fontsize=8)
            ax.axvline(lo,ymin=(i-.23+.7)/3.4,ymax=(i+.23+.7)/3.4,color=COLORS['ink'],lw=.5)
        else:ax.text(3.9,i,'No feasible nonnegative strength',va='center',ha='right',color=COLORS['muted'],fontsize=8)
        ax.text(0,i-.36,name,fontsize=8.4,ha='left',va='center',color=COLORS['ink'])
        meta.append({'case':name,'scores':z.tolist(),'reference':r.tolist(),'target':target,'interval':result.to_dict()})
    ax.set_ylim(-.7,2.7);ax.invert_yaxis();ax.set_xlim(0,4);ax.set_yticks([])
    ax.set_xlabel('Contrast strength');ax.set_title('A fixed reference defines a reachable interval',loc='left',pad=16)
    ax.spines['left'].set_visible(False);fig.subplots_adjust(left=.05,right=.98,bottom=.19,top=.84)
    save(fig,out,'MATHEMATICAL EXAMPLE  |  Three-token distributions  |  Exact inequalities, not empirical KDM results')
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2,allow_nan=False))


def risk_curves(data:Path,out:Path,model='q9b',fixture=False):
    t=pd.read_csv(data);t=t[(t.model==model)&(t.score_kind=='sequence')]
    if t.empty:raise ValueError('No risk-coverage data for selected model.')
    fig,ax=plt.subplots(figsize=(6.8,4.5))
    for st,ls in [('low_acc','-'),('high_acc','--')]:
        base=t[(t.stratum==st)&(t.method=='vcd')].sort_values('coverage_common')
        if base.empty:continue
        ax.plot(base.coverage_common,100*base.risk_direct,color=M_COLORS['direct'],linestyle=ls,
                marker='o',label=f"Direct / {'low' if st=='low_acc' else 'high'}")
        for m in ('vcd','m3id'):
            q=t[(t.stratum==st)&(t.method==m)].sort_values('coverage_common')
            ax.plot(q.coverage_common,100*q.risk_method,color=M_COLORS[m],linestyle=ls,marker='o',
                    label=f"{METHOD_NAMES[m]} / {'low' if st=='low_acc' else 'high'}")
    ax.set_xlabel('Coverage within the common-answer set');ax.set_ylabel('Error rate among selected answers (%)')
    ax.set_xlim(.18,1.02);ax.set_ylim(0,100);ax.legend(frameon=False,ncol=2,fontsize=7.6)
    ax.set_title(f'Selective answering: {MODEL_NAMES[model]}',loc='left',pad=15)
    fig.subplots_adjust(left=.12,right=.98,bottom=.17,top=.87)
    save(fig,out,('SIMULATED FIXTURE - NOT KDM RESULTS  |  Plot and tie-handling verification only' if fixture else
                  'REAL DATA  |  Common non-abstained answers; total-input coverage is reported in the accompanying CSV'))


def evidence_curve(data:Path,out:Path,model='q9b',fixture=False):
    t=pd.read_csv(data);t=t[t.model==model]
    fig,ax=plt.subplots(figsize=(6.8,4.5));order=['original','short64','short32']
    for st,ls in [('low_acc','-'),('high_acc','--')]:
        for m in ('direct','vcd','m3id'):
            q=t[(t.stratum==st)&(t.method==m)].set_index('condition')
            if any(c not in q.index for c in order):raise ValueError('Incomplete evidence conditions.')
            ax.plot(range(3),[100*q.loc[c,'accuracy'] for c in order],color=M_COLORS[m],linestyle=ls,marker='o',
                label=f"{METHOD_NAMES[m]} / {'low' if st=='low_acc' else 'high'}")
    ax.set_xticks(range(3),['Original','64-pixel short side','32-pixel short side'])
    ax.set_ylabel('Naming accuracy (%)');ax.set_ylim(0,100)
    ax.set_title('Change visual evidence while keeping image identities fixed',loc='left',pad=15)
    ax.legend(frameon=False,ncol=2,fontsize=7.6);fig.subplots_adjust(left=.12,right=.98,bottom=.17,top=.87)
    save(fig,out,('SIMULATED FIXTURE - NOT KDM RESULTS  |  End-to-end schema verification only' if fixture else
                  'REAL DATA  |  Same images and category assignment across conditions; no label hints are added'))



def observed_reachability(data:Path,out:Path,fixture=False):
    t=pd.read_csv(data)
    categories=['at_least_one_name_reachable','every_name_excluded_by_support',
                'remaining_score_constraints_incompatible','unstable_across_seeds',
                'outside_sequence_budget']
    labels=['A specified name is reachable','Each name excluded by support',
            'Other score constraints incompatible','Unstable across noise seeds',
            'Outside length scope']
    colors=[COLORS['positive'],COLORS['negative'],COLORS['blue'],COLORS['muted'],COLORS['line']]
    groups=list(t.groupby(['model','condition'],sort=True))
    fig,ax=plt.subplots(figsize=(7.2,3.7));yt=[]
    for i,((model,condition),g) in enumerate(groups):
        left=0;yt.append(f"{MODEL_NAMES[model]} / {'original' if condition=='original' else '32-pixel'}")
        for cat,label,col in zip(categories,labels,colors):
            val=100*(g.category==cat).mean()
            ax.barh(i,val,left=left,height=.5,color=col,label=label if i==0 else None)
            if val>=9:ax.text(left+val/2,i,f'{val:.0f}%',ha='center',va='center',fontsize=8,color='white' if col!=COLORS['line'] else COLORS['ink'])
            left+=val
        ax.text(101.5,i,f'n={len(g)}',ha='left',va='center',fontsize=8)
    ax.set_yticks(range(len(yt)),yt);ax.invert_yaxis();ax.set_xlim(0,112)
    ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Percentage of evaluated image-name cases')
    ax.set_title('Reachability under a fixed visual reference',loc='left',pad=12)
    ax.legend(loc='lower left',bbox_to_anchor=(-.40,-.43),frameon=False,ncol=2,fontsize=7.4)
    fig.subplots_adjust(left=.29,right=.97,bottom=.29,top=.84)
    save(fig,out,('SIMULATED FIXTURE - NOT KDM RESULTS  |  Two frozen spellings; not a test of semantic knowledge' if fixture else
                  'REAL DATA  |  Reachability of two specified name spellings, including prescribed EOS; not all acceptable answers'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,default=ROOT)
    ap.add_argument('--risk-csv',type=Path);ap.add_argument('--evidence-csv',type=Path);ap.add_argument('--reachability-csv',type=Path)
    ap.add_argument('--fixture',action='store_true');a=ap.parse_args();setup();p=a.package
    error_exchange(p/'data_sample/stage6_core_selected.csv',p/'figures/real/01_error_exchange')
    confidence_tradeoff(p/'data_sample/stage6_core_selected.csv',p/'figures/real/02_confidence_tradeoff')
    offset_reversal(p/'data_sample/correction_selected.csv',p/'figures/real/03_offset_reversal')
    reachable_intervals(p/'figures/illustrative/04_reachable_intervals')
    if a.risk_csv:risk_curves(a.risk_csv,p/'figures/illustrative/05_risk_coverage' if a.fixture else p/'figures/real/05_risk_coverage',fixture=a.fixture)
    if a.evidence_csv:evidence_curve(a.evidence_csv,p/'figures/illustrative/06_evidence_control' if a.fixture else p/'figures/real/06_evidence_control',fixture=a.fixture)
    if a.reachability_csv:observed_reachability(a.reachability_csv,p/'figures/illustrative/07_name_reachability' if a.fixture else p/'figures/real/07_name_reachability',fixture=a.fixture)
    print('Figure export complete.')

if __name__=='__main__':main()
