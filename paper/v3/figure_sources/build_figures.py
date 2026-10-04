"""Three data figures. ImageGen illustrations use compose_imagegen_figures.py.

All chart geometry derives from source CSV values. Physical print width is
174 mm; fonts are embedded in PDF and converted to paths in the portable SVG.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

ROOT=Path(__file__).resolve().parents[1]
ST,OUT=ROOT/'statistics',ROOT/'figures'
EASY=Path(os.environ.get('EASYPLOT_ROOT',str(Path.home()/'.codex/skills/easyplot')))
sys.path.insert(0,str(EASY/'scripts'))
from easyplot_py import publication_context, export_figure

BLUE,TEAL,RED,GOLD='#3569B2','#258C83','#C45A51','#C69232'
INK,GRAY,GRID='#183247','#66798A','#E5EBF0'
MODEL_IDS=['qwen25vl','llava16_mistral','minicpm26','internvl35_8b','qwen3vl','qwen35_4b','gemma3_4b','onevision','phi35']
NAMES=['Qwen2.5-VL','LLaVA-Mistral','MiniCPM-V2.6','InternVL3.5','Qwen3-VL','Qwen3.5','Gemma-3','OneVision','Phi-3.5-Vision']
SOURCES=set(); MANIFEST={}

def read(name):
    SOURCES.add(ST/name)
    return pd.read_csv(ST/name)

def save(fig,name,claim):
    MANIFEST[name]=export_figure(fig,OUT/name,formats=('pdf','svg','png'),dpi=360,overwrite=True,
        provenance={'description':claim,'sources':[str(p.relative_to(ROOT)) for p in sorted(SOURCES)]})
    plt.close(fig)

def title(ax,letter,label):
    ax.text(-.04,1.15,letter,transform=ax.transAxes,size=9.5,weight='bold',va='bottom',color=INK)
    ax.text(.04,1.15,label,transform=ax.transAxes,size=8,weight='bold',va='bottom',color=INK)

def clean(ax,grid='x'):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#A9B7C3')
    ax.spines[['left','bottom']].set_linewidth(.55)
    ax.tick_params(length=2.2,width=.5,color=GRAY,labelsize=7)
    if grid: ax.grid(axis=grid,color=GRID,lw=.55)
    ax.set_axisbelow(True)

def mechanism():
    flow=read('llava_archived_original_reasonable_abstention_flow.csv')
    cm=read('qwen25_cross_prompt_4x4.csv')
    margins=read('carpaccio_four_view_margins.csv').set_index('view')
    values=read('carpaccio_counterfactual_math.csv').iloc[0]
    f=plt.figure(figsize=(174/25.4,4.0))
    gs=f.add_gridspec(2,2,left=.13,right=.96,bottom=.10,top=.9,wspace=.4,hspace=.72)
    axes=[f.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
    a=axes[0];title(a,'a','Where supported abstentions go')
    colors=[BLUE,TEAL,RED]
    for y,(_,r) in zip([1,0],flow.iterrows()):
        start=0
        for key,col in zip(['retained_abstention','correct_answer','wrong_answer'],colors):
            n=int(r[key]);a.barh(y,n,left=start,height=.36,color=col,edgecolor='white',lw=.65)
            a.text(start+n/2,y,str(n),ha='center',va='center',color='white',size=7)
            start+=n
        assert start==287
    a.set(xlim=(0,287),ylim=(-.48,1.95),xlabel='Inputs (same 287 cases)')
    a.set_yticks([1,0],['Guided VCD','IP-VCD']);a.set_xticks([0,100,200,287]);clean(a)
    a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0,pad=5)
    a.legend(handles=[Rectangle((0,0),1,1,color=c,label=l) for c,l in zip(colors,['Abstain','Correct','Wrong'])],
        loc='upper left',bbox_to_anchor=(-.03,1.07),ncol=3,frameon=False,fontsize=6.7,handlelength=.9,columnspacing=.65)
    a=axes[1];title(a,'b','Sensitivity to the reference instruction')
    order=['UNKNOWN','UNCLEAR','UNSURE','I cannot identify it']
    matrix=100*cm.pivot(index='marker',columns='reference_marker',values='abstention_rate').reindex(index=order,columns=order).to_numpy()
    cmap=LinearSegmentedColormap.from_list('reference',['#F1F6FB','#8AAED5','#254B7D'])
    a.imshow(matrix,cmap=cmap,vmin=0,vmax=100,aspect='auto')
    for i in range(4):
        for j in range(4):
            a.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',size=7.4,color='white' if matrix[i,j]>59 else INK)
    a.set_xticks(range(4),['U','C','S','I']);a.set_yticks(range(4),['U','C','S','I'])
    a.set(xlabel='Reference instruction',ylabel='Main instruction');a.tick_params(length=0)
    for s in a.spines.values():s.set_visible(False)
    a.set_xticks(np.arange(-.5,4),minor=True);a.set_yticks(np.arange(-.5,4),minor=True)
    a.grid(which='minor',color='white',lw=.7);a.tick_params(which='minor',length=0)
    a.text(.5,-.33,'Semantic abstention rate (%)',ha='center',transform=a.transAxes,size=7,color=GRAY)
    a=axes[2];title(a,'c','Unequal instruction responses')
    val={v:float(margins.loc[v,'abstention_minus_answer_log_odds']) for v in ['c','r','g','h']}
    for x,start,end,col,delta in [(0,'c','g',BLUE,values.delta_c),(1,'r','h',TEAL,values.delta_r)]:
        a.annotate('',(x,val[end]),(x,val[start]),arrowprops={'arrowstyle':'-|>','color':col,'lw':2.1,'mutation_scale':10})
        for key in (start,end):
            a.scatter(x,val[key],s=21,color=col,zorder=3)
            a.text(x+.075,val[key],f'{key}  {val[key]:.2f}',va='center',size=7,color=GRAY)
        symbol='c' if start=='c' else 'r'
        a.text(x-.07,-5.0,r'$\Delta_'+symbol+'$'+f' = {delta:.2f}',ha='right',size=7,color=col,rotation=90,va='center')
    a.axhline(0,color='#8B9AA8',lw=.7)
    a.set(xlim=(-.38,1.53),ylim=(-15,7.3),ylabel='Log odds: UN / To')
    a.set_xticks([0,1],['Clean image','Reference image']);clean(a,'y')
    a=axes[3];title(a,'d','Reference interaction flips the decision')
    eta=np.linspace(0,1,101);curve=values.IP_VCD_margin+eta*(values.delta_c-values.delta_r)
    a.axhspan(0,.55,color='#EDF6F3');a.axhspan(-.30,0,color='#FCF0EE')
    a.axhline(0,color='#718291',lw=.8);a.plot(eta,curve,color=BLUE,lw=1.9)
    a.axvline(values.eta_crossing,color=GOLD,ls=(0,(3,2)),lw=1.1)
    a.scatter([0,1],[curve[0],curve[-1]],color=[TEAL,RED],s=25,zorder=3)
    a.text(.04,.46,'IP-VCD',size=7.2,color=TEAL)
    a.text(.98,-.269,'Guided VCD',size=7.2,color=RED,ha='right')
    a.text(.72,.30,r'$\eta^*=9/13$',size=7.5,color='#967023')
    a.set(xlim=(-.05,1.05),ylim=(-.30,.55),xlabel=r'Reference interaction $\eta$',ylabel='Abstention margin')
    clean(a,None)
    save(f,'fig02_mechanism_four_panel','Actual behaviour, prompt matrix, four-view log odds and analytic interpolation; no generated chart values.')

def joint_heatmap(dataset='food'):
    d=read('food_nine_models_best_observed_J.csv' if dataset=='food' else 'vizwiz_nine_models_recorded_working_points.csv')
    if dataset=='vizwiz':assert len(d)==68 and d.n.eq(512).all()
    methods=['direct','vcd','dola','deco','sid','cda_visual','instruction_vcd','instruction_m3id']
    labels=['Direct','VCD','DoLa','DeCo','SID','CDA','IP-VCD','IP-M3ID']
    m=100*d.pivot(index='model',columns='method',values='J').reindex(index=MODEL_IDS,columns=methods).to_numpy()
    f=plt.figure(figsize=(174/25.4,3.25));a=f.add_axes([.205,.17,.69,.73])
    cmap=LinearSegmentedColormap.from_list('joint',['#B45048','#E5A99B','#FBF7ED','#B8CFDD','#326387','#12394F'])
    vmax=65 if dataset=='food' else 80
    cmap.set_bad('#F1F3F5');im=a.imshow(np.ma.masked_invalid(m),cmap=cmap,vmin=0,vmax=vmax,aspect='auto')
    for i in range(9):
        high=np.nanmax(m[i])
        for j,v in enumerate(m[i]):
            col=GRAY if np.isnan(v) else ('white' if v<.14*vmax or v>.69*vmax else INK)
            a.text(j,i,'n/a' if np.isnan(v) else f'{v:.1f}',ha='center',va='center',size=8,
                color=col,weight='bold' if np.isfinite(v) and np.isclose(v,high,atol=1e-9) else 'normal')
    a.set_yticks(range(9),NAMES);a.set_xticks(range(8),labels,rotation=25,ha='right',rotation_mode='anchor')
    a.tick_params(length=0,labelsize=7.7,pad=5)
    a.set_xticks(np.arange(-.5,8),minor=True);a.set_yticks(np.arange(-.5,9),minor=True)
    a.grid(which='minor',color='white',lw=1);a.tick_params(which='minor',length=0)
    for x in [5.5,6.5]:a.add_patch(Rectangle((x,-.5),1,9,fill=False,edgecolor=GOLD,lw=1.65,clip_on=False,zorder=5))
    for s in a.spines.values():s.set_visible(False)
    f.text(.42,.95,'Native baselines',ha='center',size=8,color=GRAY)
    f.text(.68,.95,'CDA',ha='center',size=8,color=GRAY)
    f.text(.81,.95,'IP variants',ha='center',size=8,color='#967023',weight='bold')
    cb=f.colorbar(im,cax=f.add_axes([.921,.19,.013,.69]));cb.set_label('Joint utility J (%)',size=7.5)
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6.8,length=2)
    if dataset=='food':
        save(f,'fig04_food_nine_model_J_heatmap','Nine-model same-input joint utility; IP observed maxima; native baseline working points, unavailable architecture cells explicit.')
    else:
        save(f,'figA01_vizwiz_all_baselines_J','All 68 actual fixed512 VizWiz operating points; native DoLa and DeCo for nine models, original SID for five applicable models; no imputed cells.')

def forest():
    ci=read('vizwiz_paired_J_intervals.csv');assert len(ci)==18 and (ci.n==512).all()
    f,a=plt.subplots(figsize=(83/25.4,3.95));f.subplots_adjust(left=.31,right=.975,bottom=.16,top=.88)
    for i in range(9):
        if i%2==0:a.axhspan(i-.46,i+.46,color='#F4F7F9',zorder=0)
    a.axvline(0,color='#617585',ls=(0,(4,3)),lw=.9)
    for method,col,marker,offset,label in [('instruction_vcd',BLUE,'o',-.15,'IP-VCD'),('instruction_m3id',TEAL,'D',.15,'IP-M3ID')]:
        q=ci[ci.method.eq(method)].set_index('model').loc[MODEL_IDS]
        x=q.delta_J_pp.to_numpy();lo=q.ci95_lo_pp.to_numpy();hi=q.ci95_hi_pp.to_numpy()
        assert np.all(lo<=x) and np.all(hi>=x)
        a.errorbar(x,np.arange(9)+offset,xerr=np.stack([x-lo,hi-x]),fmt=marker,color=col,
            ms=3.6,capsize=2,capthick=.85,elinewidth=1.05,label=label,zorder=3)
    a.set_yticks(range(9),NAMES);a.set_ylim(8.55,-.55);a.set_xlim(-1.5,30.5)
    a.set_xticks([0,10,20,30]);a.set_xlabel('Gain over native VCD\nJoint utility J (pp)',labelpad=7)
    clean(a);a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0,labelsize=7.5,pad=4)
    a.legend(loc='lower center',bbox_to_anchor=(.33,1.03),ncol=2,frameon=False,handlelength=1.1,columnspacing=.75,fontsize=7.5)
    save(f,'fig05_vizwiz_paired_J_forest','Paired512-image J differences, 2000-resample95 percent percentile intervals; original values and zero baseline preserved.')

def main():
    with publication_context(base_size=8,font_family='Arial') as fonts:
        plt.rcParams.update({'text.color':INK,'axes.labelcolor':INK,'mathtext.fontset':'dejavusans',
            'pdf.fonttype':42,'svg.fonttype':'path','axes.unicode_minus':True,'axes.labelsize':7.5})
        mechanism();joint_heatmap();forest();joint_heatmap('vizwiz')
    (ROOT/'figure_sources/figure_manifest.json').write_text(json.dumps({'figures':MANIFEST,'fonts':fonts,
        'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}},
        ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    print('Rendered three main data figures and complete VizWiz appendix heatmap; ImageGen illustrations are separate PDF compositions.')

if __name__=='__main__':main()
