"""Render exact condition plots and compact instruction-versus-base summaries."""
from __future__ import annotations
import argparse,csv,hashlib,json,html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DIR=ROOT/'outputs/analysis/main_results'
IN=DEFAULT_DIR
OUT=DEFAULT_DIR
COND=['model','method','kind','marker','reference_marker','guided','reference_guided','replicate']
COLORS=['#3b82f6','#ef4444','#10b981','#f59e0b','#8b5cf6','#06b6d4','#f97316','#64748b','#ec4899','#84cc16']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def display_path(p):
 p=Path(p)
 try:return str(p.relative_to(ROOT))
 except ValueError:return str(p)
def csvrows(p):
 with Path(p).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def xml(s):return html.escape(str(s),quote=True)
def main(input_dir=DEFAULT_DIR,out_dir=DEFAULT_DIR):
 global IN,OUT
 IN=Path(input_dir);OUT=Path(out_dir);OUT.mkdir(parents=True,exist_ok=True)
 metrics=csvrows(IN/'condition_metrics.csv')
 pairs=[json.loads(x) for x in (IN/'paired_comparisons.jsonl').read_text(encoding='utf8').splitlines() if x]
 if len(metrics)!=352 or any(int(r['n'])!=2424 for r in metrics):raise ValueError('condition table coverage mismatch')
 if len(pairs)!=372:raise ValueError(f'expected 332 direct plus 40 instruction/base pairs, got {len(pairs)}')
 figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
 figs=[];pngs=[];pdfs=[]
 for model in sorted({r['model'] for r in metrics}):
  rs=sorted([r for r in metrics if r['model']==model],key=lambda r:tuple(str(r[k]) for k in COND[1:]))
  n=len(rs);left=80;top=90;barw=13;gap=5;plot_h=390;plot_top=100;chart_w=left+right if False else left+n*(barw+gap)+30
  chart_h=900
  methods=sorted({r['method'] for r in rs});colors={m:COLORS[i%len(COLORS)] for i,m in enumerate(methods)}
  out=['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">'%(chart_w,chart_h,chart_w,chart_h),'<rect width="100%" height="100%" fill="#ffffff"/>']
  out.append(f'<text x="{left}" y="34" font-size="22" font-family="sans-serif" font-weight="bold">{xml(model)}: canonical primary-name accuracy by exact condition</text>')
  out.append('<text x="%d" y="58" font-size="12" font-family="sans-serif">Each bar is one of the exact 8-key condition cells (no pooling); denominator 2,424. Hover for full keys and counts.</text>'%left)
  for t in [0,.25,.5,.75,1]:
   y=plot_top+plot_h*(1-t);out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{chart_w-20}" y2="{y:.1f}" stroke="#d1d5db" stroke-width="1"/>');out.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">{t:.2f}</text>')
  for i,r in enumerate(rs):
   x=left+i*(barw+gap);v=float(r['canonical_accuracy_lower']);h=plot_h*v;y=plot_top+plot_h-h;color=colors[r['method']]
   tip='; '.join(f'{k}={r[k]}' for k in COND)+f"; n={r['n']}; correct={r['canonical_correct']}; unknown={r['canonical_unknown']}; accuracy={v:.6f}"
   out.append(f'<rect x="{x}" y="{y:.1f}" width="{barw}" height="{h:.1f}" fill="{color}"><title>{xml(tip)}</title></rect>')
   label=f"{r['method']}|{r['kind']}|{r['marker']}:{r['reference_marker']}|g{r['guided']}{r['reference_guided']}|r{r['replicate']}"
   out.append(f'<text transform="translate({x+barw/2},{plot_top+plot_h+16}) rotate(60)" font-size="8" font-family="sans-serif">{xml(label)}</text>')
  out.append(f'<text x="20" y="{plot_top+plot_h/2}" transform="rotate(-90 20,{plot_top+plot_h/2})" font-size="12" font-family="sans-serif">Canonical primary-name accuracy</text>')
  yleg=chart_h-25;xleg=left
  for m in methods:
   out.append(f'<rect x="{xleg}" y="{yleg-12}" width="11" height="11" fill="{colors[m]}"/><text x="{xleg+15}" y="{yleg-2}" font-size="10" font-family="sans-serif">{xml(m)}</text>');xleg+=max(80,24+len(m)*6)
  out.append('</svg>')
  fp=figdir/f'{model}_exact_conditions.svg';fp.write_text('\n'.join(out)+'\n',encoding='utf8');figs.append(fp)
  fig,ax=plt.subplots(figsize=(max(18,n*0.26),8))
  ax.bar(range(n),[float(r['canonical_accuracy_lower']) for r in rs],color=[colors[r['method']] for r in rs])
  labels=[f"{r['method']}|{r['kind']}|{r['marker']}:{r['reference_marker']}|g{r['guided']}{r['reference_guided']}|r{r['replicate']}" for r in rs]
  ax.set_xticks(range(n),labels,rotation=68,ha='right',fontsize=6)
  ax.set_ylim(0,1);ax.set_ylabel('Canonical primary-name accuracy');ax.set_title(f'{model}: exact eight-key conditions (n=2,424 each)')
  handles=[plt.Rectangle((0,0),1,1,color=colors[m]) for m in methods];ax.legend(handles,methods,loc='upper left',bbox_to_anchor=(1.01,1),fontsize=7)
  fig.tight_layout();png=figdir/f'{model}_exact_conditions.png';pdf=figdir/f'{model}_exact_conditions.pdf';fig.savefig(png,dpi=180);fig.savefig(pdf);plt.close(fig);pngs.append(png);pdfs.append(pdf)
 # Exact UNKNOWN/UNKNOWN main and instruction-preserving cells retain all condition keys.
 uu=[r for r in metrics if r['kind'] in {'main','instruction_preserving'} and r['marker']=='UNKNOWN' and r['reference_marker']=='UNKNOWN']
 if len(uu)!=42:raise ValueError(f'expected 42 descriptive UNKNOWN/UNKNOWN rows, got {len(uu)}')
 with (OUT/'unknown_unknown_full_conditions.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(uu[0]));w.writeheader();w.writerows(uu)
 # Same direct-paired UNKNOWN/UNKNOWN method summary; no averaging across guided/replicate.
 uu_pairs=[d for d in pairs if d['comparison']=='same_main_prompt_direct' and d['context'].get('kind') in {'main','instruction_preserving'} and d['context'].get('marker')=='UNKNOWN' and d['context'].get('reference_marker')=='UNKNOWN']
 with (OUT/'unknown_unknown_paired_methods.jsonl').open('w',encoding='utf8',newline='\n') as f:
  for d in uu_pairs:f.write(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n')
 inst=[d for d in pairs if d['comparison']=='instruction_vs_base_method']
 if len(inst)!=40:raise ValueError(f'expected 40 instruction/base paired cells, got {len(inst)}')
 fields=['model','method_a','method_b']+[k for k in COND if k not in {'method','model'}]+['n_paired','noninferiority_at_minus_0p01','accuracy_difference_lower_ci','accuracy_difference_upper_ci','reference_preservation_numerator','reference_preservation_denominator','reference_preservation_ci_lower','reference_preservation_ci_upper','reference_preservation_lower_gt_zero','uniform_preservation_numerator','uniform_preservation_denominator','uniform_preservation_ci_lower','uniform_preservation_ci_upper','uniform_preservation_lower_gt_zero','specific_answer_correction_retention','all_input_correction_retention','main_prompt_sha_mismatch_count','seed_mismatch_count']
 with (OUT/'instruction_vs_base_summary.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for d in inst:
   c=d['context'];lo=d['canonical_accuracy_difference_conservative_lower'];hi=d['canonical_accuracy_difference_conservative_upper']
   row={'model':c['model'],'method_a':d['method_a'],'method_b':d['method_b'],'n_paired':d['n_paired'],'noninferiority_at_minus_0p01':d['noninferiority_at_minus_0p01'],'accuracy_difference_lower_ci':lo['ci95'][0] if lo else None,'accuracy_difference_upper_ci':hi['ci95'][1] if hi else None,'reference_preservation_numerator':d['direct_reasonable_abstention_set_preservation_difference_reference_gt']['numerator'],'reference_preservation_denominator':d['direct_reasonable_abstention_set_preservation_difference_reference_gt']['denominator'],'reference_preservation_ci_lower':d['direct_reasonable_abstention_set_preservation_difference_reference_gt']['ci95'][0],'reference_preservation_ci_upper':d['direct_reasonable_abstention_set_preservation_difference_reference_gt']['ci95'][1],'reference_preservation_lower_gt_zero':d['direct_reasonable_abstention_set_preservation_difference_reference_gt']['ci95'][0] is not None and d['direct_reasonable_abstention_set_preservation_difference_reference_gt']['ci95'][0]>0,'uniform_preservation_numerator':d['direct_reasonable_abstention_set_preservation_difference_uniform_gt']['numerator'],'uniform_preservation_denominator':d['direct_reasonable_abstention_set_preservation_difference_uniform_gt']['denominator'],'uniform_preservation_ci_lower':d['direct_reasonable_abstention_set_preservation_difference_uniform_gt']['ci95'][0],'uniform_preservation_ci_upper':d['direct_reasonable_abstention_set_preservation_difference_uniform_gt']['ci95'][1],'uniform_preservation_lower_gt_zero':d['direct_reasonable_abstention_set_preservation_difference_uniform_gt']['ci95'][0] is not None and d['direct_reasonable_abstention_set_preservation_difference_uniform_gt']['ci95'][0]>0,'specific_answer_correction_retention':d['specific_answer_correction_retention'],'all_input_correction_retention':d['all_input_correction_retention'],'main_prompt_sha_mismatch_count':d['main_prompt_sha_mismatch_count'],'seed_mismatch_count':d['seed_mismatch_count']}
   row.update({k:c.get(k) for k in fields if k in c});w.writerow(row)
 ni={i for i,d in enumerate(inst) if d['noninferiority_at_minus_0p01']}
 ref_improved={i for i,d in enumerate(inst) if d['direct_reasonable_abstention_set_preservation_difference_reference_gt']['ci95'][0] is not None and d['direct_reasonable_abstention_set_preservation_difference_reference_gt']['ci95'][0]>0}
 uniform_improved={i for i,d in enumerate(inst) if d['direct_reasonable_abstention_set_preservation_difference_uniform_gt']['ci95'][0] is not None and d['direct_reasonable_abstention_set_preservation_difference_uniform_gt']['ci95'][0]>0}
 # Publication figures: one paired tradeoff panel per GT policy, plus two descriptive heatmaps.
 models=sorted({d['context']['model'] for d in inst})
 marker_colors={'UNKNOWN':'#1f77b4','UNCLEAR':'#d62728','UNSURE':'#2ca02c','I cannot identify it':'#9467bd'}
 method_markers={'instruction_vcd':'o','instruction_m3id':'s'}
 from matplotlib.lines import Line2D
 for gt_label,preserve_key,suffix in (('Accepted reference','direct_reasonable_abstention_set_preservation_difference_reference_gt','accepted_gt'),('Uniform reference','direct_reasonable_abstention_set_preservation_difference_uniform_gt','uniform_gt')):
  fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True,sharey=True);flat=axes.ravel()
  for col,model in enumerate(models):
   ax=flat[col];subset=[d for d in inst if d['context']['model']==model]
   for d in subset:
    c=d['context'];x=d['resolved_accuracy_difference_a_minus_b'];ac=d['canonical_accuracy_difference_conservative_lower']['ci95'];yobj=d[preserve_key];y=yobj['estimate']
    if x is None or y is None:continue
    xerr=[[max(0,x-ac[0])*100],[max(0,ac[1]-x)*100]];yc=yobj['ci95'];yerr=[[max(0,y-yc[0])*100],[max(0,yc[1]-y)*100]]
    ax.errorbar([x*100],[y*100],xerr=xerr,yerr=yerr,fmt=method_markers.get(d['method_a'],'o'),color=marker_colors.get(c['marker'],'#444444'),markersize=5,capsize=2,alpha=.9)
   if model=='gemma3_4b':
    ns=[d[preserve_key]['denominator'] for d in subset]
    ax.text(.02,.98,'direct∩GT+ n by cell: '+', '.join(map(str,ns)),transform=ax.transAxes,va='top',fontsize=5)
   ax.axhline(0,color='#777777',lw=.6);ax.axvline(0,color='#777777',lw=.6);ax.axvline(-1,color='#777777',lw=.6,ls='--')
   ax.set_xlim(-7,8);ax.set_ylim(-65,65);ax.grid(alpha=.18);ax.set_title(model,fontsize=8)
   if col==0:ax.set_ylabel('Abstention preservation difference (pp)')
   if col>=2:ax.set_xlabel('Accuracy difference (pp)')
  flat[5].axis('off')
  legend=[Line2D([0],[0],marker=method_markers[m],color='black',linestyle='None',label=('I-VCD' if m=='instruction_vcd' else 'I-M3ID'),markersize=5) for m in method_markers]
  legend += [Line2D([0],[0],marker='o',color=v,linestyle='None',label=k,markersize=5) for k,v in marker_colors.items()]
  flat[5].legend(handles=legend,loc='center',frameon=False,title=gt_label+'\nMethod / matched marker')
  fig.suptitle('Instruction-versus-base paired differences with 95% class-cluster intervals',fontsize=10);fig.tight_layout(rect=[0,0,1,.95])
  fp=OUT/'figures'/f'instruction_base_tradeoff_{suffix}';fig.savefig(fp.with_suffix('.png'),dpi=220);fig.savefig(fp.with_suffix('.pdf'));plt.close(fig);pngs.append(fp.with_suffix('.png'));pdfs.append(fp.with_suffix('.pdf'))
 # 9 methods by 5 models, with registered/unavailable cells shown as gray dashes.
 method_order=['direct','vcd','m3id','dola','deco','sid','instruction_vcd','instruction_m3id','cda_visual']
 method_labels=['Direct','VCD','M3ID','DoLa','DeCo','SID','I-VCD','I-M3ID','CDA']
 model_order=sorted({r['model'] for r in uu})
 by_method={(r['model'],r['method']):r for r in uu}
 if len(by_method)!=len(uu):raise ValueError('duplicate UNKNOWN/UNKNOWN model-method row')
 for value_key,title,cmap_name,suffix in (('canonical_correct','Accuracy (%)','viridis','accuracy'),('abstain_true','Abstention (%)','magma','abstention')):
  matrix=[];texts=[]
  for method in method_order:
   vals=[];labels=[]
   for model in model_order:
    r=by_method.get((model,method))
    if r is None:vals.append(float('nan'));labels.append('—')
    else:
     val=100*int(r[value_key])/int(r['n']);vals.append(val);labels.append(f'{val:.1f}')
   matrix.append(vals);texts.append(labels)
  cmap=plt.get_cmap(cmap_name).copy();cmap.set_bad('#d9d9d9')
  fig,ax=plt.subplots(figsize=(8.5,4.3));im=ax.imshow(np.ma.masked_invalid(np.asarray(matrix,dtype=float)),aspect='auto',cmap=cmap,vmin=0,vmax=100)
  ax.set_xticks(range(len(model_order)),model_order,rotation=28,ha='right',fontsize=7);ax.set_yticks(range(len(method_order)),method_labels,fontsize=8)
  ax.set_title(f'Descriptive UNKNOWN/UNKNOWN {title.lower()} (42 exact condition cells; n=2,424 each)',fontsize=9)
  for i in range(len(method_order)):
   for j in range(len(model_order)):
    color='white' if texts[i][j]!='—' and float(texts[i][j])<55 else 'black';ax.text(j,i,texts[i][j],ha='center',va='center',fontsize=7,color=color)
  fig.colorbar(im,ax=ax,label=title,fraction=.035,pad=.04);fig.tight_layout()
  fp=OUT/'figures'/f'unknown_unknown_{suffix}_heatmap';fig.savefig(fp.with_suffix('.png'),dpi=220);fig.savefig(fp.with_suffix('.pdf'));plt.close(fig);pngs.append(fp.with_suffix('.png'));pdfs.append(fp.with_suffix('.pdf'))
 ni={i for i,d in enumerate(inst) if d['noninferiority_at_minus_0p01']}
 ref_improved={i for i,d in enumerate(inst) if d['direct_reasonable_abstention_set_preservation_difference_reference_gt']['ci95'][0] is not None and d['direct_reasonable_abstention_set_preservation_difference_reference_gt']['ci95'][0]>0}
 uniform_improved={i for i,d in enumerate(inst) if d['direct_reasonable_abstention_set_preservation_difference_uniform_gt']['ci95'][0] is not None and d['direct_reasonable_abstention_set_preservation_difference_uniform_gt']['ci95'][0]>0}
 summary={'instruction_paired_cells':len(inst),'instruction_cells_noninferior':len(ni),'instruction_cells_retention_improved_reference_gt':len(ref_improved),'instruction_cells_retention_improved_uniform_gt':len(uniform_improved),'instruction_cells_ni_and_retention_improved_reference_gt':len(ni & ref_improved),'instruction_cells_ni_and_retention_improved_uniform_gt':len(ni & uniform_improved),'instruction_rows_by_model_method':{},'all_condition_rows':len(metrics),'direct_paired_comparisons':sum(d['comparison']=='same_main_prompt_direct' for d in pairs),'missing_control_cells':len([1 for d in pairs if d['n_paired']!=2424]),'figures':[display_path(x) for x in [*figs,*pngs,*pdfs]]}
 for d in inst:summary['instruction_rows_by_model_method'][f"{d['context']['model']}|{d['method_a']}"]=summary['instruction_rows_by_model_method'].get(f"{d['context']['model']}|{d['method_a']}",0)+1
 (OUT/'render_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 report=OUT/'report.md';source_report=report if report.exists() else IN/'report.md';text=source_report.read_text(encoding='utf8')
 section='\n## Paired instruction and descriptive-slice results'
 if section in text:text=text.split(section,1)[0].rstrip()+'\n'
 text+=f'\n## Paired instruction and descriptive-slice results\n\nAcross 40 instruction-VCD/VCD and instruction-M3ID/M3ID paired cells, {len(ni)} meet the accuracy noninferiority criterion (paired 95% lower bound above -0.01). Preservation improves (fixed direct-abstain and GT-positive subset; 95% lower bound above zero) in {len(ref_improved)}/40 cells under the accepted three-model reference and {len(uniform_improved)}/40 under the uniform five-model reference; both criteria hold in {len(ni & ref_improved)}/40 and {len(ni & uniform_improved)}/40, respectively. The 40-row `instruction_vs_base_summary.csv` gives each condition’s numerator, denominator, and confidence interval under both references.\n\nThe UNKNOWN/UNKNOWN table and plot are descriptive: `unknown_unknown_full_conditions.csv` preserves 42 exact main and instruction-preserving conditions, with direct-paired results in `unknown_unknown_paired_methods.jsonl`. Per-model exact-condition figures are in SVG, PNG, and PDF; paired-instruction tradeoff is shown under both GT policies, with 95% intervals and marker/method legends; UNKNOWN/UNKNOWN accuracy and abstention heatmaps show the 9-method by 5-model descriptive slice in PNG/PDF.\n'
 report.write_text(text,encoding='utf8')
 inputs=[IN/'condition_metrics.csv',IN/'paired_comparisons.jsonl',IN/'missing_controls.jsonl',IN/'prompt_pair_audit.json']
 outputs=[*figs,*pngs,*pdfs,OUT/'unknown_unknown_full_conditions.csv',OUT/'unknown_unknown_paired_methods.jsonl',OUT/'instruction_vs_base_summary.csv',OUT/'render_summary.json',OUT/'report.md']
 receipt={'schema':'main_results_render_receipt','inputs':{display_path(p):sha(p) for p in inputs},'outputs':{display_path(p):sha(p) for p in outputs},'condition_cells':len(metrics),'unknown_unknown_full_condition_rows':len(uu),'instruction_base_pairs':len(inst),'render_summary':summary}
 rp=OUT/'render_receipt.json';rp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--input-dir',type=Path,default=DEFAULT_DIR,help='analysis tables directory; default is outputs/analysis/main_results')
 ap.add_argument('--out-dir',type=Path,default=DEFAULT_DIR,help='render output directory; default is outputs/analysis/main_results')
 a=ap.parse_args();main(a.input_dir,a.out_dir)
