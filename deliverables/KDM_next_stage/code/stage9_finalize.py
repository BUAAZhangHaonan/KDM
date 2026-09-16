"""Summarize stage-9 evidence interventions and exact-name reachability."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from stage9_common import ROOT,initialize,inside,read_jsonl,transition_counts,cluster_bootstrap_pair,write_json


def finalize(root:Path,replicates=2000):
    folder=root/'outputs/raw/stage9'
    records={};geometry={}
    paths=sorted(folder.glob('*_evidence_*.jsonl'))
    if not paths:raise FileNotFoundError('No stage-9 evidence records.')
    for p in paths:
        for r in read_jsonl(p):
            key=(r['model'],r['condition'],r['file'],r['method'])
            if key in records:raise ValueError(f'Duplicate stage-9 result: {key}')
            records[key]=r
    for p in sorted(folder.glob('*_geometry_*.jsonl')):
        for r in read_jsonl(p):
            key=(r['model'],r['condition'],r['file'])
            geometry.setdefault(key,[]).append(r)
    selected=list(read_jsonl(root/'data/stage9/manifest.jsonl'))
    for s in selected:
        for condition in ('original','short64','short32'):
            for method in ('direct','vcd','m3id'):
                if (s['model'],condition,s['file'],method) not in records:
                    raise RuntimeError(f'Incomplete inference matrix: {s["model"]} {s["file"]} {condition} {method}')
        if s.get('geometry'):
            for condition in ('original','short32'):
                if (s['model'],condition,s['file']) not in geometry:
                    raise RuntimeError(f'Missing geometry result for {s["file"]}')
    output=root/'outputs/tables/stage9';inside(root,output);output.mkdir(parents=True,exist_ok=True)
    curve=[];effect=[];recovery=[];interactions=[]
    for model in sorted({r['model'] for r in records.values()}):
        for st in ('low_acc','high_acc'):
            files=sorted({r['file'] for r in records.values() if r['model']==model and r['stratum']==st})
            if not files:continue
            for condition in ('original','short64','short32'):
                direct=[records[model,condition,f,'direct'] for f in files]
                yd=np.array([r['outcome']=='correct' for r in direct])
                for method in ('direct','vcd','m3id'):
                    mr=[records[model,condition,f,method] for f in files]
                    ym=np.array([r['outcome']=='correct' for r in mr])
                    curve.append(dict(model=model,stratum=st,condition=condition,method=method,n=len(files),accuracy=float(ym.mean())))
                    if method!='direct':
                        result=transition_counts(yd,ym)
                        vals=np.c_[yd,ym].astype(float);groups=[r['class'] for r in direct]
                        lo,hi=cluster_bootstrap_pair(vals,groups,lambda v:np.mean(v[:,1]-v[:,0]),replicates)
                        effect.append(dict(model=model,stratum=st,condition=condition,method=method,**result,delta_lo=lo,delta_hi=hi))
            yo=np.array([records[model,'original',f,'direct']['outcome']=='correct' for f in files])
            yd=np.array([records[model,'short32',f,'direct']['outcome']=='correct' for f in files])
            for method in ('vcd','m3id'):
                ym=np.array([records[model,'short32',f,method]['outcome']=='correct' for f in files])
                mo=np.array([records[model,'original',f,method]['outcome']=='correct' for f in files])
                values=np.c_[yo,mo,yd,ym].astype(float)
                groups=[records[model,'original',f,'direct']['class'] for f in files]
                statistic=lambda v:np.mean((v[:,3]-v[:,2])-(v[:,1]-v[:,0]))
                il,ih=cluster_bootstrap_pair(values,groups,statistic,replicates)
                dl,dh=cluster_bootstrap_pair(values,groups,lambda v:np.mean(v[:,2]-v[:,0]),replicates)
                interactions.append(dict(model=model,stratum=st,method=method,
                    effect_change=float(statistic(values)),effect_change_lo=il,effect_change_hi=ih,
                    direct_evidence_change=float(np.mean(yd.astype(float)-yo.astype(float))),
                    direct_evidence_lo=dl,direct_evidence_hi=dh))
                subset=yo&~yd
                recovery.append(dict(model=model,stratum=st,method=method,n=len(files),
                    evidence_responsive_error_n=int(subset.sum()),
                    corrected_in_that_subset=int(np.sum(subset&ym)),
                    conditional_correction_rate=float(np.mean(ym[subset])) if np.any(subset) else None,
                    original_input_accuracy=float(yo.mean()),degraded_input_accuracy=float(yd.mean()),
                    note='Original-correct/degraded-wrong is a paired observed response category, not latent knowledge.'))
    grow=[]
    for (model,condition,file),paths in sorted(geometry.items()):
        ok=[p for p in paths if p['status']=='ok']
        if not ok:label='outside_sequence_budget'
        elif any(p['feasible_any_nonnegative'] for p in ok):label='at_least_one_name_reachable'
        elif all(any(s['target_interval']['reason']=='target_outside_support' for s in p['steps']) for p in ok):label='every_name_excluded_by_support'
        else:label='remaining_score_constraints_incompatible'
        grow.append(dict(model=model,condition=condition,file=file,category=label,n_paths=len(paths),
                         n_measured=len(ok),any_default=any(p['feasible_default'] for p in ok),
                         any_up_to_two=any(p['feasible_up_to_two'] for p in ok),
                         n_validated=sum(p.get('validation_passed') is True for p in ok)))
    for name,table in [('evidence_accuracy.csv',curve),('evidence_effects.csv',effect),('evidence_recovery.csv',recovery),('evidence_interaction.csv',interactions),('name_reachability.csv',grow)]:
        with (output/name).open('w',newline='') as f:
            if table:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    write_json(output/'completion.json',dict(evidence_records=len(records),name_paths=sum(map(len,geometry.values())),
        selected_model_images=len(selected),complete=True,replicates=replicates,
        target_scope='Two frozen name spellings and one specified EOS per path; not semantic knowledge or all acceptable names.'))
    return curve,effect,grow


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--replicates',type=int,default=2000);a=ap.parse_args()
    curve,_,_=finalize(initialize(a.root),a.replicates);print(f'{len(curve)} evidence-condition summaries written.')
if __name__=='__main__':main()
