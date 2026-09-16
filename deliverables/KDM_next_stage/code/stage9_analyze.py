"""Paired analysis of existing stage-6 JSONL caches; never opens an NPZ file."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from stage9_common import (ROOT,MODELS,METHODS,initialize,inside,load_records,confidence,
                          transition_counts,risk_at_coverage,cluster_bootstrap_pair,write_json)


def index_records(path:Path) -> dict:
    result={}
    for r in load_records(path).values():
        if r.get('task','naming')!='naming' or r['method'] not in ('direct',)+METHODS:continue
        k=(r['file'],r['method'])
        if k in result:raise ValueError(f'Duplicate paired key {k}')
        result[k]=r
    return result


def analyze_pair(drows,mrows,all_common,score_kind,replicates=2000):
    ds={r['file']:r for r in drows};ms={r['file']:r for r in mrows}
    if set(ds)!=set(ms):raise ValueError('Missing paired outcomes; no silent intersection.')
    files=sorted(ds);d=[ds[f]['outcome']=='correct' for f in files];m=[ms[f]['outcome']=='correct' for f in files]
    result=transition_counts(d,m)
    vals=np.c_[d,m].astype(float);clusters=[ds[f]['class'] for f in files]
    lo,hi=cluster_bootstrap_pair(vals,clusters,lambda v:np.mean(v[:,1]-v[:,0]),replicates)
    result.update(delta_accuracy_lo=lo,delta_accuracy_hi=hi)
    eligible=[f for f in files if f in all_common]
    result['n_common_answers']=len(eligible)
    curve=[]
    if len(eligible)<20:
        result.update(risk80_direct=None,risk80_method=None,delta_risk80=None,
                      delta_risk80_lo=None,delta_risk80_hi=None,decision='insufficient_common_answers')
        return result,curve
    sd=[confidence(ds[f],score_kind) for f in eligible];sm=[confidence(ms[f],score_kind) for f in eligible]
    yd=[ds[f]['outcome']=='correct' for f in eligible];ym=[ms[f]['outcome']=='correct' for f in eligible]
    vals=np.c_[sd,sm,yd,ym];groups=[ds[f]['class'] for f in eligible]
    for c in (.2,.4,.6,.8,1.):
        rd=risk_at_coverage(sd,yd,c);rm=risk_at_coverage(sm,ym,c)
        actual_count=max(1,int(np.ceil(c*len(eligible))))
        curve.append(dict(coverage_common=c,coverage_common_actual=actual_count/len(eligible),
                          coverage_all=actual_count/len(files),risk_direct=rd,risk_method=rm))
    rd=risk_at_coverage(sd,yd,.8);rm=risk_at_coverage(sm,ym,.8)
    stat=lambda v:risk_at_coverage(v[:,1],v[:,3],.8)-risk_at_coverage(v[:,0],v[:,2],.8)
    lo,hi=cluster_bootstrap_pair(vals,groups,stat,replicates)
    result.update(risk80_direct=rd,risk80_method=rm,delta_risk80=rm-rd,
                  delta_risk80_lo=lo,delta_risk80_hi=hi,
                  decision=('higher_fixed_coverage_risk' if rm-rd>=.02 and lo>0 else 'no_confirmed_harm_at_primary_budget'))
    return result,curve


def run(root:Path,out:Path,models=MODELS,replicates=2000):
    out=inside(root,out);out.mkdir(parents=True,exist_ok=True)
    rows=[];curves=[];audit={}
    for model in models:
        path=root/'outputs/raw'/f'{model}_s6_eval_naming.jsonl'
        recs=index_records(path)
        files=sorted(f for f,m in recs if m=='direct')
        if not files:raise ValueError(f'No direct records in {path}')
        for f in files:
            for m in METHODS:
                if (f,m) not in recs:raise ValueError(f'Missing {model}/{f}/{m}')
        audit[model]={'n_inputs':len(files),'source':str(path.relative_to(root))}
        for kind in ('sequence','first'):
            common={f for f in files if all(confidence(recs[f,m],kind) is not None for m in ('direct',)+METHODS)}
            for st in ('low_acc','high_acc'):
                fs=[f for f in files if recs[f,'direct']['stratum']==st]
                if not fs:raise ValueError(f'Missing stratum {st} in {model}')
                dr=[recs[f,'direct'] for f in fs]
                for method in METHODS:
                    result,curve=analyze_pair(dr,[recs[f,method] for f in fs],common,kind,replicates)
                    meta=dict(model=model,stratum=st,method=method,score_kind=kind)
                    rows.append({**meta,**result});curves.extend({**meta,**c} for c in curve)
    for name,table in [('paired_reliability.csv',rows),('risk_coverage.csv',curves)]:
        if table:
            with (out/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    write_json(out/'cache_analysis_audit.json',{'sources':audit,'replicates':replicates,
        'primary_score':'sequence log probability including the recorded termination token',
        'population':'common non-abstained answers across all five configurations; total-input coverage reported separately',
        'novel_inference':False})
    return rows,curves


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--out',type=Path);ap.add_argument('--models',default=','.join(MODELS))
    ap.add_argument('--replicates',type=int,default=2000);args=ap.parse_args()
    root=initialize(args.root)
    out=args.out or root/'outputs/tables/stage9'
    rows,_=run(root,out,tuple(args.models.split(',')),args.replicates)
    print(json.dumps({'combinations':len(rows),'output':str(out)},ensure_ascii=False))

if __name__=='__main__':main()
