"""Supplement 3.1 validity gate for the evidence-degradation manipulation.

Runs BEFORE the formal stage-9 evidence run completes: reads the direct-decode
records of the three image conditions (original reused from stage-6 cache,
short64/short32 from the phase-A run) and decides, per model, whether the
degradation constitutes a valid evidence-strength manipulation:

  invalid if acc(short64) or acc(short32) falls in [0, 0.10], OR any pair of
  the three conditions is indistinguishable on the class-paired 95% bootstrap
  interval of the accuracy difference (interval contains 0).

No degradation parameter may be changed when the gate fails; the conclusion is
reported as-is ("this round cannot test the dataset-difficulty hypothesis").
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from stage9_common import ROOT,initialize,read_jsonl,cluster_bootstrap_pair,write_json

CONDITIONS=('original','short64','short32')
FLOOR=0.10


def load_direct(root:Path):
    records={}
    for p in sorted((root/'outputs/raw/stage9').glob('*_evidence_*.jsonl')):
        for r in read_jsonl(p):
            if r.get('method')!='direct':continue
            key=(r['model'],r['condition'],r['file'])
            if key in records and records[key]!=r:raise ValueError(f'Duplicate direct record: {key}')
            records[key]=r
    return records


def validity(root:Path,replicates=2000):
    records=load_direct(root)
    manifest={ (r['model'],r['file']):r for r in read_jsonl(root/'data/stage9/manifest.jsonl') }
    report={}
    for model in sorted({m for m,_,_ in records}):
        files=sorted(f for (m,c,f) in records if m==model and c=='original')
        missing=[(c,f) for c in CONDITIONS for f in files if (model,c,f) not in records]
        if missing:raise RuntimeError(f'{model}: incomplete direct matrix ({len(missing)} missing, e.g. {missing[:3]})')
        outcome={c:np.array([records[model,c,f]['outcome']=='correct' for f in files]) for c in CONDITIONS}
        groups=[manifest[model,f]['class'] for f in files]
        acc={c:float(outcome[c].mean()) for c in CONDITIONS}
        floor_ok=all(acc[c]>FLOOR for c in ('short64','short32'))
        pairs={}
        for a,b in (('original','short64'),('original','short32'),('short64','short32')):
            vals=np.c_[outcome[a],outcome[b]].astype(float)
            lo,hi=cluster_bootstrap_pair(vals,groups,lambda v:np.mean(v[:,1]-v[:,0]),replicates)
            pairs[f'{a}_vs_{b}']=dict(diff=float(outcome[b].mean()-outcome[a].mean()),lo=lo,hi=hi,
                                      distinguishable=bool(not lo<=0<=hi))
        all_distinguished=all(v['distinguishable'] for v in pairs.values())
        report[model]=dict(n=len(files),accuracy=acc,floor_range_ok=bool(floor_ok),
                           pairwise=pairs,
                           valid=bool(floor_ok and all_distinguished),
                           note=('valid evidence-strength manipulation' if floor_ok and all_distinguished
                                 else 'NOT a valid manipulation this round; report honestly, change nothing'))
    write_json(root/'outputs/tables/stage9/evidence_validity.json',
               dict(rule=('invalid if short64 or short32 direct accuracy in [0,0.10], '
                          'or any pair of conditions indistinguishable (class-paired 95% CI of the difference contains 0)'),
                    replicates=replicates,floor=FLOOR,models=report))
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--replicates',type=int,default=2000);a=ap.parse_args()
    report=validity(initialize(a.root),a.replicates)
    print(json.dumps(report,ensure_ascii=False,indent=1,default=float))


if __name__=='__main__':main()
