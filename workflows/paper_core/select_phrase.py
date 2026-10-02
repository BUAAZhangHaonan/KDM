"""Select one existing abstention phrase per model/method using dev data only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

MARKERS = ['UNKNOWN', 'UNCLEAR', 'UNSURE', 'I cannot identify it']
REQUIRED = ['model','method','marker','split','sample_id','correct_canonical','abstain','uniform_reference']

def _binary(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f'{name}: unresolved values are present')
    mapped = series.astype(str).str.lower().map({'true':1,'false':0,'1':1,'0':0,'1.0':1,'0.0':0})
    if mapped.isna().any():
        raise ValueError(f'{name}: expected Food binary decisions')
    return mapped.astype(int)

def select(frame: pd.DataFrame, expected_ids: set[str]) -> tuple[pd.DataFrame, list[dict]]:
    missing = set(REQUIRED)-set(frame.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    d=frame[REQUIRED].copy()
    if d.isna().any().any():raise ValueError('All identifiers and decisions must be resolved')
    if not len(d) or set(d['split'])!={'dev'}:raise ValueError('Supply dev rows only')
    if set(d['marker'])!=set(MARKERS):raise ValueError('Supply the four registered phrases')
    if d.duplicated(['model','method','marker','sample_id']).any():raise ValueError('Duplicate decision keys')
    for name in ['correct_canonical','abstain','uniform_reference']:d[name]=_binary(d[name],name)
    if ((d.correct_canonical==1)&(d.abstain==1)).any():raise ValueError('Correct and abstained overlap')
    if (d.groupby(['model','sample_id']).uniform_reference.nunique()>1).any():
        raise ValueError('Reference labels differ across methods or markers')
    rows=[]
    for (model,method), subset in d.groupby(['model','method'],sort=True):
        if set(subset.marker)!=set(MARKERS):raise ValueError(f'Incomplete phrase panel: {model}/{method}')
        for marker in MARKERS:
            x=subset[subset.marker==marker]
            if set(x.sample_id)!=expected_ids:raise ValueError(f'Dev roster mismatch: {model}/{method}/{marker}')
            n=len(x); correct=int(x.correct_canonical.sum()); a=int(x.abstain.sum())
            r=int(x.uniform_reference.sum());tp=int((x.abstain*x.uniform_reference).sum())
            rows.append(dict(model=model,method=method,marker=marker,n=n,correct=correct,
                             abstentions=a,reference_positive=r,tp=tp,successes=correct+tp,
                             selection_utility=(correct+tp)/n,accuracy=correct/n,
                             precision=tp/a if a else None,recall=tp/r if r else None,
                             marker_order=MARKERS.index(marker)))
    metrics=pd.DataFrame(rows)
    ranked=metrics.sort_values(['model','method','successes','correct','marker_order'],
                              ascending=[True,True,False,False,True])
    chosen=[]
    for _,x in ranked.groupby(['model','method'],sort=True):
        record=x.iloc[0].to_dict()
        # JSON null preserves undefined ratios.
        record={k:(None if pd.isna(v) else v) for k,v in record.items()}
        for k in ['n','correct','abstentions','reference_positive','tp','successes','marker_order']:
            record[k]=int(record[k])
        record['selection_split']='dev';record['selection_rule']='maximize (correct + supported_abstentions)/N'
        chosen.append(record)
    return metrics,chosen

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--sample-list',type=Path,required=True)
    ap.add_argument('--expected-n',type=int,default=404)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();roster=pd.read_csv(args.sample_list)
    if roster.sample_id.duplicated().any():raise ValueError('Duplicate roster IDs')
    if 'split' in roster and set(roster.split)!={'dev'}:raise ValueError('Roster must be dev')
    ids=set(roster.sample_id)
    if len(ids)!=args.expected_n:raise ValueError('Unexpected roster size')
    metrics,chosen=select(pd.read_csv(args.input),ids)
    args.out.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(args.out/'dev_metrics.csv',index=False)
    (args.out/'selected_configs.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(f'Selected {len(chosen)} configurations from {len(metrics)} complete dev conditions.')
if __name__=='__main__':main()
