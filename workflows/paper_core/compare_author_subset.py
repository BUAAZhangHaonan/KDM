#!/usr/bin/env python3
"""Compare corrected baselines with frozen main methods on identical 101 inputs."""
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from kdm.io import read_jsonl,file_hash,atomic_json,within

BASE='outputs/paper_core_20261002_dev_viz'
FROZEN=BASE+'/nine_main_food_cpu_20261003_0400/closed_main_v9_with_paired_20261003_0708/main_score_rows.parquet'
SELECTION=BASE+'/nine_dev_selected_J_main_comparison_20261003_2228/main69_actual_operating_points.csv'


def main():
    p=argparse.ArgumentParser();p.add_argument('--new-scores',action='append',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();out=within(ROOT,a.out);out.mkdir(parents=True,exist_ok=False)
    with (ROOT/'workflows/paper_core/attribution_representative101.csv').open() as f:ids={r['sample_id'] for r in csv.DictReader(f)}
    if len(ids)!=101:raise ValueError('Representative IDs differ')
    selection=pd.read_csv(ROOT/SELECTION)
    selection=selection[selection.method.isin(['direct','deco','vcd','cda_visual','instruction_vcd','instruction_m3id'])]
    if len(selection)!=54 or selection.duplicated(['model','method']).any():raise ValueError('Actual dev selected/main configuration coverage differs')
    frozen=pd.read_parquet(ROOT/FROZEN);scored=[];configuration=[]
    for condition in selection.to_dict('records'):
        group=frozen[(frozen.condition_id==condition['condition_id']) & frozen.sample_id.isin(ids)]
        if len(group)!=101 or set(group.sample_id)!=ids:raise ValueError('Frozen comparator subset missing')
        for row in group.to_dict('records'):
            scored.append({'model':row['model'],'method':row['method'],'sample_id':row['sample_id'],
                'correct':int(row['correct_canonical']),'abstain':bool(row['abstain']),
                'uniform_reference':bool(row['uniform_reference']),'condition_id':condition['condition_id'],
                'marker':condition['marker'],'implementation_revision':'frozen_accepted',
                'source':FROZEN})
        configuration.append({k:condition[k] for k in ('model','checkpoint','method','kind','marker','reference_marker','guided','reference_guided','replicate','condition_id')})
    used=set()
    for name in a.new_scores:
        for row in read_jsonl(within(ROOT,name)):
            key=(row['model'],row['method'],row['sample_id'])
            if key in used:raise ValueError('Repeated new comparison row')
            used.add(key)
            if row['sample_id'] not in ids:raise ValueError('New baseline uses a different subset')
            scored.append({'model':row['model'],'method':row['method'],'sample_id':row['sample_id'],
                'correct':row['correct_canonical'],'abstain':row['abstain'],
                'uniform_reference':row['uniform_reference'],'condition_id':None,'marker':'NONE',
                'implementation_revision':row['implementation_revision'],'source':name})
    groups=defaultdict(list)
    for r in scored:groups[(r['model'],r['method'])].append(r)
    metrics=[];by_key={}
    for (model,method),group in sorted(groups.items()):
        if len(group)!=101 or {r['sample_id'] for r in group}!=ids:raise ValueError('Unequal comparator denominator')
        complete=all(r['correct'] is not None and r['abstain'] is not None for r in group)
        c=sum(r['correct'] for r in group) if complete else None
        ac=sum(r['abstain'] for r in group) if complete else None
        tp=sum(r['abstain'] and r['uniform_reference'] for r in group) if complete else None
        metrics.append({'model':model,'method':method,'n':101,'complete':complete,'C':c,
                        'W':101-c-ac if complete else None,'A':ac,'TP':tp,
                        'J':(c+tp)/101 if complete else None,
                        'marker':group[0]['marker'],'implementation_revision':group[0]['implementation_revision']})
        by_key[(model,method)]={r['sample_id']:r for r in group}
    paired=[]
    for (model,ip),iprows in by_key.items():
        if ip not in ('instruction_vcd','instruction_m3id'):continue
        for baseline in ('direct','deco','vcd','cda_visual','dola','sid'):
            other=by_key.get((model,baseline))
            if other is None:continue
            if any(iprows[k]['uniform_reference']!=other[k]['uniform_reference'] for k in ids):raise ValueError('Paired reference differs')
            if any(other[k]['correct'] is None or other[k]['abstain'] is None for k in ids):continue
            deltas=[int(iprows[k]['correct'] or (iprows[k]['abstain'] and iprows[k]['uniform_reference']))
                    -int(other[k]['correct'] or (other[k]['abstain'] and other[k]['uniform_reference'])) for k in sorted(ids)]
            paired.append({'model':model,'method':ip,'baseline':baseline,'n':101,
                           'delta_J':sum(deltas)/101,'improved':sum(x>0 for x in deltas),
                           'harmed':sum(x<0 for x in deltas),'unchanged':sum(x==0 for x in deltas),
                           'comparison_scope':'same fixed 101 inputs; descriptive subset only'})
    pd.DataFrame(scored).to_csv(out/'same101_sample_scores.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'same101_J.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'same101_IP_paired_changes.csv',index=False)
    atomic_json(out/'frozen_configurations.json',configuration)
    atomic_json(out/'receipt.json',{'frozen_conditions':54,'new_conditions':len(groups)-54,
        'frozen_score_sha256':file_hash(ROOT/FROZEN),'selection_sha256':file_hash(ROOT/SELECTION),
        'uniform_n':101,'GPU_initialized':False,'configuration_reselection':False,
        'scope':'same subset; original 2424-eval panel is not overwritten'})
    print(json.dumps({'conditions':len(groups),'rows':len(scored),'paired':len(paired)}))

if __name__=='__main__':main()
