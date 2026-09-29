#!/usr/bin/env python3
"""Compare completed task identifiers against the entire frozen manifest."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from kdm.pipeline import census_tasks,experiment_tasks,probe_tasks,task_id
from kdm.prompts import MARKERS
from kdm.io import read_jsonl,within,atomic_json

def verify(samples,records,model,mode,methods=('vcd','m3id','dola','deco')):
    if mode not in {'census','probe','experiment'}:raise ValueError('Unknown completion mode')
    if len({s['id'] for s in samples})!=len(samples):raise ValueError('Duplicate manifest sample')
    tasks=census_tasks(samples) if mode=='census' else (probe_tasks(samples) if mode=='probe' else experiment_tasks(samples,methods,MARKERS))
    expected_tasks={task_id(model,t):t for t in tasks};expected=set(expected_tasks);found=set();duplicates=[];invalid=[]
    for r in records:
        if r['model']!=model:raise ValueError('Wrong model in completion file')
        if r['key'] in found:duplicates.append(r['key'])
        found.add(r['key'])
        task=expected_tasks.get(r['key'])
        if task is not None:
            if r.get('status')!='ok' or any(r.get(k)!=v for k,v in task.items()):invalid.append(r['key'])
            elif not isinstance(r.get('text'),str) or not isinstance(r.get('terminated'),bool):invalid.append(r['key'])
            elif not isinstance(r.get('tokens'),list) or not r['tokens']:invalid.append(r['key'])
    return {'expected':len(expected),'observed':len(found),'missing':sorted(expected-found),
            'unexpected':sorted(found-expected),'duplicates':duplicates,'invalid':sorted(set(invalid)),'complete':found==expected and not duplicates and not invalid}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--records',nargs='+',required=True);p.add_argument('--model',required=True);p.add_argument('--mode',choices=['census','experiment','probe'],required=True)
    p.add_argument('--methods',default='vcd,m3id,dola,deco');p.add_argument('--out',required=True);a=p.parse_args()
    result=verify(list(read_jsonl(a.manifest)),[r for path in a.records for r in read_jsonl(path)],a.model,a.mode,tuple(a.methods.split(',')))
    atomic_json(within(a.root,a.out),result);print(json.dumps(result))
    if not result['complete']:raise SystemExit(1)
if __name__=='__main__':main()
