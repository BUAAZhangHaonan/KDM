#!/usr/bin/env python3
import argparse,json,sys,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from kdm.analysis import annotated_rows,evaluate,probe_summary,export_csv
from kdm.annotation import validate_annotations
from kdm.reports import behavioral_validity,method_comparison
from kdm.io import read_jsonl,within,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--records',nargs='+',required=True)
    p.add_argument('--annotations',required=True);p.add_argument('--aliases',required=True);p.add_argument('--closed',nargs='+',required=True)
    p.add_argument('--normalizer',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args()
    target=within(a.root,a.out_dir);target.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('vqa_official',a.normalizer);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    ev=m.VQAEval(None,None);normalizer=lambda x:ev.processDigitArticle(ev.processPunctuation(x))
    rows=annotated_rows(a.records,validate_annotations(a.annotations),json.load(open(a.aliases)),normalizer)
    closed=[r for path in a.closed for r in read_jsonl(path)];probes=probe_summary(rows)
    reports={'paired':evaluate(rows),'probe':probes,'validity':behavioral_validity(rows,probes,closed),
             'method':method_comparison(rows,probes,closed)}
    for name,values in reports.items():atomic_json(target/(name+'.json'),values);export_csv(values,target/(name+'.csv'))
if __name__=='__main__':main()
