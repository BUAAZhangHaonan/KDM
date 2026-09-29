#!/usr/bin/env python3
import argparse,json,sys,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from kdm.analysis import annotated_rows,evaluate,probe_summary,export_csv
from kdm.annotation import validate_annotations
from kdm.reports import behavioral_validity,method_comparison,validate_report_coverage
from kdm.io import read_jsonl,within,atomic_json,file_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--records',nargs='+',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--selection',required=True);p.add_argument('--method-plan',required=True)
    p.add_argument('--annotations',required=True);p.add_argument('--aliases',required=True);p.add_argument('--closed',nargs='+',required=True)
    p.add_argument('--normalizer',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args()
    root=Path(a.root).resolve();target=within(root,a.out_dir)
    for name in ('manifest','selection','annotations','aliases','normalizer','method_plan'):
        setattr(a,name,str(within(root,getattr(a,name))))
    a.records=[str(within(root,path)) for path in a.records];a.closed=[str(within(root,path)) for path in a.closed]
    from kdm.protocol import validate_freeze
    from kdm.task_provenance import validate_task_inputs
    freeze=validate_freeze(root)
    if file_hash(a.manifest)!=freeze['files']['data/current/all.jsonl']:raise ValueError('Report manifest differs from frozen full original manifest')
    if file_hash(a.aliases)!=freeze['files']['configs/kdm/food_aliases.json']:raise ValueError('Report aliases differ from frozen scoring identity')
    if file_hash(a.normalizer)!=file_hash(root/'src/kdm/models/official_vqa_normalizer.py'):raise ValueError('Report normalizer differs from active frozen official implementation')
    raw_provenance=validate_task_inputs(root,a.records,freeze,stages={'experiment','probe'})
    closed_provenance=validate_task_inputs(root,a.closed,freeze,stages={'closed'})
    spec=importlib.util.spec_from_file_location('vqa_official',a.normalizer);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    ev=m.VQAEval(None,None);normalizer=lambda x:ev.processDigitArticle(ev.processPunctuation(x))
    from kdm.human_review import validate_human_review
    annotations=validate_human_review(a.annotations,a.records)
    for path in a.records:
        for record in read_jsonl(path):
            annotation=annotations.get(record['key'])
            if annotation is None or annotation['text']!=record['text']:
                raise ValueError('Every formal response requires a matching unified annotation record')
    rows=annotated_rows(a.records,annotations,json.load(open(a.aliases)),normalizer)
    closed=[r for path in a.closed for r in read_jsonl(path)]
    plan_path=within(a.root,a.method_plan);plan_relative=str(plan_path.relative_to(Path(a.root).resolve()))
    if freeze.get('status')!='frozen' or freeze.get('files',{}).get(plan_relative)!=file_hash(plan_path):
        raise ValueError('Report method plan differs from frozen identity')
    from kdm.selection_provenance import validate_selection
    selection_input=validate_selection(root,a.selection,a.manifest,freeze)
    method_plan=json.load(open(plan_path));selection=selection_input['selection']
    coverage=validate_report_coverage(rows,list(read_jsonl(a.manifest)),selection,closed,method_plan)
    from kdm.protocol import validate_method_runtime
    for condition in selection:
        if condition['selected']:
            runtime=json.loads((Path(a.root)/'configs/runtime'/f"{condition['model']}.json").read_text())
            validate_method_runtime(Path(a.root),runtime,method_plan[condition['model']][condition['dataset']])
    coverage.update(method_plan_sha256=file_hash(a.method_plan),manifest_sha256=file_hash(a.manifest),selection_sha256=file_hash(a.selection),
        raw_provenance=raw_provenance,closed_provenance=closed_provenance,selection_provenance=selection_input['provenance'])
    target.mkdir(parents=True,exist_ok=True)
    atomic_json(target/'coverage.json',coverage)
    probes=probe_summary(rows)
    reports={'paired':evaluate(rows),'probe':probes,'validity':behavioral_validity(rows,probes,closed),
             'method':method_comparison(rows,probes,closed)}
    for name,values in reports.items():atomic_json(target/(name+'.json'),values);export_csv(values,target/(name+'.csv'))
if __name__=='__main__':main()
