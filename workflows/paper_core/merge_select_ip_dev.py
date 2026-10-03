#!/usr/bin/env python3
"""Merge disjoint closed IP-dev parts; select only the complete registered panel."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from kdm.io import file_hash,within
from kdm.prompts import MARKERS
from workflows.paper_core.dev_viz import roster
from workflows.paper_core.select_phrase import select
from workflows.paper_core.score_ip_dev import CORE,EXTRA,FIELDS
from workflows.paper_core.score_native import save_json,save_rows
from workflows.supplemental.remaining11.score import rows
OLD_SELECTION='outputs/paper_core_20261002_dev_viz/dev_selection/all5_preserved_20261003_0010/selected_configs.json'
MAIN='outputs/paper_core_20261002_dev_viz/nine_main_food_cpu_20261003_0400/closed_main_v9_with_paired_20261003_0708'


def merge_closed(paths,expected_ids):
    values=[];proofs=[];seen=set();sources=set();reference={}
    allowed={(m,'instruction_m3id') for m in CORE}|{(m,t) for m in EXTRA for t in ('instruction_vcd','instruction_m3id')}
    for folder in paths:
        receipt_path=folder/'receipt.json';receipt=json.loads(receipt_path.read_text());raw=folder/'score_rows.jsonl.gz'
        if (receipt['schema'] not in {'kdm_nine_model_IP_only_dev_pilot_cpu_scoring_v1','kdm_nine_model_IP_only_dev_cpu_scoring_v1'}
                or receipt['passed'] is not True or receipt['reference_join_missing']!=0
                or any(receipt[k] for k in ('canonical_pending_rows','literal_pending_rows','abstain_pending_rows','pending_QA','pending_memberships'))
                or receipt['outputs']['score_rows.jsonl.gz']!=file_hash(raw)):
            raise ValueError('Merge requires the actual fully closed immutable score receipt')
        count=0
        for line,row,line_sha in rows(raw):
            if (row['key'] in seen or (row['model'],row['method']) not in allowed
                    or row['sample_id'] not in expected_ids or row['dataset']!='food101' or row['split']!='dev'
                    or row['marker'] not in MARKERS or row['reference_marker']!=row['marker']
                    or row['kind']!='instruction_preserving' or row['guided'] is not True
                    or row['reference_guided'] is not False or row['replicate']!=0):
                raise ValueError('Merged dev key is duplicate, foreign or outside its exact IP condition')
            if (row['correct_canonical'] not in (0,1) or row['correct_literal'] not in (0,1)
                    or type(row['abstain']) is not bool or type(row['uniform_reference']) is not bool
                    or (row['correct_canonical']==1 and row['abstain'])
                    or row['independent_attempts']!=10
                    or row['uniform_reference']!=(row['gold_rank']>1 and row['independent_correct_attempts']==0)):
                raise ValueError('Merged score is unresolved or differs from the original Food reference rule')
            ref_key=(row['model'],row['sample_id'])
            ref=(row['gold_rank'],row['independent_correct_attempts'],row['independent_attempts'],row['uniform_reference'])
            if ref_key in reference and reference[ref_key]!=ref:
                raise ValueError('Original per-model sample reference changed across a received condition')
            reference[ref_key]=ref;seen.add(row['key']);count+=1
            values.append({**row,'merged_score_source_path':str(raw),'merged_score_source_line':line,
                'merged_score_line_sha256':line_sha,'merged_score_receipt_path':str(receipt_path)})
            sources.add(row['source_path'])
        if count!=receipt['rows']:raise ValueError('Actual score rows differ from the closed receipt')
        proofs.append(dict(path=str(raw),sha256=file_hash(raw),rows=count,receipt_path=str(receipt_path),receipt_sha256=file_hash(receipt_path)))
    frame=pd.DataFrame(values)
    coverage=[]
    for model,method in sorted(allowed):
        for marker in MARKERS:
            group=frame[frame.model.eq(model)&frame.method.eq(method)&frame.marker.eq(marker)]
            ids=set(group.sample_id);complete=len(group)==404 and ids==expected_ids and not group.sample_id.duplicated().any()
            coverage.append(dict(model=model,method=method,marker=marker,rows=len(group),expected_n=404,
                unique_sample_ids=len(ids),missing_samples=len(expected_ids-ids),complete=complete,
                canonical_pending=0,literal_pending=0,abstain_pending=0,
                condition_completion_uses_actual_merged_ID_coverage=True))
    return values,frame,pd.DataFrame(coverage),proofs,len(sources)


def selected_eval(selected,output):
    main=ROOT/MAIN;receipt_path=main/'receipt.json';receipt=json.loads(receipt_path.read_text())
    metrics_path=main/'metrics_all_complete.csv';old_eval_path=main/'dev_selected_operating_points.csv'
    if (not receipt['passed'] or not receipt['complete_main_panel'] or receipt['complete_conditions']!=123
            or receipt['missing_or_pending_rows'] or file_hash(metrics_path)!=receipt['outputs'][metrics_path.name]
            or file_hash(old_eval_path)!=receipt['outputs'][old_eval_path.name]):
        raise ValueError('The frozen nine-model main eval panel is not fully accepted')
    metrics=pd.read_csv(metrics_path);old_eval=pd.read_csv(old_eval_path)
    if len(old_eval)!=5:raise ValueError('The five existing core IP-VCD dev points differ')
    points=[]
    for chosen in selected:
        matches=metrics[metrics.model.eq(chosen['model'])&metrics.method.eq(chosen['method'])
            &metrics.marker.eq(chosen['marker'])&metrics.reference_marker.eq(chosen['marker'])
            &metrics.kind.eq('instruction_preserving')&metrics.dataset.eq('food101')&metrics.split.eq('eval')
            &metrics.guided.eq(True)&metrics.reference_guided.eq(False)&metrics.replicate.eq(0)]
        if len(matches)!=1:raise ValueError('The selected dev identity has no unique exact accepted eval run')
        result=matches.iloc[0].to_dict()
        if result['n']!=2424 or any(result[k] for k in ['canonical_pending','literal_pending','abstain_pending','reference_pending']):
            raise ValueError('An accepted selected eval result lacks its full denominator or closure')
        if chosen.get('selection_origin')=='previous_core_IP_VCD':
            prior=old_eval[old_eval.model.eq(chosen['model'])&old_eval.method.eq(chosen['method'])&old_eval.marker.eq(chosen['marker'])]
            if len(prior)!=1:raise ValueError('The old preserved selection changed')
            for k in ['C','W','A','TP','FP','FN','reference_positive','n','J']:
                if prior.iloc[0][k]!=result[k]:raise ValueError('An old selected eval metric changed')
            result=prior.iloc[0].to_dict()
        else:
            result.update(dev_selected=True,actual_frozen_dev_selection=True,selection='dev_selected',
                dev_selection_source=str(output/'selected_configs.json'),dev_selection_source_sha256=file_hash(output/'selected_configs.json'))
        result.update(dev_n=chosen['n'],dev_correct=chosen['correct'],dev_supported_abstentions=chosen['tp'],
            dev_J=chosen['selection_utility'],selection_origin=chosen['selection_origin'])
        points.append(result)
    pd.DataFrame(points).to_csv(output/'selected_operating_points_eval.csv',index=False)
    return dict(main_receipt_path=str(receipt_path),main_receipt_sha256=file_hash(receipt_path),
        main_metrics_path=str(metrics_path),main_metrics_sha256=file_hash(metrics_path),selected_eval_rows=len(points),
        preserved_previous_core_eval_rows=5,actual_new_eval_rows=13,eval_generation=0,eval_metric_recomputed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score-dir',action='append',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only merge requires empty CUDA_VISIBLE_DEVICES')
    output=within(ROOT,args.out);output.relative_to(ROOT/'outputs/paper_core_20261002_dev_viz')
    if output.exists():raise FileExistsError('Use a fresh merge checkpoint')
    roster_path,samples=roster('dev404');expected_ids={s['id'] for s in samples}
    values,frame,coverage,proofs,source_count=merge_closed([within(ROOT,x) for x in args.score_dir],expected_ids)
    output.mkdir(parents=True,exist_ok=False)
    save_rows(output/'merged_closed_dev_rows.jsonl.gz',values)
    coverage.to_csv(output/'condition_coverage.csv',index=False)
    complete=len(frame)==52*404 and len(coverage)==52 and bool(coverage.complete.all())
    selected=[];eval_proof=None
    if complete:
        dev_metrics,chosen=select(frame,expected_ids)
        if len(chosen)!=13 or len(dev_metrics)!=52:raise ValueError('Only 13 new model-method selections are allowed')
        frozen_at=datetime.now(timezone.utc).isoformat()
        for item in chosen:item.update(selection='dev_selected',selection_origin='new_IP_only_dev404',
            dataset='food101',split='dev',kind='instruction_preserving',reference_marker=item['marker'],
            guided=True,reference_guided=False,replicate=0,
            selection_frozen_at_utc=frozen_at,source_scores=str(output/'merged_closed_dev_rows.jsonl.gz'),
            source_scores_sha256=file_hash(output/'merged_closed_dev_rows.jsonl.gz'))
        old_path=ROOT/OLD_SELECTION;old=[r for r in json.loads(old_path.read_text()) if r['method']=='instruction_vcd']
        if len(old)!=5 or {r['model'] for r in old}!=set(CORE):raise ValueError('The five original core IP-VCD selections differ')
        selected=[{**r,"selection_origin":'previous_core_IP_VCD'} for r in old]
        selected += chosen
        save_json(output/'selected_configs.json',selected)
        dev_metrics.to_csv(output/'dev_metrics52.csv',index=False)
        eval_proof=selected_eval(selected,output)
    result=dict(schema='kdm_nine_IP_dev404_closed_merge_selection_v1',passed=True,created_utc=datetime.now(timezone.utc).isoformat(),
        rows=len(frame),expected_rows=21008,unique_keys=frame.key.nunique(),source_parts=source_count,
        complete_new_conditions=int(coverage.complete.sum()),expected_new_conditions=52,missing_rows=21008-len(frame),
        duplicate_keys=0,reference_changed_across_conditions=0,pending_rows=0,
        actual_selection_performed=complete,new_selected_model_methods=13 if complete else 0,
        preserved_prior_core_model_methods=5 if complete else 0,total_dev_selected_model_methods=18 if complete else 0,
        selection_requires_all52_conditions_404=True,pilot_historical_scope_not_used_to_override_actual_coverage=True,
        score_sources=proofs,roster_path=str(roster_path),roster_sha256=file_hash(roster_path),
        reused_selection_script_path=str(ROOT/'workflows/paper_core/select_phrase.py'),
        reused_selection_script_sha256=file_hash(ROOT/'workflows/paper_core/select_phrase.py'),
        previous_core_selection_path=str(ROOT/OLD_SELECTION),previous_core_selection_sha256=file_hash(ROOT/OLD_SELECTION),
        accepted_eval_sources=eval_proof,runner_sha256=file_hash(Path(__file__)),actual_command=sys.argv,
        GPU_initialized=False,new_generation=0,new_scientific_judgments=0,
        outputs={q.name:file_hash(q) for q in output.iterdir() if q.is_file()})
    save_json(output/'receipt.json',result)
    print(json.dumps({k:result[k] for k in ['passed','rows','complete_new_conditions','missing_rows','actual_selection_performed','total_dev_selected_model_methods']}))

if __name__=='__main__':main()
