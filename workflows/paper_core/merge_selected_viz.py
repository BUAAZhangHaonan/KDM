#!/usr/bin/env python3
"""Merge only closed new Viz512 cohorts and copy the frozen core31 panel."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from kdm.io import file_hash, within
from workflows.paper_core.dev_viz import roster
from workflows.paper_core.score_dev_viz import COND, FIELDS, viz_metrics
from workflows.paper_core.score_ip_dev import EXTRA
from workflows.paper_core.score_native import save_json, save_rows
from workflows.supplemental.remaining11.score import rows
BASE=ROOT/'outputs/paper_core_20261002_dev_viz'
OLD=BASE/'merged_viz31_all_five_closed_20261003_0410'
CURATED=BASE/'viz_curated_main_mechanism_20261003_0415'
AUTH=BASE/'native_baselines/VIZ512_SELECTED21_MAIN_GAP_AUTHORITY_ACTUAL_FREEZE_20261003_2202.json'


def key(record):
    return tuple(record[field] for field in COND)


def validate_panel(records, expected_conditions, ids):
    groups={}
    seen=set()
    for record in records:
        if record['key'] in seen or record['sample_id'] not in ids:
            raise ValueError('Duplicate key or foreign fixed512 sample')
        seen.add(record['key']);groups.setdefault(key(record),[]).append(record)
        if record['quality_score'] is None or record['abstain'] is None:
            raise ValueError('An unresolved score cannot enter the final merge')
        quality=float(record['quality_score']);raw=float(record['official_raw_score'])
        if not (0<=quality<=1 and 0<=raw<=1) or type(record['abstain']) is not bool:
            raise ValueError('The official continuous score or semantic abstention is invalid')
        if record['abstain'] and quality!=0:
            raise ValueError('Abstention answer quality must remain zero')
    if len(groups)!=expected_conditions:
        raise ValueError('The actual complete condition count differs')
    for group in groups.values():
        if len(group)!=512 or {r['sample_id']for r in group}!=ids:
            raise ValueError('A final condition lacks the exact complete512 roster')
        if sum(bool(r['official_reference'])for r in group)!=166:
            raise ValueError('The official166/346 split differs')
    return groups,seen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ownership',default=str(BASE/'viz_cpu_stream/source_ownership.json'))
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    output=within(ROOT,args.out);output.relative_to(BASE)
    if output.exists():raise FileExistsError('Use an exclusive final result directory')
    ownership_path=within(ROOT,args.ownership);ownership=json.loads(ownership_path.read_text())
    if ownership['schema']!='kdm_completed_Viz_CPU_source_ownership_v1' or ownership['distinct_received_keys']!=12800:
        raise ValueError('All new25/12800 actually received keys are required')
    roster_path,samples=roster('viz512');ids={r['id']for r in samples}
    authority=json.loads(AUTH.read_text());expected={}
    if authority['passed'] is not True or authority['authorized_main_configurations']!=21:
        raise ValueError('The frozen selected21 authority differs')
    for condition in authority['conditions']:
        c={'model':condition['model'],'dataset':'vizwiz','split':'eval',**{k:condition[k]for k in FIELDS}}
        expected[key(c)]=c
    for model in EXTRA:
        c=dict(model=model,dataset='vizwiz',split='eval',method='direct',kind='unguided',marker='UNKNOWN',reference_marker='UNKNOWN',guided=False,reference_guided=False,replicate=0)
        expected[key(c)]=c
    if len(expected)!=25:raise ValueError('The registered new main configuration scope differs')
    new=[];sources=[]
    for cohort in ownership['cohorts']:
        if cohort['accepted_complete_scores'] is not True:
            raise ValueError('A received cohort still requires genuine semantic closure')
        directory=within(ROOT,cohort['score_dir']);receipt_path=directory/'receipt.json'
        if file_hash(receipt_path)!=cohort['receipt_sha256']:
            raise ValueError('A closed source receipt changed')
        receipt=json.loads(receipt_path.read_text());score_path=directory/'score_rows.jsonl.gz'
        if (receipt['passed'] is not True or receipt['quality_pending_rows']!=0 or receipt['abstain_pending_rows']!=0
                or receipt['pending_QA']!=0 or receipt['reference_join_missing']!=0
                or file_hash(score_path)!=cohort['score_sha256']
                or receipt['outputs']['score_rows.jsonl.gz']!=cohort['score_sha256']):
            raise ValueError('Only immutable genuinely closed scores may merge')
        actual=[]
        for line,record,line_sha in rows(score_path):
            actual.append(record['key']);new.append({**record,'merged_score_source_path':str(score_path),
                'merged_score_source_line':line,'merged_score_line_sha256':line_sha,
                'merged_score_receipt_path':str(receipt_path)})
        if len(actual)!=receipt['rows'] or set(actual)!=set(cohort['keys']):
            raise ValueError('The scored source differs from its original owned keys')
        sources.append(dict(path=str(score_path),sha256=file_hash(score_path),rows=len(actual),receipt_path=str(receipt_path),receipt_sha256=file_hash(receipt_path)))
    new_groups,newkeys=validate_panel(new,25,ids)
    if len(new)!=12800 or set(new_groups)!=set(expected):
        raise ValueError('New source condition identities differ from registered main25')
    newmetrics=[]
    for condition,records in new_groups.items():
        frame=pd.DataFrame(records);frame['expected_n']=512;frame['source_raw_complete']=True
        metric=viz_metrics(frame,dict(zip(COND,condition)))
        if metric['primary_complete'] is not True:raise ValueError('A final new condition is incomplete')
        newmetrics.append(metric)
    old_receipt_path=OLD/'receipt.json';old_receipt=json.loads(old_receipt_path.read_text())
    if (old_receipt['passed'] is not True or old_receipt['rows']!=15872 or old_receipt['conditions']!=31
            or old_receipt['primary_complete_conditions']!=31 or old_receipt['quality_pending_rows']!=0
            or old_receipt['abstain_pending_rows']!=0):
        raise ValueError('The frozen core31 panel is not closed')
    old_score=OLD/'score_rows.jsonl.gz';old_metrics=OLD/'metrics_all.csv'
    if file_hash(old_score)!=old_receipt['outputs']['score_rows.jsonl.gz'] or file_hash(old_metrics)!=old_receipt['outputs']['metrics_all.csv']:
        raise ValueError('A frozen core31 artifact changed')
    old=[record for _,record,_ in rows(old_score)];old_groups,oldkeys=validate_panel(old,31,ids)
    if oldkeys.intersection(newkeys) or set(old_groups).intersection(new_groups):
        raise ValueError('Old and new accepted sources overlap')
    curated_receipt_path=CURATED/'receipt.json';curated=json.loads(curated_receipt_path.read_text())
    if (curated['source_sha256']!=file_hash(old_metrics) or curated['main_conditions']!=20
            or curated['mechanism_only_conditions']!=11):raise ValueError('The frozen curated partition differs')
    coremain_path=CURATED/'core5_viz512_main_native_direct_cda_ip.csv'
    mechanism_path=CURATED/'core5_viz512_guided_mechanism_only.csv'
    coremain=pd.read_csv(coremain_path);mechanism=pd.read_csv(mechanism_path);oldframe=pd.read_csv(old_metrics)
    mainkeys={key(x)for x in coremain.to_dict('records')};mechanismkeys={key(x)for x in mechanism.to_dict('records')}
    if len(mainkeys)!=20 or len(mechanismkeys)!=11 or mainkeys.intersection(mechanismkeys) or mainkeys|mechanismkeys!=set(old_groups):
        raise ValueError('The frozen main20/mechanism11 partition differs')
    output.mkdir(parents=True,exist_ok=False)
    newframe=pd.DataFrame(newmetrics)
    newframe.to_csv(output/'new25_metrics.csv',index=False)
    pd.concat([coremain,newframe],ignore_index=True,sort=False).to_csv(output/'main45_metrics.csv',index=False)
    pd.concat([oldframe,newframe],ignore_index=True,sort=False).to_csv(output/'all56_metrics.csv',index=False)
    shutil.copyfile(mechanism_path,output/'mechanism11_metrics.csv')
    save_rows(output/'new25_scores.jsonl.gz',new)
    save_rows(output/'all56_scores.jsonl.gz',old+new)
    save_rows(output/'main45_scores.jsonl.gz',[r for r in old if key(r)in mainkeys]+new)
    save_rows(output/'mechanism11_scores.jsonl.gz',[r for r in old if key(r)in mechanismkeys])
    coverage=[]
    for condition,records in {**old_groups,**new_groups}.items():
        coverage.append({**dict(zip(COND,condition)),'n':len(records),'unique_sample_ids':len({r['sample_id']for r in records}),
            'official_unanswerable_n':166,'official_answerable_n':346,'quality_pending':0,'abstain_pending':0,
            'panel_role':'main' if condition in mainkeys or condition in new_groups else 'mechanism_only',
            'scores_copied_from_frozen_core31':condition in old_groups})
    pd.DataFrame(coverage).to_csv(output/'condition_coverage56.csv',index=False)
    save_json(output/'source_score_receipts.json',sources)
    receipt=dict(schema='kdm_nine_selected_Viz512_complete_frozen_merge_v1',passed=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),new_rows=12800,new_conditions=25,
        frozen_old_rows=15872,frozen_old_conditions=31,total_rows=28672,total_conditions=56,
        main_rows=23040,main_conditions=45,mechanism_rows=5632,mechanism_conditions=11,
        duplicate_keys=0,missing_sample_keys=0,quality_pending_rows=0,abstain_pending_rows=0,
        official_unanswerable_per_condition=166,official_answerable_per_condition=346,
        roster_path=str(roster_path),roster_sha256=file_hash(roster_path),authority_path=str(AUTH),authority_sha256=file_hash(AUTH),
        ownership_path=str(ownership_path),ownership_sha256=file_hash(ownership_path),new_score_sources=sources,
        old_receipt_path=str(old_receipt_path),old_receipt_sha256=file_hash(old_receipt_path),
        old_score_path=str(old_score),old_score_sha256=file_hash(old_score),old_metrics_path=str(old_metrics),old_metrics_sha256=file_hash(old_metrics),
        curated_receipt_path=str(curated_receipt_path),curated_receipt_sha256=file_hash(curated_receipt_path),
        frozen_old_scores_rescored_rows=0,frozen_old_metrics_recomputed=False,new_semantic_judgments=0,
        new_Viz_J_defined=False,partial_official_scores_binarized=False,Food_reference_substituted=False,
        reused_viz_metrics_sha256=file_hash(ROOT/'workflows/paper_core/score_dev_viz.py'),
        runner_sha256=file_hash(Path(__file__)),actual_command=sys.argv,GPU_initialized=False,
        outputs={p.name:file_hash(p)for p in output.iterdir()if p.is_file()})
    save_json(output/'receipt.json',receipt)
    print(json.dumps({k:receipt[k]for k in ['passed','total_rows','total_conditions','main_conditions','mechanism_conditions']}))

if __name__=='__main__':main()
