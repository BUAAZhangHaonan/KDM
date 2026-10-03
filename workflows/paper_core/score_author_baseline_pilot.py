#!/usr/bin/env python3
"""Frozen-old versus author-core-new scoring on an explicit finite subset."""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(Path(__file__).parent)]
from kdm.io import read_jsonl,file_hash,atomic_json,within
from workflows.main_results import score as frozen
from workflows.paper_core.score_native_baseline_parts import reference_map
from workflows.paper_core.score_dev_viz import native_decisions,NATIVE_RECEIPT,AUTHORITY
from workflows.supplemental.remaining11.score import load_authority,infer_qa,score_target

OLD='outputs/paper_core_20261002_dev_viz/nine_main_food_cpu_20261003_0400/closed_main_v9_with_paired_20261003_0708/main_score_rows.parquet'


def main():
    p=argparse.ArgumentParser();p.add_argument('--part',action='append',required=True)
    p.add_argument('--out',required=True);p.add_argument('--decision-file',action='append',default=[])
    a=p.parse_args();out=within(ROOT,a.out);out.mkdir(parents=True,exist_ok=False)
    gt,ref_sources=reference_map();old=pd.read_parquet(ROOT/OLD)
    old=old[old.method.isin(['dola','sid']) & ~old.guided.astype(bool)]
    if old.duplicated(['model','method','sample_id']).any():raise ValueError('Frozen baseline not unique')
    lookup={(r['model'],r['method'],r['sample_id']):r for r in old.to_dict('records')}
    source=json.loads((ROOT/AUTHORITY).read_text())
    native_paths,receipts=native_decisions(NATIVE_RECEIPT,ref_sources[0]['sha256'],file_hash(Path(frozen.__file__)))
    reviews,behavior,history,_,decisions=load_authority(out,{'census_final_labels':source['census_final_labels'],
        'historical_labels':[]},native_paths+[within(ROOT,x) for x in a.decision_file])
    samples={s['id']:s for s in read_jsonl(ROOT/'data/current/all.jsonl') if s['dataset']=='food101' and s['split']=='eval'}
    patterns=frozen.compile_classes(sorted({s['class'] for s in samples.values()}))
    scored=[];pending={};cache={};seen=set();source_rows=[]
    for part in a.part:
        directory=within(ROOT,part);receipt=json.loads((directory/'complete.json').read_text())
        raw=directory/'new_predictions.jsonl'
        if not receipt['passed'] or file_hash(raw)!=receipt['raw_sha256']:raise ValueError('Unsealed new raw')
        source_rows.append({'path':str(raw),'sha256':receipt['raw_sha256'],'n':receipt['completed']})
        for lineno,row in enumerate(read_jsonl(raw),1):
            s=row['sample'];key=(row['model'],row['method'],s['id'])
            if key in seen:raise ValueError('Overlapping subset sources')
            seen.add(key);prior=lookup[key]
            if s!=samples[s['id']] or prior['answer']!=row['old_text']:
                raise ValueError('Historical score does not bind to supplied prior response')
            reference=gt[(row['model'],s['id'])]
            if bool(prior['uniform_reference'])!=reference:raise ValueError('Reference changed')
            qkey=frozen.qah(s['question'],row['text'])
            reason='frozen_identical_full_response';inferred=None
            if row['text']==row['old_text']:
                c=int(prior['correct_canonical']);literal=int(prior['correct_literal']);abstain=bool(prior['abstain'])
            else:
                if qkey not in cache:cache[qkey]=infer_qa(s['question'],row['text'],patterns,reviews,behavior,decisions)
                inferred=cache[qkey];c,literal,reason=score_target(row['text'],s['class'],inferred,patterns)
                abstain=inferred['abstain']
            rec={'model':row['model'],'method':row['method'],'sample_id':s['id'],'target_class':s['class'],
                 'question':s['question'],'answer':row['text'],'qa_key':qkey,'correct_canonical':c,
                 'correct_literal':literal,'abstain':abstain,'uniform_reference':reference,
                 'old_answer':prior['answer'],'old_correct_canonical':int(prior['correct_canonical']),
                 'old_abstain':bool(prior['abstain']),'old_source':str(ROOT/OLD),
                 'old_condition_id':prior['condition_id'],'new_source':str(raw),'new_line':lineno,
                 'historical_source':row['historical_source'],'new_config':row['config'],
                 'decoder_sha256':row['decoder_sha256'],'score_reason':reason,'inference':inferred}
            scored.append(rec)
            if c is None or abstain is None or literal is None:
                r=pending.setdefault(qkey,{'qa_key':qkey,'question':s['question'],'answer':row['text'],
                                          'reasons':[],'memberships':[]})
                r['reasons']=sorted(set(r['reasons']+[reason]))
                r['memberships'].append({'model':row['model'],'method':row['method'],'sample_id':s['id'],
                                        'source_path':str(raw),'source_line':lineno})
    groups=defaultdict(list)
    for r in scored:groups[(r['model'],r['method'])].append(r)
    metrics=[]
    for (model,method),group in sorted(groups.items()):
        n=len(group);complete=all(r['correct_canonical'] is not None and r['abstain'] is not None for r in group)
        old_c=sum(r['old_correct_canonical'] for r in group);old_a=sum(r['old_abstain'] for r in group)
        old_tp=sum(r['old_abstain'] and r['uniform_reference'] for r in group)
        decided=[r for r in group if r['correct_canonical'] is not None and r['abstain'] is not None]
        c=sum(r['correct_canonical'] for r in decided);a_count=sum(r['abstain'] for r in decided)
        tp=sum(r['abstain'] and r['uniform_reference'] for r in decided)
        transitions=defaultdict(int)
        for r in decided:
            before='A' if r['old_abstain'] else ('C' if r['old_correct_canonical'] else 'W')
            after='A' if r['abstain'] else ('C' if r['correct_canonical'] else 'W')
            transitions[before+'_'+after]+=1
        metrics.append({'model':model,'method':method,'n':n,'decided':len(decided),'complete':complete,
                        'old_C':old_c,'old_W':n-old_c-old_a,'old_A':old_a,'old_TP':old_tp,
                        'old_J':(old_c+old_tp)/n,
                        'new_C':c if complete else None,'new_A':a_count if complete else None,
                        'new_W':n-c-a_count if complete else None,'new_TP':tp if complete else None,
                        'new_J':(c+tp)/n if complete else None,'delta_C':c-old_c if complete else None,
                        'delta_A':a_count-old_a if complete else None,
                        'delta_J':(c+tp-old_c-old_tp)/n if complete else None,
                        'reference_positive':sum(r['uniform_reference'] for r in group),
                        **{before+'_'+after:transitions[before+'_'+after] for before in 'CWA' for after in 'CWA'},
                        'scope':'paired fixed representative subset; no full-eval replacement'})
    for name,values in [('scores.jsonl',scored),('pending_complete_QA.jsonl',list(pending.values()))]:
        with (out/name).open('w') as f:
            for r in values:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    pd.DataFrame(metrics).to_csv(out/'paired_metrics.csv',index=False)
    atomic_json(out/'receipt.json',{'rows':len(scored),'pending_unique_QA':len(pending),
        'old_frozen_scores_sha256':file_hash(ROOT/OLD),'sources':source_rows,'references':ref_sources,
        'new_semantic_judgments':0,'complete':not pending,'GPU_initialized':False})
    print(json.dumps({'rows':len(scored),'pending_unique_QA':len(pending),'metrics':metrics}))

if __name__=='__main__':main()
