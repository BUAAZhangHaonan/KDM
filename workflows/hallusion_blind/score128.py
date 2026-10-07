#!/usr/bin/env python3
"""Score a separate 128-token Hallusion panel from actual generated answer text."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from kdm.io import file_hash,stable_hash
from workflows.general_vqa_direct.score import write_rows
from workflows.hallusion_blind.complete_response_quality import CompleteResponseScoring,RULE_VERSION
from workflows.hallusion_blind.score_delta import (
    LEGACY,legacy_indices,read_review_batches,insert,score_rows,csv_rows)

BASE=ROOT/'outputs/hallusion_blind128_20261006'
OLD_BASE=ROOT/'outputs/hallusion_blind_20261005'
EXCLUDED_SID={
    'qwen25vl':'Full951 original SID not applicable: 30 inputs have fewer than the required 100 visual tokens.',
    'qwen3vl':'Full951 original SID not applicable: 59 inputs have fewer than the required 100 visual tokens.'}

def eligible_conditions(conditions):
    eligible=[];applicability=[]
    for c in conditions:
        reason=EXCLUDED_SID.get(c['model']) if c['method']=='sid' else None
        applicability.append({'model':c['model'],'method':c['method'],
            'condition_identity':stable_hash(c),'registered_denominator':951,
            'required_generation':0 if reason else 951,
            'status':'not_applicable_full_panel' if reason else 'candidate',
            'reason':reason or ''})
        if not reason:eligible.append(c)
    for model,reason in [
        ('minicpm26','Original SID attention/visual-token architecture is not applicable; no adaptation.'),
        ('qwen35_4b','Original SID requires second-layer attention weights, but this model has linear attention; no adaptation.')]:
        applicability.append({'model':model,'method':'sid','condition_identity':None,
            'registered_denominator':951,'required_generation':0,'status':'architecture_not_applicable','reason':reason})
    return eligible,applicability

def review_indices(folders):
    behaviors,qualities,claimed={}, {}, set()
    for folder in folders:
        b,q,c=read_review_batches(folder)
        for key,value in b.items():
            insert(behaviors,key,value,('label','abstain','answer_text','predicted_answer','evidence_span'))
        for key,value in q.items():insert(qualities,key,value,('review',))
        claimed.update(c)
    return behaviors,qualities,claimed

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-list',type=Path,default=BASE/'validated_raw_list.txt')
    p.add_argument('--conditions',type=Path,default=BASE/'registration/conditions.json')
    p.add_argument('--reviews-dir',type=Path,action='append')
    p.add_argument('--reuse-scores',type=Path,default=ROOT/LEGACY)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    paths=[Path(x) for x in args.raw_list.read_text().splitlines() if x.strip()]
    if len(paths)!=len(set(paths)):raise ValueError('Unique validated raw list required')
    conditions=json.loads(args.conditions.read_text())
    eligible,applicability=eligible_conditions(conditions)
    if len(conditions)!=70 or len(eligible)!=68:
        raise ValueError('Frozen registered/candidate condition counts differ from 70/68')
    old_behavior,old_quality=legacy_indices(args.reuse_scores)
    folders=args.reviews_dir or [OLD_BASE/'annotations',BASE/'annotations']
    scored=score_rows(paths,eligible,CompleteResponseScoring(ROOT),old_behavior,old_quality,
                      review_indices(folders),token_budget=128)
    scores,metrics,ready,held,sources=scored
    args.out.mkdir(parents=True,exist_ok=False)
    write_rows(args.out/'scores.jsonl',scores);csv_rows(args.out/'scores.csv',scores)
    csv_rows(args.out/'condition_metrics.csv',metrics)
    csv_rows(args.out/'method_applicability.csv',applicability)
    write_rows(args.out/'pending_unique.jsonl',ready);write_rows(args.out/'claimed_pending.jsonl',held)
    csv_rows(args.out/'sources.csv',sources)
    receipt={'schema':'hallusion_score_budget128_v1','output_token_budget':128,
        'termination_policy':'EOS_or_128_tokens','score_policy':'actual_generated_text_quality_and_behavior_independent_of_finish_reason',
        'generated':len(scores),'natural_eos':sum(r['terminated'] for r in scores),
        'truncated':sum(r['truncated'] for r in scores),
        'resolved':sum(r['score'] is not None and r['abstain'] is not None for r in scores),
        'complete_conditions':sum(r['status']=='complete' for r in metrics),
        'expected_conditions':len(eligible),'registered_conditions':len(conditions),
        'candidate_denominator':len(eligible)*951,
        'pending_unique_unclaimed':len(ready),'pending_unique_claimed':len(held),
        'pending_behavior_rows':sum(r['abstain'] is None for r in scores),
        'pending_quality_rows':sum(r['score'] is None for r in scores),
        'raw_list_sha256':file_hash(args.raw_list),'conditions_sha256':file_hash(args.conditions),
        'legacy_scores_sha256':file_hash(args.reuse_scores),
        'review_directories':[str(f) for f in folders],
        'scorer_sha256':file_hash(Path(__file__)),
        'shared_score_sha256':file_hash(ROOT/'workflows/hallusion_blind/score_delta.py')}
    receipt.update({'complete_response_rule_version':RULE_VERSION,
        'complete_response_rule_sha256':file_hash(ROOT/'workflows/hallusion_blind/complete_response_quality.py'),
        'complete_response_rule_rows':sum(r['quality'] is not None and r['quality'].get('source')==RULE_VERSION for r in scores)})
    (args.out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
