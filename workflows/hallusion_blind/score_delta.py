#!/usr/bin/env python3
"""Incremental, source-bound Hallusion quality and behavior on sealed EOS rows."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import Counter,defaultdict
from dataclasses import asdict,replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from kdm.io import file_hash,stable_hash
from kdm.decoding import DecodeConfig
from workflows.general_vqa_direct.score import rule_decision,check_decision,qa_key,rows,write_rows
from workflows.general_vqa_direct.hallusion_scoring import HallusionScoring,validate_review_author

LEGACY='outputs/general_vqa_direct/scoring_snapshot_0041/scores.jsonl'
LEGACY_SHA='76bb04b3510eefea75811f5f98ed43cd2f22ce4927f275c6b0b7b23733ac9bda'
FIELDS=('model','method','kind','marker','reference_marker','guided','reference_guided','replicate')
def joint_key(qa,quality):return stable_hash(['hallusion_joint_semantic_v1',qa,quality])
def behavior_of(row):
    return {f:row.get(f) for f in ('label','abstain','answer_text','predicted_answer','evidence_span')}
def same_decision(left,right):
    return all(left.get(f)==right.get(f) for f in ('label','abstain','answer_text','predicted_answer','evidence_span'))
def insert(index,key,value,semantic_fields):
    if key in index and any(index[key].get(f)!=value.get(f) for f in semantic_fields):
        raise ValueError('Conflicting exact semantic judgments require root resolution: '+key)
    index.setdefault(key,value)

def legacy_indices(path,expected_sha=LEGACY_SHA):
    if file_hash(path)!=expected_sha:raise ValueError('Frozen legacy scoring snapshot SHA differs')
    behaviors,qualities={},{}
    for line,row in rows(path):
        if row.get('dataset')!='hallusionbench':continue
        if row.get('score') is None or row.get('abstain') is None:continue
        if qa_key({'prompt':row['question']},row['answer'])!=row['qa_key']:
            raise ValueError('Legacy QA binding differs')
        behavior={**behavior_of(row),'question':row['question'],'answer':row['answer'],
                  'source':{'reuse_snapshot':str(path),'reuse_line':line,'snapshot_sha256':expected_sha,
                            'original_decision_source':row['decision_source']}}
        insert(behaviors,row['qa_key'],behavior,('label','abstain','answer_text','predicted_answer','evidence_span'))
        quality=row['quality']
        if row['quality_key']!=quality['quality_key'] or row['score']!=quality['score']:
            raise ValueError('Legacy quality score binding differs')
        candidate={'quality':quality,'question':row['question'],'answer':row['answer'],
                   'gt_answer_details':row['gt_answer_details'],'source':{
                       'reuse_snapshot':str(path),'reuse_line':line,'snapshot_sha256':expected_sha}}
        insert(qualities,row['quality_key'],candidate,('question','answer','gt_answer_details'))
        if qualities[row['quality_key']]['quality']['quality_label']!=quality['quality_label']:
            raise ValueError('Conflicting frozen quality judgments')
    return behaviors,qualities

def read_review_batches(folder):
    behaviors,qualities,claimed={}, {}, set()
    if folder is None or not folder.exists():return behaviors,qualities,claimed
    for batch in sorted(folder.iterdir()):
        pending=batch/'pending_review.jsonl'
        if not batch.is_dir() or not pending.exists() or (batch/'RELEASED.json').exists():continue
        requests={r['joint_key']:r for _,r in rows(pending)}
        if len(requests)!=sum(1 for _ in rows(pending)):raise ValueError('Duplicate joint review assignment')
        claimed.update(requests)
        if (batch/'REVIEW_HOLD.json').exists():continue
        active=batch/'ACTIVE.json'
        evidence=[batch/n for n in ('write_receipt.json','receipt.json','root_validation_receipt.json')]
        if not active.exists() and not any(p.exists() for p in evidence):continue
        path=batch/'decisions.jsonl'
        if active.exists():
            selection=json.loads(active.read_text())
            path=batch/selection['filename']
            if path.parent!=batch or file_hash(path)!=selection['sha256']:
                raise ValueError('Active joint review source differs')
        if not path.exists():raise ValueError('Completed review has no decisions')
        records=list(rows(path))
        if {r['joint_key'] for _,r in records}!=set(requests) or len(records)!=len(requests):
            raise ValueError('Joint review completion does not cover its entire assigned batch')
        for line,r in records:
            request=requests[r['joint_key']]
            for f in ('joint_key','qa_key','quality_key','question','answer','gt_answer_details'):
                if r.get(f)!=request[f]:raise ValueError('Joint review full-input binding differs: '+f)
            if r['joint_key']!=joint_key(r['qa_key'],r['quality_key']):raise ValueError('Joint key differs')
            source={'decision_path':str(path),'decision_line':line,'decision_sha256':file_hash(path)}
            if request['need_behavior']:
                b=r.get('behavior_review')
                if not isinstance(b,dict):raise ValueError('Missing assigned behavior review')
                for f in ('qa_key','question','answer'):
                    if b.get(f)!=request[f]:raise ValueError('Behavior full QA differs')
                validate_review_author(b.get('author'))
                if b.get('model')!='gpt-5.6-luna' or b.get('effort')!='medium':
                    raise ValueError('New semantic review requires actual gpt-5.6-luna medium')
                if 'call_id' not in b or b['call_id'] is not None and (
                    not isinstance(b['call_id'],str) or not b['call_id'].strip()):
                    raise ValueError('Actual or unknown-null call_id required')
                candidate={**b['decision'],'question':b['question'],'answer':b['answer'],
                    'source':{**source,**{f:b[f] for f in ('author','model','effort','call_id')}}}
                insert(behaviors,r['qa_key'],candidate,('label','abstain','answer_text','predicted_answer','evidence_span'))
            if request['need_quality']:
                q=r.get('quality_review')
                if not isinstance(q,dict):raise ValueError('Missing assigned full-reference review')
                insert(qualities,r['quality_key'],{'review':q,'source':source},
                       ('review',))
    return behaviors,qualities,claimed

def csv_rows(path,values):
    fields=sorted({k for r in values for k in r})
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v
                         for k,v in r.items()} for r in values)

def validate_prediction_completion(row,condition,token_budget=None):
    """Validate the declared generation budget; never infer semantic quality."""
    if token_budget is None:
        if row.get('terminated') is not True or row.get('status')!='ok' or not row.get('tokens'):
            raise ValueError('Truncated/failed prediction is outside final Hall EOS scoring')
        return
    if token_budget!=128:raise ValueError('Only the explicitly authorized 128-token budget is supported')
    tok=row.get('tokens')
    if (row.get('status')!='ok' or not isinstance(tok,list) or not 1<=len(tok)<=token_budget
        or any(type(t)is not int or t<0 for t in tok) or type(row.get('terminated')) is not bool
        or type(row.get('truncated')) is not bool
        or row.get('truncated') is row['terminated']
        or row.get('generation_source') not in {'new_budget128','reused_budget128'}):
        raise ValueError('Prediction is not a validated 128-token generation')
    expected_finish='eos' if row['terminated'] else 'length'
    if row.get('finish_reason')!=expected_finish or not row['terminated'] and len(tok)!=token_budget:
        raise ValueError('Declared length/EOS termination differs from the 128-token output')
    cfg=DecodeConfig(**{**condition['config'],'max_tokens':token_budget})
    if cfg.method=='instruction_m3id':
        offset=row.get('offset_prompt_tokens')
        if not isinstance(offset,list) or any(type(t)is not int or t<0 for t in offset):
            raise ValueError('Missing registered M3ID prompt offset')
        cfg=replace(cfg,m3id_offset=len(offset))
    if row.get('config')!=asdict(cfg):
        raise ValueError('128-token generation config differs from the frozen Food operating point')

def score_rows(raw_paths,conditions,hallusion,legacy_behavior,legacy_quality,reviews,root=ROOT,token_budget=None):
    new_behavior,new_quality,claimed=reviews
    condition_index={stable_hash(c):c for c in conditions}
    if len(condition_index)!=len(conditions):raise ValueError('Duplicate frozen conditions')
    scores,pending,seen=[],{},set()
    counters=defaultdict(Counter);source_index=[];review_cache={}
    def original_record(path,line):
        p=Path(path);p=p if p.is_absolute() else root/p
        if p not in review_cache:review_cache[p]={n:r for n,r in rows(p)}
        return review_cache[p][line]
    for path in raw_paths:
        raw_sha=file_hash(path);source_index.append({'source_raw':str(path),'sha256':raw_sha})
        for line,row in rows(path):
            sample,answer=row['sample'],row['text']
            hallusion.source_record(sample)
            identity=(row['model'],row['key'])
            if identity in seen:raise ValueError('Duplicate generated model/condition/sample key')
            seen.add(identity)
            c=condition_index.get(row.get('condition_identity'))
            if c is None or any(row.get(f)!=c[f] for f in FIELDS):
                raise ValueError('Generated condition outside frozen Food operating points')
            validate_prediction_completion(row,c,token_budget)
            qa=qa_key(sample,answer);quality_key=hallusion.quality_key(sample,answer)
            joint=joint_key(qa,quality_key)
            b=new_behavior.get(qa) or legacy_behavior.get(qa)
            if b is None:
                decision=rule_decision(sample,answer)
                b=None if decision is None else {**decision,'question':sample['prompt'],'answer':answer,
                    'source':{'author':'deterministic_rule','rule':decision.get('decision_source')}}
            if b is not None:
                if b['question']!=sample['prompt'] or b['answer']!=answer:raise ValueError('Exact QA reuse differs')
                check_decision(sample,answer,b);hallusion.check_behavior_prediction(sample,b.get('predicted_answer'))
            quality=None;quality_source=None
            q=new_quality.get(quality_key)
            if q is not None:
                quality=hallusion.validate_quality(sample,answer,q['review']);quality_source=q['source']
            elif quality_key in legacy_quality:
                old=legacy_quality[quality_key]
                if old['answer']!=answer or old['gt_answer_details']!=sample['gt_answer_details']:
                    raise ValueError('Exact full-reference reuse differs')
                quality=old['quality'];quality_source=old['source']
                if quality.get('source')=='luna_full_reference_quality_v1':
                    reviewed=original_record(quality['decision_path'],quality['decision_line'])
                    validated=hallusion.validate_quality(sample,answer,reviewed)
                    if any(validated[f]!=quality[f] for f in ('quality_label','score','official_correctness')):
                        raise ValueError('Frozen quality source decision differs')
                else:
                    validated=hallusion.literal_binary_quality(sample,answer)
                    if validated!=quality:raise ValueError('Frozen literal quality source differs')
            if quality is None:
                quality=hallusion.literal_binary_quality(sample,answer)
                if quality is not None:quality_source={'author':'deterministic_rule','rule':quality['source']}
            membership={f:row[f] for f in FIELDS}|{'condition_identity':row['condition_identity'],
                'sample_id':sample['id'],'key':row['key'],'source_raw':str(path),'source_line':line,'source_sha256':raw_sha}
            out={**membership,'dataset':'hallusionbench','split':'blind_test','checkpoint':c.get('checkpoint'),
                'condition':c,'config':row.get('config'),'identity':row['identity'],
                'dataset_identity':row['dataset_identity'],'claim_identity':row['claim_identity'],
                'question':sample['prompt'],'answer':answer,'gt_answer_details':sample['gt_answer_details'],
                'qa_key':qa,'quality_key':quality_key,'joint_key':joint,'terminated':row['terminated'],
                'truncated':False if token_budget is None else row['truncated'],
                'finish_reason':'eos' if token_budget is None else row['finish_reason'],
                'output_token_budget':token_budget,'generation_source':row.get('generation_source'),
                'n_tokens':len(row['tokens']),'wall_s':row['wall_s'],'score':None if quality is None else quality['score'],
                'quality':quality,'quality_source':quality_source,'abstain':None if b is None else b['abstain'],
                'label':None if b is None else b['label'],'behavior':None if b is None else behavior_of(b),
                'behavior_source':None if b is None else b['source']}
            scores.append(out);counter=counters[row['condition_identity']]
            counter['generated']+=1;counter['pending_behavior']+=int(b is None)
            counter['truncated']+=int(out['truncated']);counter['natural_eos']+=int(out['terminated'])
            counter['pending_quality']+=int(quality is None)
            resolved=b is not None and quality is not None;counter['resolved']+=int(resolved)
            if quality is not None:counter['correct']+=quality['score']
            if b is not None:counter['abstentions']+=int(b['abstain'])
            if resolved:
                counter['A' if b['abstain'] else 'C' if quality['score']==1 else 'W']+=1
            if not resolved:
                request=hallusion.quality_request(sample,answer)
                item=pending.setdefault(joint,{**request,'qa_key':qa,'joint_key':joint,
                    'behavior_question':sample['prompt'],'need_behavior':b is None,'need_quality':quality is None,
                    'memberships':[]})
                item['memberships'].append(membership)
    metrics=[]
    for cid,c in condition_index.items():
        counter=counters[cid];n=counter['generated'];expected=c.get('hall_expected_n',951)
        if n>expected:raise ValueError('Condition exceeds frozen 951 denominator')
        final=n==expected and counter['resolved']==expected
        metrics.append({**{f:c[f] for f in FIELDS},'condition_identity':cid,'checkpoint':c.get('checkpoint'),
            'split':'blind_test','expected_n':expected,'generated':n,'missing_generation':expected-n,
            **{f:counter[f] for f in ('resolved','pending_behavior','pending_quality','correct','abstentions','C','W','A','truncated','natural_eos')},
            'output_token_budget':token_budget,
            'truncation_rate':counter['truncated']/expected if final else None,
            'observed_truncation_rate':counter['truncated']/n if n else None,
            'accuracy':counter['correct']/expected if final else None,
            'abstention_rate':counter['abstentions']/expected if final else None,
            'observed_accuracy':counter['correct']/n if n and not counter['pending_quality'] else None,
            'observed_abstention_rate':counter['abstentions']/n if n and not counter['pending_behavior'] else None,
            'status':'complete' if final else 'generation_incomplete' if n<expected else 'annotation_pending'})
    ready=[v for k,v in pending.items() if k not in claimed]
    held=[v for k,v in pending.items() if k in claimed]
    return scores,metrics,ready,held,source_index

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-list',type=Path,required=True)
    p.add_argument('--reviews-dir',type=Path)
    p.add_argument('--conditions',type=Path,default=ROOT/'outputs/hallusion_blind_20261005/registration/conditions.json')
    p.add_argument('--reuse-scores',type=Path,default=ROOT/LEGACY)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    paths=[Path(x) for x in args.raw_list.read_text().splitlines() if x.strip()]
    if not paths or len(paths)!=len(set(paths)):raise ValueError('Nonempty unique validated raw list required')
    hallusion=HallusionScoring(ROOT)
    old_behavior,old_quality=legacy_indices(args.reuse_scores)
    conditions=json.loads(args.conditions.read_text())
    result=score_rows(paths,conditions,hallusion,old_behavior,old_quality,read_review_batches(args.reviews_dir))
    scores,metrics,ready,held,sources=result
    args.out.mkdir(parents=True,exist_ok=False)
    write_rows(args.out/'scores.jsonl',scores);csv_rows(args.out/'scores.csv',scores)
    csv_rows(args.out/'condition_metrics.csv',metrics)
    write_rows(args.out/'pending_unique.jsonl',ready);write_rows(args.out/'claimed_pending.jsonl',held)
    csv_rows(args.out/'sources.csv',sources)
    receipt={'generated':len(scores),'resolved':sum(r['score'] is not None and r['abstain'] is not None for r in scores),
        'complete_conditions':sum(r['status']=='complete' for r in metrics),'expected_conditions':len(metrics),
        'pending_unique_unclaimed':len(ready),'pending_unique_claimed':len(held),
        'pending_behavior_rows':sum(r['abstain'] is None for r in scores),
        'pending_quality_rows':sum(r['score'] is None for r in scores),
        'raw_list_sha256':file_hash(args.raw_list),'conditions_sha256':file_hash(args.conditions),
        'legacy_scores_sha256':file_hash(args.reuse_scores),'scorer_sha256':file_hash(Path(__file__))}
    (args.out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
