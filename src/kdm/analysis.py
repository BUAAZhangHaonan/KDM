"""Postprocessing with explicit answer semantics and image/category pairing."""
from collections import defaultdict
import json,csv
from pathlib import Path
import numpy as np
from .io import read_jsonl,atomic_json
from .scoring import label_response,food_correct,vqa_score,normalize
from .statistics import mean_ci,ratio_ci


def annotated_rows(paths,annotations,aliases,vqa_normalizer=None):
    result=[];seen=set()
    for path in paths:
        for row in read_jsonl(path):
            if row['key'] in seen:raise ValueError('Duplicated output key across shards')
            seen.add(row['key']);s=row['sample']
            label=label_response(row['text'],annotations,row['key'])
            answer=annotations.get(row['key'],{}).get('answer_text','')
            if label in {'answer_assertive','answer_uncertain'} and (not isinstance(answer,str) or not answer or answer not in row['text']):
                raise ValueError('Concrete scoring requires an annotated exact answer span')
            if s['dataset']=='food101':correct=float(food_correct(answer,s['class'],aliases))
            elif s['dataset']=='vizwiz':
                if vqa_normalizer is None:raise ValueError('Official VQA normalizer is required')
                correct=float(vqa_score(answer,s['gold'],vqa_normalizer))
            elif s['dataset']=='fixture':correct=float(normalize(row['text'])==normalize(s['gold'][0]))
            else:raise ValueError('Unknown dataset')
            if label in {'abstain','invalid'}:correct=0.
            result.append({**row,'label':label,'correct':correct})
    return result


def condition(row):
    return (row['model'],row['sample']['dataset'],row['marker'],row['method'],
            row['reference_marker'],row['reference_guided'],row['kind'])


def evaluate(rows,bootstrap=2000,seed=20260918):
    groups=defaultdict(dict);base={}
    for r in rows:
        if r['sample']['split']!='eval' or r['kind']=='independent_attempt':continue
        sid=r['sample']['id'];key=condition(r)
        if sid in groups[key]:raise ValueError('Duplicate condition/sample')
        groups[key][sid]=r
        if r['method']=='direct' and r['guided']:
            bkey=(r['model'],r['sample']['dataset'],r['marker'],sid)
            if bkey in base and (base[bkey]['text']!=r['text']):
                raise ValueError('Conflicting direct records; keep backend protocols separate')
            base[bkey]=r
    table=[]
    for key,records in sorted(groups.items()):
        model,dataset,marker,method,refmark,refguide,kind=key
        if method=='direct':continue
        expected={sid for m,d,mk,sid in base if (m,d,mk)==(model,dataset,marker)}
        if set(records)!=expected:raise ValueError('Method condition does not cover its entire paired direct set')
        ids=sorted(records);before=[];after=[]
        for sid in ids:
            bkey=(model,dataset,marker,sid)
            if bkey not in base:raise ValueError('Missing paired direct record')
            before.append(base[bkey]);after.append(records[sid])
        clusters=[r['sample']['cluster'] for r in before]
        ab=np.array([r['label']=='abstain' for r in before]);aa=np.array([r['label']=='abstain' for r in after])
        cb=np.array([r['correct'] for r in before]);ca=np.array([r['correct'] for r in after])
        answered=np.array([r['label'] in {'answer_assertive','answer_uncertain'} for r in after])
        lost=ab&answered;retained=ab&aa
        originally_answered=np.array([r['label'] in {'answer_assertive','answer_uncertain'} for r in before])
        record={'model':model,'dataset':dataset,'marker':marker,'method':method,
                'reference_marker':refmark,'reference_guided':refguide,'kind':kind,'n':len(ids),
                'n_answered':int(answered.sum()),'n_originally_answered':int(originally_answered.sum()),
                'n_new_abstentions':int((originally_answered&aa).sum()),
                'n_new_abstentions_on_originally_correct':int(((cb==1)&aa).sum()),
                'accuracy_denominator':len(ids),'answered_accuracy_denominator':int(answered.sum()),
                'n_base_abstain':int(ab.sum()),'n_lost':int(lost.sum()),'n_retained':int(retained.sum()),
                'n_invalid':sum(r['label']=='invalid' for r in after),
                'accuracy_direct':float(cb.mean()),'accuracy_method':float(ca.mean()),
                'accuracy_change':mean_ci(ca-cb,clusters,bootstrap,seed),
                'loss_rate':ratio_ci(lost.astype(float),ab.astype(float),clusters,bootstrap,seed),
                'lost_correct_credit':float(ca[lost].sum()),
                'answer_rate':float(answered.mean()),
                'answered_accuracy':float(ca[answered].mean()) if answered.any() else None,
                'abstention_to_invalid':int(sum(a and r['label']=='invalid' for a,r in zip(ab,after)))}
        table.append(record)
    return table


def copy_abstention_comparator(direct,method):
    """Return the original full response on baseline abstentions; no token rewriting."""
    if set(direct)!=set(method):raise ValueError('Unpaired comparator inputs')
    return {key:direct[key] if direct[key]['label']=='abstain' else method[key] for key in direct}


def probe_summary(rows,repeats=10):
    groups=defaultdict(list)
    for r in rows:
        if r['kind']=='independent_attempt':groups[(r['model'],r['sample']['id'])].append(r)
    result=[]
    for (model,sid),values in sorted(groups.items()):
        reps=[r['replicate'] for r in values]
        if len(values)!=repeats or set(reps)!=set(range(repeats)):raise ValueError('Incomplete repeated attempts')
        credit=sum(r['correct'] for r in values)
        external_unanswerable=values[0]['sample'].get('annotated_answerable') is False or values[0]['sample'].get('annotated_answerable')==0
        result.append({'model':model,'sample_id':sid,'n':repeats,'mean_correctness':None if external_unanswerable else credit/repeats,
                       'annotation_answerable':values[0]['sample'].get('annotated_answerable'),
                       'n_full_correct':sum(r['correct']==1 for r in values),
                       'n_positive_credit':sum(r['correct']>0 for r in values),
                       'total_correctness_credit':float(credit),
                       'n_abstain_during_attempt':sum(r['label']=='abstain' for r in values),
                       'unique_normalized_answers':len({normalize(r['text']) for r in values})})
    return result


def export_csv(rows,path):
    fields=sorted({k for r in rows for k in r});Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
