"""Full-output annotation queues; unknown wording remains explicit work."""
import json
from pathlib import Path
from .io import read_jsonl,atomic_json
from .scoring import lexical_label,semantic_judge_request,LABELS


def build_queue(paths,out):
    rows=[]
    for path in paths:
        for r in read_jsonl(path):
            fast=lexical_label(r['text'])
            rows.append({'key':r['key'],'text':r['text'],'question':r['sample']['question'],
                         'label':fast,'source':'exact_phrase' if fast else 'requires_semantic_review',
                         'evidence':r['text'] if fast else None,'answer_text':'' if fast else None})
    with open(out,'w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    return len(rows)


def validate_annotations(queue):
    ann={}
    for r in read_jsonl(queue):
        if r.get('label') not in LABELS:raise ValueError(f"Unresolved response: {r['key']}")
        evidence=r.get('evidence')
        if r['label']!='invalid' and (not evidence or evidence not in r['text']):
            raise ValueError('Annotation evidence must quote an exact span')
        if r['key'] in ann:raise ValueError('Duplicate annotation')
        answer=r.get('answer_text','')
        if r['label'] in {'answer_assertive','answer_uncertain'} and (not answer or answer not in r['text']):
            raise ValueError('A concrete answer needs its exact text span')
        ann[r['key']]={k:r[k] for k in ('text','label','evidence','answer_text')}
    return ann
