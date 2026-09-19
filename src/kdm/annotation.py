"""Full-output annotation queues; unknown wording remains explicit work."""
import json
from pathlib import Path
from .io import read_jsonl,atomic_json
from .scoring import lexical_label,semantic_judge_request,LABELS


def build_queue(paths,out):
    rows=[];seen=set()
    for path in paths:
        for r in read_jsonl(path):
            if r['key'] in seen:raise ValueError('Duplicate response key in annotation inputs')
            seen.add(r['key'])
            if not isinstance(r.get('text'),str):raise ValueError('Annotation requires the original response text')
            fast=lexical_label(r['text'])
            rows.append({'key':r['key'],'text':r['text'],'question':r['sample']['question'],
                         'label':fast,'source':'exact_phrase' if fast else 'requires_semantic_review',
                         'evidence':r['text'] if fast else None,'answer_text':'' if fast else None})
    payload=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
    out=Path(out)
    if out.exists():
        if out.read_text(encoding='utf-8')!=payload:raise ValueError('Existing annotation queue differs; use a new output path')
    else:
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x',encoding='utf-8') as f:f.write(payload)
    return len(rows)


def validate_annotations(queue):
    ann={}
    for r in read_jsonl(queue):
        if r.get('label') not in LABELS:raise ValueError(f"Unresolved response: {r['key']}")
        if not isinstance(r.get('text'),str):raise ValueError('Annotation text must be a string')
        evidence=r.get('evidence')
        if evidence is not None and (not isinstance(evidence,str) or evidence not in r['text']):
            raise ValueError('Annotation evidence must quote an exact span')
        if r['label']!='invalid' and (not evidence or evidence not in r['text']):
            raise ValueError('Annotation evidence must quote an exact span')
        if r['key'] in ann:raise ValueError('Duplicate annotation')
        answer=r.get('answer_text','')
        if not isinstance(answer,str):raise ValueError('Answer span must be a string')
        if r['label'] in {'abstain','invalid'} and answer:raise ValueError('Abstention or invalid response cannot endorse an answer span')
        if r['label'] in {'answer_assertive','answer_uncertain'} and (not answer or answer not in r['text']):
            raise ValueError('A concrete answer needs its exact text span')
        ann[r['key']]={'text':r['text'],'label':r['label'],'evidence':evidence,'answer_text':answer}
    return ann
