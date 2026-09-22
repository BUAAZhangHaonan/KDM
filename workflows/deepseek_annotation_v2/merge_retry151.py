#!/usr/bin/env python3
"""Merge only validated labels from the one authorized retry, preserving history."""
import argparse,json
from pathlib import Path
from collections import Counter
from annotate import digest,file_hash,jsonlines,parse_label,atomic_json
from score import score_run

def merge(root):
    base=root/'outputs/annotations/deepseek_v2/census';parent=base/'reconciled_extra_type_v1';retry=base/'retry151_once_v1';out=base/'merged_after_retry151_v1'
    if out.exists():raise ValueError('Fresh derived output required')
    pid=json.loads((parent/'identity.json').read_text());rid=json.loads((retry/'identity.json').read_text())
    original_errors={r['key'] for r in jsonlines(parent/'errors.jsonl')}
    attempted=list(jsonlines(retry/'requests.jsonl'));results=list(jsonlines(retry/'results.jsonl'))
    if len(attempted)!=151 or len({r['key'] for r in attempted})!=151 or {r['key'] for r in attempted}!=original_errors:raise ValueError('Retry exceeded or missed authorized scope')
    if len(results)!=151 or len({r['key'] for r in results})!=151 or {r['key'] for r in results}!=original_errors:raise ValueError('Incomplete retry result coverage')
    delta=list(jsonlines(retry/'labels.jsonl'));errors=list(jsonlines(retry/'errors.jsonl'))
    if {r['key'] for r in delta}&{r['key'] for r in errors} or {r['key'] for r in delta}|{r['key'] for r in errors}!=original_errors:raise ValueError('Invalid retry label/failure coverage')
    definition={**pid['definition'],'schema':'deepseek_v2_merge_authorized_retry151',
        'parent_validated_identity':pid['identity'],'retry_identity':rid['identity'],
        'retry_authorization_sha256':file_hash(retry/'authorization.json'),
        'parent_labels_sha256':file_hash(parent/'labels.jsonl'),'retry_labels_sha256':file_hash(retry/'labels.jsonl'),
        'merge_source_sha256':file_hash(__file__),'policy':'Preserve every previously validated label and add only valid labels from the 151 once-retried failures; no relabeling or further API request'}
    identity=digest(definition);out.mkdir();atomic_json(out/'identity.json',{'identity':identity,'definition':definition})
    (out/'queue.sources.json').write_bytes((parent/'queue.sources.json').read_bytes())
    (out/'labels_delta.jsonl').write_bytes((retry/'labels.jsonl').read_bytes())
    (out/'errors.jsonl').write_bytes((retry/'errors.jsonl').read_bytes())
    seen=set()
    with (out/'labels.jsonl').open('x') as stream:
        for source,expected in [(parent,pid['identity']),(retry,rid['identity'])]:
            for r in jsonlines(source/'labels.jsonl'):
                if r['identity']!=expected or r['key'] in seen:raise ValueError('Duplicate or mismatched label identity')
                parse_label(json.dumps({k:r[k] for k in ['label','evidence_span','answer_text']}),r['text'])
                seen.add(r['key']);r['source_annotation_identity']=r['identity'];r['identity']=identity
                stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    queue={r['key'] for r in jsonlines(base/'queue.jsonl')}
    if len(queue)!=293344 or seen|{r['key'] for r in errors}!=queue or seen&{r['key'] for r in errors}:raise ValueError('Full merged coverage mismatch')
    receipt={'identity':identity,'expected':293344,'previous_validated':293193,'retry_requests':151,
        'retry_success':len(delta),'retry_failed':len(errors),'validated_labels':len(seen),
        'unresolved':len(errors),'unresolved_error_types':dict(Counter(r['error'] for r in errors)),
        'all_keys_accounted_for':True,'all_annotations_complete':not errors,'further_retries_performed':False,
        'reconstruction':'Use previous reconciled label archive plus labels_delta.jsonl; change only provenance identity fields as this script specifies. Original labels and all failed attempts preserved.'}
    atomic_json(out/'merge_receipt.json',receipt)
    score_run(root,out)
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();merge(Path(a.root).resolve())
