#!/usr/bin/env python3
"""Lossless projection of extra API type metadata; no requests or semantic changes."""
import argparse, json, sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from annotate import digest, file_hash, jsonlines, parse_label, atomic_json
from score import score_run

def reconcile(root, source, target):
    if not target.resolve().is_relative_to(source.resolve()) or target.exists():
        raise ValueError('Fresh child output directory required')
    identity=json.loads((source/'identity.json').read_text())
    rows={r['key']:r for r in jsonlines(source/'queue.jsonl')}
    requests={r['key']:r for r in jsonlines(source/'requests.jsonl')}
    results={}; recovered={}; failures={}
    for r in jsonlines(source/'results.jsonl'):
        key=r['key']
        if key in results or key not in rows or r['identity']!=identity['identity']:
            raise ValueError('Invalid or duplicate original result identity')
        if r['request_sha256']!=requests[key]['request_sha256']:
            raise ValueError('Original result does not match original request')
        results[key]=r
        if r['result_status']=='error':
            accepted=False
            if r['error']=='ValueError: Exactly label, evidence_span, answer_text are required' and r['http_status']==200:
                payload=json.loads(r['response_body'])
                if payload['model']!=identity['definition']['settings']['response_model']:
                    raise ValueError('Original response model mismatch')
                choices=payload['choices']
                if len(choices)==1 and choices[0]['finish_reason']=='stop':
                    obj=json.loads(choices[0]['message']['content'])
                    if isinstance(obj,dict) and set(obj)=={'label','evidence_span','answer_text','type'}:
                        projection={k:obj[k] for k in ['label','evidence_span','answer_text']}
                        try:parsed=parse_label(json.dumps(projection),rows[key]['text'])
                        except (ValueError,TypeError,KeyError):pass
                        else:
                            assert all(parsed[k]==obj[k] for k in parsed)
                            recovered[key]=(parsed,obj['type']); accepted=True
            if not accepted:failures[key]=r
    if set(requests)!=set(rows) or set(results)!=set(rows) or len(rows)!=293344:
        raise ValueError('Incomplete original request/result coverage')
    originals={r['key']:r for r in jsonlines(source/'labels.jsonl')}
    if set(originals)!={k for k,r in results.items() if r['result_status']=='ok'}:
        raise ValueError('Original successful label coverage mismatch')
    if set(originals)|set(recovered)|set(failures)!=set(rows) or set(originals)&set(recovered):
        raise ValueError('Reconciled coverage mismatch')
    definition={**identity['definition'],'schema':'deepseek_v2_extra_type_projection',
        'parent_annotation_identity':identity['identity'],
        'parent_results_path':str((source/'results.jsonl').relative_to(root)),
        'parent_results_sha256':file_hash(source/'results.jsonl'),
        'normalization_source_sha256':file_hash(__file__),
        'normalization_policy':'Only remove the extra type field when exactly the three required fields plus type exist; preserve all three values byte-for-byte as strings; apply unchanged semantic/span validator; no model request or relabeling.'}
    new_identity=digest(definition);target.mkdir()
    atomic_json(target/'identity.json',{'identity':new_identity,'definition':definition})
    (target/'queue.sources.json').write_bytes((source/'queue.sources.json').read_bytes())
    with (target/'labels.jsonl').open('x') as stream:
        for key,row in rows.items():
            if key in failures:continue
            if key in originals:
                label=dict(originals[key]);parse_label(json.dumps({a:label[a] for a in ['label','evidence_span','answer_text']}),row['text'])
            else:
                ann,_=recovered[key];r=results[key]
                label={**row,**ann,'evidence':ann['evidence_span'],'label_source':'blinded_deepseek_api',
                    'judge_response_model':r['response_model'],'judge_response_id':r['response_id'],
                    'mechanical_normalization':'drop_extra_type_only'}
            label['parent_annotation_identity']=identity['identity'];label['identity']=new_identity
            stream.write(json.dumps(label,ensure_ascii=False)+'\n')
    with (target/'errors.jsonl').open('x') as stream:
        for r in failures.values():stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (target/'projection_records.jsonl').open('x') as stream:
        for key,(ann,extra) in recovered.items():
            stream.write(json.dumps({'key':key,'original_result_sha256':digest(results[key]),
                'removed_type':extra,'unchanged_required_fields':ann,'new_identity':new_identity},ensure_ascii=False)+'\n')
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'original_identity':identity['identity'],
        'derived_identity':new_identity,'expected':len(rows),'original_success':len(originals),
        'recovered_extra_type':len(recovered),'validated_labels':len(originals)+len(recovered),
        'unresolved':len(failures),'unresolved_error_types':dict(Counter(r['error'] for r in failures.values())),
        'all_keys_accounted_for':True,'all_annotations_complete':not failures,
        'additional_api_requests':0,'semantic_label_values_changed':0,'human_review_completed':False}
    atomic_json(target/'reconciliation.json',receipt)
    score_run(root,target)
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();source=root/'outputs/annotations/deepseek_v2/census'
    reconcile(root,source,source/'reconciled_extra_type_v1')
