#!/usr/bin/env python3
"""Compile explicitly accepted semantic decisions, never infer unread labels."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from workflows.main_results.score import qah
from workflows.supplemental.remaining4.score_received import load_viz_authority

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p): return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotation',required=True)
    ap.add_argument('--accepted-manifest',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    folder,manifest,out=(Path(x).resolve() for x in (a.annotation,a.accepted_manifest,a.output))
    ids={x['id']:x['qa_key'] for x in json.loads((folder/'short_id_mapping.json').read_text(encoding='utf-8-sig'))}
    inputs={x['qa_key']:x for x in lines(folder/'pending_QA_with_sources.jsonl')}
    accepted=json.loads(manifest.read_text(encoding='utf-8-sig'))
    decisions={}
    for source in accepted['sources']:
        path=Path(source['path'])
        if not path.is_absolute():path=folder/path
        if sha(path)!=source['sha256']:raise ValueError('Accepted source SHA changed')
        if source['status']!='semantically_reviewed':raise ValueError('A draft was not accepted')
        values=lines(path)
        if len(values)!=source['rows']:raise ValueError('Source count changed')
        for line,v in enumerate(values,1):
            k=ids[v['id']]; original=inputs[k]
            if qah(original['question'],original['answer'])!=k:raise ValueError('Exact QA identity mismatch')
            if v.get('question',original['question'])!=original['question'] or v.get('answer',original['answer'])!=original['answer']:raise ValueError('Source decision changed QA')
            if v.get('uncertain') is not False or type(v['abstain']) is not bool:raise ValueError('Unresolved decision cannot become authority')
            if not isinstance(v.get('span'),str) or not v['span'] or v['span'] not in original['answer']:raise ValueError('Missing continuous original evidence span')
            if v['label'] not in {'answered','multiple','abstain','invalid'}:raise ValueError('Unknown label')
            if (v['label']=='abstain')!=v['abstain']:raise ValueError('Behavior mismatch')
            if k in decisions:raise ValueError('Overlapping authorities require explicit adjudication')
            label='abstain' if v['abstain'] else ('invalid' if v['label']=='invalid' else 'answer_assertive')
            decisions[k]={
                'qa_key':k,'question':original['question'],'answer':original['answer'],
                'abstain':v['abstain'],'label':label,
                'answer_text':'' if label in {'abstain','invalid'} else v['span'],
                'evidence_span':v['span'],'semantic_subtype':v['label'],
                'annotation_complete':True,'behavior_resolved':True,'quality_span_resolved':True,
                'reason':v['reason'],'actual_author':v.get('actual_author',source['author']),
                'actual_model':v.get('model',source['model']),'actual_annotation':v,
                'actual_effort':source['effort'],'call_id':source.get('call_id',''),
                'source_decision_path':str(path),'source_decision_sha256':sha(path),'source_decision_line':line,
                'raw_memberships':original['occurrences'], 'full_reply_fallback_used':False,
            }
    out.mkdir(parents=True,exist_ok=True)
    target=out/'qa_authority.jsonl'
    tmp=out/'qa_authority.tmp'
    tmp.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for _,v in sorted(decisions.items())),encoding='utf-8')
    tmp.replace(target)
    receipt={'schema':'kdm_actual_finite_semantic_authority_v1','passed':True,'unique_QA':len(decisions),'closed_QA':len(decisions),'pending_QA':0,
        'scope':'only explicitly listed accepted subset, not full Viz capability closure',
        'global_unique_QA':len(inputs),'global_remaining_QA':len(inputs)-len(decisions),
        'manifest_sha256':sha(manifest),'outputs':{target.name:sha(target)},'updated_utc':datetime.now(timezone.utc).isoformat()}
    (out/'receipt.json').write_text(json.dumps({'schema':'validation_pending','passed':False})+'\n')
    loaded=load_viz_authority([target])
    if len(loaded)!=len(decisions):raise ValueError('Existing authority loader rejected coverage')
    (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
