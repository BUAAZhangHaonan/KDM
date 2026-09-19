#!/usr/bin/env python3
"""Label every queued response with a blinded local judge; disagreements stay visible."""
import argparse,json,sys
from pathlib import Path
from urllib.parse import urlsplit
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from kdm.io import read_jsonl,Ledger,file_hash,within
from kdm.scoring import LABELS,semantic_judge_request


def parse_label(content,text):
    obj=json.loads(content)
    if obj.get('label') not in LABELS:raise ValueError('Invalid semantic label')
    evidence=obj.get('evidence_span','')
    if not isinstance(evidence,str) or (evidence and evidence not in text):raise ValueError('Evidence is absent from original response')
    if not evidence and text.strip():raise ValueError('Nonempty response needs an exact evidence span')
    answer=obj.get('answer_text','')
    if obj['label'] in {'answer_assertive','answer_uncertain'} and (not isinstance(answer,str) or not answer or answer not in text):raise ValueError('Answer span is absent')
    return obj


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--queue',required=True)
    p.add_argument('--endpoint',required=True);p.add_argument('--judge-model',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();u=urlsplit(a.endpoint)
    if u.hostname not in {'127.0.0.1','localhost','::1'}:raise ValueError('Use a local judge endpoint')
    ledger=Ledger(within(a.root,a.out),{'queue_sha':file_hash(a.queue),'judge':a.judge_model,'endpoint':a.endpoint})
    for row in read_jsonl(a.queue):
        key=row['key']
        if key in ledger.keys:continue
        if row.get('label') is not None:
            ledger.add(key,{**row,'label_source':'exact_phrase'});continue
        request=semantic_judge_request(row.get('question',''),row['text'])
        response=requests.post(a.endpoint.rstrip('/')+'/chat/completions',json={
            'model':a.judge_model,'messages':[{'role':'system','content':'Classify answer behavior. Output only JSON.'},
            {'role':'user','content':json.dumps(request,ensure_ascii=False)}],
            'temperature':0,'max_tokens':512,'response_format':{'type':'json_object'}},timeout=180)
        response.raise_for_status();obj=parse_label(response.json()['choices'][0]['message']['content'],row['text'])
        ledger.add(key,{'text':row['text'],**obj,'evidence':obj['evidence_span'],'label_source':'blinded_local_judge','question':row.get('question','')})

if __name__=='__main__':main()
