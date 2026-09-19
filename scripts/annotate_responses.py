#!/usr/bin/env python3
"""Label every queued response with a blinded local judge; disagreements stay visible."""
import argparse,json,sys,re
from pathlib import Path
from urllib.parse import urlsplit
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from kdm.io import read_jsonl,Ledger,file_hash,within,stable_hash
from kdm.scoring import LABELS,semantic_judge_request,lexical_label


def parse_label(content,text):
    if not isinstance(content,str):raise ValueError('Judge content must be text')
    envelope=re.fullmatch(r"\s*```json[ \t]*\r?\n(.*?)\r?\n```\s*",content,re.DOTALL)
    parse_format='single_complete_json_fence' if envelope else 'raw_json'
    obj=json.loads(envelope.group(1) if envelope else content)
    if not isinstance(obj,dict):raise ValueError('Semantic response must be a JSON object')
    if obj.get('label') not in LABELS:raise ValueError('Invalid semantic label')
    evidence=obj.get('evidence_span','')
    if not isinstance(evidence,str) or (evidence and evidence not in text):raise ValueError('Evidence is absent from original response')
    if not evidence and text.strip():raise ValueError('Nonempty response needs an exact evidence span')
    answer=obj.get('answer_text','')
    if not isinstance(answer,str):raise ValueError('Answer span must be a string')
    if obj['label'] in {'abstain','invalid'} and answer:raise ValueError('Abstention or invalid output cannot endorse an answer span')
    if obj['label'] in {'answer_assertive','answer_uncertain'} and (not isinstance(answer,str) or not answer or answer not in text):raise ValueError('Answer span is absent')
    return {'label':obj['label'],'evidence_span':evidence,'answer_text':answer,'parse_format':parse_format}


def validate_judge_identity(session,endpoint,spec_path,model,root):
    spec=json.loads(Path(spec_path).read_text());endpoint=endpoint.rstrip('/')
    if spec.get('alias')!=model or spec.get('endpoint','').rstrip('/')!=endpoint:
        raise ValueError('Requested judge alias or endpoint differs from fixed judge spec')
    panel=json.loads((Path(root)/'configs/kdm/models.json').read_text())
    subjects={row['hf_model_id'] for row in panel};independence=spec.get('independence',{})
    if independence.get('independent_checkpoint') is not True or set(independence.get('subject_hf_model_ids',[]))!=subjects or spec.get('hf_model_id') in subjects:
        raise ValueError('Judge checkpoint is not registered independently of the frozen subject panel')
    response=session.get(endpoint+'/identity',timeout=30,allow_redirects=False)
    response.raise_for_status()
    if 300<=response.status_code<400:raise ValueError('Local judge identity redirects are not allowed')
    identity=response.json();receipt=identity.get('receipt')
    if not isinstance(receipt,dict) or stable_hash(receipt)!=identity.get('receipt_sha256'):
        raise ValueError('Judge receipt digest mismatch')
    if receipt.get('spec_sha256')!=file_hash(spec_path) or receipt.get('spec')!=spec or receipt.get('model')!=model:
        raise ValueError('Running judge does not match the fixed checkpoint specification')
    if receipt.get('versions')!=spec.get('versions') or receipt.get('device_map')!=spec.get('device_map'):
        raise ValueError('Running judge software or device map differs from fixed specification')
    cards=receipt.get('physical_gpus',[])
    if len(cards)!=spec.get('gpu_count') or len(cards)!=len(set(cards)) or any(card not in {'0','1','4','5'} for card in cards):
        raise ValueError('Judge receipt physical GPU allocation is invalid')
    if receipt.get('parameter_devices')!=['cuda:0'] or not receipt.get('service_id') or not receipt.get('started_utc'):
        raise ValueError('Judge receipt lacks actual local-device service evidence')
    if receipt.get('script_sha256')!=file_hash(Path(root)/'scripts/serve_annotation_judge.py'):
        raise ValueError('Judge service code differs from its launch receipt')
    return identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--queue',required=True)
    p.add_argument('--endpoint',required=True);p.add_argument('--judge-model',required=True);p.add_argument('--judge-spec',required=True);p.add_argument('--out',required=True)
    p.add_argument('--retry-failed',action='store_true',help='Explicitly retry failed keys under the same immutable annotation identity')
    a=p.parse_args();u=urlsplit(a.endpoint)
    if u.scheme not in {'http','https'} or u.username or u.password or u.query or u.fragment or u.hostname not in {'127.0.0.1','localhost','::1'}:raise ValueError('Use a local judge endpoint')
    session=requests.Session();session.trust_env=False
    identity=validate_judge_identity(session,a.endpoint,a.judge_spec,a.judge_model,a.root)
    out=within(a.root,a.out)
    ledger=Ledger(out,{'queue_sha':file_hash(a.queue),'judge':a.judge_model,'endpoint':a.endpoint,
        'judge_spec_sha256':file_hash(a.judge_spec),'judge_receipt':identity['receipt'],
        'judge_receipt_sha256':identity['receipt_sha256'],'client_script_sha256':file_hash(__file__)})
    rows=list(read_jsonl(a.queue))
    if len({row['key'] for row in rows})!=len(rows):raise ValueError('Duplicate annotation queue key')
    queue_keys={row['key'] for row in rows}
    if not ledger.keys<=queue_keys:raise ValueError('Annotation ledger contains unknown queue keys')
    errors_path=Path(str(out)+'.errors.jsonl');failed=set()
    if errors_path.exists():
        for error in read_jsonl(errors_path):
            if error.get('annotation_identity')!=ledger.identity or error.get('queue_sha256')!=file_hash(a.queue) or error.get('key') not in queue_keys:
                raise ValueError('Previous annotation failure identity differs; use the exact original inputs and identity')
            failed.add(error['key'])
    if not out.exists():out.touch(exist_ok=False)
    for row in rows:
        key=row['key']
        if key in ledger.keys or (key in failed and not a.retry_failed):continue
        if row.get('label') is not None:
            if row.get('source')!='exact_phrase' or row['label']!=lexical_label(row['text']):
                raise ValueError('Only verified exact-phrase labels may bypass the semantic judge')
            ledger.add(key,{**row,'label_source':'exact_phrase'});continue
        request=semantic_judge_request(row.get('question',''),row['text'])
        response=None
        try:
            response=session.post(a.endpoint.rstrip('/')+'/chat/completions',json={
                'model':a.judge_model,'messages':[{'role':'system','content':'Classify answer behavior. Output only JSON.'},
                {'role':'user','content':json.dumps(request,ensure_ascii=False)}],
                'temperature':0,'max_tokens':512,'response_format':{'type':'json_object'}},timeout=180,allow_redirects=False)
            response.raise_for_status()
            if 300<=response.status_code<400:raise ValueError('Local judge redirects are not allowed')
            payload=response.json()
            if payload.get('model')!=a.judge_model:raise ValueError('Judge response model differs from requested model identity')
            if payload.get('kdm_judge_receipt_sha256')!=identity['receipt_sha256']:
                raise ValueError('Judge service identity changed during annotation')
            choice=payload['choices'][0]
            if choice.get('finish_reason')!='stop':raise ValueError('Judge output is incomplete; preserve failure for human handling')
            content=choice['message']['content'];obj=parse_label(content,row['text'])
        except Exception as exc:
            error={'key':key,'text':row['text'],'question':row.get('question',''),
                   'annotation_identity':ledger.identity,'queue_sha256':file_hash(a.queue),
                   'judge_receipt_sha256':identity['receipt_sha256'],
                   'error':type(exc).__name__+': '+str(exc),
                   'response_text':response.text if response is not None else None}
            with Path(str(out)+'.errors.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(error,ensure_ascii=False)+'\n')
            continue
        ledger.add(key,{'text':row['text'],**obj,'evidence':obj['evidence_span'],
            'label_source':'blinded_local_judge','question':row.get('question',''),
            'judge_response_model':payload['model'],'judge_response_id':payload.get('id'),
            'judge_raw_content':content,'judge_receipt_sha256':identity['receipt_sha256']})

    missing=sorted(queue_keys-ledger.keys)
    print(json.dumps({'expected':len(queue_keys),'completed':len(ledger.keys),'unresolved_keys':missing,'all_complete':not missing},ensure_ascii=False))
    if missing:raise SystemExit(1)

if __name__=='__main__':main()
