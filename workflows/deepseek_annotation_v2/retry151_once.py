#!/usr/bin/env python3
"""Exactly one user-authorized retry for the 151 unresolved original responses."""
import argparse,asyncio,json,signal,sys
from pathlib import Path
import httpx
from annotate import Runner,SYSTEM,request_body,digest,file_hash,jsonlines,atomic_json

def main(root):
    source=root/'outputs/annotations/deepseek_v2/census'
    previous=source/'reconciled_extra_type_v1'
    target=source/'retry151_once_v1'
    if target.exists():raise ValueError('Retry already initialized; never resend automatically')
    errors=list(jsonlines(previous/'errors.jsonl')); keys={r['key'] for r in errors}
    if len(errors)!=151 or len(keys)!=151:raise ValueError('Exactly the authorized 151 unique failures required')
    rows=[r for r in jsonlines(source/'queue.jsonl') if r['key'] in keys]
    if len(rows)!=151:raise ValueError('Incomplete original retry inputs')
    original=json.loads((source/'identity.json').read_text());settings=original['definition']['settings']
    if SYSTEM!=original['definition']['system_prompt'] or file_hash(Path(__file__).with_name('annotate.py'))!=original['definition']['client_sha256']:
        raise ValueError('Original prompt/client changed')
    errmap={r['key']:r for r in errors}
    for row in rows:
        if digest(request_body(row,settings))!=errmap[row['key']]['request_sha256']:
            raise ValueError('Retry request differs from original model/prompt/parameters')
    keypath=root/'cache/private/deepseek_api_key'
    if keypath.stat().st_mode&0o077:raise ValueError('Credential file permissions changed')
    credential=keypath.read_text().strip()
    target.mkdir()
    with (target/'queue.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    authorization={'user_authorization':'允许仅重试这 151 条一次','scope':'Only 151 remaining failures, once each, same API request bodies; preserve all previous records',
        'parent_annotation_identity':original['identity'],'parent_reconciliation_identity':json.loads((previous/'identity.json').read_text())['identity'],
        'previous_errors_sha256':file_hash(previous/'errors.jsonl'),'authorized_count':151,'request_bodies_exactly_match_original':True}
    atomic_json(target/'authorization.json',authorization)
    definition={**original['definition'],'schema':'deepseek_v2_explicit_retry151_once',
        'queue_sha256':file_hash(target/'queue.jsonl'),'parent_queue_sha256':original['definition']['queue_sha256'],
        'retry_authorization':authorization,'retry_entrypoint_sha256':file_hash(__file__)}
    runner=Runner(target,rows,definition)
    async def run():
        loop=asyncio.get_running_loop()
        for signum in (signal.SIGTERM,signal.SIGINT):
            loop.add_signal_handler(signum,lambda:setattr(runner,'stop_reason','Stop signal; drain requests; no further retry'))
        async with httpx.AsyncClient(headers={'Authorization':'Bearer '+credential},timeout=httpx.Timeout(settings['timeout_s']),
            follow_redirects=False,trust_env=False,limits=httpx.Limits(max_connections=settings['concurrency'],max_keepalive_connections=settings['concurrency'])) as client:
            code=await runner.run(client)
        status=json.loads((target/'status.json').read_text())
        print(json.dumps({k:status[k] for k in ['expected','attempted','success','failed','peak_concurrency','stop_reason','error_types']},ensure_ascii=False))
        return code
    return asyncio.run(run())
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    sys.exit(main(Path(args.root).resolve()))
