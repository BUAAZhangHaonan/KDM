#!/usr/bin/env python3
"""Download explicitly listed research assets; keep all writes in the project."""
from pathlib import Path
import argparse,json,sys,re,hashlib,ast,datetime,traceback,base64
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE/'src'))
from kdm.data import download,extract_safe
from kdm.io import within,atomic_json,file_hash


def extract_vqa_normalizer(source):
    lines=source.expandtabs(4).splitlines();wanted={'__init__','processPunctuation','processDigitArticle'}
    selected=[];found=set();i=0
    while i<len(lines):
        match=re.match(r'^    def (\w+)\(',lines[i])
        if match and match.group(1) in wanted:
            j=i+1
            while j<len(lines) and not re.match(r'^    def \w+\(',lines[j]):j+=1
            found.add(match.group(1));selected.extend(lines[i:j]);i=j
        else:i+=1
    if found!=wanted:raise ValueError('Official normalization source changed')
    result='import re\nclass VQAEval:\n'+'\n'.join(selected)+'\n'
    tree=ast.parse(result)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    fields={'contractions','manualMap','articles','periodStrip','commaStrip','punct'}
    init.body=[node for node in init.body if isinstance(node,ast.Assign) and any(
        isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name)
        and target.value.id=='self' and target.attr in fields for target in node.targets)]
    assigned={target.attr for node in init.body for target in node.targets if isinstance(target,ast.Attribute)}
    if assigned!=fields:raise ValueError('Official normalization fields changed')
    ast.fix_missing_locations(tree)
    result=ast.unparse(tree)+'\n';compile(result,'normalizer','exec');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--group',choices=['papers','vizwiz','vqa_code'],required=True);a=p.parse_args()
    root=Path(a.root).resolve();sources=json.load(open(a.manifest));report=[]
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report_path=within(root,'outputs/records/assets_'+a.group+'_'+stamp+'.json')
    for item in sources:
        if item['group']!=a.group:continue
        record={**item,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            dest=within(root,item['local_path'])
            if item.get('github_blob'):
                envelope=dest.with_suffix(dest.suffix+'.blob.json')
                record['response_sha256']=download(item['url'],envelope)
                payload=json.loads(envelope.read_text())
                if payload.get('sha')!=item['github_blob'] or payload.get('encoding')!='base64':
                    raise ValueError('Unexpected GitHub blob envelope')
                raw=base64.b64decode(payload['content'])
                blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
                if blob!=item['github_blob']:raise ValueError('GitHub blob content mismatch')
                if dest.exists() and dest.read_bytes()!=raw:raise ValueError('Existing source differs from official blob')
                dest.write_bytes(raw);sha=file_hash(dest)
            else:sha=download(item['url'],dest,item.get('sha256'))
            record.update(downloaded_sha256=sha,bytes=dest.stat().st_size,status='downloaded')
            if item.get('extract_to'):extract_safe(dest,within(root,item['extract_to']))
            if item.get('vqa_normalizer'):
                blob=hashlib.sha1(b'blob '+str(dest.stat().st_size).encode()+b'\0'+dest.read_bytes()).hexdigest()
                record['git_blob_sha1']=blob
                if blob!='e4ff7887d53195f12856ab1e9087e69abe2e75c8':raise ValueError('VQA source blob changed')
                target=within(root,'src/kdm/models/official_vqa_normalizer.py')
                target.write_text(extract_vqa_normalizer(dest.read_text()),encoding='utf-8')
            record['status']='ok'
        except Exception as error:
            record.update(status='error',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        record['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        report.append(record)
        atomic_json(report_path,report)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    if not report:raise ValueError('No assets selected')
    if any(x['status']!='ok' for x in report):raise SystemExit(1)
if __name__=='__main__':main()
