#!/usr/bin/env python3
"""Download explicitly listed research assets; keep all writes in the project."""
from pathlib import Path
import argparse,json,sys,re,hashlib,ast
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE/'src'))
from kdm.data import download,extract_safe
from kdm.io import within,atomic_json


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
    for item in sources:
        if item['group']!=a.group:continue
        dest=within(root,item['local_path']);sha=download(item['url'],dest,item.get('sha256'))
        report.append({**item,'downloaded_sha256':sha})
        if item.get('extract_to'):extract_safe(dest,within(root,item['extract_to']))
        if item.get('vqa_normalizer'):
            blob=hashlib.sha1(b'blob '+str(dest.stat().st_size).encode()+b'\0'+dest.read_bytes()).hexdigest()
            if blob!='e4ff7887d53195f12856ab1e9087e69abe2e75c8':raise ValueError('VQA source blob changed')
            target=within(root,'src/kdm/models/official_vqa_normalizer.py')
            target.write_text(extract_vqa_normalizer(dest.read_text()))
    atomic_json(within(root,'outputs/records/assets_'+a.group+'.json'),report)
if __name__=='__main__':main()
