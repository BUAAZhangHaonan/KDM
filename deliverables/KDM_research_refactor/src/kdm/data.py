"""Dataset manifests, streaming assets and exact split preservation."""
from __future__ import annotations
import json, os, shutil, tarfile, zipfile
from pathlib import Path
import requests
from .io import stable_hash, file_hash, atomic_json, read_jsonl, within


def write_manifest(rows,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    identifiers=set()
    with path.open('w',encoding='utf-8') as f:
        for r in rows:
            if r['id'] in identifiers: raise ValueError('Duplicate sample identifier')
            identifiers.add(r['id']);f.write(json.dumps(r,ensure_ascii=False)+'\n')
    atomic_json(str(path)+'.meta.json',{'n':len(identifiers),'sha256':file_hash(path)})


def food_from_existing(source,destination):
    rows=[]
    for x in read_jsonl(source):
        image=Path(x['image_path'])
        if not image.is_file(): raise FileNotFoundError(image)
        rows.append({'id':'food101:'+x['file'],'cluster':x['class'],'dataset':'food101',
                     'image_path':str(image.resolve()),'question':'What specific food is shown in this image?',
                     'gold':[x['class']],'class':x['class'],'split':'dev' if x['part']=='group' else 'eval',
                     'source_record':x['file']})
    if any(x['part'] not in {'group','eval'} for x in read_jsonl(source)):
        raise ValueError('Unknown existing split; no automatic reassignment')
    write_manifest(rows,destination)


def vizwiz_manifest(annotation,image_root,destination):
    """Full official validation split; deterministic image-level dev/eval 1:4.
    Every item is retained; the split limits fitting to dev records.
    """
    rows=[]
    for x in json.load(open(annotation,encoding='utf-8')):
        image=Path(image_root)/x['image']
        if not image.is_file(): raise FileNotFoundError(image)
        sid='vizwiz:'+x['image']
        rows.append({'id':sid,'cluster':x['image'],'dataset':'vizwiz','image_path':str(image.resolve()),
                     'question':x['question'],'gold':[a['answer'] for a in x['answers']],
                     'annotated_answerable':x.get('answerable'),
                     'split':'dev' if int(stable_hash(sid)[:8],16)%5==0 else 'eval'})
    write_manifest(rows,destination)


def download(url,dest,expected_sha256=None):
    """Stream to disk; interrupted partial files never become completed assets."""
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if expected_sha256 and file_hash(dest)!=expected_sha256: raise ValueError('Asset checksum mismatch')
        return file_hash(dest)
    tmp=dest.with_suffix(dest.suffix+'.part')
    with requests.get(url,stream=True,timeout=(20,120)) as response:
        response.raise_for_status()
        with tmp.open('wb') as f:
            for chunk in response.iter_content(1024*1024):
                if chunk:f.write(chunk)
    digest=file_hash(tmp)
    if expected_sha256 and digest!=expected_sha256: raise ValueError('Downloaded asset checksum mismatch')
    tmp.replace(dest);return digest


def extract_safe(archive,destination):
    """Reject symlinks and path traversal for downloaded dataset archives."""
    root=Path(destination).resolve();root.mkdir(parents=True,exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                target=within(root,root/m.filename)
                mode=m.external_attr>>16
                if (mode & 0o170000)==0o120000: raise ValueError('Archive symlink refused')
                if m.is_dir():target.mkdir(parents=True,exist_ok=True);continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(m) as src,target.open('wb') as out:shutil.copyfileobj(src,out)
    else:
        with tarfile.open(archive) as tar:
            for m in tar.getmembers():
                target=within(root,root/m.name)
                if m.issym() or m.islnk() or not (m.isdir() or m.isfile()):raise ValueError('Archive special member refused')
                if m.isdir():target.mkdir(parents=True,exist_ok=True);continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(m) as src,target.open('wb') as out:shutil.copyfileobj(src,out)
