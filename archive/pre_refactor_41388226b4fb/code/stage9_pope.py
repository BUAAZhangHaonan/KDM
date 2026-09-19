"""Small standard-task bridge: fixed 50 COCO images, two official POPE subsets.

Network I/O is tested with local byte responses in the delivery environment;
real image download and VLM inference must be executed in the project server.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,re,ssl,sys,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from stage9_common import ROOT,ALLOWED_GPUS,initialize,inside,read_jsonl,stable_key,write_json,transition_counts,cluster_bootstrap_pair
from stage9_run import MODEL_PATHS,noise_seed,clean_record

ANNOTATIONS={
'random':('https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json','cd2b137f786675decb7442675929885125eeb083'),
'adversarial':('https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json','4eb7fc57596b955756a94bc63d1f967c2afb692a')}


IMAGE_HOST='https://images.cocodataset.org/'

def fetch_bytes(url:str)->bytes:
    """Official annotations: full TLS verification. COCO image host: this
    network's S3 edge serves the default s3.amazonaws.com certificate without
    the images.cocodataset.org SAN entry (verified 2026-09-17), so hostname
    checking is disabled for this one host while CERTIFICATE CHAIN verification
    stays enabled; downloaded image bytes are sha256-logged in
    source_manifest.json as a compensating audit control. See
    docs/stage9/CODE_CHANGES_STAGE9.md."""
    if url.startswith(IMAGE_HOST):
        # Direct TLS to this host works; the local proxy resets the tunnel.
        ctx=ssl.create_default_context();ctx.check_hostname=False
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                           urllib.request.HTTPSHandler(context=ctx))
        with opener.open(url,timeout=180) as r:return r.read()
    with urllib.request.urlopen(url,timeout=120) as r:return r.read()


def blob_sha(content:bytes)->str:
    return hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()


def prepare(root:Path,n_images=50,fetch=fetch_bytes,annotations=ANNOTATIONS):
    folder=inside(root,root/'data/stage9/pope');folder.mkdir(parents=True,exist_ok=True)
    by_subset={}
    for subset,(url,expected) in annotations.items():
        p=folder/f'{subset}.jsonl'
        content=p.read_bytes() if p.exists() else fetch(url)
        if blob_sha(content)!=expected:raise ValueError(f'Official annotation hash changed: {subset}')
        p.write_bytes(content)
        rows=[json.loads(l) for l in content.decode().splitlines() if l.strip()]
        by_subset[subset]={}
        for r in rows:
            name=r['image']
            if Path(name).name!=name or r['label'] not in ('yes','no'):raise ValueError('Unexpected official record schema.')
            by_subset[subset].setdefault(name,[]).append(r)
    common=set.intersection(*(set(v) for v in by_subset.values()))
    names=sorted(common,key=lambda n:stable_key(911,n))[:n_images]
    if len(names)!=n_images:raise ValueError('Insufficient paired images.')
    image_dir=folder/'images';image_dir.mkdir(exist_ok=True)
    def download(name):
        p=image_dir/name
        if not p.exists():p.write_bytes(fetch(f'{IMAGE_HOST}val2014/{name}'))
        with Image.open(p) as im:im.verify()
        return p
    for name in names:download(name)
    image_hashes={n:hashlib.sha256((image_dir/n).read_bytes()).hexdigest() for n in names}
    selected=[]
    for subset,mapping in by_subset.items():
        for name in names:
            rows=mapping[name]
            if len(rows)!=6 or sum(r['label']=='yes' for r in rows)!=3:
                raise ValueError('Expected six balanced questions per paired image.')
            for r in rows:selected.append({**r,'subset':subset,'image_path':str(image_dir/name)})
    with (folder/'manifest.jsonl').open('w') as f:
        for r in selected:f.write(json.dumps(r)+'\n')
    write_json(folder/'source_manifest.json',dict(images=n_images,questions=len(selected),seed=911,
        annotation_blobs={k:v[1] for k,v in annotations.items()},
        image_sha256=image_hashes,
        note='Negative-object selection differs; do not relabel the subsets as easy/hard or knowledge-present/absent.'))
    return selected


def parse_answer(text:str):
    m=re.match(r'^\s*(yes|no)\b',text.lower());return m.group(1) if m else None


def run(root,model,em,shard=0,shards=1):
    rows=list(read_jsonl(root/'data/stage9/pope/manifest.jsonl'))
    names=sorted({r['image'] for r in rows});names={n for i,n in enumerate(names) if i%shards==shard}
    selected=[r for r in rows if r['image'] in names]
    out=inside(root,root/'outputs/raw/stage9'/f'{model}_pope_{shard}.jsonl');out.parent.mkdir(parents=True,exist_ok=True)
    done={r['key'] for r in read_jsonl(out)} if out.exists() else set()
    for r in selected:
        prompt=r['text'].strip()+' Answer yes or no.'
        image=Image.open(r['image_path']).convert('RGB');inputs=em.build(image,prompt)
        for method in ('direct','vcd'):
            key=f"{model}:pope:{r['subset']}:{r['question_id']}:{method}"
            if key in done:continue
            if method=='direct':res=em.decode_single(inputs,4,method='direct')
            else:res=em.decode_two_branch(inputs,em.noised_inputs(inputs,noise_seed(r['image'])),'vcd',4)
            rec={**clean_record(res),'key':key,'model':model,'method':method,'image':r['image'],
                 'question_id':r['question_id'],'subset':r['subset'],'gold':r['label'],
                 'prediction':parse_answer(res['text']),'prompt':prompt}
            rec['correct']=rec['prediction']==rec['gold']
            with out.open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
    return out


def summarize(root:Path,replicates=2000):
    d={}
    for p in sorted((root/'outputs/raw/stage9').glob('*_pope_*.jsonl')):
        for r in read_jsonl(p):
            k=(r['model'],r['subset'],r['question_id'],r['method'])
            if k in d:raise ValueError('Duplicate POPE result.')
            d[k]=r
    expected=list(read_jsonl(root/'data/stage9/pope/manifest.jsonl'))
    table=[]
    for model in MODEL_PATHS:
        for subset in ANNOTATIONS:
            ids=[r['question_id'] for r in expected if r['subset']==subset]
            if any((model,subset,i,m) not in d for i in ids for m in ('direct','vcd')):
                raise ValueError(f'Incomplete POPE matrix for {model}/{subset}')
            a=[d[model,subset,i,'direct'] for i in ids];b=[d[model,subset,i,'vcd'] for i in ids]
            yd=np.array([r['correct'] for r in a]);ym=np.array([r['correct'] for r in b])
            vals=np.c_[yd,ym].astype(float)
            lo,hi=cluster_bootstrap_pair(vals,[r['image'] for r in a],lambda v:np.mean(v[:,1]-v[:,0]),replicates)
            table.append(dict(model=model,subset=subset,**transition_counts(yd,ym),delta_lo=lo,delta_hi=hi,
                              yes_direct=float(np.mean([r['prediction']=='yes' for r in a])),
                              yes_vcd=float(np.mean([r['prediction']=='yes' for r in b]))))
    p=root/'outputs/tables/stage9/pope.csv';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    return table


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('prepare','run','summarize'))
    ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--model',choices=tuple(MODEL_PATHS))
    ap.add_argument('--gpu',type=int);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    a=ap.parse_args();root=initialize(a.root)
    if a.action=='prepare':prepare(root);return
    if a.action=='summarize':summarize(root);return
    if a.gpu not in ALLOWED_GPUS or a.model is None:raise ValueError('Authorized physical GPU and model required.')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);sys.path.insert(0,str(root/'code'))
    from stage6_engine import S6Model
    em=S6Model(MODEL_PATHS[a.model],'cuda:0');run(root,a.model,em,a.shard,a.shards)

if __name__=='__main__':main()
