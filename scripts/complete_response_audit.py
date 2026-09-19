#!/usr/bin/env python3
"""Whole-sequence probes on every donor response and the fixed marker set."""
import argparse,json,sys,os
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from kdm.io import read_jsonl,Ledger,file_hash,within,stable_hash
from kdm.pipeline import make_backend,finite_response_audit,json_safe
from kdm.scoring import label_response
from kdm.annotation import validate_annotations
from kdm.decoding import DecodeConfig
from kdm.prompts import MARKERS
from PIL import Image

def donor_pool(records,model,annotations):
    donors=defaultdict(list);samples={};seen=defaultdict(set)
    for r in records:
        if r['model']!=model or r['method']!='direct' or not r['guided'] or r['kind']!='main':continue
        sid=r['sample']['id'];marker=r['marker']
        if marker not in MARKERS or marker in seen[sid]:raise ValueError('Duplicate or unknown donor prompt')
        seen[sid].add(marker);samples[sid]=r
        label=label_response(r['text'],annotations,r['key'])
        if label=='invalid':continue
        if not r.get('tokens'):raise ValueError('Exact donor tokens are required')
        if not r.get('terminated'):raise ValueError('Truncated donor is not a complete response')
        donors[sid].append({'kind':'abstention' if label=='abstain' else 'answer','text':r['text'],'tokens':r['tokens']})
    if not samples:raise ValueError('No direct donor records')
    if any(markers!=set(MARKERS) for markers in seen.values()):raise ValueError('All four direct prompts are required per donor pool')
    return donors,samples


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--records',nargs='+',required=True)
    p.add_argument('--annotations',required=True);p.add_argument('--model-spec',required=True);p.add_argument('--model',required=True)
    p.add_argument('--methods',default='vcd,m3id,dola,deco,instruction_vcd,instruction_m3id');p.add_argument('--gpu',choices=['0','1','4','5'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
    # Must run through worker.sh so multi-device allocation is preserved.
    if os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')[0]!=a.gpu:raise ValueError('Use the allocated worker script')
    backend=make_backend(json.load(open(a.model_spec)),'cuda:0');ann=validate_annotations(a.annotations)
    donors,samples=donor_pool((r for path in a.records for r in read_jsonl(path)),a.model,ann)
    methods=tuple(a.methods.split(','))
    if not methods or not set(methods)<={'vcd','m3id','dola','deco','sid','instruction_vcd','instruction_m3id'}:raise ValueError('Unsupported complete-response method')
    ledger=Ledger(within(a.root,a.out),{'model':a.model,'model_spec':file_hash(a.model_spec),'inputs':[file_hash(x) for x in a.records],'annotations':file_hash(a.annotations),'methods':methods})
    for sid in sorted(samples):
        pool=donors[sid];record=samples[sid]
        for method in methods:
            for marker in MARKERS:
                refs=MARKERS if method in {'vcd','m3id','sid'} else (marker,)
                for refmarker in refs:
                    key=stable_hash([a.model,sid,method,marker,refmarker,'complete_response'])
                    if key in ledger.keys:continue
                    if not any(x['kind']=='answer' for x in pool):
                        ledger.add(key,{'sample_id':sid,'model':a.model,'method':method,'marker':marker,'reference_marker':refmarker,'measurement_status':'no_concrete_donor','responses':pool});continue
                    with Image.open(record['sample']['image_path']) as im:
                        result=finite_response_audit(backend,im.convert('RGB'),record['sample']['question'],marker,refmarker,pool,DecodeConfig(method=method),record['seed'])
                    ledger.add(key,{'sample_id':sid,'model':a.model,'method':method,'marker':marker,'reference_marker':refmarker,**json_safe(result)})
if __name__=='__main__':main()
