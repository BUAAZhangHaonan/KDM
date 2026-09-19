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

def donor_pool(records,model,annotations,expected_samples=None,require_annotations=False):
    donors=defaultdict(list);samples={};seen=defaultdict(set)
    for r in records:
        if r['model']!=model:raise ValueError('Donor inputs contain a different model')
        if r['method']!='direct' or not r['guided'] or r['kind']!='main':continue
        sid=r['sample']['id'];marker=r['marker']
        if marker not in MARKERS or marker in seen[sid]:raise ValueError('Duplicate or unknown donor prompt')
        if r.get('status')!='ok':raise ValueError('Donor generation did not complete successfully')
        if sid in samples and samples[sid]['sample']!=r['sample']:raise ValueError('Donor sample contents disagree')
        if expected_samples is not None:
            if expected_samples.get(sid)!=r['sample']:raise ValueError('Donor sample disagrees with frozen manifest')
            from kdm.pipeline import task_id
            task={'sample':r['sample'],'method':'direct','marker':marker,'reference_marker':marker,
                  'guided':True,'reference_guided':True,'replicate':0,'kind':'main'}
            if r['key']!=task_id(model,task) or any(r.get(k)!=v for k,v in task.items()):
                raise ValueError('Raw donor task differs from frozen direct task')
        seen[sid].add(marker);samples[sid]=r
        if require_annotations and (r['key'] not in annotations or annotations[r['key']]['text']!=r['text']):
            raise ValueError('Every donor requires its matching unified annotation')
        label=label_response(r['text'],annotations,r['key'])
        if not isinstance(r.get('tokens'),list) or not r['tokens'] or any(type(t) is not int or t<0 for t in r['tokens']):
            raise ValueError('Exact donor tokens are required')
        if not isinstance(r.get('terminated'),bool):raise ValueError('Explicit donor termination status is required')
        donors[sid].append({'kind':'invalid' if label=='invalid' else ('abstention' if label=='abstain' else 'answer'),
            'label':label,'key':r['key'],'marker':marker,'text':r['text'],
            'tokens':r['tokens'],'terminated':r['terminated']})
    if not samples:raise ValueError('No direct donor records')
    if expected_samples is not None and set(samples)!=set(expected_samples):raise ValueError('Incomplete frozen donor sample coverage')
    if any(markers!=set(MARKERS) for markers in seen.values()):raise ValueError('All four direct prompts are required per donor pool')
    return donors,samples


def donor_measurement_status(pool):
    # A budget-complete but unterminated response is not an EOS-complete event.
    # Keep the original fixed pool visible; do not append EOS or shrink the pool.
    if any(not row['terminated'] for row in pool):return 'incomplete_direct_donor_pool'
    if not any(row['kind']=='answer' for row in pool):return 'no_concrete_donor'
    return 'defined'


def execution_identity(root,spec,model,out,gpu):
    from kdm.protocol import validate_runtime,code_identity
    if spec.get('purpose')=='CPU_TEST_ONLY':
        if not out.is_relative_to(root/'outputs/verification'):
            raise ValueError('Synthetic model outputs must remain in outputs/verification')
        return {'software_fixture':True,'formal_evidence':False}
    cards=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if not cards or cards[0]!=gpu or any(card not in {'0','1','4','5'} for card in cards):
        raise ValueError('Use the allocated worker script')
    validate_runtime(root,spec,model,cards)
    return code_identity(root)


def validate_input_ledgers(paths,spec_sha,formal):
    """Check raw record identity against its immutable input sidecar."""
    for path in paths:
        path=Path(path);meta=path.with_suffix('.identity.json')
        if not meta.is_file():
            if formal:raise ValueError('Formal donor input requires a ledger identity sidecar')
            continue
        identity=json.loads(meta.read_text());definition=identity['definition']
        if stable_hash(definition)!=identity['identity']:raise ValueError('Input ledger identity digest mismatch')
        if formal and definition.get('backend_spec_sha256')!=spec_sha:
            raise ValueError('Donor backend spec differs from measurement backend')
        for row in read_jsonl(path):
            if row.get('identity')!=identity['identity']:raise ValueError('Raw donor record identity mismatch')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--records',nargs='+',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--annotations',required=True);p.add_argument('--model-spec',required=True);p.add_argument('--model',required=True)
    p.add_argument('--methods',default='vcd,m3id,dola,deco,instruction_vcd,instruction_m3id');p.add_argument('--gpu',choices=['0','1','4','5'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
    from kdm.cli import setup
    root=setup(a.root);out=within(root,a.out)
    spec=json.load(open(a.model_spec));spec_sha=file_hash(a.model_spec)
    source_blobs=execution_identity(root,spec,a.model,out,a.gpu)
    ann=validate_annotations(a.annotations)
    manifest=list(read_jsonl(a.manifest))
    if len({row['id'] for row in manifest})!=len(manifest):raise ValueError('Duplicate frozen manifest sample')
    expected={row['id']:row for row in manifest if row['split']=='eval'}
    validate_input_ledgers(a.records,spec_sha,spec.get('purpose')!='CPU_TEST_ONLY')
    donors,samples=donor_pool((r for path in a.records for r in read_jsonl(path)),a.model,ann,expected,True)
    for sample in expected.values():
        if not Path(sample['image_path']).is_file():raise ValueError('Missing frozen donor image')
    backend=None
    methods=tuple(a.methods.split(','))
    if not methods or not set(methods)<={'vcd','m3id','dola','deco','sid','instruction_vcd','instruction_m3id'}:raise ValueError('Unsupported complete-response method')
    ledger=Ledger(out,{'model':a.model,'model_spec':spec_sha,'manifest':file_hash(a.manifest),
        'inputs':[file_hash(x) for x in a.records],'annotations':file_hash(a.annotations),
        'methods':methods,'source_blobs':source_blobs,'measurement_script_sha256':file_hash(__file__),
        'measurement_schema':'complete_response_v2'})
    for sid in sorted(samples):
        pool=donors[sid];record=samples[sid]
        for method in methods:
            for marker in MARKERS:
                refs=MARKERS if method in {'vcd','m3id','sid'} else (marker,)
                for refmarker in refs:
                    key=stable_hash([a.model,sid,method,marker,refmarker,'complete_response'])
                    if key in ledger.keys:continue
                    status=donor_measurement_status(pool)
                    if status!='defined':
                        ledger.add(key,{'sample_id':sid,'model':a.model,'method':method,'marker':marker,
                            'reference_marker':refmarker,'measurement_status':status,'donor_pool':pool,
                            'finite_response_identity':None,'responses':[],
                            'measurement_scope':'undefined_on_the_frozen_direct_donor_pool'})
                        continue
                    if backend is None:backend=make_backend(spec,'cuda:0')
                    candidates=[row for row in pool if row['kind']!='invalid']
                    with Image.open(record['sample']['image_path']) as im:
                        result=finite_response_audit(backend,im.convert('RGB'),record['sample']['question'],marker,refmarker,candidates,DecodeConfig(method=method),record['seed'])
                    ledger.add(key,{'sample_id':sid,'model':a.model,'method':method,'marker':marker,'reference_marker':refmarker,'measurement_status':'defined','donor_pool':pool,**json_safe(result)})
if __name__=='__main__':main()
