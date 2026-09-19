"""Numerical SID mask-core transplant check on an actual registered backbone.

The oracle executes the unchanged selection block and mask helpers extracted from
fixed official source. Hook transport is shared; this does not run the legacy
entire official model implementation or establish task-level effectiveness.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
from types import SimpleNamespace
from typing import Optional
import numpy as np
import torch
from kdm.models.sid import SIDControl, SIDSession

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '127dd412fa6b61ab1c9babf6979ec4da98002438'
SOURCE = ROOT/'reference_repos/sid/transformers/src/transformers/models/llama/modeling_llama.py'


def official_core():
    repo = ROOT/'reference_repos/sid'
    if subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip() != COMMIT:
        raise RuntimeError('Official SID commit changed')
    rel = str(SOURCE.relative_to(repo))
    expected = subprocess.check_output(['git','show', COMMIT+':'+rel], cwd=repo)
    if SOURCE.read_bytes() != expected:
        raise RuntimeError('Official SID source differs from committed blob')
    source = expected.decode()
    tree = ast.parse(source)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('_make_causal_mask','_expand_mask')]
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name=='LlamaModel')
    method = next(node for node in model.body if isinstance(node, ast.FunctionDef)
                  and node.name=='_prepare_decoder_attention_mask')
    namespace = {'torch':torch, 'Optional':Optional}
    for node in funcs+[method]:
        node.decorator_list=[]
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),namespace)
    start = source.index('                                last_layer_attention = layer_outputs[1]',
                         source.index('elif USE_FAST_V'))
    end = source.index('                                new_attention_mask = gen_attention_mask',start)
    end += len('                                new_attention_mask = gen_attention_mask')
    snippet = textwrap.dedent(source[start:end])
    return namespace, compile(snippet,str(SOURCE),'exec'), snippet, source[:start].count('\n')+1


class OfficialControl(SIDControl):
    def __init__(self, backend, inputs):
        super().__init__(backend,inputs)
        self.namespace,self.code,self.snippet,self.first_line = official_core()
        method = self.namespace['_prepare_decoder_attention_mask']
        self.helper = SimpleNamespace(_prepare_decoder_attention_mask=lambda *args:method(None,*args))
    def _mask(self, original, hidden, attention):
        scope = dict(self.namespace, self=self.helper, layer_outputs=(None,attention[None]),
                     SYS_LENGTH=self.start, IMAGE_TOKEN_LENGTH=self.length, ATTENTION_RANK=100,
                     batch_size=1, seq_length_with_past=attention.shape[-1], seq_length=hidden.shape[1],
                     inputs_embeds=hidden, past_key_values_length=attention.shape[-1]-hidden.shape[1])
        exec(self.code,scope)
        return (scope['top_attention_rank_index']-self.start).sort().values,scope['new_attention_mask']


def oracle_session(backend,inputs):
    session=SIDSession(backend,inputs)
    session.control=OfficialControl(backend,inputs)
    return session


def trace_collector(rows):
    def record(event):
        mask=event['mask'].detach().float().cpu()
        forbidden=mask < -1e4
        q,k=mask.shape[-2:]
        future=torch.arange(k)[None,:] > torch.arange(q)[:,None]+k-q
        start,length=event['start'],event['length']
        chosen=event['selected'].detach().cpu()
        expected=torch.zeros(k,dtype=torch.bool)
        expected[start:start+length]=True
        expected[start+chosen]=False
        expected=expected[None,:] | future
        rows.append({'layer':event['layer'],'query_length':q,'kv_length':k,
                     'start':start,'length':length,'selected':chosen.tolist(),
                     'mask_forbidden_sha256':hashlib.sha256(forbidden.numpy().tobytes()).hexdigest(),
                     'mask_shape':list(mask.shape),
                     'mask_last_row_blocked':forbidden[0,0,-1].nonzero().flatten().tolist(),
                     'mask_min_finite':float(mask[torch.isfinite(mask)].min()) if torch.isfinite(mask).any() else None,
                     'negative_infinity_count':int(torch.isneginf(mask).sum()),
                     'causal_and_selection_match':bool(torch.equal(forbidden[0,0],expected))})
    return record


def compare(backend, inputs_a, inputs_b, prefixes):
    checks={key:True for key in ('official_selection','official_reference_logits',
                                'causal_mask_preserved','interleaved_sessions','nonmonotonic_prefix')}
    # Separate sessions hold independent native KV caches; all use the same model weights.
    sessions=[SIDSession(backend,inputs_a),SIDSession(backend,inputs_b)]
    refs=[oracle_session(backend,inputs_a),oracle_session(backend,inputs_b)]
    evidence=[]
    for n,prefix in enumerate(prefixes):
        for branch in (0,1):
            actual_rows=[];official_rows=[]
            sessions[branch].control.audit=trace_collector(actual_rows)
            refs[branch].control.audit=trace_collector(official_rows)
            actual=sessions[branch].next(prefix).logits
            expected=refs[branch].next(prefix).logits
            error=float(np.max(np.abs(actual-expected)))
            logits_match=bool(np.array_equal(actual,expected))
            selection_match=([r['selected'] for r in actual_rows]==[r['selected'] for r in official_rows])
            masks_match=([r['mask_forbidden_sha256'] for r in actual_rows]==
                         [r['mask_forbidden_sha256'] for r in official_rows])
            causal=all(r['causal_and_selection_match'] for r in actual_rows+official_rows)
            # Recompute a fresh reference at every visited prefix; tests prefix resets and isolation
            # independently of a simultaneously evolving oracle session's cache.
            fresh=oracle_session(backend, [inputs_a,inputs_b][branch])
            fresh_logits=fresh.next(prefix).logits
            fresh_error=float(np.max(np.abs(actual-fresh_logits)))
            fresh_match=bool(np.array_equal(actual,fresh_logits))
            del fresh
            checks['official_selection'] &= selection_match and masks_match
            checks['official_reference_logits'] &= logits_match
            checks['causal_mask_preserved'] &= causal
            checks['interleaved_sessions'] &= fresh_match
            checks['nonmonotonic_prefix'] &= fresh_match
            evidence.append({'visit':n,'branch':branch,'prefix':list(prefix),'max_abs_logit_error':error,
                             'fresh_reference_max_abs_logit_error':fresh_error,
                             'logits_equal':logits_match,'fresh_logits_equal':fresh_match,
                             'selection_equal':selection_match,'mask_blocking_equal':masks_match,
                             'actual':actual_rows,'official':official_rows,
                             'actual_logits_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),
                             'official_logits_sha256':hashlib.sha256(expected.tobytes()).hexdigest()})
    return checks,evidence


def main():
    import argparse, importlib
    from PIL import Image
    from kdm.io import within
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    spec=json.loads(within(ROOT,args.spec).read_text())
    output=within(ROOT,args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    names={'hf.py','backbone.py','sid.py'}
    if spec['key'] in {'minicpm26','minicpm45','phi35'}:names.add('remote.py')
    if spec['key']=='internvl35_8b':names.add('internvl_preprocessing.py')
    report={'passed':False,'reference_commit':COMMIT,'spec':spec,
            'runtime_adapter_sha256':{name:hashlib.sha256((ROOT/'src/kdm/models'/name).read_bytes()).hexdigest()
                                      for name in sorted(names)},
            'scope':'exact native-forward equality to fixed official selection/mask source transplant; shared hook transport',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        namespace,code,snippet,line=official_core()
        report['official_source']={'path':str(SOURCE.relative_to(ROOT)),
                                  'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                                  'selection_first_line':line,'selection_source':snippet}
        module,factory=spec['factory'].split(':')
        backend=getattr(importlib.import_module(module),factory)(**{**spec['kwargs'],'device':args.device})
        sample=json.loads((ROOT/'data/current/interface16.jsonl').read_text().splitlines()[0])
        report['sample_id']=sample['id']
        report['image_sha256']=hashlib.sha256(Path(sample['image_path']).read_bytes()).hexdigest()
        image=Image.open(sample['image_path']).convert('RGB')
        prompts=['What food is shown? Answer briefly.',
                 'Inspect this photograph carefully. What food is shown? Give a short answer.']
        report['prompts']=prompts
        a,b=[backend.em.build(image,prompt) for prompt in prompts]
        tokens=backend.encode(' food dish plate')
        if len(tokens)<3:raise RuntimeError('Verification continuation requires three native tokenizer tokens')
        prefixes=[(),tuple(tokens[:1]),tuple(tokens[:2]),tuple(tokens[:2]),tuple(tokens[2:3]),(),tuple(tokens[:3])]
        report['checks'],report['evidence']=compare(backend,a,b,prefixes)
        report['passed']=all(report['checks'].values())
        if not report['passed']:raise RuntimeError('SID numerical source comparison failed; see checks/evidence')
    except Exception as exc:
        report['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'record':str(output)}))

if __name__=='__main__':main()
