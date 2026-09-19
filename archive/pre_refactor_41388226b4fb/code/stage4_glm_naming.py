"""GLM-4.6V-Flash direct naming on a subsample (probe-only model).

Purpose: sample-level known set for the probe gate (known = open-set correct),
per PREREGISTER_STAGE4 section 1. GLM does ~16.5 s/decode (measured stage 3),
so we subsample instead of running all 1200.

Usage: ./venv/bin/python code/stage4_glm_naming.py --gpu 4 --n 200
"""
import os, sys, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for var, sub in [('HF_HOME','hf'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg'),('MPLCONFIGDIR','mpl')]:
    os.environ[var] = str(ROOT/'cache'/sub)
sys.path.insert(0, str(ROOT/'code'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--n', type=int, default=200)
    a = ap.parse_args()
    import torch
    from PIL import Image
    from stage3_engine import get_engine
    from engine import HP
    from prompts import STYLE1
    import scoring
    em = get_engine('/home/g203-4028/Models/GLM-4.6V-Flash', f'cuda:{a.gpu}')
    manifest = [json.loads(l) for l in open(ROOT/'data/samples_manifest.jsonl')]
    classes_all = sorted({m['class'] for m in manifest})
    strata = json.load(open(ROOT/'data/strata_q4b.json'))   # fixed sample selection
    stratum_of = {c: 'deficient' for c in strata['deficient']}
    stratum_of.update({c: 'known' for c in strata['known']})
    evals = [m for m in manifest if m['part']=='eval' and m['class'] in stratum_of]
    # deterministic subsample spread across classes: every k-th
    step = max(1, len(evals)//a.n)
    evals = evals[::step][:a.n]
    out = 'glm46v_stage4_naming_food101.jsonl'
    done = set()
    p = ROOT/'outputs/raw'/out
    if p.exists():
        done = {json.loads(l)['key'] for l in open(p)}
    t0=time.time(); n=0
    with open(p,'a') as fo:
        for m in evals:
            key=f"nm:{m['file']}"
            if key in done: continue
            r = em.decode_single(em.build(Image.open(m['image_path']).convert('RGB'), STYLE1),
                                 HP['max_new_tokens_naming'])
            fo.write(json.dumps({'key':key,'model':'glm46v','domain':'food101','task':'naming',
                                 'method':'direct','stratum':stratum_of[m['class']],'class':m['class'],
                                 'file':m['file'],'text':r['text'],
                                 'outcome':scoring.score_naming(r['text'], m['class'], classes_all),
                                 'answer_maxp':r['answer_maxp'],
                                 'n_forwards':r['n_forwards'],'wall_s':round(r['wall_s'],3)})+'\n')
            n+=1
            if n%20==0: print(f'[glm naming] {n}/{len(evals)} {time.time()-t0:.0f}s', flush=True)
    print('[glm naming DONE]', flush=True)

if __name__=='__main__':
    main()
