"""Stage-7 SID runner: faithful reimplementation runs on llava16 / internvl4b
(attention-matrix architectures only), food101 eval half, both strata.

Usage: ./venv/bin/python code/stage7_sid_run.py --model llava16 --gpu 4
"""
import os, sys, json, time, argparse, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ['HF_HOME'] = str(ROOT / 'cache' / 'hf')
os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache' / 'xdg')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache' / 'mpl')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, str(ROOT / 'code'))

MODELS = {'llava16': '/home/g203-4028/Models/llava-v1.6-mistral-7b-hf',
          'internvl4b': '/home/g203-4028/Models/InternVL3_5-4B'}
STRATA = {'llava16': 'data/strata_llava16_food101.json',
          'internvl4b': 'data/strata_internvl4b_food101.json'}
NAMING_TOKENS = 12


def done_keys(path):
    keys = set()
    p = ROOT / 'outputs' / 'raw' / path
    if p.exists():
        for line in open(p):
            try:
                r = json.loads(line)
                if 'error' not in r:
                    keys.add(r['key'])
            except Exception:
                pass
    return keys


def append(path, rec):
    with open(ROOT / 'outputs' / 'raw' / path, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    import torch
    from PIL import Image
    import scoring, prompts
    from stage6_engine import S6Model
    from stage7_sid import SidDecoder

    em = S6Model(MODELS[args.model], f'cuda:{args.gpu}')
    sid = SidDecoder(em)
    classes_all = sorted({m['class'] for m in
                          map(json.loads, open(ROOT / 'data' / 'samples_manifest.jsonl'))})
    strata = json.load(open(STRATA[args.model]))
    stratum_of = {c: 'low_acc' for c in strata['deficient']}
    stratum_of.update({c: 'high_acc' for c in strata['known']})
    manifest = [json.loads(l) for l in open(ROOT / 'data' / 'samples_manifest.jsonl')]
    samples = [m for m in manifest
               if m['part'] == 'eval' and m['class'] in stratum_of]
    if args.limit:
        samples = samples[:args.limit]
    naming_prompt = prompts.STYLE1
    out = f'{args.model}_s7_sid_naming.jsonl'
    done = done_keys(out)
    t0 = time.time()
    n = 0
    for m in samples:
        sid_key = f"nm:{m['file']}:sid"
        if sid_key in done:
            continue
        img = Image.open(m['image_path']).convert('RGB')
        inputs = em.build(img, naming_prompt)
        try:
            r = sid.decode(inputs, NAMING_TOKENS)
            outcome = scoring.score_naming(r['text'], m['class'], classes_all)
            append(out, {'key': sid_key, 'model': args.model, 'task': 'naming',
                         'method': 'sid', 'stratum': stratum_of[m['class']],
                         'class': m['class'], 'file': m['file'],
                         'prompt': naming_prompt, 'text': r['text'],
                         'tokens': r['tokens'], 'step_probs': r['step_probs'],
                         'outcome': outcome, 'answer_maxp': r['answer_maxp'],
                         'n_forwards': r['n_forwards'], 'wall_s': r['wall_s'],
                         'layers_selected': r['layers_selected'],
                         'variant': 'faithful_reimpl_from_official_code'})
            done.add(sid_key)
        except Exception as e:
            append(out, {'key': sid_key, 'model': args.model, 'task': 'naming',
                         'method': 'sid', 'stratum': stratum_of[m['class']],
                         'class': m['class'], 'file': m['file'],
                         'error': f'{type(e).__name__}: {e}'})
            print(f'  [error] {sid_key} {type(e).__name__}: {e}', flush=True)
        n += 1
        if n % 25 == 0:
            el = time.time() - t0
            print(f'[{args.model} sid] {n}/{len(samples)} {el:.0f}s '
                  f'({el / n:.2f}s/img)', flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    sid.close()
    print(f'[{args.model} sid] DONE in {(time.time() - t0) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
