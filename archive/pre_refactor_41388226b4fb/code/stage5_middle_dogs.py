"""Stage-5 middle-accuracy class run (dogs domain): stage-3 code path.

Replicates stage3_model.py's `main` stage verbatim (same get_engine, prep,
make_variants, prompts.STYLE1, scoring, lcd_ok handling) with the class set
replaced by dogs classes in neither deficient nor known, stratum='middle'.

Usage:
  ./venv/bin/python code/stage5_middle_dogs.py --key q4b --gpu 4
"""
import os, sys, json, time, argparse, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for var, sub in [('HF_HOME', 'hf'), ('TORCH_HOME', 'torch'), ('XDG_CACHE_HOME', 'xdg'),
                 ('MPLCONFIGDIR', 'mpl'), ('NLTK_DATA', 'nltk')]:
    os.environ[var] = str(ROOT / 'cache' / sub)
sys.path.insert(0, str(ROOT / 'code'))

XMODELS = {'q4b': '/home/g203-4028/Models/Qwen3.5-4B',
           'q9b': '/home/g203-4028/Models/Qwen3.5-9B'}
METHODS = ['direct', 'vcd', 'mib', 'lcd']


def append(path, rec):
    with open(ROOT / 'outputs' / 'raw' / path, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def done_keys(path):
    keys, p = set(), ROOT / 'outputs' / 'raw' / path
    if p.exists():
        for line in open(p):
            try:
                keys.add(json.loads(line)['key'])
            except Exception:
                pass
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True, choices=list(XMODELS))
    ap.add_argument('--gpu', type=int, required=True)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from stage3_engine import get_engine
    from engine import make_variants, jsd, HP
    import scoring, prompts

    manifest = [json.loads(l) for l in open(ROOT / 'data' / 'samples_manifest_dogs.jsonl')]
    classes_all = sorted({m['class'] for m in manifest})
    strata = json.load(open(ROOT / 'data' / f'strata_{args.key}_dogs.json'))
    excluded = set(strata['deficient']) | set(strata['known'])
    middle = [c for c in classes_all if c not in excluded]
    stratum_of = {c: 'middle' for c in middle}
    print(f'[{args.key} dogs middle] classes: {len(middle)}', flush=True)

    t0 = time.time()
    em = get_engine(XMODELS[args.key], f"cuda:{args.gpu}")
    lcd_ok = em.lcd_ready
    resize_max = 640 if em.mt == 'qwen3_vl' else None
    print(f"[{args.key}] loaded {em.mt} lcd_ready={lcd_ok} ({time.time() - t0:.0f}s)", flush=True)

    def prep(m):
        img = Image.open(m['image_path']).convert('RGB')
        if resize_max:
            img.thumbnail((resize_max, resize_max))
        return img

    out = f'{args.key}_s5middle_dogs.jsonl'
    done = done_keys(out)
    evals = [m for m in manifest if m['part'] == 'eval' and m['class'] in stratum_of]
    n = 0
    for m in evals:
        img = prep(m)
        v = make_variants(img)
        cls, st = m['class'], stratum_of[m['class']]
        sid = f"nm:{m['file']}"
        blank = None
        need = [mm for mm in METHODS if f"{sid}:{mm}" not in done]
        if 'direct' in need:
            blank = em.decode_single(em.build(v['blank'], prompts.STYLE1), 1)
        for mm in need:
            try:
                if mm == 'direct':
                    r = em.decode_single(em.build(v['clean'], prompts.STYLE1), HP['max_new_tokens_naming'])
                    signals = {'entropy': r['first_entropy'], 'maxp': r['first_maxp'],
                               'jsd': jsd(r['first_probs'], blank['first_probs'])}
                elif mm == 'lcd':
                    if not lcd_ok:
                        append(out, {'key': f"{sid}:lcd", 'model': args.key, 'domain': 'dogs',
                                     'task': 'naming', 'method': 'lcd', 'stratum': st, 'class': cls,
                                     'file': m['file'], 'unavailable': True,
                                     'reason': f'lcd not supported for {em.mt}'})
                        continue
                    r = em.decode_single(em.build(v['clean'], prompts.STYLE1), HP['max_new_tokens_naming'], lcd=True)
                    signals = {}
                elif mm == 'vcd':
                    r = em.decode_contrastive(em.build(v['clean'], prompts.STYLE1),
                                              em.build(v['noise'], prompts.STYLE1), 'vcd', HP['max_new_tokens_naming'])
                    signals = {}
                else:
                    r = em.decode_contrastive(em.build(v['clean'], prompts.STYLE1),
                                              em.build(v['blur'], prompts.STYLE1), 'mib', HP['max_new_tokens_naming'])
                    signals = {}
                append(out, {'key': f"{sid}:{mm}", 'model': args.key, 'domain': 'dogs',
                             'task': 'naming', 'method': mm, 'stratum': st, 'class': cls,
                             'file': m['file'], 'prompt': prompts.STYLE1, 'style': 'style1',
                             'text': r['text'], 'outcome': scoring.score_naming(r['text'], cls, classes_all),
                             'answer_maxp': r['answer_maxp'], 'signals': signals,
                             'n_forwards': r['n_forwards'], 'wall_s': round(r['wall_s'], 3)})
            except Exception as ex:
                append(out, {'key': f"{sid}:{mm}", 'model': args.key, 'domain': 'dogs',
                             'task': 'naming', 'method': mm, 'stratum': st, 'class': cls,
                             'file': m['file'], 'error': str(ex)[:300]})
                print(f'[{args.key}] ERROR {sid}:{mm}: {str(ex)[:120]}', flush=True)
        n += 1
        if n % 25 == 0:
            print(f'[{args.key} dogs middle] {n}/{len(evals)} {time.time() - t0:.0f}s', flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print(f'[{args.key} dogs middle] DONE {(time.time() - t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
