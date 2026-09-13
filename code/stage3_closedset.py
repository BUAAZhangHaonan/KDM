"""Stage 3: closed-set recognition on the evaluation half (strata classes).

Two variants per sample:
  full101 : all 101 food classes as numbered candidates (chance ~1%)
  near10  : gold + 9 WordNet-Wu-Palmer nearest neighbours (chance 10%),
            the "semantic-neighbour" control for the easier-task caveat
Candidate order is shuffled per sample with a deterministic seed
(random.Random(f"{file}|{variant}")), identical across models.
Scoring: first integer in the reply == gold position. Unparsed -> wrong.

Usage:
  ./venv/bin/python code/stage3_closedset.py --model q4b --gpu 4
"""
import os, sys, json, time, argparse, random, re, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for var, sub in [('HF_HOME', 'hf'), ('TORCH_HOME', 'torch'), ('XDG_CACHE_HOME', 'xdg'),
                 ('MPLCONFIGDIR', 'mpl'), ('NLTK_DATA', 'nltk')]:
    os.environ[var] = str(ROOT / 'cache' / sub)
sys.path.insert(0, str(ROOT / 'code'))

MODELS = {'q4b': '/home/g203-4028/Models/Qwen3.5-4B',
          'q9b': '/home/g203-4028/Models/Qwen3.5-9B'}


def append(path, rec):
    with open(ROOT / 'outputs' / 'raw' / path, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def done_keys(path):
    keys = set()
    p = ROOT / 'outputs' / 'raw' / path
    if p.exists():
        for line in open(p):
            try:
                keys.add(json.loads(line)['key'])
            except Exception:
                pass
    return keys


def build_neighbours(classes, k=9, cache_path=None):
    """Per class: k nearest other classes by WordNet Wu-Palmer similarity
    (max over synset pairs); token-Jaccard fallback when no synset pair
    yields a score. Deterministic; cached to json."""
    if cache_path is not None and Path(cache_path).exists():
        return json.load(open(cache_path))
    from scoring import class_synsets
    syn = {c: class_synsets(c) for c in classes}

    def sim(a, b):
        best, seen = 0.0, False
        for sa in syn[a]:
            for sb in syn[b]:
                try:
                    s = sa.wup_similarity(sb)
                except Exception:
                    s = None
                if s is not None:
                    seen = True
                    if s > best:
                        best = s
        if not seen or best == 0.0:
            ta, tb = set(a.split('_')), set(b.split('_'))
            best = len(ta & tb) / max(1, len(ta | tb))  # 0..1 fallback scale
        return best

    nb = {}
    for c in classes:
        scored = sorted(((sim(c, o), o) for o in classes if o != c),
                        key=lambda t: (-t[0], t[1]))
        nb[c] = [o for _, o in scored[:k]]
    if cache_path is not None:
        json.dump(nb, open(cache_path, 'w'), indent=1)
    return nb


def closedset_prompt(cands):
    lines = "\n".join(f"{i + 1}. {c.replace('_', ' ')}" for i, c in enumerate(cands))
    return ("Look at the image. Which of the following dishes is shown?\n"
            "Reply with the number only.\n"
            f"{lines}\nAnswer:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--variants', default='full101,near10')
    ap.add_argument('--max-new', type=int, default=6)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from engine import ExpModel

    em = ExpModel(MODELS[args.model], f'cuda:{args.gpu}')
    manifest = [json.loads(l) for l in open(ROOT / 'data' / 'samples_manifest.jsonl')]
    strata = json.load(open(ROOT / 'data' / f'strata_{args.model}.json'))
    stratum_of = {c: 'deficient' for c in strata['deficient']}
    stratum_of.update({c: 'known' for c in strata['known']})
    classes_all = sorted({m['class'] for m in manifest})
    nb = build_neighbours(classes_all, 9, ROOT / 'data' / 'neighbours_food101.json')

    out = f'{args.model}_stage3_closedset.jsonl'
    done = done_keys(out)
    evals = [m for m in manifest if m['part'] == 'eval' and m['class'] in stratum_of]
    variants = args.variants.split(',')
    t0, n_done = time.time(), 0
    total = len(evals) * len(variants)
    print(f'[{args.model}] closed-set: {len(evals)} imgs x {variants}', flush=True)

    for m in evals:
        img = Image.open(m['image_path']).convert('RGB')
        cls, st = m['class'], stratum_of[m['class']]
        for var in variants:
            key = f"cs:{m['file']}:{var}"
            if key in done:
                continue
            pool = classes_all if var == 'full101' else [cls] + nb[cls]
            order = list(range(len(pool)))
            random.Random(f"{m['file']}|{var}").shuffle(order)
            cands = [pool[i] for i in order]
            gold_idx = cands.index(cls)
            r = em.decode_single(em.build(img, closedset_prompt(cands)), args.max_new)
            mm = re.search(r'\d+', r['text'])
            parsed = int(mm.group()) if mm else None
            correct = (parsed == gold_idx + 1)
            append(out, {'key': key, 'model': args.model, 'task': 'closedset',
                         'variant': var, 'stratum': st, 'class': cls, 'file': m['file'],
                         'n_options': len(cands), 'gold_idx': gold_idx,
                         'text': r['text'], 'parsed': parsed, 'correct': correct,
                         'first_maxp': r['first_maxp'],
                         'n_forwards': r['n_forwards'], 'wall_s': round(r['wall_s'], 3)})
            n_done += 1
            if n_done % 50 == 0:
                el = time.time() - t0
                print(f'[{args.model} closedset] {n_done}/{total} {el:.0f}s ({el / n_done:.2f}s/it)', flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print(f'[{args.model}] closed-set DONE {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
