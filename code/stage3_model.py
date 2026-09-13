"""Stage 3 cross-model / cross-domain pipeline for ONE (model, domain) pair.

Stages (all resumable, run in order):
  group     : direct STYLE1 naming on the grouping half, all classes
  strata    : bottom/top 25% classes by grouping accuracy -> strata json
  validate  : STYLE1 on eval half of strata classes + Welch gate (PASS/FAIL)
  closedset : full101 + near10 on eval half of strata classes
  main      : direct/vcd/mib/lcd STYLE1 naming on eval half of strata classes

Extended models use style1 only (no abstention experiment, per task section 5).
Domains: food101 (v2 manifest) and dogs (stage-3 manifest).

Usage:
  ./venv/bin/python code/stage3_model.py --key q3vl4b --gpu 4 --domain food101
"""
import os, sys, json, time, argparse, random, re, gc, math
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
for var, sub in [('HF_HOME', 'hf'), ('TORCH_HOME', 'torch'), ('XDG_CACHE_HOME', 'xdg'),
                 ('MPLCONFIGDIR', 'mpl'), ('NLTK_DATA', 'nltk')]:
    os.environ[var] = str(ROOT / 'cache' / sub)
sys.path.insert(0, str(ROOT / 'code'))

XMODELS = {
    'q4b': '/home/g203-4028/Models/Qwen3.5-4B',
    'q9b': '/home/g203-4028/Models/Qwen3.5-9B',
    'internvl4b': '/home/g203-4028/Models/InternVL3_5-4B',
    'q3vl4b': '/home/g203-4028/Models/Qwen3-VL-4B-Instruct',
    'q3vl8b': '/home/g203-4028/Models/Qwen3-VL-8B-Instruct',
    'llava16': '/home/g203-4028/Models/llava-v1.6-mistral-7b-hf',
    'glm46v': '/home/g203-4028/Models/GLM-4.6V-Flash',
}
DOMAINS = {
    'food101': {'manifest': 'data/samples_manifest.jsonl', 'noun': 'dishes',
                'neighbours': 'data/neighbours_food101.json'},
    'dogs': {'manifest': 'data/samples_manifest_dogs.jsonl', 'noun': 'dog breeds',
             'neighbours': 'data/neighbours_stanford_dogs.json'},
}
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


def closedset_prompt(cands, noun):
    lines = "\n".join(f"{i + 1}. {c.replace('_', ' ')}" for i, c in enumerate(cands))
    return (f"Look at the image. Which of the following {noun} is shown?\n"
            "Reply with the number only.\n"
            f"{lines}\nAnswer:")


def validate_strata(cls_acc_def, cls_acc_kn):
    """Class-level Welch t CI on the eval-half accuracy difference."""
    import numpy as np

    a, b = np.asarray(cls_acc_def, float), np.asarray(cls_acc_kn, float)
    diff = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    return float(diff), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True, choices=list(XMODELS))
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--domain', required=True, choices=list(DOMAINS))
    ap.add_argument('--stages', default='group,strata,validate,closedset,main')
    args = ap.parse_args()

    import torch
    import numpy as np
    from PIL import Image
    from stage3_engine import get_engine
    from engine import make_variants, jsd, HP
    from stage3_closedset import build_neighbours
    import scoring, prompts

    dom = DOMAINS[args.domain]
    tag = f"{args.key}_{args.domain}"
    manifest = [json.loads(l) for l in open(ROOT / dom['manifest'])]
    classes_all = sorted({m['class'] for m in manifest})
    k = max(2, round(0.25 * len(classes_all)))

    t0 = time.time()
    em = get_engine(XMODELS[args.key], f"cuda:{args.gpu}")
    caps = {'model_type': em.mt, 'lcd_ready': em.lcd_ready, 'norm_path': getattr(em, 'norm_path', None)}
    json.dump(caps, open(ROOT / 'data' / f'caps_{tag}.json', 'w'), indent=1)
    print(f"[{tag}] loaded {em.mt} lcd_ready={em.lcd_ready} "
          f"({time.time() - t0:.0f}s)", flush=True)
    lcd_ok = em.lcd_ready
    # Qwen3-VL encodes at native resolution (~9 s/decode measured); cap the
    # max side to 640 px for this family so runs are feasible. Documented
    # per-model input setting; applied uniformly to every variant/config.
    resize_max = 640 if em.mt == 'qwen3_vl' else None

    def prep(m):
        img = Image.open(m['image_path']).convert('RGB')
        if resize_max:
            img.thumbnail((resize_max, resize_max))
        return img

    stages = args.stages.split(',')

    # ---------------- group ----------------
    if 'group' in stages:
        out = f'{tag}_group.jsonl'
        done = done_keys(out)
        evals = [m for m in manifest if m['part'] == 'group']
        n = 0
        for m in evals:
            key = f"gr:{m['file']}"
            if key in done:
                continue
            img = prep(m)
            r = em.decode_single(em.build(img, prompts.STYLE1), HP['max_new_tokens_naming'])
            append(out, {'key': key, 'model': args.key, 'domain': args.domain,
                         'task': 'group', 'method': 'direct', 'stratum': '',
                         'class': m['class'], 'file': m['file'],
                         'text': r['text'], 'outcome': scoring.score_naming(r['text'], m['class'], classes_all),
                         'answer_maxp': r['answer_maxp'],
                         'n_forwards': r['n_forwards'], 'wall_s': round(r['wall_s'], 3)})
            n += 1
            if n % 100 == 0:
                print(f'[{tag} group] {n}/{len(evals)} {time.time() - t0:.0f}s', flush=True)
            gc.collect(); torch.cuda.empty_cache()
        print(f'[{tag} group done]', flush=True)

    # ---------------- strata ----------------
    if 'strata' in stages:
        sp = ROOT / 'data' / f'strata_{tag}.json'
        if sp.exists():
            print(f'[{tag}] strata exist, skip', flush=True)
        else:
            acc = defaultdict(lambda: [0, 0])
            for line in open(ROOT / 'outputs' / 'raw' / f'{tag}_group.jsonl'):
                r = json.loads(line)
                acc[r['class']][0] += (r['outcome'] == 'correct')
                acc[r['class']][1] += 1
            ca = {c: v[0] / v[1] for c, v in acc.items()}
            order = sorted(ca, key=lambda c: (ca[c], c))
            st = {'model': args.key, 'domain': args.domain, 'k': k,
                  'rule': 'bottom-25%/top-25% by grouping-half accuracy',
                  'deficient': order[:k], 'known': order[-k:], 'group_acc': ca}
            json.dump(st, open(sp, 'w'), indent=1)
            print(f"[{tag}] strata: deficient head {st['deficient'][:3]}", flush=True)

    strata = json.load(open(ROOT / 'data' / f'strata_{tag}.json'))
    stratum_of = {c: 'deficient' for c in strata['deficient']}
    stratum_of.update({c: 'known' for c in strata['known']})

    # ---------------- validate ----------------
    if 'validate' in stages:
        out = f'{tag}_eval.jsonl'
        done = done_keys(out)
        evals = [m for m in manifest if m['part'] == 'eval' and m['class'] in stratum_of]
        n = 0
        for m in evals:
            key = f"ev:{m['file']}"
            if key in done:
                continue
            img = prep(m)
            r = em.decode_single(em.build(img, prompts.STYLE1), HP['max_new_tokens_naming'])
            append(out, {'key': key, 'model': args.key, 'domain': args.domain,
                         'task': 'validate', 'method': 'direct', 'stratum': stratum_of[m['class']],
                         'class': m['class'], 'file': m['file'],
                         'text': r['text'], 'outcome': scoring.score_naming(r['text'], m['class'], classes_all),
                         'answer_maxp': r['answer_maxp'],
                         'n_forwards': r['n_forwards'], 'wall_s': round(r['wall_s'], 3)})
            n += 1
            if n % 100 == 0:
                print(f'[{tag} validate] {n}/{len(evals)} {time.time() - t0:.0f}s', flush=True)
            gc.collect(); torch.cuda.empty_cache()
        # gate
        acc = defaultdict(lambda: [0, 0])
        for line in open(ROOT / 'outputs' / 'raw' / out):
            r = json.loads(line)
            acc[(r['stratum'], r['class'])][0] += (r['outcome'] == 'correct')
            acc[(r['stratum'], r['class'])][1] += 1
        ca_d = [v[0] / v[1] for (s, c), v in acc.items() if s == 'deficient']
        ca_k = [v[0] / v[1] for (s, c), v in acc.items() if s == 'known']
        diff, lo, hi = validate_strata(ca_d, ca_k)   # diff = deficient - known
        mean_d, mean_k = float(np.mean(ca_d)), float(np.mean(ca_k))
        passed = bool(hi < 0 and mean_d <= 0.25)   # CI entirely below zero
        res = {'model': args.key, 'domain': args.domain, 'eval_acc_known': round(mean_k, 4),
               'eval_acc_deficient': round(mean_d, 4), 'diff': round(diff, 4),
               'ci': [round(lo, 4), round(hi, 4)], 'PASS': passed}
        json.dump(res, open(ROOT / 'data' / f'strata_validation_{tag}.json', 'w'), indent=1)
        print(f'[{tag}] VALIDATION {"PASS" if passed else "FAIL"} {res}', flush=True)
        if not passed:
            print(f'[{tag}] domain/model excluded by gate; stopping before closedset/main', flush=True)
            return

    # ---------------- closedset ----------------
    if 'closedset' in stages:
        nb = build_neighbours(classes_all, 9, ROOT / dom['neighbours'])
        out = f'{tag}_closedset.jsonl'
        done = done_keys(out)
        evals = [m for m in manifest if m['part'] == 'eval' and m['class'] in stratum_of]
        n, total = 0, 2 * len(evals)
        for m in evals:
            img = prep(m)
            cls, st = m['class'], stratum_of[m['class']]
            for var in ['full101', 'near10']:
                key = f"cs:{m['file']}:{var}"
                if key in done:
                    continue
                pool = classes_all if var == 'full101' else [cls] + nb[cls]
                order = list(range(len(pool)))
                random.Random(f"{m['file']}|{var}").shuffle(order)
                cands = [pool[i] for i in order]
                gold_idx = cands.index(cls)
                r = em.decode_single(em.build(img, closedset_prompt(cands, dom['noun'])), 6)
                mm = re.search(r'\d+', r['text'])
                parsed = int(mm.group()) if mm else None
                append(out, {'key': key, 'model': args.key, 'domain': args.domain,
                             'task': 'closedset', 'variant': var, 'stratum': st,
                             'class': cls, 'file': m['file'], 'n_options': len(cands),
                             'gold_idx': gold_idx, 'text': r['text'], 'parsed': parsed,
                             'correct': bool(parsed == gold_idx + 1),
                             'n_forwards': r['n_forwards'], 'wall_s': round(r['wall_s'], 3)})
                n += 1
                if n % 100 == 0:
                    print(f'[{tag} closedset] {n}/{total} {time.time() - t0:.0f}s', flush=True)
            gc.collect(); torch.cuda.empty_cache()

    # ---------------- main ----------------
    if 'main' in stages:
        out = f'{tag}_main.jsonl'
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
                            append(out, {'key': f"{sid}:lcd", 'model': args.key, 'domain': args.domain,
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
                    append(out, {'key': f"{sid}:{mm}", 'model': args.key, 'domain': args.domain,
                                 'task': 'naming', 'method': mm, 'stratum': st, 'class': cls,
                                 'file': m['file'], 'prompt': prompts.STYLE1, 'style': 'style1',
                                 'text': r['text'], 'outcome': scoring.score_naming(r['text'], cls, classes_all),
                                 'answer_maxp': r['answer_maxp'], 'signals': signals,
                                 'n_forwards': r['n_forwards'], 'wall_s': round(r['wall_s'], 3)})
                except Exception as ex:
                    append(out, {'key': f"{sid}:{mm}", 'model': args.key, 'domain': args.domain,
                                 'task': 'naming', 'method': mm, 'stratum': st, 'class': cls,
                                 'file': m['file'], 'error': str(ex)[:300]})
                    print(f'[{tag}] ERROR {sid}:{mm}: {str(ex)[:120]}', flush=True)
            n += 1
            if n % 25 == 0:
                print(f'[{tag} main] {n}/{len(evals)} {time.time() - t0:.0f}s', flush=True)
            gc.collect(); torch.cuda.empty_cache()
    print(f'[{tag}] ALL DONE {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
