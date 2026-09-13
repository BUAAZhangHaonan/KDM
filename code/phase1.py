"""Phase 1 (critical path): per-model stratification + abstention pre-test +
strata validation on the held-out evaluation half.

Stages (single process, model loaded once):
  group    : STYLE1 direct naming on ALL grouping-half images (101x24)
  strata   : bottom-25% classes = knowledge-deficient, top-25% = known (by
             grouping-half accuracy; middle excluded, reported)
  abstain  : 3 prompt styles on 10 grouping-half images per deficient class
  validate : STYLE1 direct naming on EVAL-half images of strata classes only;
             PASS iff (known - deficient) class-level accuracy difference CI
             excludes 0 AND deficient mean accuracy <= 0.25

Outputs:
  outputs/raw/{model}_phase1_group.jsonl
  outputs/raw/{model}_phase1_abstain.jsonl
  outputs/raw/{model}_phase1_eval.jsonl
  data/strata_{model}.json
"""
import os, sys, json, time, argparse, random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
os.environ['HF_HOME'] = str(ROOT / 'cache' / 'hf')
os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache' / 'xdg')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache' / 'mpl')
os.environ['NLTK_DATA'] = str(ROOT / 'cache' / 'nltk')
sys.path.insert(0, str(ROOT / 'code'))

MODELS = {'q4b': '/home/g203-4028/Models/Qwen3.5-4B',
          'q9b': '/home/g203-4028/Models/Qwen3.5-9B'}


def load_manifest():
    return [json.loads(l) for l in open(ROOT / 'data' / 'samples_manifest.jsonl')]


def all_classes():
    return sorted({m['class'] for m in load_manifest()})


def append(path, rec):
    with open(path, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def done_keys(path):
    keys = set()
    p = ROOT / 'outputs' / 'raw' / path
    if p.exists():
        for line in open(p):
            try:
                r = json.loads(line)
                keys.add(r['key'])
            except Exception:
                pass
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--n-abstain', type=int, default=10)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from engine import ExpModel, make_variants, HP
    import scoring, prompts

    em = ExpModel(MODELS[args.model], f'cuda:{args.gpu}')
    classes = all_classes()
    manifest = load_manifest()
    raw = ROOT / 'outputs' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)

    # ---------------- stage: group ----------------
    gfile = f'{args.model}_phase1_group.jsonl'
    done = done_keys(gfile)
    todo = [m for m in manifest if m['part'] == 'group' and f"g:{m['file']}" not in done]
    t0 = time.time()
    for i, m in enumerate(todo):
        img = Image.open(m['image_path']).convert('RGB')
        r = em.decode_single(em.build(img, prompts.STYLE1), HP['max_new_tokens_naming'])
        outcome = scoring.score_naming(r['text'], m['class'], classes)
        append(gfile, {'key': f"g:{m['file']}", 'model': args.model, 'class': m['class'],
                       'part': 'group', 'image_path': m['image_path'], 'file': m['file'],
                       'prompt': prompts.STYLE1, 'text': r['text'], 'outcome': outcome,
                       'answer_maxp': r['answer_maxp'], 'first_entropy': r['first_entropy'],
                       'style': 'style1', 'wall_s': round(r['wall_s'], 3)})
        if (i + 1) % 200 == 0:
            print(f'[{args.model}] group {i+1}/{len(todo)} {time.time()-t0:.0f}s', flush=True)

    # ---------------- strata ----------------
    recs = [json.loads(l) for l in open(raw / gfile)]
    by_class = defaultdict(list)
    for r in recs:
        by_class[r['class']].append(r['outcome'] == 'correct')
    acc = {c: sum(v) / len(v) for c, v in by_class.items() if v}
    order = sorted(acc, key=lambda c: (acc[c], c))  # ascending, name tiebreak
    k = len(order) // 4
    deficient = order[:k]
    known = order[-k:]
    middle = order[k:-k]
    strata = {'model': args.model, 'rule': 'bottom-25%/top-25% by grouping-half accuracy',
              'k': k, 'deficient': deficient, 'known': known,
              'group_acc': {c: round(acc[c], 4) for c in order}}
    json.dump(strata, open(ROOT / 'data' / f'strata_{args.model}.json', 'w'), indent=1)
    print(f'[{args.model}] strata: deficient n={len(deficient)} mean_acc='
          f'{sum(acc[c] for c in deficient)/k:.3f} | known n={len(known)} mean_acc='
          f'{sum(acc[c] for c in known)/k:.3f}', flush=True)

    # ---------------- stage: abstention pre-test ----------------
    afile = f'{args.model}_phase1_abstain.jsonl'
    done = done_keys(afile)
    rng = random.Random(123)
    pool = [m for m in manifest if m['part'] == 'group' and m['class'] in deficient]
    picks = []
    for c in deficient:
        cs = [m for m in pool if m['class'] == c]
        rng.shuffle(cs)
        picks += cs[:args.n_abstain]
    todo = [m for m in picks if f"a:{m['file']}" not in done]
    t0 = time.time()
    for i, m in enumerate(todo):
        img = Image.open(m['image_path']).convert('RGB')
        imgg = Image.open(m['image_path']).convert('RGB')
        # style2
        r2 = em.decode_single(em.build(img, prompts.STYLE2), HP['max_new_tokens_naming'] + 4)
        abst2, name2 = scoring.parse_style2(r2['text'])
        out2 = 'abstain' if abst2 else scoring.score_naming(name2, m['class'], classes)
        append(afile, {'key': f"a:{m['file']}", 'model': args.model, 'style': 'style2',
                       'class': m['class'], 'file': m['file'], 'text': r2['text'],
                       'abstained': abst2, 'outcome': out2,
                       'answer_maxp': r2['answer_maxp']})
        # style3: round A yes/no + round B naming
        ra = em.decode_single(em.build(imgg, prompts.STYLE3_ROUND_A), 4)
        rb = em.decode_single(em.build(imgg, prompts.STYLE1), HP['max_new_tokens_naming'])
        na = scoring._norm(ra['text'])
        abst3 = bool(na.startswith('no')) and not na.startswith(('no_idea',))
        out3 = 'abstain' if abst3 else scoring.score_naming(rb['text'], m['class'], classes)
        append(afile, {'key': f"a:{m['file']}", 'model': args.model, 'style': 'style3',
                       'class': m['class'], 'file': m['file'],
                       'text': f"[A]{ra['text']} [B]{rb['text']}",
                       'abstained': abst3, 'outcome': out3,
                       'answer_maxp': rb['answer_maxp']})
        # style1 baseline from the group stage
        g = next((r for r in recs if r['file'] == m['file']), None)
        if g:
            append(afile, {'key': f"a:{m['file']}", 'model': args.model, 'style': 'style1',
                           'class': m['class'], 'file': m['file'], 'text': g['text'],
                           'abstained': g['outcome'] == 'abstain', 'outcome': g['outcome'],
                           'answer_maxp': g['answer_maxp']})
        if (i + 1) % 100 == 0:
            print(f'[{args.model}] abstain {i+1}/{len(todo)} {time.time()-t0:.0f}s', flush=True)

    # ---------------- stage: validate on eval half ----------------
    efile = f'{args.model}_phase1_eval.jsonl'
    done = done_keys(efile)
    strata_classes = set(deficient) | set(known)
    todo = [m for m in manifest if m['part'] == 'eval' and m['class'] in strata_classes
            and f"e:{m['file']}" not in done]
    t0 = time.time()
    for i, m in enumerate(todo):
        img = Image.open(m['image_path']).convert('RGB')
        r = em.decode_single(em.build(img, prompts.STYLE1), HP['max_new_tokens_naming'])
        outcome = scoring.score_naming(r['text'], m['class'], classes)
        stratum = 'deficient' if m['class'] in deficient else 'known'
        append(efile, {'key': f"e:{m['file']}", 'model': args.model, 'class': m['class'],
                       'stratum': stratum, 'part': 'eval', 'file': m['file'],
                       'prompt': prompts.STYLE1, 'text': r['text'], 'outcome': outcome,
                       'answer_maxp': r['answer_maxp'], 'first_entropy': r['first_entropy'],
                       'style': 'style1', 'wall_s': round(r['wall_s'], 3)})
        if (i + 1) % 200 == 0:
            print(f'[{args.model}] eval {i+1}/{len(todo)} {time.time()-t0:.0f}s', flush=True)

    print(f'[{args.model}] PHASE1 DONE', flush=True)


if __name__ == '__main__':
    main()
