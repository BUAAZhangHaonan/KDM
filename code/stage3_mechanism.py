"""Stage 3 mechanism check (interpretive, not a judgment condition).

For retrieval-failure samples (open-set wrong, closed-set correct, deficient
stratum): teacher-forced mean log-likelihood of the gold class name under
(image + STYLE1 prompt) vs (blank image + STYLE1 prompt). If the image
condition significantly raises the gold-name LL while the open-set output was
occupied by a common word, the "knowledge present, output occupied" reading
gains direct support.

Usage:
  ./venv/bin/python code/stage3_mechanism.py --model q4b --gpu 4
"""
import os, sys, json, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for var, sub in [('HF_HOME', 'hf'), ('TORCH_HOME', 'torch'), ('XDG_CACHE_HOME', 'xdg'),
                 ('MPLCONFIGDIR', 'mpl'), ('NLTK_DATA', 'nltk')]:
    os.environ[var] = str(ROOT / 'cache' / sub)
sys.path.insert(0, str(ROOT / 'code'))

MODELS = {'q4b': '/home/g203-4028/Models/Qwen3.5-4B',
          'q9b': '/home/g203-4028/Models/Qwen3.5-9B'}


def retrieval_files(model):
    S = {}
    for line in open(ROOT / 'outputs' / 'raw' / f'{model}_main_naming.jsonl'):
        r = json.loads(line)
        if r.get('task') != 'naming':
            continue
        d = S.setdefault(r['file'], {'stratum': r['stratum'], 'class': r['class'], 'open': {}})
        if r['method'] == 'roundA':
            d['ra'] = bool(r.get('abstained'))
        else:
            d['open'][r['method']] = r['outcome']
    for line in open(ROOT / 'outputs' / 'raw' / f'{model}_stage3_closedset.jsonl'):
        r = json.loads(line)
        if r['variant'] == 'full101':
            S.setdefault(r['file'], {}).setdefault('closed', {})
            S[r['file']]['closed'] = bool(r['correct'])
    out = []
    for f, d in S.items():
        if d.get('ra'):
            continue
        o = d['open'].get('direct')
        if o in ('wrong', 'wrong_unparsed') and d.get('closed') is True and d['stratum'] == 'deficient':
            out.append((f, d['class']))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--gpu', type=int, required=True)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from engine import ExpModel, make_variants
    from prompts import STYLE1

    em = ExpModel(MODELS[args.model], f'cuda:{args.gpu}')
    tok = em.proc.tokenizer
    manifest = {m['file']: m for m in map(json.loads, open(ROOT / 'data' / 'samples_manifest.jsonl'))}
    files = retrieval_files(args.model)
    out = f'{args.model}_stage3_mechanism.jsonl'
    done = set()
    p = ROOT / 'outputs' / 'raw' / out
    if p.exists():
        done = {json.loads(l)['key'] for l in open(p)}
    print(f'[{args.model}] mechanism: {len(files)} retrieval-failure samples', flush=True)

    def mean_ll(pil, gold_name):
        inputs = em.build(pil, STYLE1)
        gold = tok(' ' + gold_name, add_special_tokens=False)['input_ids']
        ids = torch.cat([inputs['input_ids'], torch.tensor([gold], device=em.device)], 1)
        am = torch.ones_like(ids)
        kw = {k: v for k, v in inputs.items() if k not in ('input_ids', 'attention_mask')}
        if 'input_token_type' in kw:
            kw['input_token_type'] = torch.cat(
                [kw['input_token_type'],
                 torch.zeros((1, len(gold)), dtype=kw['input_token_type'].dtype, device=em.device)], 1)
        if 'mm_token_type_ids' in kw:
            kw['mm_token_type_ids'] = torch.cat(
                [kw['mm_token_type_ids'],
                 torch.zeros((1, len(gold)), dtype=kw['mm_token_type_ids'].dtype, device=em.device)], 1)
        with torch.no_grad():
            o = em.model(input_ids=ids, attention_mask=am, **kw)
        lp = F.log_softmax(o.logits[0].float(), -1)
        L = inputs['input_ids'].shape[1]
        vals = [float(lp[L - 1 + t, g]) for t, g in enumerate(gold)]
        return sum(vals) / len(vals)

    t0 = time.time()
    with open(p, 'a') as fo:
        for i, (f, cls) in enumerate(files):
            if f in done:
                continue
            img = Image.open(manifest[f]['image_path']).convert('RGB')
            blank = make_variants(img)['blank']
            nice = cls.replace('_', ' ')
            ll_img, ll_blk = mean_ll(img, nice), mean_ll(blank, nice)
            fo.write(json.dumps({'key': f, 'model': args.model, 'file': f, 'class': cls,
                                 'll_image': round(ll_img, 4), 'll_blank': round(ll_blk, 4),
                                 'd_ll': round(ll_img - ll_blk, 4)}) + '\n')
            if (i + 1) % 50 == 0:
                print(f'[{args.model} mech] {i + 1}/{len(files)} {time.time() - t0:.0f}s', flush=True)
    print(f'[{args.model}] mechanism DONE {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
