"""Stage 4: name-restricted log-likelihood probe.

Under the SAME prompt (STYLE1) and image as open-set naming, score every
candidate class name as a continuation with the LM head (teacher forcing,
sum over the name's tokens); argmax LL = probe prediction. Also computes the
blank-image condition. No generation, no instruction following required.

Scoring for a candidate ids[0..L-1]:
  - prefill prompt once -> lp0 gives log p(ids[0] | prompt, image)
  - incremental forward fed ids[:-1] (against the shared prompt cache) ->
    its logits row j-1 gives log p(ids[j] | prompt, image, ids[..<j])
Cache is restored after each candidate via DynamicCache.crop (deepcopy fallback).

Usage:
  ./venv/bin/python code/stage4_probe.py --model q4b --gpu 4 --domain food101
"""
import os, sys, json, time, argparse, gc, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for var, sub in [('HF_HOME', 'hf'), ('TORCH_HOME', 'torch'), ('XDG_CACHE_HOME', 'xdg'),
                 ('MPLCONFIGDIR', 'mpl'), ('NLTK_DATA', 'nltk')]:
    os.environ[var] = str(ROOT / 'cache' / sub)
sys.path.insert(0, str(ROOT / 'code'))

MODEL_PATHS = {
    'q4b': '/home/g203-4028/Models/Qwen3.5-4B',
    'q9b': '/home/g203-4028/Models/Qwen3.5-9B',
    'q3vl4b': '/home/g203-4028/Models/Qwen3-VL-4B-Instruct',
    'llava16': '/home/g203-4028/Models/llava-v1.6-mistral-7b-hf',
    'internvl4b': '/home/g203-4028/Models/InternVL3_5-4B',
    'glm46v': '/home/g203-4028/Models/GLM-4.6V-Flash',
}
DOMAINS = {
    'food101': {'manifest': 'data/samples_manifest.jsonl'},
    'dogs': {'manifest': 'data/samples_manifest_dogs.jsonl'},
}


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


def strata_path(model, domain):
    if model in ('q4b', 'q9b') and domain == 'food101':
        return ROOT / 'data' / f'strata_{model}.json'
    return ROOT / 'data' / f'strata_{model}_{domain}.json'


def probe_ranks(em, pil, prompt, cand_ids, cand_names, gold_idx, is_internvl):
    """Exact pruned scoring under one image. Returns (gold_rank, gold_ll,
    top_name, n_scored). gold_rank = #{candidates with LL strictly > gold's}
    (0-based). Pruning is exact: LL(c) <= lp0[c_first] since logprobs <= 0;
    candidates are visited in descending lp0 and skipped only when they can
    neither beat the gold LL nor the current best."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        inputs = em.build(pil, prompt)
        if is_internvl:
            out = em._prefill(inputs)
        else:
            out = em.model(**inputs, use_cache=True)
        pkv = out.past_key_values
        lp0 = F.log_softmax(out.logits[:, -1, :].float(), -1)[0]
        firsts = [ids[0] for ids in cand_ids]
        lp0s = [float(lp0[t]) for t in firsts]
        order = sorted(range(len(cand_ids)), key=lambda i: -lp0s[i])

        def full_ll(i, pkv0):
            ids = cand_ids[i]
            ll = lp0s[i]
            rest = ids[:-1]
            if len(rest) > 0:
                pkv_in = copy.deepcopy(pkv0)
                ids_t = torch.tensor([rest], device=em.device)
                if is_internvl:
                    o = em.model.language_model(input_ids=ids_t, past_key_values=pkv_in)
                else:
                    o = em.model(input_ids=ids_t, past_key_values=pkv_in)
                lps = F.log_softmax(o.logits[0].float(), -1)
                for j in range(1, len(ids)):
                    ll += float(lps[j - 1, ids[j]])
            return ll

        gold_ll = full_ll(gold_idx, pkv)
        # gold seeds the best; only candidates whose first-token logprob alone
        # exceeds gold's FULL LL can possibly beat it (logprobs <= 0)
        best_ll, best_i = gold_ll, gold_idx
        n_beat, n_scored = 0, 0
        for i in order:
            if lp0s[i] <= gold_ll:
                break            # sorted desc: no later candidate can matter
            ll = full_ll(i, pkv)
            n_scored += 1
            if ll > best_ll:
                best_ll, best_i = ll, i
            if ll > gold_ll:
                n_beat += 1
        return n_beat, gold_ll, cand_names[best_i], n_scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODEL_PATHS))
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--domain', required=True, choices=list(DOMAINS))
    ap.add_argument('--limit', type=int, default=0, help='smoke-test cap')
    ap.add_argument('--strata-from', default=None,
                    help='use another model-key strata file for sample selection '
                         '(for probe-only models without their own strata)')
    args = ap.parse_args()

    import torch
    from PIL import Image
    from stage3_engine import get_engine
    from engine import make_variants
    from prompts import STYLE1

    em = get_engine(MODEL_PATHS[args.model], f'cuda:{args.gpu}')
    is_internvl = em.mt == 'internvl_chat'
    tok = em.proc.tokenizer
    manifest = [json.loads(l) for l in open(ROOT / DOMAINS[args.domain]['manifest'])]
    classes_all = sorted({m['class'] for m in manifest})
    strata = json.load(open(strata_path(args.strata_from or args.model, args.domain)))
    stratum_of = {c: 'deficient' for c in strata['deficient']}
    stratum_of.update({c: 'known' for c in strata['known']})

    cand_names = [c.replace('_', ' ') for c in classes_all]
    cand_ids = [tok(' ' + n, add_special_tokens=False)['input_ids'] for n in cand_names]
    # guard: a name must not tokenize to empty
    assert all(len(x) > 0 for x in cand_ids), [n for n, x in zip(cand_names, cand_ids) if not x]
    name_to_idx = {classes_all[i]: i for i in range(len(classes_all))}

    tag = f'{args.model}_{args.domain}'
    out = f'{args.model}_stage4_probe_{args.domain}.jsonl'
    done = done_keys(out)
    evals = [m for m in manifest if m['part'] == 'eval' and m['class'] in stratum_of]
    if args.limit:
        evals = evals[:args.limit]
    t0, n = time.time(), 0
    for m in evals:
        key = f"pr4:{m['file']}"
        if key in done:
            continue
        img = Image.open(m['image_path']).convert('RGB')
        if em.mt == 'qwen3_vl':
            img.thumbnail((640, 640))   # same cap as stage-3 main config
        blank = make_variants(img)['blank']
        gold_idx = name_to_idx[m['class']]
        rank_img, ll_img, top_img, ns_i = probe_ranks(em, img, STYLE1, cand_ids, cand_names, gold_idx, is_internvl)
        rank_blk, ll_blk, top_blk, ns_b = probe_ranks(em, blank, STYLE1, cand_ids, cand_names, gold_idx, is_internvl)
        append(out, {'key': key, 'model': args.model, 'domain': args.domain,
                     'stratum': stratum_of[m['class']], 'class': m['class'],
                     'file': m['file'], 'gold_idx': gold_idx,
                     'rank_image': rank_img, 'rank_blank': rank_blk,
                     'correct_image': rank_img == 0, 'correct_blank': rank_blk == 0,
                     'top_image': top_img, 'top_blank': top_blk,
                     'll_gold_image': round(ll_img, 4),
                     'll_gold_blank': round(ll_blk, 4),
                     'n_scored_image': ns_i, 'n_scored_blank': ns_b})
        n += 1
        if n % 25 == 0:
            el = time.time() - t0
            print(f'[{tag} probe] {n}/{len(evals)} {el:.0f}s ({el / n:.2f}s/it)', flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print(f'[{tag} probe DONE] {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
