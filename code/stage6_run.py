"""Stage-6 runner: faithful re-run of the five decoding configs (direct + VCD +
M3ID + DoLa + DeCo) on the food101 strata, with enriched per-sample recording.

Parts (per docs/stage6/PREREGISTER_STAGE6.md section 3):
  eval : evaluation half, both strata (25 classes x 24 imgs each) - experiment A
  dev  : grouping-half development subset, 200/stratum (seed 607) - fits for the
         B control-2 temperature and the F offset estimate
  dose : q4b only, VCD alpha in {0.5, 2.0} on the eval half - experiment G

q4b keeps its established style3 protocol (round-A YES/NO abstention decided once
per sample under direct decoding, then STYLE1 naming for every config); the other
models use style1. Strata labels are the stage-5 renames: low_acc / high_acc.

Usage:
  ./venv/bin/python code/stage6_run.py --model q4b --gpu 2 --parts eval,dev --dose
  ./venv/bin/python code/stage6_run.py --model q4b --gpu 2 --limit 3   # sanity
"""
import os, sys, json, time, argparse, random, gc, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ['HF_HOME'] = str(ROOT / 'cache' / 'hf')
os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache' / 'xdg')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache' / 'mpl')
os.environ['NLTK_DATA'] = str(ROOT / 'cache' / 'nltk')
sys.path.insert(0, str(ROOT / 'code'))

MODELS = {
    'q4b': ('/home/g203-4028/Models/Qwen3.5-4B', 'style3'),
    'q9b': ('/home/g203-4028/Models/Qwen3.5-9B', 'style1'),
    'llava16': ('/home/g203-4028/Models/llava-v1.6-mistral-7b-hf', 'style1'),
    'internvl4b': ('/home/g203-4028/Models/InternVL3_5-4B', 'style1'),
}
STRATA_FILE = {
    'q4b': 'data/strata_q4b.json', 'q9b': 'data/strata_q9b.json',
    'llava16': 'data/strata_llava16_food101.json',
    'internvl4b': 'data/strata_internvl4b_food101.json',
}
NAMING_TOKENS = 12
DEV_N, DEV_SEED = 200, 607
DOSE_ALPHAS = {'vcd_a05': 0.5, 'vcd_a20': 2.0}


def done_keys(path):
    """Keys of non-error records (error records are retried on resume)."""
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
    ap.add_argument('--parts', default='eval,dev')
    ap.add_argument('--dose', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='sanity: first N eval imgs')
    ap.add_argument('--methods', default='direct,vcd,m3id,dola,deco')
    args = ap.parse_args()

    import torch
    import numpy as np
    from PIL import Image
    import scoring, prompts
    from stage6_engine import S6Model, HP6

    model_path, style = MODELS[args.model]
    em = S6Model(model_path, f'cuda:{args.gpu}')
    classes_all = sorted({m['class'] for m in
                          map(json.loads, open(ROOT / 'data' / 'samples_manifest.jsonl'))})
    strata = json.load(open(ROOT / STRATA_FILE[args.model]))
    stratum_of = {c: 'low_acc' for c in strata['deficient']}
    stratum_of.update({c: 'high_acc' for c in strata['known']})
    manifest = [json.loads(l) for l in open(ROOT / 'data' / 'samples_manifest.jsonl')]

    naming_prompt = prompts.STYLES[style][-1]
    # M3ID schedule offset t0: token count of the question text (preregister sec 2.2)
    t0_sched = len(em.tokenizer(naming_prompt, add_special_tokens=False)['input_ids'])
    print(f'[{args.model}] n_layers={em.n_layers} deco_layers={em.deco_layers} '
          f't0_sched={t0_sched} style={style}', flush=True)

    methods = args.methods.split(',')
    t_start = time.time()
    n_done = 0

    def run_part(tag, samples, method_list):
        nonlocal n_done
        out = f'{args.model}_s6_{tag}_naming.jsonl'
        done = set()
        ra_abstain = {}
        p = ROOT / 'outputs' / 'raw' / out
        if p.exists():
            for line in open(p):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if 'error' in r:
                    continue
                done.add(r['key'])
                if r.get('method') == 'roundA':
                    ra_abstain[r['file']] = r.get('abstained')
        firstdist = {}
        total = len(samples)
        for m in samples:
            img = Image.open(m['image_path']).convert('RGB')
            cls, st = m['class'], stratum_of[m['class']]
            sid = f"nm:{m['file']}"
            need = [mm for mm in method_list if f"{sid}:{mm}" not in done]
            if not need:
                continue
            inputs = em.build(img, naming_prompt)
            inputs_prior = em.build_text_only(naming_prompt) if 'm3id' in need else None
            roundA_abstained = None
            if style == 'style3':
                if m['file'] in ra_abstain:
                    roundA_abstained = ra_abstain[m['file']]
                else:
                    ra = em.em.decode_single(em.build(img, prompts.STYLE3_ROUND_A), 4)
                    na = scoring._norm(ra['text'])
                    roundA_abstained = bool(na.startswith('no') and
                                            not na.startswith('no_idea'))
                    append(out, {'key': f"{sid}:ra", 'model': args.model,
                                 'task': 'naming', 'method': 'roundA', 'stratum': st,
                                 'class': cls, 'file': m['file'], 'text': ra['text'],
                                 'abstained': roundA_abstained})
                    done.add(f"{sid}:ra")
                    ra_abstain[m['file']] = roundA_abstained
            for mm in need:
                try:
                    if mm == 'direct':
                        r = em.decode_single(inputs, NAMING_TOKENS, method='direct')
                    elif mm == 'vcd':
                        seed = int(hashlib.sha256(
                            f"{m['file']}:vcd".encode()).hexdigest(), 16) % (2 ** 31)
                        r = em.decode_two_branch(
                            inputs, em.noised_inputs(inputs, seed), 'vcd',
                            NAMING_TOKENS)
                    elif mm == 'm3id':
                        r = em.decode_two_branch(inputs, inputs_prior, 'm3id',
                                                 NAMING_TOKENS, t0_sched=t0_sched)
                    elif mm == 'dola':
                        r = em.decode_single(inputs, NAMING_TOKENS, method='dola')
                    elif mm == 'deco':
                        r = em.decode_single(inputs, NAMING_TOKENS, method='deco')
                    elif mm in DOSE_ALPHAS:
                        seed = int(hashlib.sha256(
                            f"{m['file']}:vcd".encode()).hexdigest(), 16) % (2 ** 31)
                        r = em.decode_two_branch(
                            inputs, em.noised_inputs(inputs, seed), 'vcd',
                            NAMING_TOKENS,
                            alpha_override=DOSE_ALPHAS[mm])
                    else:
                        raise ValueError(mm)
                    outcome = scoring.score_naming(r['text'], cls, classes_all)
                    if style == 'style3' and roundA_abstained:
                        outcome = 'abstain'
                    fp = r.pop('first_probs', None)
                    if mm == 'direct' and fp is not None:
                        firstdist[m['file']] = fp.detach().cpu().half().numpy()
                        if len(firstdist) % 100 == 0:  # crash-safe periodic dump
                            np.savez(ROOT / 'outputs' / 'raw' /
                                     f'{args.model}_s6_firstdist_{tag}.npz',
                                     **firstdist)
                    rec = {'key': f"{sid}:{mm}", 'model': args.model, 'task': 'naming',
                           'method': mm, 'stratum': st, 'class': cls, 'file': m['file'],
                           'prompt': naming_prompt, 'style': style, 'text': r['text'],
                           'tokens': r['tokens'], 'step_probs': r['step_probs'],
                           'outcome': outcome, 'answer_maxp': r['answer_maxp'],
                           'first_entropy': r['first_entropy'],
                           'first_maxp': r['first_maxp'],
                           'n_forwards': r['n_forwards'],
                           'n_prefill_tokens': r['n_prefill_tokens'],
                           'wall_s': r['wall_s'],
                           'layers_selected': r['layers_selected'],
                           'variant': 'faithful',
                           'abstained': roundA_abstained}
                    if mm == 'm3id':
                        rec['t0_sched'] = t0_sched
                    if mm in DOSE_ALPHAS:
                        rec['alpha'] = DOSE_ALPHAS[mm]
                    append(out, rec)
                    done.add(f"{sid}:{mm}")
                except Exception as e:  # record error, retried on resume
                    append(out, {'key': f"{sid}:{mm}", 'model': args.model,
                                 'task': 'naming', 'method': mm, 'stratum': st,
                                 'class': cls, 'file': m['file'],
                                 'error': f'{type(e).__name__}: {e}'})
                    print(f'  [error] {sid}:{mm} {type(e).__name__}: {e}',
                          flush=True)
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t_start
                print(f'[{args.model} {tag}] {n_done}/{total} {el:.0f}s '
                      f'({el / n_done:.2f}s/img)', flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        if firstdist:
            np.savez(ROOT / 'outputs' / 'raw' / f'{args.model}_s6_firstdist_{tag}.npz',
                     **firstdist)
            print(f'[{args.model} {tag}] saved firstdist npz '
                  f'({len(firstdist)} entries)', flush=True)

    parts = args.parts.split(',')
    for part in parts:
        if part == 'eval':
            samples = [m for m in manifest
                       if m['part'] == 'eval' and m['class'] in stratum_of]
            if args.limit:
                samples = samples[:args.limit]
            run_part('eval', samples, methods)
        elif part == 'dev':
            rng = random.Random(DEV_SEED)
            dev = []
            for stratum in ('low_acc', 'high_acc'):
                pool = sorted([m for m in manifest if m['part'] == 'group'
                               and stratum_of.get(m['class']) == stratum],
                              key=lambda x: (x['class'], x['file']))
                dev += rng.sample(pool, min(DEV_N, len(pool)))
            if args.limit:
                dev = dev[:args.limit]
            run_part('dev', dev, methods)
        else:
            raise ValueError(part)
    if args.dose:
        samples = [m for m in manifest
                   if m['part'] == 'eval' and m['class'] in stratum_of]
        run_part('dose', samples, list(DOSE_ALPHAS))
    print(f'[{args.model}] DONE {args.parts}{"+dose" if args.dose else ""} in '
          f'{(time.time() - t_start) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
