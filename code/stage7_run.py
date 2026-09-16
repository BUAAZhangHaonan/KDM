"""Stage-7 runner: fidelity close-out inference (PREREGISTER_STAGE7.md sec 2).

Jobs:
  seed       : VCD with two fresh noise-seed salts (s2, s3), food101 eval half
  dola_subset: DoLa with the official-subset candidate layers (even layers
               0,2,...,L-2 plus mature L, README example convention), eval half
  dogs       : faithful four-method rerun on the existing dogs strata
               (direct decoding and the judgment-round abstention are reused
               from the legacy dogs main records; inputs unchanged)

Usage: ./venv/bin/python code/stage7_run.py --model q4b --gpu 2 --jobs seed,dola_subset,dogs
"""
import os, sys, json, time, argparse, gc, hashlib
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
STRATA = {
    'food101': ('data/samples_manifest.jsonl',
                {'q4b': 'data/strata_q4b.json', 'q9b': 'data/strata_q9b',
                 'llava16': 'data/strata_llava16_food101.json',
                 'internvl4b': 'data/strata_internvl4b_food101.json'}),
    'dogs': ('data/samples_manifest_dogs.jsonl',
             {'q4b': 'data/strata_q4b_dogs.json', 'q9b': 'data/strata_q9b_dogs.json'}),
}
NAMING_TOKENS = 12
SEED_JOBS = {'vcd_s2': 's2', 'vcd_s3': 's3'}
LEGACY_DOGS = {'q4b': 'outputs/raw/q4b_dogs_main.jsonl',
               'q9b': 'outputs/raw/q9b_dogs_main.jsonl'}


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
    ap.add_argument('--model', required=True, choices=['q4b', 'q9b'])
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--jobs', default='seed,dola_subset,dogs')
    args = ap.parse_args()

    import torch
    from PIL import Image
    import scoring, prompts
    from stage6_engine import S6Model

    model_path, style = MODELS[args.model]
    em = S6Model(model_path, f'cuda:{args.gpu}')
    classes_food = sorted({m['class'] for m in
                           map(json.loads, open(ROOT / 'data' / 'samples_manifest.jsonl'))})
    t_start = time.time()
    n_done = 0

    def run_food(job, method_list, dola_even=False):
        nonlocal n_done
        out = f'{args.model}_s7_{job}_naming.jsonl'
        strata = json.load(open(STRATA['food101'][1][args.model]))
        stratum_of = {c: 'low_acc' for c in strata['deficient']}
        stratum_of.update({c: 'high_acc' for c in strata['known']})
        manifest = [json.loads(l) for l in open(ROOT / STRATA['food101'][0])]
        samples = [m for m in manifest
                   if m['part'] == 'eval' and m['class'] in stratum_of]
        naming_prompt = prompts.STYLES[style][-1]
        _loop(out, samples, method_list, stratum_of, classes_food, naming_prompt,
              style, job, dola_even=dola_even)
        print(f'[{args.model} {job}] DONE in {(time.time()-t_start)/60:.1f} min '
              f'(cumulative)', flush=True)

    def run_dogs():
        nonlocal n_done
        out = f'{args.model}_s7_dogs_naming.jsonl'
        man_f, strata_f = STRATA['dogs']
        strata = json.load(open(strata_f[args.model]))
        stratum_of = {c: 'low_acc' for c in strata['deficient']}
        stratum_of.update({c: 'high_acc' for c in strata['known']})
        classes_dogs = sorted({m['class'] for m in map(json.loads, open(ROOT / man_f))})
        manifest = [json.loads(l) for l in open(ROOT / man_f)]
        samples = [m for m in manifest
                   if m['part'] == 'eval' and m['class'] in stratum_of]
        # reuse the legacy round-A abstention and direct-decode outcomes (same inputs)
        ra_abstain = {}
        legacy_direct = {}
        for line in open(ROOT / LEGACY_DOGS[args.model]):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('task') != 'naming':
                continue
            if r.get('method') == 'roundA':
                ra_abstain[r['file']] = bool(r.get('abstained'))
            elif r.get('method') == 'direct':
                legacy_direct[r['file']] = r
        naming_prompt = prompts.STYLE1  # dogs used the direct naming protocol
        _loop(out, samples, ['vcd', 'm3id', 'dola', 'deco'], stratum_of,
              classes_dogs, naming_prompt, 'style1', 'dogs',
              ra_abstain=ra_abstain, legacy_direct=legacy_direct)
        print(f'[{args.model} dogs] DONE in {(time.time()-t_start)/60:.1f} min '
              f'(cumulative)', flush=True)

    def _loop(out, samples, method_list, stratum_of, classes_all, naming_prompt,
              style, job, dola_even=False, ra_abstain=None, legacy_direct=None):
        nonlocal n_done
        done = done_keys(out)
        ra_abstain = ra_abstain or {}
        t0_sched = len(em.tokenizer(naming_prompt,
                                    add_special_tokens=False)['input_ids'])
        if dola_even:
            em._dola_candidates_override = list(range(0, em.n_layers, 2))
        else:
            em._dola_candidates_override = None
        for m in samples:
            cls, st = m['class'], stratum_of[m['class']]
            sid = f"nm:{m['file']}"
            need = [mm for mm in method_list if f"{sid}:{mm}" not in done]
            if not need:
                continue
            img = Image.open(m['image_path']).convert('RGB')
            inputs = em.build(img, naming_prompt)
            inputs_prior = em.build_text_only(naming_prompt) if 'm3id' in need else None
            roundA_abstained = ra_abstain.get(m['file'])
            if style == 'style3' and roundA_abstained is None:
                ra = em.em.decode_single(em.build(img, prompts.STYLE3_ROUND_A), 4)
                na = scoring._norm(ra['text'])
                roundA_abstained = bool(na.startswith('no') and
                                        not na.startswith('no_idea'))
            for mm in need:
                try:
                    if mm in SEED_JOBS:
                        salt = SEED_JOBS[mm]
                        seed = int(hashlib.sha256(
                            f"{m['file']}:vcd:{salt}".encode()).hexdigest(), 16) % (2 ** 31)
                        r = em.decode_two_branch(
                            inputs, em.noised_inputs(inputs, seed), 'vcd', NAMING_TOKENS)
                        r['method'] = mm
                        r['noise_salt'] = salt
                    elif mm == 'dola_subset':
                        r = em.decode_single(inputs, NAMING_TOKENS, method='dola')
                    elif mm == 'vcd':
                        seed = int(hashlib.sha256(
                            f"{m['file']}:vcd".encode()).hexdigest(), 16) % (2 ** 31)
                        r = em.decode_two_branch(
                            inputs, em.noised_inputs(inputs, seed), 'vcd', NAMING_TOKENS)
                    elif mm == 'm3id':
                        r = em.decode_two_branch(inputs, inputs_prior, 'm3id',
                                                 NAMING_TOKENS, t0_sched=t0_sched)
                    elif mm == 'dola':
                        r = em.decode_single(inputs, NAMING_TOKENS, method='dola')
                    elif mm == 'deco':
                        r = em.decode_single(inputs, NAMING_TOKENS, method='deco')
                    else:
                        raise ValueError(mm)
                    outcome = scoring.score_naming(r['text'], cls, classes_all)
                    if style == 'style3' and roundA_abstained:
                        outcome = 'abstain'
                    r.pop('first_probs', None)
                    rec = {'key': f"{sid}:{mm}", 'model': args.model,
                           'task': 'naming', 'method': mm, 'stratum': st,
                           'class': cls, 'file': m['file'], 'prompt': naming_prompt,
                           'style': style, 'text': r['text'], 'tokens': r['tokens'],
                           'step_probs': r['step_probs'], 'outcome': outcome,
                           'answer_maxp': r['answer_maxp'],
                           'n_forwards': r['n_forwards'], 'wall_s': r['wall_s'],
                           'layers_selected': r['layers_selected'],
                           'variant': 'faithful_stage7',
                           'abstained': roundA_abstained}
                    append(out, rec)
                    done.add(f"{sid}:{mm}")
                except Exception as e:
                    append(out, {'key': f"{sid}:{mm}", 'model': args.model,
                                 'task': 'naming', 'method': mm, 'stratum': st,
                                 'class': cls, 'file': m['file'],
                                 'error': f'{type(e).__name__}: {e}'})
                    print(f'  [error] {sid}:{mm} {type(e).__name__}: {e}', flush=True)
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t_start
                print(f'[{args.model} {job}] {n_done}/{len(samples)} {el:.0f}s '
                      f'({el / n_done:.2f}s/img)', flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    for job in args.jobs.split(','):
        if job == 'seed':
            run_food('seed', list(SEED_JOBS))
        elif job == 'dola_subset':
            run_food('dola_subset', ['dola_subset'], dola_even=True)
        elif job == 'dogs':
            run_dogs()
        else:
            raise ValueError(job)


if __name__ == '__main__':
    main()
