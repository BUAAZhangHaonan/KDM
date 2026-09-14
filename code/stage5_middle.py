"""Stage-5 middle-accuracy class run (food101): fill the base-rate curve's middle.

Replicates run_experiment.py's naming loop verbatim (same engine, prompts,
scoring, style, random processes); only the class set is replaced by classes
that are in neither the deficient nor the known stratum of this model's own
strata file, and records carry stratum='middle'.

Usage:
  ./venv/bin/python code/stage5_middle.py --model q4b --gpu 4
"""
import os, sys, json, time, argparse, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ['HF_HOME'] = str(ROOT / 'cache' / 'hf')
os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache' / 'xdg')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache' / 'mpl')
os.environ['NLTK_DATA'] = str(ROOT / 'cache' / 'nltk')
sys.path.insert(0, str(ROOT / 'code'))

MODELS = {'q4b': '/home/g203-4028/Models/Qwen3.5-4B',
          'q9b': '/home/g203-4028/Models/Qwen3.5-9B'}
METHODS = ['direct', 'vcd', 'mib', 'lcd']


def done_keys(path):
    keys = set()
    p = ROOT / 'outputs' / 'raw' / path
    if p.exists():
        for line in open(p):
            try:
                keys.add((json.loads(line)['key']))
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
    ap.add_argument('--methods', default=','.join(METHODS))
    args = ap.parse_args()

    import torch
    from PIL import Image
    from engine import ExpModel, make_variants, jsd, HP
    import scoring, prompts

    em = ExpModel(MODELS[args.model], f'cuda:{args.gpu}')
    manifest = [json.loads(l) for l in open(ROOT / 'data' / 'samples_manifest.jsonl')]
    classes_all = sorted({m['class'] for m in manifest})
    strata = json.load(open(ROOT / 'data' / f'strata_{args.model}.json'))
    excluded = set(strata['deficient']) | set(strata['known'])
    middle = [c for c in classes_all if c not in excluded]
    print(f'[{args.model}] middle classes: {len(middle)}', flush=True)

    decision_path = ROOT / 'data' / f'primary_outcome_decision_{args.model}.json'
    style = 'style1'
    if decision_path.exists():
        style = json.load(open(decision_path)).get('chosen_style', 'style1')

    methods = args.methods.split(',')
    out = f'{args.model}_s5middle_naming.jsonl'
    done = done_keys(out)
    evals = [m for m in manifest if m['part'] == 'eval' and m['class'] in set(middle)]
    t_start = time.time()
    n_done = 0

    for m in evals:
        img = Image.open(m['image_path']).convert('RGB')
        v = make_variants(img)
        cls = m['class']
        naming_prompt = prompts.STYLES[style][-1]
        sid = f"nm:{m['file']}"
        roundA = None
        if style == 'style3':
            needA = any(f"{sid}:ra:{mm}" not in done for mm in methods)
            if needA:
                ra = em.decode_single(em.build(v['clean'], prompts.STYLE3_ROUND_A), 4)
                na = scoring._norm(ra['text'])
                roundA = (bool(na.startswith('no')) and not na.startswith('no_idea'), ra['text'])
                append(out, {'key': f"{sid}:ra", 'model': args.model, 'task': 'naming',
                             'method': 'roundA', 'stratum': 'middle', 'class': cls,
                             'file': m['file'], 'text': ra['text'],
                             'abstained': roundA[0]})
        need = [mm for mm in methods if f"{sid}:{mm}" not in done]
        signals = {}
        if need:
            blank = em.decode_single(em.build(v['blank'], naming_prompt), 1)
        for mm in need:
            if mm == 'direct':
                r = em.decode_single(em.build(v['clean'], naming_prompt), HP['max_new_tokens_naming'])
                signals = {'entropy': r['first_entropy'], 'maxp': r['first_maxp'],
                           'jsd': jsd(r['first_probs'], blank['first_probs'])}
            elif mm == 'lcd':
                r = em.decode_single(em.build(v['clean'], naming_prompt), HP['max_new_tokens_naming'], lcd=True)
            elif mm == 'vcd':
                r = em.decode_contrastive(em.build(v['clean'], naming_prompt),
                                          em.build(v['noise'], naming_prompt), 'vcd', HP['max_new_tokens_naming'])
            elif mm == 'mib':
                r = em.decode_contrastive(em.build(v['clean'], naming_prompt),
                                          em.build(v['blur'], naming_prompt), 'mib', HP['max_new_tokens_naming'])
            outcome = scoring.score_naming(r['text'], cls, classes_all)
            append(out, {'key': f"{sid}:{mm}", 'model': args.model, 'task': 'naming',
                         'method': mm, 'stratum': 'middle', 'class': cls, 'file': m['file'],
                         'prompt': naming_prompt, 'style': style, 'text': r['text'],
                         'outcome': outcome, 'answer_maxp': r['answer_maxp'],
                         'first_entropy': r['first_entropy'], 'first_maxp': r['first_maxp'],
                         'signals': signals if mm == 'direct' else {},
                         'n_forwards': r['n_forwards'], 'wall_s': round(r['wall_s'], 3)})
        n_done += 1
        if n_done % 25 == 0:
            el = time.time() - t_start
            print(f'[{args.model} middle] {n_done}/{len(evals)} {el:.0f}s ({el/n_done:.2f}s/img)', flush=True)
        gc.collect(); torch.cuda.empty_cache()

    print(f'[{args.model} middle] DONE in {(time.time()-t_start)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
