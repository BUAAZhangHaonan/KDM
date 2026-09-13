"""Main experiment (v2): naming on the evaluation half for both strata and all
decoding configs + blank-image signals; existence task on an eval subset.

Per sample records carry answer_maxp = probability of the generated first token
under the distribution the config actually used (the config's own confidence).

Usage:
  ./venv/bin/python code/run_experiment.py --model q4b --gpu 4 --tag main
"""
import os, sys, json, time, argparse, random, gc, hashlib
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
METHODS = ['direct', 'vcd', 'mib', 'lcd']
N_EXIST_PER_CLASS = 10


def load_json(p):
    return json.load(open(ROOT / p))


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
    ap.add_argument('--tasks', default='naming,existence')
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--tag', default='main')
    args = ap.parse_args()

    import torch
    from PIL import Image
    from engine import ExpModel, make_variants, jsd, HP
    import scoring, prompts

    em = ExpModel(MODELS[args.model], f'cuda:{args.gpu}')
    classes_all = sorted({m['class'] for m in map(json.loads, open(ROOT / 'data' / 'samples_manifest.jsonl'))})
    strata = load_json(f'data/strata_{args.model}.json')
    stratum_of = {c: 'deficient' for c in strata['deficient']}
    stratum_of.update({c: 'known' for c in strata['known']})

    decision_path = ROOT / 'data' / f'primary_outcome_decision_{args.model}.json'
    style = 'style1'
    if decision_path.exists():
        style = load_json(str(decision_path)).get('chosen_style', 'style1')

    manifest = [json.loads(l) for l in open(ROOT / 'data' / 'samples_manifest.jsonl')]
    tasks = args.tasks.split(',')
    methods = args.methods.split(',')
    rng = random.Random(7)

    t_start = time.time()
    n_done = 0
    total = 0

    # ---------------- naming ----------------
    if 'naming' in tasks:
        out = f'{args.model}_{args.tag}_naming.jsonl'
        done = done_keys(out)
        evals = [m for m in manifest if m['part'] == 'eval' and m['class'] in stratum_of]
        total += len(evals)
        for m in evals:
            img = Image.open(m['image_path']).convert('RGB')
            v = make_variants(img)
            cls, st = m['class'], stratum_of[m['class']]
            naming_prompt = prompts.STYLES[style][-1]  # style3 uses STYLE1 as round B
            sid = f"nm:{m['file']}"
            # style3: round A decides abstention for ALL configs equally (recorded once)
            roundA = None
            if style == 'style3':
                needA = any(f"{sid}:ra:{mm}" not in done for mm in methods)
                if needA:
                    ra = em.decode_single(em.build(v['clean'], prompts.STYLE3_ROUND_A), 4)
                    na = scoring._norm(ra['text'])
                    roundA = (bool(na.startswith('no')) and not na.startswith('no_idea'), ra['text'])
                    append(out, {'key': f"{sid}:ra", 'model': args.model, 'task': 'naming',
                                 'method': 'roundA', 'stratum': st, 'class': cls,
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
                             'method': mm, 'stratum': st, 'class': cls, 'file': m['file'],
                             'prompt': naming_prompt, 'style': style, 'text': r['text'],
                             'outcome': outcome, 'answer_maxp': r['answer_maxp'],
                             'first_entropy': r['first_entropy'], 'first_maxp': r['first_maxp'],
                             'signals': signals if mm == 'direct' else {},
                             'n_forwards': r['n_forwards'], 'wall_s': round(r['wall_s'], 3)})
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t_start
                print(f'[{args.model} naming] {n_done}/{total} {el:.0f}s ({el/n_done:.2f}s/img)', flush=True)
            gc.collect(); torch.cuda.empty_cache()

    # ---------------- existence ----------------
    if 'existence' in tasks:
        out = f'{args.model}_{args.tag}_existence.jsonl'
        done = done_keys(out)
        by_class = defaultdict(list)
        for m in manifest:
            if m['part'] == 'eval' and m['class'] in stratum_of:
                by_class[m['class']].append(m)
        subs = {}
        for c, ms in by_class.items():
            rng.shuffle(ms)
            subs[c] = ms[:N_EXIST_PER_CLASS]
        total2 = sum(len(v) for v in subs.values())
        n2 = 0
        for c, ms in subs.items():
            st = stratum_of[c]
            same = sorted(x for x in (strata['deficient'] if st == 'deficient' else strata['known']) if x != c)
            for m in ms:
                img = Image.open(m['image_path']).convert('RGB')
                v = make_variants(img)
                neg_cls = same[int(hashlib.sha256(m['file'].encode()).hexdigest(), 16) % len(same)]
                for qkind, qname, gold in [('pos', c, True), ('neg', neg_cls, False)]:
                    sid = f"ex:{m['file']}:{qkind}"
                    prompt = prompts.existence_prompt(qname)
                    need = [mm for mm in methods if f"{sid}:{mm}" not in done]
                    for mm in need:
                        if mm == 'direct':
                            r = em.decode_single(em.build(v['clean'], prompt), HP['max_new_tokens_exist'])
                        elif mm == 'lcd':
                            r = em.decode_single(em.build(v['clean'], prompt), HP['max_new_tokens_exist'], lcd=True)
                        elif mm == 'vcd':
                            r = em.decode_contrastive(em.build(v['clean'], prompt),
                                                      em.build(v['noise'], prompt), 'vcd', HP['max_new_tokens_exist'])
                        elif mm == 'mib':
                            r = em.decode_contrastive(em.build(v['clean'], prompt),
                                                      em.build(v['blur'], prompt), 'mib', HP['max_new_tokens_exist'])
                        outcome = scoring.score_existence(r['text'], gold)
                        append(out, {'key': f"{sid}:{mm}", 'model': args.model, 'task': 'existence',
                                     'method': mm, 'stratum': st, 'class': c, 'file': m['file'],
                                     'question_kind': qkind, 'query_class': qname, 'gold_present': gold,
                                     'prompt': prompt, 'text': r['text'], 'outcome': outcome,
                                     'answer_maxp': r['answer_maxp'], 'first_maxp': r['first_maxp'],
                                     'n_forwards': r['n_forwards'], 'wall_s': round(r['wall_s'], 3)})
                    gc.collect()
                n2 += 1
                if n2 % 50 == 0:
                    print(f'[{args.model} existence] {n2}/{total2}', flush=True)
        print(f'[{args.model}] DONE naming+existence in {(time.time()-t_start)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
