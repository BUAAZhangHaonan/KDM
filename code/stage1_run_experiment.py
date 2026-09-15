"""Experiment runner: for one model on one GPU, runs all decoding configs over the
dataset for the two tasks, writing per-sample raw outputs (resumable).

Usage:
  ./venv/bin/python code/stage1_run_experiment.py --model q4b --gpu 1 --tasks naming,existence --tag main
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

MODELS = {
    'q4b': '/home/g203-4028/Models/Qwen3.5-4B',
    'q9b': '/home/g203-4028/Models/Qwen3.5-9B',
}

NAMING_PROMPT = ("Look at the image. Name the main object shown in the image. "
                 "Reply with a single English noun (one word or a short compound). "
                 "If you do not recognize the object or are not sure, reply exactly: UNKNOWN")


def _display_name(cat_name):
    import re
    return re.sub(r'\([^)]*\)', '', cat_name).replace('_', ' ').strip()


def existence_prompt(cat_name):
    nice = _display_name(cat_name)
    return (f"Look at the image. Is there a {nice} in the image? "
            "Answer with exactly one word: Yes or No.")


def load_dataset(limit=None, name='dataset.jsonl'):
    rows = [json.loads(l) for l in open(ROOT / 'data' / name)]
    if limit:
        # take first `limit` per tier for balanced pilots
        sel, cnt = [], {'high': 0, 'low': 0}
        for r in rows:
            if cnt[r['tier']] < limit:
                sel.append(r); cnt[r['tier']] += 1
        rows = sel
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--tasks', default='naming,existence')
    ap.add_argument('--methods', default='direct,vcd,mib,lcd')
    ap.add_argument('--limit', type=int, default=None, help='per-tier sample limit (pilot)')
    ap.add_argument('--tag', default='main')
    ap.add_argument('--dataset', default='dataset.jsonl')
    args = ap.parse_args()

    import torch
    from PIL import Image
    from engine import ExpModel, make_variants, jsd, HP
    import scoring

    device = f'cuda:{args.gpu}'
    em = ExpModel(MODELS[args.model], device)
    rows = load_dataset(args.limit, args.dataset)
    tasks = args.tasks.split(',')
    methods = args.methods.split(',')
    print(f'[{args.model}] {len(rows)} samples, tasks={tasks}, methods={methods}', flush=True)

    raw_dir = ROOT / 'outputs' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_files = {}
    done = set()
    for t in tasks:
        f = raw_dir / f'{args.model}_{args.tag}_{t}.jsonl'
        out_files[t] = open(f, 'a')
        if f.exists():
            for line in open(f):
                try:
                    r = json.loads(line)
                    done.add((r['sample_id'], r['method']))
                except Exception:
                    pass

    t_start = time.time()
    n_done = 0
    for idx, s in enumerate(rows):
        img = Image.open(s['image_path']).convert('RGB')
        variants = make_variants(img)
        tier = s['tier']

        # ---------------- task 1: naming ----------------
        if 'naming' in tasks:
            sid = f"nm:{s['image_id']}"
            need = [m for m in methods if (sid, m) not in done]
            signals = {}
            if need:
                # blank-condition first-token distribution for signal 2
                blank_in = em.build(variants['blank'], NAMING_PROMPT)
                blank_run = em.decode_single(blank_in, max_new_tokens=1)
                for m in need:
                    if m == 'direct':
                        r = em.decode_single(em.build(variants['clean'], NAMING_PROMPT),
                                             HP['max_new_tokens_naming'])
                        signals = {'entropy': r['first_entropy'], 'maxp': r['first_maxp'],
                                   'jsd': jsd(r['first_probs'], blank_run['first_probs'])}
                        outcome = scoring.score_naming(r['text'], s['name'], s['synonyms'], s['synset'])
                    elif m == 'lcd':
                        r = em.decode_single(em.build(variants['clean'], NAMING_PROMPT),
                                             HP['max_new_tokens_naming'], lcd=True)
                        outcome = scoring.score_naming(r['text'], s['name'], s['synonyms'], s['synset'])
                    elif m == 'vcd':
                        r = em.decode_contrastive(em.build(variants['clean'], NAMING_PROMPT),
                                                  em.build(variants['noise'], NAMING_PROMPT),
                                                  'vcd', HP['max_new_tokens_naming'])
                        outcome = scoring.score_naming(r['text'], s['name'], s['synonyms'], s['synset'])
                    elif m == 'mib':
                        r = em.decode_contrastive(em.build(variants['clean'], NAMING_PROMPT),
                                                  em.build(variants['blur'], NAMING_PROMPT),
                                                  'mib', HP['max_new_tokens_naming'])
                        outcome = scoring.score_naming(r['text'], s['name'], s['synonyms'], s['synset'])
                    rec = {'model': args.model, 'task': 'naming', 'method': m, 'tier': tier,
                           'sample_id': sid, 'image_id': s['image_id'], 'source': s['source'],
                           'gold': s['name'], 'synset': s['synset'], 'prompt': NAMING_PROMPT,
                           'text': r['text'], 'outcome': outcome,
                           'first_entropy': r['first_entropy'], 'first_maxp': r['first_maxp'],
                           'n_forwards': r['n_forwards'], 'n_prefill_tokens': r['n_prefill_tokens'],
                           'wall_s': round(r['wall_s'], 3), 'signals': signals}
                    out_files['naming'].write(json.dumps(rec) + '\n')
                    out_files['naming'].flush()
                del blank_in, blank_run
            gc.collect(); torch.cuda.empty_cache()

        # ---------------- task 2: existence ----------------
        if 'existence' in tasks:
            questions = [('pos', s['name'], True)]
            for neg in s.get('negatives', [])[:1]:
                questions.append((('neg', neg['name'], False)))
            for qkind, qname, gold in questions:
                sid = f"ex:{s['image_id']}:{qkind}:{qname}"
                prompt = existence_prompt(qname)
                need = [m for m in methods if (sid, m) not in done]
                for m in need:
                    if m == 'direct':
                        r = em.decode_single(em.build(variants['clean'], prompt), HP['max_new_tokens_exist'])
                    elif m == 'lcd':
                        r = em.decode_single(em.build(variants['clean'], prompt), HP['max_new_tokens_exist'], lcd=True)
                    elif m == 'vcd':
                        r = em.decode_contrastive(em.build(variants['clean'], prompt),
                                                  em.build(variants['noise'], prompt), 'vcd', HP['max_new_tokens_exist'])
                    elif m == 'mib':
                        r = em.decode_contrastive(em.build(variants['clean'], prompt),
                                                  em.build(variants['blur'], prompt), 'mib', HP['max_new_tokens_exist'])
                    outcome = scoring.score_existence(r['text'], gold)
                    rec = {'model': args.model, 'task': 'existence', 'method': m, 'tier': tier,
                           'sample_id': sid, 'image_id': s['image_id'], 'source': s['source'],
                           'gold': qname, 'question_kind': qkind, 'gold_present': gold,
                           'prompt': prompt, 'text': r['text'], 'outcome': outcome,
                           'first_entropy': r['first_entropy'], 'first_maxp': r['first_maxp'],
                           'n_forwards': r['n_forwards'], 'n_prefill_tokens': r['n_prefill_tokens'],
                           'wall_s': round(r['wall_s'], 3)}
                    out_files['existence'].write(json.dumps(rec) + '\n')
                    out_files['existence'].flush()
                gc.collect(); torch.cuda.empty_cache()

        n_done += 1
        if n_done % 10 == 0:
            el = time.time() - t_start
            print(f'[{args.model}] {n_done}/{len(rows)} samples, {el:.0f}s elapsed, {el/n_done:.1f}s/sample', flush=True)

    for f in out_files.values():
        f.close()
    print(f'[{args.model}] DONE in {(time.time()-t_start)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
