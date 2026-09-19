"""Stage-7: M3ID context-pressure gate inactivity ratio (PREREGISTER sec 2).

Replays the CONDITIONAL branch of the faithful M3ID decoding path for every
recorded sample (tokens already stored in stage-6 records; deterministic greedy
replay, single branch, no regeneration) and reports, per model x stratum, the
share of decoding steps where the correction was NOT applied because the
conditional top probability reached the 0.3 context-pressure threshold.

Usage: ./venv/bin/python code/stage7_m3idgate.py --model q4b --gpu 4
"""
import os, sys, json, argparse, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ['HF_HOME'] = str(ROOT / 'cache' / 'hf')
os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache' / 'xdg')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache' / 'mpl')
sys.path.insert(0, str(ROOT / 'code'))

MODELS = {'q4b': '/home/g203-4028/Models/Qwen3.5-4B',
          'q9b': '/home/g203-4028/Models/Qwen3.5-9B'}
TAU = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--gpu', type=int, required=True)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from stage6_engine import S6Model
    import prompts

    em = S6Model(MODELS[args.model], f'cuda:{args.gpu}')
    man = {m['file']: m for m in
           (json.loads(l) for l in open(ROOT / 'data' / 'samples_manifest.jsonl'))}
    recs = {}
    for line in open(ROOT / 'outputs' / 'raw' / f'{args.model}_s6_eval_naming.jsonl'):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('method') == 'm3id' and 'error' not in r and r.get('tokens'):
            recs[r['file']] = r
    naming_prompt = prompts.STYLE1
    per_cell = {}
    t0 = time.time()
    for i, (f, r) in enumerate(sorted(recs.items())):
        m = man[f]
        inputs = em.build(Image.open(m['image_path']).convert('RGB'), naming_prompt)
        toks = r['tokens']
        with torch.no_grad():
            out = em._fwd(inputs=inputs)
            closed = 0
            steps = 0
            for t in range(min(len(toks), 12)):
                p_max = float(F.softmax(out.logits[:, -1, :].float(), -1).max())
                steps += 1
                closed += p_max >= TAU
                if t + 1 >= len(toks):
                    break
                out = em._fwd(tok=toks[t], pkv=out.past_key_values)
        cell = per_cell.setdefault(r['stratum'], {'steps': 0, 'closed': 0, 'n': 0})
        cell['steps'] += steps
        cell['closed'] += closed
        cell['n'] += 1
        if (i + 1) % 100 == 0:
            print(f'[{args.model}] {i + 1}/{len(recs)} {(time.time() - t0) / 60:.1f} min',
                  flush=True)
    import csv
    outp = ROOT / 'outputs' / 'tables' / f'stage7_m3idgate_{args.model}.csv'
    with open(outp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['model', 'stratum', 'n_samples',
                                           'n_steps', 'gate_closed_ratio'])
        w.writeheader()
        for st, c in sorted(per_cell.items()):
            w.writerow({'model': args.model, 'stratum': st, 'n_samples': c['n'],
                        'n_steps': c['steps'],
                        'gate_closed_ratio': round(c['closed'] / c['steps'], 4)})
            print(args.model, st, 'closed ratio',
                  round(c['closed'] / c['steps'], 4), f"({c['n']} samples)")


if __name__ == '__main__':
    main()
