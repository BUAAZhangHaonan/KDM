"""v2 run_manifest.json: environment, versions, data fingerprints, GPUs, timings."""
import os, sys, json, hashlib, subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent


def sha256(path, n=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    import torch, transformers
    man = {'created': datetime.now().isoformat(timespec='seconds'),
           'python': sys.version.split()[0], 'torch': torch.__version__,
           'transformers': transformers.__version__}
    try:
        import datasets, accelerate, PIL, scipy, nltk, numpy
        man.update({'datasets': datasets.__version__, 'accelerate': accelerate.__version__,
                    'pillow': PIL.__version__, 'scipy': scipy.__version__,
                    'nltk': nltk.__version__, 'numpy': numpy.__version__})
    except Exception as e:
        man['lib_error'] = str(e)
    try:
        import fla
        man['flash_linear_attention'] = fla.__version__
    except Exception:
        man['flash_linear_attention'] = 'not installed'
    man['cuda'] = torch.version.cuda
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total',
                                        '--format=csv,noheader'], text=True)
        man['gpus'] = [l.strip() for l in out.strip().split('\n')]
    except Exception as e:
        man['gpus'] = f'nvidia-smi failed: {e}'
    man['gpu_assignment'] = {'q4b': 'cuda:4', 'q9b': 'cuda:5',
                             'note': 'server admin restriction: GPUs 4/5 only'}
    man['models'] = {'q4b': '/home/g203-4028/Models/Qwen3.5-4B',
                     'q9b': '/home/g203-4028/Models/Qwen3.5-9B'}
    man['data'] = {'source': 'ethz/food101 (HF datasets)',
                   'samples_manifest_sha256': sha256(ROOT / 'data/samples_manifest.jsonl'),
                   'n_manifest': sum(1 for _ in open(ROOT / 'data/samples_manifest.jsonl'))}
    dr = json.load(open(ROOT / 'data/data_report.json'))
    man['data'].update(dr)
    for m in ['q4b', 'q9b']:
        p = ROOT / 'data' / f'strata_{m}.json'
        if p.exists():
            st = json.load(open(p))
            man.setdefault('strata', {})[m] = {
                'rule': st['rule'], 'k': st['k'],
                'deficient_mean_group_acc': round(sum(st['group_acc'][c] for c in st['deficient']) / st['k'], 4),
                'known_mean_group_acc': round(sum(st['group_acc'][c] for c in st['known']) / st['k'], 4)}
    raw = {}
    for f in sorted((ROOT / 'outputs/raw').glob('*.jsonl')):
        recs = [json.loads(l) for l in open(f) if l.strip()]
        raw[f.name] = {'n': len(recs),
                       'sum_wall_s': round(sum(r.get('wall_s', 0) for r in recs), 1)}
    man['raw_outputs'] = raw
    sv = ROOT / 'outputs/tables/strata_validation.csv'
    if sv.exists():
        man['strata_validation_pass'] = open(sv).read().splitlines()[1:]
    json.dump(man, open(ROOT / 'run_manifest.json', 'w'), indent=1, ensure_ascii=False)
    print('run_manifest.json written')


if __name__ == '__main__':
    main()
