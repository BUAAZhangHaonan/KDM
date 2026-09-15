"""Write run_manifest.json: environment, versions, data fingerprints, GPUs, timings."""
import os, sys, json, time, hashlib, subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('HF_HOME', str(ROOT / 'cache' / 'hf'))


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
    man = {}
    man['created'] = datetime.now().isoformat(timespec='seconds')
    man['python'] = sys.version.split()[0]
    man['torch'] = torch.__version__
    man['transformers'] = transformers.__version__
    try:
        import accelerate, PIL, scipy, nltk, numpy
        man['accelerate'] = accelerate.__version__
        man['pillow'] = PIL.__version__
        man['scipy'] = scipy.__version__
        man['nltk'] = nltk.__version__
        man['numpy'] = numpy.__version__
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
    man['gpu_assignment'] = {'q4b': 'cuda:4 (Qwen3.5-4B)', 'q9b': 'cuda:5 (Qwen3.5-9B)'}
    man['model_paths'] = {'q4b': '/home/g203-4028/Models/Qwen3.5-4B',
                          'q9b': '/home/g203-4028/Models/Qwen3.5-9B'}
    man['data'] = {
        'lvis_val_zip': {'path': 'data/lvis_v1_val.json.zip', 'sha256': sha256(ROOT / 'data/lvis_v1_val.json.zip'),
                         'size': (ROOT / 'data/lvis_v1_val.json.zip').stat().st_size},
        'lvis_train_zip': {'path': 'data/lvis_v1_train.json.zip',
                           'sha256': sha256(ROOT / 'data/lvis_v1_train.json.zip'),
                           'size': (ROOT / 'data/lvis_v1_train.json.zip').stat().st_size},
        'dataset': {'path': 'data/dataset.jsonl', 'sha256': sha256(ROOT / 'data/dataset.jsonl'),
                    'n_samples': sum(1 for _ in open(ROOT / 'data/dataset.jsonl'))},
    }
    dsr = json.load(open(ROOT / 'data/dataset_report.json'))
    man['dataset_report'] = dsr
    # raw output fingerprints + timings
    raw = {}
    for f in sorted((ROOT / 'outputs/raw').glob('*_main_*.jsonl')):
        recs = [json.loads(l) for l in open(f) if l.strip()]
        raw[f.name] = {'n': len(recs),
                       'sum_wall_s': round(sum(r.get('wall_s', 0) for r in recs), 1),
                       'sum_forwards': int(sum(r.get('n_forwards', 0) for r in recs))}
    man['raw_outputs'] = raw
    for log, key in [('logs/main_q4b.log', 'q4b'), ('logs/main_q9b.log', 'q9b')]:
        p = ROOT / log
        if p.exists():
            lines = p.read_text().strip().split('\n')
            last = [l for l in lines if 'DONE' in l or 'elapsed' in l]
            man.setdefault('run_log_tail', {})[key] = last[-1] if last else lines[-1]
    json.dump(man, open(ROOT / 'run_manifest.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(man, indent=1)[:2000])


if __name__ == '__main__':
    main()
