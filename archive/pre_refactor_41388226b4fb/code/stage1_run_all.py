"""End-to-end orchestrator (task-book entry point).

Stage pipeline (each stage is idempotent and resumable):
  1. build dataset (needs LVIS annotations in data/, downloads COCO images)
  2. main experiments on both models (GPU 4 = q4b, GPU 5 = q9b, per server constraints)
  3. analysis tables + judgments
  4. figures
  5. run manifest

Usage:
  ./venv/bin/python code/stage1_run_all.py --stage all
  ./venv/bin/python code/stage1_run_all.py --stage exp      # resume experiments only
"""
import os, sys, subprocess, argparse, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / 'venv' / 'bin' / 'python')
# GPU 1 was unstable (hardware fault, 2026-09-13); server admin restricted us to GPU 4/5.
GPU_ASSIGN = {'q4b': 4, 'q9b': 5}


def sh(cmd, log=None):
    print('>>', ' '.join(map(str, cmd)), flush=True)
    logf = open(ROOT / 'logs' / log, 'a') if log else None
    r = subprocess.run(list(map(str, cmd)), stdout=logf or subprocess.PIPE,
                       stderr=subprocess.STDOUT if logf else None)
    if logf:
        logf.close()
    if r.returncode != 0:
        raise RuntimeError(f'stage failed: {cmd}')


def stage_data():
    p = ROOT / 'data/dataset.jsonl'
    if p.exists():
        print('dataset exists, skip (delete data/dataset.jsonl to rebuild)')
        return
    sh([PY, ROOT / 'code/stage1_build_dataset.py', '--target', 600], log='build_dataset.log')


def stage_exp(models=('q4b', 'q9b')):
    # stagger starts to smooth host-RAM during weight loading
    procs = {}
    for i, m in enumerate(models):
        if i:
            time.sleep(90)
        log = open(ROOT / 'logs' / f'main_{m}.log', 'a')
        p = subprocess.Popen([PY, ROOT / 'code/stage1_run_experiment.py',
                              '--model', m, '--gpu', GPU_ASSIGN[m],
                              '--tasks', 'naming,existence',
                              '--methods', 'direct,vcd,mib,lcd',
                              '--dataset', 'dataset.jsonl', '--tag', 'main'],
                             stdout=log, stderr=subprocess.STDOUT)
        procs[m] = (p, log)
    for m, (p, log) in procs.items():
        rc = p.wait()
        log.close()
        print(f'[{m}] exit {rc}', flush=True)
        if rc != 0:
            raise RuntimeError(f'{m} failed')


def stage_analysis():
    sh([PY, ROOT / 'code/stage1_analysis.py', '--models', 'q4b,q9b', '--tag', 'main'],
       log='analysis.log')


def stage_figures():
    sh([PY, ROOT / 'code/stage1_make_figures.py'], log='figures.log')


def stage_manifest():
    sh([PY, ROOT / 'code/stage1_make_manifest.py'], log='manifest.log')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['all', 'data', 'exp', 'analysis', 'figures', 'manifest'])
    args = ap.parse_args()
    t0 = time.time()
    if args.stage in ('all', 'data'):
        stage_data()
    if args.stage in ('all', 'exp'):
        stage_exp()
    if args.stage in ('all', 'analysis'):
        stage_analysis()
    if args.stage in ('all', 'figures'):
        stage_figures()
    if args.stage in ('all', 'manifest'):
        stage_manifest()
    print(f'run_all[{args.stage}] finished in {(time.time()-t0)/60:.1f} min')
