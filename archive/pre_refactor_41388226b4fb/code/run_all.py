"""v2 orchestrator entry point.

Stages (idempotent / resumable):
  data      : download ethz/food101, export split manifest + images + contact sheet
  phase1    : per-model grouping -> strata -> abstention pre-test -> eval-half
              validation (GPU 4 = q4b, GPU 5 = q9b)
  validate  : phase-1 analysis (strata_validation.csv + outcome decision)
  exp       : main experiments (both models parallel)
  analysis  : tables for judgments 1-3
  figures / manifest

Usage: ./venv/bin/python code/run_all.py --stage all
"""
import os, sys, subprocess, argparse, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / 'venv' / 'bin' / 'python')
# Server admin restriction (2026-09-13): use only GPU 4 and GPU 5.
GPU = {'q4b': 4, 'q9b': 5}


def sh(cmd, log=None):
    print('>>', ' '.join(map(str, cmd)), flush=True)
    if log:
        with open(ROOT / 'logs' / log, 'a') as lf:
            r = subprocess.run(list(map(str, cmd)), stdout=lf, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(list(map(str, cmd)))
    if r.returncode != 0:
        raise RuntimeError(f'stage failed: {cmd}')


def stage_data():
    if (ROOT / 'data/samples_manifest.jsonl').exists():
        print('manifest exists, skip')
        return
    sh([PY, ROOT / 'code/prepare_data.py'], 'prepare_data.log')


def stage_phase1(models=('q4b', 'q9b')):
    procs = {}
    for i, m in enumerate(models):
        if i:
            time.sleep(90)
        lf = open(ROOT / 'logs' / f'phase1_{m}.log', 'a')
        p = subprocess.Popen([PY, ROOT / 'code/phase1.py', '--model', m, '--gpu', GPU[m]],
                             stdout=lf, stderr=subprocess.STDOUT)
        procs[m] = (p, lf)
    for m, (p, lf) in procs.items():
        rc = p.wait(); lf.close()
        print(f'[phase1 {m}] exit {rc}', flush=True)
        if rc != 0:
            raise RuntimeError(f'phase1 {m} failed')


def stage_validate(models='q4b,q9b'):
    sh([PY, ROOT / 'code/phase1_analysis.py', models], 'phase1_analysis.log')


def stage_exp(models=('q4b', 'q9b')):
    procs = {}
    for i, m in enumerate(models):
        if i:
            time.sleep(90)
        lf = open(ROOT / 'logs' / f'main_{m}.log', 'a')
        p = subprocess.Popen([PY, ROOT / 'code/run_experiment.py', '--model', m,
                              '--gpu', GPU[m], '--tag', 'main'],
                             stdout=lf, stderr=subprocess.STDOUT)
        procs[m] = (p, lf)
    for m, (p, lf) in procs.items():
        rc = p.wait(); lf.close()
        print(f'[exp {m}] exit {rc}', flush=True)
        if rc != 0:
            raise RuntimeError(f'exp {m} failed')


def stage_analysis(models='q4b,q9b'):
    sh([PY, ROOT / 'code/analysis.py', '--models', models, '--tag', 'main'], 'analysis.log')


def stage_figures():
    sh([PY, ROOT / 'code/make_figures.py'], 'figures.log')


def stage_manifest():
    sh([PY, ROOT / 'code/make_manifest.py'], 'manifest.log')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['all', 'data', 'phase1', 'validate', 'exp', 'analysis', 'figures', 'manifest'])
    args = ap.parse_args()
    t0 = time.time()
    if args.stage in ('all', 'data'):
        stage_data()
    if args.stage in ('all', 'phase1'):
        stage_phase1()
    if args.stage in ('all', 'validate'):
        stage_validate()
    if args.stage in ('all', 'exp'):
        stage_exp()
    if args.stage in ('all', 'analysis'):
        stage_analysis()
    if args.stage in ('all', 'figures'):
        stage_figures()
    if args.stage in ('all', 'manifest'):
        stage_manifest()
    print(f'run_all[{args.stage}] done in {(time.time()-t0)/60:.1f} min')
