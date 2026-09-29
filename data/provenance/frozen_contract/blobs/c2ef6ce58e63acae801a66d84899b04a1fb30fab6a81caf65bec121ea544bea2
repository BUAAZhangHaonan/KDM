#!/usr/bin/env python3
"""Fixed full-panel census orchestration; no retries, substitutions or sample cuts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kdm.io import atomic_json, file_hash, read_jsonl, within
from kdm.protocol import validate_freeze
from kdm.execution import local_host, read_registry

LANES = {
    'remote_gpu1': {'host':'6403', 'cards':[1], 'models':['qwen3vl','glm46v']},
    'gpu0': {'host':'4028', 'cards': [0], 'models': ['qwen35_9b']},
    'gpu1': {'host':'4028', 'cards': [1], 'models': ['qwen35_4b', 'qwen25vl']},
    'pair45': {'host':'4028', 'cards': [4, 5], 'models': ['gemma3_12b', 'llava15_13b', 'internvl35_8b']},
    'gpu4_after_pairs': {'host':'4028', 'cards': [4], 'models': ['gemma3_4b', 'llava15_7b', 'onevision', 'llava16_mistral']},
    'gpu5_after_pairs': {'host':'4028', 'cards': [5], 'models': ['minicpm26', 'minicpm45', 'phi35', 'llava16_vicuna']},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def plan():
    panel = json.loads((ROOT / 'configs/kdm/models.json').read_text())
    keys = [key for lane in LANES.values() for key in lane['models']]
    if len(panel) != 16 or len({row['key'] for row in panel}) != 16 or len(keys) != 16 or len(set(keys)) != 16 or set(keys) != {row['key'] for row in panel}:
        raise ValueError('Schedule must contain the complete fixed 16-model panel exactly once')
    samples = list(read_jsonl(ROOT / 'data/current/all.jsonl'))
    if len(samples) != 9167 or len({row['id'] for row in samples}) != 9167:
        raise ValueError('Full original manifest required')
    registry=read_registry(ROOT)
    if set(registry["model_hosts"])!=set(keys):raise ValueError("Host registry differs from the fixed panel")
    specs = {}
    for lane in LANES.values():
        for key in lane['models']:
            if registry['model_hosts'][key]!=lane['host'] or not set(lane['cards'])<=set(registry['hosts'][lane['host']]['allowed_gpus']):
                raise ValueError('Schedule differs from registered host/model/GPU allocation')
            spec = json.loads((ROOT / f'configs/runtime/{key}.json').read_text())
            if spec.get('key') != key or spec.get('gpu_count') != len(lane['cards']):
                raise ValueError('Spec and physical GPU allocation differ: ' + key)
            specs[key] = spec
    return {'lanes': LANES, 'samples': len(samples), 'expected_per_model': 2 * len(samples),
            'expected_panel_responses': 2 * len(samples) * len(keys),
            'scope': 'Full guided/unguided direct census only; no selection or intervention'}, specs


def execute(run_dir, host):
    local_host(ROOT, host)
    schedule, specs = plan()
    active={name:lane for name,lane in LANES.items() if lane['host']==host}
    specs={key:specs[key] for lane in active.values() for key in lane['models']}
    if not specs:raise ValueError('No fixed models assigned to this host')
    freeze_path = ROOT / 'outputs/records/preregistration_freeze.json'
    freeze = validate_freeze(ROOT)
    sources = freeze['source_blobs']
    target = within(ROOT, run_dir)
    if not target.is_relative_to(ROOT / 'outputs/records'):
        raise ValueError('Launch receipt must be under outputs/records')
    target.mkdir(parents=True, exist_ok=False)
    with (ROOT / 'outputs/locks/census_panel.lock').open('a') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop = threading.Event()
        mutex = threading.Lock()
        dispatch_lock = threading.Lock()
        report = {'started_utc': now(), 'scheduler_pid': os.getpid(), 'schedule': schedule,
                  'freeze_receipt_sha256': file_hash(freeze_path), 'source_blobs': sources,
                  'host':host,'jobs': {key: {'status': 'not_started'} for key in specs}, 'host_complete':False,'complete': False,
                  'completion_scope':'host subset only; use --verify-panel after syncing all sixteen original ledgers and sidecars'}

        def update(key, **fields):
            with mutex:
                report['jobs'][key].update(fields)
                atomic_json(target / 'status.json', report)

        def lane(name):
            cards = LANES[name]['cards']
            for key in LANES[name]['models']:
                if stop.is_set():
                    return
                spec = specs[key]
                output = f'outputs/raw/current/{key}/census.jsonl'
                command = ['bash', str(ROOT / 'scripts/worker.sh'), str(ROOT), ','.join(map(str, cards)),
                           spec['environment_python'], '-m', 'kdm.cli', '--root', str(ROOT), 'run',
                           '--mode', 'census', '--manifest', 'data/current/all.jsonl', '--model', key,
                           '--model-spec', f'configs/runtime/{key}.json', '--gpu', str(cards[0]), '--out', output]
                try:
                    with (target / (key + '.log')).open('x') as log:
                        # Failure publication and process admission share one lock.
                        # A lane that checked stop earlier cannot launch after a known failure.
                        with dispatch_lock:
                            if stop.is_set():return
                            update(key, status='starting', lane=name, physical_gpus=cards, command=command, started_utc=now())
                            try:
                                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                            except Exception:
                                stop.set()
                                raise
                            update(key, status='running', worker_pid=process.pid)
                        code = process.wait()
                        if code:
                            with dispatch_lock:stop.set()
                        update(key, process_exit_code=code, generation_finished_utc=now())
                        if code:
                            raise RuntimeError('Generation process failed: ' + str(code))
                        check = [sys.executable, str(ROOT / 'scripts/verify_complete.py'), '--root', str(ROOT),
                                 '--manifest', 'data/current/all.jsonl', '--records', output, '--model', key,
                                 '--mode', 'census', '--out', str(target / (key + '_complete.json'))]
                        verify_code = subprocess.run(check, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
                        if verify_code:
                            with dispatch_lock:stop.set()
                        update(key, verification_exit_code=verify_code)
                        if verify_code:
                            raise RuntimeError('Full census coverage verification failed')
                    update(key, status='complete', finished_utc=now())
                except Exception as exc:
                    with dispatch_lock:stop.set()
                    update(key, status='failed', error=type(exc).__name__ + ': ' + str(exc), finished_utc=now())
                    return

        def pair_then_singles():
            lane('pair45')
            if not stop.is_set():
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(lane, ['gpu4_after_pairs', 'gpu5_after_pairs']))

        atomic_json(target / 'status.json', report)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = ([pool.submit(lane, 'gpu0'), pool.submit(lane, 'gpu1'), pool.submit(pair_then_singles)]
                       if host=='4028' else [pool.submit(lane,'remote_gpu1')])
            for future in futures:
                future.result()
        report['finished_utc'] = now()
        report['host_complete'] = all(job['status'] == 'complete' for job in report['jobs'].values())
        atomic_json(target / 'status.json', report)
        return 0 if report['host_complete'] else 1


def verify_panel(run_dir):
    """Only a synchronized, provenance-checked sixteen-model collection is complete."""
    from kdm.provenance import validate_census_inputs
    schedule,specs=plan()
    freeze=validate_freeze(ROOT)
    paths=[ROOT/f'outputs/raw/current/{key}/census.jsonl' for key in specs]
    receipt=validate_census_inputs(ROOT,paths,ROOT/'data/current/all.jsonl',freeze,require_complete_panel=True)
    target=within(ROOT,run_dir)
    if not target.is_relative_to(ROOT/'outputs/records'):raise ValueError('Panel receipt must be under outputs/records')
    target.mkdir(parents=True,exist_ok=False)
    atomic_json(target/'panel_complete.json',{'complete':True,'finished_utc':now(),'schedule':schedule,'provenance':receipt})
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--run-dir')
    parser.add_argument('--host',choices=['4028','6403'])
    parser.add_argument('--verify-panel',action='store_true')
    args = parser.parse_args()
    if args.verify_panel:
        if args.execute or not args.run_dir:parser.error('--verify-panel requires --run-dir and excludes --execute')
        raise SystemExit(verify_panel(args.run_dir))
    if args.execute:
        if not args.run_dir or not args.host:
            parser.error('--execute requires --host and a new --run-dir')
        raise SystemExit(execute(args.run_dir,args.host))
    print(json.dumps(plan()[0], ensure_ascii=False, indent=2))
