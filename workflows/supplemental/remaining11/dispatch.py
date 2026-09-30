"""Start explicit supplemental claims once and record the real worker PID."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from kdm.io import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--cards', required=True)
    parser.add_argument('--claim', required=True)
    parser.add_argument('--run', default='run_20260930_140337')
    parser.add_argument('--missing-keys', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--dataset', default='food101')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    args = parser.parse_args()
    folder = ROOT / 'outputs/supplemental/remaining11' / args.run / 'dispatch'
    folder.mkdir(parents=True, exist_ok=True)
    reservation = folder / (args.claim + '.json')
    with reservation.open('x') as stream:
        json.dump({'claim': args.claim, 'model': args.model,
                   'status': 'reserved', 'created_utc': datetime.now(timezone.utc).isoformat()}, stream)
    command = [
        'bash', str(ROOT / 'workflows/supplemental/remaining11/worker.sh'),
        str(ROOT), args.cards, args.python,
        str(ROOT / 'workflows/supplemental/remaining11/generate.py'),
        '--model', args.model, '--dataset', args.dataset, '--stage', args.stage,
        '--missing-keys', args.missing_keys, '--run-name', args.run,
        '--claim-id', args.claim, '--owner', 'root', '--execute',
        '--shard', str(args.shard), '--n-shards', str(args.n_shards),
    ]
    log = folder / (args.claim + '.log')
    with log.open('xb') as stream:
        worker = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                  stderr=subprocess.STDOUT, start_new_session=True,
                                  stdin=subprocess.DEVNULL, close_fds=True)
    receipt = {'claim': args.claim, 'model': args.model, 'stage': args.stage,
               'dataset': args.dataset, 'pid': worker.pid, 'cards': args.cards,
               'shard': args.shard, 'n_shards': args.n_shards,
               'created_utc': datetime.now(timezone.utc).isoformat(),
               'command': command, 'log': str(log), 'status': 'started'}
    atomic_json(reservation, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
