"""Start one explicitly checked finite event runner and retain its real PID."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from kdm.io import atomic_json, file_hash
from workflows.supplemental.remaining11.event_queue import NAME, environment, host_registration, now, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-id', required=True)
    parser.add_argument('--checked-queue-id', required=True)
    args = parser.parse_args()
    if any(not NAME.fullmatch(value) for value in (args.queue_id, args.checked_queue_id)):
        raise ValueError('Event queue identifiers must be explicit valid names')
    _registry, host, _details = host_registration()
    plan_path = ROOT / 'workflows/supplemental/remaining11/plan.json'
    plan = read(plan_path)
    folder = ROOT / 'outputs/supplemental/remaining11' / plan['run'] / 'queues' / host
    checked = folder / args.checked_queue_id
    state_path = checked / 'queue_state.json'
    state = read(state_path)
    if (state['status'] != 'checked' or state['host'] != host
            or state['plan_sha256'] != file_hash(plan_path) or read(checked / 'plan.json') != plan):
        raise ValueError('The finite runner requires its unchanged real CPU queue check')
    enabled = [node for node in plan['nodes'] if node['host'] == host and node['enabled']]
    if not enabled or any(state['nodes'][node['id']]['status'] != 'waiting' for node in enabled):
        raise ValueError('An enabled queue node lacks successful real CPU admission')
    if (folder / args.queue_id).exists():
        raise FileExistsError('Event runner identifier is already used')
    receipt_path = folder / (args.queue_id + '.launcher.json')
    if receipt_path.exists():
        raise FileExistsError('Event runner launcher receipt is already used')
    log = folder / (args.queue_id + '.launcher.log')
    command = [sys.executable, str(ROOT / 'workflows/supplemental/remaining11/event_queue.py'),
               'execute', '--host', host, '--queue-id', args.queue_id]
    with log.open('xb') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    receipt = {'host': host, 'run': plan['run'], 'queue_id': args.queue_id, 'pid': process.pid,
               'started_utc': now(), 'command': command, 'log': str(log),
               'checked_queue_state': str(state_path), 'checked_state_sha256': file_hash(state_path),
               'status': 'runner_started_pending_event_registration',
               'gpu_model_started_by_launcher': False, 'timers_created': 0,
               'scientific_parameters_changed': False}
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
