#!/usr/bin/env python3
"""One-shot event-driven CPU screening after a specific generation process exits."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from kdm.io import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--closed-record', required=True)
    args = parser.parse_args()
    allowed = {'gemma3_4b': 'panel5_engine_v6'}
    if args.model not in allowed:
        raise ValueError('This deployment is bounded to the authorized Gemma v4 producer')
    closed = f'outputs/records/acceleration_v4/{allowed[args.model]}/{args.model}'
    if args.closed_record != closed:
        raise ValueError('Unexpected closed cohort')
    if str(ROOT) != '/home/team/zhanghaonan/TAFFC/knowledge-deficit-mitigation':
        raise ValueError('This one-shot follow-up is restricted to 6403')
    if not hasattr(os, 'pidfd_open'):
        raise RuntimeError('Run this waiting process with system Python supporting pidfd_open')
    run = 'panel5_independent_food_v5'
    record = ROOT / 'outputs/records/acceleration_v4' / run / args.model
    out = ROOT / 'outputs/annotations/independent_v5' / run / args.model
    followup = ROOT / 'outputs/records/independent_v5/free_postprocess_followup_gemma_v3' / args.model
    lock_path = ROOT / 'outputs/locks' / f'free_postprocess_{run}_{args.model}.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    followup.mkdir(parents=True, exist_ok=False)
    state = dict(schema='kdm_free_postprocess_followup_gemma_v3', model=args.model,
                 watcher_pid=os.getpid(), target_pid=args.pid, status='initializing',
                 run=run, record=str(record.relative_to(ROOT)), output=str(out.relative_to(ROOT)),
                 api_requests=0, gpu_used=False, retries=0, started_unix=time.time(),
                 code_sha256=file_hash(__file__), postprocess_sha256=file_hash(Path(__file__).with_name('postprocess.py')))
    atomic_json(followup / 'status.json', state)
    try:
        if out.exists():
            raise ValueError('Output already exists; refusing duplicate screening')
        try:
            fd = os.pidfd_open(args.pid)
        except ProcessLookupError:
            fd = None
        if fd is not None:
            try:
                proc = Path('/proc') / str(args.pid)
                cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
                if args.model == 'qwen35_4b':
                    queue = ROOT / 'outputs/records/independent_v5/qwen35_after_qwen25_queue_v1.json'
                    expected_cmd = ['python3', 'workflows/acceleration_v4/adopt_queue.py', '--config', str(queue)]
                    actual_cmd = (proc / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
                    if actual_cmd != expected_cmd:
                        raise ValueError('Target queue process identity mismatch')
                    config = json.loads(queue.read_text())
                    pending = [p for lane in config['lanes'] for p in lane['pending']]
                    if (len(pending) != 1 or pending[0]['complete_path'] != str((record / 'complete.json').relative_to(ROOT))
                        or '--model' not in pending[0]['command']
                        or pending[0]['command'][pending[0]['command'].index('--model') + 1] != args.model):
                        raise ValueError('Queue does not describe the exact authorized Qwen3.5 cohort')
                    state.update(target_kind='adopt_queue', queue_config=str(queue.relative_to(ROOT)),
                                 queue_config_sha256=file_hash(queue))
                elif ('workflows/independent_v5/gemma_native_pixels_entry_v4.py' not in cmd
                      or f'--model {args.model} ' not in cmd + ' '
                      or f'--run {run}' not in cmd):
                    raise ValueError('Target process identity mismatch')
                state['target_command'] = cmd
                state['target_start_ticks'] = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
            except FileNotFoundError:
                # Target exited after pidfd_open; the fd still binds that exact process.
                state['target_exited_during_identity_read'] = True
            state.update(status='waiting_for_target_exit', wait_mechanism='pidfd_poll_no_periodic_polling')
            atomic_json(followup / 'status.json', state)
            poller = select.poll()
            poller.register(fd, select.POLLIN)
            poller.poll()
            os.close(fd)
        if state.get('queue_config_sha256') and file_hash(ROOT / state['queue_config']) != state['queue_config_sha256']:
            raise ValueError('Queue config changed while waiting')
        done = json.loads((record / 'complete.json').read_text())
        progress = json.loads((record / 'progress.json').read_text())
        if (done.get('generation_complete') is not True or done.get('independent') != 48480
            or progress.get('status') != 'generation_complete' or progress.get('independent') != 48480):
            raise ValueError('Target exited without complete generation; no screening attempted')
        state.update(status='validating_and_screening', generation_finished_unix=time.time())
        atomic_json(followup / 'status.json', state)
        command = ['/home/team/lvshuyang/anaconda3/envs/ST_LORA/bin/python',
                   '-u', str(ROOT / 'workflows/independent_v5/postprocess.py'),
                   '--record', str(record), '--closed-record', str(ROOT / closed), '--out', str(out)]
        state['postprocess_command'] = command
        atomic_json(followup / 'status.json', state)
        subprocess.run(command, cwd=ROOT, check=True)
        summary = json.loads((out / 'summary.json').read_text())
        state.update(status='complete', finished_unix=time.time(),
                     summary_sha256=file_hash(out / 'summary.json'), screened_answers=summary['screened_answers'],
                     counts=summary['counts'], question_states=summary['question_states'], final_gt=False)
        atomic_json(followup / 'status.json', state)
    except BaseException as exc:
        state.update(status='failed_preserved_no_retry', error=type(exc).__name__ + ': ' + str(exc),
                     finished_unix=time.time())
        atomic_json(followup / 'failure.json', {'error': state['error'], 'traceback': traceback.format_exc(), 'retry_performed': False})
        atomic_json(followup / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
