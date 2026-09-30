"""Validate an explicit supplemental claim before starting its real worker."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash, within

NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}')
STAGES = ('formal', 'independent', 'candidate')


def now():
    return datetime.now(timezone.utc).isoformat()


def registered_inputs(args):
    for field in ('claim', 'run', 'owner'):
        if not NAME.fullmatch(getattr(args, field)):
            raise ValueError('Invalid dispatch identifier: ' + field)
    declaration = json.loads((ROOT / 'workflows/supplemental/remaining11/generation.json').read_text())
    if args.model not in declaration['models'] or args.stage not in STAGES:
        raise ValueError('Unregistered supplemental model or generation stage')
    if args.dataset not in ('food101', 'vizwiz') or (args.stage == 'candidate' and args.dataset != 'food101'):
        raise ValueError('Unregistered dataset/stage condition')
    if args.n_shards < 1 or not 0 <= args.shard < args.n_shards:
        raise ValueError('Invalid stable sample shard')
    cards = args.cards.split(',')
    if (not cards or any(not re.fullmatch(r'[0-9]+', card) for card in cards)
            or len(cards) != len(set(cards))):
        raise ValueError('Invalid physical GPU list')
    registry = json.loads((ROOT / 'workflows/supplemental/remaining11/host_registry.json').read_text())
    hosts = [(name, item) for name, item in registry['hosts'].items()
             if item['hostname'] == socket.gethostname() and Path(item['root']).resolve() == ROOT]
    if len(hosts) != 1:
        raise ValueError('Unregistered dispatch hostname/project root')
    host, details = hosts[0]
    if not set(cards) <= {str(card) for card in details['allowed_gpus']}:
        raise ValueError('Dispatch requests an unauthorized physical GPU')
    spec = json.loads((ROOT / f'configs/runtime/{args.model}.json').read_text())
    if len(cards) != spec['gpu_count'] or spec['key'] != args.model or spec['availability'] != 'resolved':
        raise ValueError('Dispatch model identity or GPU count differs from registered runtime')
    override = registry.get('runtime_overrides', {}).get(host, {}).get(args.model)
    expected_python = override['environment_python'] if override else spec['environment_python']
    if override is None and registry['model_hosts'][args.model] != host:
        raise ValueError('Model has no runtime path admission on this dispatch host')
    python = Path(args.python)
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('Registered Python executable is unavailable')
    if os.path.abspath(args.python) != os.path.abspath(expected_python):
        raise ValueError('Dispatch Python differs from the registered runtime environment')
    missing = within(ROOT, args.missing_keys)
    if not missing.is_file() or missing.stat().st_size == 0:
        raise ValueError('Explicit missing-key input is absent or empty: ' + str(missing))
    worker = ROOT / 'workflows/supplemental/remaining11/worker.sh'
    runner = ROOT / 'workflows/supplemental/remaining11/generate.py'
    if not worker.is_file() or not runner.is_file():
        raise ValueError('Supplemental worker or generation entrypoint is unavailable')
    return {'host': host, 'cards': cards, 'missing_keys_path': str(missing),
            'missing_keys_sha256': file_hash(missing),
            'registry_sha256': file_hash(ROOT / 'workflows/supplemental/remaining11/host_registry.json')}


def checked_process(command, stdout_path, stderr_path, environment, cwd=None):
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        result = subprocess.run(command, cwd=ROOT if cwd is None else cwd, env=environment, stdout=stdout,
                                stderr=stderr, stdin=subprocess.DEVNULL, check=False)
    if result.returncode:
        raise RuntimeError(f'CPU preflight exited {result.returncode}; stderr: {stderr_path}')


def framework_import_preflight(args, folder, prefix, environment):
    source = (
        'import json,queue,sysconfig,torch,transformers; from pathlib import Path; '
        'print(json.dumps({"queue_path":queue.__file__,"queue_class":queue.Queue.__name__,'
        '"stdlib_queue_path":str(Path(sysconfig.get_path("stdlib"))/"queue.py"),'
        '"torch":torch.__version__,"transformers":transformers.__version__,'
        '"cuda_initialized":torch.cuda.is_initialized()}))'
    )
    stdout, stderr = folder / (prefix + '.framework.json'), folder / (prefix + '.framework.stderr.log')
    command = [args.python, '-c', source]
    checked_process(command, stdout, stderr, environment,
                    cwd=ROOT / 'workflows/supplemental/remaining11')
    observed = json.loads(stdout.read_text())
    spec = json.loads((ROOT / f'configs/runtime/{args.model}.json').read_text())
    if (Path(observed['queue_path']).resolve() != Path(observed['stdlib_queue_path']).resolve()
            or observed['queue_class'] != 'Queue' or observed['cuda_initialized']
            or any(observed[name] != spec['versions'][name] for name in ('torch', 'transformers'))):
        raise ValueError('Actual framework import differs from standard library or registered versions')
    return {'command': command, 'cwd': str(ROOT / 'workflows/supplemental/remaining11'),
            'stdout': str(stdout), 'sha256': file_hash(stdout), 'observed': observed,
            'model_loading_performed': False, 'generation_performed': False}


def zero_import_failure_evidence(claim):
    claim = within(ROOT, claim)
    owner_path = claim / 'owner.json'
    owner = json.loads(owner_path.read_text())
    source_id, model, dataset, stage = (owner[field] for field in ('claim_id', 'model', 'dataset', 'stage'))
    if claim.name != source_id:
        raise ValueError('Zero-load failure claim path differs from original owner')
    run = claim.parents[4]
    record = run / 'records' / model / dataset / stage / source_id
    raw = run / 'raw' / model / dataset / stage / source_id
    dispatch_path = run / 'dispatch' / (source_id + '.json')
    original = json.loads(dispatch_path.read_text())
    failure_path, progress_path, admission_path = (record / name for name in ('failure.json', 'progress.json', 'admission.json'))
    failure, progress, admission = (json.loads(path.read_text()) for path in (failure_path, progress_path, admission_path))
    error = "ImportError: cannot import name 'Queue' from 'queue' (" + str(ROOT / 'workflows/supplemental/remaining11/queue.py') + ')'
    if (progress['status'] != 'failed' or progress['completed'] != 0 or progress['completed_parts']
            or progress['current_task_key'] is not None or failure['completed_rows'] != 0
            or failure['current_task_key'] is not None or failure['current_raw_path'] is not None
            or failure['current_part_expected_keys'] or failure['error'] != error
            or progress['error'] != error or not failure['no_retry_performed']):
        raise ValueError('Source is outside the precise zero-model-load standard-library import failure')
    trace = failure['traceback']
    if ('src/kdm/models/hf.py' not in trace or 'in __init__' not in trace
            or '    import torch\n' not in trace or not trace.rstrip().endswith(error)):
        raise ValueError('Source traceback does not establish failure before checkpoint loading')
    if (owner['pid'] != progress['pid'] or original['pid'] != progress['pid']
            or (Path('/proc') / str(progress['pid'])).exists()):
        raise ValueError('Original zero-load worker PID differs or is still present')
    if not raw.is_dir() or any(raw.iterdir()) or list(record.glob('*.complete.json')) or (record / 'complete.json').exists() or (claim / 'complete.json').exists():
        raise ValueError('Original import failure has raw, part, or generation-completion artifacts')
    plan_path = within(ROOT, original['plan_preflight']['path'])
    source_plan = json.loads(plan_path.read_text())
    if (source_plan != owner['plan'] or source_plan != admission['task_plan']
            or file_hash(plan_path) != original['plan_preflight']['sha256']
            or progress['expected'] != source_plan['expected_generation_rows']):
        raise ValueError('Original import failure plan/owner/admission coverage differs')
    keys_path = claim / 'keys.jsonl'
    key_sha = file_hash(keys_path)
    if key_sha != owner['keys_sha256']:
        raise ValueError('Original ownership keys changed')
    seen, digest = set(), hashlib.sha256()
    for row in read_jsonl(keys_path):
        if row['key'] in seen or not re.fullmatch('[0-9a-f]{64}', row['key']):
            raise ValueError('Original zero-load ownership has duplicate or invalid keys')
        seen.add(row['key'])
        digest.update((row['key'] + '\n').encode())
    if len(seen) != source_plan['expected_generation_rows'] or digest.hexdigest() != source_plan['ordered_selected_keys_sha256']:
        raise ValueError('Original ownership keys are incomplete or differ from exact admitted order')
    log_path = within(ROOT, original['log'])
    if error not in log_path.read_text():
        raise ValueError('Original worker log lacks the same exact zero-load ImportError')
    paths = (owner_path, keys_path, dispatch_path, plan_path, failure_path, progress_path, admission_path, log_path)
    return {
        'failure_type': 'zero_model_load_standard_library_queue_shadowing',
        'source_claim_id': source_id, 'source_claim_path': str(claim.relative_to(ROOT)),
        'source_pid': progress['pid'], 'source_process_present': False,
        'model': model, 'dataset': dataset, 'stage': stage, 'owner': owner['owner'],
        'cards': original['cards'].split(','), 'shard': source_plan['shard'], 'n_shards': source_plan['n_shards'],
        'key_start': source_plan['key_start'], 'key_stop': source_plan['key_stop'],
        'expected_rows': len(seen), 'ordered_keys_sha256': digest.hexdigest(),
        'keys_sha256': key_sha, 'plan_sha256': stable_hash(source_plan),
        'source_files_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in paths},
        'source_error': error, 'source_raw_directory': str(raw.relative_to(ROOT)),
        'source_raw_directory_empty': True, 'model_loading_performed': False,
        'generated_rows': 0, 'scientific_parameters_changed': False,
    }


def validate_import_release(claim, replacement, plan, cards, owner):
    claim = within(ROOT, claim)
    marker_path = claim / 'released.json'
    marker = json.loads(marker_path.read_text())
    if (marker['schema'] != 'kdm_remaining11_zero_load_import_release_v1'
            or marker['replacement_claim_id'] != replacement or marker['generation_complete']
            or marker['scientific_parameters_changed'] or marker['automatic_retry_performed']):
        raise ValueError('Ownership release does not authorize this exact replacement claim')
    evidence = zero_import_failure_evidence(claim)
    if marker['source_evidence'] != evidence or marker['source_evidence_sha256'] != stable_hash(evidence):
        raise ValueError('Ownership-release evidence differs from immutable original failure')
    original = json.loads((claim / 'owner.json').read_text())
    if (plan != original['plan'] or list(cards) != evidence['cards'] or owner != evidence['owner']):
        raise ValueError('Import-failure replacement changes original keys, parameters, cards or owner')
    probe = marker['framework_import_preflight']
    if file_hash(probe['stdout']) != probe['sha256'] or probe['observed']['cuda_initialized']:
        raise ValueError('Ownership release lacks the preserved successful CPU framework import')
    return {'source_claim_id': evidence['source_claim_id'], 'replacement_claim_id': replacement,
            'release_path': str(marker_path.relative_to(ROOT)), 'release_sha256': file_hash(marker_path),
            'source_evidence_sha256': marker['source_evidence_sha256'],
            'source_keys_sha256': evidence['keys_sha256'], 'source_plan_sha256': evidence['plan_sha256'],
            'expected_rows': evidence['expected_rows'], 'ordered_keys_sha256': evidence['ordered_keys_sha256'],
            'source_generated_rows': 0, 'scientific_parameters_changed': False}


def release_import_failure(args):
    import fcntl

    registered_inputs(args)
    if args.recovery_from is None or not NAME.fullmatch(args.recovery_from) or args.recovery_from == args.claim:
        raise ValueError('Explicit release requires a different named source and replacement claim')
    run = ROOT / 'outputs/supplemental/remaining11' / args.run
    stage = run / 'claims' / args.model / args.dataset / args.stage
    claim = stage / args.recovery_from
    if (stage / args.claim).exists() or (run / 'dispatch' / (args.claim + '.json')).exists():
        raise ValueError('Explicit replacement claim is already reserved or owned')
    with (stage / 'ownership.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (claim / 'released.json').exists():
            raise ValueError('Original import-failure ownership is already explicitly released')
        evidence = zero_import_failure_evidence(claim)
        original = json.loads((claim / 'owner.json').read_text())
        plan = original['plan']
        if ((evidence['model'], evidence['dataset'], evidence['stage'], evidence['owner'],
             evidence['shard'], evidence['n_shards'], evidence['cards']) !=
                (args.model, args.dataset, args.stage, args.owner, args.shard, args.n_shards, args.cards.split(','))
                or within(ROOT, plan['missing_keys_path']) != within(ROOT, args.missing_keys)
                or file_hash(within(ROOT, args.missing_keys)) != plan['missing_keys_sha256']):
            raise ValueError('Explicit release changes the original model, inputs, range, cards or owner')
        checks = claim / 'release_checks' / args.claim
        checks.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ)
        cache = ROOT / 'cache/tmp'
        cache.mkdir(parents=True, exist_ok=True)
        env.update(TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
                   PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'src'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', KDM_SUPPLEMENTAL_DATASET=args.dataset)
        framework = framework_import_preflight(args, checks, 'release', env)
        command = [args.python, str(ROOT / 'workflows/supplemental/remaining11/generate.py'),
                   '--model', args.model, '--dataset', args.dataset, '--stage', args.stage,
                   '--missing-keys', args.missing_keys, '--shard', str(args.shard), '--n-shards', str(args.n_shards),
                   '--key-start', str(plan['key_start']), '--key-stop', str(plan['key_stop']),
                   '--check-plan', '--plan-output', str(checks / 'replacement_plan.json')]
        checked_process(command, checks / 'replacement_plan.stdout.json', checks / 'replacement_plan.stderr.log', env)
        replacement = json.loads((checks / 'replacement_plan.json').read_text())
        if replacement != plan:
            raise ValueError('Replacement CPU plan differs from the full original admitted plan')
        marker = {
            'schema': 'kdm_remaining11_zero_load_import_release_v1', 'released_utc': now(),
            'source_claim_id': args.recovery_from, 'replacement_claim_id': args.claim,
            'source_evidence': evidence, 'source_evidence_sha256': stable_hash(evidence),
            'framework_import_preflight': framework,
            'replacement_plan_path': str((checks / 'replacement_plan.json').relative_to(ROOT)),
            'replacement_plan_sha256': file_hash(checks / 'replacement_plan.json'),
            'generation_complete': False, 'model_loading_performed': False,
            'scientific_parameters_changed': False, 'automatic_retry_performed': False,
            'original_files_retained': True,
        }
        with (claim / 'released.json').open('x', encoding='utf-8') as stream:
            json.dump(marker, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    print(json.dumps({'released_claim': args.recovery_from, 'replacement_claim': args.claim,
                      'keys': evidence['expected_rows'], 'release_path': str(claim / 'released.json'),
                      'source_generated_rows': 0, 'new_gpu_worker_started': False}))


def input_failure_recovery(args, folder):
    if args.recovery_from is None:
        return None
    if not NAME.fullmatch(args.recovery_from):
        raise ValueError('Invalid input-failure source claim')
    original_path = folder / (args.recovery_from + '.json')
    original = json.loads(original_path.read_text())
    if (original['model'] != args.model or original['stage'] != args.stage
            or original.get('dataset', 'food101') != args.dataset
            or original.get('shard', 0) != args.shard
            or original.get('n_shards', 1) != args.n_shards):
        raise ValueError('Input-failure recovery changes the original task interval')
    if (Path('/proc') / str(original['pid'])).exists():
        raise ValueError('Original input-failure worker process is still present')
    log_path = within(ROOT, original['log'])
    log = log_path.read_text()
    source_claim = folder.parent / 'claims' / args.model / args.dataset / args.stage / args.recovery_from
    if (source_claim / 'released.json').is_file():
        plan = json.loads((source_claim / 'owner.json').read_text())['plan']
        released = validate_import_release(source_claim, args.claim, plan, args.cards.split(','), args.owner)
        return {'source_failure': 'zero_model_load_standard_library_queue_shadowing', **released}
    if 'FileNotFoundError' not in log or 'remaining_keys/' not in log:
        raise ValueError('Original log does not establish the missing-input failure')
    run = ROOT / 'outputs/supplemental/remaining11' / args.run
    names = ('raw', 'records', 'claims')
    previous_paths = [run / name / args.model / args.dataset / args.stage / args.recovery_from
                      for name in names]
    if any(path.exists() for path in previous_paths):
        raise ValueError('Original input-failure claim has downstream output or ownership artifacts')
    return {
        'source_dispatch_path': str(original_path), 'source_dispatch_sha256': file_hash(original_path),
        'source_log_path': str(log_path), 'source_log_sha256': file_hash(log_path),
        'source_pid': original['pid'], 'source_process_present': False,
        'source_failure': 'missing_key_file_before_gpu_admission',
        'source_raw_records_claims_absent': [str(path) for path in previous_paths],
        'source_raw_rows': 0, 'scientific_parameters_changed': False,
    }


def dispatch(args):
    for field in ('claim', 'run', 'owner'):
        if not NAME.fullmatch(getattr(args, field)):
            raise ValueError('Invalid dispatch identifier: ' + field)
    folder = ROOT / 'outputs/supplemental/remaining11' / args.run / 'dispatch'
    folder.mkdir(parents=True, exist_ok=True)
    reservation = folder / (args.claim + '.json')
    receipt = {'claim': args.claim, 'model': args.model, 'stage': args.stage,
               'dataset': args.dataset, 'cards': args.cards,
               'shard': args.shard, 'n_shards': args.n_shards,
               'owner': args.owner, 'status': 'reserved', 'created_utc': now(),
               'recovery_from': args.recovery_from, 'gpu_worker_started': False}
    with reservation.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    phase = 'input_validation'
    try:
        receipt['inputs'] = registered_inputs(args)
        receipt['input_failure_recovery_evidence'] = input_failure_recovery(args, folder)
        environment = dict(os.environ)
        cache = ROOT / 'cache/tmp'
        cache.mkdir(parents=True, exist_ok=True)
        environment.update(TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
                           PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', PYTHONPATH=str(ROOT / 'src'),
                           KDM_SUPPLEMENTAL_DATASET=args.dataset)
        phase = 'actual_framework_import_preflight'
        receipt['framework_import_preflight'] = framework_import_preflight(args, folder, args.claim, environment)
        phase = 'runtime_cpu_preflight'
        runtime_stdout = folder / (args.claim + '.runtime.json')
        runtime_stderr = folder / (args.claim + '.runtime.stderr.log')
        runtime_command = [args.python, str(ROOT / 'workflows/supplemental/remaining11/execution.py'),
                           '--model', args.model]
        checked_process(runtime_command, runtime_stdout, runtime_stderr, environment)
        runtime = json.loads(runtime_stdout.read_text())
        if runtime['runtime_spec']['key'] != args.model:
            raise ValueError('CPU runtime admission returned another model')
        receipt['runtime_preflight'] = {'command': runtime_command,
                                        'stdout': str(runtime_stdout),
                                        'sha256': file_hash(runtime_stdout)}
        phase = 'generation_cpu_preflight'
        plan_path = folder / (args.claim + '.plan.json')
        plan_stdout = folder / (args.claim + '.plan.stdout.json')
        plan_stderr = folder / (args.claim + '.plan.stderr.log')
        task_arguments = [
            '--model', args.model, '--dataset', args.dataset, '--stage', args.stage,
            '--missing-keys', args.missing_keys, '--shard', str(args.shard),
            '--n-shards', str(args.n_shards),
        ]
        plan_command = [args.python, str(ROOT / 'workflows/supplemental/remaining11/generate.py'),
                        *task_arguments, '--check-plan', '--plan-output', str(plan_path)]
        checked_process(plan_command, plan_stdout, plan_stderr, environment)
        plan = json.loads(plan_path.read_text())
        recovery = receipt['input_failure_recovery_evidence']
        if recovery is not None and recovery['source_failure'] == 'zero_model_load_standard_library_queue_shadowing':
            source_claim = folder.parent / 'claims' / args.model / args.dataset / args.stage / args.recovery_from
            validate_import_release(source_claim, args.claim, plan, args.cards.split(','), args.owner)
        if (plan['expected_generation_rows'] < 1 or not plan['checks']['explicit_missing_keys_registered']
                or not plan['checks']['native_runtime_proofs']):
            raise ValueError('Generation preflight did not establish actual runnable missing keys')
        receipt['plan_preflight'] = {'command': plan_command, 'path': str(plan_path),
                                     'sha256': file_hash(plan_path),
                                     'expected_rows': plan['expected_generation_rows'],
                                     'ordered_keys_sha256': plan['ordered_selected_keys_sha256']}
        command = [
            'bash', str(ROOT / 'workflows/supplemental/remaining11/worker.sh'),
            str(ROOT), args.cards, args.python,
            str(ROOT / 'workflows/supplemental/remaining11/generate.py'),
            *task_arguments, '--run-name', args.run, '--claim-id', args.claim,
            '--owner', args.owner, '--execute',
        ]
        receipt.update(status='preflight_passed', updated_utc=now(), command=command)
        atomic_json(reservation, receipt)
        if args.check_only:
            print(json.dumps(receipt, indent=2))
            return receipt
        phase = 'worker_start'
        log = folder / (args.claim + '.log')
        with log.open('xb') as stream:
            worker = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                      stderr=subprocess.STDOUT, start_new_session=True,
                                      stdin=subprocess.DEVNULL, close_fds=True)
        receipt.update(pid=worker.pid, log=str(log), status='started',
                       gpu_worker_started=True, started_utc=now())
        atomic_json(reservation, receipt)
        print(json.dumps(receipt, indent=2))
        return receipt
    except BaseException as exc:
        receipt.update(status='failed', failed_phase=phase, updated_utc=now(),
                       error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
        atomic_json(reservation, receipt)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', nargs='?', choices=('dispatch', 'release-import-failure'), default='dispatch')
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
    parser.add_argument('--owner', default='root')
    parser.add_argument('--recovery-from')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'release-import-failure':
        release_import_failure(args)
    else:
        dispatch(args)


if __name__ == '__main__':
    main()
