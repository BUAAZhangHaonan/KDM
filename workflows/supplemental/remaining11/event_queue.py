"""Run an explicit finite queue using kernel process-completion events."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import selectors
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from kdm.io import atomic_json, file_hash, read_jsonl, within
from workflows.supplemental.remaining11.dispatch import framework_import_preflight, registered_inputs

NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}')
TERMINAL = {'generation_complete', 'failed_held', 'blocked', 'dependency_blocked', 'preflight_failed', 'capacity_held'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def host_registration():
    registry = read(ROOT / 'workflows/supplemental/remaining11/host_registry.json')
    matches = [(host, item) for host, item in registry['hosts'].items()
               if item['hostname'] == socket.gethostname() and Path(item['root']).resolve() == ROOT]
    if len(matches) != 1:
        raise ValueError('Event queue hostname/project root is unregistered')
    return registry, *matches[0]


def generation_arguments(job, run):
    return [
        '--model', job['model'], '--dataset', job['dataset'], '--stage', job['stage'],
        '--missing-keys', job['missing_keys'], '--shard', str(job.get('shard', 0)),
        '--n-shards', str(job.get('n_shards', 1)), '--run-name', run,
        '--claim-id', job['claim'], '--owner', job.get('owner', 'root'),
    ]


def dispatch_arguments(job, run):
    return [
        '--model', job['model'], '--dataset', job['dataset'], '--stage', job['stage'],
        '--missing-keys', job['missing_keys'], '--shard', str(job.get('shard', 0)),
        '--n-shards', str(job.get('n_shards', 1)), '--run', run,
        '--claim', job['claim'], '--owner', job.get('owner', 'root'),
        '--cards', ','.join(map(str, job['cards'])), '--python', job['python'],
    ]


def dispatch_namespace(job, run):
    return argparse.Namespace(
        model=job['model'], dataset=job['dataset'], stage=job['stage'],
        missing_keys=job['missing_keys'], shard=job.get('shard', 0),
        n_shards=job.get('n_shards', 1), run=run, claim=job['claim'],
        owner=job.get('owner', 'root'), cards=','.join(map(str, job['cards'])),
        python=job['python'], recovery_from=None,
    )


def environment():
    env = dict(os.environ)
    cache = ROOT / 'cache/tmp'
    cache.mkdir(parents=True, exist_ok=True)
    env.update(TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONPATH=str(ROOT / 'src'))
    return env


def validate_plan(plan):
    if plan['schema'] != 'kdm_remaining11_event_queue_v1' or not NAME.fullmatch(plan['run']):
        raise ValueError('Invalid finite event-queue schema/run')
    if plan['active_phase'] not in ('food101', 'vizwiz'):
        raise ValueError('Queue must name its explicitly authorized dataset phase')
    declaration = read(ROOT / 'workflows/supplemental/remaining11/generation.json')
    identifiers = [entry['claim_id'] for entry in plan['existing_claims']]
    identifiers.extend(node['id'] for node in plan['nodes'])
    if len(identifiers) != len(set(identifiers)) or any(not NAME.fullmatch(value) for value in identifiers):
        raise ValueError('Queue identifiers overlap or are invalid')
    enabled = []
    for node in plan['nodes']:
        if node['model'] not in declaration['models'] or node['stage'] not in ('formal', 'independent', 'candidate'):
            raise ValueError('Queue node has an unregistered model/stage')
        if node['dataset'] not in ('food101', 'vizwiz') or (node['dataset'] == 'vizwiz' and node['stage'] == 'candidate'):
            raise ValueError('Queue node has an unregistered dataset/stage')
        if node['resource_mode'] not in ('exclusive', 'shared'):
            raise ValueError('Unknown queue GPU ownership mode')
        dependencies = node['after_finished'] + node['requires_success']
        if node['id'] in dependencies or not set(dependencies) <= set(identifiers):
            raise ValueError('Queue node has an unknown or recursive direct dependency')
        if not node['enabled']:
            if not node['blocked_reasons']:
                raise ValueError('Disabled node lacks an explicit blocking reason')
            continue
        food_priority_independent = (
            plan['active_phase'] == 'food101' and node.get('food_priority_independent') is True
            and (node['host'], node['model'], node['dataset'], node['stage']) ==
            ('k100', 'qwen35_9b', 'vizwiz', 'independent')
            and node['resource_mode'] == 'shared' and node['cards'] == [0]
            and node['requires_success'] == ['k100_qwen35_9b_candidate']
            and node['expected_missing_rows'] == 35010
        )
        if node['blocked_reasons'] or (node['dataset'] != plan['active_phase'] and not food_priority_independent):
            raise ValueError('Enabled node conflicts with its admission or dataset phase')
        if not NAME.fullmatch(node['claim']) or not isinstance(node['expected_missing_rows'], int) or node['expected_missing_rows'] < 1:
            raise ValueError('Enabled node lacks an exclusive claim or actual missing-row count')
        if not node['capacity_evidence']:
            raise ValueError('Enabled node lacks inherited or actual resource evidence')
        group = (node['model'], node['dataset'], node['stage'])
        for prior in enabled + plan['existing_claims']:
            if (prior['model'], prior['dataset'], prior['stage']) != group:
                continue
            if (prior.get('n_shards', 1) != node.get('n_shards', 1)
                    or prior.get('shard', 0) == node.get('shard', 0)):
                raise ValueError('Enabled node overlaps an existing or another queued interval')
        enabled.append(node)
    visiting, visited = set(), set()
    nodes = {node['id']: node for node in plan['nodes']}

    def visit(identifier):
        if identifier in visited or identifier not in nodes:
            return
        if identifier in visiting:
            raise ValueError('Finite queue has a dependency cycle')
        visiting.add(identifier)
        node = nodes[identifier]
        for dependency in node['after_finished'] + node['requires_success']:
            visit(dependency)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in nodes:
        visit(identifier)


def run_cpu_process(command, stdout, stderr):
    """The same pidfd completion path is exercised by real CPU admission calls."""
    with stdout.open('xb') as out, stderr.open('xb') as err:
        process = subprocess.Popen(command, cwd=ROOT, env=environment(), stdout=out,
                                   stderr=err, stdin=subprocess.DEVNULL, close_fds=True)
        descriptor = os.pidfd_open(process.pid)
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_READ)
            selector.select()
        os.close(descriptor)
        return process.wait()


def preflight_node(node, run, folder):
    arguments = dispatch_namespace(node, run)
    inputs = registered_inputs(arguments)
    framework = framework_import_preflight(arguments, folder, 'queue', environment())
    owned = set()
    claimed = ROOT / 'outputs/supplemental/remaining11' / run / 'claims' / node['model'] / node['dataset'] / node['stage']
    for path in claimed.glob('*/keys.jsonl'):
        owned.update(row['key'] for row in read_jsonl(path))
    if owned.intersection(row['key'] for row in read_jsonl(within(ROOT, node['missing_keys']))):
        raise ValueError('Queued missing keys are already owned by an existing immutable claim')
    runtime_stdout, runtime_stderr = folder / 'runtime.json', folder / 'runtime.stderr.log'
    runtime_command = [node['python'], str(ROOT / 'workflows/supplemental/remaining11/execution.py'),
                       '--model', node['model']]
    code = run_cpu_process(runtime_command, runtime_stdout, runtime_stderr)
    if code:
        return {'status': 'preflight_failed', 'phase': 'runtime_cpu_admission', 'exit_code': code,
                'stderr_path': str(runtime_stderr), 'command': runtime_command}
    runtime = read(runtime_stdout)
    runtime['framework_import_preflight'] = framework
    plan_output = folder / 'generation_plan.json'
    command = [node['python'], str(ROOT / 'workflows/supplemental/remaining11/generate.py'),
               *generation_arguments(node, run), '--check-plan', '--plan-output', str(plan_output)]
    code = run_cpu_process(command, folder / 'generation_plan.stdout.json', folder / 'generation_plan.stderr.log')
    if code:
        return {'status': 'preflight_failed', 'phase': 'generation_cpu_plan', 'exit_code': code,
                'stderr_path': str(folder / 'generation_plan.stderr.log'), 'command': command}
    planned = read(plan_output)
    if (planned['expected_generation_rows'] != node['expected_missing_rows']
            or runtime['runtime_spec']['key'] != node['model']
            or not planned['checks']['native_runtime_proofs']
            or not planned['checks']['explicit_missing_keys_registered']):
        raise ValueError('CPU admission differs from the exact finite queue declaration')
    return {'status': 'waiting', 'inputs': inputs, 'runtime_receipt': str(runtime_stdout),
            'framework_import_preflight': framework,
            'runtime_receipt_sha256': file_hash(runtime_stdout), 'generation_plan': str(plan_output),
            'generation_plan_sha256': file_hash(plan_output),
            'expected_rows': planned['expected_generation_rows'],
            'ordered_keys_sha256': planned['ordered_selected_keys_sha256'],
            'pidfd_cpu_completion_verified': True, 'gpu_initialized': False}


def claim_paths(entry, run):
    folder = ROOT / 'outputs/supplemental/remaining11' / run / 'records'
    return folder / entry['model'] / entry['dataset'] / entry['stage'] / entry['claim_id']


def observe_claim(entry, run):
    folder = claim_paths(entry, run)
    progress = read(folder / 'progress.json')
    admission = read(folder / 'admission.json')
    if (progress['claim_id'], progress['model'], progress['dataset'], progress['stage']) != (
            entry['claim_id'], entry['model'], entry['dataset'], entry['stage']):
        raise ValueError('Existing claim identity differs from the explicit queue')
    cards = list(map(int, admission['runtime_admission']['execution']['physical_gpus']))
    if cards != entry['cards']:
        raise ValueError('Existing claim physical cards differ from queue ownership')
    if progress['status'] == 'generation_complete':
        complete = read(folder / 'complete.json')
        if progress['completed'] != progress['expected'] or not complete['generation_complete']:
            raise ValueError('Claim terminal receipt lacks actual full generation coverage')
        status = 'generation_complete'
    elif progress['status'] == 'failed':
        status = 'failed_held'
    else:
        status = 'running'
    return {'status': status, 'pid': progress['pid'], 'expected_rows': progress['expected'],
            'observed_rows': progress['completed'], 'progress_path': str(folder / 'progress.json'),
            'admission_path': str(folder / 'admission.json'),
            'admission_sha256': file_hash(folder / 'admission.json'),
            'failure_path': str(folder / 'failure.json') if (folder / 'failure.json').is_file() else None}


def process_descriptor(pid, entry):
    """ProcessLookupError means the process completed before the kernel watch opened."""
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return None
    try:
        command = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
    except FileNotFoundError:
        return descriptor
    expected = [entry['claim_id'].encode()]
    if entry.get('exclusive_job_path'):
        expected.append(entry['exclusive_job_path'].encode())
    if command and not any(value in command for value in expected):
        os.close(descriptor)
        raise ValueError('Recorded PID now belongs to an unrelated process')
    return descriptor


class EventQueue:
    def __init__(self, plan, plan_path, host, folder):
        self.plan, self.host, self.folder = plan, host, folder
        self.registry, actual, self.details = host_registration()
        if actual != host:
            raise ValueError('Requested queue host differs from its actual execution host')
        self.state = {'schema': 'kdm_remaining11_event_queue_state_v1', 'run': plan['run'],
                      'host': host, 'plan_path': str(plan_path), 'plan_sha256': file_hash(plan_path),
                      'started_utc': now(), 'queue_pid': os.getpid(), 'status': 'checking',
                      'claims': {}, 'nodes': {}, 'events': [], 'gpu_queries': 0,
                      'timers_created': 0, 'scientific_retries': 0, 'research_complete': False}
        self.entries = {entry['claim_id']: entry for entry in plan['existing_claims'] if entry['host'] == host}
        self.nodes = {node['id']: node for node in plan['nodes'] if node['host'] == host}
        self.active = {}
        self.selector = selectors.DefaultSelector()

    def save(self, event=None):
        if event is not None:
            self.state['events'].append({'when': now(), **event})
        self.state['updated_utc'] = now()
        atomic_json(self.folder / 'queue_state.json', self.state)

    def check(self):
        for claim, entry in self.entries.items():
            self.state['claims'][claim] = observe_claim(entry, self.plan['run'])
        for identifier, node in self.nodes.items():
            if not node['enabled']:
                self.state['nodes'][identifier] = {'status': 'blocked', 'status_hint': node.get('status_hint'),
                                                  'reasons': node['blocked_reasons']}
                self.save({'event': 'candidate_blocked', 'node': identifier, 'reasons': node['blocked_reasons']})
                continue
            out = self.folder / 'preflight' / identifier
            out.mkdir(parents=True, exist_ok=False)
            job_path = out / 'cpu_job.json'
            atomic_json(job_path, {'node': node, 'run': self.plan['run'], 'folder': str(out)})
            stdout, stderr = out / 'cpu_check.json', out / 'cpu_check.stderr.log'
            command = [sys.executable, str(Path(__file__)), 'check-node', '--job-file', str(job_path)]
            code = run_cpu_process(command, stdout, stderr)
            result = read(stdout) if code == 0 else {
                'status': 'preflight_failed', 'phase': 'isolated_node_cpu_preflight',
                'exit_code': code, 'stderr_path': str(stderr), 'command': command,
                'gpu_initialized': False, 'scientific_retry_performed': False,
            }
            self.state['nodes'][identifier] = result
            self.save({'event': 'cpu_preflight', 'node': identifier, 'status': result['status']})
        self.state['status'] = 'checked'
        self.save()

    def watch(self, identifier, entry, pid):
        descriptor = process_descriptor(pid, entry)
        if descriptor is None:
            self.finish(identifier, entry)
            return
        self.selector.register(descriptor, selectors.EVENT_READ, identifier)
        self.active[identifier] = {'entry': entry, 'pidfd': descriptor, 'pid': pid}

    def finish(self, identifier, entry):
        folder = claim_paths(entry, self.plan['run'])
        if (folder / 'progress.json').is_file():
            result = observe_claim(entry, self.plan['run'])
            if result['status'] == 'running':
                result.update(status='failed_held', reason='process_completed_without_terminal_generation_receipt')
        else:
            result = {'status': 'failed_held', 'reason': 'worker_completed_before_generation_progress',
                      'claim_id': entry['claim_id'], 'raw_completion_claimed': False}
        target = self.state['nodes'] if identifier in self.nodes else self.state['claims']
        target[identifier] = {**target[identifier], **result, 'process_completed_event_utc': now()}
        self.save({'event': 'process_completed', 'identifier': identifier, 'status': result['status']})

    def status(self, identifier):
        if identifier in self.state['claims']:
            return self.state['claims'][identifier]['status']
        if identifier in self.state['nodes']:
            return self.state['nodes'][identifier]['status']
        return 'foreign_host_dependency'

    def resource_available(self, node):
        occupancy = Counter()
        exclusive = set()
        for active in self.active.values():
            entry = active['entry']
            occupancy.update(entry['cards'])
            if entry['resource_mode'] == 'exclusive':
                exclusive.update(entry['cards'])
        cards = set(node['cards'])
        if cards.intersection(exclusive):
            return False
        if node['resource_mode'] == 'exclusive':
            return all(occupancy[card] == 0 for card in cards)
        return all(occupancy[card] < self.details.get('max_workers_per_gpu', 1) for card in cards)

    def start(self, identifier, node):
        record = self.state['nodes'][identifier]
        if node.get('minimum_free_mib') is not None:
            command = ['nvidia-smi', '-i', ','.join(map(str, node['cards'])),
                       '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits']
            observed = {}
            for line in subprocess.check_output(command, text=True).splitlines():
                card, uuid, available = [field.strip() for field in line.split(',')]
                observed[card] = {'uuid': uuid, 'free_mib': int(available)}
            if set(observed) != set(map(str, node['cards'])) or any(
                    item['uuid'] != self.details['gpu_uuids'][card] for card, item in observed.items()):
                raise ValueError('Completion-event capacity check found a different physical GPU')
            record['event_capacity_observation'] = observed
            self.state['gpu_queries'] += 1
            if any(item['free_mib'] < node['minimum_free_mib'] for item in observed.values()):
                record.update(status='capacity_held', minimum_free_mib=node['minimum_free_mib'])
                self.save({'event': 'event_capacity_held', 'node': identifier, 'observed': observed})
                return
        record.update(status='starting', started_utc=now())
        self.save({'event': 'dispatch_starting', 'node': identifier})
        exclusive = node['resource_mode'] == 'exclusive' and self.details.get('max_workers_per_gpu', 1) > 1
        command = [sys.executable, str(ROOT / 'workflows/supplemental/remaining11/dispatch.py'),
                   *dispatch_arguments(node, self.plan['run'])]
        if exclusive:
            command.append('--check-only')
        stdout, stderr = self.folder / (identifier + '.dispatch.json'), self.folder / (identifier + '.dispatch.stderr.log')
        code = run_cpu_process(command, stdout, stderr)
        if code:
            record.update(status='preflight_failed', exit_code=code, stderr_path=str(stderr), command=command)
            self.save({'event': 'dispatch_failed', 'node': identifier, 'exit_code': code})
            return
        receipt = read(stdout)
        if exclusive:
            job_path = self.folder / (identifier + '.exclusive_job.json')
            atomic_json(job_path, {'run': self.plan['run'], 'node': node,
                                   'cpu_dispatch_receipt': str(stdout), 'cpu_dispatch_sha256': file_hash(stdout)})
            log = self.folder / (identifier + '.exclusive_worker.log')
            worker_command = [sys.executable, str(Path(__file__)), 'exclusive-worker', '--job-file', str(job_path)]
            with log.open('xb') as stream:
                worker = subprocess.Popen(worker_command, cwd=ROOT, env=environment(), stdout=stream,
                                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                          start_new_session=True, close_fds=True)
            pid = worker.pid
            record.update(exclusive_worker_command=worker_command, worker_log=str(log),
                          exclusive_gpu_ownership=True, cpu_dispatch_receipt=str(stdout))
        else:
            if receipt['status'] != 'started' or not receipt['gpu_worker_started']:
                raise ValueError('Dispatcher lacks its actual worker-start receipt')
            pid = receipt['pid']
            record.update(dispatch_receipt=str(stdout), worker_log=receipt['log'])
        record.update(status='running', pid=pid, claim_id=node['claim'])
        entry = {'claim_id': node['claim'], 'model': node['model'], 'dataset': node['dataset'],
                 'stage': node['stage'], 'cards': node['cards'], 'resource_mode': node['resource_mode']}
        if exclusive:
            entry['exclusive_job_path'] = str(job_path)
        self.save({'event': 'worker_started', 'node': identifier, 'pid': pid})
        self.watch(identifier, entry, pid)

    def execute(self):
        enabled_nodes = [node for node in self.nodes.values()
                         if self.state['nodes'][node['id']]['status'] == 'waiting']
        cards = {card for node in enabled_nodes for card in node['cards']}
        dependencies = {dependency for node in enabled_nodes
                        for dependency in node['after_finished'] + node['requires_success']}
        for claim, entry in self.entries.items():
            if claim in dependencies or cards.intersection(entry['cards']):
                self.watch(claim, entry, self.state['claims'][claim]['pid'])
        self.state['status'] = 'running'
        self.save()
        while True:
            for identifier, node in self.nodes.items():
                if self.state['nodes'][identifier]['status'] != 'waiting':
                    continue
                failed = [dependency for dependency in node['requires_success']
                          if self.status(dependency) in TERMINAL and self.status(dependency) != 'generation_complete']
                if failed:
                    self.state['nodes'][identifier].update(status='dependency_blocked', failed_dependencies=failed)
                    self.save({'event': 'dependency_blocked', 'node': identifier, 'dependencies': failed})
                    continue
                if not all(self.status(dependency) == 'generation_complete' for dependency in node['requires_success']):
                    continue
                if not all(self.status(dependency) in TERMINAL for dependency in node['after_finished']):
                    continue
                if self.resource_available(node):
                    self.start(identifier, node)
            pending = [identifier for identifier, result in self.state['nodes'].items() if result['status'] == 'waiting']
            if not self.active:
                self.state['status'] = 'event_source_blocked' if pending else 'finite_queue_exhausted'
                self.state['pending_event_sources'] = pending
                self.save()
                return
            for event, _mask in self.selector.select():
                identifier = event.data
                active = self.active.pop(identifier)
                self.selector.unregister(active['pidfd'])
                os.close(active['pidfd'])
                self.finish(identifier, active['entry'])


def exclusive_worker(job_path):
    import fcntl

    job = read(job_path)
    node, run = job['node'], job['run']
    if node['resource_mode'] != 'exclusive' or file_hash(job['cpu_dispatch_receipt']) != job['cpu_dispatch_sha256']:
        raise ValueError('Exclusive worker lacks unchanged CPU admission or GPU ownership declaration')
    registered_inputs(dispatch_namespace(node, run))
    receipt = read(job['cpu_dispatch_receipt'])
    if receipt['status'] != 'preflight_passed' or receipt['gpu_worker_started']:
        raise ValueError('Exclusive worker CPU preflight was not the preserved check-only dispatch')
    locks = ROOT / 'outputs/locks'
    locks.mkdir(parents=True, exist_ok=True)
    for offset, card in enumerate(sorted(node['cards'])):
        descriptor = os.open(locks / f'gpu_{card}.lock', os.O_WRONLY | os.O_CREAT, 0o600)
        target = 20 + offset
        os.dup2(descriptor, target, inheritable=True)
        if descriptor != target:
            os.close(descriptor)
        fcntl.flock(target, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = environment()
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, node['cards'])), KDM_GPU_SLOTS='')
    for key, relative in {
        'HF_HOME': 'cache/hf', 'TORCH_HOME': 'cache/torch', 'XDG_CACHE_HOME': 'cache/xdg',
        'MPLCONFIGDIR': 'cache/mpl', 'TRITON_CACHE_DIR': 'cache/triton',
        'TORCHINDUCTOR_CACHE_DIR': 'cache/inductor', 'TORCH_EXTENSIONS_DIR': 'cache/torch_extensions',
        'CUDA_CACHE_PATH': 'cache/cuda',
    }.items():
        env[key] = str(ROOT / relative)
    env['HF_HUB_DISABLE_XET'] = '1'
    print(json.dumps({'event': 'exclusive_gpu_locks_acquired', 'cards': node['cards'],
                      'claim_id': node['claim'], 'when': now()}), flush=True)
    command = [node['python'], str(ROOT / 'workflows/supplemental/remaining11/generate.py'),
               *generation_arguments(node, run), '--execute']
    os.chdir(ROOT)
    os.execvpe(node['python'], command, env)


def audit_locks():
    import fcntl

    registry, host, details = host_registration()
    evidence = []
    for card in details['allowed_gpus']:
        path = ROOT / 'outputs/locks' / f'gpu_{card}.lock'
        if not path.is_file():
            evidence.append({'card': card, 'path': str(path), 'status': 'lock_file_absent'})
            continue
        descriptor = os.open(path, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            status = 'exclusive_lock_blocked_by_existing_owner'
        else:
            status = 'exclusive_lock_available'
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        evidence.append({'card': card, 'path': str(path), 'status': status})
    print(json.dumps({'host': host, 'observed_utc': now(), 'kernel_flock_evidence': evidence,
                      'file_contents_changed': False, 'gpu_queries': 0, 'gpu_worker_started': False}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'check-node', 'execute', 'exclusive-worker', 'audit-locks'))
    parser.add_argument('--plan', type=Path, default=ROOT / 'workflows/supplemental/remaining11/plan.json')
    parser.add_argument('--host')
    parser.add_argument('--queue-id')
    parser.add_argument('--job-file', type=Path)
    args = parser.parse_args()
    if args.action == 'audit-locks':
        audit_locks()
        return
    if args.action == 'check-node':
        if args.job_file is None:
            raise ValueError('Isolated CPU check requires its explicit saved node')
        job = read(args.job_file)
        print(json.dumps(preflight_node(job['node'], job['run'], Path(job['folder']))))
        return
    if args.action == 'exclusive-worker':
        if args.job_file is None:
            raise ValueError('Exclusive worker requires its real saved CPU preflight')
        exclusive_worker(args.job_file)
        return
    if not hasattr(os, 'pidfd_open'):
        raise ValueError('Native kernel pidfd event support is required')
    registry, host, details = host_registration()
    if args.host is not None and args.host != host:
        raise ValueError('Requested event-queue host differs from the actual host')
    if not args.queue_id or not NAME.fullmatch(args.queue_id):
        raise ValueError('A finite queue requires an exclusive queue identifier')
    plan_path = args.plan.resolve()
    plan = read(plan_path)
    validate_plan(plan)
    for node in plan['nodes']:
        if node['host'] not in registry['hosts']:
            raise ValueError('Queue node has an unregistered execution host')
    folder = ROOT / 'outputs/supplemental/remaining11' / plan['run'] / 'queues' / host / args.queue_id
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / 'plan.json', plan)
    queue = EventQueue(plan, plan_path, host, folder)
    queue.check()
    if args.action == 'execute':
        import fcntl

        lock = folder.parent / 'event_runner.lock'
        with lock.open('a+') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue.execute()
    print(json.dumps({'queue_state': str(folder / 'queue_state.json'),
                      'host': host, 'status': queue.state['status'],
                      'candidate_states': {key: value['status'] for key, value in queue.state['nodes'].items()},
                      'gpu_queries': queue.state['gpu_queries'], 'timers_created': 0,
                      'research_complete': False}, indent=2))


if __name__ == '__main__':
    main()
