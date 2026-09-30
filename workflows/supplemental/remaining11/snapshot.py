"""Capture this run's receipts and verify explicitly transferred immutable parts."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash, stable_seed, within

NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}')
FIELDS = ('method', 'marker', 'reference_marker', 'guided', 'reference_guided', 'replicate', 'kind')


def now():
    return datetime.now(timezone.utc).isoformat()


def run_folder(run):
    if not NAME.fullmatch(run):
        raise ValueError('Invalid supplemental run identifier')
    return ROOT / 'outputs/supplemental/remaining11' / run


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def process_evidence(pid):
    folder = Path('/proc') / str(pid)
    if not folder.exists():
        return {'pid': pid, 'present': False, 'command': None}
    command = (folder / 'cmdline').read_bytes().replace(b'\x00', b' ').decode('utf-8')
    return {'pid': pid, 'present': True, 'command': command.strip()}


def local_host():
    registry = read(ROOT / 'workflows/supplemental/remaining11/host_registry.json')
    matches = [name for name, details in registry['hosts'].items()
               if details['hostname'] == socket.gethostname() and Path(details['root']).resolve() == ROOT]
    if len(matches) != 1:
        raise ValueError('Snapshot hostname/project root is unregistered')
    return matches[0]


def part_entry(complete_path, host):
    receipt = read(complete_path)
    raw = within(ROOT, receipt['raw_path'])
    identity_path = raw.with_suffix('.identity.json')
    meta = read(identity_path)
    if meta['identity'] != stable_hash(meta['definition']):
        raise ValueError('Completed part identity definition is invalid')
    if file_hash(identity_path) != receipt['identity_sha256']:
        raise ValueError('Completed part identity sidecar changed')
    if not raw.is_file() or receipt['rows'] < 1:
        raise ValueError('Completed receipt lacks its actual immutable raw part')
    if receipt.get('generation_complete') is not True and receipt.get('part_validation_complete') is not True:
        raise ValueError('Part receipt does not establish complete immutable rows')
    relative = raw.relative_to(ROOT).as_posix()
    return {
        'part_id': stable_hash([host, relative, receipt['raw_sha256']]),
        'host': host, 'model': receipt['model'], 'dataset': receipt['dataset'],
        'stage': receipt['stage'], 'claim_id': receipt['claim_id'], 'part': receipt.get('part', 0),
        'rows': receipt['rows'], 'raw_relative_path': relative,
        'raw_source_path': str(raw), 'raw_sha256': receipt['raw_sha256'],
        'identity_relative_path': identity_path.relative_to(ROOT).as_posix(),
        'identity_source_path': str(identity_path), 'identity_sha256': receipt['identity_sha256'],
        'identity': meta['identity'], 'complete_relative_path': complete_path.relative_to(ROOT).as_posix(),
        'complete_source_path': str(complete_path), 'complete_sha256': file_hash(complete_path),
        'source_scope': receipt.get('complete_scope', 'completed_generation_part'),
        'claimed_stage_complete': receipt.get('claimed_stage_complete', False),
        'finished_utc': receipt['finished_utc'],
        'stage_counts': receipt.get('stage_counts', {}),
    }


def recent_part_measurements(parts):
    """Read only the latest three immutable parts for observed method throughput."""
    measurements = []
    for part in sorted(parts, key=lambda value: value['part'])[-3:]:
        raw = Path(part['raw_source_path'])
        methods, rows, tokens = Counter(), 0, 0
        for row in read_jsonl(raw):
            methods[(row.get('method', 'candidate'), row.get('kind', 'candidate'))] += 1
            tokens += len(row.get('tokens', []))
            rows += 1
        if rows != part['rows'] or file_hash(raw) != part['raw_sha256']:
            raise ValueError('Recent immutable part differs from its preserved receipt')
        measurements.append({
            'part': part['part'], 'rows': rows, 'finished_utc': part['finished_utc'],
            'complete_source_path': part['complete_source_path'],
            'complete_sha256': part['complete_sha256'], 'raw_sha256': part['raw_sha256'],
            'method_counts': [{'method': method, 'kind': kind, 'rows': count}
                              for (method, kind), count in sorted(methods.items())],
            'generated_tokens': tokens,
        })
    return measurements


def build_snapshot(run, host=None):
    host = host or local_host()
    if host != local_host():
        raise ValueError('Snapshot host name differs from actual execution host')
    folder = run_folder(run)
    jobs = []
    for path in sorted((folder / 'records').rglob('progress.json')):
        progress = read(path)
        admission_path = path.parent / 'admission.json'
        admission = read(admission_path)
        failure_path = path.parent / 'failure.json'
        claim = folder / 'claims' / progress['model'] / progress['dataset'] / progress['stage'] / progress['claim_id']
        release = None
        if (claim / 'released.json').is_file():
            from workflows.supplemental.remaining11.dispatch import validate_import_release

            marker = read(claim / 'released.json')
            owner = read(claim / 'owner.json')
            release = validate_import_release(claim, marker['replacement_claim_id'], owner['plan'],
                                              marker['source_evidence']['cards'], owner['owner'])
        jobs.append({
            'host': host, 'model': progress['model'], 'dataset': progress['dataset'],
            'stage': progress['stage'], 'claim_id': progress['claim_id'],
            'progress_path': str(path), 'progress': progress,
            'process': process_evidence(progress['pid']),
            'admission_path': str(admission_path), 'admission_sha256': file_hash(admission_path),
            'admission': {
                'schema': admission['schema'], 'model': admission['model'], 'dataset': admission['dataset'],
                'stage': admission['stage'], 'claim_id': admission['claim_id'], 'owner': admission['owner'],
                'backend_sha256': stable_hash(admission['backend']),
                'registered_backend_sha256': stable_hash(admission['registered_backend']),
                'runtime_admission': admission['runtime_admission'],
                'source_provenance': admission['source_provenance'], 'runner_sha256': admission['runner_sha256'],
                'expected_rows': admission['task_plan']['expected_generation_rows'],
                'ordered_keys_sha256': admission['task_plan']['ordered_selected_keys_sha256'],
            },
            'failure_path': str(failure_path) if failure_path.exists() else None,
            'failure': read(failure_path) if failure_path.exists() else None,
            'ownership_release': release,
        })
    completed = [part_entry(path, host) for path in sorted((folder / 'records').rglob('*.complete.json'))]
    completed.extend(part_entry(path, host) for path in sorted((folder / 'sealed_failed').rglob('*.complete.json')))
    for job in jobs:
        parts = [part for part in completed if part['claim_id'] == job['claim_id']
                 and part['source_scope'] == 'completed_generation_part']
        job['recent_part_measurements'] = recent_part_measurements(parts)
    return {
        'schema': 'kdm_remaining11_host_snapshot_v1', 'run': run, 'host': host,
        'project_root': str(ROOT), 'hostname': socket.gethostname(), 'captured_utc': now(),
        'jobs': jobs, 'completed_parts': completed, 'only_completed_parts_exported': True,
        'gpu_queries_performed': 0,
    }


def validate_raw(raw, meta, model, stage, expected_rows, sample_map):
    from kdm.pipeline import task_id
    from kdm.prompts import task_prompt

    if stable_hash(meta['definition']) != meta['identity']:
        raise ValueError('Raw sidecar identity is invalid')
    definition = meta['definition']
    spec = read(ROOT / f'configs/runtime/{model}.json')
    if definition['registered_backend'] != spec or definition['model'] != model or definition['stage'] != stage:
        raise ValueError('Raw part differs from original registered model identity or stage')
    keys = []
    for row in read_jsonl(raw):
        sample = row['sample']
        if (row['model'] != model or sample != sample_map.get(sample['id'])
                or row['identity'] != meta['identity'] or row['status'] != 'ok'):
            raise ValueError('Raw row has an unknown sample, model, identity or status')
        if stage == 'candidate':
            key = stable_hash([model, sample['id'], 'closed'])
            names, scores = definition['names'], row['candidate_scores']
            if (len(names) != 101 or len(scores) != 101 or {value['label'] for value in scores} != set(names)
                    or row['target'] != sample['class'] or row['ranking_rule'] != 'mean_log_probability'):
                raise ValueError('Candidate ranking does not retain all original canonical classes')
            if any(not math.isfinite(value[field]) or value['n_tokens'] < 1
                   for value in scores for field in ('sum_logp', 'mean_logp')):
                raise ValueError('Candidate ranking has invalid probability evidence')
            target = next(value for value in scores if value['label'] == row['target'])
            if row['gold_rank'] != 1 + sum(value['mean_logp'] > target['mean_logp'] for value in scores):
                raise ValueError('Candidate rank differs from the registered comparison rule')
            if row['seed'] != stable_seed(sample['id'], model, 0):
                raise ValueError('Candidate seed differs from the original sample/model seed')
        else:
            task = {field: row[field] for field in FIELDS}
            task['sample'] = sample
            if 'attempt' in row:
                task['attempt'] = row['attempt']
            key = task_id(model, task)
            if row['prompt'] != task_prompt(sample['question'], row['marker'], row['guided'], row.get('attempt', False)):
                raise ValueError('Raw prompt differs from the registered task prompt')
            config = dict(definition['base_config'])
            config['method'] = row['method']
            if row['method'] in ('m3id', 'instruction_m3id'):
                config['m3id_offset'] = len(row['offset_prompt_tokens'])
            if row['config'] != config or row['config']['max_tokens'] != 32:
                raise ValueError('Raw decoding parameters differ from the admitted configuration')
            tokens = row['tokens']
            if (not isinstance(row['terminated'], bool) or not 1 <= len(tokens) <= 32
                    or len(tokens) != len(row['selected_log_probabilities'])
                    or not all(math.isfinite(value) for value in row['selected_log_probabilities'])):
                raise ValueError('Raw token or probability coverage is invalid')
            if row['seed'] != stable_seed(sample['id'], model, row['replicate']):
                raise ValueError('Raw stable sample/model/replicate seed differs')
        if row['key'] != key:
            raise ValueError('Raw key differs from the original task-key function')
        keys.append(key)
    if len(keys) != expected_rows or len(keys) != len(set(keys)):
        raise ValueError('Raw rows have duplicates or incomplete receipt coverage')
    return keys


def receive_locked(run, incoming):
    folder = run_folder(run)
    incoming = within(folder, incoming)
    snapshot = read(incoming / 'host_snapshot.json')
    if snapshot['schema'] != 'kdm_remaining11_host_snapshot_v1' or snapshot['run'] != run:
        raise ValueError('Incoming snapshot belongs to another run or schema')
    host = snapshot['host']
    if not NAME.fullmatch(host):
        raise ValueError('Invalid incoming source host')
    destination = folder / 'remote_completed' / host
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'remote_completed/received_manifest.json'
    previous = read(manifest_path) if manifest_path.exists() else {'parts': []}
    parts = {entry['part_id']: entry for entry in previous['parts']}
    sample_map = {sample['id']: sample for sample in read_jsonl(ROOT / 'data/current/all.jsonl')}
    transfer = read(incoming / 'transfer_manifest.json')
    if transfer['run'] != run or transfer['host'] != host:
        raise ValueError('Transfer selection belongs to another host or run')
    selected_ids = set(transfer['part_ids'])
    entries = [entry for entry in snapshot['completed_parts'] if entry['part_id'] in selected_ids]
    if len(entries) != len(selected_ids):
        raise ValueError('Transfer selection contains missing or duplicate snapshot parts')
    received_now = []
    for entry in entries:
        if entry['part_id'] in parts:
            continue
        paths = {}
        for kind in ('raw', 'identity', 'complete'):
            relative = PurePosixPath(entry[kind + '_relative_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Incoming part path escapes its source project')
            path = incoming / 'files' / Path(*relative.parts)
            if file_hash(path) != entry[kind + '_sha256']:
                raise ValueError('Transferred ' + kind + ' SHA differs from its immutable source')
            paths[kind] = path
        meta = read(paths['identity'])
        if meta['identity'] != entry['identity']:
            raise ValueError('Transferred raw identity differs from the source snapshot')
        receipt = read(paths['complete'])
        if receipt['rows'] != entry['rows'] or receipt['raw_sha256'] != entry['raw_sha256']:
            raise ValueError('Transferred completion receipt disagrees with the source map')
        keys = validate_raw(paths['raw'], meta, entry['model'], entry['stage'], entry['rows'], sample_map)
        known_keys = set()
        for prior in parts.values():
            if (prior['model'], prior['stage']) == (entry['model'], entry['stage']):
                known_keys.update(read(within(ROOT, prior['keys_path'])))
        if known_keys.intersection(keys):
            raise ValueError('Incoming raw part overlaps previously received model/stage keys')
        target = destination / 'parts' / entry['part_id']
        target.mkdir(parents=True, exist_ok=False)
        final = {}
        for kind, path in paths.items():
            final[kind] = target / path.name
            path.rename(final[kind])
        atomic_json(target / 'keys.json', keys)
        source_map = {
            **entry, 'path': final['raw'].relative_to(ROOT).as_posix(),
            'identity_path': final['identity'].relative_to(ROOT).as_posix(),
            'complete_path': final['complete'].relative_to(ROOT).as_posix(),
            'keys_path': (target / 'keys.json').relative_to(ROOT).as_posix(),
            'received_raw_path': final['raw'].relative_to(ROOT).as_posix(),
            'received_receipt_path': (target / 'receive_receipt.json').relative_to(ROOT).as_posix(),
            'received_utc': now(), 'receive_validation_complete': True,
            'status': 'selected_complete', 'cohort': 'remaining11_native_supplement',
            'labels_complete': False, 'reference_complete': False,
        }
        atomic_json(target / 'receive_receipt.json', source_map)
        parts[entry['part_id']] = source_map
        received_now.append(source_map)
        values = sorted(parts.values(), key=lambda item: (item['host'], item['model'], item['stage'], item['claim_id'], item['part']))
        manifest = {
            'schema': 'kdm_remaining11_received_immutable_parts_v1', 'run': run, 'updated_utc': now(),
            'parts': values, 'sources': values,
            'chosen_sources': [item for item in values if item['stage'] in ('formal', 'independent')],
            'candidate_sources': [item for item in values if item['stage'] == 'candidate'],
            'rows': sum(item['rows'] for item in values),
        }
        atomic_json(manifest_path, manifest)
    saved_snapshot = destination / 'snapshots' / incoming.name / 'host_snapshot.json'
    atomic_json(saved_snapshot, snapshot)
    print(json.dumps({'host': host, 'received_parts': len(received_now),
                      'received_rows': sum(item['rows'] for item in received_now),
                      'manifest': str(manifest_path), 'total_received_rows': sum(item['rows'] for item in parts.values())}))


def receive(run, incoming):
    import fcntl

    folder = run_folder(run) / 'remote_completed'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'receive.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        receive_locked(run, incoming)
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def seal_failed(run, claim_id, seal_id):
    if not NAME.fullmatch(claim_id) or not NAME.fullmatch(seal_id):
        raise ValueError('Invalid explicit failed-claim seal identifier')
    folder = run_folder(run)
    matches = [path for path in (folder / 'records').rglob('progress.json') if path.parent.name == claim_id]
    if len(matches) != 1:
        raise ValueError('Failed claim is not uniquely registered')
    progress = read(matches[0])
    if progress['status'] != 'failed' or process_evidence(progress['pid'])['present']:
        raise ValueError('Only a stopped, explicitly failed claim can be sealed')
    failure_path = matches[0].parent / 'failure.json'
    failure = read(failure_path)
    if list((folder / 'sealed_failed' / claim_id).glob('*/*.complete.json')):
        raise ValueError('This failed claim already has an immutable sealed prefix')
    raw = within(ROOT, failure['current_raw_path'])
    identity = raw.with_suffix('.identity.json')
    signature = (raw.stat().st_size, raw.stat().st_mtime_ns)
    rows = list(read_jsonl(raw))
    meta = read(identity)
    sample_map = {sample['id']: sample for sample in read_jsonl(ROOT / 'data/current/all.jsonl')}
    keys = validate_raw(raw, meta, progress['model'], progress['stage'], len(rows), sample_map)
    if not keys or not set(keys) <= set(failure['current_part_expected_keys']):
        raise ValueError('Failed prefix has empty or unadmitted successful keys')
    if signature != (raw.stat().st_size, raw.stat().st_mtime_ns):
        raise ValueError('Failed source changed during prefix validation')
    target = folder / 'sealed_failed' / claim_id / seal_id
    target.mkdir(parents=True, exist_ok=False)
    copied_raw, copied_identity = target / raw.name, target / identity.name
    shutil.copyfile(raw, copied_raw)
    shutil.copyfile(identity, copied_identity)
    receipt = {
        'rows': len(keys), 'model': progress['model'], 'dataset': progress['dataset'],
        'stage': progress['stage'], 'claim_id': claim_id, 'part': meta['definition']['part'],
        'raw_path': str(copied_raw.relative_to(ROOT)), 'raw_sha256': file_hash(copied_raw),
        'identity_sha256': file_hash(copied_identity), 'finished_utc': now(),
        'source_raw_path': str(raw), 'source_raw_sha256': file_hash(raw),
        'source_identity_path': str(identity), 'source_failure_path': str(failure_path),
        'source_failure_sha256': file_hash(failure_path),
        'complete_scope': 'validated_successful_prefix_of_failed_claim',
        'part_validation_complete': True, 'generation_complete': False,
        'claimed_stage_complete': False, 'original_claim_failed': True,
        'retry_performed': False, 'labels_complete': False, 'research_complete': False,
    }
    if receipt['raw_sha256'] != receipt['source_raw_sha256']:
        raise ValueError('Sealed prefix differs from its preserved original source')
    atomic_json(target / (raw.stem + '.complete.json'), receipt)
    atomic_json(target / 'source_map.json', receipt)
    print(json.dumps({'sealed_rows': len(keys), 'claim_id': claim_id, 'receipt': str(target / 'source_map.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('capture', 'receive', 'known', 'seal-failed'))
    parser.add_argument('--run', default='run_20260930_140337')
    parser.add_argument('--host')
    parser.add_argument('--snapshot-id')
    parser.add_argument('--incoming', type=Path)
    parser.add_argument('--claim-id')
    parser.add_argument('--seal-id')
    args = parser.parse_args()
    if args.action == 'capture':
        if not args.snapshot_id or not NAME.fullmatch(args.snapshot_id):
            raise ValueError('Capture requires an exclusive snapshot identifier')
        out = run_folder(args.run) / 'snapshots' / args.snapshot_id
        out.mkdir(parents=True, exist_ok=False)
        snapshot = build_snapshot(args.run, args.host)
        atomic_json(out / 'host_snapshot.json', snapshot)
        print(json.dumps({'host': snapshot['host'], 'snapshot_path': str(out / 'host_snapshot.json'),
                          'completed_parts': len(snapshot['completed_parts']),
                          'completed_rows': sum(item['rows'] for item in snapshot['completed_parts'])}))
    elif args.action == 'receive':
        if args.incoming is None:
            raise ValueError('Receive requires the explicit incoming directory')
        receive(args.run, args.incoming)
    elif args.action == 'seal-failed':
        if not args.claim_id or not args.seal_id:
            raise ValueError('Sealing requires an explicit failed claim and exclusive seal identifier')
        seal_failed(args.run, args.claim_id, args.seal_id)
    else:
        path = run_folder(args.run) / 'remote_completed/received_manifest.json'
        manifest = read(path) if path.exists() else {'parts': []}
        print(json.dumps({'part_ids': [item['part_id'] for item in manifest['parts']]}))


if __name__ == '__main__':
    main()
