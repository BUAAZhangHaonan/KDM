"""Explicit administrative retirement of one sealed Phi ranking claim.

This only changes key ownership after a confirmed input-boundary stop. It never
retries a scientific failure or changes the registered candidate computation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from kdm.io import file_hash, read_jsonl, stable_hash, within

SCHEMA = 'kdm_phi_candidate_admin_retirement_v1'
SCIENCE = ('model', 'dataset', 'stage', 'methods', 'base_config', 'source_provenance',
           'ordered_registered_keys_sha256', 'registered_full_stage_rows')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def ordered_sha(keys):
    return hashlib.sha256(''.join(key + '\n' for key in keys).encode()).hexdigest()


def science(plan):
    return {field: plan[field] for field in SCIENCE}


def validate_retirement(root, path, replacement, plan, selected_keys, cards, owner):
    from workflows.supplemental.remaining11.execution import host_registration

    root = Path(root).resolve()
    path = within(root, path)
    marker = read(path)
    if (marker['schema'] != SCHEMA or marker['scientific_failure']
            or marker['automatic_retry_performed'] or marker['scientific_parameters_changed']
            or marker['full_original_claim_complete']):
        raise ValueError('Continuation is outside explicit sealed administrative Phi ranking retirement')
    _registry, host, _details = host_registration(root)
    for name, digest in marker['bundle_files_sha256'].items():
        if file_hash(path.parent / name) != digest:
            raise ValueError('Immutable retirement source bundle changed: ' + name)
    original = read(path.parent / 'source/owner.json')
    stopped = read(path.parent / 'source/stopped_receipt.json')
    seal = read(path.parent / 'source/seal_receipt.json')
    if (original['claim_id'] != marker['source_claim_id']
            or original['plan'] != read(path.parent / 'source/admission.json')['task_plan']
            or stopped['pidfd_exit_event_observed'] is not True
            or stopped['administratively_stopped'] is not True
            or seal['scientific_failure'] or seal['source_originals_modified']
            or science(plan) != marker['scientific_plan']):
        raise ValueError('Original stop, source admission, or scientific configuration differs')
    candidates = [item for item in marker['replacements'] if item['claim_id'] == replacement]
    if len(candidates) != 1:
        raise ValueError('Retirement has no unique explicit replacement claim')
    target = candidates[0]
    keys = [row['key'] for row in read_jsonl(path.parent / target['keys_path'])]
    if (target['host'] != host or target['cards'] != list(map(str, cards))
            or target['owner'] != owner or set(keys) != selected_keys
            or target['plan'] != plan or ordered_sha(keys) != plan['ordered_selected_keys_sha256']
            or len(keys) != plan['expected_generation_rows']):
        raise ValueError('Retirement replacement changes its exact host, cards, owner, or key interval')
    return {'schema': SCHEMA, 'source_claim_id': marker['source_claim_id'],
            'retirement_path': str(path.relative_to(root)), 'retirement_sha256': file_hash(path),
            'source_owner_sha256': marker['bundle_files_sha256']['source/owner.json'],
            'source_keys_sha256': marker['bundle_files_sha256']['source/keys.jsonl'],
            'source_completed_rows': seal['total_generated_complete_rows'],
            'replacement_claim_id': replacement, 'replacement_rows': len(keys),
            'replacement_keys_sha256': marker['bundle_files_sha256'][target['keys_path']],
            'scientific_parameters_changed': False, 'automatic_retry_performed': False}


def prepare(args):
    import fcntl
    from workflows.supplemental.remaining11.generate import load_plan, selected_tasks, generation_key

    root = ROOT
    stop_dir = within(root, args.stop_dir)
    stopped = read(stop_dir / 'stopped_receipt.json')
    seal = read(stop_dir / 'seal_receipt.json')
    if (stopped['project_root'] != str(root) or stopped['model'] != 'phi35'
            or stopped['dataset'] != 'food101' or stopped['stage'] != 'candidate'
            or not stopped['pidfd_exit_event_observed'] or not stopped['administratively_stopped']
            or (Path('/proc') / str(stopped['pid'])).exists() or seal['scientific_failure']
            or seal['source_originals_modified'] or seal['full_claim_generation_complete']):
        raise ValueError('Only an actually stopped, sealed incomplete Phi Food ranking claim can retire')
    claim = Path(stopped['owner_path']).parent
    owner = read(claim / 'owner.json')
    admission_path = Path(stopped['admission_path'])
    if (file_hash(claim / 'owner.json') != stopped['owner_sha256']
            or file_hash(claim / 'keys.jsonl') != stopped['keys_sha256']
            or file_hash(admission_path) != stopped['admission_sha256']
            or owner['plan'] != read(admission_path)['task_plan']):
        raise ValueError('Original generating identity or ownership changed after the confirmed stop')
    source_keys = [row['key'] for row in read_jsonl(claim / 'keys.jsonl')]
    completed_keys = []
    for part in [*seal['complete_parts'], seal['sealed_partial']]:
        if part is None:
            continue
        raw = root / part['raw_path']
        if file_hash(raw) != part['raw_sha256']:
            raise ValueError('An immutable completed source part changed after validation')
        completed_keys.extend(row['key'] for row in read_jsonl(raw))
    remainder = Path(seal['remaining_keys_path'])
    remaining_keys = [row['key'] for row in read_jsonl(remainder)]
    if (completed_keys + remaining_keys != source_keys or len(set(source_keys)) != len(source_keys)
            or len(completed_keys) != seal['total_generated_complete_rows']
            or file_hash(remainder) != seal['remaining_keys_sha256']):
        raise ValueError('Remaining keys are not the exact owned keys minus real immutable completed keys')
    output = within(root, args.output)
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    source.mkdir()
    for name, path in (('owner.json', claim / 'owner.json'), ('keys.jsonl', claim / 'keys.jsonl'),
                       ('admission.json', admission_path), ('stopped_receipt.json', stop_dir / 'stopped_receipt.json'),
                       ('seal_receipt.json', stop_dir / 'seal_receipt.json')):
        shutil.copy2(path, source / name)
    shutil.copy2(remainder, output / 'missing_keys.jsonl')
    replacements = read(within(root, args.replacements))
    jobs, union = [], set()
    for job in replacements:
        request = SimpleNamespace(model='phi35', dataset='food101', stage='candidate',
                                  missing_keys=str(output / 'missing_keys.jsonl'), key_start=0, key_stop=None,
                                  shard=job['shard'], n_shards=job['n_shards'])
        plan = load_plan(request, root)
        if science(plan['summary']) != science(owner['plan']):
            raise ValueError('Replacement changes the original scientific candidate plan')
        keys = [generation_key('phi35', 'candidate', task) for task in selected_tasks(plan)]
        if union.intersection(keys) or job['claim_id'] in {item['claim_id'] for item in jobs}:
            raise ValueError('Replacement claims overlap or repeat an identifier')
        union.update(keys)
        key_path = output / (job['claim_id'] + '.keys.jsonl')
        with key_path.open('x', encoding='utf-8') as stream:
            for key in keys:
                stream.write(json.dumps({'key': key}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        jobs.append({**job, 'keys_path': key_path.name, 'plan': plan['summary']})
    if union != set(remaining_keys):
        raise ValueError('Explicit replacement jobs do not exactly partition all remaining keys')
    marker = {'schema': SCHEMA, 'retired_utc': datetime.now(timezone.utc).isoformat(),
              'reason': 'Human authorized sealed Phi candidate parallel continuation at original parameters',
              'source_claim_id': owner['claim_id'], 'source_host': stopped['host'],
              'source_original_paths': {'owner': str(claim / 'owner.json'), 'keys': str(claim / 'keys.jsonl'),
                                        'stop_dir': str(stop_dir)},
              'source_completed_rows': len(completed_keys), 'remaining_rows': len(remaining_keys),
              'scientific_plan': science(owner['plan']), 'replacements': jobs,
              'bundle_files_sha256': {str(path.relative_to(output)): file_hash(path)
                                     for path in sorted(output.rglob('*')) if path.is_file()},
              'scientific_failure': False, 'scientific_parameters_changed': False,
              'automatic_retry_performed': False, 'full_original_claim_complete': False}
    save(output / 'retirement.json', marker)
    with (claim.parent / 'ownership.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        marker_name = 'retired.json' if not (claim / 'retired.json').exists() else 'retired_' + file_hash(output / 'retirement.json')[:16] + '.json'
        save(claim / marker_name, {'schema': SCHEMA, 'retirement_path': str((output / 'retirement.json').relative_to(root)),
                                   'retirement_sha256': file_hash(output / 'retirement.json'),
                                   'source_owner_sha256': stopped['owner_sha256'],
                                   'source_keys_sha256': stopped['keys_sha256'],
                                   'replacement_claim_ids': [item['claim_id'] for item in jobs]})
    print(json.dumps({'retirement': str(output / 'retirement.json'), 'completed': len(completed_keys),
                      'remaining': len(remaining_keys), 'replacement_rows': {
                          job['claim_id']: job['plan']['expected_generation_rows'] for job in jobs}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--replacements', required=True)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
