"""Scientific provenance validation helpers. No job-launching entry point."""

from __future__ import annotations

import hashlib

import json

from pathlib import Path

from kdm.io import file_hash, read_jsonl, stable_hash, within

SCHEMA = 'kdm_phi_candidate_admin_retirement_v1'

SCIENCE = ('model', 'dataset', 'stage', 'methods', 'base_config', 'source_provenance',
           'ordered_registered_keys_sha256', 'registered_full_stage_rows')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

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
