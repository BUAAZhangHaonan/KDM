"""Scientific provenance validation helpers. No job-launching entry point."""

from __future__ import annotations

import sys

import hashlib

import json

from pathlib import Path

import re

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT / 'src'))

sys.path.insert(0, str(ROOT))

from kdm.io import atomic_json, file_hash, read_jsonl, stable_hash, within

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
