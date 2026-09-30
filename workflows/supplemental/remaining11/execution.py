"""Supplemental host admission with unchanged frozen research specifications.

The original registry and native proofs retain their bytes. This separate
registry records the physical devices authorized for the current supplement.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from kdm.io import file_hash
from kdm.protocol import validate_environment, validate_local_checkpoint

REGISTRY = 'workflows/supplemental/remaining11/host_registry.json'


def host_registration(root):
    root = Path(root).resolve()
    registry = json.loads((root / REGISTRY).read_text())
    matches = [(key, value) for key, value in registry['hosts'].items()
               if value['hostname'] == socket.gethostname()
               and Path(value['root']) == root]
    if len(matches) != 1:
        raise ValueError('Unregistered supplemental hostname/project root')
    return registry, *matches[0]


def runtime_spec(root, frozen_spec, model):
    registry, host, _details = host_registration(root)
    if frozen_spec['key'] != model:
        raise ValueError('Model differs from frozen runtime specification')
    spec = copy.deepcopy(frozen_spec)
    override = registry.get('runtime_overrides', {}).get(host, {}).get(model)
    if override is None:
        if registry['model_hosts'][model] != host:
            raise ValueError('Cross-host execution requires an explicit runtime path registration')
    else:
        if set(override) != {'environment_python', 'model_path', 'source_evidence'}:
            raise ValueError('Only registered environment/checkpoint paths can move')
        spec['environment_python'] = override['environment_python']
        spec['kwargs']['model_path'] = override['model_path']
        spec['processor']['path'] = override['model_path']
    return spec


def checkpoint_identity(spec):
    validate_local_checkpoint(spec)
    folder = Path(spec['kwargs']['model_path'])
    observed = {}
    for filename, digest in spec['processor']['files'].items():
        path = folder / filename
        if file_hash(path) != digest:
            raise ValueError('Processor/tokenizer/template changed: ' + filename)
        observed[filename] = digest
    if file_hash(folder / 'config.json') != spec['model_config_sha256']:
        raise ValueError('Checkpoint model config differs from its registered identity')
    return {'model_path': str(folder), 'model_config_sha256': spec['model_config_sha256'],
            'processor_files': observed,
            'weight_files': [{'filename': item['filename'], 'size_bytes': item['size_bytes'],
                              'hub_revision': item.get('hub_revision'),
                              'hub_recorded_sha256': item.get('hub_recorded_sha256')}
                             for item in spec['weights']],
            'weight_verification': 'registered file size and inherited immutable identity'}


def validate_supplemental_runtime(root, spec, model, cards, stage, claim_id, owner):
    root = Path(root).resolve()
    registry, host, details = host_registration(root)
    cards = [str(card) for card in cards]
    if len(cards) != spec['gpu_count'] or len(cards) != len(set(cards)):
        raise ValueError('Physical GPU count differs from the unchanged model device map')
    if not set(cards) <= {str(card) for card in details['allowed_gpus']}:
        raise ValueError('Unauthorized physical GPU')
    for offset, card in enumerate(sorted(cards, key=int)):
        expected = root / 'outputs' / 'locks' / f'gpu_{card}.lock'
        if Path(os.readlink(f'/proc/self/fd/{20 + offset}')) != expected:
            raise ValueError('Missing inherited supplemental worker GPU lock')
    lines = subprocess.check_output([
        'nvidia-smi', '-i', ','.join(cards),
        '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'],
        text=True).splitlines()
    observed = {parts[0].strip(): {'uuid': parts[1].strip(),
                                 'free_mib': int(parts[2].strip())}
                for parts in (line.split(',') for line in lines)}
    if set(observed) != set(cards) or any(
            observed[card]['uuid'] != details['gpu_uuids'][card] for card in cards):
        raise ValueError('Physical GPU UUID changed')
    actual = runtime_spec(root, spec, model)
    environment = validate_environment(actual)
    checkpoint = checkpoint_identity(actual)
    # Activate only the independent image-path registry for this worker process.
    # Native/method proof validation runs against the original registry first.
    import kdm.execution
    kdm.execution.REGISTRY = REGISTRY
    receipt = {
        'host': host, 'hostname': details['hostname'], 'project_root': str(root),
        'physical_gpus': cards, 'gpu_observation_before_loading': observed,
        'registry_path': REGISTRY, 'registry_sha256': file_hash(root / REGISTRY),
        'admission_source_sha256': file_hash(Path(__file__)),
        'model': model, 'stage': stage, 'claim_id': claim_id, 'owner': owner,
        'environment': environment, 'checkpoint': checkpoint,
        'gpu_sharing_authorized': True,
        'gpu_worker_slots': os.environ.get('KDM_GPU_SLOTS', ''),
        'max_workers_per_gpu': details.get('max_workers_per_gpu', 1),
        'capacity_evidence': details.get('capacity_evidence'),
        'runtime_path_evidence': registry.get('runtime_overrides', {}).get(host, {}).get(model),
    }
    return {'execution': receipt, 'runtime_spec': actual}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT / f'configs/runtime/{args.model}.json').read_text())
    actual = runtime_spec(ROOT, spec, args.model)
    print(json.dumps({'runtime_spec': actual, 'environment': validate_environment(actual),
                      'checkpoint': checkpoint_identity(actual)}, indent=2))


if __name__ == '__main__':
    main()
