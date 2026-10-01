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
    candidate_capacity = None
    model_capacity = details.get('minimum_free_mib_by_model', {}).get(model)
    if model_capacity is not None and any(observed[card]['free_mib'] < model_capacity for card in cards):
        raise ValueError('Unchanged model load requires its registered per-card free-memory budget')
    if model == 'phi35' and stage == 'candidate' and os.environ.get('KDM_SUPPLEMENTAL_DATASET') == 'food101':
        candidate_capacity = {
            'minimum_free_mib': 16384,
            'source': '2026-09-30 original Phi candidate PID2877080 on 4029 GPU2 used 10750 MiB; unchanged 101-label serial rank runtime',
            'fixed_headroom_mib': 5634,
            'scope': 'one unchanged Phi Food candidate worker per authorized physical card',
            'new_gpu_peak_measured': False,
        }
        if len(cards) != 1 or any(observed[card]['free_mib'] < 16384 for card in cards):
            raise ValueError('Phi Food candidate load requires 16384 MiB free from original usage plus fixed headroom')
    slots = [slot.split(':') for slot in os.environ.get('KDM_GPU_SLOTS', '').split(',') if slot]
    if any(int(slot[1]) >= 2 for slot in slots) and host == '6403':
        dual_budget = details.get('dual_card_food_candidate_budgets_mib', {}).get(model)
        dual_formal_budget = details.get('dual_card_food_formal_budgets_mib', {}).get(model)
        if dual_budget is not None and stage == 'candidate' and len(cards) == spec['gpu_count'] == 2:
            budget = dual_budget
        elif (dual_formal_budget is not None and stage == 'formal'
              and model == 'internvl35_8b' and len(cards) == spec['gpu_count'] == 2):
            budget = dual_formal_budget
        else:
            budget = details.get('third_slot_food_budgets_mib', {}).get(model) if len(cards) == 1 else None
        if (budget is None
                or os.environ.get('KDM_SUPPLEMENTAL_DATASET') != 'food101'
                or any(observed[card]['free_mib'] < budget for card in cards)):
            raise ValueError('Third A100 slot requires the explicitly registered stage and model Food capacity budget')
    if host == 'k100' and os.environ.get('KDM_SUPPLEMENTAL_DATASET') == 'vizwiz':
        native_budget = details.get('vizwiz_native_m3id_budgets_mib', {}).get(model)
        ip_budget = details.get('vizwiz_ip_m3id_budgets_mib', {}).get(model)
        if stage == 'formal' and ip_budget is not None:
            marker = os.environ.get('KDM_VIZ_IP_MARKER')
            phase = os.environ.get('KDM_VIZ_IP_PHASE')
            source_path = (root / os.environ.get('KDM_VIZ_IP_SOURCE', '')).resolve()
            source_path.relative_to(root)
            if (model != 'onevision' or len(cards) != 1
                    or os.environ.get('KDM_SUPPLEMENTAL_METHOD') != 'instruction_m3id'
                    or marker not in ('UNKNOWN', 'UNCLEAR', 'UNSURE')
                    or marker not in details.get('vizwiz_ip_m3id_markers', [])
                    or phase not in ('gate', 'production')
                    or any(observed[card]['free_mib'] < ip_budget for card in cards)):
                raise ValueError('K100 Viz IP-M3ID requires its exact OneVision marker/phase and separate capacity budget')
            source = json.loads(source_path.read_text())
            previous_path = root / source['previous_Food_software_gate_path']
            previous = json.loads(previous_path.read_text())
            first_keys_path = root / source['first8_keys_path']
            if (source['schema'] != 'kdm_selected4_one_viz_ip_m3id_CPU_fixture_v1'
                    or source['model'] != model or source['dataset'] != 'vizwiz'
                    or source['method'] != 'instruction_m3id' or source['marker'] != marker
                    or source['original_cuda_initialized']
                    or file_hash(previous_path) != source['previous_Food_software_gate_sha256']
                    or not previous['passed'] or not previous['production_allowed'] or previous['differences']
                    or previous['software_compatibility']['marker_scope'] != [marker]
                    or previous['software_compatibility']['dataset_scope'] != 'food101'
                    or file_hash(first_keys_path) != source['first8_keys_sha256']
                    or len(first_keys_path.read_text().splitlines()) != 8):
                raise ValueError('K100 Viz IP-M3ID actual original-input/software source is absent')
            if phase == 'production':
                gate_path = (root / os.environ.get('KDM_VIZ_IP_GATE', '')).resolve()
                gate_path.relative_to(root)
                gate = json.loads(gate_path.read_text())
                audit_path = root / gate['operator_audit_path']
                audit = json.loads(audit_path.read_text())
                if (not gate['passed'] or not gate['production_allowed'] or gate['completed_inputs'] != 8
                        or gate['dataset'] != 'vizwiz' or gate['marker'] != marker
                        or gate['method'] != 'instruction_m3id'
                        or gate['original_CPU_source_sha256'] != file_hash(source_path)
                        or file_hash(audit_path) != gate['operator_audit_sha256']
                        or not audit['passed'] or audit['completed'] != 8
                        or not audit['three_original_routes_per_input'] or audit['noise_called']):
                    raise ValueError('K100 Viz IP-M3ID production requires this marker actual eight-input operator gate')
        elif stage == 'native_unguided' and os.environ.get('KDM_SUPPLEMENTAL_METHOD') == 'm3id' and native_budget is not None:
            software_gate = root / details['native_m3id_software_gate']
            actual_gate = json.loads(software_gate.read_text())
            actual_operator_path = root / actual_gate['actual_operator_audit_path']
            actual_operator = json.loads(actual_operator_path.read_text())
            if (model not in ('onevision', 'qwen3vl') or len(cards) != 1 or not actual_gate['passed']
                    or file_hash(actual_operator_path) != actual_gate['actual_operator_audit_sha256']
                    or actual_operator['model'] != model or actual_operator['method'] != 'm3id'
                    or actual_gate['differences'] or not actual_gate['production_allowed']
                    or actual_gate['software_compatibility']['actual_method_scope'] != 'm3id'
                    or any(observed[card]['free_mib'] < native_budget for card in cards)):
                raise ValueError('K100 native M3ID requires the actual no-noise software gate and its separate free-memory budget')
        else:
            budget = details.get('vizwiz_independent_budgets_mib', {}).get(model)
            if (budget is None or stage != 'independent' or len(cards) != 1
                    or any(observed[card]['free_mib'] < budget for card in cards)):
                raise ValueError('K100 VizWiz independent generation requires its registered free-memory budget')
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
        'candidate_capacity_guard': candidate_capacity,
        'registered_model_minimum_free_mib': model_capacity,
    }
    return {'execution': receipt, 'runtime_spec': actual}


def main():
    global REGISTRY
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--host-registry', default=REGISTRY)
    args = parser.parse_args()
    selected_registry = (ROOT / args.host_registry).resolve()
    REGISTRY = str(selected_registry.relative_to(ROOT))
    if not selected_registry.is_file():
        raise ValueError('Explicit runtime host registry is unavailable')
    spec = json.loads((ROOT / f'configs/runtime/{args.model}.json').read_text())
    actual = runtime_spec(ROOT, spec, args.model)
    print(json.dumps({'runtime_spec': actual, 'environment': validate_environment(actual),
                      'checkpoint': checkpoint_identity(actual)}, indent=2))


if __name__ == '__main__':
    main()
