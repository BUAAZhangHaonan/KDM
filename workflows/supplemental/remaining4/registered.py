#!/usr/bin/env python3
"""Run explicit retained selected-four keys through the unchanged registered algorithm."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, within
from workflows.supplemental.remaining11 import execution, generate

MODELS = ('internvl35_8b', 'onevision', 'phi35', 'qwen3vl')


def retained(plan):
    count, samples = 0, set()
    for task in generate.selected_tasks(plan):
        if task['method'] in {'vcd', 'm3id'}:
            raise ValueError('New plain VCD/M3ID grids are excluded; native NONE baselines use their separate completed entry')
        if task['method'] in {'instruction_vcd', 'instruction_m3id'}:
            if task['marker'] not in generate.MARKERS or task['reference_marker'] != task['marker']:
                raise ValueError('Retained IP execution requires each registered marker unchanged on both routes')
            if not task['guided'] or task['reference_guided'] or task['kind'] != 'instruction_preserving':
                raise ValueError('Retained IP must keep the original instruction_preserving task fields')
        count += 1
        samples.add(task['sample']['id'])
    if count != plan['summary']['expected_generation_rows']:
        raise ValueError('Explicit retained task count differs from its registered key plan')
    return {'rows': count, 'unique_inputs': len(samples), 'plain_guided_vcd_m3id_rows': 0,
            'ip_scope': 'all four registered diagonal IP conditions; original guided=True/reference_guided=False and neutral route'}


def source_ownership(args, plan):
    if args.stage != 'formal' or args.dataset != 'food101':
        return None
    if not args.ownership_receipt:
        raise ValueError('Food retained methods require explicit appended stopped-source ownership')
    receipt_path = within(ROOT, args.ownership_receipt)
    receipt = json.loads(receipt_path.read_text())
    capacity_migration = receipt['schema'] == 'kdm_selected4_retained_capacity_migration_ownership_v1'
    if (receipt['schema'] not in {'kdm_selected4_retained_admin_ownership_v1',
                                 'kdm_selected4_retained_capacity_migration_ownership_v1'}
            or any(receipt[field] != getattr(args, field) for field in ('model', 'dataset', 'stage'))
            or args.claim_id not in receipt['replacement_claim_ids']):
        raise ValueError('Administrative retained ownership does not bind this claim')
    allowed = within(ROOT, receipt['allowed_keys_path'])
    completed = within(ROOT, receipt['source_completed_keys_path'])
    if file_hash(allowed) != receipt['allowed_keys_sha256'] or file_hash(completed) != receipt['source_completed_keys_sha256']:
        raise ValueError('Immutable allowed/completed source keys changed')
    allowed_keys = {json.loads(line)['key'] for line in allowed.read_text().splitlines()}
    completed_keys = {json.loads(line)['key'] for line in completed.read_text().splitlines()}
    if not plan['selected_keys'] <= allowed_keys or plan['selected_keys'].intersection(completed_keys):
        raise ValueError('Replacement reruns completed original source keys or leaves explicit retained scope')
    for source in receipt['sources']:
        marker = within(ROOT, source['appended_marker_path'])
        owner = within(ROOT, source['owner_path'])
        keys = within(ROOT, source['keys_path'])
        if (file_hash(marker) != source['appended_marker_sha256']
                or file_hash(owner) != source['owner_sha256'] or file_hash(keys) != source['keys_sha256']):
            raise ValueError('Original stopped ownership or appended replacement marker changed')
        if capacity_migration:
            failure = within(ROOT, source['original_failure_path'])
            sealed = within(ROOT, source['failed_prefix_receipt_path'])
            if (file_hash(failure) != source['original_failure_sha256']
                    or file_hash(sealed) != source['failed_prefix_receipt_sha256']
                    or not receipt['original_failed_claim_remains_held']
                    or receipt['automatic_scientific_retry']
                    or receipt['authorized_recovery'] != 'new_host_capacity_first8_then_explicit_retained_missing_keys'):
                raise ValueError('Explicit capacity-migration authority or preserved failure/prefix differs')
            original_failure = json.loads(failure.read_text())
            prefix = json.loads(sealed.read_text())
            if (not original_failure['error'].startswith('OutOfMemoryError: CUDA out of memory')
                    or not prefix['original_claim_failed'] or prefix['retry_performed']
                    or prefix['rows'] != original_failure['completed_rows']):
                raise ValueError('Capacity migration does not bind the actual held original OOM and sealed successful prefix')
        original = json.loads(owner.read_text())['plan']
        if any(original[field] != plan['summary'][field] for field in
               ('model', 'dataset', 'stage', 'methods', 'base_config')):
            raise ValueError('Retained continuation changes original registered science')
    return {'receipt_path': str(receipt_path.relative_to(ROOT)), 'receipt_sha256': file_hash(receipt_path),
            'source_claim_ids': [s['claim_id'] for s in receipt['sources']],
            'source_completed_rows': len(completed_keys), 'scientific_parameters_changed': False,
            'capacity_migration': capacity_migration,
            'original_failed_claim_remains_held': capacity_migration,
            'cancelled_plain_guided_grids_restarted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--dataset', choices=('food101', 'vizwiz'), required=True)
    parser.add_argument('--stage', choices=('formal', 'independent'), required=True)
    parser.add_argument('--missing-keys', required=True)
    parser.add_argument('--key-start', type=int, default=0)
    parser.add_argument('--key-stop', type=int)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    parser.add_argument('--run-name', default='dispatch_20261001_1600')
    parser.add_argument('--run-root', default='outputs/supplemental/remaining4/dispatch_20261001_1600')
    parser.add_argument('--claim-id')
    parser.add_argument('--owner')
    parser.add_argument('--chunk-rows', type=int, default=512)
    parser.add_argument('--plan-output')
    parser.add_argument('--host-registry', required=True)
    parser.add_argument('--ownership-receipt')
    args = parser.parse_args()
    args.continuation_receipt = None
    selected = within(ROOT, args.host_registry)
    if not selected.is_file() or not args.run_root.startswith('outputs/supplemental/remaining4/'):
        raise ValueError('Explicit selected-four registry/output scope is missing')
    within(ROOT, args.run_root).relative_to(ROOT / 'outputs/supplemental/remaining4')
    execution.REGISTRY = str(selected.relative_to(ROOT))
    plan = generate.load_plan(args)
    scope = retained(plan)
    ownership = source_ownership(args, plan) if args.ownership_receipt else None
    if args.execute:
        if not all(generate.NAME_PATTERN.fullmatch(value or '') for value in
                   (args.run_name, args.claim_id, args.owner)):
            parser.error('Execution requires explicit unique run/claim and owner names')
        args.retained_source_ownership = ownership or source_ownership(args, plan)
        os.environ['KDM_SUPPLEMENTAL_DATASET'] = args.dataset
        generate.execute(args)
        if (args.retained_source_ownership or {}).get('capacity_migration') and scope['rows'] == 8:
            import torch
            record = within(ROOT, args.run_root) / 'records' / args.model / args.dataset / args.stage / args.claim_id
            admission_path, complete_path = record / 'admission.json', record / 'complete.json'
            complete = json.loads(complete_path.read_text())
            if not complete['generation_complete'] or complete['completed'] != 8:
                raise ValueError('Capacity receipt requires actual completion of the eight claimed inputs')
            output = record / 'actual_same_process_capacity.json'
            with output.open('x', encoding='utf-8') as stream:
                json.dump({'schema': 'kdm_selected4_same_process_capacity_v1',
                           'model': args.model, 'dataset': args.dataset, 'stage': args.stage,
                           'claim_id': args.claim_id, 'completed_inputs': 8,
                           'admission_path': str(admission_path.relative_to(ROOT)),
                           'admission_sha256': file_hash(admission_path),
                           'complete_path': str(complete_path.relative_to(ROOT)),
                           'complete_sha256': file_hash(complete_path),
                           'logical_gpus': [{'logical_gpu': index,
                                'peak_allocated_bytes': torch.cuda.max_memory_allocated(index),
                                'peak_reserved_bytes': torch.cuda.max_memory_reserved(index)}
                               for index in range(plan['spec']['gpu_count'])],
                           'peak_scope': 'same fresh process model loading plus eight actual retained-method inputs; not full-input peak admission',
                           'dtype_device_map_batch_parameters_unchanged': True,
                           'extra_forward_calls': 0}, stream, indent=2, allow_nan=False)
                stream.write('\n')
    else:
        receipt = {'plan': plan['summary'], 'scope': scope,
                   'run_root': args.run_root, 'registry': str(selected.relative_to(ROOT)),
                   'registry_sha256': file_hash(selected), 'entrypoint_sha256': file_hash(Path(__file__)),
                   'algorithm': 'original registered generate/run_tasks', 'parameters_changed': False}
        receipt['retained_source_ownership'] = ownership
        if args.plan_output:
            output = within(ROOT, args.plan_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x', encoding='utf-8') as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
        print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
