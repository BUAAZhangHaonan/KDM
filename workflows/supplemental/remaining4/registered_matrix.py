#!/usr/bin/env python3
"""Run only the user-restored registered VCD/M3ID crossed and ref-off gaps."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from kdm.io import atomic_json, file_hash, within
from workflows.supplemental.remaining11 import execution, generate

MODELS = ('internvl35_8b', 'onevision', 'phi35', 'qwen3vl')


def restored_scope(plan):
    counts = {'vcd_main': 0, 'm3id_main': 0, 'vcd_reference_instruction_removed': 0,
              'm3id_reference_instruction_removed': 0}
    for task in generate.selected_tasks(plan):
        if task['method'] not in {'vcd', 'm3id'} or not task['guided']:
            raise ValueError('The restored matrix scope contains a different registered method or main guidance')
        if task['marker'] not in generate.MARKERS or task['reference_marker'] not in generate.MARKERS:
            raise ValueError('A restored matrix marker differs from its original four registered expressions')
        if task['kind'] == 'main':
            if not task['reference_guided']:
                raise ValueError('A registered crossed main condition must retain both guided branches')
        elif task['kind'] == 'reference_instruction_removed':
            if task['reference_guided'] or task['reference_marker'] != task['marker']:
                raise ValueError('A registered ref-off condition must retain its diagonal marker and unguided reference')
        else:
            raise ValueError('This entry cannot add IP, native, SID matrices, or other methods')
        if task['replicate'] != 0 or task['sample']['split'] != 'eval':
            raise ValueError('The restored Food matrix requires registered eval inputs and replicate zero')
        counts[task['method'] + '_' + task['kind']] += 1
    if sum(counts.values()) != plan['summary']['expected_generation_rows']:
        raise ValueError('The restored scope differs from the registered missing-key plan')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--missing-keys', required=True)
    parser.add_argument('--key-start', type=int, default=0)
    parser.add_argument('--key-stop', type=int)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    parser.add_argument('--run-name', default='restored_matrix_20261003')
    parser.add_argument('--run-root', default='outputs/supplemental/remaining4/restored_matrix_20261003')
    parser.add_argument('--claim-id')
    parser.add_argument('--owner')
    parser.add_argument('--chunk-rows', type=int, default=512)
    parser.add_argument('--host-registry', required=True)
    parser.add_argument('--plan-output')
    parser.add_argument('--executor-agent')
    parser.add_argument('--executor-model')
    parser.add_argument('--executor-effort')
    args = parser.parse_args()
    args.dataset, args.stage, args.continuation_receipt = 'food101', 'formal', None
    registry = within(ROOT, args.host_registry)
    run = within(ROOT, args.run_root)
    run.relative_to(ROOT / 'outputs/supplemental/remaining4')
    if not registry.is_file():
        raise FileNotFoundError('The explicit host/runtime registry is missing')
    execution.REGISTRY = str(registry.relative_to(ROOT))
    plan = generate.load_plan(args)
    scope = restored_scope(plan)
    proof = {
        'schema': 'kdm_user_restored_registered_matrix_scope_v1',
        'model': args.model, 'dataset': args.dataset, 'stage': args.stage,
        'registered_plan': plan['summary'], 'scope_counts': scope,
        'host_registry_path': str(registry.relative_to(ROOT)), 'host_registry_sha256': file_hash(registry),
        'entrypoint_path': str(Path(__file__).relative_to(ROOT)), 'entrypoint_sha256': file_hash(Path(__file__)),
        'original_generate_path': str(Path(generate.__file__).relative_to(ROOT)),
        'original_generate_sha256': file_hash(Path(generate.__file__)),
        'authorization': 'User explicitly restored original Food VCD16/M3ID16/ref-off4+4 on 2026-10-03; original scope exclusion remains unchanged in registered.py',
        'old_retained_guard_modified': False, 'generation_algorithm_changed': False,
        'new_SID_matrix_allowed': False, 'shared_prefix_speedup_claimed': False,
        'actual_command': [sys.executable, *sys.argv],
        'actual_executor': {'agent': args.executor_agent, 'model': args.executor_model,
                            'effort': args.executor_effort, 'call_id': ''},
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
    }
    if args.execute:
        if not all(generate.NAME_PATTERN.fullmatch(value or '') for value in
                   (args.run_name, args.claim_id, args.owner)):
            parser.error('Execution requires explicit unique run/claim and owner names')
        if not all((args.executor_agent, args.executor_model, args.executor_effort)):
            parser.error('Execution requires the actual executor agent/model/effort')
        target = run / 'commands' / (args.claim_id + '.restored_scope.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError('The explicit restored-scope execution proof already exists')
        atomic_json(target, proof)
        os.environ['KDM_SUPPLEMENTAL_DATASET'] = args.dataset
        generate.execute(args)
    else:
        if args.plan_output:
            target = within(ROOT, args.plan_output)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError('The CPU plan output already exists')
            atomic_json(target, proof)
        print(json.dumps({'scope_counts': scope, 'expected_rows': plan['summary']['expected_generation_rows'],
                          'registered_plan_sha256': plan['summary']['ordered_selected_keys_sha256'],
                          'registry_sha256': file_hash(registry), 'no_GPU_execution': True}))


if __name__ == '__main__':
    main()
