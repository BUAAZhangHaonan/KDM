"""Verify an explicitly released, real zero-load failure without claiming keys."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from kdm.io import atomic_json, stable_hash, within
from workflows.supplemental.remaining11.dispatch import input_failure_recovery, validate_import_release
from workflows.supplemental.remaining11.generate import load_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--replacement', required=True)
    parser.add_argument('--cards', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run', default='run_20260930_140337')
    args = parser.parse_args()
    run = ROOT / 'outputs/supplemental/remaining11' / args.run
    claim = run / 'claims' / args.model / 'food101/formal' / args.source
    original = json.loads((claim / 'owner.json').read_text())
    summary = original['plan']
    plan = load_plan(argparse.Namespace(
        model=args.model, dataset='food101', stage='formal',
        missing_keys=summary['missing_keys_path'], shard=summary['shard'],
        n_shards=summary['n_shards'], key_start=summary['key_start'], key_stop=summary['key_stop'],
    ))
    assert plan['summary'] == summary
    assert len(plan['selected_keys']) == summary['expected_generation_rows']
    released = validate_import_release(claim, args.replacement, plan['summary'],
                                       args.cards.split(','), original['owner'])
    branch = input_failure_recovery(argparse.Namespace(
        recovery_from=args.source, claim=args.replacement, model=args.model,
        dataset='food101', stage='formal', shard=summary['shard'], n_shards=summary['n_shards'],
        cards=args.cards, owner=original['owner'], run=args.run,
    ), run / 'dispatch')
    assert branch == {'source_failure': 'zero_model_load_standard_library_queue_shadowing', **released}
    output = within(ROOT, args.output)
    if output.exists():
        raise FileExistsError('Actual import-recovery audit output already exists')
    receipt = {
        'schema': 'kdm_remaining11_import_recovery_cpu_audit_v1', 'source_claim_id': args.source,
        'replacement_claim_id': args.replacement, 'model': args.model, 'cards': args.cards.split(','),
        'fresh_full_cpu_plan_equal': True, 'full_selected_keys': len(plan['selected_keys']),
        'full_plan_sha256': stable_hash(plan['summary']), 'explicit_recovery_branch_verified': True,
        'source_release': released, 'source_generated_rows': 0,
        'generation_performed': False, 'new_claim_created': False, 'gpu_worker_started': False,
    }
    atomic_json(output, receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
