#!/usr/bin/env python3
"""Explicitly admitted prefill reuse; original candidate algorithm and runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from kdm.io import file_hash, within
from workflows.supplemental.remaining11 import execution, generate
from workflows.supplemental.remaining4.prefill_cache import PrefillCacheBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--model', choices=('internvl35_8b', 'phi35'), required=True)
    parser.add_argument('--missing-keys', required=True)
    parser.add_argument('--key-start', type=int, default=0)
    parser.add_argument('--key-stop', type=int)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    parser.add_argument('--run-name')
    parser.add_argument('--claim-id')
    parser.add_argument('--owner')
    parser.add_argument('--chunk-rows', type=int, default=64)
    parser.add_argument('--host-registry', required=True)
    parser.add_argument('--gate-receipt', required=True)
    parser.add_argument('--plan-output')
    args = parser.parse_args()
    args.dataset, args.stage, args.continuation_receipt = 'food101', 'candidate', None
    registry = within(ROOT, args.host_registry)
    execution.REGISTRY = str(registry.relative_to(ROOT))
    gate_path = within(ROOT, args.gate_receipt)
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    cache = Path(__file__).with_name('prefill_cache.py')
    if (gate.get('status') != 'pass' or gate.get('model') != args.model
            or gate.get('inputs') != 8 or gate.get('candidate_score_rows') != 808
            or gate.get('max_absolute_score_difference') != 0 or gate.get('rank_differences') != 0
            or gate.get('cache_sha256') != file_hash(cache)
            or gate.get('original_algorithm_sha256') != file_hash(ROOT / 'src/kdm/pipeline.py')
            or gate.get('registry_sha256') != file_hash(registry)):
        raise ValueError('Actual finite cache-concordance receipt does not match this runtime route')
    for name, field in (('admission.json', 'admission_sha256'),
                        ('comparisons.jsonl', 'comparisons_sha256')):
        if file_hash(gate_path.parent / name) != gate[field]:
            raise ValueError('Immutable gate source changed: ' + name)
    old_load, old_backend = generate.load_plan, generate.make_backend
    optimization = {'name': 'identical_clean_prompt_prefill_reuse', 'gate_path': args.gate_receipt,
                    'gate_sha256': file_hash(gate_path), 'cache_sha256': file_hash(cache),
                    'execution_entrypoint': str(Path(__file__).relative_to(ROOT)),
                    'execution_entrypoint_sha256': file_hash(Path(__file__)),
                    'original_runner_sha256': file_hash(Path(generate.__file__)),
                    'scientific_parameters_changed': False,
                    'original_candidate_token_forwards': True, 'independent_KV_clones': True}

    def explicit_plan(namespace):
        plan = old_load(namespace)
        plan['summary']['source_provenance']['compute_optimization'] = optimization
        return plan

    def explicit_backend(*a, **kw):
        return PrefillCacheBackend(old_backend(*a, **kw))

    # This isolated entry point changes only the candidate session implementation;
    # admissions, claims, task keys, original closed_rank and sealed proof checks
    # stay in the original runner.  The distinct entry point and gate are recorded.
    generate.load_plan, generate.make_backend = explicit_plan, explicit_backend
    try:
        if args.execute:
            if not all(generate.NAME_PATTERN.fullmatch(value or '') for value in
                       (args.run_name, args.claim_id, args.owner)) or args.chunk_rows < 1:
                parser.error('Unique run/claim, owner and positive chunk rows are required')
            generate.execute(args)
        else:
            plan = explicit_plan(args)
            if args.plan_output:
                output = within(ROOT, args.plan_output)
                if output.exists():
                    raise ValueError('CPU plan output already exists')
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open('x', encoding='utf-8') as stream:
                    json.dump(plan['summary'], stream, ensure_ascii=False, indent=2, allow_nan=False)
            print(json.dumps(plan['summary'], ensure_ascii=False, indent=2))
    finally:
        generate.load_plan, generate.make_backend = old_load, old_backend


if __name__ == '__main__':
    main()
