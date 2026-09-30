#!/usr/bin/env python3
"""Route selected-four original Food reference work through its explicit host registry."""
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

MODELS = ('internvl35_8b', 'onevision', 'phi35', 'qwen3vl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--stage', choices=('independent', 'candidate'), required=True)
    parser.add_argument('--missing-keys', required=True)
    parser.add_argument('--key-start', type=int, default=0)
    parser.add_argument('--key-stop', type=int)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    parser.add_argument('--run-name')
    parser.add_argument('--claim-id')
    parser.add_argument('--owner')
    parser.add_argument('--chunk-rows', type=int, default=512)
    parser.add_argument('--plan-output')
    parser.add_argument('--host-registry', default='workflows/supplemental/remaining11/host_registry.json')
    args = parser.parse_args()
    args.dataset = 'food101'
    args.continuation_receipt = None
    registry = within(ROOT, args.host_registry)
    if not registry.is_file():
        raise ValueError('Original-reference route requires an existing explicit host registry')
    execution.REGISTRY = str(registry.relative_to(ROOT))
    plan = generate.load_plan(args)
    if args.execute:
        if not all(generate.NAME_PATTERN.fullmatch(value or '') for value in
                   (args.run_name, args.claim_id, args.owner)) or args.chunk_rows < 1:
            parser.error('--execute requires a unique run/claim, owner, and positive chunk rows')
        # The original runner performs physical UUID, inherited-lock, native
        # proof, checkpoint, environment and unchanged parameter checks.
        generate.execute(args)
    else:
        if args.plan_output:
            output = within(ROOT, args.plan_output)
            if output.exists():
                raise ValueError('Explicit reference CPU output already exists')
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x', encoding='utf-8') as stream:
                json.dump(plan['summary'], stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
        print(json.dumps({'plan': plan['summary'], 'routing_registry': str(registry.relative_to(ROOT)),
                          'routing_registry_sha256': file_hash(registry),
                          'wrapper_sha256': file_hash(Path(__file__)),
                          'sampling_engine': 'original registered native runner',
                          'scientific_parameters_changed': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
