#!/usr/bin/env python3
"""Explicit same-model no-noise M3ID production under its actual software gate identity."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from kdm.io import file_hash, within
from kdm.protocol import validate_environment
from workflows.supplemental.remaining11 import execution
from workflows.supplemental.remaining4 import native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--model', choices=('onevision', 'qwen3vl'), required=True)
    parser.add_argument('--method', choices=('m3id',), required=True)
    parser.add_argument('--dataset', choices=('vizwiz',), required=True)
    parser.add_argument('--missing-keys', required=True)
    parser.add_argument('--key-start', type=int, default=0)
    parser.add_argument('--key-stop', type=int)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--claim-id', required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--chunk-rows', type=int, default=512)
    parser.add_argument('--host-registry', required=True)
    parser.add_argument('--software-gate', required=True)
    parser.add_argument('--retirement-receipt', required=True)
    parser.add_argument('--plan-output')
    args = parser.parse_args()
    gate_path = within(ROOT, args.software_gate)
    gate = json.loads(gate_path.read_text())
    retirement_path = within(ROOT, args.retirement_receipt)
    retirement = json.loads(retirement_path.read_text())
    audit_path = within(ROOT, gate['actual_operator_audit_path'])
    actual_audit = json.loads(audit_path.read_text())
    if (not gate['passed'] or gate['differences'] or not gate['production_allowed']
            or file_hash(audit_path) != gate['actual_operator_audit_sha256']
            or actual_audit['model'] != args.model or actual_audit['method'] != args.method
            or gate['software_compatibility']['actual_method_scope'] != 'm3id'
            or retirement['model'] != args.model
            or not retirement['future_nodes_removed'] or not retirement['zero_m3id_gpu_claim_and_raw']
            or retirement['retired_future_method'] != 'm3id' or retirement['retired_rows'] != 3501):
        raise ValueError('Actual M3ID software/zero-claim ownership gate is absent')
    execution.REGISTRY = str(within(ROOT, args.host_registry).relative_to(ROOT))
    original_spec = execution.runtime_spec
    def actual_spec(root, frozen_spec, model):
        if model != args.model:
            raise ValueError('Compatibility identity cannot admit a different model')
        value = original_spec(root, frozen_spec, model)
        value['versions'] = copy.deepcopy(frozen_spec['versions'])
        value['versions'].update(torch='2.9.0+cu128', torchvision='0.24.0+cu128')
        return value
    execution.runtime_spec = actual_spec
    plan = native.load_plan(args)
    actual = execution.runtime_spec(ROOT, plan['spec'], args.model)
    environment = validate_environment(actual)
    checkpoint = execution.checkpoint_identity(actual)
    provenance = {**gate['software_compatibility'], 'actual_software_gate_path': args.software_gate,
                  'actual_software_gate_sha256': file_hash(gate_path),
                  'native_pending_queue_retirement_path': args.retirement_receipt,
                  'native_pending_queue_retirement_sha256': file_hash(retirement_path),
                  'production_method_scope': 'native_unguided/m3id only',
                  'compatible_entrypoint_sha256': file_hash(Path(__file__)),
                  'noise_based_methods_admitted': False}
    if args.check_plan:
        receipt = {'plan': plan['summary'], 'actual_environment': environment,
                   'checkpoint': checkpoint, 'software_compatibility': provenance,
                   'gpu_admission': 'not_run'}
        if args.plan_output:
            native.write_new_json(within(ROOT, args.plan_output), receipt)
        print(json.dumps(receipt), flush=True)
        return
    original_admission = execution.validate_supplemental_runtime
    def admission(*values, **kwargs):
        result = original_admission(*values, **kwargs)
        result['software_compatibility'] = provenance
        return result
    execution.validate_supplemental_runtime = admission
    os.environ['KDM_SUPPLEMENTAL_METHOD'] = 'm3id'
    native.execute(args)
    import torch
    record = within(ROOT, 'outputs/supplemental/remaining4/' + args.run_name) / 'records' / args.model / args.method / args.claim_id
    native.write_new_json(record / 'actual_software_capacity.json', {
        'model': args.model, 'dataset': args.dataset, 'method': args.method,
        'software_compatibility': provenance,
        'actual_cuda_capability': list(torch.cuda.get_device_capability(0)),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
        'registered_map_precision_and_parameters_unchanged': True,
        'same_process_completed_generation': True,
    })


if __name__ == '__main__':
    main()
