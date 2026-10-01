"""Exact original-task OneVision IP-M3ID compatibility and its explicitly admitted missing keys."""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from kdm.decoding import step_distribution
from kdm.io import file_hash, read_jsonl, within
from kdm.probability import log_normalize
from kdm.prompts import task_prompt
from kdm.protocol import validate_environment
from workflows.paper_core.native_audit import input_description
from workflows.supplemental.remaining11 import execution, generate
from workflows.supplemental.remaining4 import registered

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

class IPObserver:
    def __init__(self, backend, tasks, cfg, expected):
        self.backend, self.tasks, self.cfg = backend, tasks, replace(cfg, method='instruction_m3id')
        self.expected = {row['sample_id']: row for row in expected}
        self.records, self.pending = [], {}
        self.session_count = 0
    def __getattr__(self, name):
        return getattr(self.backend, name)
    def session(self, image, prompt, reference='clean', seed=0, need_layers=False):
        index, role = divmod(self.session_count, 3)
        self.session_count += 1
        task = self.tasks[index]
        sample = task['sample']
        expected = self.expected[sample['id']]
        name = ('main', 'reference', 'neutral')[role]
        expected_prompt = expected[('main_prompt', 'reference_prompt', 'neutral_prompt')[role]]
        if prompt != expected_prompt or need_layers or reference != ('text_only' if role == 1 else 'clean'):
            raise ValueError('Actual IP-M3ID route or canonical prompt differs')
        if role == 1 and seed != expected['seed']:
            raise ValueError('Actual IP text-only seed differs')
        session = self.backend.session(image, prompt, reference=reference, seed=seed, need_layers=False)
        inputs = input_description(session.inputs)
        field = ('clean_inputs', 'reference_inputs', 'neutral_inputs')[role]
        ids = session.inputs['input_ids'].detach().cpu().tolist()
        ids_field = ('main_prompt_token_ids', 'reference_prompt_token_ids', 'neutral_prompt_token_ids')[role]
        if inputs != expected[field] or ids != expected[ids_field]:
            raise ValueError('Actual original-environment processor tensors or token IDs differ: ' + name)
        if role == 0:
            self.records.append({'sample_id': sample['id'], **expected, 'step_checks': []})
        return IPSession(self, session, index, role)
    def proof(self, rows):
        if len(rows) != len(self.tasks) or self.session_count != 3 * len(rows) or self.pending:
            raise ValueError('Actual IP operator coverage is incomplete')
        for record, row in zip(self.records, rows):
            if row['sample']['id'] != record['sample_id'] or len(row['tokens']) != len(record['step_checks']):
                raise ValueError('Actual IP operator does not cover the saved answer tokens')
            if row['offset_prompt_tokens'] != record['offset_prompt_tokens']:
                raise ValueError('IP-M3ID offset differs from the actual neutral prompt')
            for check, token, trace in zip(record['step_checks'], row['tokens'], row['trace']):
                if token != check['argmax'] or trace['weight'] != check['weight'] or trace['active'] != check['active']:
                    raise ValueError('Saved IP token or trace differs from the original step operator')
            record.update(tokens=row['tokens'], text=row['text'], config=row['config'],
                          terminated=row['terminated'], wall_s=row['wall_s'])
        return {'model': 'onevision', 'method': 'instruction_m3id', 'passed': True,
                'completed': len(rows), 'records': self.records,
                'three_original_routes_per_input': True, 'noise_called': False,
                'gate_generation_wall_s': sum(row['wall_s'] for row in rows)}

class IPSession:
    def __init__(self, owner, session, index, role):
        self.owner, self.session, self.index, self.role = owner, session, index, role
    def next(self, prefix):
        step = self.session.next(prefix)
        if not np.isfinite(step.logits).all():
            raise ValueError('Actual IP full-vocabulary logits are not finite')
        key = self.index, tuple(prefix)
        values = self.owner.pending.setdefault(key, {})
        values[self.role] = step
        if self.role == 2:
            if set(values) != {0, 1, 2}:
                raise ValueError('IP three-route steps are not evaluated at the same actual prefix')
            cfg = replace(self.owner.cfg, m3id_offset=len(self.owner.records[self.index]['offset_prompt_tokens']))
            actual, meta = step_distribution(values[0], values[1], cfg, len(prefix), values[2])
            p, q, neutral = [log_normalize(values[index].logits) for index in (0, 1, 2)]
            active = bool(np.exp(neutral.max()) < cfg.m3id_threshold)
            weight = float(np.expm1(cfg.m3id_lambda * (len(prefix) + cfg.m3id_offset))) if active else 0.0
            official = log_normalize(p + weight * (neutral - q))
            if (not np.allclose(actual, official, atol=1e-8, rtol=1e-10)
                    or weight != meta['weight'] or active != meta['active']):
                raise ValueError('Actual IP-M3ID scores differ from the registered original formula')
            self.owner.records[self.index]['step_checks'].append({
                'argmax': int(np.argmax(actual)), 'active': active, 'weight': weight,
                'maximum_normalized_logp_error': float(np.max(np.abs(actual - official)))})
            del self.owner.pending[key]
        return step

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--phase', choices=('gate', 'production'), required=True)
    parser.add_argument('--marker', choices=generate.MARKERS, default='UNKNOWN')
    parser.add_argument('--source', required=True)
    parser.add_argument('--host-registry', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--claim-id', required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--software-gate')
    parser.add_argument('--ownership-receipt')
    parser.add_argument('--plan-output')
    args = parser.parse_args()
    args.model, args.dataset, args.stage = 'onevision', 'food101', 'formal'
    args.shard, args.n_shards, args.key_start, args.key_stop = 0, 1, 0, None
    args.chunk_rows = 8 if args.phase == 'gate' else 512
    args.continuation_receipt = None
    source_path = within(ROOT, args.source)
    source = json.loads(source_path.read_text())
    if (source['model'] != 'onevision' or source['dataset'] != args.dataset
            or source['method'] != 'instruction_m3id' or source['markers'] != [args.marker]
            or source['original_cuda_initialized']):
        raise ValueError('Narrow actual original source scope differs')
    if (source['original_processor_source_sha256'] != file_hash(ROOT / 'src/kdm/models/backbone.py')
            or source['original_text_adapter_source_sha256'] != file_hash(ROOT / 'src/kdm/models/hf.py')):
        raise ValueError('Actual original processor or text adapter source changed')
    field = 'keys' if args.phase == 'gate' else 'future_missing_keys'
    args.missing_keys = source[field + '_path']
    if file_hash(within(ROOT, args.missing_keys)) != source[field + '_sha256']:
        raise ValueError('Bound original or unclaimed task keys changed')
    execution.REGISTRY = str(within(ROOT, args.host_registry).relative_to(ROOT))
    original_spec = execution.runtime_spec
    def actual_spec(root, frozen, model):
        if model != 'onevision':
            raise ValueError('This software scope is OneVision only')
        actual = original_spec(root, frozen, model)
        actual['versions'] = copy.deepcopy(frozen['versions'])
        actual['versions'].update(torch='2.9.0+cu128', torchvision='0.24.0+cu128')
        return actual
    execution.runtime_spec = actual_spec
    plan = generate.load_plan(args)
    registered.retained(plan)
    tasks = list(generate.selected_tasks(plan))
    if not tasks or any(task['method'] != 'instruction_m3id' or task['marker'] != args.marker for task in tasks):
        raise ValueError('Every task must belong to the independently gated IP-M3ID marker')
    spec = execution.runtime_spec(ROOT, plan['spec'], args.model)
    environment = validate_environment(spec)
    checkpoint = execution.checkpoint_identity(spec)
    gate = None
    if args.phase == 'production':
        gate_path = within(ROOT, args.software_gate)
        gate = json.loads(gate_path.read_text())
        audit = within(ROOT, gate['operator_audit_path'])
        if (not gate['passed'] or gate['differences'] or not gate['production_allowed']
                or file_hash(audit) != gate['operator_audit_sha256']
                or gate['original_source_sha256'] != file_hash(source_path)
                or gate['software_compatibility']['marker_scope'] != [args.marker]
                or gate['software_compatibility']['entrypoint_sha256'] != file_hash(Path(__file__))
                or gate['software_compatibility']['registry_sha256'] != file_hash(within(ROOT, args.host_registry))):
            raise ValueError('Actual same-task IP-M3ID software proof is absent')
        args.retained_source_ownership = registered.source_ownership(args, plan)
    else:
        if len(tasks) != 8 or {task['sample']['id'] for task in tasks} != {row['sample_id'] for row in source['original_cpu_inputs']}:
            raise ValueError('Actual software gate must contain exactly the original eight distinct inputs')
        args.retained_source_ownership = {'software_comparison_only': True,
            'original_source_path': args.source, 'original_source_sha256': file_hash(source_path)}
    provenance = {'model': 'onevision', 'actual_method_scope': 'instruction_m3id',
        'dataset_scope': 'food101', 'marker_scope': [args.marker], 'original_source_sha256': file_hash(source_path),
        'original_torch': plan['spec']['versions']['torch'], 'actual_torch': spec['versions']['torch'],
        'transformers_and_other_registered_versions_unchanged': True, 'noise_based_methods_admitted': False,
        'entrypoint_sha256': file_hash(Path(__file__)),
        'registry_sha256': file_hash(within(ROOT, args.host_registry)), 'comparison_only': args.phase == 'gate'}
    if gate is not None:
        provenance['actual_software_gate_sha256'] = file_hash(gate_path)
    summary = {'plan': plan['summary'], 'actual_environment': environment, 'checkpoint': checkpoint,
               'software_compatibility': provenance, 'gpu_model_admission': 'not_run'}
    if args.plan_output:
        write(within(ROOT, args.plan_output), summary)
    if args.check_plan:
        print(json.dumps(summary), flush=True)
        return
    original_admission = execution.validate_supplemental_runtime
    def admitted(*values, **kwargs):
        result = original_admission(*values, **kwargs)
        result['software_compatibility'] = provenance
        return result
    execution.validate_supplemental_runtime = admitted
    os.environ['KDM_SUPPLEMENTAL_DATASET'] = 'food101'
    observers = []
    if args.phase == 'gate':
        original_backend = generate.make_backend
        def observed_backend(*values, **kwargs):
            observer = IPObserver(original_backend(*values, **kwargs), tasks, plan['cfg'], source['original_cpu_inputs'])
            observers.append(observer)
            return observer
        generate.make_backend = observed_backend
    out = within(ROOT, args.run_root) / 'compatibility'
    try:
        generate.execute(args)
        if args.phase == 'gate':
            record = within(ROOT, args.run_root) / 'records/onevision/food101/formal' / args.claim_id
            receipt = json.loads((record / 'formal_onevision_part_00000.complete.json').read_text())
            rows = list(read_jsonl(within(ROOT, receipt['raw_path'])))
            audit = observers[0].proof(rows)
            import torch
            audit['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
            audit['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
            write(record / 'ip_operator_audit.json', audit)
            previous = {item['row']['key']: item['row'] for item in source['original_rows']}
            fields = ('sample', 'method', 'kind', 'marker', 'reference_marker', 'guided',
                      'reference_guided', 'replicate', 'config', 'tokens', 'text', 'terminated', 'offset_prompt_tokens')
            differences = [{'key': row['key'], 'field': field} for row in rows for field in fields
                           if row.get(field) != previous[row['key']].get(field)]
            result = {'passed': not differences, 'production_allowed': not differences,
                'differences': differences, 'original_source_sha256': file_hash(source_path),
                'operator_audit_path': str((record / 'ip_operator_audit.json').relative_to(ROOT)),
                'operator_audit_sha256': file_hash(record / 'ip_operator_audit.json'),
                'software_compatibility': provenance, 'completed_inputs': len(rows),
                'wall_s_generation': audit['gate_generation_wall_s'],
                'output_tokens': sum(len(row['tokens']) for row in rows),
                'completed_utc': datetime.now(timezone.utc).isoformat()}
            if differences:
                write(out / 'failed_comparison.json', result)
                raise ValueError('Actual IP-M3ID software output differs from its original same-task source')
            write(out / 'success.json', result)
            print(json.dumps(result), flush=True)
    except BaseException as error:
        write(out / 'failure.json', {'error': type(error).__name__ + ': ' + str(error),
            'traceback': traceback.format_exc(), 'production_allowed': False, 'automatic_retry': False})
        raise

if __name__ == '__main__':
    main()
