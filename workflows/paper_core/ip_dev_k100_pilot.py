#!/usr/bin/env python3
"""Authorized IP-only fixed-dev eight-input timing; no full generation dispatch."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from kdm.decoding import DecodeConfig
from kdm.frozen import load_contract
from kdm.io import file_hash, read_jsonl, stable_hash, stable_seed, within
from kdm.pipeline import make_backend, run_tasks, task_id
from kdm.prompts import MARKERS, task_prompt
from workflows.paper_core.dev_viz import roster, condition, PilotInputs, now
from workflows.supplemental.remaining11.generate import validate_proofs
from workflows.supplemental.remaining11 import execution as supplemental
from workflows.paper_core import execution as core

CORE = ('qwen25vl', 'qwen35_4b', 'llava16_mistral', 'minicpm26', 'gemma3_4b')
EXTRA = ('internvl35_8b', 'onevision', 'phi35', 'qwen3vl')
METHODS = ('instruction_vcd', 'instruction_m3id')

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def tasks_for(samples, methods, markers):
    return [dict(sample=s, method=m, marker=k, reference_marker=k, guided=True,
                 reference_guided=False, replicate=0, kind='instruction_preserving')
            for m in methods for k in markers for s in samples]

def audit(model, tasks, raw):
    expected = {task_id(model, t): t for t in tasks}
    rows = list(read_jsonl(raw))
    if len(rows) != len(expected) or {r['key'] for r in rows} != expected.keys():
        raise ValueError('Pilot keys are missing, foreign or duplicated')
    for row in rows:
        task = expected[row['key']]
        if row['status'] != 'ok' or not 1 <= len(row['tokens']) <= 32:
            raise ValueError('Actual token/status evidence failed')
        if row['sample'] != task['sample'] or row['seed'] != stable_seed(task['sample']['id'], model, 0):
            raise ValueError('Original sample/seed differs')
        for field in ('method', 'marker', 'reference_marker', 'guided', 'reference_guided', 'replicate', 'kind'):
            if row[field] != task[field]:
                raise ValueError('Original task field differs: ' + field)
        for field, marker, guided in [('prompt', task['marker'], True),
                                     ('reference_prompt', task['reference_marker'], False),
                                     ('neutral_prompt', task['marker'], False)]:
            if row[field] != task_prompt(task['sample']['question'], marker, guided):
                raise ValueError('Original branch prompt differs: ' + field)
        cfg = asdict(DecodeConfig(method=task['method']))
        if task['method'] == 'instruction_m3id':
            cfg['m3id_offset'] = len(row['offset_prompt_tokens'])
        if row['config'] != cfg or not isinstance(row['terminated'], bool):
            raise ValueError('Original decode configuration differs')
        if len(row['selected_log_probabilities']) != len(row['tokens']) or not all(
                math.isfinite(p) for p in row['selected_log_probabilities']):
            raise ValueError('Actual finite token-probability evidence failed')
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=CORE + EXTRA, required=True)
    parser.add_argument('--methods', nargs='+', choices=METHODS, required=True)
    parser.add_argument('--markers', nargs='+', choices=MARKERS, default=list(MARKERS))
    parser.add_argument('--mode', choices=('plan', 'pilot'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--claim-id', required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--physical-gpus')
    parser.add_argument('--host-registry', required=True)
    parser.add_argument('--original-cpu-fixture', required=True)
    parser.add_argument('--software-gates', nargs='+', required=True)
    args = parser.parse_args()
    import re
    if not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', v) for v in
               (args.run_id, args.claim_id, args.owner)):
        raise ValueError('Invalid exclusive identity')
    if len(set(args.methods)) != len(args.methods) or len(set(args.markers)) != len(args.markers):
        raise ValueError('Repeated method/marker')
    if args.model in CORE and args.methods != ['instruction_m3id']:
        raise ValueError('Core IP-VCD already complete; only missing IP-M3ID authorized')
    if args.model != 'onevision' or args.methods != ['instruction_m3id'] or set(args.markers) - {'UNKNOWN','UNCLEAR','UNSURE'}:
        raise ValueError('K100 scope only independently passed OneVision IP-M3ID Food markers')
    fixture_path = within(ROOT, args.original_cpu_fixture)
    fixture = json.loads(fixture_path.read_text())
    if fixture['original_cuda_initialized'] or fixture['model'] != 'onevision' or fixture['split'] != 'dev':
        raise ValueError('Original dev CPU fixture absent')
    for field, relative in [('original_processor_source_sha256','src/kdm/models/backbone.py'),
                            ('original_text_adapter_source_sha256','src/kdm/models/hf.py'),
                            ('original_operator_source_sha256','workflows/supplemental/remaining4/ip_m3id_compatibility.py')]:
        if fixture[field] != file_hash(ROOT/relative): raise ValueError('Original CPU/operator source changed')
    software_gates = {}
    for relative in args.software_gates:
        path = within(ROOT, relative)
        gate = json.loads(path.read_text())
        scope = gate['software_compatibility']
        operator_path = within(ROOT, gate['operator_audit_path'])
        if not gate['passed'] or not gate['production_allowed'] or gate['differences'] or gate['completed_inputs'] != 8 or scope['model'] != 'onevision' or scope['actual_method_scope'] != 'instruction_m3id' or scope['dataset_scope'] != 'food101' or file_hash(operator_path) != gate['operator_audit_sha256']:
            raise ValueError('Existing marker-bound actual software gate failed')
        software_gates[scope['marker_scope'][0]] = {'path':relative,'sha256':file_hash(path),'gate':gate}
    if set(software_gates) != set(args.markers): raise ValueError('Exact successful marker scope required')
    original_runtime = supplemental.runtime_spec
    def runtime(root, frozen, model):
        import copy
        actual = original_runtime(root, frozen, model)
        actual['versions'] = copy.deepcopy(frozen['versions'])
        actual['versions'].update(torch='2.9.0+cu128',torchvision='0.24.0+cu128')
        return actual
    supplemental.runtime_spec = runtime
    roster_path, samples = roster('dev404')
    if fixture['roster_sha256'] != file_hash(roster_path):raise ValueError('Original fixed roster changed')
    full_tasks = tasks_for(samples, args.methods, args.markers)
    tasks = tasks_for(samples[:8], args.methods, args.markers)
    spec = json.loads((ROOT / f'configs/runtime/{args.model}.json').read_text())
    manifest, freeze = load_contract(ROOT)
    proofs = validate_proofs(ROOT, spec, args.model, args.methods, 'formal', manifest, freeze)
    # Supplemental source_inputs additionally checks the pinned algorithm blobs.
    frozen_source = None
    if args.model in EXTRA:
        from workflows.supplemental.remaining4.native import source_inputs
        _, frozen_spec, _, frozen_source = source_inputs(args.model, 'm3id', 'food101')
        if frozen_spec != spec:
            raise ValueError('Original native runtime differs')
    keys = [task_id(args.model, t) for t in tasks]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate pilot scientific keys')
    plan = dict(schema='kdm_ip_only_dev404_pilot_plan_v1', model=args.model,
                stage='dev404', methods=args.methods, markers=args.markers,
                authorized_inputs_per_condition=8, actual_planned_rows=len(tasks),
                total_missing_registered_dev_rows=len(full_tasks), full_budget_released=False,
                fixed_roster_sha256=file_hash(roster_path), sample_ids=[s['id'] for s in samples[:8]],
                pilot_keys=keys, frozen_spec=spec, proofs=proofs, frozen_native_source=frozen_source,
                runner_sha256=file_hash(Path(__file__)), host_registry=args.host_registry,
                registry_sha256=file_hash(within(ROOT, args.host_registry)),
                inherited_marker_software_gates=software_gates, original_CPU_fixture_sha256=file_hash(fixture_path),
                original_dev_GPU_same_answer_comparison_available=False, noise_methods_admitted=False,
                claim_id=args.claim_id, owner=args.owner, scientific_parameters_changed=False)
    base = ROOT / 'outputs/paper_core_20261002_dev_viz' / args.run_id / args.model
    if args.mode == 'plan':
        write(base / 'cpu_plan.json', plan)
        print(json.dumps({k: plan[k] for k in ('model','actual_planned_rows','total_missing_registered_dev_rows','full_budget_released')}))
        return
    if not args.physical_gpus:
        raise ValueError('Explicit physical cards required')
    cards = args.physical_gpus.split(',')
    if any(int(v.split(':')[1]) >= 2 for v in os.environ.get('KDM_GPU_SLOTS', '').split(',') if v):
        raise ValueError('Current A100 authorization is limited to slots zero and one')
    base.mkdir(parents=True, exist_ok=True)
    lock = (base / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(base / 'claimed_pilot.json', plan)
    started = time.perf_counter()
    try:
        os.environ['KDM_SUPPLEMENTAL_DATASET'] = 'food101'
        if args.model in EXTRA:
            supplemental.REGISTRY = args.host_registry
            admission = supplemental.validate_supplemental_runtime(ROOT, spec, args.model, cards, 'formal', args.claim_id, args.owner)
            actual_spec = admission['runtime_spec']
        else:
            core.REGISTRY = args.host_registry
            admission = core.admit(ROOT, spec, args.model, cards)
            actual_spec = core.runtime_spec(ROOT, spec, args.model)
        identity = dict(plan=plan, actual_admission=admission, runtime_spec=actual_spec,
                        executor={'agent':'/root/tail_scheduler','model':'unknown','effort':'unknown'},
                        claimed_at_utc=now(), pid=os.getpid())
        write(base / 'identity.json', identity)
        backend = make_backend(actual_spec, 'cuda:0')
        load_wall = time.perf_counter() - started
        original_backend = backend
        groups = defaultdict(list)
        for task in tasks:
            groups[condition(task)].append(task)
        timing = []
        for cid, group in groups.items():
            raw = base / 'sealed_pilot' / f'{group[0]["method"]}_{group[0]["marker"]}_{cid}.jsonl'
            if raw.exists():
                raise FileExistsError('No implicit repeat or recovery of pilot keys')
            from workflows.supplemental.remaining4.ip_m3id_compatibility import IPObserver
            expected = fixture['conditions'][group[0]['marker']]
            if {t['sample']['id'] for t in group} != {r['sample_id'] for r in expected}:
                raise ValueError('Original CPU eight inputs differ')
            import kdm.execution
            for record in expected:
                if file_hash(kdm.execution.resolve_image_path(record['image_source_path'],ROOT)) != record['original_image_sha256']:
                    raise ValueError('Original dev image bytes changed')
            backend = IPObserver(original_backend, group, DecodeConfig(), expected)
            before = time.perf_counter()
            for task in group:
                backend.current = task
                run_tasks(backend, args.model, [task], raw, identity, DecodeConfig())
            rows = audit(args.model, group, raw)
            operator = backend.proof(rows)
            write(raw.with_suffix('.operator.json'), operator)
            detail = dict(method=group[0]['method'], marker=group[0]['marker'], rows=len(rows),
                          output_tokens=sum(len(r['tokens']) for r in rows),
                          generation_wall_s=sum(r['wall_s'] for r in rows), execution_wall_s=time.perf_counter()-before,
                          raw=str(raw.relative_to(ROOT)), raw_sha256=file_hash(raw),
                          operator_audit_sha256=file_hash(raw.with_suffix('.operator.json')), three_original_routes_per_input=True,
                          completed_keys=[r['key'] for r in rows], passed=True, completed_utc=now(), scope='pilot_only')
            write(raw.with_suffix('.complete.json'), detail)
            timing.append(detail)
        elapsed = time.perf_counter()-started
        write(base / 'pilot_receipt.json', dict(passed=True, plan_sha256=stable_hash(plan),
              actual_rows=sum(v['rows'] for v in timing), conditions=timing, model_load_wall_s=load_wall,
              elapsed_s=elapsed, gpu_count=len(cards), gpu_seconds_including_load=elapsed*len(cards),
              full_budget_released=False, no_parameter_fallback=True, completed_utc=now()))
        print(json.dumps({'passed':True,'model':args.model,'actual_rows':len(tasks),'elapsed_s':elapsed}), flush=True)
    except BaseException as error:
        write(base / 'failure.json', dict(error=type(error).__name__+': '+str(error),
              traceback=traceback.format_exc(), elapsed_s=time.perf_counter()-started,
              automatic_retry=False, parameters_changed=False, failed_utc=now()))
        raise

if __name__ == '__main__':
    main()
