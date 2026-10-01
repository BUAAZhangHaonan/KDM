"""Viz thin caller of byte-identical accepted vLLM independent Generators."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from kdm.io import Ledger, atomic_json, file_hash, read_jsonl, stable_hash, within
from workflows.supplemental.remaining11 import generate

def recovered_generator(model, route):
    base = ROOT / 'outputs/supplemental/remaining4/dispatch_20261001_1600/vllm_exact_source_scope_event_20261001_2000/original_source_snapshot'
    common = base / 'workflows/acceleration_v4/vllm_closed_v3.py'
    implementation = base / f'workflows/independent_k100_{model}_v1/independent.py'
    if (file_hash(common) != '91fd30a36e345a8b93e759aa5427fa94227d22b8ccf4a492b1f2ac0f7febe8b0'
            or file_hash(implementation) != route['implementation_sha256']):
        raise ValueError('Original accepted engine implementation bytes differ')
    sys.path.insert(0, str(common.parent))
    name = 'kdm_recovered_' + model + '_independent'
    spec = importlib.util.spec_from_file_location(name, implementation)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.closed.ROOT = ROOT
    return module, module.Generator(model)

def cast_tree(value, dtype):
    import torch
    if torch.is_tensor(value):
        return value.to(dtype=dtype) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: cast_tree(item, dtype) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cast_tree(item, dtype) for item in value]
    return value

def actual_worker_peaks(worker):
    import torch
    return {'pid': os.getpid(), 'logical_device': torch.cuda.current_device(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}

def memory_admission(registry_path):
    registry = json.loads(registry_path.read_text())
    details = registry['hosts']['k100']
    if (socket.gethostname() != details['hostname'] or str(ROOT) != details['root']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0'
            or Path(os.readlink('/proc/self/fd/20')) != ROOT / 'outputs/locks/gpu_0.lock'):
        raise ValueError('Original exclusive K100 GPU0 inherited lock is absent')
    values = subprocess.check_output(['nvidia-smi', '-i', '0',
        '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    observed = {'index': int(values[0]), 'uuid': values[1].strip(), 'free_mib': int(values[2])}
    if observed['uuid'] != details['gpu_uuids']['0'] or observed['free_mib'] < details['minimum_free_mib']:
        raise ValueError('Actual original engine load exceeds the registered exclusive free-memory guard')
    return {'host': 'k100', 'hostname': socket.gethostname(), 'project_root': str(ROOT),
            'physical_gpus': ['0'], 'gpu_worker_slots': '', 'max_workers_per_gpu': 1,
            'gpu_observation_before_loading': {'0': observed},
            'registry_sha256': file_hash(registry_path), 'minimum_free_mib': details['minimum_free_mib']}

def validate_input(module, scorer, reference_row, eos):
    from kdm.execution import resolve_image_path
    image, prompt, base, inputs = scorer.prepare(reference_row['sample'])
    if (file_hash(resolve_image_path(reference_row['sample']['image_path'], ROOT)) != reference_row['image_sha256']
            or prompt != reference_row['prompt'] or base != reference_row['unexpanded_prompt_ids']
            or inputs['input_ids'][0].tolist() != reference_row['expanded_prompt_ids']
            or module.closed.summarize(dict(inputs)) != reference_row['processor_summary']
            or scorer.proc.tokenizer.eos_token_id not in eos):
        raise ValueError('Actual Viz image, native prompt, tokenizer, processor inputs or EOS semantics differ')
    return image, prompt, base, inputs

def gate_input(module, scorer, reference_row, eos, image, base, inputs, attempts):
    from vllm import SamplingParams
    from vllm.multimodal.processing import ProcessorInputs, TimingContext
    processor = scorer.llm.llm_engine.input_processor.renderer.mm_processor
    model_config = scorer.llm.llm_engine.model_config
    engine_ids = module.engine_base_ids(scorer.model, base, reference_row['expanded_prompt_ids'],
                                        scorer.proc.tokenizer.bos_token_id)
    processed = processor.apply(ProcessorInputs(prompt=engine_ids,
        mm_data_items=processor.info.parse_mm_data({'image': image}),
        hf_processor_mm_kwargs={}, tokenization_kwargs={}), TimingContext(enabled=False))
    actual = module.closed.summarize(processed['mm_kwargs'].get_data(device='cpu', pin_memory=False))
    expected = module.closed.summarize(cast_tree(dict(inputs), model_config.dtype))
    required = {'llava_onevision': {'pixel_values', 'image_sizes'},
                'qwen3_vl': {'pixel_values', 'image_grid_thw'}}[scorer.mt]
    parity = {key: key in actual and key in expected and actual[key] == expected[key] for key in required}
    if not all(parity.values()) or processed['prompt_token_ids'] != reference_row['expanded_prompt_ids']:
        raise ValueError('Actual vLLM Viz multimodal processor tensors or expanded IDs differ')
    params = module.parameters(scorer.model, reference_row['sample']['id'], 0, eos)
    params['temperature'] = 0.0
    request = {'prompt_token_ids': engine_ids, 'multi_modal_data': {'image': image},
               'multi_modal_uuids': {'image': reference_row['sample']['id']}}
    started = time.perf_counter()
    result = scorer.llm.generate([request], SamplingParams(**params), use_tqdm=False)[0]
    output = result.outputs[0]
    tokens = list(output.token_ids)
    probabilities = [float(values[token].logprob) for token, values in zip(tokens, output.logprobs)]
    if (result.prompt_token_ids != reference_row['expanded_prompt_ids'] or not 1 <= len(tokens) <= 32
            or len(tokens) != len(probabilities) or not all(math.isfinite(value) for value in probabilities)
            or any(token in eos for token in tokens[:-1])
            or (tokens[-1] not in eos and len(tokens) != 32)):
        raise ValueError('Actual engine greedy comparator has invalid expansion, probabilities or native-EOS handling')
    common = 0
    for new, old in zip(tokens, reference_row['greedy_tokens']):
        if new != old:
            break
        common += 1
    cached = sum(row['num_cached_tokens'] for row in attempts)
    if cached <= 0 or len({row['seed'] for row in attempts}) != 10:
        raise ValueError('Actual first genuine request plus nine distinct-seed cached requests did not pass')
    return {'sample_id': reference_row['sample']['id'], 'sample': reference_row['sample'],
        'engine_greedy_tokens': tokens, 'native_greedy_tokens': reference_row['greedy_tokens'],
        'engine_greedy_text': scorer.proc.tokenizer.decode(tokens, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False),
        'native_greedy_text': reference_row['greedy_text'],
        'engine_greedy_selected_log_probabilities': probabilities,
        'native_greedy_selected_log_probabilities': reference_row['greedy_selected_log_probabilities'],
        'greedy_tokens_equal': tokens == reference_row['greedy_tokens'], 'matched_prefix_tokens': common,
        'matched_prefix_max_abs_logprob_error': max((abs(probabilities[index] -
            reference_row['greedy_selected_log_probabilities'][index]) for index in range(common)), default=None),
        'engine_processor_tensor_parity': parity, 'actual_engine_processor_tensors': actual,
        'engine_expanded_prompt_ids_equal': True, 'ten_distinct_stable_seeds': True,
        'native_eos_handling': True, 'finite_selected_logprobs': True, 'num_cached_tokens': cached,
        'ten_request_wall_s': attempts[0]['wall_s'], 'greedy_wall_s': time.perf_counter() - started,
        'sampling_parameters': module.parameters(scorer.model, reference_row['sample']['id'], 0, eos),
        'distribution_scope': 'raw selected-token log probabilities; no native random-draw equivalence'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--phase', choices=('gate', 'production'), required=True)
    parser.add_argument('--model', choices=('onevision', 'qwen3vl'), required=True)
    parser.add_argument('--missing-keys', required=True)
    parser.add_argument('--native-reference', required=True)
    parser.add_argument('--route-evidence', required=True)
    parser.add_argument('--host-registry', required=True)
    parser.add_argument('--claim-id', required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--engine-gate')
    parser.add_argument('--run-root', default='outputs/supplemental/remaining4/dispatch_20261001_1600')
    args = parser.parse_args()
    args.dataset, args.stage, args.shard, args.n_shards = 'vizwiz', 'independent', 0, 1
    args.key_start, args.key_stop = (0, 80) if args.phase == 'gate' else (80, None)
    args.run_name, args.chunk_rows, args.continuation_receipt = 'dispatch_20261001_1600', 520, None
    plan = generate.load_plan(args)
    tasks = list(generate.selected_tasks(plan))
    groups = {}
    for task in tasks:
        groups.setdefault(task['sample']['id'], []).append(task)
    if any({task['replicate'] for task in group} != set(range(10)) for group in groups.values()):
        raise ValueError('Only complete ten-attempt groups of explicit unclaimed keys may be sent to the original engine')
    evidence_path = within(ROOT, args.route_evidence)
    route = json.loads(evidence_path.read_text())['models'][args.model]
    native_path = within(ROOT, args.native_reference)
    native = json.loads(native_path.read_text())
    if (not native['comparison_complete'] or native['model'] != args.model or native['dataset'] != 'vizwiz'
            or len(native['rows']) != 8 or native['formal_completed_rows'] != 0):
        raise ValueError('Actual eight-input original-runtime Viz comparator is absent')
    versions = {name: importlib.metadata.version(name) for name in ('vllm', 'torch', 'transformers')}
    expected_versions = {name: route['independent_verification']['versions'][name]
                         for name in ('vllm', 'torch', 'transformers')}
    if versions != expected_versions:
        raise ValueError('The actual original accepted engine software versions differ')
    registry_path = within(ROOT, args.host_registry)
    import kdm.execution
    kdm.execution.REGISTRY = str(registry_path.relative_to(ROOT))
    module, scorer = recovered_generator(args.model, route)
    if any(task != module.task(task['sample'], task['replicate']) for task in tasks):
        raise ValueError('Explicit registered task fields differ from the original accepted independent engine')
    if (asdict(module.CONFIG) != asdict(plan['cfg']) or scorer.path != route['original_checkpoint_path']
            or native['dtype'] != route['actual_definition_parameters']['dtype']):
        raise ValueError('Current scientific decoding, checkpoint or precision differs from the original accepted engine')
    from workflows.supplemental.remaining11.execution import checkpoint_identity
    registered = copy.deepcopy(plan['spec'])
    registered['kwargs']['model_path'] = scorer.path
    registered['processor']['path'] = scorer.path
    checkpoint = checkpoint_identity(registered)
    eos = native['eos_token_ids']
    for row in native['rows']:
        validate_input(module, scorer, row, eos)
    cohort = {'model': args.model, 'dataset': 'vizwiz', 'engine': 'vllm_separate_cohort_not_numpy_draw_equivalence',
        'versions': versions, 'dtype': native['dtype'], 'checkpoint': checkpoint,
        'original_implementation_sha256': route['implementation_sha256'],
        'original_closed_preparer_sha256': file_hash(Path(module.closed.__file__)),
        'original_sampling_rule': route['actual_definition_parameters']['sampling_seed_rule'],
        'original_prefix_cache': route['actual_definition_parameters']['prefix_cache'],
        'base_config': asdict(module.CONFIG), 'eos_token_ids': eos,
        'native_reference_sha256': file_hash(native_path)}
    gate = None
    if args.phase == 'production':
        gate_path = within(ROOT, args.engine_gate)
        gate = json.loads(gate_path.read_text())
        if (not gate['passed'] or gate['cohort_identity'] != stable_hash(cohort)
                or gate['entrypoint_sha256'] != file_hash(Path(__file__))
                or gate['registry_sha256'] != file_hash(registry_path)
                or gate['completed_inputs'] != 8 or gate['completed_attempts'] != 80):
            raise ValueError('Actual model-specific Viz engine input/sampling/cache/EOS gate is absent')
    summary = {'plan': plan['summary'], 'engine_cohort': cohort, 'cohort_identity': stable_hash(cohort),
               'native_sampling_equivalence_claimed': False, 'gpu_model_admission': 'not_run'}
    if args.check_plan:
        print(json.dumps(summary), flush=True)
        return
    run = within(ROOT, args.run_root)
    run.relative_to(ROOT / 'outputs/supplemental/remaining4')
    admission = memory_admission(registry_path)
    claim = generate.claim_directory(args, plan, run)
    output = run / 'raw' / args.model / 'vizwiz/independent' / args.claim_id
    record = run / 'records' / args.model / 'vizwiz/independent' / args.claim_id
    output.mkdir(parents=True, exist_ok=False)
    record.mkdir(parents=True, exist_ok=False)
    identity = {'schema': 'kdm_remaining11_generation_v1', 'model': args.model,
        'dataset': 'vizwiz', 'stage': 'independent', 'claim_id': args.claim_id, 'owner': args.owner,
        'backend': cohort, 'registered_backend': plan['spec'],
        'runtime_admission': {'execution': admission, 'runtime_spec': cohort},
        'source_provenance': plan['summary']['source_provenance'], 'task_plan': plan['summary'],
        'runner_sha256': file_hash(Path(__file__)), 'engine_cohort_identity': stable_hash(cohort),
        'engine_route_evidence_sha256': file_hash(evidence_path), 'actual_native_reference_sha256': file_hash(native_path),
        'actual_Viz_engine_gate_sha256': file_hash(gate_path) if gate is not None else None,
        'sampling_backend': 'vllm_separate_cohort_not_numpy_draw_equivalence',
        'paid_api_calls': 0, 'no_automatic_retry': True, 'native_sampling_equivalence_claimed': False}
    atomic_json(record / 'admission.json', identity)
    progress = {'model': args.model, 'dataset': 'vizwiz', 'stage': 'independent',
        'claim_id': args.claim_id, 'owner': args.owner, 'pid': os.getpid(), 'started_utc': generate.now(),
        'status': 'loading', 'completed': 0, 'expected': len(tasks), 'completed_parts': [],
        'claim_path': str(claim.relative_to(ROOT)), 'current_task_key': None}
    atomic_json(record / 'progress.json', progress)
    current_path = None
    report = {'rows': []}
    try:
        scorer.load()
        progress['status'] = 'running'
        samples = list(groups.values())
        for part, start in enumerate(range(0, len(samples), 52)):
            selected = samples[start:start + 52]
            current_tasks = [task for group in selected for task in group]
            basename = f'independent_{args.model}_part_{part:05d}'
            current_path = output / (basename + '.jsonl')
            part_identity = {**identity, 'part': part, 'expected_part_rows': len(current_tasks)}
            ledger = Ledger(current_path, part_identity)
            for group in selected:
                sample = group[0]['sample']
                if args.phase == 'gate':
                    original = next(row for row in native['rows'] if row['sample']['id'] == sample['id'])
                    image, _, base, inputs = validate_input(module, scorer, original, eos)
                rows = scorer.generate_ten(sample, eos)
                if args.phase == 'gate':
                    report['rows'].append(gate_input(module, scorer, original, eos, image, base, inputs, rows))
                for task, row in zip(group, rows):
                    if task != module.task(sample, row['replicate']):
                        raise ValueError('Original accepted engine task or independent replicate differs')
                    key = generate.generation_key(args.model, 'independent', task)
                    ledger.add(key, row)
                progress.update(completed=progress['completed'] + len(rows), updated_utc=generate.now())
                atomic_json(record / 'progress.json', progress)
            coverage = generate.verify_output(current_path, plan, current_tasks, part_identity, 0, 1)
            receipt = {**coverage, 'model': args.model, 'dataset': 'vizwiz', 'stage': 'independent',
                'claim_id': args.claim_id, 'owner': args.owner, 'part': part,
                'raw_path': str(current_path.relative_to(ROOT)), 'finished_utc': generate.now()}
            atomic_json(record / (basename + '.complete.json'), receipt)
            progress['completed_parts'].append({'part': part, 'rows': len(current_tasks), 'raw_path': receipt['raw_path']})
            print(json.dumps({'event': 'raw_part_complete', **receipt}), flush=True)
        if progress['completed'] != progress['expected']:
            raise ValueError('Actual original engine omitted admitted attempt keys')
        peaks = scorer.llm.collective_rpc(actual_worker_peaks)
        if args.phase == 'gate':
            report.update(passed=True, completed_inputs=8, completed_attempts=80,
                cohort_identity=stable_hash(cohort), cohort=cohort,
                greedy_mismatch_count=sum(not row['greedy_tokens_equal'] for row in report['rows']),
                native_sampling_equivalence_claimed=False, separate_engine_cohort=True,
                input_config_EOS_sampling_semantics_passed=True, worker_actual_cuda_peaks=peaks,
                native_reference_sha256=file_hash(native_path), entrypoint_sha256=file_hash(Path(__file__)),
                registry_sha256=file_hash(registry_path), completed_utc=generate.now())
            atomic_json(record / 'engine_gate_success.json', report)
        progress.update(status='generation_complete', updated_utc=generate.now(), worker_actual_cuda_peaks=peaks)
        atomic_json(record / 'progress.json', progress)
        atomic_json(record / 'complete.json', {**progress, 'generation_complete': True,
            'admission_sha256': file_hash(record / 'admission.json'), 'labeling_complete': False,
            'gt_complete': False, 'research_complete': False})
        atomic_json(claim / 'complete.json', {'claim_id': args.claim_id, 'owner': args.owner,
            'rows': progress['completed'], 'record_complete_path': str((record / 'complete.json').relative_to(ROOT))})
    except BaseException as error:
        atomic_json(record / 'failure.json', {'model': args.model, 'dataset': 'vizwiz', 'stage': 'independent',
            'claim_id': args.claim_id, 'error': type(error).__name__ + ': ' + str(error),
            'traceback': traceback.format_exc(), 'completed_rows': progress['completed'],
            'current_raw_path': str(current_path.relative_to(ROOT)) if current_path is not None else None,
            'actual_gate_rows': report['rows'], 'no_retry_performed': True, 'when': generate.now()})
        progress.update(status='failed', updated_utc=generate.now())
        atomic_json(record / 'progress.json', progress)
        raise

if __name__ == '__main__':
    main()
