"""Eight actual original-runtime Viz independent-prompt comparators; no formal rows."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
import numpy as np
from kdm.io import atomic_json, file_hash, stable_seed, within
from kdm.probability import log_normalize
from kdm.prompts import task_prompt
from kdm.protocol import validate_environment
from workflows.supplemental.remaining11 import execution, generate

def tensor_summary(values):
    return {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                  'sha256': hashlib.sha256(value.detach().cpu().contiguous().view(
                      __import__('torch').uint8).numpy().tobytes()).hexdigest()}
            for key, value in values.items()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--model', choices=('onevision', 'qwen3vl'), required=True)
    parser.add_argument('--missing-keys', required=True)
    parser.add_argument('--host-registry', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--claim-id', required=True)
    parser.add_argument('--owner', required=True)
    args = parser.parse_args()
    args.dataset, args.stage, args.shard, args.n_shards = 'vizwiz', 'independent', 0, 1
    args.key_start, args.key_stop, args.chunk_rows = 0, 80, 80
    execution.REGISTRY = str(within(ROOT, args.host_registry).relative_to(ROOT))
    plan = generate.load_plan(args)
    tasks = list(generate.selected_tasks(plan))
    samples = {}
    repeats = {}
    for task in tasks:
        sample = task['sample']
        samples[sample['id']] = sample
        repeats.setdefault(sample['id'], set()).add(task['replicate'])
    if len(tasks) != 80 or len(samples) != 8 or any(values != set(range(10)) for values in repeats.values()):
        raise ValueError('The original explicit first80 keys must be eight eval inputs times ten attempts')
    actual_spec = execution.runtime_spec(ROOT, plan['spec'], args.model)
    cpu = {'schema': 'kdm_selected4_viz_independent_native_comparison_plan_v1',
           'model': args.model, 'dataset': 'vizwiz', 'comparison_only': True,
           'formal_completed_rows': 0, 'native_original_environment': validate_environment(actual_spec),
           'checkpoint': execution.checkpoint_identity(actual_spec), 'task_plan': plan['summary'],
           'samples': list(samples.values()), 'generation_gpu_admission': 'not_run'}
    if args.check_plan:
        print(json.dumps(cpu), flush=True)
        return
    import os
    import torch
    from PIL import Image
    from kdm.execution import resolve_image_path
    folder = within(ROOT, args.out)
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / 'plan.json', cpu)
    report = {'schema': 'kdm_selected4_viz_native_independent_reference_v1',
              'model': args.model, 'dataset': 'vizwiz', 'comparison_only': True,
              'formal_completed_rows': 0, 'rows': [], 'expected_inputs': 8,
              'task_plan': plan['summary'], 'pid': os.getpid(), 'started_utc': generate.now(),
              'source_entry_sha256': file_hash(Path(__file__)),
              'source_provenance': plan['summary']['source_provenance']}
    try:
        os.environ['KDM_SUPPLEMENTAL_DATASET'] = 'vizwiz'
        admission = execution.validate_supplemental_runtime(ROOT, plan['spec'], args.model,
            os.environ['CUDA_VISIBLE_DEVICES'].split(','), 'independent', args.claim_id, args.owner)
        report['native_runtime_admission'] = admission
        backend = generate.make_backend(admission['runtime_spec'], 'cuda:0')
        report['eos_token_ids'] = sorted(backend.eos)
        report['dtype'] = admission['runtime_spec']['kwargs']['dtype']
        report['processor_files'] = admission['runtime_spec']['processor']['files']
        for ordinal, sample in enumerate(samples.values()):
            image_path = resolve_image_path(sample['image_path'], ROOT)
            image_copy = folder / f'image_{ordinal:02d}{image_path.suffix}'
            shutil.copy2(image_path, image_copy)
            prompt = task_prompt(sample['question'], guided=False, attempt=True)
            seed = stable_seed(sample['id'], args.model, 0)
            with Image.open(image_path) as image:
                image = image.convert('RGB')
                session = backend.session(image, prompt, reference='clean', seed=seed)
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
                kwargs = {'add_generation_prompt': True, 'tokenize': False}
                if backend.em.mt == 'qwen3_vl':
                    kwargs['enable_thinking'] = False
                chat = backend.em.proc.apply_chat_template(messages, **kwargs)
            input_path = folder / f'processor_inputs_{ordinal:02d}.pt'
            torch.save({key: value.detach().cpu() for key, value in session.inputs.items()}, input_path)
            tokens, selected = [], []
            started = time.perf_counter()
            for _ in range(plan['cfg'].max_tokens):
                logits = session.next(tokens).logits
                probabilities = log_normalize(logits)
                if not np.isfinite(probabilities).all():
                    raise ValueError('Native independent comparator has nonfinite probabilities')
                token = int(np.argmax(probabilities))
                tokens.append(token)
                selected.append(float(probabilities[token]))
                if token in backend.eos:
                    break
            report['rows'].append({'sample': sample, 'prompt': prompt, 'chat_text': chat,
                'seed': seed, 'unexpanded_prompt_ids': backend.encode(chat),
                'expanded_prompt_ids': session.inputs['input_ids'][0].tolist(),
                'processor_summary': tensor_summary(session.inputs),
                'processor_inputs_path': str(input_path.relative_to(ROOT)),
                'processor_inputs_sha256': file_hash(input_path),
                'copied_image_path': str(image_copy.relative_to(ROOT)), 'image_sha256': file_hash(image_copy),
                'original_image_path': str(image_path), 'greedy_tokens': tokens,
                'greedy_text': backend.decode(tokens), 'greedy_selected_log_probabilities': selected,
                'greedy_wall_s': time.perf_counter() - started,
                'terminated': tokens[-1] in backend.eos,
                'comparison_config': {**asdict(plan['cfg']), 'temperature': 0.0}})
            atomic_json(folder / 'partial.json', report)
            print(json.dumps({'event': 'native_comparison_complete', 'model': args.model,
                              'completed_inputs': len(report['rows']), 'formal_completed_rows': 0}), flush=True)
        report.update(comparison_complete=True, finished_utc=generate.now(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        atomic_json(folder / 'reference.json', report)
        print(json.dumps({'model': args.model, 'completed_inputs': 8, 'comparison_complete': True,
                          'reference': str((folder / 'reference.json').relative_to(ROOT))}), flush=True)
    except BaseException as error:
        atomic_json(folder / 'failure.json', {'error': type(error).__name__ + ': ' + str(error),
            'traceback': traceback.format_exc(), 'comparison_inputs_complete': len(report['rows']),
            'formal_completed_rows': 0, 'automatic_retry': False})
        raise

if __name__ == '__main__':
    main()
