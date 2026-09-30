#!/usr/bin/env python3
"""Finite, actual original-vs-cached closed-rank concordance and timing gate."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from kdm.execution import resolve_image_path
from kdm.io import atomic_json, file_hash, Ledger, stable_hash, within
from kdm.pipeline import closed_rank, make_backend
from workflows.supplemental.remaining11 import execution, generate
from workflows.supplemental.remaining4.prefill_cache import PrefillCacheBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('internvl35_8b', 'phi35'), required=True)
    parser.add_argument('--missing-keys', required=True)
    parser.add_argument('--key-start', type=int, default=0)
    parser.add_argument('--key-stop', type=int, default=8)
    parser.add_argument('--host-registry', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--claim-id', required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--original-raw', help='Optional sealed original records for these same eight inputs')
    args = parser.parse_args()
    args.dataset = 'food101'
    args.stage = 'candidate'
    args.shard = 0
    args.n_shards = 1
    args.continuation_receipt = None
    args.run_name = args.claim_id
    registry = within(ROOT, args.host_registry)
    execution.REGISTRY = str(registry.relative_to(ROOT))
    plan = generate.load_plan(args)
    tasks = list(generate.selected_tasks(plan))
    if len(tasks) != 8:
        raise ValueError('The finite cache admission requires exactly eight registered inputs')
    out = within(ROOT, args.out)
    out.mkdir(parents=True, exist_ok=False)
    source = {}
    if args.original_raw:
        raw = within(ROOT, args.original_raw)
        if not raw.with_suffix('.identity.json').is_file():
            raise ValueError('Original raw identity is missing')
        from kdm.io import read_jsonl
        for row in read_jsonl(raw):
            sid = row.get('sample', {}).get('id')
            if row.get('model') == args.model and row.get('status') == 'ok':
                source[sid] = row
        if not all(task['sample']['id'] in source for task in tasks):
            raise ValueError('Original source does not cover the exact eight registered inputs')
        source_receipt = {'path': str(raw.relative_to(ROOT)), 'sha256': file_hash(raw),
                          'identity_sha256': file_hash(raw.with_suffix('.identity.json'))}
    else:
        source_receipt = None
    admission = execution.validate_supplemental_runtime(
        ROOT, plan['spec'], args.model, os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','),
        'candidate', args.claim_id, args.owner)
    identity = {'schema': 'kdm_closed_rank_prefill_gate_v1', 'model': args.model,
                'admission': admission, 'plan': plan['summary'],
                'original_source': source_receipt,
                'gate_sha256': file_hash(Path(__file__)),
                'cache_sha256': file_hash(Path(__file__).with_name('prefill_cache.py')),
                'original_algorithm_sha256': file_hash(ROOT / 'src/kdm/pipeline.py'),
                'scientific_parameters_changed': False, 'gate_inputs': 8}
    atomic_json(out / 'admission.json', identity)
    ledger = Ledger(out / 'comparisons.jsonl', identity)
    import torch
    from PIL import Image
    cards = list(range(torch.cuda.device_count()))
    rows = []
    try:
        backend = make_backend(admission['runtime_spec'], 'cuda:0')
        cached = PrefillCacheBackend(backend)
        for task in tasks:
            sample = task['sample']
            with Image.open(resolve_image_path(sample['image_path'], ROOT)) as raw_image:
                image = raw_image.convert('RGB')
                for card in cards:
                    torch.cuda.synchronize(card)
                    torch.cuda.reset_peak_memory_stats(card)
                start = time.perf_counter()
                original = source.get(sample['id'])
                if original is None:
                    original = closed_rank(backend, image, sample['question'], plan['names'], sample['class'])
                    for card in cards:
                        torch.cuda.synchronize(card)
                    original_seconds = time.perf_counter() - start
                else:
                    original_seconds = original['wall_s']
                original_peak = [torch.cuda.max_memory_allocated(card) for card in cards]
                for card in cards:
                    torch.cuda.reset_peak_memory_stats(card)
                start = time.perf_counter()
                actual = closed_rank(cached, image, sample['question'], plan['names'], sample['class'])
                for card in cards:
                    torch.cuda.synchronize(card)
                cached_seconds = time.perf_counter() - start
                cached_peak = [torch.cuda.max_memory_allocated(card) for card in cards]
            if original['prompt'] != actual['prompt'] or original['target'] != actual['target']:
                raise ValueError('Actual prompts or target identities differ')
            a, b = original['candidate_scores'], actual['candidate_scores']
            if len(a) != 101 or len(b) != 101 or any(
                    x['label'] != y['label'] or x['n_tokens'] != y['n_tokens'] for x, y in zip(a, b)):
                raise ValueError('Original label order or tokenization differs')
            max_difference = max(abs(x[field] - y[field]) for x, y in zip(a, b)
                                 for field in ('sum_logp', 'mean_logp'))
            row = {'model': args.model, 'sample': sample,
                   'original': {key: original[key] for key in actual}, 'cached': actual,
                   'original_seconds': original_seconds, 'cached_seconds': cached_seconds,
                   'original_peak_allocated_bytes': original_peak,
                   'cached_peak_allocated_bytes': cached_peak,
                   'max_absolute_score_difference': max_difference,
                   'rank_equal': original['gold_rank'] == actual['gold_rank'],
                   'original_timing_same_loaded_backend': source_receipt is None,
                   'cache_counters': cached.stats()}
            ledger.add(stable_hash([args.model, sample['id'], 'cache_gate']), row)
            rows.append(row)
            if max_difference != 0 or not row['rank_equal']:
                raise ValueError('Cache gate failed exact original-score/rank concordance')
            gc.collect()
            print(json.dumps({'event': 'cache_gate_input_complete', 'sample_id': sample['id'],
                              'original_seconds': original_seconds, 'cached_seconds': cached_seconds,
                              'max_absolute_score_difference': max_difference}), flush=True)
        result = {'status': 'pass', 'model': args.model, 'inputs': 8,
                  'candidate_score_rows': 808, 'max_absolute_score_difference': 0,
                  'rank_differences': 0, 'admission_sha256': file_hash(out / 'admission.json'),
                  'comparisons_sha256': file_hash(out / 'comparisons.jsonl'),
                  'cache_sha256': identity['cache_sha256'],
                  'original_algorithm_sha256': identity['original_algorithm_sha256'],
                  'gate_sha256': identity['gate_sha256'], 'registry_sha256': file_hash(registry),
                  'cache_counters': cached.stats(),
                  'original_seconds': sum(row['original_seconds'] for row in rows),
                  'cached_seconds': sum(row['cached_seconds'] for row in rows),
                  'same_loaded_backend_timing': source_receipt is None,
                  'scientific_parameters_changed': False}
        atomic_json(out / 'success.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        atomic_json(out / 'failure.json', {'error': type(exc).__name__ + ': ' + str(exc),
                                          'completed_inputs': len(rows), 'no_retry_performed': True})
        raise


if __name__ == '__main__':
    main()
