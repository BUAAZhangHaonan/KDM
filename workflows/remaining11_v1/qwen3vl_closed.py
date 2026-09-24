#!/usr/bin/env python3
"""Qwen3-VL separate vLLM closed-score cohort, with native 16-image gate."""
import os
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
import argparse, hashlib, json, math, socket, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from workflows.acceleration_v4 import vllm_closed_v6 as engine
from kdm.io import Ledger, atomic_json, file_hash, read_jsonl, stable_hash
MODEL = 'qwen3vl'
MODEL_PATH = '/home/team/lvshuyang/Models/Qwen3-VL-8B-Instruct'
engine.PATHS[MODEL] = MODEL_PATH

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    record = ROOT / 'outputs/records/remaining11_v1' / args.run / MODEL
    record.mkdir(parents=True, exist_ok=False)
    try:
        spec_path = ROOT / 'configs/runtime/qwen3vl.json'
        ref_path = ROOT / 'outputs/records/acceleration_v4/qwen3vl_reference16.json'
        spec = json.loads(spec_path.read_text())
        reference = json.loads(ref_path.read_text())
        assert len(reference) == 16
        checkpoint = Path(MODEL_PATH)
        assert file_hash(checkpoint / 'config.json') == spec['model_config_sha256']
        for name, digest in spec['processor']['files'].items():
            assert file_hash(checkpoint / name) == digest, name
        for item in spec['weights']:
            assert (checkpoint / item['filename']).stat().st_size == item['size_bytes']
        scorer = engine.Scorer(MODEL)
        for row in reference:
            _, prompt, unexpanded, inputs = scorer.prepare(row['reference']['sample'])
            assert prompt == row['reference']['prompt']
            assert unexpanded == row['unexpanded_prompt_ids']
            assert inputs['input_ids'][0].tolist() == row['expanded_prompt_ids']
            assert hashlib.sha256(inputs['pixel_values'].numpy().tobytes()).hexdigest() == row['pixel_sha256']
            assert list(inputs['pixel_values'].shape) == row['pixel_shape']
            assert inputs['image_grid_thw'].tolist() == row['image_grid_thw']
            assert scorer.tokens == row['candidates_tokens']
        atomic_json(record / 'preflight.json', {'input_parity': True, 'rows': 16,
            'reference_sha256': file_hash(ref_path), 'checkpoint_spec_sha256': file_hash(spec_path)})
        scorer.load()
        checks, audited = [], {}
        for row in reference:
            old = row['reference']; new = scorer.score(old['sample'])
            assert len(new['candidate_scores']) == 101
            original = {x['label']: x for x in old['candidate_scores']}
            assert set(original) == set(scorer.names)
            old_top = max(original, key=lambda n: original[n]['mean_logp'])
            new_top = max(new['candidate_scores'], key=lambda x: x['mean_logp'])['label']
            check = {'sample_id': new['sample']['id'], 'top1_equal': old_top == new_top,
                'old_gold_rank': old['gold_rank'], 'gold_rank': new['gold_rank'],
                'max_mean_logp_error': max(abs(x['mean_logp'] - original[x['label']]['mean_logp']) for x in new['candidate_scores']),
                'wall_s': new['wall_s'], 'old_wall_s': old['wall_s'],
                'new': new}
            checks.append(check); audited[new['sample']['id']] = new
            atomic_json(record / 'benchmark_progress.json', {'completed': len(checks), 'checks': checks})
        eligible = all(x['top1_equal'] and (x['gold_rank'] == 1) == (x['old_gold_rank'] == 1) for x in checks)
        proof = {'input_parity': True, 'finite_101_class_scores': True, 'full_16_complete': True,
            'top1_and_gold_correctness_parity': eligible, 'numerically_identical': False,
            'cohort': 'separate vLLM batch; old Transformers rows are not mixed', 'checks': checks,
            'speedup': sum(x['old_wall_s'] for x in checks) / sum(x['wall_s'] for x in checks)}
        atomic_json(record / 'benchmark.json', proof)
        assert eligible, 'Native comparison gate failed; no production'
        if not args.execute:
            return
        import vllm, torch, transformers
        identity = {'schema': 'kdm_remaining11_closed_v1', 'model': MODEL, 'checkpoint_path': MODEL_PATH,
            'backend_spec_sha256': file_hash(spec_path), 'native_reference_sha256': file_hash(ref_path),
            'benchmark_sha256': file_hash(record / 'benchmark.json'), 'manifest_sha256': file_hash(ROOT / 'data/current/all.jsonl'),
            'implementation_sha256': file_hash(Path(__file__)), 'engine_sha256': file_hash(Path(engine.__file__)),
            'source_blobs': spec['adapter_source_sha256'], 'host': socket.gethostname(),
            'physical_gpus': os.environ['CUDA_VISIBLE_DEVICES'], 'gpu_uuid': 'GPU-6f5dc226-6850-9f93-d4b7-b6f2d618b402',
            'dtype': 'bfloat16', 'ranking': 'mean original token log probabilities, no EOS',
            'versions': {'vllm': vllm.__version__, 'torch': torch.__version__, 'transformers': transformers.__version__},
            'precision_note': 'separate vLLM numerical cohort; no old-row reuse'}
        output = ROOT / 'outputs/raw/remaining11_v1' / args.run / MODEL / 'closed.jsonl'
        assert not output.exists()
        ledger = Ledger(output, identity)
        food = [s for s in read_jsonl(ROOT / 'data/current/all.jsonl') if s['dataset'] == 'food101']
        assert len(food) == len({s['id'] for s in food}) == 4848
        start = time.time()
        for sample in food:
            row = audited.pop(sample['id'], None)
            if row is None:
                row = scorer.score(sample)
            ledger.add(stable_hash([MODEL, sample['id'], 'closed']), row)
            atomic_json(record / 'progress.json', {'status': 'running', 'pid': os.getpid(), 'model': MODEL,
                'closed': len(ledger.keys), 'expected': 4848, 'started_unix': start, 'updated_unix': time.time(),
                'last_sample_id': sample['id'], 'last_wall_s': row['wall_s'], 'output': str(output.relative_to(ROOT))})
        seen = set()
        for row in read_jsonl(output):
            assert row['sample']['id'] not in seen and row['identity'] == ledger.identity
            seen.add(row['sample']['id'])
            scores = row['candidate_scores']
            assert len(scores) == 101 and {x['label'] for x in scores} == set(scorer.names)
            assert all(math.isfinite(x['mean_logp']) and math.isfinite(x['sum_logp']) for x in scores)
        assert seen == {s['id'] for s in food}
        atomic_json(record / 'complete.json', {'complete': True, 'closed': 4848, 'output_sha256': file_hash(output), 'identity': identity})
        atomic_json(record / 'progress.json', {'status': 'complete', 'closed': 4848, 'expected': 4848, 'pid': os.getpid(), 'updated_unix': time.time()})
    except BaseException as exc:
        atomic_json(record / 'error.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
if __name__ == '__main__':
    main()
