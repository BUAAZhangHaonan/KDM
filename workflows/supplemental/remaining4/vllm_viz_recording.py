"""Recording-only adapter for the fixed accepted Viz independent engine caller."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from kdm.io import Ledger, atomic_json, file_hash, read_jsonl, stable_hash, within
from workflows.supplemental.remaining4 import vllm_viz as original

ORIGINAL_SHA = 'b475b205128b28a334bc582136e428a370fbac4fcd59ff2abfb4fbb003fdc125'
if file_hash(Path(original.__file__)) != ORIGINAL_SHA:
    raise ValueError('The actual original Viz engine caller source changed')


def recording_identity():
    return {'entrypoint': str(Path(__file__).relative_to(ROOT)), 'sha256': file_hash(Path(__file__)),
        'original_engine_caller_sha256': ORIGINAL_SHA,
        'change': 'Ledger identity adds existing canonical shard0/n_shards1/base_config fields only; engine computation and every scientific parameter unchanged',
        'new_engine_algorithm': False, 'automatic_scientific_retry': False}


class RegisteredLedger(Ledger):
    def __init__(self, path, identity):
        if identity['dataset'] != 'vizwiz' or identity['stage'] != 'independent':
            raise ValueError('The recording adapter is restricted to the actual Viz independent engine cohort')
        identity.update(shard=0, n_shards=1, base_config=copy.deepcopy(identity['backend']['base_config']),
                        recording_adapter=recording_identity())
        super().__init__(path, identity)


def bound_atomic_json(path, value):
    if (value.get('schema') == 'kdm_remaining11_generation_v1'
            or Path(path).name == 'engine_gate_success.json'):
        value['recording_adapter'] = recording_identity()
    atomic_json(path, value)


def existing_gate(args):
    run = within(ROOT, args.run_root)
    old_record = run / 'records' / args.model / 'vizwiz/independent' / args.claim_id
    failure_path, progress_path = old_record / 'failure.json', old_record / 'progress.json'
    failure, progress = json.loads(failure_path.read_text()), json.loads(progress_path.read_text())
    if (failure['error'] != 'ValueError: Raw-part identity sidecar differs from admitted configuration'
            or failure['completed_rows'] != 80
            or progress['status'] != 'failed' or len(failure['actual_gate_rows']) != 8
            or progress['expected'] != 80 or progress['completed'] != 80):
        raise ValueError('Only the completed original eighty-row recording failure can be sealed without generation')
    raw = within(ROOT, failure['current_raw_path'])
    if raw.parent != run / 'raw' / args.model / 'vizwiz/independent' / args.claim_id:
        raise ValueError('The actual failed original raw path differs from its claim')
    proc = Path('/proc') / str(progress['pid'])
    command = (proc / 'cmdline').read_bytes().decode(errors='replace') if proc.exists() else ''
    alive = '--claim-id\0' + args.claim_id + '\0' in command
    if alive:
        for fd in (proc / 'fd').iterdir():
            try:
                if fd.readlink() == raw:
                    raise ValueError('The original raw still has an open producer file descriptor')
            except FileNotFoundError:
                continue
    sources = [raw, raw.with_suffix('.identity.json'), old_record / 'admission.json', failure_path,
        progress_path, ROOT / progress['claim_path'] / 'owner.json', ROOT / progress['claim_path'] / 'keys.jsonl']
    source_hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in sources}
    if (ROOT / progress['claim_path'] / 'recording_derivative_completion.json').exists():
        raise ValueError('An existing recording derivative must remain immutable')
    old_meta = json.loads(raw.with_suffix('.identity.json').read_text())
    if old_meta['identity'] != stable_hash(old_meta['definition']):
        raise ValueError('The actual original identity source is corrupt')
    payload = raw.read_bytes()
    if not payload.endswith(b'\n'):
        raise ValueError('The original fully generated response prefix lacks its final newline')
    rows = [json.loads(line) for line in payload.splitlines()]
    if len(rows) != 80 or any(row['identity'] != old_meta['identity'] for row in rows):
        raise ValueError('The actual original eighty-row source is incomplete or has a different identity')
    for path in sources:
        if file_hash(path) != source_hashes[str(path.relative_to(ROOT))]:
            raise ValueError('Original failed source changed during bounded CPU sealing')
    plan_args = SimpleNamespace(model=args.model, dataset='vizwiz', stage='independent',
        missing_keys=args.missing_keys, key_start=0, key_stop=80, shard=0, n_shards=1)
    plan = original.generate.load_plan(plan_args)
    tasks = list(original.generate.selected_tasks(plan))
    actual_comparisons = failure['actual_gate_rows']
    for comparison in actual_comparisons:
        if (not all(comparison['engine_processor_tensor_parity'].values())
                or not all(comparison[field] for field in ('engine_expanded_prompt_ids_equal',
                    'ten_distinct_stable_seeds', 'native_eos_handling', 'finite_selected_logprobs'))
                or comparison['num_cached_tokens'] <= 0):
            raise ValueError('The actual original Viz image/processor/EOS/seed/cache comparison did not pass')
    if {comparison['sample_id'] for comparison in actual_comparisons} != {task['sample']['id'] for task in tasks}:
        raise ValueError('The actual eight compared inputs differ from the original eighty attempt keys')
    identity = copy.deepcopy(old_meta['definition'])
    if (identity['runner_sha256'] != ORIGINAL_SHA or identity['backend']['base_config'] != asdict(plan['cfg'])
            or identity['expected_part_rows'] != 80 or identity['model'] != args.model):
        raise ValueError('The actual original caller and scientific configuration differ')
    definition_additions = {'source_recording_failure': {'original_identity': old_meta['identity'],
        'source_members_sha256': source_hashes, 'original_claim_path': progress['claim_path'],
        'completed_eighty_before_failure': True, 'original_producer_failed_waiting_cleanup': alive,
        'original_raw_open_file_descriptors': 0, 'new_GPU_generation_calls': 0,
        'only_added_recording_fields': ['shard', 'n_shards', 'base_config', 'recording_adapter']}}
    identity.update(definition_additions)
    target = within(ROOT, args.output)
    target.relative_to(run / 'derivatives')
    target.mkdir(parents=True, exist_ok=False)
    target_raw = target / ('independent_' + args.model + '_part_00000.jsonl')
    ledger = RegisteredLedger(target_raw, identity)
    for row in rows:
        ledger.add(row['key'], {**row, 'original_record_identity': row['identity']})
    coverage = original.generate.verify_output(target_raw, plan, tasks, identity, 0, 1)
    derived_rows = list(read_jsonl(target_raw))
    for old, new in zip(rows, derived_rows):
        if {key: value for key, value in old.items() if key != 'identity'} != {
                key: value for key, value in new.items() if key not in ('identity', 'original_record_identity')}:
            raise ValueError('The derived source changed an original generated scientific field')
    admission = json.loads((old_record / 'admission.json').read_text())
    admission.update(definition_additions, recording_adapter=recording_identity())
    atomic_json(target / 'admission.json', admission)
    receipt = {**coverage, 'model': args.model, 'dataset': 'vizwiz', 'stage': 'independent',
        'claim_id': args.claim_id, 'owner': progress['owner'], 'part': 0,
        'raw_path': str(target_raw.relative_to(ROOT)), 'finished_utc': original.generate.now(),
        'source_kind': 'CPU_recording_derivative_of_actual_completed80_not_new_generation'}
    complete_path = target / ('independent_' + args.model + '_part_00000.complete.json')
    atomic_json(complete_path, receipt)
    report = {'passed': True, 'completed_inputs': 8, 'completed_attempts': 80,
        'rows': actual_comparisons, 'cohort_identity': identity['engine_cohort_identity'],
        'cohort': identity['backend'], 'entrypoint_sha256': ORIGINAL_SHA,
        'recording_adapter': recording_identity(), 'registry_sha256': identity['runtime_admission']['execution']['registry_sha256'],
        'native_reference_sha256': identity['actual_native_reference_sha256'],
        'input_config_EOS_sampling_semantics_passed': True,
        'greedy_mismatch_count': sum(not item['greedy_tokens_equal'] for item in actual_comparisons),
        'native_sampling_equivalence_claimed': False, 'separate_engine_cohort': True,
        'worker_actual_cuda_peaks': None, 'peak_unavailable_reason': 'Original writer refused before the existing post-generation peak RPC',
        'original_actual_GPU_gate_failure_source': definition_additions['source_recording_failure'],
        'recording_only_CPU_coverage_passed': True, 'scientific_fields_exactly_preserved': True,
        'completed_utc': original.generate.now()}
    gate_path = target / 'engine_gate_success.json'
    atomic_json(gate_path, report)
    files = [{'kind': kind, 'path': str(path), 'sha256': file_hash(path)} for kind, path in (
        ('raw', target_raw), ('identity', target_raw.with_suffix('.identity.json')),
        ('complete', complete_path), ('admission', target / 'admission.json'),
        ('engine_gate', gate_path), ('owner', ROOT / progress['claim_path'] / 'owner.json'),
        ('keys', ROOT / progress['claim_path'] / 'keys.jsonl'))]
    manifest = {'schema': 'kdm_selected4_immutable_delta_manifest_v1', 'created_utc': original.generate.now(),
        'rows': 80, 'parts': [{'model': args.model, 'dataset': 'vizwiz', 'stage': 'independent',
            'source_host': 'k100', 'source_root': str(ROOT), 'claim_id': args.claim_id,
            'part': 0, 'rows': 80, 'finished_utc': receipt['finished_utc'], 'raw_path': str(target_raw),
            'identity_path': str(target_raw.with_suffix('.identity.json')), 'receipt_path': str(complete_path),
            'raw_sha256': coverage['raw_sha256'], 'identity_sha256': coverage['identity_sha256'], 'files': files}],
        'original_source_members_sha256': source_hashes, 'original_raw_open_file_descriptors': 0,
        'new_GPU_generation_calls': 0, 'all_generation_complete': False}
    atomic_json(target / 'source_manifest.json', manifest)
    atomic_json(ROOT / progress['claim_path'] / 'recording_derivative_completion.json', {
        'claim_id': args.claim_id, 'original_failure_preserved': True,
        'source_manifest': str((target / 'source_manifest.json').relative_to(ROOT)),
        'source_manifest_sha256': file_hash(target / 'source_manifest.json'), 'actual_completed_keys': 80,
        'new_GPU_generation_calls': 0, 'cpu_coverage_passed': True})
    print(json.dumps({'model': args.model, 'rows': 80, 'new_GPU_generation_calls': 0,
        'scientific_fields_exactly_preserved': True, 'greedy_mismatch_count': report['greedy_mismatch_count'],
        'manifest': str((target / 'source_manifest.json').relative_to(ROOT)),
        'engine_gate': str(gate_path.relative_to(ROOT))}), flush=True)


def main():
    if '--seal-existing-gate' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seal-existing-gate', action='store_true')
        parser.add_argument('--model', choices=('onevision', 'qwen3vl'), required=True)
        parser.add_argument('--claim-id', required=True)
        parser.add_argument('--missing-keys', required=True)
        parser.add_argument('--output', required=True)
        parser.add_argument('--run-root', default='outputs/supplemental/remaining4/dispatch_20261001_1600')
        existing_gate(parser.parse_args())
        return
    if '--phase' in sys.argv and sys.argv[sys.argv.index('--phase') + 1] == 'production':
        gate_path = within(ROOT, sys.argv[sys.argv.index('--engine-gate') + 1])
        gate = json.loads(gate_path.read_text())
        if gate.get('recording_adapter') != recording_identity():
            raise ValueError('The exact recording adapter is not bound by the admitted actual engine gate')
    original.Ledger = RegisteredLedger
    original.atomic_json = bound_atomic_json
    original.main()


if __name__ == '__main__':
    main()
