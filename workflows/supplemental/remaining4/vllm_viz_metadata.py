"""Isolate only the optional post-generation CUDA peak RPC in the accepted Viz engine caller."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from kdm.io import atomic_json, file_hash, within
from workflows.supplemental.remaining4 import vllm_viz_recording as recording

ORIGINAL_RECORDING_SHA = '1f041f9fd973c303a25c56c0f563183b7069fae64e2152fe8bdcef78b08f27af'


def metadata_identity():
    return {
        'entrypoint': str(Path(__file__).relative_to(ROOT)),
        'sha256': file_hash(Path(__file__)),
        'original_recording_adapter_sha256': ORIGINAL_RECORDING_SHA,
        'change': 'Only optional actual_worker_peaks callable RPC after all admitted output parts are sealed is suppressed; CUDA peaks remain explicitly unmeasured.',
        'scientific_generation_changed': False,
        'insecure_serialization_enabled': False,
        'automatic_scientific_retry': False,
    }


def install_metadata_rpc(llm_class, original, record):
    actual_rpc = llm_class.collective_rpc

    def limited_metadata_rpc(self, method, *args, **kwargs):
        if method is not original.actual_worker_peaks:
            return actual_rpc(self, method, *args, **kwargs)
        progress = json.loads((record / 'progress.json').read_text())
        parts = list(record.glob('*_part_*.complete.json'))
        receipts = [json.loads(path.read_text()) for path in parts]
        if (not parts or progress['completed'] != progress['expected']
                or any(not receipt['generation_complete'] for receipt in receipts)
                or sum(receipt['rows'] for receipt in receipts) != progress['expected']
                or progress['current_task_key'] is not None):
            raise ValueError('Metadata RPC isolation requires every original admitted generation part to be sealed')
        observation = {
            'pid': None, 'logical_device': None,
            'peak_allocated_bytes': None, 'peak_reserved_bytes': None,
            'status': 'not_measured',
            'reason': 'Accepted original vLLM serializer rejects a Python callable RPC; no unsafe pickle or replacement GPU diagnostic was used.',
            'source_adapter': metadata_identity(),
        }
        atomic_json(record / 'optional_cuda_peak_rpc_not_measured.json', observation)
        return [observation]

    llm_class.collective_rpc = limited_metadata_rpc


def main():
    if file_hash(Path(recording.__file__)) != ORIGINAL_RECORDING_SHA:
        raise ValueError('The fixed accepted recording adapter bytes changed')
    if '--phase' not in sys.argv or sys.argv[sys.argv.index('--phase') + 1] != 'production':
        raise ValueError('The metadata-only wrapper is restricted to the admitted original production phase')
    model = sys.argv[sys.argv.index('--model') + 1]
    claim = sys.argv[sys.argv.index('--claim-id') + 1]
    run_root = (sys.argv[sys.argv.index('--run-root') + 1] if '--run-root' in sys.argv
                else 'outputs/supplemental/remaining4/dispatch_20261001_1600')
    record = within(ROOT, run_root) / 'records' / model / 'vizwiz/independent' / claim
    original_ledger = recording.RegisteredLedger
    original_atomic = recording.bound_atomic_json

    class MetadataProvenanceLedger(original_ledger):
        def __init__(self, path, identity):
            identity['postgeneration_metadata_adapter'] = metadata_identity()
            super().__init__(path, identity)

    def metadata_atomic(path, value):
        if value.get('schema') == 'kdm_remaining11_generation_v1':
            value['postgeneration_metadata_adapter'] = metadata_identity()
        original_atomic(path, value)

    recording.RegisteredLedger = MetadataProvenanceLedger
    recording.bound_atomic_json = metadata_atomic
    if '--execute' in sys.argv:
        from vllm import LLM
        install_metadata_rpc(LLM, recording.original, record)
    recording.main()


if __name__ == '__main__':
    main()
