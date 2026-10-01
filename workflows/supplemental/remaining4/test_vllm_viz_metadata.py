"""CPU contracts for restricting the adapter to optional final metadata."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from workflows.supplemental.remaining4 import vllm_viz_metadata as adapter


class MetadataIsolationTest(unittest.TestCase):
    def setUp(self):
        folder = adapter.ROOT / 'cache/tmp'
        folder.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=folder)
        self.addCleanup(self.temp.cleanup)
        self.record = Path(self.temp.name)
        self.marker = object()
        self.calls = []
        calls = self.calls

        class ActualLLM:
            def collective_rpc(self, method, *args, **kwargs):
                calls.append((method, args, kwargs))
                return 'original_scientific_result'

        self.llm = ActualLLM()
        adapter.install_metadata_rpc(ActualLLM, SimpleNamespace(actual_worker_peaks=self.marker), self.record)

    def progress(self, completed, expected):
        (self.record / 'progress.json').write_text(json.dumps({
            'completed': completed, 'expected': expected, 'current_task_key': None}))
        (self.record / 'independent_part_00000.complete.json').write_text(json.dumps({
            'generation_complete': True, 'rows': completed}))

    def test_other_rpc_method_and_arguments_are_preserved(self):
        sentinel = object()
        result = self.llm.collective_rpc('execute_model', sentinel, task='unchanged')
        self.assertEqual(result, 'original_scientific_result')
        self.assertEqual(self.calls, [('execute_model', (sentinel,), {'task': 'unchanged'})])
        self.assertFalse((self.record / 'optional_cuda_peak_rpc_not_measured.json').exists())

    def test_complete_output_records_unknown_peak_without_rpc(self):
        self.progress(520, 520)
        result = self.llm.collective_rpc(self.marker)
        self.assertEqual(self.calls, [])
        self.assertIsNone(result[0]['peak_allocated_bytes'])
        self.assertIsNone(result[0]['peak_reserved_bytes'])
        self.assertEqual(result[0]['status'], 'not_measured')
        self.assertFalse(result[0]['source_adapter']['scientific_generation_changed'])
        self.assertTrue((self.record / 'optional_cuda_peak_rpc_not_measured.json').is_file())

    def test_unfinished_generation_cannot_bypass_rpc(self):
        self.progress(520, 1040)
        with self.assertRaises(ValueError):
            self.llm.collective_rpc(self.marker)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.record / 'optional_cuda_peak_rpc_not_measured.json').exists())


if __name__ == '__main__':
    unittest.main()
