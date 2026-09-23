"""Audit-only conversion: native half-open spans to vLLM inclusive spans."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
VLLM_ROOT = Path('/home/team/lvshuyang/anaconda3/envs/ST_LORA/lib/python3.12/site-packages/vllm')
PINNED = {
 'v1/attention/ops/triton_attention_helpers.py': '98cb76f662dabc7005c0ad61e201d239eb1c46d8a10a5839bbc8e0065356fab8',
 'v1/attention/backends/triton_attn.py': '1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605',
 'v1/attention/backends/utils.py': 'f1375ec8e3a2d5c77ab15773eef2c00a4117826a3c238ceb5bcd4efe9fa55381',
 'v1/worker/gpu_model_runner.py': 'd2b8ec101122749e1a2610383d2ab74c5a21605e9d7995eb68192bc769986694',
 'multimodal/inputs.py': '18f8549124fab2c0012b3a1f49aeec10d4abc8423d66378124049ea776092c37',
}

def source_binding():
 for name, expected in PINNED.items():
  if hashlib.sha256((VLLM_ROOT / name).read_bytes()).hexdigest() != expected:
   raise ValueError('vLLM range source changed: ' + name)
 return {'schema': 'gemma_attention_range_endpoint_audit_v1', 'module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'vllm_root': str(VLLM_ROOT), 'source_sha256': PINNED, 'native_convention': '[start,end)', 'vllm_convention': '[start,end]', 'source_lines': {'multimodal/inputs.py': [179, 202], 'v1/attention/ops/triton_attention_helpers.py': [335, 352], 'v1/worker/gpu_model_runner.py': [2329, 2336]}, 'mask_modified': False}

def check(original, path, expected_ranges, physical_gpu):
 binding = source_binding()
 native = {(int(a), int(b)) for a, b in expected_ranges}
 if not native or any(a < 0 or b <= a for a, b in native):
  raise ValueError('Invalid native half-open ranges')
 inclusive = {(a, b - 1) for a, b in native}
 # Delegate all original metadata/GPU/module provenance checks unchanged.
 result = original(path, inclusive, physical_gpu)
 receipt = json.loads(Path(path).read_text())
 requested = {(int(a), int(b)) for spans in receipt['ranges'].values() for a, b in spans if b > a}
 converted = {(a, b + 1) for a, b in requested}
 if converted != native:
  raise ValueError('Converted vLLM image ranges differ from native token types')
 return {**result, 'expected_native_ranges': [list(r) for r in sorted(native)], 'native_half_open_ranges': [list(r) for r in sorted(native)], 'raw_vllm_inclusive_ranges': receipt['ranges'], 'raw_vllm_inclusive_tensor_ranges': receipt['tensor_ranges'], 'vllm_converted_half_open_ranges': [list(r) for r in sorted(converted)], 'endpoint_audit': binding, 'mask_modified': False}

def install(base):
 original = base.check_gemma_attention_receipt
 def corrected(path, expected_ranges, physical_gpu):
  return check(original, path, expected_ranges, physical_gpu)
 base.check_gemma_attention_receipt = corrected
 return source_binding()
