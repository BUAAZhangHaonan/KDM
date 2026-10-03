"""Project-local audit of real Triton multimodal attention metadata.

This hook observes metadata only; it does not change masks, scores, or kernels.
Load as workflows.acceleration_v4.gemma_audit_worker.GemmaAuditWorker.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import threading
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
_lock = threading.Lock()
_written_paths: set[str] = set()
_local_rank = 0


def _record_metadata(metadata, common_metadata):
    target = os.environ.get("KDM_ATTN_AUDIT_PATH")
    if not target or target in _written_paths:
        return
    ranges = getattr(common_metadata, "mm_req_doc_ranges", None)
    tensor = getattr(metadata, "mm_prefix_range_tensor", None)
    # Profiling and non-image calls have no real image interval.
    if not ranges or tensor is None or tensor.numel() == 0:
        return
    valid_ranges = {
        str(key): [[int(start), int(end)] for start, end in value if int(end) > int(start)]
        for key, value in ranges.items()
    }
    valid_ranges = {key: value for key, value in valid_ranges.items() if value}
    if not valid_ranges:
        return
    if tensor.ndim != 3 or tensor.shape[-1] != 2:
        raise RuntimeError("Unexpected Triton multimodal range tensor layout")
    values = tensor.detach().cpu().tolist()
    actual_intervals = {
        (int(start), int(end))
        for row in values for start, end in row if int(end) > int(start)
    }
    expected_intervals = {
        (start, end) for row in valid_ranges.values() for start, end in row
    }
    if not expected_intervals.issubset(actual_intervals):
        raise RuntimeError("Triton metadata lost a real image interval")
    path = Path(target).resolve()
    if not path.is_relative_to(ROOT / "outputs"):
        raise RuntimeError("Attention audit output must remain inside project outputs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [part.strip() for part in visible.split(",") if part.strip()]
    receipt = {
        "backend": "TRITON_ATTN",
        "scope": "real_metadata_builder_output_not_kernel_execution_proof",
        "shape": list(tensor.shape),
        "ranges": valid_ranges,
        "tensor_ranges": values,
        "physical_gpu": devices[_local_rank] if _local_rank < len(devices) else None,
        "cuda_visible_devices": visible,
        "local_rank": _local_rank,
        "pid": os.getpid(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "num_actual_tokens": int(getattr(common_metadata, "num_actual_tokens", 0)),
        "nonempty_real_ranges": True,
    }
    with _lock:
        if target in _written_paths:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never replace evidence from another attempt or worker.
        with path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        _written_paths.add(target)


def install_audit_hook():
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
    original = TritonAttentionMetadataBuilder.build
    if getattr(original, "_kdm_attention_audit", False):
        return

    @functools.wraps(original)
    def audited_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        result = original(self, common_prefix_len, common_attn_metadata, fast_build)
        _record_metadata(result, common_attn_metadata)
        return result

    audited_build._kdm_attention_audit = True
    TritonAttentionMetadataBuilder.build = audited_build


install_audit_hook()

from vllm.v1.worker.gpu_worker import Worker


class GemmaAuditWorker(Worker):
    def __init__(self, *args, **kwargs):
        global _local_rank
        _local_rank = int(kwargs.get("local_rank", args[1] if len(args) > 1 else 0))
        install_audit_hook()
        super().__init__(*args, **kwargs)
