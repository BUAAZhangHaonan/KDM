"""Exercise real HFSession and TF DynamicCache CPU storage isolation."""
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from types import SimpleNamespace
import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from workflows.supplemental.remaining4.test_prefill_cache import FakeBackend
from workflows.supplemental.remaining4.prefill_cache_readonly import PrefillCacheBackend


class DynamicBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.torch = torch

    def _forward(self, inputs=None, token=None, cache=None, **unused):
        if inputs is not None:
            self.prefills += 1
            cache = DynamicCache()
            self.rotary.rope_deltas = 7
            value = inputs['image'] + len(inputs['prompt'])
            for layer in range(2):
                tensor = torch.full((1, 2, 3, 4), value + layer, dtype=torch.float32)
                cache.update(tensor, tensor + 2, layer)
        else:
            for layer in range(2):
                tensor = torch.full((1, 2, 1, 4), token + layer, dtype=torch.float32)
                cache.update(tensor, tensor + 2, layer)
            self.rotary.rope_deltas += token + 1
        value = sum(float(layer.keys.sum() + layer.values.sum()) for layer in cache.layers) + self.rotary.rope_deltas
        logits = torch.arange(40, dtype=torch.float32).reshape(1, 1, -1) * value
        return SimpleNamespace(logits=logits, past_key_values=cache)


def main():
    original, cached_base = DynamicBackend(), DynamicBackend()
    cached = PrefillCacheBackend(cached_base, verify_integrity=True)
    prefixes = [(), (1,), (1, 2), (), (4,), (), (1,), (1, 7), (1, 7, 5), (), (), (6,)]
    for image, prompt in ((9, 'full original prompt'), (14, 'different input')):
        a, b = original.session(image, prompt), cached.session(image, prompt)
        for prefix in prefixes:
            np.testing.assert_array_equal(a.next(prefix).logits, b.next(prefix).logits)
        cached.verify_last_gate_session()
    stats = cached.stats()
    assert stats['sessions'] == 2 and stats['actual_empty_prefix_prefills'] == 2
    assert stats['actual_base_tensor_equality_checks'] == 40
    assert cached.gate_session is None
    assert cached_base.prefills == 2 and original.prefills == 8
    print({'status': 'pass', 'real_HFSession_and_DynamicCache': True,
           'all_prefix_logits_exact_equal': True, 'base_tensors_unchanged': True,
           'cross_input_cache_and_rotary_isolation': True, 'stats': stats})


if __name__ == '__main__':
    main()
