"""Actual HFSession prefix/rotary semantics with deliberately mutable test KV."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import numpy as np
from kdm.models.hf import HFSession
from workflows.supplemental.remaining4.prefill_cache import PrefillCacheBackend


class Array:
    def __init__(self, array):
        self.array = np.asarray(array)

    def __getitem__(self, key):
        return Array(self.array[key])

    def float(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array.copy()


class FakeBackend:
    def __init__(self):
        self.rotary = SimpleNamespace(rope_deltas=None)
        self.model = SimpleNamespace(training=False,
                                     named_modules=lambda: [('rotary', self.rotary)])
        self.prefills = 0

    def session(self, image, prompt, reference='clean'):
        return HFSession(self, {'image': image, 'prompt': prompt}, False, False)

    def _forward(self, inputs=None, token=None, cache=None, **unused):
        if inputs is not None:
            self.prefills += 1
            cache = {'mutable_tokens': [], 'nested': {'history': []},
                     'base': inputs['image'] + len(inputs['prompt'])}
            self.rotary.rope_deltas = 7
        else:
            # Deliberate in-place mutation of both cache branches catches an
            # incorrectly shallow-copied base prefill.
            cache['mutable_tokens'].append(token)
            cache['nested']['history'].append(self.rotary.rope_deltas)
            self.rotary.rope_deltas += token + 1
        value = (cache['base'] + sum((i + 1) * tok for i, tok in enumerate(cache['mutable_tokens']))
                 + sum(cache['nested']['history']) + self.rotary.rope_deltas)
        logits = np.arange(40, dtype=np.float32) * value
        return SimpleNamespace(logits=Array(logits.reshape(1, 1, -1)), past_key_values=cache)


def main():
    original = FakeBackend()
    optimized_base = FakeBackend()
    optimized = PrefillCacheBackend(optimized_base)
    tokens = [(), (1,), (1, 2), (1, 2, 3), (), (9,), (), (1,), (1, 3),
              (), (5,), (5, 6), (), (), (7,), (7, 8, 9), (), (2,)]
    for image, prompt in ((11, 'original prompt'), (19, 'another input')):
        a, b = original.session(image, prompt), optimized.session(image, prompt)
        for prefix in tokens:
            np.testing.assert_array_equal(a.next(prefix).logits, b.next(prefix).logits)
    assert optimized.stats()['sessions'] == 2
    assert optimized.stats()['actual_empty_prefix_prefills'] == 2
    assert optimized.stats()['reused_empty_prefixes'] == 12
    assert original.prefills == 14 and optimized_base.prefills == 4
    # The non-incremental nonempty prefix above is intentionally delegated to
    # the original runner. Candidate scoring uses empty+incremental prefixes.
    assert not hasattr(optimized, 'sessions')
    print({'status': 'pass', 'mutable_KV_and_rotary_isolation': True,
           'actual_HFSession_prefix_rules': True, 'cross_input_isolation': True,
           'counter_only_no_retained_GPU_sessions': True})


if __name__ == '__main__':
    main()
