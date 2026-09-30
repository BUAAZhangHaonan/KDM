"""Clone DynamicCache objects while sharing verified immutable prompt tensors.

This optimization is admitted only for actual full-attention DynamicLayer caches.
The registered DynamicLayer.update concatenates into new tensors.  A finite GPU
gate checks the original 101 scores and every base tensor across all candidates.
"""
from __future__ import annotations

import copy
import hashlib
import inspect

from workflows.supplemental.remaining4.prefill_cache import PrefillCacheSession


def clone_full_attention_cache(cache):
    from transformers.cache_utils import DynamicCache, DynamicLayer
    if type(cache) is not DynamicCache or cache.offloading or not cache.layers:
        raise TypeError('Read-only prefill reuse requires the registered non-offloaded DynamicCache')
    if any(type(layer) is not DynamicLayer or not layer.is_initialized for layer in cache.layers):
        raise TypeError('Only initialized full-attention DynamicLayer cache storage is admitted')
    cloned = copy.copy(cache)
    cloned.layers = [copy.copy(layer) for layer in cache.layers]
    return cloned


class ReadonlyPrefillSession(PrefillCacheSession):
    def __init__(self, session, counters, verify_integrity):
        super().__init__(session, counters)
        self.verify_integrity = verify_integrity
        self.base_values = None
        self.base_tensors = None

    def verify_base_integrity(self):
        if not self.verify_integrity or self.base_values is None:
            return
        for current, expected in zip(self.base_tensors, self.base_values):
            if current.shape != expected.shape or not self.session.b.torch.equal(current, expected):
                raise ValueError('A candidate forward modified a shared original prompt KV tensor')
        self.counters['actual_base_tensor_equality_checks'] += len(self.base_tensors)

    def next(self, prefix):
        prefix = tuple(prefix)
        if prefix:
            return self.session.next(prefix)
        if self.base_output is None:
            step = self.session.next(())
            self.base_output = copy.copy(self.session.output)
            self.base_cache = clone_full_attention_cache(self.session.output.past_key_values)
            self.base_rope = copy.deepcopy(self.session.rope_state)
            self.base_tensors = [tensor for layer in self.base_cache.layers for tensor in (layer.keys, layer.values)]
            if self.verify_integrity:
                self.base_values = [tensor.clone() for tensor in self.base_tensors]
            self.counters['actual_empty_prefix_prefills'] += 1
            return step
        self.verify_base_integrity()
        restored = copy.copy(self.base_output)
        restored.past_key_values = clone_full_attention_cache(self.base_cache)
        self.session.output = restored
        self.session.prefix = ()
        self.session.rope_state = copy.deepcopy(self.base_rope)
        self.counters['reused_empty_prefixes'] += 1
        return self.session.next(())

class PrefillCacheBackend:
    def __init__(self, backend, verify_integrity=False):
        self.backend = backend
        self.verify_integrity = verify_integrity
        self.counters = {'sessions': 0, 'actual_empty_prefix_prefills': 0,
                         'reused_empty_prefixes': 0, 'actual_base_tensor_equality_checks': 0}
        self.gate_session = None

    def session(self, *args, **kwargs):
        if kwargs.get('reference', 'clean') != 'clean':
            raise ValueError('Read-only ranking cache reuse is restricted to clean inputs')
        if self.verify_integrity and self.gate_session is not None:
            self.gate_session.verify_base_integrity()
            self.gate_session = None
        session = ReadonlyPrefillSession(self.backend.session(*args, **kwargs), self.counters,
                                        self.verify_integrity)
        if self.verify_integrity:
            self.gate_session = session
        self.counters['sessions'] += 1
        return session

    def verify_last_gate_session(self):
        if self.gate_session is not None:
            self.gate_session.verify_base_integrity()
            self.gate_session = None

    def stats(self):
        from transformers.cache_utils import DynamicLayer
        return {**self.counters, 'original_candidate_token_forward': True,
                'independent_cache_objects': True, 'readonly_shared_prefill_tensors': True,
                'tensor_integrity_gate_enabled': self.verify_integrity,
                'DynamicLayer_update_sha256': hashlib.sha256(inspect.getsource(DynamicLayer.update).encode()).hexdigest()}

    def __getattr__(self, name):
        return getattr(self.backend, name)
