"""Reuse the identical empty-prefix prefill in original closed-set scoring.

The original HFSession still runs every candidate-token forward.  Each new
candidate receives an independent clone of the same prompt KV cache and rotary
state.  No labels, tokens, log-normalization or ranking rules are changed.
"""
from __future__ import annotations

import copy


class PrefillCacheSession:
    def __init__(self, session, counters=None):
        required = ("prefix", "output", "rope_state", "inputs", "ohs", "b")
        if any(not hasattr(session, name) for name in required) or session.ohs:
            raise TypeError("Prefill reuse requires the original HFSession without layer projections")
        if session.b.model.training:
            raise ValueError("Prefill reuse requires the registered eval-mode model")
        self.session = session
        self.base_output = None
        self.base_cache = None
        self.base_rope = None
        self.prefill_calls = 0
        self.reused_empty_prefixes = 0
        self.counters = counters

    def next(self, prefix):
        prefix = tuple(prefix)
        if prefix:
            return self.session.next(prefix)
        if self.base_output is None:
            step = self.session.next(())
            # The initial cache must be copied before the first candidate can
            # mutate a DynamicCache in an incremental original-model forward.
            self.base_output = copy.copy(self.session.output)
            self.base_cache = copy.deepcopy(self.session.output.past_key_values)
            self.base_rope = copy.deepcopy(self.session.rope_state)
            self.prefill_calls += 1
            if self.counters is not None:
                self.counters["actual_empty_prefix_prefills"] += 1
            return step
        restored = copy.copy(self.base_output)
        restored.past_key_values = copy.deepcopy(self.base_cache)
        self.session.output = restored
        self.session.prefix = ()
        self.session.rope_state = copy.deepcopy(self.base_rope)
        self.reused_empty_prefixes += 1
        if self.counters is not None:
            self.counters["reused_empty_prefixes"] += 1
        return self.session.next(())

    def __getattr__(self, name):
        return getattr(self.session, name)


class PrefillCacheBackend:
    """A candidate-scoring proxy over an unchanged registered model backend."""
    def __init__(self, backend):
        self.backend = backend
        self.counters = {"sessions": 0, "actual_empty_prefix_prefills": 0,
                         "reused_empty_prefixes": 0}

    def session(self, *args, **kwargs):
        if kwargs.get("reference", "clean") != "clean":
            raise ValueError("Closed-rank cache reuse is restricted to registered clean inputs")
        session = PrefillCacheSession(self.backend.session(*args, **kwargs), self.counters)
        self.counters["sessions"] += 1
        return session

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def stats(self):
        return {**self.counters,
                "original_candidate_token_forward": True,
                "independent_KV_clones": True}
