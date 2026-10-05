"""Natural termination and frozen positive-budget behavior, without a GPU."""
import numpy as np
import pytest

from kdm.cda import generate_cda
from kdm.decoding import DecodeConfig, Step, generate


class SequenceSession:
    def __init__(self, eos_position=40, failure_at=None):
        self.eos_position = eos_position
        self.failure_at = failure_at
        self.calls = []

    def next(self, prefix):
        self.calls.append(prefix)
        if len(prefix) == self.failure_at:
            raise RuntimeError("fixture context exhausted")
        logits = np.array([-8., 8.]) if len(prefix) < self.eos_position else np.array([8., -8.])
        return Step(logits, {0: np.zeros(2)}, {1: logits.copy()})


METHODS = [
    "direct", "vcd", "m3id", "dola", "deco", "sid", "icd",
    "instruction_vcd", "instruction_m3id", "cda_visual",
]


def run(method, budget, eos={0}, failure_at=None):
    sessions = [SequenceSession(failure_at=failure_at)] + [SequenceSession() for _ in range(4)]
    cfg = DecodeConfig(method=method, max_tokens=budget)
    if method == "cda_visual":
        out = generate_cda(*sessions, cfg, eos, str, seed=17)
    else:
        out = generate(sessions[0], sessions[1], cfg, eos, str, seed=17, neutral_main=sessions[2])
    return out, sessions


@pytest.mark.parametrize("method", METHODS)
def test_eos_only_reaches_natural_eos_after_legacy_budget(method):
    out, sessions = run(method, None)
    assert out["tokens"] == [1] * 40 + [0]
    assert out["terminated"]
    assert len(out["trace"]) == len(out["selected_log_probabilities"]) == 41
    assert sessions[0].calls == [tuple(out["tokens"][:n]) for n in range(41)]
    assert np.isfinite(out["selected_log_probabilities"]).all()
    if method == "cda_visual":
        assert all(s.calls == sessions[0].calls for s in sessions)


@pytest.mark.parametrize("method", METHODS)
def test_positive_budget_preserves_exact_old_stop_and_prefix(method):
    capped, _ = run(method, 32)
    natural, _ = run(method, None)
    assert capped["tokens"] == natural["tokens"][:32] == [1] * 32
    assert capped["trace"] == natural["trace"][:32]
    assert capped["selected_log_probabilities"] == natural["selected_log_probabilities"][:32]
    assert not capped["terminated"]


@pytest.mark.parametrize("method", ["direct", "cda_visual"])
@pytest.mark.parametrize("budget", [0, -1])
def test_invalid_positive_budget_is_rejected(method, budget):
    with pytest.raises(ValueError, match="Positive token budget"):
        run(method, budget)


@pytest.mark.parametrize("method", ["direct", "cda_visual"])
def test_eos_only_requires_a_real_eos(method):
    with pytest.raises(ValueError, match="requires an EOS token"):
        run(method, None, eos=set())


@pytest.mark.parametrize("method", ["direct", "cda_visual"])
def test_context_failure_propagates_without_becoming_a_completed_result(method):
    with pytest.raises(RuntimeError, match="fixture context exhausted"):
        run(method, None, failure_at=4)
