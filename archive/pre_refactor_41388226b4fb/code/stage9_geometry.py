"""Exact VCD score analysis at a fixed prefix; no model fitting or new decoder.

The target is one explicitly specified token sequence. Empty intervals never
certify absence of semantic knowledge or impossibility of alternative names.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable
import math
import numpy as np


def vector(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size < 2 or not np.isfinite(a).all():
        raise ValueError('Expected a finite one-dimensional score vector.')
    return a


def softmax(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if np.isnan(z).any() or np.isposinf(z).any() or not np.isfinite(z).any():
        raise ValueError('Scores must contain finite mass and no NaN/+inf.')
    e = np.exp(z - np.max(z)); return e / e.sum()


def logsumexp(z) -> float:
    a = np.asarray(z, dtype=np.float64)
    m = float(np.max(a))
    if not np.isfinite(m):
        raise ValueError('Empty or non-finite support.')
    return m + float(np.log(np.exp(a-m).sum()))


@dataclass(frozen=True)
class Interval:
    lower: float = 0.0
    upper: float = math.inf
    lower_closed: bool = True
    upper_closed: bool = False
    empty: bool = False
    reason: str = ''

    def contains(self, alpha: float) -> bool:
        if self.empty or not math.isfinite(alpha): return False
        left = alpha >= self.lower if self.lower_closed else alpha > self.lower
        right = alpha <= self.upper if self.upper_closed else alpha < self.upper
        return bool(left and right)

    def intersect(self, other: 'Interval') -> 'Interval':
        if self.empty: return self
        if other.empty: return other
        lo, hi = max(self.lower, other.lower), min(self.upper, other.upper)
        lc = ((self.lower_closed if lo == self.lower else True) and
              (other.lower_closed if lo == other.lower else True))
        uc = ((self.upper_closed if hi == self.upper else True) and
              (other.upper_closed if hi == other.upper else True))
        bad = lo > hi or (lo == hi and not (lc and uc))
        return Interval(lo, hi, lc, uc, bad, 'incompatible_constraints' if bad else '')

    def witness(self, cap: float = 4.0) -> float | None:
        i = self.intersect(Interval(0, cap, True, True))
        if i.empty: return None
        if i.lower == i.upper: return i.lower
        a = i.lower + (i.upper-i.lower) / 2
        return a if i.contains(a) else None

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ('lower_closed', 'upper_closed', 'empty'):
            d[key] = bool(d[key])
        d['lower'] = float(d['lower'])
        d['upper'] = float(d['upper'])
        # Strict JSON: represent an unbounded upper endpoint explicitly.
        if not math.isfinite(d['upper']): d['upper'] = None
        return d


def target_interval(z, reference, target: int, beta: float = .1) -> Interval:
    """All nonnegative alpha selecting target at this prefix (NumPy tie rule).

VCD scores are z + alpha*(z-reference), on the ORIGINAL plausibility set.
Bounds include exact open/closed endpoints, including argmax tie handling.
"""
    z, r = vector(z), vector(reference)
    if z.shape != r.shape or not 0 <= target < z.size or not 0 < beta <= 1:
        raise ValueError('Invalid dimensions, target or beta.')
    keep = z >= z.max() + math.log(beta)
    if not keep[target]: return Interval(empty=True, reason='target_outside_support')
    direction = z-r
    interval = Interval()
    for k in np.flatnonzero(keep):
        if k == target: continue
        intercept = float(z[target]-z[k])
        slope = float(direction[target]-direction[k])
        # A lower token index wins a tie in torch/NumPy argmax.
        allow_equal = target < k
        if slope == 0:
            if intercept < 0 or (intercept == 0 and not allow_equal):
                return Interval(empty=True, reason='target_never_overtakes_competitor')
            continue
        boundary = -intercept / slope
        if slope > 0:
            constraint = Interval(boundary, math.inf, allow_equal, False)
        else:
            constraint = Interval(-math.inf, boundary, False, allow_equal)
        interval = interval.intersect(constraint)
        if interval.empty: break
    return interval


def sequence_interval(steps: Iterable[tuple]) -> Interval:
    result = Interval()
    for z, r, token in steps:
        result = result.intersect(target_interval(z, r, int(token)))
    return result


def confidence_terms(z, reference, token: int, alpha: float, beta: float=.1) -> dict:
    """Exact log-probability identity, derivatives, and a local 2nd-order term.

log p_alpha(j) - log p(j) = -log P(S) + alpha*d_j -
                          log E_{p(.|S)} exp(alpha*d).
This is conditional on the SAME prefix and the SAME target token.
"""
    z, r = vector(z), vector(reference)
    if z.shape != r.shape or alpha < 0 or not 0 <= token < len(z):
        raise ValueError('Invalid arguments.')
    keep = z >= z.max()+math.log(beta)
    if not keep[token]:
        return {'retained':False, 'log_gain':None, 'support_log_gain':None}
    d = z-r
    # Use log probabilities throughout; don't clip a near-zero probability.
    lp = z-logsumexp(z)
    log_mass = logsumexp(lp[keep])
    lps = lp[keep]-log_mass
    ps = np.exp(lps)
    cgf = logsumexp(lps+alpha*d[keep])
    gain = -log_mass + alpha*d[token]-cgf
    star = np.where(keep, z+alpha*d, -np.inf)
    lpstar = star-logsumexp(star)
    q = np.exp(lpstar[keep])
    mean0 = float(ps@d[keep]); var0=float(ps@((d[keep]-mean0)**2))
    mean_a=float(q@d[keep]); var_a=float(q@((d[keep]-mean_a)**2))
    zmean = float(ps@z[keep]); zc=z[keep]-zmean; dc=d[keep]-mean0
    variance_z=float(ps@(zc**2))
    scale=float(ps@(zc*dc)/variance_z) if variance_z>0 else 0.0
    residual=dc-scale*zc
    energy=float(ps@(dc**2)); residual_energy=float(ps@(residual**2))
    return dict(retained=True, log_gain=float(gain),
                direct_log_gain=float(lpstar[token]-lp[token]),
                identity_error=float(abs(gain-(lpstar[token]-lp[token]))),
                support_log_gain=float(-log_mass),
                reference_log_gain=float(alpha*d[token]-cgf),
                derivative=float(d[token]-mean_a), second_derivative=-var_a,
                local_second_order=float(-log_mass+alpha*(d[token]-mean0)-.5*alpha**2*var0),
                local_scale=scale, direction_variance=energy,
                residual_variance=residual_energy,
                local_scale_energy_fraction=(float(scale**2*variance_z/energy) if energy>0 else None),
                confidence_before=float(np.exp(lp[token])),
                confidence_after=float(np.exp(lpstar[token])))
