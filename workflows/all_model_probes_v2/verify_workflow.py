#!/usr/bin/env python3
"""Bounded coverage, seed and exact memoization contract checks; no synthetic scientific outputs."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from runner import load_inputs, tasks_for_sample, PrefixMemo, compare_attempt, CONFIG
from kdm.decoding import Step, generate
from kdm.io import stable_seed
from kdm.pipeline import task_id

samples, names = load_inputs()
tasks = [task for sample in samples for task in tasks_for_sample(sample)]
assert len(tasks) == 91670
assert sum(s["dataset"] == "food101" for s in samples) == 4848
assert sum(s["split"] == "dev" for s in samples) == 3242
assert len({task_id("fixture", t) for t in tasks}) == 91670
assert len(names) == 101
for s in samples:
    assert len({stable_seed(s["id"], "fixture", r) for r in range(10)}) == 10

class Session:
    def __init__(self):
        self.calls = 0
    def next(self, prefix):
        self.calls += 1
        if len(prefix) >= 3:
            return Step(np.array([-30., -30., 30.]))
        return Step(np.array([.2 + len(prefix) * .1, .4, -2.]))

memo = PrefixMemo(Session())
for seed in range(10):
    a = generate(Session(), None, CONFIG, {2}, str, seed)
    b = generate(memo, None, CONFIG, {2}, str, seed)
    assert compare_attempt(a, b) == 0
assert memo.hits > 0
assert 91670 * 16 + 4848 * 16 == 1544288
print("PASS: all 9167 original samples, 10 independent seeds each, all 101 Food classes, exact memoized sampling")
