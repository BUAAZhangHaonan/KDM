Phi Food candidate parallel continuation, 2026-09-30

The original single-card claim `4029_phi35_candidate_inputs_v2` stopped after a
complete fsync/close input boundary. Its raw, owner, full 4848-key interval and
admission remain unchanged. CPU validation sealed 498 completed rankings and
retained exactly 4350 keys. This administrative stop is separate from scientific
failures; no OOM retry, precision, processor, ranking rule, or parameter changed.

`retirement.py` records an immutable source bundle, original scientific plan and
six disjoint replacements. Their union is exactly the original remaining keys.
Only an explicitly named replacement, registered host/card and equal CPU plan
can pass dispatch and generator ownership checks. Retirement markers are appended;
old markers and failed preflight receipts remain available.

The active bundle is
`outputs/supplemental/remaining11/run_20260930_140337/continuations/phi_candidate_parallel_inputs_v2/retirement.json`.
Six single-card workers use 4028 physical 0/1/4/5 and 4029 physical 1/2. Each load
requires 16384 MiB free, based on the original rank process's 10750 MiB usage plus
5634 MiB fixed headroom. It is a capacity guard, not a newly measured peak.
Other users' processes remain unchanged.

Actual CPU verification expanded all registered keys, confirmed the six partitions
(705, 694, 758, 723, 722, 748), imported the real registered framework before Popen
and passed all six existing generation tests. The first observed new ranks were
19 complete rows across the six workers, with no failure. Per-worker timing was
46.1–62.3 seconds per rank from only 3–4 rows; a roughly 13-hour slowest-shard
estimate applies only conditionally to rank work, not generation methods or VizWiz.

The independent d4030 registry and dual-card wrapper live under `remaining4/`.
Its CPU identity and first-eight VCD plan passed. GPU admission and new InternVL
claim ownership are held by the root operator; the old InternVL formal OOM stays
archived and is not automatically resumed.
