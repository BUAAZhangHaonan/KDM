# P1 attribution details

`run_attribution_details.py` defaults to CPU-only planning. It validates the
sealed 2,836-row four-branch input package, the 198-row diagnostic CSV, and the
fixed 12-case list. It computes diagnostic token positions from the actual
guided-VCD/IP token LCP and records no semantic labels.

Example plan check:

```text
python workflows/paper_core/run_attribution_details.py \
  --inputs task/core/attribution_inputs/records.jsonl \
  --diagnostic-csv <sealed-198-row-csv> \
  --cases <sealed-case12-jsonl> \
  --prior-budget-dir outputs/paper_core_20260930/<prior-run> \
  --phase representative --phase diagnostic --phase cases \
  --plan-out outputs/paper_core_20260930/attribution_details_plan.json
```

The explicit execution path calls the existing admission/backend worker,
carries forward each model's budget, and charges loading and failed calls.
Planning does not start GPU work, create free guided responses, re-encode saved
text, or substitute labels for missing token evidence.

After review, an admitted worker can select one or more phases with
`--execute`, for example `--phase representative --phase diagnostic`, and must
also provide `--model`, `--run-id`, `--physical-gpus`, `--diagnostic-csv`, and
the sealed `--cases` JSONL when one is supplied. The representative manifest
is the frozen 101-row file; its five-model join is the 505-item panel. The
diagnostic manifest has 198 rows. When no separate case file is supplied, the
entry point derives the twelve case IDs only from packet rows whose frozen
`purpose` is exactly `case12`, retaining their source keys; it never invents
case content.

The actual r2 run completed 505 natural-reference generations, 198 diagnostic
ledger rows, and 12 cases with two frozen response paths each. Nine diagnostic
rows have no observed divergence and retain that status. The diagnostic worker
uses actual guided-VCD/IP and guided-Direct/VCD LCPs, then the real unguided/guided
Direct LCP when neither priority pair differs; it never fabricates a position.
Prior completed rows remain immutable and their source identity is preserved.

The independent CPU verification in `verify_attribution_exports.py` closed 505
representative positions, 208 diagnostic positions, and 101 complete-case
positions, checking prompt/prefix/noise evidence and float64 formula closure.
Actual coverage, budget, one verified numerical argmax tie, and limitations are
recorded in `docs/paper_core/attribution_finite_20261001.md`. The 505 natural
reference responses are scored by `score_attribution_reference.py`; five true
boundary QA judgments were reviewed and saved separately by root.
