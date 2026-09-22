# Cheap preliminary text matching

This independent CPU-only command screens census responses, independent responses, and Food-101 closed candidate scores without any API calls. Original records and frozen scoring files are read-only. Every derived row binds to its source key, identity, line, and SHA-256. It never claims semantic judging, human review, or final abstention GT.

Rules:
- Exact whole-answer abstention markers use the frozen `lexical_label` rule.
- Food answers equal to one frozen class alias after frozen normalization are marked preliminary correct/incorrect. No substring search or guessing which phrase a long answer endorses.
- VizWiz whole answers use the vendored official normalizer and frozen consensus score. Partial credit remains explicit.
- Other text is `needs_confirmation`, not automatically incorrect. Empty text is also flagged for confirmation.
- Closed Food records require 101 finite scores and the original mean-log-probability ranking. Unique top1 is compared to gold; top1 ties remain unresolved. These are top1 correctness labels, not the final protocol's should-abstain threshold or GT.

From project root, for completed/stopped files:

```bash
venv/bin/python workflows/quick_match_v1/match.py \
  --input outputs/raw/current/qwen25vl/census.jsonl \
  --input outputs/raw/probes_v2/all16_20260922_v2/gemma3_12b/independent.jsonl \
  --input outputs/raw/probes_v2/all16_20260922_v2/gemma3_12b/closed.jsonl \
  --out outputs/annotations/quick_match_v1/example
```

Output must be fresh and within project `outputs/`. `--input` can repeat; duplicate keys cause failure. An input modified during a full scan causes failure with a preserved partial file. Do not scan growing live files as a complete dataset. Use `--limit 3` only for a small schema snapshot: its receipt explicitly marks the limited scope. `labels.jsonl` holds initial labels; `summary.json` includes complete screened denominators, per-model/dataset/split/kind/guided counts, rules hashes, and input digests. Preserve unresolved rows when calculating any rates. Model selection, final GT aggregation, and formal acceptance remain separate steps.
