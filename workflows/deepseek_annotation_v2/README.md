# DeepSeek annotation v2

This independent workflow implements the explicit 2026-09-22 amendment. It never modifies the frozen source or old Ministral annotations.

- Prepares exactly 293344 original census answers from the 16 source ledgers, checking each original sidecar and complete per-model/dataset/prompt-condition coverage.
- Sends **every** answer to the exact configured API model, including literal UNKNOWN and empty answers. API inputs contain only the original question and answer; no subject model, prompt condition, image, or ground truth.
- Runs up to 256 fixed async workers. No SDK retries, model fallback, rule bypass, automatic failed-record retry, or blanket human approval gate.
- Stores the exact response body (including returned model, fingerprint and usage) and validation failures. Accepts only one complete JSON object with the four agreed behavior labels and exact original evidence/answer spans.
- Auth/model/rate-limit/server errors stop further dispatch; already in-flight requests finish and are retained. SIGTERM/SIGINT also drain in-flight calls.
- Request journal is flushed before sending. Explicit resume skips successes, errors and uncertain started requests. A crash cannot silently cause a duplicate paid request; indeterminate requests remain unresolved.
- Derived metrics keep full denominators, separately by model/dataset/guided/dev/eval/all. Frozen Food alias and official VizWiz scoring are used. Missing annotation yields explicit lower/upper bounds, never a dropped row.
- API judgment of behavior is not a claim of correctness, knowledge, abstention justification, or human review.

Run locally without paid API calls:
```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python workflows/deepseek_annotation_v2/test_annotate.py
PYTHONDONTWRITEBYTECODE=1 venv/bin/python workflows/deepseek_annotation_v2/annotate.py --root "$PWD" --prepare-only
```

Launch (key file must be project-local, gitignored, mode 0600):
```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -u workflows/deepseek_annotation_v2/annotate.py \
  --root "$PWD" --out outputs/annotations/deepseek_v2/census \
  --endpoint https://api.deepseek.com --model deepseek-flash --response-model deepseek-flash \
  --thinking disabled --max-tokens 512 --concurrency 256 \
  --key-file "$PWD/cache/private/deepseek_api_key"
```

Only use `--resume` deliberately with the exact same immutable identity. To retry failed requests requires a separately authorized new attempt, not a hidden flag. Initial implementation deliberately has no retry option. Frozen selection downstream does not automatically accept v2 labels; protocol integration must be explicit.

Canonical evidence: queue.jsonl, queue.sources.json, identity.json, requests.jsonl, results.jsonl, invocations.jsonl.
Mutable status: status.json. Derived terminal exports: labels.jsonl, errors.jsonl, metrics.json.


## Completed first pass and extra-field reconciliation (2026-09-22)

All 293344 original requests finished. The initial validator accepted 292748 and retained 596 errors. Of these, 445 JSON responses contained exactly the required three fields plus an auxiliary `type` field. `reconcile_extra_type.py` removes only that extra field, checks the unchanged three values with the original validator, and creates a distinct derived annotation identity. It makes no API requests, performs no semantic relabeling, and preserves the original results/errors.

Current derived evidence lives at `outputs/annotations/deepseek_v2/census/reconciled_extra_type_v1/`: `reconciliation.json`, `identity.json`, `projection_records.jsonl`, `labels.jsonl`, `errors.jsonl`, `metrics.json`, and `SUMMARY.md`. There are 293193 validated labels and 151 unresolved records. This is not a fully complete annotation set and is not human-reviewed. No failed requests have been retried.

Original and derived ledgers have lossless gzip shards by original model in `outputs/annotations/deepseek_v2/census/archived_by_model/`. Its manifest records counts, original hashes and compressed-artifact hashes; all uncompressed originals remain preserved on 4028. Use the derived metrics for the latest full-denominator statistics. Original `status.json` deliberately keeps its original 596 failures instead of rewriting history.

## User-authorized one-time retry of 151 failures

The user explicitly authorized exactly one retry of the remaining 151 records. `retry151_once.py` verifies each request body matches its original request (same model, prompt, temperature, thinking mode and maximum output length), submits each key once, and refuses an already-created run directory. Evidence is under `census/retry151_once_v1/`. All 151 requests completed: 43 passed and 108 remain unresolved. No further retry is authorized or performed.

`merge_retry151.py` creates `census/merged_after_retry151_v1/`, preserving all prior validated judgments and adding only the 43 successful retry labels. The latest total is 293236 validated labels out of 293344, with 108 unresolved. Use its `merge_receipt.json`, `metrics.json` and `SUMMARY.md` for current counts and rates. `labels_delta.jsonl` plus the earlier reconciled per-model archives reconstruct the merged labels; the full materialized JSONL remains on 4028. All older failed responses remain intact. No human review or complete annotation claim is made.
