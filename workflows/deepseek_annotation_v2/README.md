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

