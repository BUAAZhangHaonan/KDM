# Remaining census labels and two formal fragments

Authorized by the user on2026-09-24. The110 blind inputs are108 remaining original sixteen-model census responses and2 previously unresolved Qwen2.5 formal-control fragments. This is separate from the new independent-answer Luna queue.

The existing GPT-6 Luna medium subagent reads all110 question/answer pairs without model/method/reference answers. prepare.py verifies the108 exact error keys and preserves source mappings separately. finalize.py validates real authored results, exact spans, execution receipts and source identity; it adds only validated census labels into a fresh merged view and reuses unchanged frozen scoring. No DeepSeek call or inference rerun.

The two original formal fragments receive zero correctness credit as explicitly clarified by the user. Behavior labels are a separate question; any true remaining behavior ambiguity must not imply that the correctness is unknown. The new formal view retains all26664 stage rows with the9047 semantic rows and derives updated per-condition metrics. Originalv3 and its latest pointer remain unchanged because ongoing independent-annotation preparation binds those exact historical hashes. New current fragment-corrected statistics are under outputs/annotations/luna_census_remaining_v1/remaining108_plus2_20260924/formal_fragments_resolved_v1. Its completion.json becomes the receipt only after actual validation.

No automatic label is described as human review or final should-abstain GT. Old raw replies, old errors, old labels and original scoring remain preserved.

Run the CPU-only finalizer with the project runtime: `PYTHONDONTWRITEBYTECODE=1 venv/bin/python workflows/luna_census_remaining_v1/finalize.py`. System Python lacks httpx required by the imported validation module; the initial system-Python import failure occurred before any output creation or API call.
