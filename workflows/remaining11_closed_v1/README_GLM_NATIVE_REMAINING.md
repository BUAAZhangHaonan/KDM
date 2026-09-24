# GLM remaining native candidate scores

Active entry `glm_native_remaining_v1.py`, run `glm_native_remaining_20260924_v1`, 6403 physicalGPU1, UUID GPU-5c45e961-7442-eb90-9b8c-295a1cf14995.

This continues previously paused native candidate scoring. GLM and Qwen3.5-9B vLLM numerical admission failures are preserved and are not retried or silently accepted. Native checkpoint, environment, BF16, original image/template/class tokens, frozen closed_rank, and deterministic prefix-Step memoization stay unchanged.

Exactly1895 existing rows are verified and re-ledgered with original provenance from `food_closed_20260923_v3/glm46v/closed.jsonl`. Source identity, unique IDs, sample identity,101 finite scores, token means and gold ranks are checked. Only2953 missing rows are newly inferred. The first new row is compared with original non-memoized closed_rank at1e-6, as in the original verified workflow. No original output is modified.

Dispatch: `outputs/records/remaining11_closed_v1/glm_native_remaining_v1/dispatch.json`.
Progress/reuse/checks: `outputs/records/probes_v2/glm_native_remaining_20260924_v1`.
Raw: `outputs/raw/probes_v2/glm_native_remaining_20260924_v1/glm46v`.
No automatic restart or failure retry. Report reused and newly inferred counts separately. Completion remains candidate scoring only, not independent-generation annotation or final abstention GT.
