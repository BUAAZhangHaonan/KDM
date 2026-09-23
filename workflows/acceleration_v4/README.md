# Five-model acceleration and formal Food study

Authority: docs/current/PROTOCOL_AMENDMENT_20260923_ACCELERATION.md. The fixed panel is qwen25vl, qwen35_4b, llava16_mistral, minicpm26, gemma3_4b. Other models and VizWiz are deferred, not deleted.

## Live dispatch (2026-09-23)

- 6403 GPU1: qwen35_4b vLLM closed scores, run panel5_engine_v2; then MiniCPM v3 integration/preflight and closed production only if its real reference gate passes.
- 6403 GPU0: qwen25vl vLLM closed scores, run panel5_engine_v2; then LLaVA v5, then Gemma v6 with corrected attention-backend selection, each gated before production.
- Engine queue: outputs/records/acceleration_v4/engine_queue_status_v3.json on6403, with exact commands in engine_queue_config_v3.json. Linux pidfd waits, no busy polling and no retries. A failed lane stops; old evidence stays.
- 4028 physical GPU0/1/4/5: original formal methods for Qwen2.5/LLaVA/MiniCPM/Gemma respectively, run panel5_food_formal_v4. Qwen3.5 formal has moved to4029 GPU4/5/6/7 in four disjoint shards (panel5_food_formal_4029_v1). The old4028 CPU waiting queue was deliberately stopped; its old status file is historical. See workflows/acceleration_4029_v1/README.md.
- Old v3 schedulers and old selected/nonselected slow candidate workers have been deliberately paused. Old status.json may still say running; use actual PIDs and handover receipts.

## Entrypoints and identities

export_native_reference.py exports unchanged native image/token/pixel and101-class score references. Five complete reference files are under outputs/records/acceleration_v4/native_reference. Qwen3.5 reuses four actual old closed records, rather than rerunning them.

vllm_closed_v2.py uses batched candidate-prefix trees, shared image/prompt cache, raw next-token log probabilities for every original label token, and the original mean-logprob ranking. It never replaces101-class scoring with generated class text or first-token scoring. BF16, model, prompt, processor and full image remain fixed. New vLLM rows have new identities; no mixing with old Transformers rows or claim of bitwise equivalence.

vllm_closed_v3.py only adapts the existing MiniCPM tokenizer's read-only special-token properties by validating their values. vllm_closed_v4.py additionally selects TRITON_ATTN for Gemma to preserve its image bidirectional mask. Earlier software failures and Gemma discrepancy remain in panel5_engine_v1/v2. v5 adds the native leading BOS only to LLaVA engine requests. v6 uses a project-local Gemma worker to record the actual nonempty Triton image ranges and rejects missing evidence. Gemma v6 is pending real GPU validation, not a completed pass.

The first preflight compares an external native processor reconstruction, IDs and candidate tokens. Do not interpret that alone as proof of vLLM internal image processing. The CPU actual-engine processor audit is separate; GPU four-reference scoring and top1/closed-correctness checks remain required. Small reference checks do not prove equality on all4848 images. Numerical differences are recorded.

formal_runner.py uses the unchanged frozen run_tasks and DecodeConfig. All original methods and4x4 conditions remain853248 tasks on2424 Food eval questions. Order: UNKNOWN main results, UNKNOWN controls, remaining prompt matrix. VCD/M3ID/CDA need multi-condition full logits, DoLa/DeCo need internal layers, and SID needs real attention intervention; an ordinary vLLM text endpoint is not an equivalent implementation.

formal_postprocess.py accepts completed, verified shards only and reuses workflows/quick_match_v1. Example:

    venv/bin/python workflows/acceleration_v4/formal_postprocess.py --shard outputs/records/acceleration_v4/panel5_food_formal_v4/qwen25vl/shard_000_of_001 --out outputs/annotations/acceleration_v4/FRESH_SCREEN

It separates exact correct/wrong matches, exact abstentions and unresolved rows. There are no paid judge calls or human-review claims. Closed scoring alone is not the original joint abstention GT; ten independent answers and their validated labels are still needed. Any ordinary-generation engine entrypoint must pass its own prompt/EOS/token-distribution checks before scheduling.

## Monitoring and completion

Read actual workers and new progress/error/complete records once per host. Do not restart old v3, duplicate running jobs, retry failures or submit API requests. Sync completed6403 raw files and sidecars to4028; verify unique sample/replicate coverage, finite101 scores, input identities and hashes. Main-stage progress is not whole-model completion; whole-model generation is not final semantic labeling or GT. Full study acceptance still includes mechanism tests, uncertainty intervals, actual figures and paper materials. Keep the existing two-hour monitor; no continuous goal loop.

See docs/current/ACCELERATION_RUNTIME_20260923.md for measured results and limits.

Prepared independent entry: vllm_independent.py (default CPU plan only; not ready to enqueue). Stage-only free matching: formal_stage_postprocess.py, requiring the exact finished-stage task set even while later stages run. See each --help for evidence requirements.
