# K100 GPU0 renewed use authorization, 2026-09-24

The user explicitly states that K100 has no active user workload beyond an approximately700MB resident process and may be used. This supersedes the earlier monitor policy that withheld KDM because of safa-facegen PID204268. Preserve that process; no termination or suspension is authorized or needed. KDM may share GPU0 with this resident process, while retaining physical UUID/device locks and all existing model-specific admission requirements.

First-five workers on4028 retain their full task ownership. Do not duplicate their remaining tasks onK100. Prefer an independently admitted missing measurement for the remaining eleven models when no verified disjoint first-five task is available. Qwen3VL independent answers are complete and must not be repeated. Current next-task engineering is assigned to the existing glm_closed_expansion agent; consult its actual versioned K100 dispatch before launching.

Two-hour monitoring, no paid API, and the4029 exclusion remain unchanged.

## Actual next task

LLaVA1.6-Vicuna candidate scoring completed4848 on6403, freeing GPU0. Its independent native16 reference is now running with the original registered FP16 environment: PID377651, `workflows/independent_vicuna_native_v1/export_native16_v1.py`, dispatch `outputs/records/independent_vicuna_native_v1/dispatch.json`. Exact checkpoint copying toK100 is in progress. K100 independent vLLM admission and48480-answer production follow only after the real reference/weight/input/EOS/sampling checks. Preparation is not production; inspect the latest dispatch before scheduling.

## K100 GPU admission started

All three Vicuna checkpoint shards (14,126,954,312 bytes) finished copying with full registered SHA checks. Independent native16 completed in the original6403 environment with native generate/frozen decoder argmax agreement16/16. K100 actual GPU vLLM16-image/160-answer audit started PID226983. Dispatch `outputs/records/independent_k100_vicuna_v1/verify_dispatch.json`; check `outputs/records/independent_k100_vicuna_v1/engine16/check.json`. Candidate completion4848 and native source evidence are centrally synchronized at `outputs/records/remaining11_sync_20260924/vicuna_closed_and_native_complete`. Admission is not yet production or a completed independent cohort.
