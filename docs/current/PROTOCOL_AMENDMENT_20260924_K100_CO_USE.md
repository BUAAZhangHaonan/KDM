# K100 GPU0 renewed use authorization, 2026-09-24

The user explicitly states that K100 has no active user workload beyond an approximately700MB resident process and may be used. This supersedes the earlier monitor policy that withheld KDM because of safa-facegen PID204268. Preserve that process; no termination or suspension is authorized or needed. KDM may share GPU0 with this resident process, while retaining physical UUID/device locks and all existing model-specific admission requirements.

First-five workers on4028 retain their full task ownership. Do not duplicate their remaining tasks onK100. Prefer an independently admitted missing measurement for the remaining eleven models when no verified disjoint first-five task is available. Qwen3VL independent answers are complete and must not be repeated. Current next-task engineering is assigned to the existing glm_closed_expansion agent; consult its actual versioned K100 dispatch before launching.

Two-hour monitoring, no paid API, and the4029 exclusion remain unchanged.
