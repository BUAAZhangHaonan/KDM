# K100 independent generation deployment

Authorized host RTX_Pro_6000 (`k100-X785-H30`), project `/home/k100/projects/knowledge-deficit-mitigation`, physical GPU0 UUID `GPU-4320e83b-1158-0546-73c5-b715c4dbe1ea`.

Use project-local `.environments/vllm024/bin/python`. `worker.sh` establishes fd20 GPU0 lock and project-only caches. Main independent_v5 entry imports `admission.register(model)` before loading GPU weights and uses its returned `model_paths`. Only LLaVA1.6-Mistral is admitted initially; it reuses `/home/k100/Models/llava-v1.6-mistral-7b-hf` read-only. The existing MiniCPM folder lacks checkpoint files and is not admitted.

Frozen source/registry/specs remain unchanged. Code is staged from central HEAD75a579c3d4fdcc3e7153e8096d1c8e5b7c1638da, frozen source committed locally as a deployment snapshot to preserve identical tracked blob IDs. All9167 manifest images were copied and SHA256checked; all5 closed cohorts and source references were copied with SHA256 verification. Receipts are under `outputs/records/independent_k100_v1`. No secret files or private API credentials were transferred.

This deployment is CPU/precondition readiness only. No GPU inference was launched by deployment. Native16, actual processor, EOS, distribution and prefix-cache throughput validation must pass on this new GPU/engine before production; separate new result identities must be retained. No old Transformer rows are merged into a new vLLM numerical cohort.

## Current production entry

Read `latest_dispatch.json` as the current authoritative run/path index. LLaVA fresh run is `panel5_independent_food_v5_spawn1`, using `worker_v3.sh` plus guarded `entry_v3.py`. The worker explicitly chooses supported vLLM `spawn`; original fork initialization stalled before any output. Actual frozen16-processor CPU reproduction stalled at the same128x32768 CPU tensor initialization under fork and completed under spawn; live ptrace was unavailable. Evidence and0-row interruption were preserved. No sampling/checkpoint/precision parameters changed, and unrelated SAFA processes were not touched.

`run_llava_food_v2.py` supervises this one GPU run, then invokes one free CPU postprocess bound to the fresh source record and writes labels under `outputs/annotations/independent_v5/panel5_independent_food_v5_spawn1/llava16_mistral`. Nonzero exits stop without automatic retries. Original `panel5_independent_food_v5` LLaVA is interrupted with0 rows and is not the active run. Earlier `worker.sh`, `worker_v2.sh`, `entry.py`, `entry_v2.py`, old failures and dispatches remain unchanged.
