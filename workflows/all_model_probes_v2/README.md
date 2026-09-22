# All-model knowledge probes v2

This workflow implements the user change recorded in `docs/current/PROTOCOL_AMENDMENT_20260922.md`. It does not modify the original frozen source, model specifications, census or annotation records.

## Scope and exact work count

All original 9,167 questions are retained, including development and evaluation splits, for all 16 models.

- Independent attempts: 9,167 x 10 x 16 = 1,466,720 generated responses.
- Food-101 closed scores: 4,848 x 16 = 77,568 score records.
- Total expected records: 1,544,288.
- Each closed record contains all 101 candidate class sequence scores. It is not equivalent to one model forward pass.
- VizWiz has independent attempts only. Closed Food scores do not establish anything about VizWiz.
- These runs do not select models or establish semantic correctness automatically. New independent responses still need the authorized semantic annotation/scoring pipeline.

Independent generation uses the existing prompt, decoder, 32-token budget, temperature 1, top-p 1, and ten deterministic independent seeds. All 16 x 9,167 groups were checked to have ten distinct seeds. Closed ranking calls the original `closed_rank` function and uses mean token log probability over all 101 class names, without adding EOS or changing tokenization.

## Efficiency and validation

A small wrapper memoizes deterministic CPU prefix logits. It does not copy or alter GPU KV caches and does not change checkpoint, precision, image processing, prompts, sampling or ranking. Each model's first actual Food image is compared with the unchanged functions: two complete independently sampled responses and all 101 closed scores must agree to within 1e-6 and have identical generated tokens and gold rank. Failure exits that model and preserves its evidence. There is no automatic fallback or retry.

The model stays loaded while ten independent answers and, for Food, one complete closed score record are generated per question. The original session implementation still reconstructs a non-extension prefix. Hence this memoization saves repeated measured prefixes but is not a native batched decoder or a fully branching KV implementation.

## Running processes and fixed lanes

Run name: `all16_20260922_v2`.

4028 project root: `/home/g203-4028/projects/knowledge-deficit-mitigation`.

- Scheduler PID 417944.
- GPU 0: qwen35_9b, worker PID 418119.
- GPU 1: qwen35_4b, worker PID 418120, then qwen25vl.
- GPUs 4 and 5: gemma3_12b, worker PID 418124, then llava15_13b and internvl35_8b.
- After the dual-card lane finishes, GPU 4 runs gemma3_4b, llava15_7b, onevision, llava16_mistral; GPU 5 runs minicpm26, minicpm45, phi35, llava16_vicuna.

6403 project root: `/home/team/zhanghaonan/TAFFC/knowledge-deficit-mitigation`.

- Scheduler PID 14359.
- GPU 1: qwen3vl, worker PID 14446, then glm46v.

All workers inherit original `worker.sh` physical GPU locks, UUID validation and their registered environments/device maps. Future PIDs are recorded in status.json when dispatched. The two previously migrated models remain on their authorized 6403 GPU 1.

These queues are fixed in the bound scheduler source. GPU 0 can become idle after qwen35_9b finishes while other lanes continue. Rebalancing must use an explicit new dispatch receipt and avoid killing/restarting active model workers or duplicating queued models; do not edit an already-bound runner/scheduler in place.

## Files and monitoring

Paths below are relative to the relevant host project root.

- Scheduler status: `outputs/records/probes_v2/all16_20260922_v2/status.json`.
- Model progress: `outputs/records/probes_v2/all16_20260922_v2/<model>_progress.json`.
- Actual numerical check: `outputs/records/probes_v2/all16_20260922_v2/<model>_memoization_verification.json`.
- Failure evidence: `outputs/records/probes_v2/all16_20260922_v2/<model>.errors.jsonl`.
- Model logs: `outputs/records/probes_v2/all16_20260922_v2/<model>.log`.
- Completion receipt: `outputs/records/probes_v2/all16_20260922_v2/<model>_complete.json`.
- Independent responses: `outputs/raw/probes_v2/all16_20260922_v2/<model>/independent.jsonl`.
- Closed scores: `outputs/raw/probes_v2/all16_20260922_v2/<model>/closed.jsonl`.
- Each output ledger has a matching `.identity.json` sidecar binding source, model, execution, full manifest and amendment.
- The central `task_expansion_verification.json` records all-model seed and task-count checks.

Progress advances after each whole question. A long first question also includes numerical validation, so status counters may temporarily remain zero while raw records are already written.

The scheduler preserves model failures and continues independent jobs. It never reruns a failed model. Both local schedulers report only host-level generation coverage; neither claims global scientific completion. Completion requires all 91,670 independent and 4,848 closed records per model, verified unique identities and every class score, followed by semantic annotation/analysis and remaining research work.

Use the existing hourly heartbeat to inspect process identity, counts, current errors and completed artifacts. Do not launch another scheduler for the same model set. Preserve and synchronize remote raw records, sidecars and completion evidence before central analysis.
