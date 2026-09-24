# 2026-09-24 resource utilization and annotation progress

The user's direct resource instruction is enacted in `PROTOCOL_AMENDMENT_20260924_RESOURCE_UTILIZATION.md`. GPU inference does not wait for annotation acceptance. Do not restore old queues or access4029.

## Actual new work

- 6403 GPU0 Qwen3-VL vLLM closed scores: PID334841, 16 native comparisons pass input and top1/gold-correctness gate; benchmark9.34x, not a full-dataset speed guarantee. Production has actual unique101-class finite rows. Records `outputs/records/remaining11_v1/qwen3vl_closed_food_20260924_v1/qwen3vl`; raw `outputs/raw/remaining11_v1/qwen3vl_closed_food_20260924_v1/qwen3vl/closed.jsonl`. Full target4848, a separate engine cohort.
- K100 GPU0 Qwen3-VL vLLM independent answers: supervisor206830, worker206831, engine207296. Native16 exact greedy and actual input parity, 160 sampling checks and all checkpoint-shard SHA256 comparisons passed. Source records `outputs/records/independent_k100_qwen3vl_v1`, run `extra11_independent_food_qwen3vl_v1`, target48480. Startup actual590 rows/59questions verified. Candidate completion is explicitly pending; a versioned postprocessor must bind the completed new closed identity, not pretend the old full-closed dependency exists.
- 6403 GPU1 GLM native missing closed scores: PID350937, active original backend. Target2953 new plus1895 verified reused records. Records `outputs/records/probes_v2/glm_native_remaining_20260924_v1`, dispatch `outputs/records/remaining11_closed_v1/glm_native_remaining_v1/dispatch.json`. Actual GPU loading/execution verified. GLM-vLLM and Qwen3.5-9B-vLLM did not pass the strict rank gate; failures retained and no production or automatic retry on those entries.
- 4028 retains all four original formal workers. Qwen3.5 full155136 complete. Other models' main stages are complete, controls/matrices continue. No automatic new overlapping formal shards.

These PIDs are launch evidence only; every monitor must check the source host's actual process and current raw/progress files. When any resource is free, advance an eligible missing measurement or safe first-five formal task; do not idle it merely because semantic acceptance is pending. Individual engine admission remains required.

## Three-model independent annotation export now completed

`outputs/annotations/luna_independent_v2/completed3_20260923_v1/validated_v1` was exported using the unchanged finalizer. Its145440 input answers have145248 valid automatic labels and192 unresolved (Qwen25:8, Mini:11, LLaVA:173). All499 original batches plus corrections are bound to actual execution provenance. Root independently verified1469 referenced original-session event hashes/call IDs for recovered168–333; the canonical execution retains the original pilot-only receipt. Batch161's actual blind reread and four span corrections remain explicit; this is not a comprehensive semantic audit guarantee.

Accepted-export record is `accepted_export_v1.json` in that base. `semantic_acceptance_current.json` now resolves the historical161 hold and168–333 provenance gate. Do not repeat these completed batches. Automatic labels are not human review or final abstention GT. Question-level preliminary evidence:14544questions,9228 with an observed correct independent answer,3866 joint-deficit supported,1369 closed rank1,81 with unresolved independent answers.

The v3 CPU preparation tools are deployed for completed Qwen3.5/Gemma answers. Check the actual fresh queue under `outputs/annotations/luna_independent_v3/qwen35_gemma_semantic_20260924_v1`; do not mistake the earlier ready directory for completed labeling. Reuse exact valid judgments, preserve held unresolved, and assign only new blind groups to actual GPT-6 Luna medium after a real first-batch quality check. This semantic work does not block GPU scheduling.
