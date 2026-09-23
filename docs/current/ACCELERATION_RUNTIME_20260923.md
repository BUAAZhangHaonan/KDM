# Acceleration deployment evidence - 2026-09-23

This is an interim runtime report, not a completed study or a guarantee of meeting a conference deadline.

## Measured speed and correspondence

A sixteen-image Qwen3-VL engineering test on the same6403 A100 reduced101-class scoring from315.562s to28.228s (11.179x, model loading excluded). All16 top classes agreed; three gold ranks changed. Maximum mean-logprob difference was0.4157, so the engines are not numerically identical. Qwen3-VL is an engineering test, not a member of the fixed five-model panel.

For selected panel models, four real native references were compared before production. Qwen2.5 and Qwen3.5 passed exact reconstructed inputs/tokenization and all four top-class/closed-correctness checks; scores are stored as a separate engine cohort. Their four-image totals improved7.015x and4.931x including initial per-image kernel warmup. These panel comparisons include a3090-to-A100 hardware move, so they are not isolated engine speedups.

At2026-09-23 05:07:47 UTC, Qwen2.5 had538/4848 new closed rows, with recent100-row mean0.682s/image. Qwen3.5 had206/4848, mean2.160s/image. If those recent rates continue, remaining scoring alone is about49min and2.8h respectively. This excludes waiting, untested model integrations, validation, independent answers, formal interventions and paper preparation. Do not extrapolate those two rates to the entire five-model study.

Sources: outputs/records/acceleration_v4/qwen3vl_vllm_tree_benchmark.json; panel5_engine_v2/{model}/benchmark.json; runtime_snapshot_6403.json. The snapshot records raw row counts and capture time; progress remains live only on the running host.

## Failures preserved and fixes bounded

- Qwen3.5 engine v1 completed its numerical benchmark but hit missing top-level source_blobs in a reused reference schema before production. v2 reads the original nested identity, preserving v1 error and all old data; v2 is producing real rows.
- MiniCPM v2 stopped during CPU tokenizer setup because its existing special-token IDs are read-only properties. v3 validates those existing properties rather than overwriting them. It is queued, not GPU-validated yet.
- Gemma v1 had a real numerical/top-class discrepancy and was blocked before production. Inspection found vLLM's child text configuration loses the multimodal-prefix flag while auto-selecting FlashAttention2, whose code path lacks the image bidirectional mask. The outer processor still supplies the correct image range. v6 explicitly requests TRITON_ATTN and records the real nonempty attention-range metadata with a project-local observing worker, which consumes that range; the unchanged real-score gate must pass before production. This correction is queued and has not passed a new GPU test yet. The observer does not change masks or kernel scores; its CPU synthetic test is not passed off as a real GPU receipt.

No OOM fallback, lower precision, smaller image, dropped class or relaxed numerical gate was used. Shared model and environment files remain read-only.

## Formal study now running

At05:08:48 UTC,4028 native workers had Qwen2.5:663, LLaVA:274, MiniCPM:562, Gemma:154 real formal responses, all in UNKNOWN-main stage with no recorded failure. Corresponding GPUs are0,1,4,5. Qwen3.5 formal is queued after the faster measured Qwen2.5 lane. Four-model main tables can begin before the fifth finishes, without changing the five-model selection rule.

The full task count remains853248 (including all original4x4 settings); neither that count nor a successful launch establishes throughput for later, more expensive methods. The first main stage contains14544 rows/model for Qwen2.5 and LLaVA,12120/model for MiniCPM/Gemma/Qwen3.5. Remaining formal timing will be estimated by actual method/stage rates, not candidate-score throughput.

CPU audits now confirm the actual vLLM processor inputs for all five models, after normalizing only the native FP32-to-BF16 input cast and equivalent MiniCPM tensor container layout. LLaVA initially omitted its leading BOS token; v5 supplies that token to the engine only, yielding exact native expanded IDs on all four references. The original prepare/native reference is unchanged. Evidence: actual_engine_input_audit_v1.json and cpu_processor_audit_v2/v3/v4. This CPU proof does not replace real GPU score or Gemma mask checks.

The full five-model Food independent-answer component would be242400 new responses. It has not been silently restarted in the old slow pipeline. Fast independent generation requires its separate prompt/EOS/distribution validation. No new DeepSeek or Ministral calls have been started. Zero-API matching is a preliminary screen; unresolved cases and missing joint GT remain explicit.

## Scope and scientific limits

Selection was fixed before intervention outcomes, using family, model size and baseline abstention coverage: Qwen2.5-VL7B, Qwen3.5-4B, LLaVA1.6-Mistral7B, MiniCPM2.6, Gemma3-4B. Gemma's near-zero baseline abstention remains a boundary case and must not be presented as a strong estimate of abstention-retention improvement. Five-model Food results do not establish VizWiz or16-model conclusions.

Preserved deliverables still include full formal-method comparisons, mechanisms, genuine results/figures and paper materials. Current deliverable is a measured acceleration deployment and bounded queue, not completion of these future results.
