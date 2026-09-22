# Protocol amendment authorized by the user on 2026-09-22

This revision supersedes the original execution order, annotation model, and independent-probe sample scope. The original frozen sources, manifests, census raw records, failed attempts, and existing annotation ledgers remain unchanged as historical evidence. This amendment does not assert any scientific outcome.

## Automatic annotation

The user stopped the local Ministral annotation service and authorized DeepSeek V4.1 Flash through the official API with up to 256 concurrent requests. All 293,344 original census responses receive a new automatic judgment; no old local-model labels are accepted as DeepSeek labels. The registered official endpoint is https://api.deepseek.com/chat/completions and the requested/returned model is deepseek-flash. The API's server fingerprint, raw response, usage, time, prompt version, source task key, and exact original answer are retained. API credentials remain in a permission-600 ignored private file, never in version control or receipts.

The behavioral labels and fixed dataset correctness definitions stay unchanged. The annotation judge reads only the question and original answer, does not infer correctness or model knowledge, and sees no generator/model/method identity. Schema validity is not proof of semantic correctness. Classification errors remain explicit. No success may be fabricated or silently skipped. No implicit retry of old failures. New output identities distinguish this work from the prior local model.

Every response is automatically annotated; a blanket requirement for every response to have a human approval is withdrawn in accordance with the user's instruction. Genuine human checks, if performed, must remain identifiable and may not be fabricated by a model. Automated quality checks and unresolved disagreements must be reported separately. Future selection/reporting must use the new automatic annotation identity and retain unresolved counts rather than call all results human-verified.

## All-model independent behavior tests now run in parallel

Do not wait for annotation, model selection, or human review to start the following tests. Include every one of the fixed 16 candidate models and all original 9,167 samples, preserving each original dev/eval tag.

1. Independent answers: 10 separately seeded attempts for every model/sample. Total 16 * 9,167 * 10 = 1,466,720 generated answers. Preserve temperature 1, top_p 1, original neutral best-estimate prompt, 32-token limit, actual tokens, termination state, and original deterministic seed derivation.
2. Food-101 class scores: all 4,848 Food-101 samples per model, all 101 original class names, same mean-token-log-probability ranking. Total 16 * 4,848 = 77,568 records, each holding 101 candidate scores. These are not 77,568 single-token forward passes.
3. Combined conceptual task count is 1,544,288. No equivalent-runtime claim follows from dividing this count by census response count. Closed-set scoring applies to Food-101 only. VizWiz's external answerability and repeated-answer performance remain separate evidence sources.

These tests measure each model/sample once per declared attempt or class-score record. Their results are reused across later method comparisons and are not rerun ten times for every intervention answer. Full-data collection does not silently replace original eval-only denominators of final formal comparisons; dev/eval are reported distinctly.

## Resources and no-retry boundary

4028 project root /home/g203-4028/projects/knowledge-deficit-mitigation uses physical GPUs 0,1,4,5 only, with existing worker.sh locks and already registered model/device mappings. The user separately confirmed Qwen3-VL and GLM continue on 6403 physical GPU1 in /home/team/zhanghaonan/TAFFC/knowledge-deficit-mitigation using the previously verified checkpoints/environments. Existing models/environments are read-only. No CPU/disk weight offload, unapproved precision/input/batch/algorithm changes, or silent retries after failures. Preserve all failures and continue independent authorized work where safe.

## Implementation and acceptance

New versioned entry points live under workflows/deepseek_annotation_v2 and workflows/all_model_probes_v2. They use the unchanged frozen source tree as their algorithmic reference while explicitly defining the revised scope and recording their own source identity. Old artifacts are not overwritten or relabeled. A model/host/panel is complete only when all expected unique task identities and required 101-class scores are verified. Final research acceptance still requires actual experiments, all required scores, traceable real tables/figures, and evidence-consistent manuscript deliverables.

Hourly monitoring remains hourly; no continuous goal restart or additional schedules. Monitor these new runs instead of continuing the retired Ministral batches.
