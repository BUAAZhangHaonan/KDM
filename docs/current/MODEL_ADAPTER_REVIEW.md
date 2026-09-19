# Model adapter review — 2026-09-19

The fixed sixteen candidate checkpoints are now locally resolved. Six originated in the read-only `/home/g203-4028/Models`; the other ten were copied by the main agent from the user-confirmed `/home/team/lvshuyang/Models` on 6403 into this project's `cache/models`. Exact source shard filenames/sizes and safetensors header offsets passed for all ten copies (162,461,857,144 bytes; `outputs/records/model_transfer_complete.json`). Earlier missing statuses in archived interface records described the original 4028 root and are superseded by the current inventory. No candidate was substituted.

This review covers software interfaces and identities. The executor independently read the original research writing requirements, required current documents, paper outline and AGENTS.md. No census, selection, intervention or probe was run by this agent. The sixteen images are the fixed software interface manifest, not a reduced research sample.

## Runtime identity

Each candidate has a resolved spec in `configs/runtime`. The original six use the existing project venv (torch 2.9.0+cu128, Transformers 5.17.0). Two isolated project environments were built without system site packages: `mprisk-tf553` uses torch 2.6.0+cu124 / Transformers 5.5.3; `phi443` uses torch 2.3.0+cu121 / Transformers 4.43.0. Detailed frozen versions are in `environments.json`, installation/import records and actual downloaded wheel SHA256 fingerprints in `outputs/records/environment_builds/`.

The environments derive from read-only mprisk discovery, not an assumption that a Python path reproduces historical user-site packages. Historical six-model mprisk prefill evidence is not KDM generation evidence. The project Python is 3.11.14, whereas the Phi source used 3.11.11; native token checks below verify the actual project environment. Shared environments and model sources remain unchanged. GPU workers put caches, temporary files and compiled artifacts under the project.

Most weight fingerprints reuse existing Hugging Face revision/SHA256 metadata with live sizes/mtimes and clearly identify that provenance. Only three LLaVA-1.5-7B shards lacked that metadata and were actually SHA256-hashed once (`weight_fingerprints_missing_metadata.json`). Config, processor and remote-code small files receive content hashes. Discovery rejects missing/truncated shards and reports ambiguity explicitly.

## Adapter behavior

- Whitespace is preserved when decoding ordinary tokens; only declared special tokens are removed, and tokenizer cleanup is disabled. EOS derives from model/text/generation configuration and tokenizer, including Gemma turn terminators.
- VCD uses the pinned official float32 schedule, step 500, and Gaussian samples in the original image tensor dtype. CPU float32 and bfloat16 tests match the official arithmetic exactly. A private generator avoids changing global RNG state; MiniCPM's nested image slices share one ordered stream without flattening their structure.
- InternVL uses its checkpoint's native conversation template and system message plus model-card dynamic tiling (max 12 plus thumbnail), ImageNet preprocessing and matching text-only template. Its native generate returns continuation-only tokens.
- MiniCPM2.6/4.5 adapters derive from mprisk commit `cc6c0d82a77a958fd20c58e35efdc18c1ce0c036` and checkpoint remote code. They preserve the original nested pixels, image bounds, target sizes, temporal IDs and positions in the `data` dictionary. Incremental steps use the original LLM. MiniCPM4.5 thinking is disabled explicitly. Native generation preserves its terminators and continuation-only output.
- Phi3.5 retains num_crops=4, the original user/image/assistant delimiter template, bf16 and eager attention from the frozen source generation configuration. No SDPA replacement is made.
- OneVision uses its native still-image processor for KDM images. The mprisk video-F8 wrapper is not copied into this image protocol.
- Gemma12B and LLaVA13B use explicit, complete two-card module maps, with no automatic placement, CPU or disk offload. The first Gemma map used a root entry overlapping child entries; Accelerate failed on cross-card normalization. Its log is retained; the corrected partition lists each layer and non-layer module separately, with unchanged dtype, inputs and model.

The inherited Qwen Conv3d-to-linear optimization is still present. Native/backend equivalence is measured on the same loaded model and does not independently prove equivalence to an unmodified Conv3d implementation. InternVL's Transformers 5.17 tokenizer Mistral-regex warning persists even with model-card use_fast=False; its actual tokenizer configuration is recorded rather than silently replaced.

## Verification contract and current records

`python -m kdm.models.verify` requires exactly sixteen distinct manifest entries. It compares all greedy continuation token IDs, including EOS, under native generate and backend next with max_new_tokens=32. Six separate conditions (guided/unguided clean, noise and text-only) each see the same empty/one-token/two-token prefixes; independent and interleaved sessions must produce exactly equal logits. The first image additionally checks full-sequence final-head projection, early raw projection and normalized projection. This checks interfaces, not the DoLa/DeCo scientific formulas or SID algorithm.

Native records snapshot the actual spec, manifest hash, loaded adapter dependency hashes and verification-script hash. Newer records include physical GPUs, device maps and peak allocation. SID is outside these six conditions and therefore has its own independent method evidence; a SID source edit does not invalidate untouched native paths. GPU locks are inherited by every worker; no unrestricted GPU use or fallback is permitted.

The records below are software evidence; incomplete runs are not passes. Earlier four-condition records remain as history and are superseded only when a complete final record is available.

| Candidate | Final six-condition native16 record | Layer projection |
| --- | --- | --- |
| gemma3_4b | 16/16 passed: `gemma3_4b_final16_v3.json` | passed |
| gemma3_12b | 16/16 passed: `gemma3_12b_final16_v2.json` | passed |
| glm46v | 16/16 passed: `glm46v_final16.json` | passed |
| internvl35_8b | 16/16 passed: `internvl35_8b_dual_final16.json` (explicit 4/5 dual factory) | passed |
| llava15_7b | 16/16 passed: `llava15_7b_final16.json` | passed |
| onevision | 16/16 passed: `onevision_final16.json` | passed |
| minicpm26 | 16/16 passed: `minicpm26_final16_v4.json` | passed |
| minicpm45 | 16/16 passed: `minicpm45_final16.json` | passed |
| phi35 | 16/16 passed: `phi35_final16.json` | passed |
| qwen25vl | 16/16 passed: `qwen25vl_final16_v2.json` | passed |
| qwen3vl | 16/16 passed: `qwen3vl_final16.json` | passed |
| qwen35_4b | 16/16 passed: `qwen35_4b_final16.json` | passed |
| qwen35_9b | 16/16 passed: `qwen35_9b_final16.json` | passed |
| llava16_mistral | 16/16 passed: `llava16_mistral_final16.json` | passed |
| llava15_13b | 16/16 passed: `llava15_13b_final16.json` | passed |
| llava16_vicuna | 16/16 passed: `llava16_vicuna_final16.json` | passed |

Twelve adapter unit tests pass, including exact official VCD noise, original MiniCPM mapping preservation, nested noise structure, whitespace, incomplete shard detection and rejection of offload/automatic maps. Initial missing scipy imports, MiniCPM native image_sizes keyword, and incorrectly shaped last-position projection diagnostics are retained as failure records; the corrected checks passed without changing the fixed images or generation budget. SID multi-session hook and attention-source bugs found independently were repaired by the data agent and validated separately. Nine candidates now have separate passing current-source SID reference proofs, including the explicitly authorized dual InternVL factory. Each uses two prompts, 14 prefix visits, exact official selection/mask-core logits and native clean logits before/after SID; see SID_CAPABILITY_REVIEW.md for individual receipts and the separate fixed-reference/rank/mask limitations. These are not inferred from native16 passes. No native pass claims SID support or formal scientific acceptance.

Qwen2.5-VL initially disagreed on two first tokens because its checkpoint generation config defaults to repetition_penalty=1.05, while the registered decoder uses unmodified greedy logits. The native harness now explicitly applies do_sample=False, num_beams=1 and repetition_penalty=1.0, preserving the fixed greedy protocol. With these same operations, all sixteen samples pass exact token/state and layer projection checks; the original disagreement record remains. The sixteen checkpoint generation configs were inspected: no other candidate declared a non-unit repetition penalty.

## Explicit InternVL dual factory acceptance

The authorized new factory `kdm.models.internvl_dual:InternVLDualBackend` replaces only model construction with an explicit 18/18 decoder split across logical0/1 (physical4/5). Vision, projector and text embedding remain together on logical0; final normalization/head are on logical1. It inherits native build/template/visual processing and forward methods unchanged; the shared hf/backbone/remote/sid sources were not modified for this placement change. Seven constructor/layout/inherited-method CPU checks pass.

The complete new native16 record `outputs/verification/internvl35_8b_dual_final16.json` passes all16 token/state checks and native layer projections, with peaks14,374,173,184 and10,517,976,576 bytes. SID `outputs/verification/internvl35_8b_sid_reference_v3_summary.json` separately passes all14 visits, exact official selection/mask-core logits, interleaving/nonmonotonic-prefix checks and native clean logits before/after SID. Both records include the additional internvl_dual.py dependency; the runtime method gate was actually checked successfully. The original single-GPU spec and both single-GPU failure records remain as history. This hardware-placement change does not reduce images, native tiling, precision, rank100, layer2 or continuation inputs.

GLM current-source native16 completed normally with all sixteen token/state checks and layer projections exact. Physical GPU1 peak allocation was 21,163,414,528 bytes. Total elapsed time including loading was 3,616.72 seconds; the observed process task count was 252 at each image completion. Per-image timings are retained in `outputs/verification/glm46v_final16_timing.jsonl`; no image, token budget, operator or runtime parameter was changed during the run. All sixteen candidates now have complete current native-interface evidence. GLM SID remains a separate pending method check at this native-record closeout.
