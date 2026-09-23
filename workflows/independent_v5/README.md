# Five-model independent answers (v5)

This is a separate vLLM measurement cohort, authorized on 2026-09-23 after the user asked whether the observed differences came from temperature. The greedy comparison uses temperature 0. Production preserves the original independent-sampling protocol: temperature 1, top_p 1, top_k -1, repetition penalty 1, 32 generated tokens, BF16, native EOS, ten distinct stable seeds per question. Backend RNG streams and numerical outputs are not claimed equivalent to Transformers.

## Scope and resources

Prioritize qwen25vl, qwen35_4b, llava16_mistral, minicpm26 and gemma3_4b on all 4,848 Food-101 questions (2,424 dev and 2,424 eval): 48,480 independent answers each, 242,400 total. The original broader research scope including VizWiz is retained; this Food-first batch does not complete it. The remaining 11 models are opportunistic only after primary needs are satisfied; they require their own engine admission.

6403 GPU 0/1 and RTX_Pro_6000 (K100) GPU 0 are the independent-generation resources. Existing formal method experiments continue on 4028 GPU 0/1/4/5. As of 2026-09-24,4029 is removed from future dispatch and monitoring; its completed formal evidence is preserved on4028. Never start a duplicate process based on copied status files; inspect the originating host and GPU locks.

## Admission and sources

`independent.py` is a versioned copy of the earlier independent implementation. The first real replicate is generated before the other nine parallel requests, allowing supported models to reuse the same visual/prompt prefix. All ten are included; there is no extra warmup answer in the production count. `verify_engine.py` checks 16 native references, actual vLLM multimodal tensors and expanded tokens, EOS, seed uniqueness, finite selected-token log probabilities, and real ten-answer timing/cache counts. Native references also match the frozen native greedy decoder on all 16 images. This does not prove full-vocabulary distribution equivalence.

Original failed exact-parity checks remain unchanged. `reviewed_entry.py` admits only a separately reviewed cohort, with `review.json` binding the original check hash, source hashes, runtime versions, checkpoint/processor identity, and explicit non-equivalence. Qwen2.5 has 10/16 greedy differences; MiniCPM and Qwen3.5 each have 5/16. No input, precision, EOS or sampling configuration mismatch was found for these three; a specific numerical-kernel cause has not been proven. The differences may include substantive answers and must remain visible in conclusions.

Qwen3.5's vLLM 0.24 hybrid Mamba cache requires a 528-token alignment block exceeding these unchanged short prompts. All 16 measured prefix-cache counts are zero. Its review explicitly records this exception: parallel generation is used, but prefix-cache acceleration is not claimed. Do not pad prompts or change cache mode/precision to manufacture a cache pass.

Gemma had a real image-resizing difference between the native and new Torch environments. `gemma_native_pixels_v1.py` calls the exact native image processor in a persistent CPU subprocess, with no model weights on CPU. Its CPU audit matches all external tensors and actual vLLM processor outputs on 16/16 inputs. Production still requires the separate GPU engine audit and review; the earlier mismatch remains a failed attempt.

K100 uses its project-local `.environments/vllm024` and versioned `workflows/independent_k100_v1/entry_v3.py` / `worker_v3.sh`. The first engine check failed because ninja was absent from PATH; the corrected worker exposes the existing project environment and CUDA tools. That failure is retained. Its second 16-image check passes input/EOS/cache checks, with one greedy difference; review is separate from exact parity. The first production startup then stalled before generating any records. A same-environment CPU reproduction hung after fork but passed with spawn. The original zero-row attempt is retained; v3 uses spawn without changing scientific settings. The live K100 run is `panel5_independent_food_v5_spawn1`, as indexed by its latest dispatch and workflow README.

## Runs, queues and labeling

Production run on 6403: `outputs/{records,raw}/acceleration_v4/panel5_independent_food_v5/MODEL`. The live K100 LLaVA run uses `panel5_independent_food_v5_spawn1`; the older loading attempt is interrupted, not an active generation job. Native/engine/review/dispatch evidence is under `outputs/records/independent_v5`. Initial live jobs are Qwen2.5 and MiniCPM on 6403. A one-shot pidfd queue starts reviewed Qwen3.5 on GPU 0 after Qwen2.5 has a completion receipt. GPU 1 then audits Gemma after MiniCPM. LLaVA runs on K100 after its reviewed admission. Queue failure does not trigger a retry.

`postprocess.py` validates completed 48,480-record coverage (ten unique replicates/seeds per question), identities, hashes, frozen parameters and the bound complete 101-class score cohort before free character matching. One-shot completion followers process the two initial live jobs; later jobs must receive the same postprocessing. Results are under `outputs/annotations/independent_v5/panel5_independent_food_v5/MODEL`.

Free labels distinguish correct, incorrect, abstain, invalid/empty and unresolved. A recognized correct attempt or gold rank one rules out the joint abstention condition. Zero correct across ten resolved attempts plus gold rank above one supports the preliminary joint condition; unresolved attempts remain unknown. `final_gt=false` and `human_reviewed=false` are explicit. Generation, automatic screening and resolved question evidence are different counts. No new paid DeepSeek or Ministral requests are allowed.

Keep all failures, source identities, reviewed backend differences and dev/eval separation. Synchronize complete source-host evidence to 4028. Do not mix vLLM rows into older Transformers cohorts, change frozen formal code, claim human review, or claim completion of the full research from this batch.
