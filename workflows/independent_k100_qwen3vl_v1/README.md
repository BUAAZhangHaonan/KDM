# Qwen3-VL independent Food extension

One fresh cohort on K100 physical GPU 0, original Qwen3-VL-8B-Instruct BF16 checkpoint. All 4848 original Food questions, ten distinct stable seeds per question, temperature 1, top_p 1, max 32 tokens, original native EOS and unchanged original prompt/images. Prefix cache is measured on genuine first attempt plus nine requests.

This version extends independent_v5 to qwen3vl without editing original workflows. Full class scores are not a prerequisite to response generation; independent identity records pending_separate_closed_cohort. Final joint deficit GT remains unavailable until separately validated class scoring and semantic/free annotation. Existing v5 postprocess cannot be used unmodified against this pending proof; no annotation/API starts here.

Native reference is measured with the registered exact environment on 6403 GPU0. Actual K100 vLLM engine verification compares all16 original samples, expanded prompt IDs, real processor tensors, native EOS, seeds, finite selected raw logprobs, cache tokens, speed, and greedy differences. Different random/greedy backend results stay in a separate cohort. Explicit review must bind actual check and entry hashes before production.

Checkpoint admission verifies full weight SHA256 against registered Hub digests, all config/processor hashes, GPU UUID, environment versions, and GPU lock. Files changed since full hashing fail admission. Fresh output only; failures preserved; no retries. Production has no paid API or semantic judge and is not a GT-completion claim.
