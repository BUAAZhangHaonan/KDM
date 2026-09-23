# Gemma v4: audit range endpoint correction

The v3 spawn GPU audit stopped at the first metadata check. It recorded `[5,260]`; the native audit represented the same 256 image tokens as `[5,261)`. This was an audit comparison mismatch, not evidence that the real attention mask dropped a token.

Installed vLLM 0.24 source evidence under `/home/team/lvshuyang/anaconda3/envs/ST_LORA/lib/python3.12/site-packages/vllm`:

- `multimodal/inputs.py:179-202`: `extract_embeds_range` explicitly returns inclusive endpoints, with `offset + length - 1`.
- `v1/worker/gpu_model_runner.py:2329-2336`: copies these intervals and computes length with `end-start+1`.
- `v1/attention/backends/triton_attn.py:238-243`: passes ranges to the metadata tensor; `:640-668` passes the tensor to unified attention.
- `v1/attention/ops/triton_attention_helpers.py:335-352`: both query and key comparisons use `<=range_end`.
- Existing project `independent.py:94-99` returns half-open intervals; `:110-113` compared them without conversion.

`gemma_range_audit_v1.py` converts native expected half-open endpoints to inclusive endpoints for the unchanged original receipt validator. It preserves all original backend/GPU/module provenance checks, records raw and converted intervals, and binds five actual installed vLLM source hashes. It never alters the model attention mask, processor, image, weights, precision, EOS or sampling configuration. `gemma_native_pixels_entry_v4.py` retains the v3 spawn and original pixel bridge and adds this audit binding plus the CPU proof hash. Old sources and failures remain.

The CPU proof `outputs/records/independent_v5/gemma_range_endpoint_v4/cpu_check.json` verifies all 16 native token-type tensor byte hashes, matches the real v3 first-image receipt and rejects five wrong/empty interval sets. Native first-image tensor hash is `bca291c8ab763d7ff0a44bbb3fc1c8e1a1a293bc0455c35f64d2034a5914a8e7`, with ones at positions 5 through 260 inclusive.

One new GPU1 audit was dispatched under the existing resource admission and lock. Dispatch/log: `outputs/records/independent_v5/gemma_range_endpoint_v4/`. Result: `outputs/records/independent_v5/engine_verify_gemma_pixels_ranges_v4/gemma3_4b/`. Audit completion and numerical differences must be reviewed before any production admission. Metadata checks do not claim full kernel numerical equivalence. No production process is started by this correction.
