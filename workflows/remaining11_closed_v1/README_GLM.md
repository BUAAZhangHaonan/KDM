# GLM candidate scoring expansion

User authorized use of idle 6403 GPU1 to expand remaining11 on 2026-09-24. This entry reuses 16 already-measured native TF scores only as engineering references. Original 1895 TF rows remain unchanged; new vLLM output is a complete separate numerical cohort, not a continuation mixing engines.

Entry: `glm_entry_v1.py`. Native CPU export uses the exact registered native environment and source identity. Runtime checks actual vLLM multimodal tensors after declared BF16 cast, prompt expansion, original class tokens, checkpoint/config/processor identity. Inherited `vllm_closed_v6.py` runs real GPU scoring on the 16 references and refuses production if top1 or gold rank1 agreement fails. Nonzero raw logprob differences are retained. 101 original class-name token mean log probabilities, no EOS, unchanged image/template/precision, no top-k approximation.

Dispatch and native/engine audit: `outputs/records/remaining11_closed_v1/glm46v/`.
Production/gate records: `outputs/records/acceleration_v4/remaining11_glm_closed_20260924_v1/glm46v/`.
Production output: `outputs/raw/acceleration_v4/remaining11_glm_closed_20260924_v1/glm46v/closed.jsonl`.
Physical GPU1 UUID GPU-5c45e961-7442-eb90-9b8c-295a1cf14995; inherited project flock required. Original `worker.sh` validates host/UUID.
No automatic retries. This entry does not label independent answers or claim final abstention ground truth.

## Active version 2
`glm_entry_v1.py` stopped during CPU auditing before any GPU model load because BatchFeature was summarized as a type marker. Its original error and reference remain. `glm_entry_v2.py` explicitly serializes `dict(inp)`; no tensors, inputs, model parameters or scoring rules changed. Active evidence is `outputs/records/remaining11_closed_v1/glm46v_v2/`, run `remaining11_glm_closed_20260924_v2`. All16 actual multimodal processor checks passed. GPU numerical gate remains mandatory and authoritative in benchmark.json.
