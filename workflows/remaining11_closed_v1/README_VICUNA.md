# LLaVA1.6-Vicuna candidate expansion

Authorized6403 physicalGPU0, UUID GPU-6f5dc226-6850-9f93-d4b7-b6f2d618b402. GPU1's active GLM continuation is untouched. No prior Vicuna candidate scores exist. Exact checkpoint weights/config/processor identity and all13 original native package versions were checked before native inference.

Frozen Vicuna precision is FP16. Both native reference and new vLLM scorer retain FP16; no BF16 conversion or quantization is introduced. Native exporter `export_vicuna_closed16_v1.py` computes original101-class token mean scores on16 evenly spaced Food101 samples. Original registered environment is exactly copied to6403; this relocation is explicitly recorded, not claimed as original4028 hardware.

`vicuna_after_native_once_v1.py` waits on the exact native process via pidfd once, checks completion/reference hash, then launches `vicuna_entry_v1.py`. Actual multimodal processor input/tensor parity and16 real GPU top1 and gold-correctness agreement are mandatory before production. Scorer `vicuna_closed_scorer_v1.py` is a versioned v6 copy with registered FP16 precision; original scorer/source remains unchanged. Candidate names, word tokens, images, prompt, raw logprob mean rule and no-EOS convention remain unchanged. New numerical cohort does not mix old TF rows or claim exact numerical parity. Failure stops without retry or relaxed gates.

Evidence/dispatch/progress: `outputs/records/remaining11_closed_v1/llava16_vicuna_v1/`.
Run: `remaining11_llava16_vicuna_closed_20260924_v1`, under `outputs/{records,raw}/acceleration_v4`, model directory `llava16_vicuna`.
Monitor native_progress then engine_dispatch plus run progress/benchmark/error. Queue existence is not completion. All16 class scoring data must pass complete4848 unique coverage before results are consumed as completed candidate data.

## Current version2
Native16 completed with referenceSHA8ec0472b389aa7616cce752d1fb3f03a446f1e2e4cf0e4ba255ff20a504e08e0. Old event queue successfully fired v1; its CPU ModelConfig rejected generic32768 before GPU loading. All original failure/log/dispatch bytes remain.

Active entry `vicuna_entry_v2.py`, scorer `vicuna_closed_scorer_v2.py`, run `remaining11_llava16_vicuna_closed_20260924_v2`. Original model context capacity4096 is used, with strict expanded prompt+longest candidate≤4096 checks in audit and everyproduction row. No truncation, cropping, sliding-window override, precision change or numerical gate relaxation. Native16 maximum combined length3527. Only the new engine process is active; old native, queue and engine processes were checked exited before launch. Current dispatch/evidence: `outputs/records/remaining11_closed_v1/llava16_vicuna_v2/engine_dispatch.json`; reference remains in v1. There is no remaining active follower or recurring launcher.
