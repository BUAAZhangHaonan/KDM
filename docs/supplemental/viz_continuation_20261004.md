# VizWiz ten-attempt continuation

The five core models lack the full 3,501 eval images × 10 independent attempts.
The recovered generator is exactly the accepted 20260923 vLLM implementation
from commit `b81fdafb078310e2fe7e15ea0f42e68566cc362e`. The new caller changes
only dataset selection, explicit key ownership, output destinations and evidence
capture. It does not call the old Food-only completion gate or claim new native
numerical equivalence. Runtime versions, checkpoint configuration, processor files,
weight sizes/available Hub metadata, native EOS and the old acceptance identity
are checked before GPU generation.

Actual source recovery and old identity SHA256 records are in
`outputs/paper_core_20261002_dev_viz/closeout_20261004/viz/source_recovery.json`.
The 8-image pilot is included in the 3,501 images. The four remaining input files
are disjoint whole-image shards, each with all ten separately seeded attempts.
No generated answer is reused as a new draw. All old rows and scores are frozen.

The original engine uses BF16, temperature 1, top-p 1, top-k -1, max 32 tokens,
native stop-token IDs and the registered stable per-model/image/replicate seed.
One genuine attempt is generated first, then nine independently seeded requests
reuse its exact image/prompt cache. The recorded group wall time must be counted
once per image, not ten times. Fresh actual request checks cover expanded prompt
IDs, native termination, finite selected log-probabilities and ten unique seeds.
The current-input processor summaries and prompt tokens are saved separately.

6403 uses `/home/team/lvshuyang/anaconda3/envs/ST_LORA/bin/python` for Qwen2.5,
Qwen3.5, MiniCPM and Gemma. K100 uses the original `.environments/vllm024/bin/python`
for LLaVA-Mistral. Both reported the historically accepted vLLM 0.24.0, torch
2.11.0 and transformers 5.12.1. Gemma retains its native-pixel CPU bridge and
real Triton image-range audit. No host/runtime substitution is implicit.

Example CPU verification (no GPU load):

```bash
python workflows/paper_core/viz_independent_resume.py --model qwen25vl \
  --host 6403 --gpu 1 --pieces pilot8 --run pilot8 --plan
```

Actual execution additionally requires the named physical GPU to be released by
the coordinator, the matching `CUDA_VISIBLE_DEVICES`, and a fresh output name.
The caller takes the common GPU lock and records PID/starttick/UUID. Each task-key
claim checks prior output and other claims under a serialized model admission lock.
After a failure there is no automatic retry or parameter change.

`generation/<model>/<run>/complete.json` certifies generation coverage only.
Semantic labels, continuous official answer quality, and model-level capability
cross-validation are separate downstream work and are not marked complete by it.
