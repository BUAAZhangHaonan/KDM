# Model adapter review — 2026-09-19

Executor: independent model_adapters agent. Read the original research writing requirements, all six required current protocol documents, paper outline and AGENTS.md before implementation. Scope: model discovery, runtime identity, thin model adapters, fixed-image interface checks. No census, selection, intervention experiment or probe has been run by this agent.

## Availability

The historical project reconnaissance identifies `/home/g203-4028/Models` as the model root. Live discovery found six of the sixteen candidates: GLM-4.6V-Flash, InternVL3.5-8B, Qwen3-VL-8B, Qwen3.5-4B, Qwen3.5-9B and LLaVA-v1.6-Mistral-7B. All indexed weight shards exist and are nonempty. The other ten are missing; MiniCPM-V-2_6 has only README/assets and no model config or weights. No mprisk directory or link was found in the authorized user's home/project directory listing. Other model roots and mprisk environments remain an explicit input requirement; missing candidates were not replaced or downloaded.

Each candidate has a spec in `configs/runtime/`. Missing candidates have null factories/environments. Existing candidates use the project venv: Python 3.11, torch 2.9.0+cu128, transformers 5.17.0. Weights are identified by existing Hugging Face download metadata revision and SHA256 plus live filename/size/mtime; these hashes are clearly labelled as download metadata, not fresh rehashes. Processor/config/remote code and active adapter source receive small-file content fingerprints required by the protocol. No shared environment or model file was edited.

## Changes and findings

- `decode` preserves ordinary leading/trailing whitespace and disables tokenizer cleanup; only declared special tokens are removed. Original tokens remain in the records.
- Discovery now checks all indexed checkpoint shards and reports incomplete and ambiguous checkpoints separately.
- VCD noise uses the official float32 schedule and the image tensor's dtype for Gaussian noise. A private generator preserves condition independence without changing global random state. CPU float32 and bfloat16 outputs match the pinned official implementation bit for bit. The prior package's float64 noise stream was not equivalent.
- The extracted InternVL adapter used a hand-written template without the model's system message and a single CLIP crop. It now uses the installed checkpoint's actual conversation template/system message and the dynamic tiling/normalization functions from its model card (max 12 tiles plus thumbnail). Text-only input uses the same native conversation template. Verification accounts for InternVL native generate returning continuation-only tokens.
- The inherited Qwen visual Conv3d-to-linear optimization remains in the extracted model constructor. Native/backend checks use that same loaded model; they do not independently establish equivalence to an unmodified Conv3d checkpoint implementation.
- MiniCPM and Phi remain unavailable. No invented generic HF adaptation is claimed: MiniCPM still needs the original mprisk data dictionary, image slices and token mapping before implementation/testing.
- Gemma-12B and LLaVA-13B are missing. No device map is guessed or tested. Their required explicit two-GPU maps must be set from the actual checkpoint architecture when paths become available. CPU/disk placement remains rejected by the backend.

## Verification contract

`python -m kdm.models.verify` accepts only the fixed 16 distinct image manifest. For each image it compares complete greedy tokens (including EOS) from native generate and backend next, and preserves visible whitespace. It then checks four clean/noisy guided/unguided conditions on the same three prefixes (empty, first native token, first two native tokens), comparing independent execution with interleaved execution using separate sessions. Passing requires exact logits equality for this interface check. These 16 images are only software checks, not a reduced research sample or model selection evidence.

GPU checks use worker locks on physical 0 or 1, with the explicitly coordinated official-noise rechecks on physical 4 and 5. Records are in `outputs/verification/`; errors and unfinished checks are reported explicitly. Interface pass does not mean layer methods or SID have passed their distinct reference-algorithm checks.

## Observed status

| Candidate | Fixed 16 native token checks | Four-condition state checks |
| --- | --- | --- |
| Qwen3.5-4B | 16/16 passed | exact; official-noise v2 |
| Qwen3.5-9B | 16/16 passed | exact |
| Qwen3-VL-8B | 16/16 passed | exact; official-noise v2 |
| LLaVA-v1.6-Mistral-7B | 16/16 passed | exact |
| InternVL3.5-8B | 16/16 passed | exact; restored native template and tiles |
| GLM-4.6V-Flash | running; first image passed | first image exact; remaining pending |
| Remaining ten | missing checkpoints | not run |

Nine adapter unit tests pass. The official-noise v1 records are retained as earlier software evidence, superseded by v2 for Qwen4B/Qwen3-VL state checks. Formal experiments remain gated by missing candidates, full census/selection and independent method verification. Native checks compare ordinary greedy generation and VCD's four clean/noisy guided/unguided states; text-only M3ID states, layer projections and SID are explicitly not validated by these records. No MiniCPM/Phi adapter or 12B/13B placement is claimed as complete.

InternVL's tokenizer emits a Transformers 5.17 warning about the Mistral-regex heuristic. A read-only check with the model-card `use_fast=False` also produces Qwen2Tokenizer with is_fast=True and the same warning in this environment. This was not silently changed to a different tokenizer; the exact installed checkpoint/processor configuration is recorded, and native/backend equivalence uses the same tokenizer.
