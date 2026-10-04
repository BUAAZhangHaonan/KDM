# VizWiz512 original SID: scope and current sources

The current SID operator uses decoder block 2 attention, head averaging, the last query, the lowest 100 visual attention indices, and downstream visual-key masks. Greedy generation uses alpha 0.5 and the full vocabulary. The 512 official evaluation inputs, question text, native preprocessing, checkpoint, precision, and 32-token generation budget are unchanged.

The current source is `src/kdm/models/sid.py` SHA256 `e1b8457e6dc0f470c62e333a161fca1b65370dd2383b6c9633c6d32b79fa1299`; decoder SHA256 is `3381e2b09f206755fec1e90b182c5643a0c8b44609dd3e3cec494c273f922b88`. The legacy six-model SID receipts retain their original older source identity. They are recorded as historical sources and are not presented as admission of the current file. Each current model requires an actual eight-input attention, rank100, mask, finite-distribution, prompt-equality and clean-session-isolation check before the disjoint remaining 504 inputs.

Commit `523d35fb` adds Gemma-specific selection of the existing native mask and visual token index lookup. The existing six architectures retain their rank100 and downstream-mask calculations. A missing second-block softmax attention raises a scope error. The current thin runner and CPU visual-token scope inspector were committed as `2b760cb0`.

## Actual applicability findings

Qwen2.5-VL's first real Viz input, `vizwiz:VizWiz_val_00000001.jpg`, is 121 by 162 pixels. The unchanged processor returns 24 visual positions, so fixed rank100 fails before any generated answer. The exact native processor was subsequently run on all 512 inputs on CPU, without model weights or CUDA initialization. Seven inputs have fewer than 100 visual positions: 4 (1 input), 16 (1), 24 (2), 54 (1), and 63 (2). The remaining 505 have at least 100, and all 512 spans are contiguous. The full-512 SID denominator is therefore not available under the unchanged operator and preprocessing. No reduced-rank, resized-image, or partial-505 generation was launched.

Source directory on central 4028: `outputs/paper_core_20261002_dev_viz/viz_native_baselines_20261004_1645/qwen25_viz512_CPU_token_scope/`; `native_visual_tokens.csv` gives every input and `complete.json` gives the full count. Frozen roster SHA256: `09431e8838ae86a0cb314484583f0601bda058bfb8cdd3b9354958188c67ad38`.

Qwen3-VL's exact native-processor CPU check also covers all 512 inputs. Seven have fewer than 100 visual positions: 70 (6 inputs) and 72 (1). The remaining 505 have at least 100, and all 512 spans are contiguous. The first input above has 70 visual positions. The CPU check took 48.996 seconds without loading weights or initializing CUDA. Its full-512 SID denominator is likewise unavailable under the unchanged operator and preprocessing. The exact source directory `qwen3_viz512_CPU_token_scope/` was copied from the registered 6403 runtime into the central source directory alongside the Qwen2.5 result.

MiniCPM-V2.6 and Qwen3.5 retain the existing architectural non-applicability notes. No SID algorithm adaptation is used to fill these cells. Neither Qwen2.5-VL nor Qwen3-VL runs a partial-505 SID condition.

## Current production sources

All directories below are relative to the corresponding project root, under `outputs/paper_core_20261002_dev_viz/viz_native_baselines_20261004_1645/`.

| Model | Host / physical GPU | Passed 8-input source | Disjoint production source |
|---|---|---|---|
| Gemma3-4B | 4028 / 5 | `gemma_sid_pilot8/` | `gemma_sid_full504/` |
| InternVL3.5-8B | K100 / 0 | `intern_sid_pilot8_frozen_source/` | `intern_sid_full504/` |
| Phi-3.5-Vision | 4029 / 2 | `phi_sid_pilot8/` | `phi_sid_full504/` |
| LLaVA-Mistral | 4028 / 0 for pilot; 4029 / 1 for production | `mistral_sid_pilot8/` | `mistral_sid_full504/` |

Each source directory contains the actual identity, PID/starttick ownership, predictions, progress and final receipt. Only directories with a passed `complete.json` are complete. CPU answer scoring and semantic abstention decisions are performed separately by the root process.
