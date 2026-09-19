# mprisk runtime discovery — 2026-09-19

This bounded review performed read-only queries through `6403-lvshuyang`, with `PYTHONDONTWRITEBYTECODE=1`. It did not import model libraries, instantiate a model, allocate a GPU, modify the source host, or copy an environment. Current package results come from `importlib.metadata`; historical runtime claims are separately attributed to existing records. The machine-readable evidence is `outputs/records/mprisk_runtime_discovery.json`.

The initially supplied `/home/team/lvshuyang/TAFFC/mprisk` contains curation/delivery material and no model-wrapper source. The repository's committed runtime documents point to `/home/team/zhanghaonan/TAFFC/mprisk`, which exists and contains the actual model source, runtime configuration and prior smoke records. No unrelated research project was inspected. The reviewed upstream source ref is `cc6c0d82a77a958fd20c58e35efdc18c1ce0c036`.

## Verified historical prefill runtimes

The following six files under `/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_smoke_matrix_20260722/source/<model>/SMOKE_COMPLETE.json` each report `PASS`, 48 expected/completed tasks, zero failures, and Transformers **5.5.3**:

| Models | Historical Python | Dtype |
| --- | --- | --- |
| minicpm_v_2_6, minicpm_v_4_5 | `/home/team/zhanghaonan/miniconda3/envs/mind-py311/bin/python` | bfloat16 |
| phi3_5_vision | same | bfloat16 |
| gemma3_4b, gemma3_12b | same | bfloat16 |
| llava_onevision_qwen2_7b | same | float16 |

These records establish prior mprisk prefill success, not KDM native-generation equivalence. The configuration `configs/cache/complete_cache_matrix.yaml` sets `python_no_user_site:false` and `env_isolation:false`. The MiniCPM runtime log explicitly resolves Transformers under `/home/team/zhanghaonan/.local/lib/python3.11/site-packages/transformers`. The current `lvshuyang` account cannot read that user-site directory. Direct metadata inspection of `mind-py311` therefore exposes the environment-local **4.57.1**, torch **2.6.0+cu124**, and torchvision **0.21.0+cu124**. This is not a contradiction in the historical record: the complete effective dependency path differs. It must not be presented as a currently accessible, identical 5.5.3 runtime.

The file named `FINAL_CACHE_AUDIT.json` currently has top-level status `ready`, not `complete`; this review does not infer full experiment completion from its filename. The individual smoke records are the historical execution evidence used above.

## Phi-3.5 generation-specific environment

The existing recovery configuration `configs/recovery/phi3_5_vision_in_domain_pipeline_20260727.yaml` pins:

- Python: `/home/team/zhanghaonan/.venvs/mprisk-phi3-vision-4.43-cu121/bin/python`
- `PYTHONNOUSERSITE=1`, Python **3.11.11**
- torch **2.3.0+cu121**, torchvision **0.18.0+cu121**
- Transformers **4.43.0**, tokenizers **0.19.1**, accelerate **0.30.0**
- numpy **1.26.4**, Pillow **10.3.0**

Current metadata inspection matches these declared versions. The description configuration selects eager attention. This is an explicit existing generation-runtime contract, rather than a version recommendation inferred from package age. Two requested historical execution receipts are unreadable under the current account: `outputs/diagnostics/phi3_description_watcher_smoke_official443_20260804_8a6649a/runtime_contract_receipt.json` and `outputs/in_domain_recovery_formal1934_20260804/queue_runtime_receipt.json`. Their existence is not proof that their contents passed. No new execution validation was performed.

## Other inspected environments

`/home/team/zhanghaonan/.venvs/mprisk-kv-transformers-5.5.3/bin/python` is readable and currently exposes Transformers5.5.3, inheriting torch2.6.0+cu124 and other packages from `mind-py311`. It is explicitly assigned to Qwen-VL in the mprisk configuration. This review found no evidence that the six requested models were validated in that exact venv; matching a Transformers version alone does not establish interchangeability.

`/home/team/lvshuyang/anaconda3/envs/mprisk_resp/bin/python` currently has Python3.11.15, torch2.13.0+cu130 and torchvision0.28.0+cu130, but no installed Transformers, tokenizers or accelerate distribution. It is not the historical model runtime above.

The same user's `MiniCPM-V-2_6` and `gemma-3` conda environments expose Python3.12.12, Transformers4.57.3, torch2.9.0, torchvision0.24.0 and vLLM0.13.0; neither has an accelerate distribution in the inspected environment. No bounded-search evidence connects these environments to the recorded mprisk validations. They are inventoried, not recommended as verified replacements.

## Existing adapter constraints

The reviewing agent read the source at the pinned ref through the official GitHub repository with lazy Git fetching disabled. The primary agent also has the pinned model-wrapper files in the project-local sparse checkout; the environment review did not modify that checkout.

- [MiniCPM wrapper](https://github.com/BUAAZhangHaonan/mprisk/blob/cc6c0d82a77a958fd20c58e35efdc18c1ce0c036/src/mprisk/models/minicpm_v.py): remote-code `AutoModel`, a `data` dictionary for `input_ids`, `pixel_values`, `tgt_sizes`, `image_bound`, `position_ids`, optional `temporal_ids`; attention mask is separate. Position IDs derive from cumulative attention mask. The wrapper populates tokenizer image/slice/ref attributes and disables 4.5 thinking. This source uses `use_cache=False` for prefill and is not a validated autoregressive-session implementation.
- [Phi wrapper](https://github.com/BUAAZhangHaonan/mprisk/blob/cc6c0d82a77a958fd20c58e35efdc18c1ce0c036/src/mprisk/models/phi3_vision.py): remote-code `AutoModelForCausalLM`, processor `num_crops=4`, indexed image placeholders and explicit Phi user/assistant delimiters. Model attention is passed via `_attn_implementation`. The generation contract above is separate from the older prefill smoke environment.
- [Gemma3 wrapper](https://github.com/BUAAZhangHaonan/mprisk/blob/cc6c0d82a77a958fd20c58e35efdc18c1ce0c036/src/mprisk/models/gemma3.py): explicit Gemma3 model/processor classes, processor chat template, and nested image batch. Existing prefill records use bfloat16.
- [OneVision wrapper](https://github.com/BUAAZhangHaonan/mprisk/blob/cc6c0d82a77a958fd20c58e35efdc18c1ce0c036/src/mprisk/models/llava_onevision.py): explicit OneVision classes and nested `text_config`; this mprisk wrapper freezes native-video F8 and explicitly rejects still-image requests. KDM's single-image task needs its own image adapter validation; copying the video restriction would change the task.

All paths and versions above distinguish recorded previous execution, current metadata, and inaccessible evidence. The historical multi-frame protocol, its GPU allocation and its checkpoints are not transplanted into KDM's frozen task by this review.
