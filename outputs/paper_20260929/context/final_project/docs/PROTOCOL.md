# Study protocol

## Population and study boundary

The frozen study inventory contains 9,167 unique image-question records: 4,848 Food-101 records and 4,319 VizWiz validation records. Food-101 has 2,424 development and 2,424 evaluation samples. VizWiz uses an image-identity SHA-256 split: 818 development and 3,501 evaluation records. The 16-model registry and model-specific runtime specifications are in `configs/kdm/models.json` and `configs/runtime/`.

Each model first receives guided and unguided census prompts for the full inventory: 18,334 records per model and 293,344 total. Subsequent model-dataset inclusion uses the original guided-census abstention decision. The five-model Food-101 panel is a defined sub-study within the 16-model study: Qwen2.5-VL, Qwen3.5-4B, LLaVA-v1.6-Mistral, MiniCPM-V-2.6, and Gemma-3-4B. The panel has 2,424 evaluation questions per model.

The response snapshots are stored in `data/responses/census/` and `data/responses/formal/`. Their manifests bind source and output hashes, row counts, identities, and task coverage. The original 16-model baseline labels are in `data/annotations/census_baseline_auto/`; their producer and source identities are retained. The formal five-model response manifest records 853,248 evaluation rows across 352 registered conditions.

## Prompts and methods

Prompt builders and task expansion are implemented in `src/kdm/prompts.py` and `src/kdm/pipeline.py`. Guided prompts request a concise answer and include the assigned abstention marker. Unguided prompts omit the abstention instruction. The registered markers are `UNKNOWN`, `UNCLEAR`, `UNSURE`, and `I cannot identify it`. Independent best-estimate prompts request a specific short answer.

The full method matrix is registered in `configs/kdm/method_plan.json`. Every selected model-dataset condition includes VCD, M3ID, DoLa, and DeCo. SID is included where the model's native adapter and frozen proof register support. VCD, M3ID, and SID use the marker-by-reference-marker matrix. DoLa and DeCo use the corresponding clear-marker prompts. Reference-instruction-removed VCD/M3ID, instruction-preserving VCD/M3ID, and visual CDA are separate registered conditions. The exact matrix and counts are represented in the machine-readable task plan.

Generation uses a 32-token limit, greedy decoding (`temperature=0`, `top_p=1`), and the unchanged model-specific processor, dtype, weight path, and method settings. Defaults are VCD `alpha=1, beta=0.1`; M3ID `lambda=0.02, threshold=0.3`; DeCo `alpha=0.6, top_k=20, top_p=0.9`. Model-specific layers, visual inputs, SID settings, and M3ID prompt-token offsets come from the task and runtime records.

## Food-101 primary score

The primary field is `canonical_name_in_primary_score`. For each question-answer pair, the target-blind primary-name record supplies literal names and their roles. Main, coequal, and explicit competing candidates enter classification; side dishes and explanatory mentions remain in the record without entering the primary class. The selected name is compared with the 101 canonical Food-101 labels and the exact underscore-to-space spelling. Matching uses word boundaries, case folding, and outer-format cleanup. The method does not add aliases, synonym mappings, spelling repairs, plural conversions, or hyphen-to-space conversion.

A unique canonical class matching the target receives 1; a different unique class receives 0. An explicit primary answer outside the 101 labels receives 0. Multiple coequal explicit primary answers receive 0 for this single-label task, including a set that contains both a canonical Food-101 name and an explicit name outside the taxonomy. Missing or unresolved primary-name evidence remains unknown. A complete response equal to its assigned marker is scored as abstention; the behavior field is resolved from the exact-QA review chain or the raw-line-bound census label.

`literal_extracted_name_score` is the sensitivity column for the extracted literal name. Food-101 correctness, abstention, and selection metrics use the same 2,424 evaluation records per condition. Full behavior-label definitions and semantic review conventions are documented in `docs/SCORING.md`.

## Freeze and provenance

The original frozen contract is preserved at `data/provenance/frozen_contract/contract.json`. Its 95 registered files are stored as content-addressed blobs; `manifest.json` maps each original path and SHA to its blob. `src/kdm/frozen.py` verifies those bytes, the original source commit, unchanged decoding/model algorithms, and current frozen configs and manifests. Runtime receipts also bind the current migration-code identity and the proof manifest.
