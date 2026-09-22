# Food-only v3
User-authorized scope: all 16 models, 4848 Food101 rows each, 77568 total. Independent generation paused. No KV optimization or paid API.
Run: food_closed_20260923_v3. Status and progress: outputs/records/probes_v2/food_closed_20260923_v3/. Raw: outputs/raw/probes_v2/food_closed_20260923_v3/.
Complete validated v2 closed records are re-ledgered with original identities and source hashes, computing only missing samples. Old partial or failed runs stay unchanged.
4028: shared single-card queue on GPUs0/1, with GPUs4/5 running three registered dual-card models then joining the shared queue. 6403: Qwen3VL GPU1, GLM GPU0. Admission overlay explicitly records the newly authorized GPU0 UUID; frozen registry/source unchanged.
Each model's first new closed row must agree with the unchanged frozen closed_rank across all101 class scores within 1e-6. Full coverage verified before complete. No retries or fallbacks.
After this run, use workflows/quick_match_v1 for CPU-only preliminary answer/rank matching. Preserve ambiguous items. Do not automatically launch stage five, independent generation or DeepSeek; wait for user resumption after Codex reset.
