# 4029 additional extraction resources - 2026-09-23

The user explicitly authorized the last four4029 GPUs and a KDM project/environment under /home/hdd3/zhanghaonan/projects. The admitted project is /home/hdd3/zhanghaonan/projects/knowledge-deficit-mitigation on hostname WS-4029GP-TRT. Physical GPUs4,5,6,7 are RTX3090 cards; their UUIDs are pinned in workflows/acceleration_4029_v1/admission.py. GPU0..3 remain outside this KDM authorization.

The total resource pool is now two A100s (6403 GPU0/1) and eight3090s (4028 GPU0/1/4/5 plus4029 GPU4/5/6/7). Existing4028 and6403 model workers remain active.

## First dispatch

The previously unstarted qwen35_4b formal job moves from the4028 waiting queue to4029. Only that waiting CPU queue was stopped; no existing model was interrupted. Four disjoint original sample-hash partitions run concurrently under panel5_food_formal_4029_v1:

| Physical GPU | Shard | Eval questions | Formal tasks |
|---|---:|---:|---:|
|4|0/4|617|39488|
|5|1/4|605|38720|
|6|2/4|615|39360|
|7|3/4|587|37568|

The union is exactly2424 eval questions and155136 original tasks. Both filtering layers use the same sample hash, so the second filter is idempotent. No sample, method, prompt combination or32-token decoding parameter is removed. UNKNOWN-main remains first. This is inference/data generation, not model training.

## Environment, weights and evidence

The original Python3.11 base with torch2.9.0+cu128 and the original KDM venv overlay with transformers5.17.0 were copied into this project. A local native311 venv was regenerated with the local base;9 relevant package versions match exactly. Shared4029 environments were not modified. The source files and frozen runtime spec remain unchanged; a versioned admission layer relocates only host/device/path values and records their hashes.

The existing /home/hdd3/zhanghaonan/projects/holocue/models/Qwen3.5-4B is reused read-only through models/Qwen3.5-4B. Configuration/tokenizer/processor hashes, weight sizes and stored Hub identities match the original spec; the large weights were not newly rehashed. All4848 copied Food images have actual byte hashes matching the frozen catalog. Original full manifest, model census and existing automatic labels retain their hashes. VizWiz images were not deployed because this dispatch is Food-only.

Each worker holds its physical-GPU lock and a second model/shard lock, verifies hostname/root/UUID/package versions/checkpoint files/original proofs, and emits a new execution receipt. Frozen source/spec/host-registry files are never edited. New-host bitwise numerical equivalence is not claimed merely from environment copying.

Evidence is under outputs/records/deployment_4029_v1, including model_reuse.json, environment_verification.json, input_verification.json, four_shard_plan.json, cpu_admission_shard0.json, dispatch.json and the4028 handover receipt. Actual progress/raw output is under outputs/{records,raw}/acceleration_v4/panel5_food_formal_4029_v1/qwen35_4b/shard_XXX_of_004.

## Monitoring and scope

The existing two-hour monitor now covers all three hosts. No extra automation or continuous goal loop was created. Synchronize completed4029 records/raw outputs/sidecars to4028 and require exact four-shard union coverage before declaring full-model completion. Completed main stages may be screened separately using the existing free stage postprocessor, with unresolved items explicit. No new paid annotation call, model substitution, precision reduction or automatic failed-task retry is authorized here. The five-model Food study and full research completion standards remain those in the acceleration amendment.
