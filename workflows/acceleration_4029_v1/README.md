# Exact native4029 relocation, qwen35_4b first

Authority: docs/current/PROTOCOL_AMENDMENT_20260923_4029.md. User-authorized physical GPUs4/5/6/7 only. The original4028 queued Qwen3.5 formal job is retired; do not restart it.

Project: /home/hdd3/zhanghaonan/projects/knowledge-deficit-mitigation.
Python: .environments/native311/bin/python (local copy of original Python3.11/torch2.9+cu128 plus transformers5.17 overlay).
Model: models/Qwen3.5-4B, read-only link to the exact existing holocue checkpoint.

CPU checks:

    .environments/native311/bin/python workflows/acceleration_4029_v1/runner.py --check-four-shards
    .environments/native311/bin/python workflows/acceleration_4029_v1/runner.py --model qwen35_4b --shard 0 --check-plan

The real four launches are already recorded in outputs/records/deployment_4029_v1/dispatch.json; do not repeat them. Each shard i uses physical GPU4+i and n_shards=4. The run is panel5_food_formal_4029_v1, with outputs in outputs/records/acceleration_v4 and outputs/raw/acceleration_v4. No automatic resume/retry; failures remain in the original folder.

admission.py verifies exact versions, original frozen files and proofs, registered GPU UUIDs, model configuration and original image bytes. It explicitly adapts host/path admission without modifying frozen modules on disk. runner.py invokes the unchanged formal_runner/run_tasks decoder with the original task order. worker.sh enforces project-local caches and a GPU lock; runner adds an independent data-shard lock.

Expected tasks per shard:39488,38720,39360,37568; sum155136. Complete merged evidence requires four unique shards, no duplicate keys and the exact original2424 eval-question/method matrix. Free formal_stage_postprocess can process an exactly completed UNKNOWN stage before later conditions finish. Generation completion is not semantic-label, joint-GT or paper completion.

Source project for central evidence:4028 /home/g203-4028/projects/knowledge-deficit-mitigation. Copy completed raw data and sidecars back there; preserve relative paths and source receipts. Monitor only every two hours unless an actionable failure/completion occurs. All other active4028/6403 workers continue independently.
