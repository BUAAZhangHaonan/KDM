# Resource withdrawal — 2026-09-24

The user explicitly removed4029 from the available KDM server queue. This overrides the4029 resource authorization in previous amendments for all future dispatch and routine monitoring. Existing completed Qwen3.5 formal results (155136 tasks) have already been synchronized and verified on4028; preserve those records and their historical source identities. No deletion or rerun is authorized by this change.

Current available resources:4028 physicalGPU0/1/4/5;6403 physicalGPU0/1;K100 (SSH RTX_Pro_6000) GPU0. At the live check,4028 runs Qwen2.5, LLaVA, MiniCPM and Gemma formal methods on0/1/4/5 respectively;6403 GPU1 runs Gemma independent answers.6403 GPU0 and K100 have no active KDM inference. Leave unrelated tasks untouched.

Current operational resource plan is workflows/independent_v5/priority_plan.json. Historical4029 sources/specs/receipts remain unchanged for reproducibility, not as launch authorization. No new allocation to idle GPUs is implied by this status report. Existing five-model priority and separate engine/annotation/GT boundaries remain.
