#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/team/zhanghaonan/TAFFC/knowledge-deficit-mitigation
export VLLM_WORKER_MULTIPROC_METHOD=spawn
exec bash "$ROOT/workflows/food_closed_v3/worker.sh" "$ROOT" 0 /home/team/lvshuyang/anaconda3/envs/ST_LORA/bin/python -u workflows/remaining11_llava15_v1/closed_entry_spawn_v2.py
