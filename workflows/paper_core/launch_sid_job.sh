#!/usr/bin/env bash
set -euo pipefail
PROJECT=$1; GPUS=$2; PYTHON=$3; REGISTRY=$4; MODEL=$5; MODE=$6; NAME=$7
shift 7
BASE=outputs/paper_core_20261002_dev_viz/viz_native_baselines_20261004_1645
mkdir -p "$PROJECT/$BASE/logs"
LOG="$PROJECT/$BASE/logs/$NAME.log"
[[ ! -e "$LOG" && ! -e "$PROJECT/$BASE/$NAME" ]]
if [[ "$REGISTRY" == configs/runtime/hosts.json ]]; then
  WORKER=(bash "$PROJECT/scripts/worker.sh" "$PROJECT" "$GPUS" "$PYTHON")
else
  WORKER=(bash "$PROJECT/workflows/supplemental/remaining4/worker_registered.sh" "$PROJECT" "$GPUS" "$PYTHON" "$PROJECT/$REGISTRY")
fi
nohup "${WORKER[@]}" workflows/paper_core/viz_sid_author.py --model "$MODEL" --mode "$MODE" \
  --output "$BASE/$NAME" --physical-gpus "$GPUS" --registry "$REGISTRY" \
  --run-id "$NAME" "$@" >"$LOG" 2>&1 </dev/null &
printf 'SID %s PID=%s GPU=%s LOG=%s\n' "$NAME" "$!" "$GPUS" "$LOG"
