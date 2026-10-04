#!/usr/bin/env bash
set -euo pipefail
PROJECT=$1; GPUS=$2; PYTHON=$3; REGISTRY=$4; MODEL=$5; NAME=$6
shift 6
BASE=outputs/paper_core_20261002_dev_viz/viz_native_baselines_20261004_1645
mkdir -p "$PROJECT/$BASE/logs"
LOG="$PROJECT/$BASE/logs/$NAME.log"
[[ ! -e "$LOG" && ! -e "$PROJECT/$BASE/$NAME" ]]
if [[ "$REGISTRY" == configs/runtime/hosts.json ]]; then
  WORKER=(bash "$PROJECT/scripts/worker.sh" "$PROJECT" "$GPUS" "$PYTHON")
else
  WORKER=(bash "$PROJECT/workflows/supplemental/remaining4/worker_registered.sh" "$PROJECT" "$GPUS" "$PYTHON" "$PROJECT/$REGISTRY")
fi
nohup "${WORKER[@]}" workflows/paper_core/viz_author_native.py --model "$MODEL" --methods dola deco \
  --output "$BASE/$NAME" --physical-gpus "$GPUS" --registry "$REGISTRY" \
  --run-id "$NAME" --owner /root/sid_minicpm_full "$@" >"$LOG" 2>&1 </dev/null &
printf 'native %s PID=%s GPU=%s LOG=%s\n' "$NAME" "$!" "$GPUS" "$LOG"
