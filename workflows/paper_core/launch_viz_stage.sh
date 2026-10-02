#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$1"
MODEL="$2"
CARDS="$3"
PYTHON="$4"
REGISTRY="$5"
RUN_ID="$6"
SELECTED="$7"
MISSING="$8"
MODE="$9"
RELEASE="${10:-}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$ ]]
[[ "$MODEL" =~ ^[A-Za-z0-9_]+$ ]]
[[ -f "$PROJECT_ROOT/$SELECTED" && -f "$PROJECT_ROOT/$MISSING" ]]
options=()
case "$MODE" in
  pilot) ;;
  full) [[ -n "$RELEASE" && -f "$PROJECT_ROOT/$RELEASE" ]]; options=(--budget-release "$RELEASE") ;;
  *) echo 'VizWiz mode must be pilot or full' >&2; exit 2 ;;
esac
directory="$PROJECT_ROOT/outputs/paper_core_20261002_dev_viz/$RUN_ID/logs"
mkdir -p "$directory"
log="$directory/${MODEL}_${MODE}.log"
[[ ! -e "$log" ]] || { echo "Existing execution log: $log" >&2; exit 1; }
nohup bash "$PROJECT_ROOT/workflows/paper_core/worker.sh" "$PROJECT_ROOT" "$CARDS" "$PYTHON" \
  workflows/paper_core/dev_viz.py --model "$MODEL" --stage viz512 --run-id "$RUN_ID" \
  --physical-gpus "$CARDS" --registry "$REGISTRY" --mode "$MODE" \
  --selected-configs "$SELECTED" --missing-keys "$MISSING" "${options[@]}" \
  > "$log" 2>&1 < /dev/null &
pid="$!"
printf '%s\n' "$pid" > "$directory/${MODEL}_${MODE}.pid"
printf '%s MODE=%s PID=%s GPU=%s\n' "$MODEL" "$MODE" "$pid" "$CARDS"
