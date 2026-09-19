#!/usr/bin/env bash
set -euo pipefail
# Usage: worker.sh PROJECT PHYSICAL_GPUS PYTHON [python arguments...]
ROOT="$(realpath "$1")"; GPUS="$2"; PYTHON="$3"; shift 3
[[ "$ROOT" == /home/g203-4028/projects/knowledge-deficit-mitigation ]] || { echo 'Unexpected project root' >&2; exit 2; }
mkdir -p "$ROOT/outputs/locks"
IFS=',' read -r -a cards <<< "$GPUS"
[[ ${#cards[@]} -gt 0 ]] || exit 2
# Sorted lock acquisition prevents a pair of workers deadlocking.
mapfile -t cards < <(printf '%s\n' "${cards[@]}" | sort -nu)
fd=20
for card in "${cards[@]}"; do
  [[ "$card" =~ ^(0|1|4|5)$ ]] || { echo "Unauthorized GPU $card" >&2; exit 2; }
  eval "exec ${fd}>\"$ROOT/outputs/locks/gpu_${card}.lock\""
  flock -n "$fd" || { echo "GPU $card already reserved" >&2; exit 3; }
  fd=$((fd+1))
done
export CUDA_VISIBLE_DEVICES="$GPUS" HF_HUB_DISABLE_XET=1
export HF_HOME="$ROOT/cache/hf" TORCH_HOME="$ROOT/cache/torch"
export XDG_CACHE_HOME="$ROOT/cache/xdg" MPLCONFIGDIR="$ROOT/cache/mpl" NLTK_DATA="$ROOT/cache/nltk"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$PYTHON" "$@"
