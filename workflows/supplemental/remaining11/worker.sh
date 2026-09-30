#!/usr/bin/env bash
set -euo pipefail
# Usage: worker.sh PROJECT PHYSICAL_GPUS PYTHON [python arguments...]
ROOT="$(realpath "$1")"
GPUS="$2"
PYTHON="$3"
shift 3
mkdir -p "$ROOT/outputs/locks"
MAX_WORKERS="$($PYTHON -c 'import json,socket,sys; r=json.load(open(sys.argv[1])); print(next(h.get("max_workers_per_gpu",1) for h in r["hosts"].values() if h["hostname"]==socket.gethostname()))' "$ROOT/workflows/supplemental/remaining11/host_registry.json")"
IFS=',' read -r -a cards <<< "$GPUS"
[[ ${#cards[@]} -gt 0 ]]
mapfile -t cards < <(printf '%s\n' "${cards[@]}" | sort -nu)
fd=20
slot_fd=100
SLOTS=''
for card in "${cards[@]}"; do
  eval "exec ${fd}>\"$ROOT/outputs/locks/gpu_${card}.lock\""
  if [[ "$MAX_WORKERS" -eq 1 ]]; then
    flock -n "$fd" || { echo "GPU $card already reserved by a KDM worker" >&2; exit 3; }
  else
    flock -sn "$fd" || { echo "GPU $card has an exclusive KDM worker" >&2; exit 3; }
    acquired=0
    for ((slot=0; slot<MAX_WORKERS; slot++)); do
      eval "exec ${slot_fd}>\"$ROOT/outputs/locks/gpu_${card}_slot_${slot}.lock\""
      if flock -n "$slot_fd"; then
        SLOTS="${SLOTS}${SLOTS:+,}${card}:${slot}"
        acquired=1
        break
      fi
      eval "exec ${slot_fd}>&-"
    done
    [[ "$acquired" -eq 1 ]] || { echo "GPU $card supplemental worker slots are reserved" >&2; exit 3; }
    slot_fd=$((slot_fd + 1))
  fi
  fd=$((fd + 1))
done
export KDM_GPU_SLOTS="$SLOTS"
export CUDA_VISIBLE_DEVICES="$GPUS" PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$ROOT/cache/tmp" TMP="$ROOT/cache/tmp" TEMP="$ROOT/cache/tmp"
mkdir -p "$TMPDIR"
export HF_HOME="$ROOT/cache/hf" TORCH_HOME="$ROOT/cache/torch"
export XDG_CACHE_HOME="$ROOT/cache/xdg" MPLCONFIGDIR="$ROOT/cache/mpl"
export TRITON_CACHE_DIR="$ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions" CUDA_CACHE_PATH="$ROOT/cache/cuda"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
cd "$ROOT"
exec "$PYTHON" "$@"
