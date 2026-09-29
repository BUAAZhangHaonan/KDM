#!/usr/bin/env bash
set -euo pipefail
# Usage: worker.sh PROJECT PHYSICAL_GPUS PYTHON [python arguments...]
ROOT="$(realpath "$1")"; GPUS="$2"; PYTHON="$3"; shift 3
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src" "$PYTHON" -m kdm.execution --root "$ROOT" --cards "$GPUS"
mkdir -p "$ROOT/outputs/locks"
IFS=',' read -r -a cards <<< "$GPUS"
[[ ${#cards[@]} -gt 0 ]] || exit 2
# Sorted lock acquisition prevents a pair of workers deadlocking.
mapfile -t cards < <(printf '%s\n' "${cards[@]}" | sort -nu)
fd=20
for card in "${cards[@]}"; do
  eval "exec ${fd}>\"$ROOT/outputs/locks/gpu_${card}.lock\""
  flock -n "$fd" || { echo "GPU $card already reserved" >&2; exit 3; }
  fd=$((fd+1))
done
export CUDA_VISIBLE_DEVICES="$GPUS" HF_HUB_DISABLE_XET=1
export TMPDIR="$ROOT/cache/tmp" TMP="$ROOT/cache/tmp" TEMP="$ROOT/cache/tmp"
export PYTHONDONTWRITEBYTECODE=1 PIP_CACHE_DIR="$ROOT/cache/pip"
mkdir -p "$TMPDIR"
export HF_HOME="$ROOT/cache/hf" TORCH_HOME="$ROOT/cache/torch"
export XDG_CACHE_HOME="$ROOT/cache/xdg" MPLCONFIGDIR="$ROOT/cache/mpl" NLTK_DATA="$ROOT/cache/nltk"
export TRITON_CACHE_DIR="$ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions" CUDA_CACHE_PATH="$ROOT/cache/cuda"
export HF_HUB_CACHE="$ROOT/cache/hf/hub" HF_DATASETS_CACHE="$ROOT/cache/datasets" VLLM_CACHE_ROOT="$ROOT/cache/vllm"
export NUMBA_CACHE_DIR="$ROOT/cache/numba"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$PYTHON" "$@"
