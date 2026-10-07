#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$1"
CARDS="$2"
PYTHON="$3"
shift 3
IFS=',' read -r -a assigned_cards <<< "$CARDS"
fd=20
for card in "${assigned_cards[@]}"; do
  [[ "$card" == 0 || "$card" == 1 ]]
  eval "exec ${fd}>\"$PROJECT_ROOT/outputs/locks/gpu_${card}.lock\""
  flock -xn "$fd" || { echo "GPU $card is already locked" >&2; exit 3; }
  fd=$((fd + 1))
done
export CUDA_VISIBLE_DEVICES="$CARDS" PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PROJECT_ROOT/cache/tmp" TMP="$PROJECT_ROOT/cache/tmp" TEMP="$PROJECT_ROOT/cache/tmp"
export HF_HOME="$PROJECT_ROOT/cache/hf" TORCH_HOME="$PROJECT_ROOT/cache/torch"
export XDG_CACHE_HOME="$PROJECT_ROOT/cache/xdg" MPLCONFIGDIR="$PROJECT_ROOT/cache/mpl"
export TRITON_CACHE_DIR="$PROJECT_ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/cache/inductor"
export TORCH_EXTENSIONS_DIR="$PROJECT_ROOT/cache/torch_extensions" CUDA_CACHE_PATH="$PROJECT_ROOT/cache/cuda"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
unset KDM_GPU_SLOTS
cd "$PROJECT_ROOT"
exec "$PYTHON" "$@"
