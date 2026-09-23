#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/k100/projects/knowledge-deficit-mitigation"
PYTHON="$ROOT/.environments/vllm024/bin/python"
[[ "$(hostname)" == "k100-X785-H30" ]] || exit 2
mkdir -p "$ROOT/outputs/locks"
exec 20>"$ROOT/outputs/locks/gpu_0.lock"
flock -n 20 || { echo "K100 GPU0 is already reserved" >&2; exit 3; }
export PATH="$ROOT/.environments/vllm024/bin:/usr/local/cuda/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/src:$ROOT" HF_HUB_DISABLE_XET=1
export TMPDIR="$ROOT/cache/tmp" TMP="$ROOT/cache/tmp" TEMP="$ROOT/cache/tmp"
export PIP_CACHE_DIR="$ROOT/cache/pip" HF_HOME="$ROOT/cache/hf" TORCH_HOME="$ROOT/cache/torch"
export XDG_CACHE_HOME="$ROOT/cache/xdg" MPLCONFIGDIR="$ROOT/cache/mpl" NLTK_DATA="$ROOT/cache/nltk"
export TRITON_CACHE_DIR="$ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions" CUDA_CACHE_PATH="$ROOT/cache/cuda"
export HF_HUB_CACHE="$ROOT/cache/hf/hub" HF_DATASETS_CACHE="$ROOT/cache/datasets" VLLM_CACHE_ROOT="$ROOT/cache/vllm"
export NUMBA_CACHE_DIR="$ROOT/cache/numba"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME/torch/kernels"
cd "$ROOT"
exec "$PYTHON" "$@"
