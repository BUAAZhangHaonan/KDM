#!/usr/bin/env bash
set -euo pipefail
# Exact4029 target; one registered GPU and original single-card model per worker.
ROOT="$(realpath "$1")"; GPUS="$2"; PYTHON="$3"; shift 3
[[ "$ROOT" == "/home/hdd3/zhanghaonan/projects/knowledge-deficit-mitigation" ]] || exit 2
[[ "$GPUS" =~ ^[4567]$ ]] || exit 2
[[ "$PYTHON" == "$ROOT/.environments/native311/bin/python" ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/src:$ROOT"
"$PYTHON" "$ROOT/workflows/acceleration_4029_v1/admission.py" --root "$ROOT" --cards "$GPUS"
mkdir -p "$ROOT/outputs/locks"
exec 20>"$ROOT/outputs/locks/gpu_$GPUS.lock"
flock -n 20 || { echo "GPU $GPUS already reserved" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES="$GPUS" HF_HUB_DISABLE_XET=1
export TMPDIR="$ROOT/cache/tmp" TMP="$ROOT/cache/tmp" TEMP="$ROOT/cache/tmp"
export PIP_CACHE_DIR="$ROOT/cache/pip" HF_HOME="$ROOT/cache/hf" TORCH_HOME="$ROOT/cache/torch"
export XDG_CACHE_HOME="$ROOT/cache/xdg" MPLCONFIGDIR="$ROOT/cache/mpl" NLTK_DATA="$ROOT/cache/nltk"
export TRITON_CACHE_DIR="$ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions" CUDA_CACHE_PATH="$ROOT/cache/cuda"
export HF_HUB_CACHE="$ROOT/cache/hf/hub" HF_DATASETS_CACHE="$ROOT/cache/datasets" VLLM_CACHE_ROOT="$ROOT/cache/vllm"
export NUMBA_CACHE_DIR="$ROOT/cache/numba"
mkdir -p "$TMPDIR"
cd "$ROOT"
exec "$PYTHON" "$@"

