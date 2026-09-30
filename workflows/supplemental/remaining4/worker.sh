#!/usr/bin/env bash
set -euo pipefail
# Usage: worker.sh ROOT PHYSICAL_GPUS PYTHON HOST_REGISTRY [python arguments...]
ROOT="$(realpath "$1")"
GPUS="$2"
PYTHON="$3"
REGISTRY="$(realpath "$4")"
shift 4
"$PYTHON" -c 'import json,socket,sys; from pathlib import Path; r,p,c=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3].split(","); p.relative_to(r); h=json.load(p.open())["hosts"]; q=[v for v in h.values() if v["hostname"]==socket.gethostname() and Path(v["root"])==r]; assert len(q)==1 and len(c)==len(set(c)) and set(c)=={"0","1"} and set(map(int,c))<=set(q[0]["allowed_gpus"]) and q[0]["max_workers_per_gpu"]==1' "$ROOT" "$REGISTRY" "$GPUS"
mkdir -p "$ROOT/outputs/locks"
IFS=',' read -r -a cards <<< "$GPUS"
mapfile -t cards < <(printf '%s\n' "${cards[@]}" | sort -nu)
fd=20
for card in "${cards[@]}"; do
  eval "exec ${fd}>\"$ROOT/outputs/locks/gpu_${card}.lock\""
  flock -n "$fd" || { echo "GPU $card already has a KDM owner" >&2; exit 3; }
  fd=$((fd + 1))
done
export CUDA_VISIBLE_DEVICES="$GPUS" KDM_GPU_SLOTS='' PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$ROOT/cache/tmp" TMP="$ROOT/cache/tmp" TEMP="$ROOT/cache/tmp"
export HF_HOME="$ROOT/cache/huggingface" HF_MODULES_CACHE="$ROOT/cache/huggingface/modules"
export TORCH_HOME="$ROOT/cache/torch" XDG_CACHE_HOME="$ROOT/cache/xdg"
export MPLCONFIGDIR="$ROOT/cache/mpl" TRITON_CACHE_DIR="$ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor" TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions"
export CUDA_CACHE_PATH="$ROOT/cache/cuda" PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
mkdir -p "$TMPDIR" "$HF_MODULES_CACHE"
cd "$ROOT"
exec "$PYTHON" "$@"
