#!/usr/bin/env bash
# Train ANetV1 from scratch on MI300X — run in a second terminal while YOLO trains.
#
# Usage (terminal 2, while run_mi300x.sh / YOLO is in terminal 1):
#   cd ANetV1 && ./run_anet_mi300x.sh
#
# Env:
#   DATA_ROOT   dataset root (default: <repo>/datasets/suas-synth-50k)
# Fast defaults (2026-07-08 memory/throughput work): batch 16 x accum 4
# (effective 64 unchanged), compile ON (inductor fuses + rematerializes ->
# ~10GB VRAM), 4 spawn workers over the memmap dataset cache (first run
# builds it: ~70GB under $DATA_ROOT/.anet_cache, ~10 min one-time).
# Escape hatches if the container misbehaves:
#   ANET_COMPILE=0        eager (trainer also auto-falls-back on any error)
#   ANET_CKPT=1           per-round checkpointing (pair with eager: caps VRAM)
#   ANET_NUM_WORKERS=0    in-process loader
#   ANET_CACHE=0          no disk cache (slow PIL decode path)
set -euo pipefail

cd "$(dirname "$0")"
ANET_DIR="$(pwd)"
REPO_ROOT="$(dirname "$ANET_DIR")"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/datasets/suas-synth-50k}"
STAGE_DIR="$ANET_DIR/runs/.stages"
LOG_DIR="$ANET_DIR/logs"
mkdir -p "$STAGE_DIR" "$LOG_DIR"

export DATA_ROOT ANET_DATA_ROOT="$DATA_ROOT"
# The old epoch-0 hangs are root-caused and fixed (fork->spawn workers,
# inductor compile-workers capped to 1); the old 115GB VRAM is fixed at the
# source (fused BN, bf16 stream, per-round checkpointing — ~10GB at batch 32).
# Presets now default to the fast path; only pin what differs per-box here.
export ANET_NUM_WORKERS="${ANET_NUM_WORKERS:-4}"
export ANET_COMPILE="${ANET_COMPILE:-1}"
export ANET_BATCH="${ANET_BATCH:-16}"
export ANET_ACCUM="${ANET_ACCUM:-4}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export MIOPEN_LOG_LEVEL="${MIOPEN_LOG_LEVEL:-0}"
export NNPACK_DISABLE=1
export PYTHONUNBUFFERED=1
# torch.compile (on for CUDA/ROCm via presets): cap inductor to ONE compile
# worker so it can't fork ncpu full-torch processes and OOM the host compiling
# the backward graph. A warm on-disk cache makes reruns skip recompilation.
# Set ANET_COMPILE=0 to disable.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ANET_DIR/.torchinductor}"
_ALLOC="expandable_segments:True,garbage_collection_threshold:0.8"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-$_ALLOC}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$_ALLOC}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$_ALLOC}"  # unified name (newer torch/ROCm)

if [[ -z "${PYTHON:-}" && -x /opt/venv/bin/python3 ]]; then
    PY=/opt/venv/bin/python3
else
    PY="${PYTHON:-python3}"
fi

# settings live IN scripts/train_anet.py now (no yaml) — edit that file to tune.
# Running this script MEANS "train ANet now" — no stage-marker gate. The only
# guard: never start a second trainer on top of a live one.
if pgrep -f "train_anet.py" > /dev/null 2>&1; then
    echo "a train_anet.py is ALREADY RUNNING:"
    pgrep -af "train_anet.py"
    echo "watch it:  tail -f logs/anet.log"
    echo "kill it:   pkill -f train_anet.py      then rerun this script"
    exit 1
fi

printf '\n== ANetV1 MI300X | data=%s | python=%s ==\n' "$DATA_ROOT" "$PY"

"$PY" scripts/train_anet.py 2>&1 | tee "$LOG_DIR/anet.log"
touch "$STAGE_DIR/anet.done"
echo "done -> $ANET_DIR/runs/anet/best.pt"
