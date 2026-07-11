#!/usr/bin/env bash
# Train ANetV1 v9 from scratch on MI300X (20 GB VRAM budget).
#
# Usage:
#   cd ANetV1 && ./run_anet_mi300x.sh
#
# Env:
#   DATA_ROOT   dataset root (default: <repo>/datasets/suas-synth-50k)
#
# What happens on startup (all automatic, watch the log):
#   1. DeployNorm stat seeding — 8 forward-only batches (D39).
#   2. Fused Stage-1 parity checks (D40): the Triton forward is compared
#      against the PyTorch reference on real frames, then the Triton backward
#      against the chunked-autograd backward. Any mismatch demotes one level:
#        triton fwd+bwd  ->  triton fwd + chunked-autograd bwd  ->  PyTorch
#      dense path (batch auto-drops to ANET_FALLBACK-safe 32). The log prints
#      which level you landed on — "fused Stage-1 ON (bwd=triton...)" is the
#      fast path.
#   3. Training. Expected epoch times (19k train images):
#        fused triton bwd    ~15-45 s/epoch + ~5-15 s eval
#        fused chunked bwd   ~45-120 s/epoch
#        dense fallback      ~3-6 min/epoch (batch 32)
#      First epoch adds one-time costs: memmap cache build (~10 min if cold),
#      triton/inductor compile (~1-5 min, cached on disk afterwards).
#
# VRAM: fused path ~6-9 GB at batch 96; dense fallback ~12-16 GB at batch 32.
# Both inside the 20 GB budget; the allocator is capped so an overrun raises
# a catchable OOM instead of the driver SIGTERMing the process.
#
# Escape hatches:
#   ANET_FUSED=0          skip the fused path entirely (PyTorch dense)
#   ANET_FUSED_BWD=chunked  keep the triton forward, autograd backward
#   ANET_COMPILE=0        eager (trainer also auto-falls-back on any error)
#   ANET_BATCH=64         if a bigger config OOMs
#   ANET_CACHE=0          no disk cache (slow PIL decode path)
#   ANET_NUM_WORKERS=0    in-process loader (default on ROCm; the background
#                         prefetcher hides the loader cost)
set -euo pipefail

cd "$(dirname "$0")"
ANET_DIR="$(pwd)"
REPO_ROOT="$(dirname "$ANET_DIR")"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/datasets/suas-synth-50k}"
STAGE_DIR="$ANET_DIR/runs/.stages"   # run_mi300x.sh reads anet.done to skip this stage
LOG_DIR="$ANET_DIR/logs"
mkdir -p "$STAGE_DIR" "$LOG_DIR"

export DATA_ROOT ANET_DATA_ROOT="$DATA_ROOT"

# ---- batch/loader: fused Stage-1 keeps activations tiny, so batch 96 fits
# easily; if the fused path demotes to dense the trainer rebuilds the loader
# at fallback_batch=32 by itself. workers=0: spawn dataloader workers deadlock
# epoch-0 on this HIP container (fork + locked MIOpen mutexes); the in-process
# loader + background prefetch thread keeps the GPU fed from the memmap cache.
export ANET_NUM_WORKERS="${ANET_NUM_WORKERS:-0}"
export ANET_BATCH="${ANET_BATCH:-96}"
export ANET_ACCUM="${ANET_ACCUM:-1}"

# ---- compile: fuses the tail (neck/head/loss — dozens of tiny tensors) even
# when the Triton Stage-1 is on; inductor breaks the graph around the custom
# op and compiles the rest. Cap to ONE compile worker (host-OOM guard) and
# keep a warm on-disk cache.
export ANET_COMPILE="${ANET_COMPILE:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ANET_DIR/.torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ANET_DIR/.triton}"

# ---- ROCm/MIOpen hygiene
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export MIOPEN_LOG_LEVEL="${MIOPEN_LOG_LEVEL:-0}"
export NNPACK_DISABLE=1
export PYTHONUNBUFFERED=1

# ---- 20 GB VRAM budget: expandable segments avoid fragmentation, and the
# allocator cap (cuda_memory_frac in presets, applied to the VISIBLE VRAM)
# turns an overrun into a catchable torch OOM with a traceback.
_ALLOC="expandable_segments:True,garbage_collection_threshold:0.8"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-$_ALLOC}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$_ALLOC}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$_ALLOC}"

if [[ -z "${PYTHON:-}" && -x /opt/venv/bin/python3 ]]; then
    PY=/opt/venv/bin/python3
else
    PY="${PYTHON:-python3}"
fi

# settings live IN scripts/train_anet.py (no yaml) — edit that file to tune.
# The only guard: never start a second trainer on top of a live one.
if pgrep -f "train_anet.py" > /dev/null 2>&1; then
    echo "a train_anet.py is ALREADY RUNNING:"
    pgrep -af "train_anet.py"
    echo "watch it:  tail -f logs/anet.log"
    echo "kill it:   pkill -f train_anet.py      then rerun this script"
    exit 1
fi

printf '\n== ANetV1 v9 MI300X | data=%s | python=%s ==\n' "$DATA_ROOT" "$PY"
printf '== batch=%s fused=%s compile=%s ==\n' \
    "$ANET_BATCH" "${ANET_FUSED:-1}" "$ANET_COMPILE"

"$PY" scripts/train_anet.py 2>&1 | tee "$LOG_DIR/anet.log"
touch "$STAGE_DIR/anet.done"   # keeps the run_mi300x.sh pipeline from retraining
echo "done -> $ANET_DIR/runs/anet/best.pt (weight-EMA checkpoint, D48)"
