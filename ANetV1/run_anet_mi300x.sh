#!/usr/bin/env bash
# Train ANetV1 from scratch on MI300X — run in a second terminal while YOLO trains.
#
# Usage (terminal 2, while run_mi300x.sh / YOLO is in terminal 1):
#   cd ANetV1 && ./run_anet_mi300x.sh
#
# Env:
#   DATA_ROOT   dataset root (default: <repo>/datasets/suas-synth-50k)
#   FORCE=1     rerun even if runs/.stages/anet.done exists
set -euo pipefail

cd "$(dirname "$0")"
ANET_DIR="$(pwd)"
REPO_ROOT="$(dirname "$ANET_DIR")"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/datasets/suas-synth-50k}"
STAGE_DIR="$ANET_DIR/runs/.stages"
LOG_DIR="$ANET_DIR/logs"
mkdir -p "$STAGE_DIR" "$LOG_DIR"

export DATA_ROOT ANET_DATA_ROOT="$DATA_ROOT"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export MIOPEN_LOG_LEVEL="${MIOPEN_LOG_LEVEL:-0}"
export NNPACK_DISABLE=1
export PYTHONUNBUFFERED=1
_ALLOC="expandable_segments:True,garbage_collection_threshold:0.8"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-$_ALLOC}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$_ALLOC}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$_ALLOC}"  # unified name (newer torch/ROCm)

if [[ -z "${PYTHON:-}" && -x /opt/venv/bin/python3 ]]; then
    PY=/opt/venv/bin/python3
else
    PY="${PYTHON:-python3}"
fi

# settings live IN scripts/train_anet.py now (no yaml) — edit that file to tune
MARKER="$STAGE_DIR/anet.done"
if [[ -f "$MARKER" && "${FORCE:-0}" != 1 ]]; then
    echo "anet stage already done (rm $MARKER or FORCE=1 to redo)"
    exit 0
fi

printf '\n== ANetV1 MI300X | data=%s | python=%s ==\n' "$DATA_ROOT" "$PY"

# --- forensics: something external SIGTERMs training ~5min in ("Terminated",
# no traceback, survives compile-off/worker cuts/VRAM bounds). Record system
# state every 10s so the death leaves evidence, and run python in its OWN
# SESSION (setsid) so signals aimed at this shell's process group miss it.
SYSMON="$LOG_DIR/sysmon.log"
( while true; do
    rss=$(ps -o rss= -C python3 2>/dev/null | sort -rn | head -1)
    vram=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -1 | cut -d, -f3)
    echo "$(date +%H:%M:%S) mem_avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo) py_rss_kb=${rss:-0} vram_used=${vram:-?}"
    sleep 10
  done >> "$SYSMON" ) &
MON_PID=$!
trap 'kill $MON_PID 2>/dev/null' EXIT

set +e
setsid "$PY" scripts/train_anet.py < /dev/null 2>&1 | tee "$LOG_DIR/anet.log"
rc=${PIPESTATUS[0]}
set -e
if [[ "$rc" != 0 ]]; then
    echo "=== training exited rc=$rc — last system samples ==="
    tail -6 "$SYSMON" || true
    echo "=== kernel log (look for amdgpu/KFD/oom lines) ==="
    dmesg 2>/dev/null | tail -20 || echo "(dmesg not permitted in this container)"
    exit "$rc"
fi
touch "$MARKER"
echo "done -> $ANET_DIR/runs/anet/best.pt"
