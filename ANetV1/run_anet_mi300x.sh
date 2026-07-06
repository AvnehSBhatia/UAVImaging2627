#!/usr/bin/env bash
# Train ANetV1 from scratch on MI300X — run in a second terminal while YOLO trains.
#
# Usage (terminal 2, while run_mi300x.sh / YOLO is in terminal 1):
#   cd ANetV1 && ./run_anet_mi300x.sh
#
# Env:
#   DATA_ROOT   dataset root (default: <repo>/datasets/suas-synth-50k)
#   PARALLEL=1  use batch-32 config safe alongside YOLO (default: 1)
#   PARALLEL=0  use full batch-64 anet_mi300x.yaml (YOLO should be finished)
#   FORCE=1     rerun even if runs/.stages/anet.done exists
set -euo pipefail

cd "$(dirname "$0")"
ANET_DIR="$(pwd)"
REPO_ROOT="$(dirname "$ANET_DIR")"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/datasets/suas-synth-50k}"
PARALLEL="${PARALLEL:-1}"
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

if [[ -z "${PYTHON:-}" && -x /opt/venv/bin/python3 ]]; then
    PY=/opt/venv/bin/python3
else
    PY="${PYTHON:-python3}"
fi

if [[ "$PARALLEL" == "1" ]]; then
    CONFIG="$ANET_DIR/configs/anet_mi300x_parallel.yaml"
else
    CONFIG="$ANET_DIR/configs/anet_mi300x.yaml"
fi

MARKER="$STAGE_DIR/anet.done"
if [[ -f "$MARKER" && "${FORCE:-0}" != 1 ]]; then
    echo "anet stage already done (rm $MARKER or FORCE=1 to redo)"
    exit 0
fi

printf '\n== ANetV1 MI300X | data=%s | parallel=%s | python=%s ==\n' \
    "$DATA_ROOT" "$PARALLEL" "$PY"

"$PY" scripts/train_anet.py --config "$CONFIG" 2>&1 | tee "$LOG_DIR/anet.log"
touch "$MARKER"
echo "done -> $ANET_DIR/runs/anet/best.pt"
