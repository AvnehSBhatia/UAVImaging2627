#!/usr/bin/env bash
# =============================================================================
# ANetV1 full experiment pipeline for a single AMD MI300X (ROCm -> torch "cuda")
#
#   data load/verify -> YOLO baseline (early stop) -> ANetV1 from scratch
#   (early stop) -> teacher cache -> ANetV1 distilled (early stop)
#   -> three-way benchmark: accuracy (cell + object + slices) and latency
#
# Usage:
#   DATA_ROOT=/data/suas-synth-50k ./run_mi300x.sh
#
# Env knobs (all optional):
#   DATA_ROOT        dataset dir (default: <repo>/datasets/suas-synth-50k)
#   DATA_SRC         rsync source (user@host:/path) or .tar.gz URL to fetch the
#                    dataset from if DATA_ROOT does not exist
#   PYTHON           python binary (default: python3; box must have ROCm torch)
#   FORCE=1          rerun all stages (default: completed stages are skipped)
#   FORCE_STAGE=x    rerun one stage: data|smoke|yolo|anet|teacher|distill|eval
#
# Stages are checkpointed in runs/.stages/ — the script is safe to rerun after
# any failure; it resumes at the first incomplete stage. All output tees to
# logs/<stage>.log.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"
ANET_DIR="$(pwd)"
REPO_ROOT="$(dirname "$ANET_DIR")"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/datasets/suas-synth-50k}"
DATA_SRC="${DATA_SRC:-}"
YOLO_BEST="$ANET_DIR/runs/yolo/baseline/weights/best.pt"
STAGE_DIR="$ANET_DIR/runs/.stages"
LOG_DIR="$ANET_DIR/logs"
mkdir -p "$STAGE_DIR" "$LOG_DIR"

export DATA_ROOT ANET_DATA_ROOT="$DATA_ROOT"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"   # NORMAL needs writable MIOpen SQLite DBs, which this container lacks (miopenStatusInternalError); FAST is the safe default
export MIOPEN_LOG_LEVEL="${MIOPEN_LOG_LEVEL:-0}"
export NNPACK_DISABLE=1
export ANET_SMOKE_SKIP_CPU="${ANET_SMOKE_SKIP_CPU:-1}"  # MI300X: skip 40s CPU path, cuda is the target
export PYTHONUNBUFFERED=1
# Curb ROCm reserved-memory growth (mosaic high-water mark + fragmentation) so YOLO
# doesn't creep toward the VF ceiling across epochs. ROCm honors the HIP key and
# aliases the CUDA one; set both. Must be exported before any torch process starts.
_ALLOC="expandable_segments:True,garbage_collection_threshold:0.8"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-$_ALLOC}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$_ALLOC}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$_ALLOC}"  # unified name (newer torch/ROCm)

# Container images usually preinstall ROCm torch in /opt/venv, not system python3.
if [[ -z "${PYTHON:-}" && -x /opt/venv/bin/python3 ]]; then
    PY=/opt/venv/bin/python3
else
    PY="${PYTHON:-python3}"
fi

say()  { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
die()  { printf '\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

# stage <name> <command...>: run once, tee log, mark done; skip if already done.
stage() {
    local name="$1"; shift
    local marker="$STAGE_DIR/$name.done"
    if [[ -f "$marker" && "${FORCE:-0}" != 1 && "${FORCE_STAGE:-}" != "$name" ]]; then
        say "stage '$name' already done (rm $marker to redo) — skipping"
        return 0
    fi
    say "stage: $name"
    set +e
    ( "$@" ) 2>&1 | tee "$LOG_DIR/$name.log"
    local rc=${PIPESTATUS[0]}
    set -e
    [[ "$rc" == 0 ]] || exit "$rc"
    touch "$marker"
}

# ---------------------------------------------------------------- data -------
provision_data() {
    if [[ ! -d "$DATA_ROOT/images" ]]; then
        [[ -n "$DATA_SRC" ]] || die "dataset not at $DATA_ROOT and DATA_SRC unset.
  Either: DATA_SRC=user@host:/path/suas-synth-50k  (rsync)
      or: DATA_SRC=https://.../suas-synth-50k.tar.gz (tarball)
      or: rsync it yourself and set DATA_ROOT"
        mkdir -p "$DATA_ROOT"
        case "$DATA_SRC" in
            http://*|https://*) curl -fL "$DATA_SRC" | tar -xz -C "$DATA_ROOT" --strip-components=1 ;;
            *)                  rsync -a --info=progress2 "$DATA_SRC/" "$DATA_ROOT/" ;;
        esac
    fi
    "$PY" - <<EOF
from pathlib import Path
root = Path("$DATA_ROOT")
assert (root / "data.yaml").is_file(), f"missing {root}/data.yaml"
for split in ("train", "val", "test"):
    imgs = len(list((root / "images" / split).glob("*.[jp][pn]g")))
    lbls = len(list((root / "labels" / split).glob("*.txt")))
    print(f"{split}: {imgs} images / {lbls} labels")
    assert imgs > 0, f"empty split {split}"
    assert lbls >= imgs * 0.85, f"{split}: labels look incomplete ({lbls}/{imgs})"
print("dataset ok:", root)
EOF
}

# --------------------------------------------------------------- stages ------
run_smoke()   { "$PY" scripts/smoke_test.py; }

run_yolo()    { "$PY" scripts/train_yolo.py; }

check_yolo()  { [[ -f "$YOLO_BEST" ]] || die "expected $YOLO_BEST after YOLO training"; }

run_anet()    { "$PY" scripts/train_anet.py; }

run_teacher() { "$PY" scripts/cache_teacher.py \
                    --weights "$YOLO_BEST" --splits train val; }

run_distill() { "$PY" scripts/train_anet_distill.py; }

run_eval()    { "$PY" scripts/evaluate_all.py \
                    --yolo "$YOLO_BEST" \
                    --anet runs/anet/best.pt \
                    --anet-distill runs/anet_distill/best.pt \
                    --latency \
                    --out runs/comparison.json; }

# ---------------------------------------------------------------- main -------
say "ANetV1 MI300X pipeline | repo=$REPO_ROOT | data=$DATA_ROOT | python=$PY"
stage data    provision_data
stage smoke   run_smoke
stage yolo    run_yolo
check_yolo
stage teacher run_teacher
stage anet    run_anet
stage distill run_distill
stage eval    run_eval

say "all done"
echo "  YOLO baseline:   $YOLO_BEST"
echo "  ANetV1:          $ANET_DIR/runs/anet/best.pt        (log: runs/anet/log.csv)"
echo "  ANetV1-distill:  $ANET_DIR/runs/anet_distill/best.pt (log: runs/anet_distill/log.csv)"
echo "  comparison:      $ANET_DIR/runs/comparison.json      (table above, in logs/eval.log)"
