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
#   TORCH_INDEX_URL  ONLY if torch is NOT preinstalled: pip index to install it
#                    from. Default assumes the box's ROCm torch (e.g. ROCm 7.2)
#                    is already installed and must not be touched.
#   VENV             virtualenv path   (default: <repo>/.venv-rocm; created with
#                    --system-site-packages so it sees the preinstalled torch)
#   FORCE=1          rerun all stages (default: completed stages are skipped)
#   FORCE_STAGE=x    rerun one stage: deps|data|smoke|yolo|anet|teacher|distill|eval
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
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"   # unset = require the preinstalled ROCm torch
VENV="${VENV:-$REPO_ROOT/.venv-rocm}"
CONFIG="$ANET_DIR/configs/anet_mi300x.yaml"
YOLO_BEST="$ANET_DIR/runs/yolo/baseline/weights/best.pt"
STAGE_DIR="$ANET_DIR/runs/.stages"
LOG_DIR="$ANET_DIR/logs"
mkdir -p "$STAGE_DIR" "$LOG_DIR"

export DATA_ROOT ANET_DATA_ROOT="$DATA_ROOT"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"   # skip exhaustive conv tuning on first calls
export PYTHONUNBUFFERED=1

PY="$VENV/bin/python"

say()  { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
die()  { printf '\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

# stage <name> <command...>: run once, tee log, mark done; skip if already done
stage() {
    local name="$1"; shift
    local marker="$STAGE_DIR/$name.done"
    if [[ -f "$marker" && "${FORCE:-0}" != 1 && "${FORCE_STAGE:-}" != "$name" ]]; then
        say "stage '$name' already done (rm $marker to redo) — skipping"
        return 0
    fi
    say "stage: $name"
    ( "$@" ) 2>&1 | tee "$LOG_DIR/$name.log"
    touch "$marker"
}

# ---------------------------------------------------------------- deps -------
provision_deps() {
    command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showproductname || \
        echo "rocm-smi not found (ok inside some containers) — trusting torch probe below"
    # --system-site-packages: reuse the box's preinstalled ROCm torch/torchvision
    [[ -d "$VENV" ]] || python3 -m venv --system-site-packages "$VENV"
    "$VENV/bin/pip" install --upgrade pip

    if ! "$PY" -c "import torch" 2>/dev/null; then
        [[ -n "$TORCH_INDEX_URL" ]] || die "no torch visible in $VENV.
  This box should have ROCm torch preinstalled (system python). Either point
  VENV at an env that has it, or set TORCH_INDEX_URL=<rocm wheel index> to install."
        "$VENV/bin/pip" install --index-url "$TORCH_INDEX_URL" torch torchvision
    fi
    # torchvision must exist BEFORE installing ultralytics: if pip has to
    # resolve it, it grabs a CUDA/cpu wheel from PyPI and shadows the ROCm torch
    "$PY" -c "import torchvision" 2>/dev/null || die \
        "torchvision missing but torch present — install the ROCm 7.2 torchvision
  build matching your torch (same source as the preinstalled torch), then rerun."
    "$VENV/bin/pip" install ultralytics numpy pillow pyyaml
    "$PY" - <<'EOF'
import torch, torchvision
assert torch.cuda.is_available(), "torch.cuda unavailable — ROCm runtime/driver problem?"
hip = getattr(torch.version, "hip", None)
assert hip, f"torch {torch.__version__} is not a ROCm build (hip={hip}) — a PyPI wheel shadowed it"
name = torch.cuda.get_device_name(0)
print(f"torch {torch.__version__} (hip {hip}) | torchvision {torchvision.__version__} | {name}")
if "MI300" not in name:
    print(f"WARNING: expected MI300X, got '{name}' — continuing anyway")
EOF
    "$VENV/bin/pip" freeze > "$LOG_DIR/pip-freeze.txt"
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

run_yolo()    { "$PY" scripts/train_yolo.py --config "$CONFIG"; }

check_yolo()  { [[ -f "$YOLO_BEST" ]] || die "expected $YOLO_BEST after YOLO training"; }

run_anet()    { "$PY" scripts/train_anet.py --config "$CONFIG"; }

run_teacher() { "$PY" scripts/cache_teacher.py --config "$CONFIG" \
                    --weights "$YOLO_BEST" --splits train val; }

run_distill() { "$PY" scripts/train_anet_distill.py --config "$CONFIG"; }

run_eval()    { "$PY" scripts/evaluate_all.py --config "$CONFIG" \
                    --yolo "$YOLO_BEST" \
                    --anet runs/anet/best.pt \
                    --anet-distill runs/anet_distill/best.pt \
                    --latency \
                    --out runs/comparison.json; }

# ---------------------------------------------------------------- main -------
say "ANetV1 MI300X pipeline | repo=$REPO_ROOT | data=$DATA_ROOT"
stage deps    provision_deps
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
