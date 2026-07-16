"""Train a YOLO26n on suas-synth-50k and export it to ONNX — nothing else.

No yaml, no flags — edit SETTINGS below and run (from inside ANetV1/):
    python scripts/train_export_yolo26n.py

Writes runs/yolo/yolo26n/weights/best.pt (separate run dir from the
distillation baseline in runs/yolo/baseline — this script never touches that)
and exports runs/yolo/yolo26n/weights/best.onnx: static 960 input, no in-graph
NMS, opset 12 — the shape the Hailo-8 DFC ingests.

Env overrides (all optional): YOLO_EPOCHS, YOLO_BATCH, YOLO_IMGSZ, YOLO_OPSET,
YOLO_EXPORT_ONLY=1 (skip training, just export an existing best.pt), DATA_ROOT.
"""

import os
import sys
from pathlib import Path

# MPS fallback + allocator tuning must land before torch initializes a backend.
# expandable_segments/garbage_collection: mosaic's variable instance counts keep
# ratcheting the caching allocator's high-water mark on CUDA/ROCm (looks like a
# leak); these let blocks return to the driver. ROCm honors the HIP key.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_alloc = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _alloc)
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", _alloc)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anet.train.presets import IS_CUDA, anet_cfg  # noqa: E402
from anet.train.trainer import yolo_device  # noqa: E402

# --------------------------------------------------------------------------
# EDIT HERE
# --------------------------------------------------------------------------
SETTINGS = dict(
    weights="yolo26n.pt",        # falls back to yolo11n.pt if unavailable
    imgsz=int(os.environ.get("YOLO_IMGSZ", 960)),   # spec comparison point (ARCH §10)
    epochs=int(os.environ.get("YOLO_EPOCHS", 60)),
    patience=10,                 # ultralytics native early stop on fitness
    batch=int(os.environ["YOLO_BATCH"]) if "YOLO_BATCH" in os.environ
    else (64 if IS_CUDA else 16),
    workers=8 if IS_CUDA else 0,  # 0 on MPS: ultralytics worker RAM leak (#22791)
    amp=IS_CUDA,                 # CUDA amp is well-trodden; MPS autocast is flaky
    project="runs/yolo",
    name="yolo26n",
    opset=int(os.environ.get("YOLO_OPSET", 12)),  # Hailo DFC-supported opset
)
DATA_ROOT = anet_cfg().data.root  # env DATA_ROOT or <repo>/datasets/suas-synth-50k


def main():
    from ultralytics import YOLO

    anet_root = Path(__file__).resolve().parents[1]
    run_dir = anet_root / SETTINGS["project"] / SETTINGS["name"]
    best = run_dir / "weights" / "best.pt"

    if os.environ.get("YOLO_EXPORT_ONLY") == "1":
        if not best.exists():
            raise SystemExit(f"YOLO_EXPORT_ONLY=1 but {best} does not exist — "
                             "train first (unset YOLO_EXPORT_ONLY)")
    else:
        try:
            model = YOLO(SETTINGS["weights"])
        except Exception as e:  # yolo26n not in this ultralytics version yet
            print(f"{SETTINGS['weights']} unavailable ({e}); falling back to yolo11n.pt")
            model = YOLO("yolo11n.pt")
        model.train(
            data=str(Path(DATA_ROOT) / "data.yaml"),
            imgsz=SETTINGS["imgsz"],
            epochs=SETTINGS["epochs"],
            patience=SETTINGS["patience"],
            batch=SETTINGS["batch"],
            device=yolo_device(),
            project=str(anet_root / SETTINGS["project"]),
            name=SETTINGS["name"],
            exist_ok=True,  # resume-safe: never silently forks yolo26n2/
            cache=False,
            workers=SETTINGS["workers"],
            amp=SETTINGS["amp"],
        )

    # Export the selected-best weights, not the live training model: best.pt is
    # the fitness-selected checkpoint, model after .train() holds last-epoch state.
    # YOLO26 is natively NMS-free: the graph ends in decoded detections
    # (1, 300, 6) = (xyxy, conf, cls) with no NMS op — nms=False just keeps
    # ultralytics from appending an extra postprocess stage. The Hailo DFC will
    # still cut before the topk/gather decode tail; that decode runs host-side.
    onnx_path = YOLO(str(best)).export(
        format="onnx",
        imgsz=SETTINGS["imgsz"],
        opset=SETTINGS["opset"],
        dynamic=False,
        simplify=True,
        nms=False,
        device="cpu",  # export shape/tracing is device-independent; cpu is safest
    )
    print(f"\ntrained : {best}\nexported: {onnx_path}")


if __name__ == "__main__":
    main()
