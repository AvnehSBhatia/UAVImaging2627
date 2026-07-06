"""Cache YOLO teacher predictions as soft cell grids (D27).

Boxes come back in original-image pixels; they are normalized then pushed
through the same letterbox transform as the GT labels so student and teacher
share one geometry.
"""

from pathlib import Path

import numpy as np

from ..data.rasterize import boxes_to_soft_grid, transform_boxes


def cache_teacher(weights, dataset_root, split, out_dir, imgsz=960, conf=0.15,
                  batch=16, device=None):
    from ultralytics import YOLO

    model = YOLO(weights)
    img_dir = Path(dataset_root) / "images" / split
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    for i in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[i : i + batch]]
        results = model.predict(chunk, imgsz=imgsz, conf=conf, device=device, verbose=False)
        for path, res in zip(paths[i : i + batch], results):
            h0, w0 = res.orig_shape
            rows = []
            if res.boxes is not None and len(res.boxes):
                xywhn = res.boxes.xywhn.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for (cx, cy, w, h), k, cf in zip(xywhn, cls, confs):
                    rows.append([k, cx, cy, w, h, cf])
            boxes = np.asarray(rows, np.float32).reshape(-1, 6)
            canvas_boxes = np.concatenate(
                [transform_boxes(boxes[:, :5], w0, h0), boxes[:, 5:6]], 1
            ) if len(boxes) else boxes
            grid = boxes_to_soft_grid(canvas_boxes)
            np.savez_compressed(out / f"{path.stem}.npz", grid=grid)
        done = min(i + batch, len(paths))
        if done % 1600 < batch:
            print(f"{split}: {done}/{len(paths)} cached")
