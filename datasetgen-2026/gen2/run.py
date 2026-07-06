"""gen2 CLI: generate the dataset with multiprocessing.

Usage:
  .venv/bin/python -m gen2.run --config gen2/config.yaml [--total N] [--out DIR]
                               [--workers N] [--preview-only]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .assets import AssetLibrary
from .config import load_config
from .scene import compose_frame
from .sensor import apply_sensor

_G: dict = {}


def _init_worker(cfg_path: str):
    cv2.setNumThreads(1)  # avoid oversubscription across processes
    cfg = load_config(cfg_path)
    _G["cfg"] = cfg
    _G["lib"] = AssetLibrary(cfg.project.asset_root)


def _split_for(idx: int, total: int, splits: dict) -> str:
    # deterministic hash-interleaved split (judges: contiguous splits risk drift)
    u = ((idx * 2654435761) & 0xFFFFFFFF) / 2**32
    if u < splits["train"]:
        return "train"
    return "val" if u < splits["train"] + splits["val"] else "test"


def _generate_one(args: tuple[int, int, str]) -> dict:
    idx, total, out_dir = args
    cfg, lib = _G["cfg"], _G["lib"]
    rng = random.Random(int(cfg.project.seed) + idx)

    split = _split_for(idx, total, cfg.dataset.splits.raw())
    img_path = Path(out_dir) / "images" / split / f"{idx:06d}.jpg"
    lbl_path = Path(out_dir) / "labels" / split / f"{idx:06d}.txt"
    if img_path.exists() and lbl_path.exists():
        return {"idx": idx, "skipped": True}

    canvas, labels, meta = compose_frame(cfg, lib, idx)

    tiers = cfg.sensor.tier_weights.raw()
    tier_name = rng.choices(list(tiers), weights=list(tiers.values()), k=1)[0]
    tier = cfg.sensor[tier_name].raw()
    img, q = apply_sensor(canvas, rng, tier,
                          tuple(cfg.sensor.wb_shift),
                          tuple(cfg.sensor.exposure_ev),
                          tuple(cfg.sensor.unsharp_amount))

    img_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, q])
    lbl_path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                                for c, x, y, w, h in labels))

    meta.update({"idx": idx, "split": split, "tier": tier_name, "jpeg_q": q})
    return meta


def write_data_yaml(out_dir: Path, classes: list[str]) -> None:
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )


def render_previews(cfg_path: str, out_dir: Path, count: int) -> None:
    """Draw labeled previews for quick visual QA."""
    _init_worker(cfg_path)
    cfg = _G["cfg"]
    prev = out_dir / "previews"
    prev.mkdir(parents=True, exist_ok=True)
    colors = {0: (60, 60, 255), 1: (60, 220, 60)}
    import random as _rnd
    for split in ("train", "val", "test"):
        pool = sorted((out_dir / "images" / split).glob("*.jpg"))
        n = max(1, int(round(count * len(pool) / max(1, sum(
            len(list((out_dir / "images" / s).glob("*.jpg"))) for s in ("train", "val", "test"))))))
        imgs = _rnd.Random(0).sample(pool, min(n, len(pool)))
        for p in imgs:
            img = cv2.imread(str(p))
            lp = out_dir / "labels" / split / (p.stem + ".txt")
            if not lp.exists():
                continue
            H, W = img.shape[:2]
            for line in lp.read_text().splitlines():
                c, x, y, w, h = line.split()
                c = int(c)
                x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
                p1 = (int(x - w / 2), int(y - h / 2))
                p2 = (int(x + w / 2), int(y + h / 2))
                cv2.rectangle(img, p1, p2, colors.get(c, (255, 255, 0)), 2)
                cv2.putText(img, cfg.classes[c], (p1[0], p1[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(c, (255, 255, 0)), 1)
            cv2.imwrite(str(prev / f"{split}_{p.name}"), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--total", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--start", type=int, default=0, help="first image index")
    ap.add_argument("--preview-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    total = args.total or int(cfg.dataset.total_images)
    out_dir = Path(args.out or cfg.project.out_dir)
    workers = args.workers or int(cfg.project.workers)

    if args.preview_only:
        render_previews(args.config, out_dir, int(cfg.dataset.preview_count))
        print(f"previews -> {out_dir / 'previews'}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    write_data_yaml(out_dir, list(cfg.classes))

    # index once up front to fail fast on missing assets
    lib = AssetLibrary(cfg.project.asset_root)
    print("asset library:\n" + lib.summary(), flush=True)
    if not lib.backgrounds:
        sys.exit("FATAL: no backgrounds indexed")

    jobs = [(i, total, str(out_dir)) for i in range(args.start, total)]
    t0 = time.time()
    stats: dict = {"images": 0, "labels": 0, "per_class": {}, "tiers": {}, "buckets": {}}

    from multiprocessing import Pool
    with Pool(workers, initializer=_init_worker, initargs=(args.config,)) as pool:
        for i, meta in enumerate(pool.imap_unordered(_generate_one, jobs, chunksize=16)):
            if not meta.get("skipped"):
                stats["images"] += 1
                stats["labels"] += meta.get("n_labels", 0)
                stats["tiers"][meta.get("tier", "?")] = stats["tiers"].get(meta.get("tier", "?"), 0) + 1
                stats["buckets"][meta.get("bucket", "?")] = stats["buckets"].get(meta.get("bucket", "?"), 0) + 1
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(jobs) - i - 1) / max(rate, 1e-6) / 60
                print(f"{i + 1}/{len(jobs)}  {rate:.1f} img/s  ETA {eta:.0f} min", flush=True)

    stats["elapsed_s"] = round(time.time() - t0, 1)
    (out_dir / "gen_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
