"""CLI for the hyper-accurate 3k bg + 3k single-object dataset.

Usage (cwd = datasetgen-2026/):
  ../.venv/bin/python -m gen_hyper.run
  ../.venv/bin/python -m gen_hyper.run --smoke          # 8 frames, quick QA
  ../.venv/bin/python -m gen_hyper.run --workers 8
  ../.venv/bin/python -m gen_hyper.run --out /data/suas-hyper-6k

Index layout (deterministic, resume-safe):
  0 .. n_bg-1              background-only (empty labels)
  n_bg .. n_bg+n_single-1  single object (mannequin XOR tent)

Train with:
  cd ANetV1 && DATA_ROOT=../datasets/suas-hyper-6k python scripts/train_anet.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from gen2.config import load_config
from gen2.sensor import apply_sensor
from gen2.assets import AssetLibrary

from .compose import attach_web_backgrounds, compose_hyper

_G: dict = {}


def _init_worker(cfg_path: str):
    cv2.setNumThreads(1)
    cfg = load_config(cfg_path)
    lib = AssetLibrary(cfg.project.asset_root)
    web = getattr(cfg.project, "web_bg_dir", None)
    if web:
        attach_web_backgrounds(lib, web)
    _G["cfg"] = cfg
    _G["lib"] = lib


def _split_for(idx: int, splits: dict) -> str:
    u = ((idx * 2654435761) & 0xFFFFFFFF) / 2**32
    if u < splits["train"]:
        return "train"
    return "val" if u < splits["train"] + splits["val"] else "test"


def _plan(cfg) -> list[tuple[int, str, str | None]]:
    """Return list of (idx, mode, cls)."""
    n_bg = int(cfg.dataset.n_background)
    n_single = int(cfg.dataset.n_single)
    frac = float(cfg.dataset.mannequin_fraction)
    n_mann = int(round(n_single * frac))
    n_tent = n_single - n_mann
    plan: list[tuple[int, str, str | None]] = []
    for i in range(n_bg):
        plan.append((i, "bg", None))
    # shuffle class assignment with fixed seed so resume stays stable
    classes = ["mannequin"] * n_mann + ["tent"] * n_tent
    random.Random(int(cfg.project.seed) + 17).shuffle(classes)
    for j, cls in enumerate(classes):
        plan.append((n_bg + j, "single", cls))
    return plan


def _generate_one(args: tuple) -> dict:
    idx, mode, cls, out_dir = args
    cfg, lib = _G["cfg"], _G["lib"]
    rng = random.Random(int(cfg.project.seed) + idx)

    split = _split_for(idx, cfg.dataset.splits.raw())
    img_path = Path(out_dir) / "images" / split / f"{idx:06d}.jpg"
    lbl_path = Path(out_dir) / "labels" / split / f"{idx:06d}.txt"
    if img_path.exists() and lbl_path.exists():
        return {"idx": idx, "skipped": True}

    canvas, labels, meta = compose_hyper(cfg, lib, idx, mode, cls)

    # contract: single-object frames that lost the label become bg-only
    if mode == "single" and not labels:
        meta["demoted_to_bg"] = True

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
    lbl_path.write_text("".join(
        f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels))

    meta.update({"idx": idx, "split": split, "tier": tier_name, "jpeg_q": q,
                 "skipped": False})
    return meta


def write_data_yaml(out_dir: Path, classes: list[str]) -> None:
    (out_dir / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )


def write_readme(out_dir: Path, cfg) -> None:
    (out_dir / "README.md").write_text(
        "# suas-hyper-6k\n\n"
        "Hyper-accurate SUAS set from `datasetgen-2026/gen_hyper`.\n\n"
        f"- background-only: {cfg.dataset.n_background}\n"
        f"- single-object: {cfg.dataset.n_single} "
        f"(mannequin_fraction={cfg.dataset.mannequin_fraction})\n"
        "- scenarios: open_field, runway_drygrass, tree_clearing, brush_occlusion\n"
        "- classes: 0=mannequin, 1=tent\n"
        "- canvas: 1920x1080 YOLO labels\n\n"
        "Train:\n"
        "```bash\n"
        "cd ANetV1\n"
        "DATA_ROOT=../datasets/suas-hyper-6k python scripts/train_anet.py\n"
        "```\n"
    )


def render_previews(cfg_path: str, out_dir: Path, count: int) -> None:
    _init_worker(cfg_path)
    prev = out_dir / "previews"
    prev.mkdir(parents=True, exist_ok=True)
    colors = {0: (60, 60, 255), 1: (60, 220, 60)}
    pool = []
    for split in ("train", "val", "test"):
        pool += sorted((out_dir / "images" / split).glob("*.jpg"))
    if not pool:
        return
    sample = random.Random(0).sample(pool, min(count, len(pool)))
    for p in sample:
        img = cv2.imread(str(p))
        lp = out_dir / "labels" / p.parent.name / (p.stem + ".txt")
        if lp.exists():
            for line in lp.read_text().splitlines():
                c, cx, cy, w, h = line.split()
                c = int(c)
                H, W = img.shape[:2]
                x0 = int((float(cx) - float(w) / 2) * W)
                y0 = int((float(cy) - float(h) / 2) * H)
                x1 = int((float(cx) + float(w) / 2) * W)
                y1 = int((float(cy) + float(h) / 2) * H)
                cv2.rectangle(img, (x0, y0), (x1, y1), colors.get(c, (255, 255, 0)), 2)
                cv2.putText(img, ("M" if c == 0 else "T"), (x0, max(12, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(c, (255, 255, 0)), 1)
        cv2.imwrite(str(prev / f"{p.parent.name}_{p.name}"), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="gen_hyper/config.yaml")
    ap.add_argument("--out", default=None, help="override project.out_dir")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="8 frames (4 bg + 2 mann + 2 tent) for QA")
    ap.add_argument("--preview-only", action="store_true")
    args = ap.parse_args()

    cfg_path = str(Path(args.config).resolve())
    cfg = load_config(cfg_path)
    out_dir = Path(args.out or cfg.project.out_dir)
    workers = int(args.workers or cfg.project.workers)

    if args.smoke:
        # monkeypatch counts via a temp override by editing plan manually
        cfg._d["dataset"]["n_background"] = 4
        cfg._d["dataset"]["n_single"] = 4
        cfg._d["dataset"]["mannequin_fraction"] = 0.5
        out_dir = out_dir.parent / (out_dir.name + "-smoke")
        workers = min(workers, 2)

    if args.preview_only:
        render_previews(cfg_path, out_dir, int(cfg.dataset.preview_count))
        print(f"previews → {out_dir / 'previews'}")
        return

    plan = _plan(cfg)
    print(f"gen_hyper → {out_dir}")
    print(f"  plan: {sum(1 for _, m, _ in plan if m == 'bg')} bg + "
          f"{sum(1 for _, m, _ in plan if m == 'single')} single "
          f"(workers={workers})")
    lib = AssetLibrary(cfg.project.asset_root)
    attach_web_backgrounds(lib, getattr(cfg.project, "web_bg_dir", ""))
    print(lib.summary())
    if "web" in lib.backgrounds:
        print(f"  web backgrounds: {len(lib.backgrounds['web'])}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_data_yaml(out_dir, list(cfg._d["classes"]))
    write_readme(out_dir, cfg)

    jobs = [(idx, mode, cls, str(out_dir)) for idx, mode, cls in plan]
    t0 = time.time()
    done = skipped = demoted = 0
    stats = {"scenario": {}, "cls": {}, "bucket": {}}

    # re-dump cfg to a resolved path workers can read; for smoke we need the
    # mutated counts — write a sidecar yaml
    worker_cfg = out_dir / "_run_config.yaml"
    import yaml
    yaml.safe_dump(cfg._d, worker_cfg.open("w"))

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker,
                             initargs=(str(worker_cfg),)) as ex:
        futs = [ex.submit(_generate_one, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            meta = fut.result()
            if meta.get("skipped"):
                skipped += 1
            else:
                done += 1
                if meta.get("demoted_to_bg"):
                    demoted += 1
                sc = meta.get("scenario", "?")
                stats["scenario"][sc] = stats["scenario"].get(sc, 0) + 1
                if meta.get("cls"):
                    stats["cls"][meta["cls"]] = stats["cls"].get(meta["cls"], 0) + 1
                stats["bucket"][meta.get("bucket", "?")] = (
                    stats["bucket"].get(meta.get("bucket", "?"), 0) + 1)
            if k % 100 == 0 or k == len(futs):
                rate = k / max(time.time() - t0, 1e-6)
                print(f"  {k}/{len(futs)}  ({rate:.1f} img/s)  "
                      f"new={done} skip={skipped} demoted={demoted}",
                      flush=True)

    (out_dir / "gen_stats.json").write_text(json.dumps({
        "done": done, "skipped": skipped, "demoted_to_bg": demoted,
        "elapsed_s": time.time() - t0, **stats,
    }, indent=2))
    render_previews(str(worker_cfg), out_dir, int(cfg.dataset.preview_count))
    print(f"done → {out_dir}  (previews in previews/)")
    print(f"train: DATA_ROOT={out_dir} python scripts/train_anet.py")


if __name__ == "__main__":
    main()
