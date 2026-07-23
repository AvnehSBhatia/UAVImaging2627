"""Turn HERIDAL (real wilderness aerial persons) into mission-scale ANet training
frames — the D93.1 fix: the model's features do not encode REAL person appearance
(texture, foreshortening, real limbs/shadow, vegetation occlusion), because gen2
only ever showed it rendered mannequins. HERIDAL is real people, near-nadir, at
40-65 m over Mediterranean wilderness — the mission geometry and terrain.

WHY tiles, not whole frames. HERIDAL frames are 4000x3000 with persons ~71 px
median (GSD ~2.4 cm/px, the coarse end of the mission's 1.3-2.4). Downscaling a
whole frame to 960x540 shrinks a person to ~17 px (sub-decile, VisDrone-like).
Instead this crops a mission-scale WINDOW around each person and resizes THAT to
960x540, so the person lands at a sampled mission size (~28-90 px) with real
wilderness background context — no compositing, no synthetic anything.

Output: `hd_*.jpg` + YOLO `hd_*.txt` (class 0 = mannequin/person) into the
dataset train split, mirroring the `vd_` convention so `SUASCells` picks them up
and `hd_weight` can down/upweight them. Deterministic per (image, person, scale)
so a fresh checkout on the MI300X box regenerates byte-identical tiles — the
dataset is gitignored, so this SCRIPT (committed) is how HERIDAL reaches training.

License: HERIDAL is CC BY-3.0 (IPSAR / University of Split). Attribution belongs
in any release that trains on these tiles.

  python scripts/prepare_heridal.py --test       # 24 tiles + preview, no dataset write
  python scripts/prepare_heridal.py              # full generation into the train split
  python scripts/prepare_heridal.py --crops-per-person 2
"""
import argparse
import glob
import hashlib
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
HERIDAL = REPO / "datasets" / "heridal"
VOC = HERIDAL / "extracted" / "heridal_keras_retinanet_voc"
DATASET = REPO / "datasets" / "suas-synth-50k"
ZENODO = ("https://zenodo.org/records/5662351/files/"
          "heridal_keras_retinanet_voc.zip?download=1")
OUT_W, OUT_H = 960, 540
ASPECT = OUT_H / OUT_W                      # 0.5625
SCALE_PX = (28, 90)                         # sampled output person longest-side
JITTER = 0.30                               # crop-centre offset as frac of window


def ensure_heridal():
    """Download + extract HERIDAL from Zenodo if the VOC tree is not present."""
    if (VOC / "Annotations").is_dir() and (VOC / "JPEGImages").is_dir():
        return
    HERIDAL.mkdir(parents=True, exist_ok=True)
    zp = HERIDAL / "heridal_voc.zip"
    if not zp.exists():
        print(f"[heridal] downloading 8.3 GB from Zenodo -> {zp}", flush=True)
        subprocess.run(["curl", "-sL", "-o", str(zp), ZENODO], check=True)
    print("[heridal] extracting...", flush=True)
    subprocess.run(["unzip", "-q", "-o", str(zp), "-d", str(HERIDAL / "extracted")],
                   check=True)
    assert (VOC / "Annotations").is_dir(), "extraction did not yield the VOC tree"


def _persons(xml_path):
    """Return [(cx, cy, w, h)] person boxes in native pixels. Dimensions come
    from the image itself in the caller — the XML <size> element is missing on
    some HERIDAL annotations, and the box coords are absolute either way."""
    t = ET.parse(xml_path).getroot()
    out = []
    for o in t.findall("object"):
        if o.findtext("name") != "person":
            continue
        b = o.find("bndbox")
        if b is None:
            continue
        try:
            x0, x1 = int(float(b.findtext("xmin"))), int(float(b.findtext("xmax")))
            y0, y1 = int(float(b.findtext("ymin"))), int(float(b.findtext("ymax")))
        except (TypeError, ValueError):
            continue
        if x1 > x0 and y1 > y0:
            out.append(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return out


def _rng(*keys):
    """Deterministic RNG from string keys — same tiles on every machine."""
    h = hashlib.sha1("|".join(map(str, keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def make_tile(img, persons, target, W, H, rng):
    """Crop a mission-scale window around person `target`, resize to 960x540.

    Returns (tile_uint8 HxWx3, [(cls,cx,cy,w,h) normalized]) or None if degenerate.
    """
    pcx, pcy, pw, ph = persons[target]
    L = max(pw, ph)
    s = rng.uniform(*SCALE_PX)                       # desired output person px
    win_w = float(np.clip(L * OUT_W / s, 200, W))    # native window width
    win_h = win_w * ASPECT
    if win_h > H:
        win_h = float(H); win_w = win_h / ASPECT
    # centre on the person, jittered so it is not always centred
    cx = pcx + rng.uniform(-JITTER, JITTER) * win_w
    cy = pcy + rng.uniform(-JITTER, JITTER) * win_h
    x0 = float(np.clip(cx - win_w / 2, 0, W - win_w))
    y0 = float(np.clip(cy - win_h / 2, 0, H - win_h))
    crop = img.crop((round(x0), round(y0), round(x0 + win_w), round(y0 + win_h)))
    crop = crop.resize((OUT_W, OUT_H), Image.LANCZOS)
    sx, sy = OUT_W / win_w, OUT_H / win_h
    labels = []
    for qx, qy, qw, qh in persons:                   # every person inside the window
        nx, ny = (qx - x0) * sx, (qy - y0) * sy
        if 0 <= nx <= OUT_W and 0 <= ny <= OUT_H:
            labels.append((0, nx / OUT_W, ny / OUT_H,
                           min(qw * sx / OUT_W, 1.0), min(qh * sy / OUT_H, 1.0)))
    if not labels:
        return None
    return np.asarray(crop, np.uint8), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="write 24 tiles + a preview montage to a tmp dir, no dataset change")
    ap.add_argument("--crops-per-person", type=int, default=1)
    ap.add_argument("--split", default="train", help="dataset split to write into")
    ap.add_argument("--limit", type=int, default=0, help="cap source images (0=all)")
    args = ap.parse_args()

    ensure_heridal()
    xmls = sorted(p for p in glob.glob(f"{VOC}/Annotations/*.xml") if "__MACOSX" not in p)
    if args.limit:
        xmls = xmls[:args.limit]

    if args.test:
        out = REPO / "ANetV1" / "runs" / "heridal_test"
        out.mkdir(parents=True, exist_ok=True)
        montage, made = [], 0
        for xp in xmls:
            stem = Path(xp).stem
            ip = VOC / "JPEGImages" / f"{stem}.jpg"
            if not ip.exists():
                continue
            persons = _persons(xp)
            if not persons:
                continue
            img = Image.open(ip).convert("RGB")
            W, H = img.size
            r = make_tile(img, persons, 0, W, H, _rng(stem, 0, 0))
            if r is None:
                continue
            tile, labels = r
            vis = tile.copy()
            for _, cx, cy, w, h in labels:            # draw boxes for the preview
                x0, y0 = int((cx-w/2)*OUT_W), int((cy-h/2)*OUT_H)
                x1, y1 = int((cx+w/2)*OUT_W), int((cy+h/2)*OUT_H)
                vis[max(0,y0):y0+2, x0:x1] = [0,255,0]; vis[y1:y1+2, x0:x1] = [0,255,0]
                vis[y0:y1, max(0,x0):x0+2] = [0,255,0]; vis[y0:y1, x1:x1+2] = [0,255,0]
                montage.append(np.asarray(Image.fromarray(vis).resize((320,180))))
            made += 1
            if made >= 24:
                break
        cols = 4
        rows_ = (len(montage) + cols - 1) // cols
        sheet = np.full((rows_*180, cols*320, 3), 20, np.uint8)
        for i, m in enumerate(montage):
            sheet[(i//cols)*180:(i//cols)*180+180, (i%cols)*320:(i%cols)*320+320] = m
        Image.fromarray(sheet).save(out / "preview.png")
        print(f"[test] {made} tiles, preview -> {out/'preview.png'}")
        return

    img_dir = DATASET / "images" / args.split
    lbl_dir = DATASET / "labels" / args.split
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    # clean any prior hd_ tiles so re-runs are idempotent, not additive
    for d, ext in ((img_dir, "jpg"), (lbl_dir, "txt")):
        for f in d.glob(f"hd_*.{ext}"):
            f.unlink()
    n_tiles = n_persons = 0
    for xi, xp in enumerate(xmls):
        stem = Path(xp).stem
        ip = VOC / "JPEGImages" / f"{stem}.jpg"
        if not ip.exists():
            continue
        persons = _persons(xp)
        if not persons:
            continue
        img = Image.open(ip).convert("RGB")
        W, H = img.size
        for pi in range(len(persons)):
            n_persons += 1
            for k in range(args.crops_per_person):
                r = make_tile(img, persons, pi, W, H, _rng(stem, pi, k))
                if r is None:
                    continue
                tile, labels = r
                name = f"hd_{stem}_{pi}_{k}"
                Image.fromarray(tile).save(img_dir / f"{name}.jpg", quality=92)
                (lbl_dir / f"{name}.txt").write_text(
                    "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                            for c, cx, cy, w, h in labels))
                n_tiles += 1
        if (xi + 1) % 300 == 0:
            print(f"[heridal] {xi+1}/{len(xmls)} imgs, {n_tiles} tiles", flush=True)
    print(f"[heridal] DONE: {n_tiles} hd_ tiles from {n_persons} person annotations "
          f"-> {img_dir}\n  NOTE: delete the .anet_cache so SUASCells rebuilds with "
          f"the new items; set ANET_HD_WEIGHT to down/upweight (mirrors vd_weight).")


if __name__ == "__main__":
    main()
