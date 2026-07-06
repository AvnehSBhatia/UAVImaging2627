"""Asset library index for gen2.

Contracts (produced by the asset workflow):

backgrounds/{runway,grass,forest,dirt}/bg_*.jpg
    + meta.jsonl per bucket: {"file", "gsd_m", "scene_uuid", ...}

objects/{mannequin,tent}/<variant>/az{A:03d}_el{E:02d}.png        RGBA, 100 px/m
objects/{mannequin,tent}/<variant>/az{A:03d}_el{E:02d}_shadow.png grayscale, 255=full shadow
objects/{mannequin,tent}/<variant>/meta.json {"class","px_per_m","azimuths","elevations",...}

occluders/{vehicle,vegetation,debris}/*.png RGBA
    + meta.jsonl: {"file", "px_per_m", "kind", ...}
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

RENDER_PX_PER_M = 100.0


@dataclass
class Background:
    path: Path
    gsd_m: float
    bucket: str


@dataclass
class ObjectVariant:
    cls: str
    variant_dir: Path
    meta: dict
    azimuths: list[int]
    elevations: list[int]

    def nearest_az(self, az: float) -> int:
        az = az % 360.0
        return min(self.azimuths, key=lambda a: min(abs(a - az), 360 - abs(a - az)))

    def nearest_el(self, el: float) -> int:
        return min(self.elevations, key=lambda e: abs(e - el))

    def render_path(self, az: int, el: int) -> Path:
        return self.variant_dir / f"az{az:03d}_el{el:02d}.png"

    def shadow_path(self, az: int, el: int) -> Path:
        return self.variant_dir / f"az{az:03d}_el{el:02d}_shadow.png"


@dataclass
class Occluder:
    path: Path
    px_per_m: float
    kind: str
    otype: str  # vehicle | vegetation | debris

    # nominal height above ground in meters, used for synthesized shadows
    HEIGHTS = {"vehicle": 1.5, "vegetation": 1.1, "debris": 0.3}

    @property
    def height_m(self) -> float:
        return self.HEIGHTS.get(self.otype, 0.5)


@lru_cache(maxsize=256)
def _imread_rgba(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an RGBA png -> (bgr float32, alpha float32 0..1)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        a = img[:, :, 3].astype(np.float32) / 255.0
    else:
        bgr = (img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)).astype(np.float32)
        a = np.ones(bgr.shape[:2], np.float32)
    return bgr, a


@lru_cache(maxsize=256)
def _imread_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


class AssetLibrary:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.backgrounds: dict[str, list[Background]] = {}
        self.objects: dict[str, list[ObjectVariant]] = {}
        self.occluders: dict[str, list[Occluder]] = {}
        self._index()

    # ------------------------------------------------------------- indexing
    def _index(self) -> None:
        bg_root = self.root / "backgrounds"
        for bucket_dir in sorted(p for p in bg_root.iterdir() if p.is_dir()) if bg_root.exists() else []:
            bucket = bucket_dir.name
            gsd_by_file = {}
            meta = bucket_dir / "meta.jsonl"
            if meta.exists():
                for line in meta.read_text().splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    gsd_by_file[Path(rec["file"]).name] = float(rec["gsd_m"])
            items = []
            for p in sorted(bucket_dir.glob("*.jpg")):
                gsd = gsd_by_file.get(p.name) or self._gsd_from_name(p.name)
                if gsd:
                    items.append(Background(p, gsd, bucket))
            if items:
                self.backgrounds[bucket] = items

        obj_root = self.root / "objects"
        for cls_dir in sorted(p for p in obj_root.iterdir() if p.is_dir()) if obj_root.exists() else []:
            variants = []
            for vdir in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
                mpath = vdir / "meta.json"
                if not mpath.exists():
                    continue
                meta = json.loads(mpath.read_text())
                azs = sorted(int(a) for a in meta.get("azimuths", []))
                els = sorted(int(e) for e in meta.get("elevations", []))
                if not azs or not els:
                    continue
                v = ObjectVariant(cls_dir.name, vdir, meta, azs, els)
                if v.render_path(azs[0], els[0]).exists():
                    variants.append(v)
            if variants:
                self.objects[cls_dir.name] = variants

        occ_root = self.root / "occluders"
        for tdir in sorted(p for p in occ_root.iterdir() if p.is_dir()) if occ_root.exists() else []:
            otype = tdir.name
            recs = {}
            meta = tdir / "meta.jsonl"
            if meta.exists():
                for line in meta.read_text().splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    recs[Path(rec["file"]).name] = rec
            items = []
            for p in sorted(tdir.glob("*.png")):
                rec = recs.get(p.name, {})
                items.append(Occluder(p, float(rec.get("px_per_m", RENDER_PX_PER_M)),
                                      rec.get("kind", otype), otype))
            if items:
                self.occluders[otype] = items

    @staticmethod
    def _gsd_from_name(name: str) -> float | None:
        # bg_{bucket}_{gsd_mm*10:04d}mm_{seq}.jpg
        for part in name.split("_"):
            if part.endswith("mm") and part[:-2].isdigit():
                return int(part[:-2]) / 10.0 / 1000.0
        return None

    # -------------------------------------------------------------- access
    def pick_background(self, rng: random.Random, bucket: str) -> Background:
        pool = self.backgrounds.get(bucket)
        if not pool:  # bucket missing: fall back to any
            pool = [b for v in self.backgrounds.values() for b in v]
        return rng.choice(pool)

    def pick_variant(self, rng: random.Random, cls: str) -> ObjectVariant:
        return rng.choice(self.objects[cls])

    OCC_TYPE_WEIGHTS = {"vegetation": 0.55, "debris": 0.30, "vehicle": 0.15}

    def pick_occluder(self, rng: random.Random, otype: str | None = None) -> Occluder:
        if otype is None or otype not in self.occluders:
            types = [t for t in self.occluders]
            w = [self.OCC_TYPE_WEIGHTS.get(t, 0.2) for t in types]
            otype = rng.choices(types, weights=w, k=1)[0]
        return rng.choice(self.occluders[otype])

    @staticmethod
    def load_render(v: ObjectVariant, az: int, el: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        bgr, a = _imread_rgba(str(v.render_path(az, el)))
        sp = v.shadow_path(az, el)
        shadow = _imread_gray(str(sp)) if sp.exists() else None
        return bgr, a, shadow

    @staticmethod
    def load_cutout(o: Occluder) -> tuple[np.ndarray, np.ndarray]:
        return _imread_rgba(str(o.path))

    def summary(self) -> str:
        lines = []
        for b, items in self.backgrounds.items():
            lines.append(f"backgrounds/{b}: {len(items)}")
        for c, vs in self.objects.items():
            lines.append(f"objects/{c}: {len(vs)} variants")
        for t, os_ in self.occluders.items():
            lines.append(f"occluders/{t}: {len(os_)}")
        return "\n".join(lines)
