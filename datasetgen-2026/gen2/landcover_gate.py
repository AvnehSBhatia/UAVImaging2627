"""CLIP linear-probe land-cover gate for background crops.

Trained on 304 hand-labeled OAM crops (2026-07-05). Classes:
buildings, clutter, dirt, farmland, forest, grass, pavement, water.

Acceptance rule (empirically tuned on CV):
  top-1 in {grass, forest, dirt}  AND  margin >= 0.05  AND  terrain_mass >= 0.55

Usage:
  from gen2.landcover_gate import LandcoverGate
  gate = LandcoverGate(probe_path)
  results = gate.classify_files([...paths...])   # -> list of dicts
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TERRAIN = ("grass", "forest", "dirt")
DEFAULT_PROBE = Path(__file__).parent / "landcover_probe.npz"


class LandcoverGate:
    def __init__(self, probe_path: str | Path = DEFAULT_PROBE, device: str | None = None):
        import torch
        import open_clip

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k"
        )
        self.model = self.model.to(self.device).eval()
        probe = np.load(probe_path, allow_pickle=True)
        self.coef = probe["coef"]
        self.intercept = probe["intercept"]
        self.classes = [str(c) for c in probe["classes"]]
        self.tidx = [self.classes.index(c) for c in TERRAIN]

    def embed_pil(self, images) -> np.ndarray:
        with self.torch.no_grad():
            batch = self.torch.stack([self.preprocess(im) for im in images]).to(self.device)
            f = self.model.encode_image(batch)
            f /= f.norm(dim=-1, keepdim=True)
        return f.cpu().float().numpy()

    def classify_features(self, X: np.ndarray) -> list[dict]:
        logits = X @ self.coef.T + self.intercept
        e = np.exp(logits - logits.max(1, keepdims=True))
        proba = e / e.sum(1, keepdims=True)
        out = []
        for p in proba:
            order = np.argsort(p)
            top1 = self.classes[order[-1]]
            margin = float(p[order[-1]] - p[order[-2]])
            terrain_mass = float(p[self.tidx].sum())
            accept = top1 in TERRAIN and margin >= 0.05 and terrain_mass >= 0.55
            out.append({"bucket": top1, "margin": margin,
                        "terrain_mass": terrain_mass, "accept": bool(accept),
                        "proba": {c: float(v) for c, v in zip(self.classes, p)}})
        return out

    def classify_files(self, paths: list, batch: int = 24) -> list[dict]:
        from PIL import Image

        out = []
        for i in range(0, len(paths), batch):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            out += self.classify_features(self.embed_pil(ims))
        return out
