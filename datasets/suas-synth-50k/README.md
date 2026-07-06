# SUAS Synthetic Dataset (mannequin + tent, nadir 150 ft)

Built 2026-07-05. 22,482 images, YOLO format, classes `0=mannequin`, `1=tent`.
Train with: `yolo train data=datasets/suas-synth-50k/data.yaml ...`

## Composition

| split | total  | synthetic | VisDrone (`vd_*`) | mannequin boxes | tent boxes |
|-------|--------|-----------|-------------------|-----------------|------------|
| train | 19,185 | 13,501    | 5,684             | 118,467         | 11,126     |
| val   | 1,581  | 1,050     | 531               | 14,909          | 857        |
| test  | 1,716  | 449       | 1,267             | 27,805          | 369        |

- **Synthetic (15,000)**: photorealistic composites at 1920x1080, GSD 1.3–2.4 cm/px
  (SUAS 2026 mission floor: 150 ft AGL). Real aerial backgrounds (1,824
  CLIP-gated OpenAerialMap crops: runway/grass/forest/dirt), Blender-rendered
  clothed mannequins (60 pose variants: laying/sitting/face-down/kneeling, per
  rules) and pop-up tents (14 dome + 6 canopy) with sun-consistent baked
  shadows (8 azimuths x 4 elevations), deliberate partial occlusion by
  vegetation/debris (rules: "surrounded/covered"), Reinhard-harmonized
  blend-ensemble compositing, full sensor sim (linear-domain noise, Bayer
  round-trip, motion/defocus blur, vignette, JPEG QF 45-96) in hi-fi (45%) and
  grainy (55%) tiers. ~9.5% background-only frames. Labels reflect *visible*
  extent under occlusion; instances <25% visible are unlabeled.
- **VisDrone person subset (7,482)**: every VisDrone-DET image containing
  pedestrian/people, those boxes remapped to `mannequin`; all other VisDrone
  classes intentionally unlabeled (background for this task). Adds real aerial
  humans; note these are oblique urban scenes with small dense boxes — they
  dominate mannequin box counts (weigh/subsample if synthetic balance matters).

## Regenerate / extend

Generator: `datasetgen-2026/gen2` (deterministic per-index seeds, resume-safe).
- Extend to 50k: `.venv/bin/python -m gen2.run --workers 10` (from `datasetgen-2026/`)
- Assets: `datasets/gen-assets/` (backgrounds meta includes per-crop GSD + source scene)
- Preview labeled frames: `python -m gen2.run --preview-only` → `previews/`

## Known caveats

- Test split is VisDrone-heavy (1,267 vd vs 449 synthetic) — for synthetic-only
  eval, filter out `vd_*`.
- Runway backgrounds come from 4 distinct real airfields (OAM coverage limit),
  diversified by tiling/augmentation.
- Residual background-bucket impurity ~10% (measured by human QA sampling).
