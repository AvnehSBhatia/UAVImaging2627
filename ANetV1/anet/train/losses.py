import torch
import torch.nn.functional as F


def _cls_param(v, c):
    """Per-class alpha/beta: scalar applies to all classes; a (mannequin, tent)
    pair indexes by foreground class id (1->0, 2->1)."""
    if isinstance(v, (tuple, list)):
        return v[c - 1]
    return v


def focal_loss(logits, target, gamma=2.0, alpha=(1.0, 8.0, 4.0), mask=None):
    """logits (B,3,54,96), target (B,54,96) int in {0,1,2}.
    mask (B,54,96) bool, True = cell counts: used to drop boundary-band
    background cells (partial-coverage label noise) from the dense anchor."""
    logp = F.log_softmax(logits, 1)
    at = logits.new_tensor(alpha)[target]
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    fl = -at * (1.0 - pt) ** gamma * logpt
    if mask is None:
        return fl.mean()
    m = mask.float()
    return (fl * m).sum() / m.sum().clamp_min(1.0)


def _tversky_terms(p, target, c, band, dims):
    """Soft TP/FP/FN for foreground class c. band (B,2,54,96) bool marks
    partial-coverage cells per class (coverage in [band_lo, thresh)): they are
    labeled background but genuinely contain object — predicting class c there
    is not evidence of a false positive, so they leave the FP sum for c.
    Cells in ANOTHER class's band still count as FP for c (the mannequin ring
    around a tent boundary must stay punished as mannequin FP)."""
    pc = p[:, c]
    gc = (target == c).float()
    notg = 1.0 - gc
    if band is not None:
        notg = notg * (1.0 - band[:, c - 1].float())
    tp = (pc * gc).sum(dim=dims) if dims else (pc * gc).sum()
    fp = (pc * notg).sum(dim=dims) if dims else (pc * notg).sum()
    fn = ((1.0 - pc) * gc).sum(dim=dims) if dims else ((1.0 - pc) * gc).sum()
    return tp, fp, fn


def tversky_loss(logits, target, alpha=0.7, beta=0.3, classes=(1, 2),
                 smooth=1.0, band=None):
    """Soft Tversky over the foreground classes — the set-level FP/FN control
    focal lacks. Per class: TI = (TP+s) / (TP + alpha*FP + beta*FN + s);
    loss = 1 - TI. alpha>beta penalizes FALSE POSITIVES harder.

    smooth ~1 (one virtual TP cell), NOT an epsilon: with eps=1e-6 the index
    saturated at ~0 whenever a class was absent (TP=0), so the gradient wrt FP
    was ~1e-14 — the loss could not push down false positives on exactly the
    frames where they live (measured: the mannequin "objectness halo").
    """
    p = F.softmax(logits, 1)
    total = 0.0
    for c in classes:
        tp, fp, fn = _tversky_terms(p, target, c, band, None)
        a, b = _cls_param(alpha, c), _cls_param(beta, c)
        total = total + (1.0 - (tp + smooth) / (tp + a * fp + b * fn + smooth))
    return total / len(classes)


def focal_tversky_loss(logits, target, alpha=0.7, beta=0.3, gamma=0.75,
                       classes=(1, 2), per_image=True, smooth=1.0, band=None):
    """Focal-Tversky (Abraham & Khan 2019): FTL = (1 - TverskyIndex)**gamma per
    foreground class. ONE term that is simultaneously
      - size-invariant (set-level ratio, not per-cell -> small mannequins and big
        tents weigh equally; fixes the fewer-cells bias),
      - FP/FN-tunable (alpha>beta punishes false positives; per-class pairs
        supported — the tent blob was shrinking to ~half its GT cells under a
        global 0.8 while mannequin needed the full FP pressure),
      - hard-class-focused (the (1-TI)**gamma with gamma<1 amplifies gradient on a
        class that's doing badly, damps it once it's doing well).
    Because balance/precision/recall live in a SINGLE term, there is no focal-vs-
    Tversky tug-of-war — the source of the fp 0.3<->28 limit cycle.

    per_image=True computes the index per frame then averages, so one over-predicted
    frame can't dominate the batch statistic (steadier gradient than batch-pooled).

    smooth ~1 replaces the old eps=1e-6: on a frame without the class the index
    becomes s/(s + alpha*FP) — finite and steep in FP — instead of eps/(alpha*FP)
    ~ 0 with a vanishing (~1e-14, measured) gradient. This was the main defect:
    class-absent frames are where false positives live, and they got no push-down.

    band: per-class boundary-band ignore mask from the rasterizer (see
    _tversky_terms) — kills the loss's fight over partial-coverage cells.
    """
    p = F.softmax(logits, 1)
    dims = (1, 2) if per_image else None  # pc is (B,H,W): reduce over H,W per image
    total = 0.0
    for c in classes:
        tp, fp, fn = _tversky_terms(p, target, c, band, dims)
        a, b = _cls_param(alpha, c), _cls_param(beta, c)
        ti = (tp + smooth) / (tp + a * fp + b * fn + smooth)
        ftl = (1.0 - ti).clamp_min(0.0) ** gamma
        total = total + (ftl.mean() if per_image else ftl)
    return total / len(classes)


def distill_kl(logits, teacher_probs, temperature=2.0):
    """teacher_probs (B,3,54,96) from the cached soft grids."""
    t = temperature
    logp = F.log_softmax(logits / t, 1)
    soft = (teacher_probs.clamp_min(1e-9) ** (1.0 / t))
    soft = soft / soft.sum(1, keepdim=True)
    return F.kl_div(logp, soft, reduction="batchmean") * t * t
