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


def balanced_tversky_loss(logits, target, alpha=(0.5, 0.5, 0.5),
                          beta=(0.5, 0.5, 0.5), gamma=0.75, smooth=1.0,
                          band=None, difficulty_temp=None, class_weights=None):
    """Class-balanced Focal-Tversky over ALL classes {bg, mannequin, tent}, per
    image — one unified term, so there is no focal-vs-Tversky anchor to fight
    (the fight that made the mannequin channel oscillate 0<->over-predict).

    Per class c, per image (softmax probs p_c, GT mask g_c):
        TP=Σ p_c·g_c   FP=Σ p_c·(1−g_c)   FN=Σ (1−p_c)·g_c
        TI = (TP+s) / (TP + α_c·FP + β_c·FN + s)     # α_c = FP-per-right-cell, β_c = miss
        L_c = (1 − TI)^γ
    Aggregate = mean_c L_c, averaged over images.

    Why every class weighs the same: each L_c is a BOUNDED [0,1] ratio that does
    not scale with the class's cell count, and they are averaged with equal
    weight. So a ~60-cell mannequin frame and a ~500k-cell background contribute
    equally — the size bias that pins rare classes is gone by construction, no
    per-class alpha juggling needed. Background is a real term here: bg-FP == a
    missed target, bg-FN == a hallucinated target, so a balanced bg score puts
    symmetric size-normalized pressure on both recall and precision.

    band (B, n_fg, H, W) bool: per-foreground-class partial-coverage cells that
    are labeled bg but genuinely contain object — dropped from that class's FP
    (predicting it there is not a real false positive) and from bg's FN.

    difficulty_temp: if set, up-weights the class currently doing worst via a
    DETACHED softmax over per-class losses (z-score-style "focus the loser"
    without the weight adding its own gradient dynamics — which would re-introduce
    an oscillation). None -> plain equal weight (the stable default).

    class_weights: FIXED per-class weights in index order (bg, mann, tent),
    e.g. (0.06, 0.6, 0.34) to hard-prioritize mannequin. Normalized to sum 1,
    so loss = Σ w_c·(1−TI_c)^γ. Takes precedence over difficulty_temp — a
    deliberate fixed prior on class importance instead of an adaptive one. Note
    each (1−TI_c) already folds recall (via FN) and FP-per-real-cell (via FP)
    into one bounded score, so this is exactly "w·(miss + fp)" per class.
    """
    p = F.softmax(logits, 1)                       # (B, C, H, W)
    C = p.shape[1]
    per_class = []
    for c in range(C):
        pc = p[:, c]
        gc = (target == c).float()
        notg = 1.0 - gc
        # a foreground class's band cells leave its FP; bg's band cells leave FN
        if band is not None:
            if c == 0:
                gc_eff = gc * (1.0 - band.any(1).float())  # bg not penalized for
                fn = ((1.0 - pc) * gc_eff).sum((1, 2))      # missing partial-object cells
                fp = (pc * notg).sum((1, 2))
                tp = (pc * gc).sum((1, 2))
            else:
                notg = notg * (1.0 - band[:, c - 1].float())
                tp = (pc * gc).sum((1, 2))
                fp = (pc * notg).sum((1, 2))
                fn = ((1.0 - pc) * gc).sum((1, 2))
        else:
            tp = (pc * gc).sum((1, 2))
            fp = (pc * notg).sum((1, 2))
            fn = ((1.0 - pc) * gc).sum((1, 2))
        a = _cls_param2(alpha, c)
        b = _cls_param2(beta, c)
        ti = (tp + smooth) / (tp + a * fp + b * fn + smooth)
        per_class.append((1.0 - ti).clamp_min(0.0) ** gamma)   # (B,)
    L = torch.stack(per_class, 1)                  # (B, C)
    if class_weights is not None:                  # FIXED prior on class importance
        w = L.new_tensor(list(class_weights))
        w = w / w.sum().clamp_min(1e-8)
        return (L.mean(0) * w).sum()
    if difficulty_temp:
        w = torch.softmax(L.detach().mean(0) / difficulty_temp, 0)  # (C,), detached
        return (L.mean(0) * w).sum()               # sums to a class-weighted mean
    return L.mean()


def _cls_param2(v, c):
    """alpha/beta indexing for balanced loss: full (bg, mann, tent) triple, a
    (mann, tent) pair applied to the two fg classes (bg gets 0.5), or a scalar."""
    if isinstance(v, (tuple, list)):
        if len(v) == 3:
            return v[c]
        return 0.5 if c == 0 else v[c - 1]
    return v


def focal_norm_loss(logits, target, gamma=2.0, class_weights=(1.0, 2.0, 1.0),
                    min_pos=1.0, min_pos_bg=8.0, mask=None):
    """v9 default (D47): per-class positive-normalized focal loss — ONE smooth
    per-cell term that is simultaneously size-invariant and oscillation-free.

    The Focal-Tversky stack failed two ways: the set-level ratio makes the
    gradient per cell depend nonlinearly on the batch's TP/FP totals (spiky,
    and the documented source of the fp 0.3<->28 and mannequin 0<->overshoot
    limit cycles), and pairing it with a focal anchor created a two-term
    tug-of-war. This loss keeps focal's dense smooth gradient and gets size
    invariance the CenterNet/FCOS way — normalize each class's summed focal
    term by that class's positive-cell count in the BATCH:

        L = sum_c w_c * [ sum_{cells: t=c} FL(cell) ] / max(N_c, min_pos)
          + w_bg    * [ sum_{t=0, mask}  FL(cell) ] / max(N_fg_total, min_pos)

    Every positive cell of a rare class carries O(1) gradient no matter how
    few there are (a 60-cell mannequin and a 500-cell tent weigh the same per
    class), and the background push scales with how much foreground actually
    exists — at prior-bias init the foreground pull dominates, so recall
    rises first and precision pressure grows as predictions appear.
    Batch-level counts (not per-image) keep the normalizer steady.

    Floors are asymmetric on purpose: min_pos=1 keeps a 1-4-cell mannequin
    (exactly the worst-decile regime the decision metric targets) at full
    per-class pull — flooring foreground at 8 measurably inverted the fg/bg
    balance whenever a small batch's only foreground was one tiny object.
    min_pos_bg=8 still bounds the background push in all-background batches
    (n_fg=0). mask drops boundary-band background cells."""
    assert len(class_weights) == logits.shape[1], \
        f"class_weights must have {logits.shape[1]} entries (bg, mann, tent)"
    logp = F.log_softmax(logits, 1)
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    fl = -((1.0 - logpt.exp()) ** gamma) * logpt  # (B, H, W)
    total = logits.new_zeros(())
    n_fg = (target > 0).float().sum()
    for c, w in enumerate(class_weights):
        m = target == c
        if c == 0:
            if mask is not None:
                m = m & mask
            total = total + w * (fl * m.float()).sum() / \
                torch.clamp(n_fg, min=min_pos_bg)
        else:
            n_c = m.float().sum()
            total = total + w * (fl * m.float()).sum() / torch.clamp(n_c, min=min_pos)
    return total


def distill_kl(logits, teacher_probs, temperature=2.0):
    """teacher_probs (B,3,54,96) from the cached soft grids."""
    t = temperature
    logp = F.log_softmax(logits / t, 1)
    soft = (teacher_probs.clamp_min(1e-9) ** (1.0 / t))
    soft = soft / soft.sum(1, keepdim=True)
    return F.kl_div(logp, soft, reduction="batchmean") * t * t
