import torch
import torch.nn.functional as F


def focal_loss(logits, target, gamma=2.0, alpha=(1.0, 8.0, 4.0)):
    """logits (B,3,54,96), target (B,54,96) int in {0,1,2}."""
    logp = F.log_softmax(logits, 1)
    at = logits.new_tensor(alpha)[target]
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    return (-at * (1.0 - pt) ** gamma * logpt).mean()


def tversky_loss(logits, target, alpha=0.7, beta=0.3, classes=(1, 2), eps=1e-6):
    """Soft Tversky over the foreground classes — the set-level FP/FN control
    focal lacks. Per class: TI = TP / (TP + alpha*FP + beta*FN); loss = 1 - TI.
    alpha>beta penalizes FALSE POSITIVES harder (precision-leaning) — the direct
    knob for the fp/img creep. TP/FP/FN are soft (probabilities), summed over the
    whole batch of cells so diffuse over-prediction on background is penalized.
    """
    p = F.softmax(logits, 1)
    total = 0.0
    for c in classes:
        pc = p[:, c]
        gc = (target == c).float()
        tp = (pc * gc).sum()
        fp = (pc * (1.0 - gc)).sum()
        fn = ((1.0 - pc) * gc).sum()
        total = total + (1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps))
    return total / len(classes)


def distill_kl(logits, teacher_probs, temperature=2.0):
    """teacher_probs (B,3,54,96) from the cached soft grids."""
    t = temperature
    logp = F.log_softmax(logits / t, 1)
    soft = (teacher_probs.clamp_min(1e-9) ** (1.0 / t))
    soft = soft / soft.sum(1, keepdim=True)
    return F.kl_div(logp, soft, reduction="batchmean") * t * t
