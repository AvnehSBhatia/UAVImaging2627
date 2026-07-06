import torch
import torch.nn.functional as F


def focal_loss(logits, target, gamma=2.0, alpha=(1.0, 8.0, 4.0)):
    """logits (B,3,54,96), target (B,54,96) int in {0,1,2}."""
    logp = F.log_softmax(logits, 1)
    at = logits.new_tensor(alpha)[target]
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    return (-at * (1.0 - pt) ** gamma * logpt).mean()


def distill_kl(logits, teacher_probs, temperature=2.0):
    """teacher_probs (B,3,54,96) from the cached soft grids."""
    t = temperature
    logp = F.log_softmax(logits / t, 1)
    soft = (teacher_probs.clamp_min(1e-9) ** (1.0 / t))
    soft = soft / soft.sum(1, keepdim=True)
    return F.kl_div(logp, soft, reduction="batchmean") * t * t
