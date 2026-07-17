"""Side probes — standalone micro-models, separate from ANetV1 proper.

P1 WhiteboxDQ and P2 FiveStack live here (results ledger: OBSERVATIONS.md).
They share the repo's train protocol (env knobs, device pick, best.pt
selection) but nothing of ANet's trunk — that separation is the point:
each probe isolates one hypothesis about what the conv blocks are or
aren't doing.
"""

from .fivestack import FiveStack
from .patches import PatchCrops, collate_patches
from .whitebox import WhiteboxDQ

__all__ = ["FiveStack", "PatchCrops", "WhiteboxDQ", "collate_patches"]
