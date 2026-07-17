"""P2: fixed+learned 5-filter bank -> tiny window scorer (OBSERVATIONS.md P2).

Owner spec: duplicate the NxN patch into 5 images —
  2 fixed Sobel filters, 1 learned dual-quaternion transform,
  1 fixed 10x10 Gaussian blur, 1 learned 10x10 texture-removal kernel —
then a quick per-pixel embedding transform over the stack, scan with a
10x10 kernel outputting per-window class logits, and sum the windows into
the patch's class probability.

The window scorer is stride-10 and the sum is position-blind, so the model
can only vote on local 10x10 evidence — it probes whether hand-chosen
filter diversity + a ~5k-param scorer matches what ANet's conv blocks
learn end-to-end. Patch sides must be multiples of 10 (40 and 100 are).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.blocks import DualQuaternionRGB


def _gauss10(sigma=3.0):
    a = torch.arange(10, dtype=torch.float32) - 4.5
    g = torch.exp(-a ** 2 / (2 * sigma * sigma))
    k = g[:, None] * g[None, :]
    return (k / k.sum()).reshape(1, 1, 10, 10)


class FiveStack(nn.Module):
    def __init__(self, embed=16):
        super().__init__()
        self.dq = DualQuaternionRGB()
        sob = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4
        self.register_buffer("sobel", torch.stack([sob, sob.t()]).unsqueeze(1))
        self.register_buffer("gauss", _gauss10())
        self.register_buffer("luma",
                             torch.tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1))
        self.texture = nn.Conv2d(1, 1, 10, bias=False)
        with torch.no_grad():
            self.texture.weight.copy_(_gauss10())  # blur init = texture killer
        self.embed = nn.Conv2d(5, embed, 1)
        self.window = nn.Conv2d(embed, 3, 10, stride=10)

    def forward(self, x):  # (B,3,S,S), S % 10 == 0 -> (B,3) patch logits
        gray = (x * self.luma).sum(1, keepdim=True)
        e = F.conv2d(gray, self.sobel, padding=1)          # 2 Sobel images
        d = (self.dq(x) * self.luma).sum(1, keepdim=True)  # learned DQ image
        pad = F.pad(gray, (4, 5, 4, 5))                    # even kernel: 4/5
        g = F.conv2d(pad, self.gauss)                      # fixed 10x10 blur
        t = self.texture(pad)                              # learned 10x10
        z = F.silu(self.embed(torch.cat([e, d, g, t], 1)))
        # window-count-normalized sum: a 100x100 patch has 100 windows vs a
        # 40x40's 16 — a raw sum gives the two sizes different logit
        # temperatures under ONE shared softmax (measured: init CE 41 vs 1.1)
        return self.window(z).mean((2, 3))
