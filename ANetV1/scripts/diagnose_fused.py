"""Isolate why the fused Stage-1 demoted, on the training box.

Runs each layer independently with random input (no dataset needed) and prints
the FULL error for whichever one fails, so we don't debug through the MIOpen
log spam. Run on the MI300X box:

    /opt/venv/bin/python3 scripts/diagnose_fused.py 2>&1 | tee logs/diag.log

Paste the output. It answers, in order:
  1. is triton importable, what version, is this ROCm?
  2. does the fused FORWARD kernel compile + match the reference? (full TB if not)
  3. does the fused BACKWARD kernel compile + match? (full TB if not)
  4. eager vs torch.compile dense-path timing (is compile even helping?)
"""

import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet import ANetV1  # noqa: E402


def hr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72, flush=True)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hr("0. environment")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available(),
          "| hip:", getattr(torch.version, "hip", None))
    if dev.type == "cuda":
        print("device:", torch.cuda.get_device_name(0))
        free, total = torch.cuda.mem_get_info()
        print(f"vram: {free/2**30:.1f} GB free / {total/2**30:.1f} GB total")
    for k in ("MIOPEN_FIND_MODE", "ANET_FUSED", "ANET_FUSED_BWD", "ANET_COMPILE",
              "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        print(f"  {k}={os.environ.get(k)}")
    try:
        import triton
        print("triton:", triton.__version__, "| importable: YES")
    except Exception as e:
        print("triton import FAILED:", repr(e))

    from anet.train import fused as FU
    print("fused_available():", FU.fused_available())

    b = int(os.environ.get("ANET_BATCH", 8))
    model = ANetV1(arch="v9", stem="edge_dq4", hidden=32, h1=48,
                   use_checkpoint=False, aux_head=True, prior_fg=0.05).to(dev)
    img = torch.rand(b, 3, 540, 960, device=dev)
    print(f"model built, batch={b}, params="
          f"{sum(p.numel() for p in model.parameters()):,}")

    hr("1. fused FORWARD parity (compiles + runs the fwd kernel)")
    try:
        ok, delta = FU.parity_forward(model, img)
        print(f"RESULT: ok={ok} max_delta={delta:.3e}"
              + ("" if ok else "  <-- FORWARD KERNEL WRONG"))
    except Exception:
        print("FORWARD KERNEL RAISED:\n")
        traceback.print_exc()

    hr("2. fused BACKWARD parity (triton bwd vs chunked-autograd)")
    try:
        ok, rel = FU.parity_backward(model, img)
        print(f"RESULT: ok={ok} worst_rel={rel:.3e}"
              + ("" if ok else "  <-- BACKWARD KERNEL WRONG (would demote to chunked)"))
    except Exception:
        print("BACKWARD KERNEL RAISED:\n")
        traceback.print_exc()

    hr("3. dense-path timing (eager vs torch.compile)")
    model.train()

    def step(m):
        out = m(img)  # v9 training returns {cells, aux, z}
        cells, aux = out["cells"], out["aux"]
        (cells.square().mean() + 0.3 * aux.square().mean()).backward()
        m.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()

    try:
        for _ in range(2):
            step(model)                     # warm MIOpen
        t0 = time.time()
        for _ in range(5):
            step(model)
        eager = (time.time() - t0) / 5
        print(f"eager dense:    {eager*1000:.0f} ms/step ({eager/b*1000:.0f} ms/img)")
    except Exception:
        print("EAGER DENSE RAISED:\n")
        traceback.print_exc()
        eager = None

    try:
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        cm = torch.compile(model, mode="default")
        step(cm)                            # triggers compile (slow)
        step(cm)
        t0 = time.time()
        for _ in range(5):
            step(cm)
        comp = (time.time() - t0) / 5
        print(f"compiled dense: {comp*1000:.0f} ms/step ({comp/b*1000:.0f} ms/img)")
        if eager:
            print(f"compile speedup: {eager/comp:.1f}x")
    except Exception:
        print("torch.compile RAISED (this is why the run is eager-slow):\n")
        traceback.print_exc()

    hr("done — paste everything above")


if __name__ == "__main__":
    main()
