"""Export a trained ANetV1 checkpoint to a self-contained ONNX for fast local
inference (ONNX Runtime CoreML/CPU), verify parity vs PyTorch, optionally bench.

The torch dynamo exporter writes weights to a sidecar `.onnx.data` file; the ORT
CoreML partitioner fails to initialize from it ("model_path must not be empty"),
so the weights are re-inlined into one .onnx and the sidecar removed.

  python scripts/export_onnx.py --ckpt runs/anet/best.pt              # -> runs/anet/anet.onnx
  python scripts/export_onnx.py --ckpt runs/anet/best.pt --bench
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anet import ANetV1  # noqa: E402
from anet.onnxrt import OnnxANet, default_providers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/anet/best.pt")
    ap.add_argument("--out", default=None, help="default: <ckpt dir>/anet.onnx")
    ap.add_argument("--batch", type=int, default=1,
                    help="fixed export batch (1 measured fastest on CoreML/MPS)")
    ap.add_argument("--bench", action="store_true", help="time ORT vs eager PyTorch")
    args = ap.parse_args()
    out = Path(args.out or Path(args.ckpt).parent / "anet.onnx")

    sd = torch.load(args.ckpt, map_location="cpu")
    model = ANetV1.from_state_dict(sd, use_checkpoint=False).eval()
    print(f"ckpt {args.ckpt} | hidden={model.encoder.hidden} stem={model.stem} | "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    model.export_onnx(str(out), batch=args.batch)

    # inline the sidecar weights -> single self-contained file (CoreML EP requirement)
    import onnx

    onnx.save(onnx.load(str(out)), str(out))
    sidecar = out.with_suffix(out.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB, self-contained)")

    # parity vs eager PyTorch
    ort_model = OnnxANet(out)
    print(f"providers: {ort_model.sess.get_providers()}")
    x = torch.rand(args.batch, 3, 540, 960)
    with torch.no_grad():
        ref = model(x).numpy()
    got = ort_model(x)
    diff = float(np.abs(ref - got).max())
    agree = float((ref.argmax(1) == got.argmax(1)).mean())
    print(f"parity: max|Δlogit|={diff:.2e} argmax agree={agree:.6f}")
    assert diff < 1e-3 and agree > 0.9999, "ONNX output diverges from PyTorch"

    if args.bench:
        def timed(fn, iters=40, warmup=8):
            for _ in range(warmup):
                fn()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            return (time.perf_counter() - t0) / iters / args.batch

        xn = x.numpy()
        dt = timed(lambda: ort_model.sess.run(None, {"frame": xn}))
        print(f"  ORT {ort_model.sess.get_providers()[0]:28s} "
              f"{dt * 1e3:7.1f} ms/img | {1 / dt:6.1f} img/s | {dt * 1000:6.1f} s per 1k")
        if len(default_providers()) > 1:  # CPU EP reference when CoreML is primary
            cpu = OnnxANet(out, providers=["CPUExecutionProvider"])
            dt = timed(lambda: cpu.sess.run(None, {"frame": xn}))
            print(f"  ORT CPUExecutionProvider         "
                  f"{dt * 1e3:7.1f} ms/img | {1 / dt:6.1f} img/s | {dt * 1000:6.1f} s per 1k")
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        m = model.to(dev)
        xd = x.to(dev)

        def eager():
            with torch.no_grad():
                m(xd)
            if dev.type == "mps":
                torch.mps.synchronize()

        dt = timed(eager, iters=20)
        print(f"  eager PyTorch ({dev})            "
              f"{dt * 1e3:7.1f} ms/img | {1 / dt:6.1f} img/s | {dt * 1000:6.1f} s per 1k")


if __name__ == "__main__":
    main()
