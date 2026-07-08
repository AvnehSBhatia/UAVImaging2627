"""ONNX Runtime inference wrapper — the fast local path.

Measured on an M-series Mac at batch 1 (hidden=24, edge_dq, D34 eval graph):
~21 ms/img via the CoreML EP (GPU) vs ~92 ms eager PyTorch on MPS and ~436 ms
eager CPU. Exact parity with the PyTorch model (max logit delta ~1e-6, cell
argmax agreement 1.0 on real val frames).

Export first, then use the wrapper like the torch model:

    python scripts/export_onnx.py --ckpt runs/anet/best.pt

    from anet.onnxrt import OnnxANet
    model = OnnxANet("runs/anet/anet.onnx")
    cells = model(images)   # (B,3,540,960) float32 in [0,1] -> (B,3,54,96) logits
"""

import numpy as np


def default_providers():
    import onnxruntime as ort

    providers = []
    if "CoreMLExecutionProvider" in ort.get_available_providers():
        # MLProgram + ALL routes the graph (single partition since D34) to the
        # GPU. CPUAndNeuralEngine measured far slower: the ANE handles this
        # op mix (reshape/elementwise-heavy) poorly. fp16 also measured slower
        # than fp32 on Apple GPUs here — don't "optimize" without re-measuring.
        providers.append(("CoreMLExecutionProvider",
                          {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}))
    providers.append("CPUExecutionProvider")
    return providers


class OnnxANet:
    """Callable session with the torch model's interface for bulk inference."""

    def __init__(self, path, providers=None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(path), so, providers=providers or default_providers()
        )
        self.batch = int(self.sess.get_inputs()[0].shape[0])  # fixed at export

    def __call__(self, images):
        """images: torch tensor or ndarray, (3,540,960) or (B,3,540,960) in [0,1].
        Returns (B,3,54,96) float32 cell logits. Batches larger than the export
        batch run in chunks; the tail chunk is zero-padded then trimmed."""
        x = images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images)
        x = x.astype(np.float32, copy=False)
        if x.ndim == 3:
            x = x[None]
        n, b, outs = x.shape[0], self.batch, []
        for i in range(0, n, b):
            chunk = x[i : i + b]
            take = chunk.shape[0]
            if take < b:
                chunk = np.concatenate(
                    [chunk, np.zeros((b - take, *chunk.shape[1:]), np.float32)], 0
                )
            outs.append(self.sess.run(None, {"frame": chunk})[0][:take])
        return np.concatenate(outs, 0)
