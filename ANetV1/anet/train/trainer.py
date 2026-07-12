import csv
import math
import os
import threading
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

try:  # progress bars — degrade to a no-op wrapper if tqdm isn't installed
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it=None, *a, **k):
        return it if it is not None else _NullBar()

    class _NullBar:
        def update(self, *a, **k): pass
        def set_postfix(self, *a, **k): pass
        def set_description(self, *a, **k): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass


class _Heartbeat:
    """Prints an elapsed-seconds line from a daemon thread, so a long BLOCKING
    call (first-step compile / MIOpen autotune / first forward) still visibly
    ticks — tqdm can't animate during a single blocked iteration, which reads
    as 'frozen'. Use as a context manager around the slow region."""
    def __init__(self, msg, interval=5.0, writer=None):
        self.msg, self.interval = msg, interval
        self.writer = writer or (lambda m: print(m, flush=True))
        self._stop = threading.Event()
        self._th = None

    def __enter__(self):
        t0 = time.time()

        def run():
            while not self._stop.wait(self.interval):
                self.writer(f"  … {self.msg}: {int(time.time() - t0)}s elapsed "
                            "(alive — not hung)")
        self._th = threading.Thread(target=run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=0.2)

from ..model.norm import apply_norm_updates
from .losses import (balanced_tversky_loss, distill_kl, focal_loss,
                     focal_norm_loss, focal_tversky_loss, tversky_loss,
                     weighted_fp_tp_loss)
from .metrics import CellConfusion, ObjectMetrics, confident_pred


class ModelEMA:
    """EMA of the PARAMETERS, evaluated and checkpointed instead of the raw
    weights (v9, D48): the object-recall selection metric is noisy epoch to
    epoch, and averaging removes the last-steps jitter from what gets flown.

    Parameters only, deliberately: DeployNorm's running buffers are already
    EMAs of data statistics — shadowing them from before stat-seeding and
    re-smoothing at decay 0.998 (~100x slower) would make every early-epoch
    eval normalize against stale garbage. The live buffers are used as-is.
    A warmup ramp (timm-style) debiases the cold start so epoch-0 eval isn't
    ~20-50% random-init weights."""

    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.updates = 0
        self.shadow = {k: p.detach().clone()
                       for k, p in model.named_parameters()}
        self._backup = None

    def _decay_now(self):
        # ramp: ~n/(n+10) early (tracks the live weights), -> decay later
        self.updates += 1
        return min(self.decay, self.updates / (self.updates + 10.0))

    @torch.no_grad()
    def update(self, model):
        d = self._decay_now()
        for k, p in model.named_parameters():
            self.shadow[k].lerp_(p.detach(), 1.0 - d)

    @torch.no_grad()
    def swap_in(self, model):
        """Load EMA weights into the model in place (optimizer/compile refs
        stay valid); swap_out restores."""
        self._backup = {}
        for k, p in model.named_parameters():
            self._backup[k] = p.detach().clone()
            p.copy_(self.shadow[k])

    @torch.no_grad()
    def swap_out(self, model):
        if self._backup is None:
            return
        for k, p in model.named_parameters():
            p.copy_(self._backup[k])
        self._backup = None


class CudaPrefetcher:
    """Background-thread batch pipeline for the in-process loader (ROCm boxes
    where spawn workers deadlock, D38): the next batch's memmap memcpy +
    pinned H2D copy run on a side thread/stream while the GPU crunches the
    current one. Without it the ~150 MB/step of loader work serializes with
    the training step."""

    def __init__(self, loader, device, depth=2):
        self.loader = loader
        self.device = device
        self.depth = depth

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        import queue
        q = queue.Queue(maxsize=self.depth)
        stream = torch.cuda.Stream()
        stop = object()
        cancel = threading.Event()

        def put(item):
            # bounded put that honors cancellation, so an abandoned consumer
            # (exception/early-stop escaping the epoch loop) can't leave the
            # worker blocked forever holding GPU batches + the loader iterator
            while not cancel.is_set():
                try:
                    q.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    continue
            return False

        def worker():
            try:
                for cpu in self.loader:
                    if cancel.is_set():
                        return
                    with torch.cuda.stream(stream):
                        gpu = {k: (v.to(self.device, non_blocking=True)
                                   if torch.is_tensor(v) else v)
                               for k, v in cpu.items()}
                        ev = torch.cuda.Event()
                        ev.record(stream)
                    if not put((gpu, ev)):
                        return
            except Exception as e:  # surface loader errors on the main thread
                put(e)
            put(stop)

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        try:
            while True:
                item = q.get()
                if item is stop:
                    break
                if isinstance(item, Exception):
                    raise item
                batch, ev = item
                torch.cuda.current_stream().wait_event(ev)
                yield batch
        finally:
            cancel.set()
            while not q.empty():  # release any queued GPU batches
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            th.join(timeout=2.0)


def pick_device():
    if torch.cuda.is_available():  # includes ROCm (MI300X presents as cuda)
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def yolo_device():
    """Device arg for ultralytics calls, same preference order as pick_device."""
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Trainer:
    def __init__(self, model, train_ds, val_ds, cfg, distill=False):
        self.cfg = cfg
        self.distill = distill
        self.device = pick_device()

        # Worker processes are forked lazily on the FIRST loader iteration, which is
        # after model.to(cuda) has initialized the CUDA/HIP + OpenMP runtimes in the
        # parent. On ROCm, fork() then copies already-locked native mutexes into the
        # child, which deadlocks in C — Ctrl+C can't interrupt it (needs kill -9).
        # Forcing "spawn" starts each worker as a fresh interpreter, so there is no
        # inherited GPU/threading state and the deadlock class is impossible.
        self._train_ds, self._val_ds = train_ds, val_ds
        self._build_loaders(cfg.train.batch_size)

        self.model = model.to(self.device)

        # hard allocator cap: fail loudly instead of swap-freezing the machine
        frac = getattr(cfg.train, "mps_memory_frac", None)
        if frac and self.device.type == "mps":
            torch.mps.set_per_process_memory_fraction(float(frac))
        if self.device.type == "cuda":
            # cap the allocator below physical VRAM: overallocation then raises a
            # catchable torch OOM (traceback) instead of the amdgpu/KFD driver
            # SIGTERMing the process ("Terminated", no traceback)
            cfrac = getattr(cfg.train, "cuda_memory_frac", None)
            if cfrac:
                torch.cuda.set_per_process_memory_fraction(float(cfrac))
            # TF32 for the fp32 GEMMs that stay outside autocast (ManualBatchNorm
            # runs fp32 on purpose) — free speed on Ampere+/CDNA3
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")  # tensor-core/MFMA fp32 GEMMs
            # benchmark=True on ROCm forces an exhaustive MIOpen search per unique
            # conv shape — measured ~27 min of "hang" in epoch 0 for zero steady-
            # state gain (tiny launch-bound convs). MIOPEN_FIND_MODE=FAST does the
            # right thing instead. Preset default: True on NVIDIA, False on HIP.
            if getattr(cfg.train, "cudnn_benchmark", False):
                torch.backends.cudnn.benchmark = True

        # amp: "fp16" | "bf16" | none — MPS/CUDA only (D30).
        # fp16 measured NaN on this model at batch 8; leave off unless re-validated.
        amp = getattr(cfg.train, "amp", None)
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.amp_dtype is torch.float16
        )
        # self.model stays the RAW module (clean state_dicts, eager eval paths);
        # compiled entry points wrap it instead:
        #   _loss_fn   — the whole train-step math (normalize + forward + loss +
        #                reg) in ONE compiled region, so inductor fuses fwd AND
        #                bwd elementwise chains into a few Triton kernels and the
        #                loss tail (dozens of tiny-tensor kernels) stops paying
        #                per-op launch overhead
        #   model_eval — forward-only compiled wrapper for the val loop
        self._loss_fn = self._loss_tensors
        self.model_eval = self.model
        self._compiled = False  # bound methods aren't `is`-comparable; track explicitly
        # ANET_COMPILE=0/1 is a fast off/on switch that beats editing presets.
        want_compile = getattr(cfg.train, "compile", False)
        env_compile = os.environ.get("ANET_COMPILE")
        if env_compile is not None:
            want_compile = env_compile.strip().lower() in ("1", "true", "yes")
        if want_compile:
            # The model is launch-bound: thousands of tiny kernels at ~1% util.
            # Inductor fuses the elementwise chains (cos/tanh/sigmoid/silu, the
            # BN affine, the gated-pool multiplies) into a handful of Triton
            # kernels — the "fused kernels/ops" win — collapsing the dispatch
            # storm that dominates wall time on this tiny net.
            #   mode "default": Triton fusion, NO cudagraph capture. Chosen as
            #     the safe default because reduce-overhead's HIP-graph capture
            #     aliases the compiled output buffer, which fights grad-accum +
            #     the on-GPU loss accumulation (loss_win += loss.detach()) and
            #     crashed on this ROCm build. reduce-overhead is still available
            #     via compile_mode for a machine where cudagraphs behave.
            #   host-OOM guard: inductor forks one compile worker PER thread
            #     (default min(32,ncpu)), each a full torch import — that is what
            #     "Terminated" the box compiling the BACKWARD graph before. Cap
            #     to 1 unless the caller already set it. setdefault must run
            #     before the first (lazy) compile, i.e. here.
            os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
            # ANET_COMPILE_MODE overrides the preset. "reduce-overhead" captures
            # the whole step as a HIP/CUDA graph -> the ~1000s of kernel launches
            # replay as ONE launch (the launch-bound win). Safe only at accum=1
            # (no cross-step grad accumulation into a static-buffer graph) with
            # the cudagraph-safe loss .clone() above.
            mode = os.environ.get("ANET_COMPILE_MODE") or \
                getattr(cfg.train, "compile_mode", "default")
            if mode == "reduce-overhead" and cfg.train.accum_steps != 1:
                print("WARN: reduce-overhead (cudagraphs) needs accum_steps=1; "
                      f"have {cfg.train.accum_steps} — set ANET_ACCUM=1 "
                      "(e.g. ANET_BATCH=64 ANET_ACCUM=1)", flush=True)
            try:
                # dynamic=False: train shapes are static (drop_last=True), which
                # lets inductor specialize + skips guard overhead every step.
                self._loss_fn = torch.compile(self._loss_tensors, mode=mode,
                                              dynamic=False)
                self.model_eval = torch.compile(self.model, mode=mode)
                self._compiled = True
                print(f"torch.compile ON (mode={mode}, "
                      f"threads={os.environ['TORCHINDUCTOR_COMPILE_THREADS']}) — "
                      "first step compiles (slow), then fast", flush=True)
            except Exception as e:  # never let a compile setup error kill training
                self._loss_fn, self.model_eval = self._loss_tensors, self.model
                print(f"torch.compile setup FAILED ({type(e).__name__}: {e}) — "
                      "falling back to eager", flush=True)

        # fused AdamW: one multi-tensor kernel per step instead of ~6 tiny
        # launches per parameter tensor — real money on a launch-bound model
        try:
            self.opt = torch.optim.AdamW(
                self.model.parameters(), lr=cfg.train.lr, weight_decay=0.0,
                fused=self.device.type == "cuda",
            )
        except RuntimeError:  # fused unsupported on this build — foreach fallback
            self.opt = torch.optim.AdamW(
                self.model.parameters(), lr=cfg.train.lr, weight_decay=0.0,
                foreach=True,
            )
        opt_steps_per_epoch = max(len(self.train_loader) // cfg.train.accum_steps, 1)
        steps = cfg.train.epochs * opt_steps_per_epoch
        warmup = getattr(cfg.train, "warmup_steps", 0) or 0
        sched_mode = os.environ.get("ANET_SCHED") or getattr(cfg.train, "sched", "cosine")
        self.sched_mode = sched_mode
        self.sched_per_epoch = False       # plateau steps on the val metric, not per-batch
        self._warmup_total = warmup
        self._warmup_left = 0
        self._base_lrs = [g["lr"] for g in self.opt.param_groups]
        L = torch.optim.lr_scheduler
        if sched_mode == "plateau":
            # ReduceLROnPlateau on the val selection metric (higher=better),
            # stepped per-EPOCH after eval. Warmup is done manually per-step
            # below (it doesn't compose with SequentialLR).
            self.sched = L.ReduceLROnPlateau(
                self.opt, mode="max",
                factor=getattr(cfg.train, "plateau_factor", 0.5),
                patience=getattr(cfg.train, "plateau_patience", 3), min_lr=1e-5)
            self.sched_per_epoch = True
            self._warmup_left = warmup
        elif sched_mode == "restarts":
            # cosine warm restarts: LR blasts to peak, cosine-decays over T_0
            # opt-steps, then RESTARTS to peak (re-escapes a plateau). T_mult=2
            # lengthens each cycle. Restarts supply their own high-LR kicks, so
            # a separate warmup is unnecessary.
            t0 = max(getattr(cfg.train, "restart_epochs", 5) * opt_steps_per_epoch, 1)
            self.sched = L.CosineAnnealingWarmRestarts(self.opt, T_0=t0, T_mult=2)
        else:  # cosine (default): warmup -> smooth cosine to 0
            cosine = L.CosineAnnealingLR(self.opt, T_max=max(steps - warmup, 1))
            if warmup:
                self.sched = L.SequentialLR(
                    self.opt,
                    [L.LinearLR(self.opt, start_factor=0.05, total_iters=warmup), cosine],
                    milestones=[warmup])
            else:
                self.sched = cosine
        print(f"LR schedule: {sched_mode} | peak lr={self._base_lrs[0]:.1e} "
              f"warmup={warmup} steps", flush=True)

        self.out_dir = Path(cfg.train.checkpoint_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best = -1.0
        decay = getattr(cfg.train, "ema_decay", 0.0) or 0.0
        self.ema = ModelEMA(self.model, decay) if decay else None

    # ------------------------------------------------------------- v9 setup
    @torch.no_grad()
    def _seed_norm_stats(self, n_batches=8):
        """Run a few forward-only batches through the dense path so every
        DeployNorm buffer holds real data statistics before step 0 (D39).
        No autograd graph -> memory is a fraction of a training step."""
        if not any(m.__class__.__name__ == "DeployNorm" for m in self.model.modules()):
            return
        self.model.train()
        print(f"seeding DeployNorm stats ({n_batches} forward-only batches)...",
              flush=True)
        it = iter(self.train_loader)
        for _ in range(n_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            with self._autocast():
                self.model(self._prep_img(
                    batch["image"].to(self.device, non_blocking=True)))
            apply_norm_updates(self.model)  # sequential seeding (pending -> buffers)

    def _setup_fused(self):
        """Install the fused Triton Stage-1 with startup parity verification
        and layered demotion: triton bwd -> chunked-autograd bwd -> PyTorch
        dense (with a safe batch size)."""
        c = self.cfg.train
        want = getattr(c, "fused", False)
        env = os.environ.get("ANET_FUSED")
        if env is not None:
            want = env.strip().lower() in ("1", "true", "yes")
        if not (want and self.device.type == "cuda"
                and getattr(self.model, "arch", "v8") == "v9"):
            return
        import traceback

        def _dump(prefix, e):
            print(f"\n{'=' * 70}\n{prefix} ({type(e).__name__}: {e})\n"
                  f"{traceback.format_exc()}{'=' * 70}\n", flush=True)

        # FORWARD tier: import + parity. Failure here => the whole fused path is
        # unusable, drop to the dense fallback (smaller batch).
        try:
            from .fused import (FusedStage1, fused_available, parity_backward,
                                parity_forward)
            if not fused_available():
                raise RuntimeError("triton not importable on this box")
            n = min(2, len(self.val_loader.dataset))
            img = self._prep_img(torch.stack(
                [self.val_loader.dataset[i]["image"] for i in range(n)]
            ).to(self.device)).float()
            ok_f, delta = parity_forward(self.model, img)
            if not ok_f:
                raise RuntimeError(f"forward parity failed (max delta {delta:.3e})")
        except Exception as e:
            _dump("FUSED STAGE-1 FORWARD unavailable — PyTorch dense path", e)
            fb = getattr(c, "fallback_batch", 0) or 0
            if fb and c.batch_size > fb:
                print(f"dense fallback: rebuilding loaders at batch {fb} "
                      f"(was {c.batch_size}) to stay inside the VRAM budget",
                      flush=True)
                c.batch_size = fb
                self._build_loaders(fb)  # both loaders, same worker/spawn setup
            return

        # BACKWARD tier: a triton-backward CRASH (not just a numeric mismatch)
        # now demotes to chunked-autograd backward — the fused FORWARD and the
        # large batch are KEPT (recovers most of the fused speed) instead of
        # collapsing to dense/batch-32. parity_backward only reports numeric
        # mismatch; a kernel compile/launch crash escapes it, so it's caught.
        mode = os.environ.get("ANET_FUSED_BWD") or getattr(c, "fused_bwd", "triton")
        if mode == "triton":
            try:
                ok_b, rel = parity_backward(self.model, img)
            except Exception as e:
                _dump("FUSED TRITON BACKWARD crashed — keeping the fused "
                      "FORWARD, demoting only the backward to chunked-autograd "
                      "(batch stays large)", e)
                mode = "chunked"
            else:
                if not ok_b:
                    print(f"fused TRITON backward parity failed (worst rel "
                          f"{rel:.3e}) — demoting to chunked-autograd backward",
                          flush=True)
                    mode = "chunked"
                else:
                    print(f"fused backward parity OK (worst rel {rel:.3e})",
                          flush=True)
        try:
            self.model.fused_pool = FusedStage1(self.model, bwd_mode=mode)
            print(f"fused Stage-1 ON (bwd={mode}, fwd max delta {delta:.3e})",
                  flush=True)
        except Exception as e:
            _dump(f"FusedStage1 construction failed (bwd={mode}) — dense path", e)

    def _build_loaders(self, batch_size):
        """(Re)build both loaders at batch_size. One code path, so the fused-
        fallback rebuild keeps the SAME worker/spawn/prefetch settings — the
        first version dropped multiprocessing_context and reintroduced the
        ROCm fork-after-CUDA-init deadlock the constructor guards against."""
        cfg = self.cfg
        n_samples = getattr(cfg.train, "samples_per_epoch", None) or len(self._train_ds)
        sampler = WeightedRandomSampler(
            self._train_ds.sample_weights(), num_samples=n_samples, replacement=True
        )
        nw = cfg.train.num_workers
        common = dict(
            batch_size=batch_size,
            num_workers=nw,
            persistent_workers=nw > 0,
            pin_memory=self.device.type == "cuda",
        )
        if nw > 0:
            mp_ctx = getattr(cfg.train, "mp_context", None)
            if mp_ctx is None and self.device.type == "cuda":
                mp_ctx = "spawn"
            if mp_ctx:
                common["multiprocessing_context"] = mp_ctx
            common["prefetch_factor"] = getattr(cfg.train, "prefetch_factor", 2)
        self.train_loader = DataLoader(self._train_ds, sampler=sampler,
                                       drop_last=True, **common)
        self.val_loader = DataLoader(self._val_ds, shuffle=False, **common)

    def _autocast(self):
        return torch.autocast(self.device.type, self.amp_dtype,
                              enabled=self.amp_dtype is not None)

    @staticmethod
    def _prep_img(img):
        # uint8 loader path (cfg.data.uint8): normalize on-GPU. Inside the
        # compiled region this fuses straight into the stem's first ops.
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        return img

    def _loss(self, batch):
        img = batch["image"].to(self.device, non_blocking=True)
        grid = batch["grid"].to(self.device, non_blocking=True)
        teacher = (batch["teacher"].to(self.device, non_blocking=True)
                   if self.distill else None)
        band = (batch["band"].to(self.device, non_blocking=True)
                if "band" in batch else None)
        if not self._compiled:  # eager: no fallback needed
            return self._loss_fn(img, grid, teacher, band)
        try:  # lazy compilation happens on the FIRST call — catch + degrade here
            return self._loss_fn(img, grid, teacher, band)
        except Exception as e:
            print(f"torch.compile failed at first step ({type(e).__name__}: {e}) — "
                  "falling back to eager for the rest of training", flush=True)
            self._loss_fn = self._loss_tensors
            self.model_eval = self.model
            self._compiled = False
            return self._loss_fn(img, grid, teacher, band)

    def _loss_tensors(self, img, grid, teacher=None, band=None):
        out = self.model(self._prep_img(img))
        cells, aux = out if isinstance(out, tuple) else (out, None)
        c = self.cfg.train
        ta = getattr(c, "tversky_alpha", 0.7)
        tb = getattr(c, "tversky_beta", 0.3)
        smooth = getattr(c, "ft_smooth", 1.0)
        if isinstance(ta, (list, tuple)):
            ta = tuple(ta)
        # boundary-band background cells (partial coverage in [band_lo, thresh))
        # are label noise — drop them from the dense anchor so it stops pushing
        # "background" on cells that are half object (the ring tug-of-war)
        amask = None
        if band is not None:
            amask = ~(band.any(1) & (grid == 0))
        if getattr(c, "loss_mode", "combo") == "fp_tp":
            # v11 default: weighted per-class soft FP/TP ratio, one term. The
            # +smooth numerator makes predict-nothing cost ~sum(w) instead of 0,
            # so the collapse fixed point that killed focal_norm is gone.
            kw = dict(class_weights=tuple(getattr(c, "fp_tp_weights",
                                                  (0.05, 0.8, 0.15))),
                      smooth=getattr(c, "fp_tp_smooth", 1.0),
                      band=band)
            hard = weighted_fp_tp_loss(cells, grid, **kw)
            if aux is not None:
                hard = hard + getattr(c, "aux_weight", 0.3) * \
                    weighted_fp_tp_loss(aux, grid, **kw)
        elif getattr(c, "loss_mode", "combo") == "focal_norm":
            # v9 default (D47): ONE smooth per-cell term, per-class positive-
            # normalized (CenterNet-style size invariance without set-ratio
            # dynamics — no tug-of-war, no limit cycles).
            kw = dict(gamma=c.focal_gamma,
                      class_weights=tuple(getattr(c, "focal_norm_weights",
                                                  (1.0, 2.0, 1.0))),
                      mask=amask)
            hard = focal_norm_loss(cells, grid, **kw)
            if aux is not None:
                # deep supervision (D46): direct encoder gradient that a
                # collapsed head can't block; train-only, dropped at export
                hard = hard + getattr(c, "aux_weight", 0.3) * \
                    focal_norm_loss(aux, grid, **kw)
        elif getattr(c, "loss_mode", "combo") == "balanced":
            # class-balanced Focal-Tversky over {bg, mannequin, tent}, one term.
            # No anchor to fight -> no mannequin 0<->over-predict oscillation.
            # balanced_alpha/beta (3-tuple bg,mann,tent | 2-tuple mann,tent | scalar)
            ba = getattr(c, "balanced_alpha", None)
            bb = getattr(c, "balanced_beta", None)
            ba = tuple(ba) if isinstance(ba, (list, tuple)) else (ba if ba is not None else ta)
            bb = tuple(bb) if isinstance(bb, (list, tuple)) else (bb if bb is not None else tb)
            cw = getattr(c, "balanced_class_weights", None)
            hard = balanced_tversky_loss(
                cells, grid, alpha=ba, beta=bb,
                gamma=getattr(c, "ft_gamma", 0.75), smooth=smooth, band=band,
                difficulty_temp=getattr(c, "difficulty_temp", None),
                class_weights=cw)
        elif getattr(c, "loss_mode", "combo") == "focal_tversky":
            # single balanced term (no focal-vs-Tversky fight) + a GENTLE per-cell
            # focal anchor for dense, stable gradient early on. The anchor uses a
            # mild alpha (it stabilizes; focal-Tversky does the class balancing).
            ft = focal_tversky_loss(cells, grid, alpha=ta, beta=tb,
                                    gamma=getattr(c, "ft_gamma", 0.75),
                                    smooth=smooth, band=band)
            anchor = focal_loss(cells, grid, gamma=c.focal_gamma,
                                alpha=tuple(getattr(c, "ft_anchor_alpha", (1.0, 2.0, 2.0))),
                                mask=amask)
            hard = ft + getattr(c, "ft_anchor_weight", 0.5) * anchor
        else:  # legacy: focal + separately-weighted Tversky (can tug-of-war)
            hard = focal_loss(cells, grid, gamma=c.focal_gamma,
                              alpha=tuple(c.class_alpha), mask=amask)
            tw = getattr(c, "tversky_weight", 0.0) or 0.0
            if tw:
                hard = hard + tw * tversky_loss(cells, grid, alpha=ta, beta=tb,
                                                smooth=smooth, band=band)
        loss = hard
        if teacher is not None:
            loss = (1.0 - self.cfg.distill.kl_weight) * hard + \
                self.cfg.distill.kl_weight * distill_kl(
                    cells, teacher, temperature=self.cfg.distill.temperature,
                )
        l2, l1 = self.model.reg_losses()
        return loss + self.cfg.train.l2_score_reg * l2 + self.cfg.train.l1_kernel_reg * l1

    def train(self):
        accum = self.cfg.train.accum_steps
        patience = getattr(self.cfg.train, "early_stop_patience", 0) or 0
        clip = getattr(self.cfg.train, "grad_clip", 0.0) or 0.0
        # loss.item() every step is a full device sync — it stops the CPU from
        # queueing ahead, which on a launch-bound model IS the throughput.
        # check>1: accumulate the loss on-GPU, sync + NaN-check once per window.
        check = max(int(getattr(self.cfg.train, "nan_check_every", 1) or 1), 1)
        stale, nan_streak = 0, 0
        n_batches = len(self.train_loader)
        log_path = self.out_dir / "log.csv"
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "mannequin_recall",
                                    "mannequin_recall_synthetic", "tent_recall",
                                    "fp_per_image", "seconds"])
        print(
            f"train: device={self.device} batches/epoch={n_batches} "
            f"accum={accum} amp={self.amp_dtype or 'off'} "
            f"workers={self.cfg.train.num_workers}",
            flush=True,
        )
        self._seed_norm_stats(getattr(self.cfg.train, "seed_stat_batches", 8) or 0)
        self._setup_fused()
        n_batches = len(self.train_loader)  # a fused demotion rebuilds loaders
        use_prefetch = (self.device.type == "cuda"
                        and self.cfg.train.num_workers == 0
                        and (getattr(self.cfg.train, "prefetch", True)))
        # unmissable status line, printed LAST (below the MIOpen autotune spam)
        # so the actual perf tier is obvious without grepping the log
        fused = "ON" if getattr(self.model, "fused_pool", None) is not None \
            else "OFF (dense fallback)"
        print(f"=== ANetV1 {getattr(self.model, 'arch', '?')} ready: "
              f"fused={fused} | compile={'ON' if self._compiled else 'OFF'} | "
              f"batch={self.cfg.train.batch_size} | {n_batches} batches/epoch ===",
              flush=True)
        for epoch in range(self.cfg.train.epochs):
            self.model.train()
            t0, running, n = time.time(), 0.0, 0
            loss_win = torch.zeros((), device=self.device)  # on-GPU window sum
            win_n = 0
            self.opt.zero_grad()
            if epoch == 0 and self.cfg.train.num_workers > 0:
                print(f"epoch 0: spawning {self.cfg.train.num_workers} dataloader "
                      "workers + fetching first batch...", flush=True)
            # the FIRST step of epoch 0 blocks for minutes (compile + MIOpen
            # autotune). tqdm shows elapsed time + a live bar so it never looks
            # hung; the bar sits at 0/N with a climbing clock while it compiles.
            first = epoch == 0 and self.device.type == "cuda"
            bar = tqdm(total=n_batches, desc=f"epoch {epoch}",
                       unit="step", dynamic_ncols=True, leave=True,
                       initial=0)
            if first:
                bar.set_description(f"epoch {epoch} [first step warming up]")
            # heartbeat ticks from a daemon thread through the whole first step
            # (loader wait + MIOpen autotune + optional compile), which is ONE
            # blocking call tqdm can't animate — this is how you tell alive from
            # hung. Started before the loop so it also covers the loader wait.
            hb = _Heartbeat("first step warming up", interval=5.0) if first else None
            if hb is not None:
                compiled = bool(getattr(self.cfg.train, "compile", False))
                print(f"epoch {epoch}: entering first step — MIOpen autotune"
                      + (" + torch.compile" if compiled else "")
                      + " (one-time, can take minutes; heartbeat every 5s)",
                      flush=True)
                hb.__enter__()
            loader = (CudaPrefetcher(self.train_loader, self.device)
                      if use_prefetch else self.train_loader)
            for step, batch in enumerate(loader):
                if step == 0 and hb is not None:
                    hb.writer(f"epoch {epoch}: first batch loaded — now in the "
                              "first forward (loader OK)")
                with self._autocast():
                    loss = self._loss(batch) / accum
                if step == 0 and first:
                    bar.set_description(f"epoch {epoch}")
                self.scaler.scale(loss).backward()
                # DeployNorm stat updates run AFTER backward: mutating the
                # buffers inside the fwd->bwd window trips AOT autograd's
                # saved-tensor version check under torch.compile (norm.py)
                apply_norm_updates(self.model)
                # copy the loss OUT of its buffer NOW: under cudagraphs (compile
                # mode reduce-overhead) `loss` is a static output tensor that the
                # NEXT replay overwrites, so a plain view-add would read stale
                # values once we accumulate >1 step (nan_check_every>1). .clone()
                # forces a real copy this step; it's a scalar, so ~free.
                loss_win += loss.detach().clone()
                win_n += 1
                if step == 0 and hb is not None:
                    hb.__exit__()  # first step done — steps flow now, bar animates
                    hb = None
                if step == 0 or win_n >= check or step + 1 == n_batches:
                    wsum = loss_win.item() * accum  # the ONE sync per window
                    loss_win.zero_()
                    if step == 0:
                        compiled = bool(getattr(self.cfg.train, "compile", False))
                        note = ((f" ({'compile + ' if compiled else ''}MIOpen/cuDNN "
                                 "autotune done — steady state now)")
                                if epoch == 0 and self.device.type == "cuda" else "")
                        bar.write(f"epoch {epoch}: first step loss={wsum:.4f}{note}")
                    if not math.isfinite(wsum):
                        # drop the pending accumulation window. With check>1 the
                        # window's earlier optimizer steps already ran, so a NaN
                        # is caught up to `check` steps late — real divergence
                        # keeps producing NaN windows and dies fast below.
                        nan_streak += 1
                        win_n = 0
                        self.opt.zero_grad()
                        if nan_streak == 1 or nan_streak % 10 == 0:
                            bar.write(f"epoch {epoch} step {step}: non-finite loss "
                                      f"(streak {nan_streak}) — skipping step")
                        if nan_streak * check >= 25:
                            bar.close()
                            raise RuntimeError(
                                "non-finite loss for >=25 consecutive steps — model "
                                "has diverged; lower lr / check amp before rerunning")
                        bar.update(1)
                        continue
                    nan_streak = 0
                    running, n = running + wsum, n + win_n
                    win_n = 0
                    bar.set_postfix(loss=f"{running / max(n, 1):.4f}",
                                    lr=f"{self.opt.param_groups[0]['lr']:.1e}")
                bar.update(1)
                if (step + 1) % accum == 0:
                    if clip:
                        if self.scaler.is_enabled():
                            self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad()
                    if self.ema is not None:
                        self.ema.update(self.model)
                    if self.sched_per_epoch:
                        # plateau mode: no per-batch decay; run a manual linear
                        # warmup during the warmup window, then hold at peak
                        # until the per-epoch plateau step drops LR.
                        if self._warmup_left > 0:
                            self._warmup_left -= 1
                            frac = 1.0 - self._warmup_left / max(self._warmup_total, 1)
                            for g, base in zip(self.opt.param_groups, self._base_lrs):
                                g["lr"] = base * (0.05 + 0.95 * frac)
                    else:
                        self.sched.step()
            bar.close()
            # evaluate + checkpoint the EMA weights when EMA is on (D48):
            # what gets selected and saved is exactly what gets flown
            if self.ema is not None:
                self.ema.swap_in(self.model)
            try:
                stats = self.evaluate(self.val_loader, desc=f"epoch {epoch} eval")
                # SYNTHETIC mannequin recall is the mission metric (ARCH §10; overall
                # recall is ~94% VisDrone and pinned at 0), but selecting on it alone
                # saved a mannequin-maximal/tent-dead checkpoint (ep5 over-prediction).
                # Select on mannequin + w*tent so best.pt tracks a deployable model.
                mann = stats["mannequin_recall_synthetic"]
                if math.isnan(mann):
                    mann = stats["mannequin_recall"]
                tent_w = getattr(self.cfg.train, "select_tent_weight", 0.5)
                key = mann + tent_w * stats["tent_recall"]
                if self.sched_per_epoch and self._warmup_left == 0:
                    self.sched.step(key)  # ReduceLROnPlateau on the selection metric
                row = [epoch, running / max(n, 1), stats["mannequin_recall"],
                       stats["mannequin_recall_synthetic"], stats["tent_recall"],
                       stats["fp_per_image"], round(time.time() - t0)]
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow(row)
                mc = stats["cells"]["mannequin"]
                print(f"epoch {epoch}: loss={row[1]:.4f} mannequin_r={mann:.3f} "
                      f"(synth {stats['mannequin_recall_synthetic']:.3f}, "
                      f"cell_r={mc['recall']:.3f}, pred_cells={mc['pred_cells']}/{mc['gt_cells']}) "
                      f"tent_r={stats['tent_recall']:.3f} fp/img={stats['fp_per_image']:.2f} "
                      f"sel={key:.3f} lr={self.opt.param_groups[0]['lr']:.2e} ({row[-1]}s)")
                # threshold-free progress: if these climb while mannequin_r stays 0,
                # the model IS learning under the argmax/conf_thresh bar (not stuck)
                print(f"  soft p(fg on gt): mann={stats['soft_mann']:.3f} "
                      f"tent={stats['soft_tent']:.3f} | argmax fg cells={stats['argmax_fg']} "
                      f"(threshold-free — cross ~0.5 to win argmax)", flush=True)
                state = getattr(self.model, "_orig_mod", self.model).state_dict()
                torch.save(state, self.out_dir / "last.pt")
                # never promote a diverged epoch to best.pt (recall 0.0 "beats" the
                # -1.0 sentinel, which is how a NaN model got saved as best once)
                improved = math.isfinite(row[1]) and key > self.best
                if improved:
                    self.best, stale = key, 0
                    torch.save(state, self.out_dir / "best.pt")
            finally:
                # exceptions between swap_in and here (eval crash, disk-full
                # torch.save) must not leave EMA weights in the live model
                if self.ema is not None:
                    self.ema.swap_out(self.model)
            if not improved:
                stale += 1
                min_ep = getattr(self.cfg.train, "early_stop_min_epochs", 0) or 0
                if patience and stale >= patience and epoch + 1 >= min_ep:
                    print(f"early stop at epoch {epoch}: selection metric stuck at "
                          f"{self.best:.3f} for {patience} epochs")
                    break
            if self.device.type == "mps":
                torch.mps.empty_cache()  # release train-shape reservations before eval shapes re-cache
            # NOTE: no cuda.empty_cache() — on a 192GB GPU there's no pressure and
            # it hands blocks back to the driver, stalling the next epoch's first steps

    @torch.no_grad()
    def evaluate(self, loader, desc="eval"):
        self.model.eval()
        cells_m, obj_m = CellConfusion(), ObjectMetrics()
        thresh = getattr(self.cfg.train, "conf_thresh", 0.0) or 0.0
        # THRESHOLD-FREE soft signal: mean softmax prob for each foreground class
        # over the cells where it is the GT, and the argmax-only fg cell count.
        # These move BEFORE the thresholded metrics do — so an all-zero
        # mannequin_r with rising soft_mann means the model IS learning, just
        # under the conf_thresh/argmax bar (not stuck). Removes the "loss drops
        # but nothing improves" ambiguity.
        soft_sum = {1: 0.0, 2: 0.0}
        soft_n = {1: 0, 2: 0}
        argmax_fg = {1: 0, 2: 0}
        for batch in tqdm(loader, desc=desc, unit="batch",
                          dynamic_ncols=True, leave=False):
            img = self._prep_img(batch["image"].to(self.device, non_blocking=True))
            with self._autocast():
                logits = self.model_eval(img)
            fl = logits.float()
            probs = torch.softmax(fl, 1)
            am = fl.argmax(1)
            target = batch["grid"].numpy()
            tgt = batch["grid"].to(fl.device)
            for c in (1, 2):
                m = tgt == c
                soft_sum[c] += float(probs[:, c][m].sum()) if m.any() else 0.0
                soft_n[c] += int(m.sum())
                argmax_fg[c] += int((am == c).sum())
            pred = confident_pred(fl, thresh).cpu().numpy()
            cells_m.update(pred, target)
            for i in range(pred.shape[0]):
                obj_m.update(pred[i], batch["boxes"][i].numpy(), bool(batch["vd"][i]))
        out = obj_m.summary()
        out["cells"] = cells_m.summary()
        out["soft_mann"] = soft_sum[1] / max(soft_n[1], 1)
        out["soft_tent"] = soft_sum[2] / max(soft_n[2], 1)
        out["argmax_fg"] = argmax_fg[1] + argmax_fg[2]
        return out
