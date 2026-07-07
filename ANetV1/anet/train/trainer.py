import csv
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .losses import distill_kl, focal_loss
from .metrics import CellConfusion, ObjectMetrics


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
        n_samples = getattr(cfg.train, "samples_per_epoch", None) or len(train_ds)
        sampler = WeightedRandomSampler(
            train_ds.sample_weights(), num_samples=n_samples, replacement=True
        )
        nw = cfg.train.num_workers
        common = dict(
            batch_size=cfg.train.batch_size,
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
        self.train_loader = DataLoader(train_ds, sampler=sampler, drop_last=True, **common)
        self.val_loader = DataLoader(val_ds, shuffle=False, **common)

        self.model = model.to(self.device)

        # hard allocator cap: fail loudly instead of swap-freezing the machine
        frac = getattr(cfg.train, "mps_memory_frac", None)
        if frac and self.device.type == "mps":
            torch.mps.set_per_process_memory_fraction(float(frac))
        # benchmark=True on ROCm forces an exhaustive MIOpen search per unique
        # conv shape — measured ~27 min of "hang" in epoch 0 for zero steady-state
        # gain (tiny launch-bound convs). MIOPEN_FIND_MODE=FAST does the right
        # thing instead. Opt back in via cudnn_benchmark: true (NVIDIA, fat convs).
        if self.device.type == "cuda" and getattr(cfg.train, "cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True

        # amp: "fp16" | "bf16" | none — MPS/CUDA only (D30).
        # fp16 measured NaN on this model at batch 8; leave off unless re-validated.
        amp = getattr(cfg.train, "amp", None)
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.amp_dtype is torch.float16
        )
        if getattr(cfg.train, "compile", False):
            self.model = torch.compile(self.model)

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.train.lr, weight_decay=0.0
        )
        steps = cfg.train.epochs * (len(self.train_loader) // cfg.train.accum_steps)
        warmup = getattr(cfg.train, "warmup_steps", 0) or 0
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=max(steps - warmup, 1)
        )
        if warmup:  # linear warmup -> cosine; large batches need the ramp
            self.sched = torch.optim.lr_scheduler.SequentialLR(
                self.opt,
                [torch.optim.lr_scheduler.LinearLR(
                    self.opt, start_factor=0.05, total_iters=warmup), cosine],
                milestones=[warmup],
            )
        else:
            self.sched = cosine

        self.out_dir = Path(cfg.train.checkpoint_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best = -1.0

    def _autocast(self):
        return torch.autocast(self.device.type, self.amp_dtype,
                              enabled=self.amp_dtype is not None)

    def _reg_losses(self):
        m = getattr(self.model, "_orig_mod", self.model)
        return m.reg_losses()

    def _loss(self, batch):
        cells = self.model(batch["image"].to(self.device, non_blocking=True))
        loss = focal_loss(
            cells, batch["grid"].to(self.device, non_blocking=True),
            gamma=self.cfg.train.focal_gamma, alpha=tuple(self.cfg.train.class_alpha),
        )
        if self.distill:
            loss = (1.0 - self.cfg.distill.kl_weight) * loss + \
                self.cfg.distill.kl_weight * distill_kl(
                    cells, batch["teacher"].to(self.device),
                    temperature=self.cfg.distill.temperature,
                )
        l2, l1 = self._reg_losses()
        return loss + self.cfg.train.l2_score_reg * l2 + self.cfg.train.l1_kernel_reg * l1

    def train(self):
        accum = self.cfg.train.accum_steps
        patience = getattr(self.cfg.train, "early_stop_patience", 0) or 0
        clip = getattr(self.cfg.train, "grad_clip", 0.0) or 0.0
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
        for epoch in range(self.cfg.train.epochs):
            self.model.train()
            t0, running, n = time.time(), 0.0, 0
            self.opt.zero_grad()
            if epoch == 0 and self.cfg.train.num_workers > 0:
                print(f"epoch 0: spawning {self.cfg.train.num_workers} dataloader "
                      "workers + fetching first batch...", flush=True)
            for step, batch in enumerate(self.train_loader):
                if step == 0:
                    print(f"epoch {epoch}: first batch loaded, forward...", flush=True)
                with self._autocast():
                    loss = self._loss(batch) / accum
                self.scaler.scale(loss).backward()
                lv = loss.item() * accum
                if step == 0:
                    print(
                        f"epoch {epoch}: first step loss={lv:.4f} "
                        f"(MIOpen may be slow for ~1-2 min on first CUDA ops)",
                        flush=True,
                    )
                if not math.isfinite(lv):
                    # drop the poisoned accumulation window; a persistent NaN
                    # means diverged weights — die fast, don't train garbage
                    nan_streak += 1
                    self.opt.zero_grad()
                    if nan_streak == 1 or nan_streak % 10 == 0:
                        print(f"epoch {epoch} step {step}: non-finite loss "
                              f"(streak {nan_streak}) — skipping step", flush=True)
                    if nan_streak >= 25:
                        raise RuntimeError(
                            "non-finite loss for 25 consecutive steps — model has "
                            "diverged; lower lr / check amp before rerunning")
                    continue
                nan_streak = 0
                running, n = running + lv, n + 1
                if step > 0 and (step + 1) % 100 == 0:
                    print(f"epoch {epoch}: {step + 1}/{n_batches} steps", flush=True)
                if (step + 1) % accum == 0:
                    if clip:
                        if self.scaler.is_enabled():
                            self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad()
                    self.sched.step()
            stats = self.evaluate(self.val_loader)
            # select/early-stop on SYNTHETIC mannequin recall (the mission metric,
            # ARCHITECTURE §10) — overall recall is ~94% VisDrone boxes, which
            # pinned the key at 0.000 and made epoch-0 weights "best" once
            key = stats["mannequin_recall_synthetic"]
            if math.isnan(key):
                key = stats["mannequin_recall"]
            row = [epoch, running / max(n, 1), stats["mannequin_recall"],
                   stats["mannequin_recall_synthetic"], stats["tent_recall"],
                   stats["fp_per_image"], round(time.time() - t0)]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(f"epoch {epoch}: loss={row[1]:.4f} mannequin_r={key:.3f} "
                  f"(synth {stats['mannequin_recall_synthetic']:.3f}) "
                  f"tent_r={stats['tent_recall']:.3f} fp/img={stats['fp_per_image']:.2f} "
                  f"lr={self.opt.param_groups[0]['lr']:.2e} ({row[-1]}s)")
            state = getattr(self.model, "_orig_mod", self.model).state_dict()
            torch.save(state, self.out_dir / "last.pt")
            # never promote a diverged epoch to best.pt (recall 0.0 "beats" the
            # -1.0 sentinel, which is how a NaN model got saved as best once)
            if math.isfinite(row[1]) and key > self.best:
                self.best, stale = key, 0
                torch.save(state, self.out_dir / "best.pt")
            else:
                stale += 1
                min_ep = getattr(self.cfg.train, "early_stop_min_epochs", 0) or 0
                if patience and stale >= patience and epoch + 1 >= min_ep:
                    print(f"early stop at epoch {epoch}: selection metric stuck at "
                          f"{self.best:.3f} for {patience} epochs")
                    break
            if self.device.type == "mps":
                torch.mps.empty_cache()  # release train-shape reservations before eval shapes re-cache
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()  # return cached blocks; keeps reserved from creeping up

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        cells_m, obj_m = CellConfusion(), ObjectMetrics()
        for batch in loader:
            with self._autocast():
                logits = self.model(batch["image"].to(self.device, non_blocking=True))
            pred = logits.argmax(1).cpu().numpy()
            target = batch["grid"].numpy()
            cells_m.update(pred, target)
            for i in range(pred.shape[0]):
                obj_m.update(pred[i], batch["boxes"][i].numpy(), bool(batch["vd"][i]))
        out = obj_m.summary()
        out["cells"] = cells_m.summary()
        return out
