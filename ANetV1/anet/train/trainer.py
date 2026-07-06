import csv
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
        self.model = model.to(self.device)

        # hard allocator cap: fail loudly instead of swap-freezing the machine
        frac = getattr(cfg.train, "mps_memory_frac", None)
        if frac and self.device.type == "mps":
            torch.mps.set_per_process_memory_fraction(float(frac))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # fixed shapes; MIOpen/cuDNN autotune

        # amp: "fp16" | "bf16" | none — MPS/CUDA only (D30).
        # fp16 measured NaN on this model at batch 8; leave off unless re-validated.
        amp = getattr(cfg.train, "amp", None)
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.amp_dtype is torch.float16
        )
        if getattr(cfg.train, "compile", False):
            self.model = torch.compile(self.model)

        # replacement=True means an "epoch" is just a checkpoint/eval cadence;
        # samples_per_epoch shortens it without changing the sample distribution
        n_samples = getattr(cfg.train, "samples_per_epoch", None) or len(train_ds)
        sampler = WeightedRandomSampler(
            train_ds.sample_weights(), num_samples=n_samples, replacement=True
        )
        common = dict(
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            persistent_workers=cfg.train.num_workers > 0,
            pin_memory=self.device.type == "cuda",
        )
        self.train_loader = DataLoader(train_ds, sampler=sampler, drop_last=True, **common)
        self.val_loader = DataLoader(val_ds, shuffle=False, **common)

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
        stale = 0
        log_path = self.out_dir / "log.csv"
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "mannequin_recall",
                                    "tent_recall", "fp_per_image", "seconds"])
        for epoch in range(self.cfg.train.epochs):
            self.model.train()
            t0, running, n = time.time(), 0.0, 0
            self.opt.zero_grad()
            for step, batch in enumerate(self.train_loader):
                with self._autocast():
                    loss = self._loss(batch) / accum
                self.scaler.scale(loss).backward()
                running, n = running + loss.item() * accum, n + 1
                if (step + 1) % accum == 0:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad()
                    self.sched.step()
            stats = self.evaluate(self.val_loader)
            key = stats["mannequin_recall"]
            row = [epoch, running / max(n, 1), stats["mannequin_recall"],
                   stats["tent_recall"], stats["fp_per_image"], round(time.time() - t0)]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(f"epoch {epoch}: loss={row[1]:.4f} mannequin_r={key:.3f} "
                  f"tent_r={stats['tent_recall']:.3f} fp/img={stats['fp_per_image']:.2f} "
                  f"({row[-1]}s)")
            state = getattr(self.model, "_orig_mod", self.model).state_dict()
            torch.save(state, self.out_dir / "last.pt")
            if key > self.best:
                self.best, stale = key, 0
                torch.save(state, self.out_dir / "best.pt")
            else:
                stale += 1
                if patience and stale >= patience:
                    print(f"early stop at epoch {epoch}: mannequin_recall stuck at "
                          f"{self.best:.3f} for {patience} epochs")
                    break
            if self.device.type == "mps":
                torch.mps.empty_cache()  # release train-shape reservations before eval shapes re-cache

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
