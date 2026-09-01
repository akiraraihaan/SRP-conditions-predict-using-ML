"""One training implementation, used identically by all five arms.

Selection criterion: **validation loss on the fold's 10 % slice, computed
UNWEIGHTED**, i.e. plain cross-entropy with no class weighting, regardless of
whether the training loss is weighted.

That combination is deliberate:

* Unweighted, because the ablation compares a weighted-loss arm against an
  unweighted-loss arm. If the selection criterion followed the training loss,
  the two arms would be selecting checkpoints against different objectives and
  the comparison would confound "did weighting help" with "were the two arms
  even choosing the same kind of checkpoint". A single fixed criterion isolates
  the weighting effect.
* Loss rather than macro-F1, because the slice is 54 images and the rarest class
  contributes 2 of them (artifacts/folds_report.md). Macro-F1 on 2 images moves
  in steps of 0.5 for that class; loss is continuous and far less noisy.

Class weights on the TRAINING loss are the sklearn "balanced" form,
w_c = N / (K * n_c), computed from the fold's own training portion only -- never
from val or test. `verify_class_weights_applied()` proves they actually reach the
loss, because the legacy pipeline computed exactly these weights and then never
passed them anywhere (MIGRATION_NOTES.md section 5.4).

No augmentation is applied. Dynamometer cards are not flip- or rotation-
invariant -- a horizontal flip reverses the traversal direction of the load
curve -- so the transform chain is letterbox -> normalise, and nothing else.
This is stated rather than assumed; the legacy runs used ultralytics' default
augmentation, which is one reason script 01 keeps the legacy trainer.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_arms_config, load_data_config, set_seed
from .data import load_letterboxed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


class ImageCache:
    """Letterboxed uint8 images held in RAM, keyed by idx.

    668 images at 224x224x3 uint8 is about 100 MB, so the whole corpus fits and
    every fold reuses it. Decoding once also removes I/O jitter from the
    latency figures recorded alongside each run.
    """

    def __init__(self, index, data_root: Path, image_size: int = 224):
        self.image_size = image_size
        self._data_root = Path(data_root)
        self._relpath = dict(zip(index["idx"].tolist(), index["relpath"].tolist()))
        self._cache: dict[int, np.ndarray] = {}

    def get(self, idx: int) -> np.ndarray:
        cached = self._cache.get(idx)
        if cached is None:
            image = load_letterboxed(self._data_root / self._relpath[idx], self.image_size)
            cached = np.asarray(image, dtype=np.uint8)
            self._cache[idx] = cached
        return cached

    def warm(self, idxs) -> None:
        for i in idxs:
            self.get(int(i))


class FoldDataset:
    """torch Dataset over a list of idx values. Normalise on the fly."""

    def __init__(self, cache: ImageCache, idxs, labels_by_idx: dict[int, int]):
        import torch

        self.cache = cache
        self.idxs = [int(i) for i in idxs]
        self.labels = [int(labels_by_idx[int(i)]) for i in self.idxs]
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, position: int):
        import torch

        array = self.cache.get(self.idxs[position])
        # copy(): the cache array is shared and must stay read-only
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)
        tensor = (tensor - self._mean) / self._std
        return tensor, self.labels[position], self.idxs[position]


def balanced_class_weights(labels, num_classes: int) -> np.ndarray:
    """w_c = N / (K * n_c), the sklearn "balanced" form.

    Classes absent from `labels` get weight 0 rather than infinity; with the
    stratified folds used here that never happens, but it must not divide by zero
    if it ever does.
    """
    labels = np.asarray(labels)
    total = len(labels)
    weights = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        n_c = int((labels == c).sum())
        weights[c] = (total / (num_classes * n_c)) if n_c > 0 else 0.0
    return weights


# --------------------------------------------------------------------------
# configuration of the uniform loop
# --------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int
    batch: int
    lr: float
    optimizer: str = "SGD"
    class_weights: str = "balanced"          # "balanced" | "none"
    image_size: int = 224
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    lrf: float = 0.01                        # final lr as a fraction of lr0
    cos_lr: bool = False
    patience: int = 20
    num_classes: int = 10
    num_workers: int = 0
    amp: bool = True

    @classmethod
    def from_arm(cls, arm: str, arms_cfg: dict[str, Any] | None = None, **overrides):
        arms_cfg = arms_cfg or load_arms_config()
        from .models import resolve_arm_hyperparameters

        hyper = resolve_arm_hyperparameters(arm, arms_cfg)
        shared = arms_cfg["shared"]
        uniform = arms_cfg.get("uniform_protocol", {})
        kwargs: dict[str, Any] = {
            "epochs": hyper["epochs"],
            "batch": hyper["batch"],
            "lr": hyper["lr"],
            "optimizer": hyper["optimizer"],
            "class_weights": shared.get("class_weights", "balanced"),
            "image_size": int(shared["image_size"]),
            "patience": int(shared.get("patience", 20)),
            "num_classes": int(shared["num_classes"]),
        }
        for key in (
            "momentum",
            "weight_decay",
            "warmup_epochs",
            "warmup_momentum",
            "lrf",
            "cos_lr",
            "num_workers",
            "amp",
        ):
            if key in uniform:
                kwargs[key] = uniform[key]
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class TrainResult:
    best_state: dict
    best_epoch: int
    best_val_loss: float
    history: list[dict[str, Any]] = field(default_factory=list)
    wall_time_s: float = 0.0
    stopped_early: bool = False
    class_weights: list[float] | None = None
    determinism: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def _build_optimizer(module, cfg: TrainConfig):
    import torch

    params = [p for p in module.parameters() if p.requires_grad]
    name = (cfg.optimizer or "SGD").lower()
    if name == "musgd":
        try:
            from ultralytics.optim.muon import MuSGD
        except ImportError:
            from ultralytics.engine.trainer import MuSGD  # type: ignore
        return MuSGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError("Unknown optimizer %r" % cfg.optimizer)


def _lr_factor(epoch: int, cfg: TrainConfig) -> float:
    """Ultralytics-shaped schedule: linear (or cosine) decay from 1.0 to lrf."""
    progress = epoch / max(cfg.epochs - 1, 1)
    if cfg.cos_lr:
        return cfg.lrf + (1 - cfg.lrf) * 0.5 * (1 + math.cos(math.pi * progress))
    return (1 - progress) * (1.0 - cfg.lrf) + cfg.lrf


def train_fold(
    bundle,
    cache: ImageCache,
    train_idx,
    val_idx,
    labels_by_idx: dict[int, int],
    cfg: TrainConfig,
    seed: int,
    *,
    device: str | None = None,
    verbose: bool = True,
) -> TrainResult:
    """Train one arm on one fold. Returns the best-by-validation-loss state."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    determinism = set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    module = bundle.module.to(device)

    train_ds = FoldDataset(cache, train_idx, labels_by_idx)
    val_ds = FoldDataset(cache, val_idx, labels_by_idx)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(cfg.batch, 32), shuffle=False, num_workers=cfg.num_workers
    )

    # --- training-loss class weights, from the fold's TRAINING portion only ---
    weight_tensor = None
    weights_list = None
    if cfg.class_weights == "balanced":
        weights = balanced_class_weights(train_ds.labels, cfg.num_classes)
        weights_list = [float(w) for w in weights]
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    elif cfg.class_weights != "none":
        raise ValueError("class_weights must be 'balanced' or 'none', got %r" % cfg.class_weights)

    optimizer = _build_optimizer(module, cfg)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and device == "cuda"))

    warmup_iters = max(int(cfg.warmup_epochs * len(train_loader)), 1)
    best_val_loss = float("inf")
    best_state: dict = {}
    best_epoch = -1
    history: list[dict[str, Any]] = []
    since_improved = 0
    started = time.perf_counter()
    global_step = 0

    for epoch in range(cfg.epochs):
        module.train()
        factor = _lr_factor(epoch, cfg)
        running = 0.0
        seen = 0

        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # warmup on lr and momentum, mirroring ultralytics
            if global_step < warmup_iters:
                ratio = (global_step + 1) / warmup_iters
                for group, base in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base * factor * ratio
                    if "momentum" in group:
                        group["momentum"] = (
                            cfg.warmup_momentum + (cfg.momentum - cfg.warmup_momentum) * ratio
                        )
            else:
                for group, base in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base * factor
                    if "momentum" in group:
                        group["momentum"] = cfg.momentum

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(cfg.amp and device == "cuda")):
                logits = module(images)
                loss = F.cross_entropy(logits, targets, weight=weight_tensor)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.detach()) * images.size(0)
            seen += images.size(0)
            global_step += 1

        train_loss = running / max(seen, 1)

        # --- selection criterion: UNWEIGHTED validation cross-entropy ---
        module.eval()
        val_loss_sum = 0.0
        val_seen = 0
        val_correct = 0
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = module(images)
                # weight=None on purpose -- see the module docstring
                val_loss_sum += float(F.cross_entropy(logits, targets, reduction="sum"))
                val_correct += int((logits.argmax(dim=1) == targets).sum())
                val_seen += images.size(0)
        val_loss = val_loss_sum / max(val_seen, 1)
        val_acc = val_correct / max(val_seen, 1)

        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1

        history.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "best": improved,
            }
        )
        if verbose:
            print(
                "    epoch %3d/%d  lr %.5f  train_loss %.4f  val_loss %.4f  val_acc %.3f%s"
                % (
                    epoch + 1,
                    cfg.epochs,
                    optimizer.param_groups[0]["lr"],
                    train_loss,
                    val_loss,
                    val_acc,
                    "  <- best" if improved else "",
                )
            )

        if cfg.patience and since_improved >= cfg.patience:
            if verbose:
                print("    early stop: no val_loss improvement for %d epochs" % cfg.patience)
            break

    if not best_state:  # pathological, but never return an untracked model
        best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        best_epoch = cfg.epochs - 1

    module.load_state_dict(best_state)
    return TrainResult(
        best_state=best_state,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        history=history,
        wall_time_s=round(time.perf_counter() - started, 2),
        stopped_early=len(history) < cfg.epochs,
        class_weights=weights_list,
        determinism=determinism,
        device=device,
    )


# --------------------------------------------------------------------------
# class-weight verification
# --------------------------------------------------------------------------


def verify_class_weights_applied(num_classes: int = 10, seed: int = 0) -> dict[str, Any]:
    """Prove the balanced weights actually reach the training loss.

    The legacy pipeline computed exactly these weights, printed them, charted
    them -- and never passed them to the trainer (MIGRATION_NOTES.md section
    5.4). This test exists so that failure mode cannot recur silently.

    Three checks:
      1. w_c = N / (K * n_c) matches sklearn's compute_class_weight("balanced").
      2. Weighted and unweighted cross-entropy DIFFER on imbalanced data, and
         the weighted value equals a hand-computed weighted mean of the
         per-sample losses.
      3. One optimiser step taken with weights produces different parameters
         from the same step taken without them.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    set_seed(seed)
    results: dict[str, Any] = {}

    # imbalanced label vector: class 0 common, class 9 rare
    labels = np.array([0] * 60 + [1] * 20 + list(range(2, num_classes)) * 2)
    weights = balanced_class_weights(labels, num_classes)

    try:
        from sklearn.utils.class_weight import compute_class_weight

        reference = compute_class_weight(
            "balanced", classes=np.arange(num_classes), y=labels
        )
        results["matches_sklearn_balanced"] = bool(np.allclose(weights, reference))
        results["max_abs_diff_vs_sklearn"] = float(np.max(np.abs(weights - reference)))
    except ImportError:
        results["matches_sklearn_balanced"] = None

    torch.manual_seed(seed)
    logits = torch.randn(len(labels), num_classes)
    targets = torch.tensor(labels, dtype=torch.long)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    unweighted = float(F.cross_entropy(logits, targets))
    weighted = float(F.cross_entropy(logits, targets, weight=weight_tensor))

    per_sample = F.cross_entropy(logits, targets, reduction="none")
    sample_weights = weight_tensor[targets]
    manual = float((per_sample * sample_weights).sum() / sample_weights.sum())

    results["unweighted_loss"] = round(unweighted, 6)
    results["weighted_loss"] = round(weighted, 6)
    results["losses_differ"] = abs(weighted - unweighted) > 1e-6
    results["weighted_matches_manual"] = abs(weighted - manual) < 1e-5

    # a real optimiser step, with and without weights
    def step(use_weights: bool):
        set_seed(seed)
        torch.manual_seed(seed)
        model = nn.Linear(num_classes, num_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        optimizer.zero_grad()
        out = model(logits)
        loss = F.cross_entropy(out, targets, weight=weight_tensor if use_weights else None)
        loss.backward()
        optimizer.step()
        return model.weight.detach().clone()

    with_weights = step(True)
    without_weights = step(False)
    delta = float((with_weights - without_weights).abs().max())
    results["param_delta_after_one_step"] = round(delta, 8)
    results["optimiser_step_differs"] = delta > 1e-8

    results["passed"] = bool(
        results.get("matches_sklearn_balanced", True) is not False
        and results["losses_differ"]
        and results["weighted_matches_manual"]
        and results["optimiser_step_differs"]
    )
    results["class_weights"] = [round(float(w), 6) for w in weights]
    return results


def labels_by_idx_map(index, data_cfg: dict[str, Any] | None = None) -> dict[int, int]:
    """idx -> integer label in canonical class order."""
    data_cfg = data_cfg or load_data_config()
    lookup = {name: i for i, name in enumerate(data_cfg["classes"])}
    return {int(i): lookup[c] for i, c in zip(index["idx"], index["class"])}
