"""One interface returning a trainable model plus efficiency stats, for all five arms.

    bundle = build_model("yolo26n")      # or yolo26s, yolo26m,
    bundle = build_model("resnet18")     #    mobilenetv3_small
    logits = bundle.module(images)       # ALWAYS logits, in train and eval alike

Downstream code never branches on architecture. Ultralytics supplies the three
YOLO backbones, torchvision the two baselines; both are wrapped so that the
forward signature, the output convention and the head are identical.

Two wrinkles this module absorbs so nothing else has to:

1. Ultralytics `ClassificationModel` returns raw logits in train mode but a
   SOFTMAX-normalised tuple in eval mode. Computing a validation loss on those
   probabilities with cross_entropy would be silently wrong. The wrapper probes
   the behaviour once at build time and, when the eval output is a probability
   vector, converts it back with log(p) -- which differs from the true logits
   only by a per-row constant, and so gives an identical cross-entropy.
2. The pretrained checkpoints carry a 1000-class ImageNet head. It is replaced
   with a freshly initialised 10-output layer; everything else is fine-tuned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, MissingInputError, load_arms_config, load_data_config, set_seed
from .efficiency import profile

YOLO_ARMS = {"yolo26n", "yolo26s", "yolo26m"}
TORCHVISION_ARMS = {"mobilenetv3_small", "resnet18"}
ARM_NAMES = sorted(YOLO_ARMS | TORCHVISION_ARMS)


@dataclass
class ModelBundle:
    """A trainable model plus everything downstream needs to describe it."""

    arm: str
    architecture: str
    backend: str
    module: Any
    efficiency: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def to(self, device):
        self.module.to(device)
        return self


# --------------------------------------------------------------------------
# output-convention wrapper
# --------------------------------------------------------------------------


def _make_logits_wrapper(net, image_size: int = 224):
    """Wrap a backbone so forward() returns logits in BOTH train and eval mode.

    Returns (wrapped_module, notes). The probe runs once, here, rather than
    per-batch.
    """
    import torch
    import torch.nn as nn

    def _first(out):
        while isinstance(out, (tuple, list)):
            out = out[0]
        return out

    was_training = net.training
    net.eval()
    with torch.no_grad():
        probe = _first(net(torch.zeros(2, 3, image_size, image_size)))
    net.train(was_training)

    row_sums = probe.sum(dim=-1)
    eval_returns_probs = bool(
        torch.all(probe >= 0) and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    )

    class LogitWrapper(nn.Module):
        """Uniform forward contract: logits, always."""

        def __init__(self, backbone, converts_probs: bool):
            super().__init__()
            self.backbone = backbone
            self.converts_probs = converts_probs

        def forward(self, x):
            out = self.backbone(x)
            while isinstance(out, (tuple, list)):
                out = out[0]
            if self.converts_probs and not self.training:
                # log(softmax(z)) == z - logsumexp(z); cross-entropy is invariant
                # to that per-row constant, so this is exact, not an approximation.
                out = torch.log(out.clamp_min(1e-12))
            return out

    notes = {
        "eval_returns_probabilities": eval_returns_probs,
        "logit_recovery": "log(p)" if eval_returns_probs else "none needed",
    }
    return LogitWrapper(net, eval_returns_probs), notes


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def _resolve_pretrained(filename: str, data_cfg: dict[str, Any]) -> str:
    """Locate a pretrained checkpoint, or hand the bare name to ultralytics.

    Search order: $SRPCARD_WEIGHTS_DIR, configs/data.yaml:weights_dir, the repo
    root, the current directory. If none holds it, the bare filename is returned
    and ultralytics downloads it -- which is what happens on Kaggle.
    """
    candidates: list[Path] = []
    env = os.environ.get("SRPCARD_WEIGHTS_DIR")
    if env:
        candidates.append(Path(env))
    configured = data_cfg.get("weights_dir")
    if configured:
        path = Path(configured)
        candidates.append(path if path.is_absolute() else REPO_ROOT / path)
    candidates += [REPO_ROOT, Path.cwd()]

    for directory in candidates:
        candidate = directory / filename
        if candidate.exists():
            return str(candidate)
    return filename


def _build_yolo(arm_cfg: dict[str, Any], num_classes: int, data_cfg: dict[str, Any]):
    import torch.nn as nn
    from ultralytics import YOLO

    filename = "%s.pt" % arm_cfg["architecture"]
    resolved = _resolve_pretrained(filename, data_cfg)
    try:
        net = YOLO(resolved).model
    except Exception as exc:  # noqa: BLE001
        fallback = arm_cfg.get("pretrained_fallback")
        if not fallback:
            raise MissingInputError(
                "Could not load pretrained weights %r for arm %r: %s"
                % (resolved, arm_cfg["architecture"], exc)
            ) from exc
        resolved = _resolve_pretrained(fallback, data_cfg)
        net = YOLO(resolved).model

    head = net.model[-1]
    if not hasattr(head, "linear"):
        raise RuntimeError(
            "Unexpected ultralytics classification head %r: no .linear to replace"
            % type(head).__name__
        )
    in_features = head.linear.in_features
    head.linear = nn.Linear(in_features, num_classes)
    net.nc = num_classes
    net.names = {i: str(i) for i in range(num_classes)}
    return net, {"pretrained_source": resolved, "head_in_features": in_features}


def _build_torchvision(arm_cfg: dict[str, Any], num_classes: int, pretrained: bool):
    import torch.nn as nn
    import torchvision.models as tvm

    architecture = arm_cfg["architecture"]
    if architecture == "mobilenet_v3_small":
        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.mobilenet_v3_small(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    elif architecture == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.resnet18(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unknown torchvision architecture %r" % architecture)
    return net, {
        "pretrained_source": str(weights) if pretrained else "random-init",
        "head_in_features": in_features,
    }


def build_model(
    arm: str,
    arms_cfg: dict[str, Any] | None = None,
    data_cfg: dict[str, Any] | None = None,
    *,
    num_classes: int | None = None,
    pretrained: bool = True,
    image_size: int | None = None,
    with_efficiency: bool = True,
    latency: bool = False,
    seed: int | None = None,
) -> ModelBundle:
    """The single entry point. Same call, same return shape, for all five arms.

    Pass `seed` whenever the run must be reproducible. The 10-output head is
    randomly initialised HERE, before any training code runs, so seeding only
    inside the training loop leaves the head initialisation unseeded and two
    runs at the same run_seed diverge from the first step.
    """
    arms_cfg = arms_cfg or load_arms_config()
    data_cfg = data_cfg or load_data_config()
    if seed is not None:
        set_seed(seed)

    if arm not in arms_cfg["arms"]:
        raise ValueError(
            "Unknown arm %r. configs/arms.yaml defines: %s"
            % (arm, sorted(arms_cfg["arms"]))
        )
    arm_cfg = arms_cfg["arms"][arm]
    shared = arms_cfg["shared"]
    num_classes = num_classes or int(shared["num_classes"])
    image_size = image_size or int(shared["image_size"])

    backend = arm_cfg["backend"]
    if backend == "ultralytics":
        net, notes = _build_yolo(arm_cfg, num_classes, data_cfg)
    elif backend == "torchvision":
        net, notes = _build_torchvision(arm_cfg, num_classes, pretrained)
    else:
        raise ValueError("Unknown backend %r for arm %r" % (backend, arm))

    module, wrapper_notes = _make_logits_wrapper(net, image_size)
    notes.update(wrapper_notes)
    notes["num_classes"] = num_classes

    notes["init_seed"] = seed
    bundle = ModelBundle(
        arm=arm,
        architecture=arm_cfg["architecture"],
        backend=backend,
        module=module,
        notes=notes,
    )
    if with_efficiency:
        bundle.efficiency = profile(module, image_size, latency=latency)
    return bundle


def resolve_arm_hyperparameters(arm: str, arms_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """epochs / batch / lr for an arm, refusing any lr still marked TBD."""
    arms_cfg = arms_cfg or load_arms_config()
    arm_cfg = arms_cfg["arms"][arm]
    if arm_cfg.get("lr") is None:
        raise MissingInputError(
            "Arm %r has no learning rate: configs/arms.yaml says lr_source=%r.\n"
            "  Run scripts/02_lr_sweep_baselines.py first; it writes the winner back."
            % (arm, arm_cfg.get("lr_source"))
        )
    return {
        "epochs": int(arm_cfg["epochs"]),
        "batch": int(arm_cfg["batch"]),
        "lr": float(arm_cfg["lr"]),
        "optimizer": arm_cfg.get("optimizer", "SGD"),
        "architecture": arm_cfg["architecture"],
        "backend": arm_cfg["backend"],
    }
