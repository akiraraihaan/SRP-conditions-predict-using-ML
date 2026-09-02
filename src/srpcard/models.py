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
3. `pretrained_fallback` in configs/arms.yaml names a YOLO11 checkpoint. Taking
   it SILENTLY would change the architecture the paper reports. It is therefore
   refused unless the caller passes allow_pretrained_fallback=True (the
   `--allow-pretrained-fallback` flag on scripts 01-05), and every bundle
   carries `checkpoint_resolved` and `pretrained_fallback_used` so the registry
   records what was actually loaded rather than what was requested.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, MissingInputError, load_arms_config, load_data_config, set_seed
from .efficiency import profile


class ArchitectureMismatchError(RuntimeError):
    """The checkpoint actually loaded is not the arm's declared architecture.

    Raised rather than warned about: a run that trains YOLO11 while every table,
    figure and registry record says YOLO26 is not a degraded result, it is a
    wrong one, and nothing downstream could detect it.
    """


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

    @property
    def checkpoint_resolved(self) -> str:
        """The checkpoint FILENAME actually loaded, for the registry record."""
        return str(self.notes.get("checkpoint_resolved", "unknown"))

    @property
    def pretrained_fallback_used(self) -> bool:
        """True when a checkpoint other than the declared architecture was used."""
        return bool(self.notes.get("pretrained_fallback_used", False))


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


def checkpoint_stem(name_or_path: str) -> str:
    """'weights/yolo26n-cls.pt' -> 'yolo26n-cls'. The name the assert compares."""
    return Path(str(name_or_path)).stem


def assert_checkpoint_matches_architecture(arm: str, architecture: str, resolved: str) -> None:
    """Refuse to train when the checkpoint on disk is not the declared architecture.

    The failure this exists to catch: `pretrained_fallback: yolo11n-cls.pt` gets
    taken because the YOLO26 download failed, the run completes, and every table
    in the paper then says YOLO26 while the weights were YOLO11.
    """
    actual = checkpoint_stem(resolved)
    if actual != architecture:
        raise ArchitectureMismatchError(
            "Checkpoint / architecture mismatch for arm %r -- refusing to train.\n"
            "  declared architecture (configs/arms.yaml) : %s\n"
            "  checkpoint actually resolved              : %s\n"
            "  full path                                 : %s\n"
            "  Training this would silently change the architecture that every\n"
            "  table, figure and registry record attributes to this arm."
            % (arm, architecture, actual, resolved)
        )


def _build_yolo(
    arm: str,
    arm_cfg: dict[str, Any],
    num_classes: int,
    data_cfg: dict[str, Any],
    *,
    allow_pretrained_fallback: bool = False,
):
    import torch.nn as nn
    from ultralytics import YOLO

    architecture = arm_cfg["architecture"]
    filename = "%s.pt" % architecture
    resolved = _resolve_pretrained(filename, data_cfg)
    fallback_used = False

    try:
        net = YOLO(resolved).model
    except Exception as exc:  # noqa: BLE001
        fallback = arm_cfg.get("pretrained_fallback")
        if not fallback:
            raise MissingInputError(
                "Could not load pretrained weights %r for arm %r: %s"
                % (resolved, architecture, exc)
            ) from exc
        if not allow_pretrained_fallback:
            raise ArchitectureMismatchError(
                "Pretrained checkpoint for arm %r could not be loaded, and the\n"
                "configured fallback is a DIFFERENT ARCHITECTURE. Refusing.\n"
                "  declared architecture : %s\n"
                "  fallback checkpoint   : %s\n"
                "  underlying error      : %s\n"
                "\n"
                "  Taking the fallback would change the architecture the paper\n"
                "  reports without anything downstream noticing. Either:\n"
                "    - make %s available (SRPCARD_WEIGHTS_DIR, configs/data.yaml\n"
                "      weights_dir, or internet access for the download), or\n"
                "    - re-run with --allow-pretrained-fallback if you have decided\n"
                "      to accept %s as the architecture, in which case every\n"
                "      affected registry record is flagged pretrained_fallback_used."
                % (arm, architecture, fallback, exc, filename, fallback)
            ) from exc

        resolved = _resolve_pretrained(fallback, data_cfg)
        net = YOLO(resolved).model
        fallback_used = True
        print(warn_fallback_banner(arm, architecture, resolved))
    else:
        assert_checkpoint_matches_architecture(arm, architecture, resolved)

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
    return net, {
        "pretrained_source": resolved,
        "checkpoint_resolved": Path(resolved).name,
        "pretrained_fallback_used": fallback_used,
        "head_in_features": in_features,
    }


def warn_fallback_banner(arm: str, architecture: str, resolved: str) -> str:
    """The loud warning printed whenever --allow-pretrained-fallback is exercised."""
    bar = "!" * 74
    return "\n".join(
        [
            "",
            bar,
            "PRETRAINED FALLBACK TAKEN -- THE ARCHITECTURE IS NOT WHAT THE ARM DECLARES",
            "  arm                   : %s" % arm,
            "  declared architecture : %s" % architecture,
            "  checkpoint loaded     : %s" % resolved,
            "  Permitted only because --allow-pretrained-fallback was passed.",
            "  Every run from here carries pretrained_fallback_used=true in the",
            "  registry. Do NOT report these runs as %s." % architecture,
            bar,
            "",
        ]
    )


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
    # torchvision resolves weights by enum, not by filename: there is no fallback
    # path here, and so nothing that could substitute a different architecture.
    source = str(weights) if pretrained else "random-init"
    return net, {
        "pretrained_source": source,
        "checkpoint_resolved": source,
        "pretrained_fallback_used": False,
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
    allow_pretrained_fallback: bool = False,
) -> ModelBundle:
    """The single entry point. Same call, same return shape, for all five arms.

    Pass `seed` whenever the run must be reproducible. The 10-output head is
    randomly initialised HERE, before any training code runs, so seeding only
    inside the training loop leaves the head initialisation unseeded and two
    runs at the same run_seed diverge from the first step.

    `allow_pretrained_fallback` is OFF by default and must stay that way for any
    run that is reported. With it off, a YOLO arm whose declared checkpoint
    cannot be loaded raises ArchitectureMismatchError instead of quietly
    training the YOLO11 checkpoint named in `pretrained_fallback`.
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
        net, notes = _build_yolo(
            arm,
            arm_cfg,
            num_classes,
            data_cfg,
            allow_pretrained_fallback=allow_pretrained_fallback,
        )
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


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------


FALLBACK_FLAG = "--allow-pretrained-fallback"


def add_fallback_argument(parser) -> None:
    """The one flag that permits a pretrained fallback. Same wording everywhere."""
    parser.add_argument(
        "--allow-pretrained-fallback",
        action="store_true",
        help=(
            "permit an arm to load configs/arms.yaml:pretrained_fallback when its "
            "declared checkpoint cannot be downloaded. OFF by default: taking it "
            "changes the architecture. Every affected run is flagged "
            "pretrained_fallback_used in the registry."
        ),
    )


def _torchvision_weights_enum(architecture: str):
    import torchvision.models as tvm

    if architecture == "mobilenet_v3_small":
        return tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    if architecture == "resnet18":
        return tvm.ResNet18_Weights.IMAGENET1K_V1
    raise ValueError("Unknown torchvision architecture %r" % architecture)


def preflight_pretrained(
    arms_cfg: dict[str, Any] | None = None,
    data_cfg: dict[str, Any] | None = None,
    arms: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve -- and actually LOAD -- the pretrained checkpoint of every arm.

    Called by scripts/00_build_folds.py so a missing YOLO26 checkpoint surfaces in
    the first minute of a Kaggle session rather than 40 runs in. Never raises:
    each arm gets a record with status "ok" or "FAILED", and the caller decides.

    Loading, not merely resolving, is the point: on Kaggle the checkpoint does not
    exist locally and ultralytics downloads it, so only a real load proves the
    download works.
    """
    arms_cfg = arms_cfg or load_arms_config()
    data_cfg = data_cfg or load_data_config()
    names = arms or sorted(arms_cfg["arms"])
    results: list[dict[str, Any]] = []

    for arm in names:
        arm_cfg = arms_cfg["arms"][arm]
        architecture = arm_cfg["architecture"]
        backend = arm_cfg["backend"]
        record: dict[str, Any] = {
            "arm": arm,
            "architecture": architecture,
            "backend": backend,
            "declared_fallback": arm_cfg.get("pretrained_fallback"),
            "status": "FAILED",
            "acquisition": "-",
            "checkpoint_resolved": None,
            "error": None,
        }

        if backend == "ultralytics":
            filename = "%s.pt" % architecture
            record["expected"] = filename
            try:
                resolved = _resolve_pretrained(filename, data_cfg)
                present_before = Path(resolved).exists()
                from ultralytics import YOLO

                YOLO(resolved)
                record["acquisition"] = "local" if present_before else "downloaded"
                record["checkpoint_resolved"] = Path(resolved).name
                assert_checkpoint_matches_architecture(arm, architecture, resolved)
                record["status"] = "ok"
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                record["error"] = "%s: %s" % (type(exc).__name__, exc)
        elif backend == "torchvision":
            try:
                import os as _os

                import torch

                weights = _torchvision_weights_enum(architecture)
                record["expected"] = str(weights)
                cached = (
                    Path(torch.hub.get_dir())
                    / "checkpoints"
                    / _os.path.basename(weights.url)
                )
                present_before = cached.exists()
                weights.get_state_dict(progress=False)
                record["acquisition"] = "local" if present_before else "downloaded"
                record["checkpoint_resolved"] = str(weights)
                record["status"] = "ok"
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                record["error"] = "%s: %s" % (type(exc).__name__, exc)
        else:
            record["expected"] = "-"
            record["error"] = "unknown backend %r" % backend

        results.append(record)
    return results


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
