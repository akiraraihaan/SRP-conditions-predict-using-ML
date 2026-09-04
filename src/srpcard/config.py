"""Paths, configuration loading, seeding and run provenance.

Every path in this repository resolves through here. Nothing is hardcoded to
/kaggle or to a local directory: `data_root` comes from configs/data.yaml and
may be overridden by the SRPCARD_DATA_ROOT environment variable.

This module is an addition to the module list in the brief. It exists because
section 7 requires one `set_seed` helper and one place that resolves paths, and
neither belongs in data.py, folds.py or train.py.

Import has no side effects.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# repo root = .../extra-deep  (this file is src/srpcard/config.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"

DATA_ROOT_ENV = "SRPCARD_DATA_ROOT"


# --------------------------------------------------------------------------
# failing loudly, with the name of the file
# --------------------------------------------------------------------------


class MissingInputError(FileNotFoundError):
    """A required input is absent. The message always names the file."""


def require_file(path: Path, produced_by: str = "") -> Path:
    """Return `path`, or raise naming the file and what would have produced it."""
    path = Path(path)
    if not path.exists():
        hint = ("\n  Produce it with: " + produced_by) if produced_by else ""
        raise MissingInputError(f"Required input not found: {path}{hint}")
    return path


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    path = require_file(Path(path))
    with open(path, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not parse to a mapping (got {type(loaded).__name__})")
    return loaded


def load_data_config() -> dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "data.yaml")


def load_folds_config() -> dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "folds.yaml")


def load_arms_config() -> dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "arms.yaml")


def artifacts_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_data_config()
    path = Path(cfg.get("artifacts_dir", "artifacts"))
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def runs_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_data_config()
    path = Path(cfg.get("runs_dir", "runs"))
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_data_root(cfg: dict[str, Any] | None = None, *, verbose: bool = True) -> Path:
    """Resolve DATA_ROOT, descending through a single wrapper directory if needed.

    Precedence: $SRPCARD_DATA_ROOT, then configs/data.yaml:data_root.

    Kaggle mounts this dataset as /kaggle/input/srp-dyna-card/dataset -- note the
    dataset/ wrapper. If the resolved directory contains exactly one subdirectory
    and that subdirectory is the one actually holding the class directories, we
    descend into it and say so, rather than failing on a path that is only one
    level off.
    """
    cfg = cfg or load_data_config()
    env_value = os.environ.get(DATA_ROOT_ENV)
    root = Path(env_value) if env_value else Path(cfg["data_root"])
    source = f"${DATA_ROOT_ENV}" if env_value else "configs/data.yaml:data_root"

    if not root.exists():
        raise MissingInputError(
            f"DATA_ROOT does not exist: {root}  (from {source})\n"
            f"  On Kaggle, attach the dataset raihanakirar/srp-dyna-card and use\n"
            f"    {cfg['data_root']}\n"
            f"  Locally, set {DATA_ROOT_ENV} to the directory holding the 10 class "
            f"directories.\n"
            f"  See docs/KAGGLE_SETUP.md."
        )

    if cfg.get("descend_single_wrapper", True):
        subdirs = sorted(d for d in root.iterdir() if d.is_dir())
        if len(subdirs) == 1:
            inner = sorted(d for d in subdirs[0].iterdir() if d.is_dir())
            if len(inner) > 1:
                if verbose:
                    print(
                        f"[data_root] descended through single wrapper directory "
                        f"'{subdirs[0].name}/' ({len(inner)} class directories inside)"
                    )
                root = subdirs[0]

    if verbose:
        print(f"[data_root] {root}  (from {source})")
    return root


# --------------------------------------------------------------------------
# the resolved-hyperparameter snapshot
# --------------------------------------------------------------------------
#
# configs/arms.yaml is rewritten by scripts 01 and 02 with the hyperparameters
# they select. On Colab that file lives in the clone and dies with the session,
# while artifacts/ is a symlink into Drive -- so the snapshot below is written
# there and survives, even if the session ends before arms.yaml is committed.
#
# This matters more than convenience. epochs, batch and lr feed the run_id hash
# (registry.RUN_ID_FIELDS). A clone carrying the PROVISIONAL yolo26m config after
# a dropped session computes DIFFERENT run_ids, so nothing is skipped, all 15
# folds are retrained under the old settings, and the registry ends up holding
# two hyperparameter regimes for one arm. See registry.assert_config_matches_registry.

ARMS_SNAPSHOT_NAME = "resolved_arms.yaml"

# The fields that feed the run_id hash, and therefore the ones a drift is
# dangerous in rather than merely untidy.
RUN_DEFINING_HYPERPARAMETERS = ("epochs", "batch", "lr")

SNAPSHOT_MARKER = "# ==== BEGIN VERBATIM COPY OF configs/arms.yaml ===="


def arms_path() -> Path:
    return CONFIGS_DIR / "arms.yaml"


def resolved_arms_path(cfg: dict[str, Any] | None = None) -> Path:
    return artifacts_dir(cfg) / ARMS_SNAPSHOT_NAME


def snapshot_arms(resolved_by: str, cfg: dict[str, Any] | None = None) -> Path:
    """Write artifacts/resolved_arms.yaml: a header, then arms.yaml verbatim.

    Called by scripts 01 and 02 immediately after they rewrite arms.yaml. The
    header is YAML comments, so the snapshot parses as the same document the
    original does and can be compared with it by loading both.
    """
    from datetime import datetime, timezone

    source = arms_path()
    target = resolved_arms_path(cfg)
    header = [
        "# artifacts/%s -- GENERATED SNAPSHOT. Do not edit by hand." % ARMS_SNAPSHOT_NAME,
        "#",
        "# Written by  : %s" % resolved_by,
        "# At          : %s" % datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "# Git commit  : %s" % git_commit(),
        "#",
        "# A full snapshot of configs/arms.yaml as it stood after the resolving script",
        "# wrote its results back. configs/arms.yaml lives in the clone and dies with",
        "# the session; artifacts/ is the Drive symlink, so this copy survives.",
        "#",
        "# Restore it over configs/arms.yaml in a fresh clone with:",
        "#     python scripts/restore_arms.py",
        "#",
        SNAPSHOT_MARKER,
    ]
    body = source.read_text(encoding="utf-8")
    target.write_text("\n".join(header) + "\n" + body, encoding="utf-8", newline="\n")
    return target


def snapshot_body(path: Path) -> str:
    """The verbatim arms.yaml text inside a snapshot, header stripped."""
    text = Path(path).read_text(encoding="utf-8")
    if SNAPSHOT_MARKER in text:
        return text.split(SNAPSHOT_MARKER, 1)[1].lstrip("\n")
    return text


def arm_hyperparameters(arms_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """{arm: {epochs, batch, lr}} -- just the run-id-defining fields."""
    return {
        name: {field: arm.get(field) for field in RUN_DEFINING_HYPERPARAMETERS}
        for name, arm in (arms_cfg.get("arms") or {}).items()
    }


def compare_arms_configs(
    left: dict[str, Any], right: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per-arm differences in epochs/batch/lr between two loaded arms configs."""
    left_hp, right_hp = arm_hyperparameters(left), arm_hyperparameters(right)
    differences = []
    for name in sorted(set(left_hp) | set(right_hp)):
        a, b = left_hp.get(name), right_hp.get(name)
        if a != b:
            differences.append({"arm": name, "left": a, "right": b})
    return differences


def unresolved_arms(arms_cfg: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Arms whose hyperparameters are not settled yet.

    `lr_null`   -- 02_lr_sweep_baselines.py has not run, or its result was lost.
    `provisional` -- 01_complete_medium_grid.py has not run, or its result was lost.
    """
    arms_cfg = arms_cfg or load_arms_config()
    null_lr, provisional = [], []
    for name, arm in (arms_cfg.get("arms") or {}).items():
        if arm.get("lr") is None:
            null_lr.append(name)
        if arm.get("provisional"):
            provisional.append(name)
    return {"lr_null": sorted(null_lr), "provisional": sorted(provisional)}


def arms_snapshot_status(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Does a snapshot exist, and does it still agree with configs/arms.yaml?

    A snapshot that DIFFERS is the signature of a config lost between sessions:
    the snapshot holds what a resolving script decided, and the clone's
    arms.yaml has reverted to what was committed.
    """
    target = resolved_arms_path(cfg)
    status: dict[str, Any] = {"path": str(target), "exists": target.exists()}
    if not target.exists():
        return status

    live = load_yaml(arms_path())
    snapshot = load_yaml(target)
    # left = what configs/arms.yaml says NOW, right = what the snapshot resolved to.
    # Callers print them in that order; do not swap them.
    differences = compare_arms_configs(live, snapshot)
    status["differs"] = bool(differences)
    status["differences"] = differences
    status["resolved_by"] = None
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.startswith("# Written by"):
            status["resolved_by"] = line.split(":", 1)[1].strip()
        if line.startswith("# At"):
            status["written_at"] = line.split(":", 1)[1].strip()
        if line.startswith(SNAPSHOT_MARKER):
            break
    return status


# --------------------------------------------------------------------------
# seeding
# --------------------------------------------------------------------------


def run_seed_for(fold_cfg: dict[str, Any], repeat: int, fold: int) -> int:
    """run_seed = run_base + repeat*100 + fold. A pure function of (repeat, fold).

    Identical across every arm, which is what makes comparisons paired.
    """
    return int(fold_cfg["seeds"]["run_base"]) + repeat * 100 + fold


def val_seed_for(fold_cfg: dict[str, Any], repeat: int, fold: int) -> int:
    """val_seed = run_seed + val_slice_offset. Also identical across every arm."""
    return run_seed_for(fold_cfg, repeat, fold) + int(fold_cfg["seeds"]["val_slice_offset"])


def set_seed(seed: int) -> dict[str, Any]:
    """Seed random, numpy, torch, torch.cuda and PYTHONHASHSEED.

    Also forces cuDNN into deterministic mode: benchmark mode picks convolution
    algorithms by timing them, which is nondeterministic. The ~10-15% slowdown is
    accepted.

    Ultralytics may not honour torch.use_deterministic_algorithms, so the achieved
    status is RETURNED for the registry to record rather than hard-failing.
    """
    status: dict[str, Any] = {"seed": int(seed)}

    os.environ["PYTHONHASHSEED"] = str(seed)
    status["pythonhashseed"] = str(seed)
    # required by cuBLAS for deterministic GEMM on CUDA >= 10.2; must be set
    # before the first cuBLAS handle is created
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    status["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
        status["numpy"] = True
    except ImportError:
        status["numpy"] = False

    try:
        import torch
    except ImportError:
        status["torch"] = "unavailable"
        return status

    torch.manual_seed(seed)
    status["torch"] = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        status["cuda"] = True
    else:
        status["cuda"] = False

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    status["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    status["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        status["use_deterministic_algorithms"] = "warn_only"
    except Exception as exc:  # noqa: BLE001 - status is recorded, not raised
        status["use_deterministic_algorithms"] = f"unavailable: {exc}"

    return status


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def git_commit() -> str:
    """Current commit hash, or a marker. Never raises."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return "not-a-git-repo"
        commit = out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "git-unavailable"

    try:
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            return f"{commit}-dirty"
    except Exception:  # noqa: BLE001
        pass
    return commit


def hardware() -> dict[str, Any]:
    """The device a run actually executed on.

    Recorded at the TOP LEVEL of every registry record, not nested inside
    `library_versions`. The GPU name was already captured there, but nothing
    could query it: a mixed-hardware check has to compare one field across
    records, and `library_versions` is a free-form blob.

    Never raises. Every field falls back to None so a CPU run, or a machine
    without nvidia-smi, records what it can rather than failing.
    """
    info: dict[str, Any] = {
        "gpu": None,
        "gpu_count": 0,
        "cuda_version": None,
        "driver_version": None,
        "compute_capability": None,
        "device_kind": "cpu",
    }
    try:
        import torch
    except ImportError:
        return info

    info["cuda_version"] = torch.version.cuda
    if not torch.cuda.is_available():
        return info

    info["device_kind"] = "cuda"
    try:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        major, minor = torch.cuda.get_device_capability(0)
        info["compute_capability"] = "%d.%d" % (major, minor)
    except Exception:  # noqa: BLE001 - provenance never blocks a run
        pass

    # The driver is not exposed by the public torch API in every version, so
    # try the private accessor first and fall back to nvidia-smi.
    try:
        raw = torch._C._cuda_getDriverVersion()  # noqa: SLF001
        info["driver_version"] = "%d.%d" % (raw // 1000, (raw % 1000) // 10)
    except Exception:  # noqa: BLE001
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                info["driver_version"] = out.stdout.strip().splitlines()[0].strip()
        except Exception:  # noqa: BLE001
            pass
    return info


def library_versions() -> dict[str, str]:
    """Versions ACTUALLY installed at runtime, not the pins in requirements.txt."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in (
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "torch",
        "torchvision",
        "ultralytics",
        "PIL",
        "cv2",
        "matplotlib",
        "yaml",
    ):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # noqa: BLE001
            versions[name] = "not-installed"

    try:
        import torch

        versions["torch_cuda"] = str(torch.version.cuda)
        versions["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return versions
