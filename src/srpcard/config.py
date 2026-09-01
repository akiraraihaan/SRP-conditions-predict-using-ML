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
        hint = f"  Produce it with: {produced_by}" if produced_by else ""
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
