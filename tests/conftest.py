"""Fixtures that keep every test off the real artifacts/ directory.

A test in this repository must never write inside `artifacts/`. Several files
there are tracked -- `folds.json`, `image_index.csv`, `dev_split.json`,
`registry.jsonl`, `resolved_arms.yaml` -- and an earlier ad-hoc test that used
save-and-restore around the real paths deleted a tracked snapshot twice.
Save-and-restore is not enough: a test that fails partway through never restores.

So the isolation is structural rather than disciplined. Every fixture below hands
back a path under pytest's `tmp_path`, and nothing here resolves to the
repository's own artifacts directory.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    """A throwaway artifacts directory."""
    path = tmp_path / "artifacts"
    path.mkdir()
    return path


@pytest.fixture
def data_cfg(artifacts: Path) -> dict:
    """A data config whose artifacts_dir points into tmp_path.

    `config.artifacts_dir()` honours an absolute `artifacts_dir`, so passing this
    cfg to anything that writes artefacts redirects it wholesale.
    """
    return {"artifacts_dir": str(artifacts), "runs_dir": str(artifacts / "runs")}


@pytest.fixture
def registry_path(artifacts: Path) -> Path:
    """An empty registry file to append to."""
    path = artifacts / "registry.jsonl"
    path.write_text("", encoding="utf-8")
    return path


@pytest.fixture
def arms_file(tmp_path: Path) -> Path:
    """A copy of the real configs/arms.yaml, in tmp_path and safe to rewrite."""
    target = tmp_path / "arms.yaml"
    shutil.copy2(REPO_ROOT / "configs" / "arms.yaml", target)
    return target


@pytest.fixture(autouse=True)
def _guard_real_artifacts():
    """Fail any test that leaves a new file in the repository's artifacts/.

    A backstop for the fixtures above: if a test ever writes through to the real
    directory, it is caught here rather than in `git status` days later.
    """
    real = REPO_ROOT / "artifacts"
    before = {p.name for p in real.iterdir()} if real.is_dir() else set()
    yield
    after = {p.name for p in real.iterdir()} if real.is_dir() else set()
    created, removed = after - before, before - after
    assert not created, "test created file(s) in the real artifacts/: %s" % sorted(created)
    assert not removed, "test removed file(s) from the real artifacts/: %s" % sorted(removed)
