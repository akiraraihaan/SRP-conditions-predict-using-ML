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


def _artifact_state(root: Path) -> dict[str, tuple[int, int]]:
    """Every file under artifacts/, by size and mtime. Subdirectories included."""
    if not root.is_dir():
        return {}
    state = {}
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            state[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return state


@pytest.fixture(autouse=True)
def _guard_real_artifacts():
    """Fail any test that touches the repository's artifacts/ at all.

    A backstop for the fixtures above: if a test ever writes through to the real
    directory, it is caught here rather than in `git status` days later.

    It used to compare only the top-level FILENAMES, which missed two whole
    categories. A test that OVERWRITES an existing file changed no name, and a
    test that wrote into artifacts/figures/ changed nothing at the top level at
    all -- so `test_script06_needs_no_torch_and_no_dataset`, which runs the real
    script 06 in a subprocess, quietly regenerated all ten tracked figures on
    every test run and the churn only showed up in `git status`. Size and mtime
    of every file, recursively, catches both.
    """
    real = REPO_ROOT / "artifacts"
    before = _artifact_state(real)
    yield
    after = _artifact_state(real)

    created = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    assert not created, "test created file(s) in the real artifacts/: %s" % created
    assert not removed, "test removed file(s) from the real artifacts/: %s" % removed
    assert not changed, "test MODIFIED file(s) in the real artifacts/: %s" % changed
