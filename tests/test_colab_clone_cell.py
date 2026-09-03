"""The clone cell must survive being re-run inside one session.

That cell is re-run every time the repository changes, which is this project's
normal workflow. It used to end with `os.chdir(WORK)`, so the second run called
`shutil.rmtree` on the process's own working directory and `git clone` exited 128
with an invalid working directory.

The cells are read out of the notebooks and executed against tmp_path with
`subprocess.run` stubbed -- these tests are about the cell's directory handling,
not about git.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def clone_cell(notebook: str) -> str:
    nb = json.loads((REPO_ROOT / "notebooks" / notebook).read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "shutil.rmtree(WORK)" in source:
            return source
    pytest.fail("no clone cell found in %s" % notebook)


@pytest.fixture(autouse=True)
def _restore_cwd():
    before = os.getcwd()
    yield
    os.chdir(before)


@pytest.fixture
def fake_clone(monkeypatch):
    """Stub subprocess.run so the cell's own `import subprocess` gets it too."""
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        # git resolves its own cwd first: this is precisely what used to fail
        os.getcwd()
        if "clone" in argv:
            target = Path(argv[-1])
            (target / "artifacts").mkdir(parents=True)
            (target / "artifacts" / "registry.jsonl").write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def run_cell(source: str, *, work: Path, anchor: Path) -> None:
    """Execute the cell with its /content (or /kaggle/working) anchor redirected."""
    for literal in ("os.chdir('/content')", "os.chdir('/kaggle/working')"):
        source = source.replace(literal, "os.chdir(ANCHOR)")
    # the kaggle cell assigns WORK itself; drop that so the injected path wins
    source = re.sub(r"^WORK\s*=\s*'[^']*'$", "", source, flags=re.M)
    exec(
        compile(source, "<clone-cell>", "exec"),
        {
            "WORK": str(work),
            "BRANCH": "main",
            "REPO_URL": "https://example.invalid/repo.git",
            "ANCHOR": str(anchor),
        },
    )


@pytest.mark.parametrize(
    "notebook,anchor",
    [("colab_runner.ipynb", "/content"), ("kaggle_runner.ipynb", "/kaggle/working")],
)
def test_clone_cell_leaves_the_tree_before_deleting_it(notebook, anchor):
    """Read straight off the cell: chdir out, then rmtree, then chdir back in."""
    source = clone_cell(notebook)
    chdir_out = source.index("os.chdir('%s')" % anchor)
    rmtree = source.index("shutil.rmtree(WORK)")
    chdir_in = source.index("os.chdir(WORK)")
    assert chdir_out < rmtree, "%s: rmtree runs while still inside the tree" % notebook
    assert rmtree < chdir_in, "%s: unexpected ordering" % notebook


@pytest.mark.parametrize(
    "notebook", ["colab_runner.ipynb", "kaggle_runner.ipynb"]
)
def test_clone_cell_is_rerunnable(notebook, tmp_path, fake_clone):
    """Run the real cell twice, exactly as pushing a change does."""
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    work = anchor / "SRP-conditions-predict-using-ML"
    source = clone_cell(notebook)

    run_cell(source, work=work, anchor=anchor)
    assert work.is_dir()

    # the second run is what used to exit 128 with an invalid working directory
    run_cell(source, work=work, anchor=anchor)
    assert work.is_dir()
    assert sum(1 for argv in fake_clone if "clone" in argv) == 2


def test_clone_cell_warns_that_the_symlink_is_gone(tmp_path, fake_clone, capsys):
    """A fresh clone replaces the Drive symlink with a plain directory, and the
    only symptom otherwise is a completed-run count of 0."""
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    run_cell(clone_cell("colab_runner.ipynb"), work=anchor / "repo", anchor=anchor)
    out = capsys.readouterr().out
    assert "RE-RUN CELL 5" in out
    assert "ephemeral disk" in out


def test_symlink_cell_exposes_a_reusable_check(tmp_path):
    """The run cell calls check_artifacts_symlink before spending a session."""
    nb = json.loads(
        (REPO_ROOT / "notebooks" / "colab_runner.ipynb").read_text(encoding="utf-8")
    )
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    definitions = [s for s in sources if "def check_artifacts_symlink" in s]
    callers = [s for s in sources if "check_artifacts_symlink(" in s]
    assert len(definitions) == 1, "the check must be defined exactly once"
    assert len(callers) >= 2, "the run cell must call it as well as the symlink cell"


def test_the_check_detects_a_plain_directory(tmp_path, capsys):
    """Drive the real function out of the notebook against tmp_path."""
    nb = json.loads(
        (REPO_ROOT / "notebooks" / "colab_runner.ipynb").read_text(encoding="utf-8")
    )
    source = next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and "def check_artifacts_symlink" in "".join(c["source"])
    )
    body = source[source.index("def check_artifacts_symlink"):]
    body = body[: body.index("\ncheck_artifacts_symlink()")]

    work = tmp_path / "repo"
    drive = tmp_path / "drive"
    (work / "artifacts").mkdir(parents=True)
    drive.mkdir()

    namespace = {"os": os, "WORK": str(work), "ARTIFACTS_DRIVE": str(drive)}
    exec(compile(body, "<check>", "exec"), namespace)
    check = namespace["check_artifacts_symlink"]

    assert check() is False
    out = capsys.readouterr().out
    assert "IS NOT THE DRIVE SYMLINK" in out
    assert "a plain directory" in out

    # and true once the symlink is in place
    (work / "artifacts").rmdir()
    os.symlink(drive, work / "artifacts")
    assert check() is True
