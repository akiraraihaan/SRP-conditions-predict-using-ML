"""Cell 5 of notebooks/colab_runner.ipynb -- the artifacts/ -> Drive symlink.

That cell deletes and recreates a directory holding tracked artefacts, and its
copying is deliberately asymmetric: frozen inputs come from the clone, while
`registry.jsonl` must come from Drive or a fresh clone's empty committed copy
would erase every completed run.

The cell's source is read out of the notebook and executed against tmp_path
directories, so the test exercises the real code without a Drive or a clone.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "colab_runner.ipynb"

FROZEN = [
    "image_index.csv", "folds.json", "dev_split.json", "excluded_images.csv",
    "folds_report.md", "legacy_contamination.json", "legacy_grid_metrics.csv",
    "legacy_test_predictions.csv",
]


@pytest.fixture(scope="module")
def cell_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "os.symlink" in source:
            return source
    pytest.fail("no symlink cell found in %s" % NOTEBOOK)


def make_clone(root: Path) -> Path:
    """A fresh clone: frozen artefacts present, registry.jsonl committed EMPTY."""
    work = root / "clone"
    (work / "artifacts").mkdir(parents=True)
    for name in FROZEN:
        (work / "artifacts" / name).write_text("committed:" + name, encoding="utf-8")
    (work / "artifacts" / "registry.jsonl").write_text("", encoding="utf-8")
    return work


def run_cell(source: str, work: Path, drive: Path) -> None:
    exec(compile(source, "<colab-cell-5>", "exec"),
         {"WORK": str(work), "ARTIFACTS_DRIVE": str(drive)})


@pytest.fixture
def drive(tmp_path: Path) -> Path:
    return tmp_path / "drive" / "srp-artifacts"


def test_symlink_is_created_and_frozen_inputs_survive(cell_source, tmp_path, drive):
    work = make_clone(tmp_path)
    run_cell(cell_source, work, drive)

    artifacts = work / "artifacts"
    assert artifacts.is_symlink()
    assert artifacts.resolve() == drive.resolve()
    for name in FROZEN:
        assert (artifacts / name).read_text(encoding="utf-8") == "committed:" + name


def test_registry_survives_a_dropped_session(cell_source, tmp_path, drive):
    """The failure this guards: a fresh clone's EMPTY registry overwriting Drive's."""
    work = make_clone(tmp_path)
    run_cell(cell_source, work, drive)

    with open(work / "artifacts" / "registry.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"run_id": "aaa"}) + "\n")
        fh.write(json.dumps({"run_id": "bbb"}) + "\n")
    (work / "artifacts" / "summary_cv.csv").write_text("generated", encoding="utf-8")

    shutil.rmtree(work)                      # the session dies
    work = make_clone(tmp_path)              # a brand-new clone next session
    run_cell(cell_source, work, drive)

    lines = [
        line for line in
        (work / "artifacts" / "registry.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2, "the clone's empty registry clobbered Drive's"
    assert (work / "artifacts" / "summary_cv.csv").exists()


def test_rerunning_the_cell_is_idempotent(cell_source, tmp_path, drive):
    work = make_clone(tmp_path)
    run_cell(cell_source, work, drive)
    with open(work / "artifacts" / "registry.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"run_id": "aaa"}) + "\n")

    run_cell(cell_source, work, drive)       # same session, cell run twice

    assert (work / "artifacts").is_symlink()
    lines = [
        line for line in
        (work / "artifacts" / "registry.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1, "re-running the cell destroyed the registry"


def test_drive_is_created_when_absent(cell_source, tmp_path, drive):
    assert not drive.exists()
    run_cell(cell_source, make_clone(tmp_path), drive)
    assert drive.is_dir()
    assert (drive / "registry.jsonl").exists()


def test_clone_only_records_are_merged_into_drive(cell_source, tmp_path, drive):
    """The divergence that actually happened: the repository held a backfill that
    never reached Drive, while Drive held runs that were never committed.
    Seeding one-directionally loses one side; merging keeps both."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from srpcard import registry

    work = make_clone(tmp_path)
    # Drive: two runs the repository has never seen
    drive.mkdir(parents=True)
    (drive / "registry.jsonl").write_text(
        "".join(json.dumps({"run_id": "drive%d" % i, "script": "03_run_cv",
                            "arm": "yolo26n", "f1_macro": 0.5}) + "\n" for i in range(2)),
        encoding="utf-8",
    )
    # the clone: a committed record Drive lacks
    (work / "artifacts" / "registry.jsonl").write_text(
        json.dumps({"run_id": "committed0", "script": "01b_uniform_grid",
                    "arm": "yolo26m", "f1_macro": 0.6, "gpu": "Tesla T4"}) + "\n",
        encoding="utf-8",
    )

    run_cell(cell_source, work, drive)

    ids = {r["run_id"] for r in registry.load_registry(work / "artifacts" / "registry.jsonl")}
    assert ids == {"drive0", "drive1", "committed0"}, (
        "the merge lost records: %s" % sorted(ids)
    )


def test_a_conflicting_registry_stops_the_session(cell_source, tmp_path, drive):
    """Two records sharing a run_id but disagreeing on a measured value must
    abort rather than silently discard one."""
    work = make_clone(tmp_path)
    drive.mkdir(parents=True)
    (drive / "registry.jsonl").write_text(
        json.dumps({"run_id": "same", "script": "03_run_cv", "arm": "yolo26n",
                    "f1_macro": 0.50}) + "\n", encoding="utf-8")
    (work / "artifacts" / "registry.jsonl").write_text(
        json.dumps({"run_id": "same", "script": "03_run_cv", "arm": "yolo26n",
                    "f1_macro": 0.87}) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        run_cell(cell_source, work, drive)
