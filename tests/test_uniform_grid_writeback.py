"""01b writes three arms' locked configurations back into configs/arms.yaml.

Every test drives `write_back_winner` against a tmp copy; the repository's own
configs/arms.yaml is never opened for writing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def grid():
    """Import scripts/01b_uniform_grid.py -- the leading digit blocks `import`."""
    spec = importlib.util.spec_from_file_location(
        "uniform_grid", REPO_ROOT / "scripts" / "01b_uniform_grid.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["uniform_grid"] = module
    spec.loader.exec_module(module)
    return module


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["arms"]


def test_writes_one_arm_and_leaves_the_others(grid, arms_file):
    before = load(arms_file)
    grid.write_back_winner("yolo26s", 25, 32, 0.001, "s_ep25_bs32_lr1e-03", 0.81,
                           path=arms_file)
    after = load(arms_file)

    assert (after["yolo26s"]["epochs"], after["yolo26s"]["batch"],
            after["yolo26s"]["lr"]) == (25, 32, 0.001)
    for arm in ("yolo26n", "yolo26m", "mobilenetv3_small", "resnet18"):
        assert after[arm] == before[arm], "%s was modified" % arm


def test_all_three_arms_in_sequence(grid, arms_file):
    wanted = {
        "yolo26n": (25, 8, 0.001),
        "yolo26s": (50, 32, 0.01),
        "yolo26m": (25, 16, 0.0001),
    }
    for arm, (epochs, batch, lr) in wanted.items():
        grid.write_back_winner(arm, epochs, batch, lr, "%s_key" % arm, 0.7, path=arms_file)

    after = load(arms_file)
    for arm, (epochs, batch, lr) in wanted.items():
        assert (after[arm]["epochs"], after[arm]["batch"], after[arm]["lr"]) == (
            epochs, batch, lr
        ), arm
        assert after[arm]["locked"] is True


def test_lr_source_names_the_grid_and_the_protocol(grid, arms_file):
    """Two grids now exist for these arms; a locked value that does not say
    which one produced it is the ambiguity 01b was written to remove."""
    grid.write_back_winner("yolo26n", 50, 16, 0.01, "n_ep50_bs16_lr1e-02", 0.8364,
                           path=arms_file)
    source = load(arms_file)["yolo26n"]["lr_source"]
    assert "uniform_grid:01b" in source
    assert "UNIFORM protocol" in source
    assert "no augmentation" in source
    assert "class weights applied" in source
    assert "n_ep50_bs16_lr1e-02" in source
    assert "0.8364" in source


def test_stale_provisional_commentary_is_dropped(grid, arms_file):
    """The yolo26m block carries a comment describing a state two grids ago."""
    grid.write_back_winner("yolo26m", 50, 8, 0.01, "m_ep50_bs8_lr1e-02", 0.77,
                           path=arms_file)
    text = arms_file.read_text(encoding="utf-8")
    block = text[text.index("  yolo26m:"):text.index("  mobilenetv3_small:")]
    assert "PROVISIONAL" not in block
    assert load(arms_file)["yolo26m"]["provisional"] is False


def test_the_file_still_parses_and_keeps_every_arm(grid, arms_file):
    for arm in ("yolo26n", "yolo26s", "yolo26m"):
        grid.write_back_winner(arm, 25, 8, 0.01, "k", 0.5, path=arms_file)
    parsed = yaml.safe_load(arms_file.read_text(encoding="utf-8"))
    assert set(parsed["arms"]) == {
        "yolo26n", "yolo26s", "yolo26m", "mobilenetv3_small", "resnet18",
    }
    # the sections after the arms block must survive untouched
    for key in ("shared", "uniform_protocol", "medium_grid", "learning_curve"):
        assert key in parsed
    assert parsed["uniform_protocol"]["augmentation"] == "none"


def test_baseline_arms_are_not_writable_by_this_script(grid, arms_file):
    assert grid.YOLO_ARMS == ["yolo26n", "yolo26s", "yolo26m"]
    assert "mobilenetv3_small" not in grid.YOLO_ARMS
    assert "resnet18" not in grid.YOLO_ARMS


def test_run_ids_differ_from_script_01(grid):
    """01 and 01b must never collide: same arm, same hyperparameters, different
    protocol, so the records have to be distinguishable by run_id."""
    from srpcard import registry

    common = dict(
        arm="yolo26m", architecture="yolo26m-cls", split_kind="dev",
        repeat=None, fold=None, epochs=25, batch=8, lr=0.01,
        class_weights="none_legacy_bug", run_seed=42,
    )
    legacy = registry.compute_run_id(
        script="01_complete_medium_grid", extra="legacy_protocol", **common
    )
    uniform = registry.compute_run_id(
        script=grid.SCRIPT, extra="uniform_grid", **common
    )
    assert legacy != uniform


def test_projection_scales_with_model_and_epochs(grid):
    assert grid.project_seconds("yolo26m", 50) > grid.project_seconds("yolo26m", 25)
    assert grid.project_seconds("yolo26m", 50) > grid.project_seconds("yolo26n", 50)
    total = sum(
        grid.project_seconds(arm, epochs)
        for arm in grid.YOLO_ARMS
        for epochs in (25, 50)
        for _ in range(9)
    )
    assert 3600 < total < 3 * 3600, "projection outside the budget stated to the user"
