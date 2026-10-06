"""Architecture names and in-plot titles on the figures.

The manuscript says MobileNetV3-Small, YOLO26n-cls, ResNet18; the figures said
mobilenetv3_small, yolo26n, resnet18, because the arm key went straight from the
registry onto the plot. And --for-publication removed the provenance strip but
kept every in-plot title, repeating what the caption already says.

Nothing here writes a file: the figures live in memory and are closed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

from srpcard import figures  # noqa: E402

ARM_KEYS = sorted(
    yaml.safe_load((REPO_ROOT / "configs" / "arms.yaml").read_text(encoding="utf-8"))["arms"]
)


@pytest.mark.parametrize("arm", ARM_KEYS)
def test_every_configured_arm_has_a_display_name(arm):
    name = figures.display_name(arm)
    assert "_" not in name, name
    assert name != arm


@pytest.mark.parametrize(
    "arm, flat, wrapped",
    [
        ("mobilenetv3_small", "MobileNetV3-Small", "MobileNetV3-\nSmall"),
        ("resnet18", "ResNet18", "ResNet18"),
        ("yolo26n", "YOLO26n-cls", "YOLO26n-cls"),
        # split before the parenthetical, NOT at the last hyphen, which would
        # give "YOLO26n-\ncls (50 epochs)" and break the architecture name
        ("yolo26n_ep50", "YOLO26n-cls (50 epochs)", "YOLO26n-cls\n(50 epochs)"),
    ],
)
def test_wrapping_is_deterministic(arm, flat, wrapped):
    assert figures.display_name(arm) == flat
    assert figures.display_name(arm, wrap=True) == wrapped


def test_every_wrapped_name_is_the_flat_name_with_one_line_break():
    """Wrapping may move a break in, never drop or change a character."""
    for arm in ARM_KEYS:
        flat = figures.display_name(arm)
        wrapped = figures.display_name(arm, wrap=True)
        assert wrapped.count("\n") <= 1, wrapped
        assert wrapped.replace("-\n", "-").replace("\n", " ") == flat


def test_an_unknown_arm_is_returned_unchanged():
    assert figures.display_name("sesuatu_yang_tak_dikenal") == "sesuatu_yang_tak_dikenal"


@pytest.fixture
def restore_titles():
    before = figures.RENDER_TITLES
    yield
    figures.set_render_titles(before)


def test_titles_can_be_switched_off_and_back_on(restore_titles):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        figures.set_render_titles(False)
        figures._title(ax, "x")
        assert ax.get_title() == ""

        figures.set_render_titles(True)
        figures._title(ax, "x")
        assert ax.get_title() == "x"
    finally:
        plt.close(fig)
