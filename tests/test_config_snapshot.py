"""The resolved-config snapshot and its restore path.

`config.arms_path()` is monkeypatched to a tmp copy throughout, so no test can
write configs/arms.yaml, and `data_cfg` redirects the snapshot into tmp_path.
"""

from __future__ import annotations

import pytest
import yaml

from srpcard import config


@pytest.fixture
def arms(monkeypatch, arms_file):
    """Redirect config.arms_path() at a throwaway copy of arms.yaml."""
    monkeypatch.setattr(config, "arms_path", lambda: arms_file)
    return arms_file


def rewrite_yolo26m(path, *, epochs, batch, lr):
    """Stand in for script 01's write-back."""
    text = path.read_text(encoding="utf-8")
    start, end = text.index("  yolo26m:"), text.index("  mobilenetv3_small:")
    out = []
    for line in text[start:end].splitlines():
        stripped = line.strip()
        if stripped.startswith("epochs:") and line.startswith("    "):
            out.append("    epochs: %d" % epochs)
        elif stripped.startswith("batch:") and line.startswith("    "):
            out.append("    batch: %d" % batch)
        elif stripped.startswith("lr:") and line.startswith("    "):
            out.append("    lr: %r" % lr)
        elif stripped.startswith("provisional:"):
            out.append("    provisional: false")
        else:
            out.append(line)
    path.write_text(text[:start] + "\n".join(out) + "\n" + text[end:], encoding="utf-8")


def test_snapshot_parses_as_the_same_document(arms, data_cfg):
    target = config.snapshot_arms("01_complete_medium_grid", data_cfg)
    assert target.parent == config.artifacts_dir(data_cfg)
    assert yaml.safe_load(target.read_text(encoding="utf-8")) == yaml.safe_load(
        arms.read_text(encoding="utf-8")
    )


def test_snapshot_header_records_provenance(arms, data_cfg):
    text = config.snapshot_arms("02_lr_sweep_baselines", data_cfg).read_text(encoding="utf-8")
    assert "Written by  : 02_lr_sweep_baselines" in text
    assert "Git commit" in text
    assert config.SNAPSHOT_MARKER in text


def test_snapshot_body_round_trips(arms, data_cfg):
    target = config.snapshot_arms("01_complete_medium_grid", data_cfg)
    assert config.snapshot_body(target) == arms.read_text(encoding="utf-8")


def test_status_is_clean_when_nothing_changed(arms, data_cfg):
    config.snapshot_arms("01_complete_medium_grid", data_cfg)
    status = config.arms_snapshot_status(data_cfg)
    assert status["exists"] and not status["differs"]


def test_status_detects_a_lost_config(arms, data_cfg):
    """Resolve, snapshot, then revert arms.yaml as a fresh clone would."""
    original = arms.read_text(encoding="utf-8")
    before = yaml.safe_load(original)["arms"]["yolo26m"]
    rewrite_yolo26m(arms, epochs=50, batch=16, lr=0.001)
    config.snapshot_arms("01_complete_medium_grid", data_cfg)
    arms.write_text(original, encoding="utf-8")          # the session died

    status = config.arms_snapshot_status(data_cfg)
    assert status["differs"]
    (difference,) = [d for d in status["differences"] if d["arm"] == "yolo26m"]
    # left is what arms.yaml says NOW, right is what the snapshot resolved to
    assert difference["left"] == {
        "epochs": before["epochs"], "batch": before["batch"], "lr": before["lr"]
    }
    assert difference["right"] == {"epochs": 50, "batch": 16, "lr": 0.001}


def test_restoring_the_body_recovers_the_resolved_values(arms, data_cfg):
    original = arms.read_text(encoding="utf-8")
    rewrite_yolo26m(arms, epochs=50, batch=16, lr=0.001)
    snapshot = config.snapshot_arms("01_complete_medium_grid", data_cfg)
    arms.write_text(original, encoding="utf-8")

    arms.write_text(config.snapshot_body(snapshot), encoding="utf-8")

    restored = yaml.safe_load(arms.read_text(encoding="utf-8"))["arms"]["yolo26m"]
    assert (restored["epochs"], restored["batch"], restored["lr"]) == (50, 16, 0.001)
    assert config.arms_snapshot_status(data_cfg)["differs"] is False


def test_unresolved_arms_are_named():
    """Contract only -- not tied to whatever configs/arms.yaml currently says,
    which changes as scripts 01 and 02 resolve their arms."""
    pending = config.unresolved_arms({
        "arms": {
            "settled": {"lr": 0.01, "locked": True},
            "needs_sweep": {"lr": None},
            "needs_grid": {"lr": 0.01, "provisional": True},
            "needs_both": {"lr": None, "provisional": True},
        }
    })
    assert pending["lr_null"] == ["needs_both", "needs_sweep"]
    assert pending["provisional"] == ["needs_both", "needs_grid"]


def test_status_when_no_snapshot_exists(arms, data_cfg):
    status = config.arms_snapshot_status(data_cfg)
    assert status["exists"] is False
    assert "differs" not in status
