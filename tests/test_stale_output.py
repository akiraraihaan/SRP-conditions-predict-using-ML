"""Script 06 must not be able to leave stale output behind.

The incident: a figure and two tables built from a one-fold smoke run sat in
artifacts/ after that run's registry record had been removed. They looked exactly
like current output, and this repository is cited in the manuscript's data
availability statement.

Two defences, both tested here: the output set is cleared before anything is
regenerated, and everything written carries a provenance stamp so a stale file
announces itself.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from srpcard import aggregate  # noqa: E402


@pytest.fixture(scope="module")
def script06():
    spec = importlib.util.spec_from_file_location(
        "export_figures", REPO_ROOT / "scripts" / "06_export_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_figures"] = module
    spec.loader.exec_module(module)
    return module


def make_records(n, arm="yolo26n", digest="abc123"):
    return [
        {
            "run_id": "r%d" % i,
            "script": "03_run_cv",
            "arm": arm,
            "corpus_fingerprint": {"sha1_of_sorted_included_sha1s": digest},
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------- clearing


def test_clear_removes_tables_and_figures(script06, artifacts):
    figures_dir = artifacts / "figures"
    figures_dir.mkdir()
    for name in aggregate.TABLE_NAMES:
        (artifacts / name).write_text("stale", encoding="utf-8")
    for name in ("fig_pareto.pdf", "fig_pareto.png", "fig_cv_box.pdf"):
        (figures_dir / name).write_text("stale", encoding="utf-8")

    removed = script06.clear_outputs(artifacts, figures_dir)

    assert removed == len(aggregate.TABLE_NAMES) + 3
    assert not any((artifacts / n).exists() for n in aggregate.TABLE_NAMES)
    assert list(figures_dir.iterdir()) == []


def test_clear_leaves_everything_else_alone(script06, artifacts):
    """Frozen inputs, the registry and other scripts' outputs are not ours."""
    keep = [
        "image_index.csv", "folds.json", "dev_split.json", "registry.jsonl",
        "resolved_arms.yaml", "uniform_grid.csv", "learning_curve.csv",
        "ablation_paired.csv", "legacy_grid_metrics.csv",
    ]
    for name in keep:
        (artifacts / name).write_text("keep", encoding="utf-8")
    (artifacts / "summary_cv.csv").write_text("stale", encoding="utf-8")
    figures_dir = artifacts / "figures"
    figures_dir.mkdir()

    script06.clear_outputs(artifacts, figures_dir)

    for name in keep:
        assert (artifacts / name).exists(), "%s was deleted" % name
    assert not (artifacts / "summary_cv.csv").exists()


def test_clear_is_safe_when_nothing_exists(script06, artifacts):
    assert script06.clear_outputs(artifacts, artifacts / "figures") == 0


def test_clear_does_not_touch_non_image_files_in_the_figure_dir(script06, artifacts):
    figures_dir = artifacts / "figures"
    figures_dir.mkdir()
    (figures_dir / "notes.txt").write_text("hand written", encoding="utf-8")
    (figures_dir / "fig_pareto.pdf").write_text("stale", encoding="utf-8")

    script06.clear_outputs(artifacts, figures_dir)

    assert (figures_dir / "notes.txt").exists()
    assert not (figures_dir / "fig_pareto.pdf").exists()


# ---------------------------------------------------------------- provenance


def test_provenance_reports_what_it_was_built_from(registry_path):
    registry_path.write_text("one line\n", encoding="utf-8")
    block = aggregate.provenance(make_records(3), registry_path)
    assert block["n_records"] == 3
    assert block["arms"] == ["yolo26n"]
    assert block["scripts"] == ["03_run_cv"]
    assert block["corpus_fingerprint"] == "abc123"
    assert len(block["registry_sha1"]) == 16
    assert block["generated_at"].endswith("+00:00")


def test_provenance_distinguishes_stale_from_fresh(registry_path):
    """The whole point: 1 record against 75 is visible on the artefact."""
    stale = aggregate.provenance(make_records(1), registry_path)
    fresh = aggregate.provenance(make_records(75), registry_path)
    assert stale["n_records"] != fresh["n_records"]
    assert "1 registry record" in aggregate.provenance_lines(stale)[0]
    assert "75 registry record" in aggregate.provenance_lines(fresh)[0]


def test_provenance_flags_a_mixed_corpus(registry_path):
    records = make_records(2, digest="aaa") + make_records(2, digest="bbb")
    block = aggregate.provenance(records, registry_path)
    assert block["corpus_fingerprint"] == ["aaa", "bbb"]


def test_provenance_survives_a_missing_registry(tmp_path):
    block = aggregate.provenance([], tmp_path / "nope.jsonl")
    assert block["registry_sha1"] == "absent"
    assert block["n_records"] == 0


# ---------------------------------------------------------------- stamped tables


def test_stamped_csv_is_still_readable(artifacts, registry_path):
    frame = pd.DataFrame({"arm": ["yolo26n"], "f1_macro_mean": [0.83]})
    block = aggregate.provenance(make_records(15), registry_path)
    target = aggregate.write_csv_with_provenance(frame, artifacts / "t.csv", block)

    # every consumer in this repository reads these with pandas
    reloaded = pd.read_csv(target, comment="#")
    assert list(reloaded.columns) == ["arm", "f1_macro_mean"]
    assert reloaded["f1_macro_mean"].iloc[0] == pytest.approx(0.83)

    text = target.read_text(encoding="utf-8")
    assert text.startswith("# built from 15 registry record(s)")
    assert "# registry sha1:" in text
    assert "# corpus:" in text


def test_script06_reads_its_own_stamped_tables(script06):
    """A stamped table must not break the readers in script 06."""
    source = (REPO_ROOT / "scripts" / "06_export_figures.py").read_text(encoding="utf-8")
    for name in ("summary_path", "epochs_path", "lc_path", "paired_path", "per_class_path"):
        if "pd.read_csv(%s" % name in source:
            assert 'pd.read_csv(%s, comment="#")' % name in source, (
                "%s is read without comment='#', so a provenance header would "
                "break it" % name
            )


def test_figures_expose_a_provenance_hook():
    from srpcard import figures

    figures.set_provenance(None)
    assert figures.PROVENANCE is None
    block = {"n_records": 5, "arms": ["yolo26n"], "corpus_fingerprint": "abc",
             "registry_sha1": "def", "generated_at": "2026-01-01T00:00:00+00:00"}
    figures.set_provenance(block)
    try:
        caption = aggregate.provenance_caption(block)
        assert "5 record(s)" in caption and "yolo26n" in caption and "def" in caption
    finally:
        figures.set_provenance(None)
