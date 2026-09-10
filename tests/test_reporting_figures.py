"""What script 06 reports, and what its artefacts claim about themselves.

Two failures these pin:

  - the confusion matrix was drawn for whichever arm had the highest mean F1,
    which is not the model the paper recommends, and nothing in the filename
    said which model it was;
  - one run-wide provenance stamp was reused for every figure, so the
    learning-curve and ablation figures claimed `scripts: 03_run_cv` when they
    are built from script 05 and 04 records.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

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


def cv_record(arm, repeat, fold, script="03_run_cv", **overrides):
    base = {
        "run_id": "%s-%s-r%df%d" % (script, arm, repeat, fold),
        "script": script,
        "arm": arm,
        "split_kind": "cv",
        "repeat": repeat,
        "fold": fold,
        "epochs": 50,
        "batch": 16,
        "lr": 0.01,
        "f1_macro": 0.5,
        "confusion_matrix": [[10, 2], [3, 20]],
        "corpus_fingerprint": {"sha1_of_sorted_included_sha1s": "abc"},
        "extra": {"protocol": "uniform"},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------- detailed arm


def test_config_names_the_detailed_arm():
    cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "arms.yaml").read_text(encoding="utf-8")
    )
    assert cfg["reporting"]["detailed_arm"] in cfg["arms"], (
        "reporting.detailed_arm must name a real arm"
    )


def test_no_arm_name_is_hardcoded_in_the_confusion_section():
    """The arm must come from the config, not from a literal or from argmax."""
    source = (REPO_ROOT / "scripts" / "06_export_figures.py").read_text(encoding="utf-8")
    section = source[source.index("# ---- 4. confusion matrices"):]
    section = section[: section.index("# ---- 5.")]
    assert "reporting" in section and "detailed_arm" in section
    for literal in ("mobilenetv3_small", "resnet18", "yolo26n", "yolo26s", "yolo26m"):
        assert literal not in section, "%s is hardcoded in the confusion section" % literal
    assert "summarise_cv" not in section, "the detailed arm must not come from argmax"


def test_the_arm_is_in_the_figure_filename():
    source = (REPO_ROOT / "scripts" / "06_export_figures.py").read_text(encoding="utf-8")
    assert '"fig_confusion_%s" % arm' in source


# ---------------------------------------------------------------- summed matrix


def test_confusion_is_summed_over_folds_not_taken_from_one():
    records = [cv_record("a", i // 5, i % 5) for i in range(15)]
    matrix, used = aggregate.summed_confusion_matrix("a", records)
    assert len(used) == 15
    # each fold contributes [[10,2],[3,20]] = 35 predictions
    assert int(np.asarray(matrix).sum()) == 15 * 35
    assert matrix[0][0] == 150


def test_only_the_requested_arm_is_summed():
    records = [cv_record("a", 0, 0), cv_record("b", 0, 0)]
    matrix, used = aggregate.summed_confusion_matrix("a", records)
    assert len(used) == 1 and used[0]["arm"] == "a"


def test_missing_arm_returns_nothing():
    assert aggregate.summed_confusion_matrix("nope", [cv_record("a", 0, 0)]) == (None, [])


def test_expected_total_is_the_corpus_times_the_repeats():
    """668 clean images x 3 repeats = 2004 predictions."""
    assert aggregate.expected_confusion_total() == 2004


def test_a_complete_arm_with_the_wrong_total_raises(monkeypatch):
    """A complete arm must account for every image once per repeat. If it does
    not, the matrix is not the corpus-wide matrix it would be reported as."""
    bad = np.array([[1, 0], [0, 1]])          # 2 predictions, not 2004
    with pytest.raises(ValueError, match="expected 2004"):
        aggregate.check_confusion_total(bad, "a", 15)


def test_an_incomplete_arm_is_labelled_not_raised():
    """A run still in progress is not an error, but must not pass as complete."""
    partial = np.array([[500, 0], [0, 500]])
    total, expected, note = aggregate.check_confusion_total(partial, "a", 14)
    assert total == 1000 and expected == 2004
    assert "INCOMPLETE" in note and "14 of 15" in note


def test_the_real_matrices_total_2004():
    records = aggregate.cv_records()
    if not records:
        pytest.skip("no 03_run_cv records in this checkout")
    for arm in sorted({r["arm"] for r in records}):
        matrix, used = aggregate.summed_confusion_matrix(arm, records)
        if len(used) == 15:
            total, expected, _ = aggregate.check_confusion_total(matrix, arm, len(used))
            assert total == expected == 2004, arm


# ---------------------------------------------------------------- provenance


def test_provenance_records_the_scripts_that_fed_it():
    block = aggregate.provenance([cv_record("a", 0, 0, script="04_run_ablation")])
    assert block["scripts"] == ["04_run_ablation"]
    assert block["n_records"] == 1


def test_a_stamp_naming_the_wrong_script_is_refused():
    """The exact failure: an ablation figure claiming 03-only provenance."""
    ablation = [cv_record("a", 0, i, script="04_run_ablation") for i in range(15)]
    cv = [cv_record("a", 0, i, script="03_run_cv") for i in range(15)]

    wrong = aggregate.provenance(cv)          # a 03 stamp ...
    with pytest.raises(ValueError, match="scripts says"):
        aggregate.assert_provenance_covers(wrong, ablation)   # ... on 04 records

    right = aggregate.provenance(ablation)
    aggregate.assert_provenance_covers(right, ablation)


def test_a_stamp_with_the_wrong_count_is_refused():
    records = [cv_record("a", 0, i) for i in range(15)]
    block = aggregate.provenance(records[:5])
    with pytest.raises(ValueError, match="n_records says"):
        aggregate.assert_provenance_covers(block, records)


def test_sources_appear_for_a_figure_built_from_no_records():
    block = aggregate.provenance([], sources=["artifacts/image_index.csv"])
    assert block["n_records"] == 0
    assert block["sources"] == ["artifacts/image_index.csv"]
    assert "sources: artifacts/image_index.csv" in aggregate.provenance_lines(block)
    assert "artifacts/image_index.csv" in aggregate.provenance_caption(block)


def test_the_caption_names_the_scripts():
    block = aggregate.provenance([cv_record("a", 0, 0, script="05_learning_curve")])
    assert "05_learning_curve" in aggregate.provenance_caption(block)


def test_script06_verifies_every_stamp(script06):
    """`stamped()` is the only way 06 installs provenance, and it checks."""
    source = (REPO_ROOT / "scripts" / "06_export_figures.py").read_text(encoding="utf-8")
    assert "assert_provenance_covers" in source
    # set_provenance must be reached through stamped(), never called directly
    assert source.count("figures.set_provenance(") == 1
    assert "def stamped(" in source

    ablation = [cv_record("a", 0, i, script="04_run_ablation") for i in range(3)]
    block = script06.stamped(ablation)
    assert block["scripts"] == ["04_run_ablation"]


# ---------------------------------------------------------------- per class


def test_per_class_carries_mean_sd_and_support():
    records = []
    for i in range(15):
        records.append(
            cv_record(
                "a", i // 5, i % 5,
                f1_per_class={"x": 0.5, "y": 0.7},
                recall_per_class={"x": 0.4, "y": 0.6},
                precision_per_class={"x": 0.6, "y": 0.8},
                support_per_class={"x": 3, "y": 5},
            )
        )
    data_cfg = {
        "classes": ["x", "y"],
        "clean_corpus": {"expected_counts": {"x": 1, "y": 2}},
    }
    frame = aggregate.summarise_per_class(records, data_cfg).set_index("class")
    for column in ("precision_mean", "precision_std", "recall_mean", "recall_std",
                   "f1_mean", "f1_std", "support_total", "support_mean", "n_clean"):
        assert column in frame.columns, column
    assert frame.loc["x", "support_total"] == 45      # 3 per fold x 15
    assert frame.loc["x", "support_mean"] == pytest.approx(3.0)
    assert frame.loc["x", "precision_mean"] == pytest.approx(0.6)
    assert frame.loc["x", "f1_std"] == pytest.approx(0.0)


def test_per_class_support_matches_the_corpus():
    """support_total must be the class's clean count times the repeat count."""
    records = aggregate.cv_records()
    if not records:
        pytest.skip("no 03_run_cv records in this checkout")
    from srpcard.config import load_data_config, load_folds_config

    repeats = int(load_folds_config()["cv"]["n_repeats"])
    frame = aggregate.summarise_per_class(records)
    complete = frame[frame["n_folds"] == 15]
    for row in complete.to_dict("records"):
        assert row["support_total"] == row["n_clean"] * repeats, row["class"]


# ---------------------------------------------------------------- publication mode


@pytest.fixture
def restore_render():
    from srpcard import figures

    before = figures.RENDER_PROVENANCE
    yield
    figures.set_render_provenance(before)


BLOCK = {
    "n_records": 15, "arms": ["a"], "scripts": ["03_run_cv"], "sources": [],
    "corpus_fingerprint": "abc", "registry_sha1": "def",
    "generated_at": "2026-01-01T00:00:00+00:00",
}


def test_the_strip_is_drawn_by_default(restore_render):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from srpcard import figures

    figures.set_provenance(BLOCK)
    figures.set_render_provenance(True)
    fig, _ = plt.subplots()
    figures._stamp(fig)
    try:
        assert len(fig.texts) == 1
        assert "03_run_cv" in fig.texts[0].get_text()
    finally:
        plt.close(fig)


def test_publication_mode_omits_the_strip(restore_render):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from srpcard import figures

    figures.set_provenance(BLOCK)
    figures.set_render_provenance(False)
    fig, _ = plt.subplots()
    figures._stamp(fig)
    try:
        assert fig.texts == []
    finally:
        plt.close(fig)


def test_publication_mode_still_writes_the_metadata(tmp_path, restore_render):
    """The point of the flag: the strip goes, the provenance does not."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from srpcard import figures

    figures.set_provenance(BLOCK)
    figures.set_render_provenance(False)
    fig, _ = plt.subplots()
    written = figures.save(fig, tmp_path, "fig_x")
    plt.close(fig)

    pdf = next(p for p in written if p.suffix == ".pdf")
    assert b"03_run_cv" in pdf.read_bytes(), "PDF metadata lost the script list"

    png = next(p for p in written if p.suffix == ".png")
    raw = png.read_bytes()
    comment, i = None, 8
    while i < len(raw) - 8:
        length = int.from_bytes(raw[i:i + 4], "big")
        kind = raw[i + 4:i + 8]
        if kind == b"tEXt":
            key, _, value = raw[i + 8:i + 8 + length].partition(b"\x00")
            if key == b"Comment":
                comment = value.decode("latin-1")
        i += 12 + length
    assert comment and "03_run_cv" in comment, "PNG metadata lost the script list"


def test_publication_figures_go_to_their_own_directory():
    source = (REPO_ROOT / "scripts" / "06_export_figures.py").read_text(encoding="utf-8")
    assert '"figures_pub" if args.for_publication else "figures"' in source


def test_publication_mode_does_not_rewrite_or_prune_the_tables():
    """The default set's tables are inputs here; touching them would let the two
    modes overwrite each other's work."""
    source = (REPO_ROOT / "scripts" / "06_export_figures.py").read_text(encoding="utf-8")
    assert "artifacts=None if args.for_publication else artifacts_dir(data_cfg)" in source
    assert '[table] not rewritten in --for-publication' in source


def test_pruning_can_spare_the_tables(script06, artifacts):
    """Publication mode passes artifacts=None, so tables are never pruned."""
    figures_dir = artifacts / "figures_pub"
    figures_dir.mkdir()
    (figures_dir / "fig_pareto.pdf").write_text("stale", encoding="utf-8")
    for name in aggregate.TABLE_NAMES:
        (artifacts / name).write_text("keep", encoding="utf-8")

    removed = script06.prune_outputs(figures_dir, [])

    assert len(removed) == 1
    for name in aggregate.TABLE_NAMES:
        assert (artifacts / name).exists(), "%s was pruned in publication mode" % name


# ---------------------------------------------------------------- decoupling


def test_script06_needs_no_torch_and_no_dataset():
    """It must run on a laptop with no GPU and no copy of the images."""
    import subprocess

    probe = (
        "import sys, os, runpy\n"
        "sys.path.insert(0, 'src')\n"
        "os.environ['SRPCARD_DATA_ROOT'] = '/nonexistent-on-purpose'\n"
        "sys.argv = ['06']\n"
        "try:\n"
        "    runpy.run_path('scripts/06_export_figures.py', run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    assert not exc.code, exc.code\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'torch','torchvision','ultralytics','cv2','thop'})\n"
        "print('HEAVY:' + ','.join(heavy))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    line = next(l for l in result.stdout.splitlines() if l.startswith("HEAVY:"))
    assert line == "HEAVY:", "script 06 pulled in %s" % line


def test_script06_reads_only_artifacts_and_configs():
    source = (REPO_ROOT / "scripts" / "06_export_figures.py").read_text(encoding="utf-8")
    assert "resolve_data_root" not in source, "06 must not resolve DATA_ROOT"
    for module in ("torch", "ultralytics", "torchvision"):
        assert "import %s" % module not in source
