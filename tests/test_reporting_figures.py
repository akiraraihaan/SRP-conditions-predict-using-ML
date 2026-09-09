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
