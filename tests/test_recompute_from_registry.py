"""Task A: the two things the registry can answer without retraining.

Nothing here touches artifacts/. The confusion-matrix arithmetic and the
Nadeau-Bengio correction are both checkable against independent references --
sklearn for macro-F1, the closed-form factor for the interval width -- so the
parts that decide what the manuscript claims are pinned here rather than only
in the --verify-reference mode that runs against the real registry.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def recompute():
    spec = importlib.util.spec_from_file_location(
        "recompute", REPO_ROOT / "scripts" / "08_recompute_from_registry.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["recompute"] = module
    spec.loader.exec_module(module)
    return module


CLASSES = [
    "collide_pump_and_vibration",
    "full_load_production",
    "gas_influence",
    "gas_influence_and_vibration",
    "insufficient_liquid_supply_and_vibration",
    "natural_flowing",
    "pump_leakage",
    "severe_insufficient_liquid_supply",
    "severe_vibration",
    "vibration",
]


# ------------------------------------------------------------------ macro-F1


def test_macro_f1_matches_sklearn(recompute):
    """The recorded f1_macro came from sklearn, so this must agree with it or
    the baseline scenario silently reports a different quantity."""
    sklearn = pytest.importorskip("sklearn.metrics")

    rng = np.random.default_rng(0)
    truth = rng.integers(0, 10, size=400)
    predicted = np.where(rng.random(400) < 0.6, truth, rng.integers(0, 10, size=400))

    matrix = sklearn.confusion_matrix(truth, predicted, labels=list(range(10)))
    expected = sklearn.f1_score(
        truth, predicted, labels=list(range(10)), average="macro", zero_division=0
    )

    assert recompute.macro_f1_from_confusion(matrix) == pytest.approx(expected, abs=1e-12)


def test_an_absent_class_scores_zero_not_nan(recompute):
    """A class with no support and no predictions must count as 0, which is what
    sklearn does with zero_division=0 over an explicit labels list."""
    matrix = np.zeros((3, 3))
    matrix[0, 0] = 5
    matrix[1, 1] = 5
    # class 2 never appears at all
    assert recompute.macro_f1_from_confusion(matrix) == pytest.approx(2.0 / 3.0)


def test_a_perfect_matrix_scores_one(recompute):
    assert recompute.macro_f1_from_confusion(np.eye(10) * 7) == pytest.approx(1.0)


# -------------------------------------------------------------------- merging


def test_merging_sums_both_the_row_and_the_column(recompute):
    """Merging i and j means a confusion BETWEEN them stops being an error."""
    matrix = np.array([[5.0, 3.0], [2.0, 4.0]])
    merged, names = recompute.merge_confusion(matrix, ["a", "b"], [("a", "b")])

    assert merged.shape == (1, 1)
    assert merged[0, 0] == 14.0          # every cell folds into the diagonal
    assert names == ["a+b"]
    assert recompute.macro_f1_from_confusion(merged) == pytest.approx(1.0)


def test_merging_preserves_the_total_count(recompute):
    rng = np.random.default_rng(3)
    matrix = rng.integers(0, 9, size=(10, 10)).astype(float)
    merged, _ = recompute.merge_confusion(
        matrix, CLASSES, [("vibration", "severe_vibration"),
                          ("pump_leakage", "natural_flowing")]
    )
    assert merged.sum() == pytest.approx(matrix.sum())
    assert merged.shape == (8, 8)


def test_merging_can_only_raise_macro_f1_for_the_merged_pair(recompute):
    """Sanity: the control exists because this is true for ANY pair."""
    rng = np.random.default_rng(11)
    matrix = rng.integers(1, 20, size=(10, 10)).astype(float)
    base = recompute.macro_f1_from_confusion(matrix)
    gains = []
    for left, right in [("vibration", "severe_vibration"),
                        ("gas_influence", "pump_leakage"),
                        ("full_load_production", "natural_flowing")]:
        merged, _ = recompute.merge_confusion(matrix, CLASSES, [(left, right)])
        gains.append(recompute.macro_f1_from_confusion(merged) - base)
    assert all(g > -1e-9 for g in gains), "a merge lowered macro-F1: %s" % gains


def test_unmerged_classes_keep_their_identity(recompute):
    matrix = np.eye(10) * 3
    merged, names = recompute.merge_confusion(
        matrix, CLASSES, [("vibration", "severe_vibration")]
    )
    assert names[0] == "severe_vibration+vibration"
    assert set(names[1:]) == set(CLASSES) - {"vibration", "severe_vibration"}


def test_a_scenario_naming_an_unknown_class_fails_loudly(recompute):
    """A rename upstream must break here rather than merge the wrong pair."""
    with pytest.raises(SystemExit) as caught:
        recompute.scenario_groups(["a", "b", "c"])
    assert "not in the registry's class_order" in str(caught.value)


def test_the_scenarios_are_what_the_manuscript_proposes(recompute):
    groups = recompute.scenario_groups(CLASSES)
    assert groups["baseline"] == []
    assert groups["M1"] == [("vibration", "severe_vibration")]
    assert groups["M2"] == [("pump_leakage", "natural_flowing")]
    assert groups["M3"] == [("vibration", "severe_vibration"),
                            ("pump_leakage", "natural_flowing")]
    assert len(groups["M4"][0]) == 5          # every vibration-bearing class


# ------------------------------------------------------ Nadeau-Bengio


def test_the_width_factor_is_2_1794(recompute):
    """The number the whole correction turns on: sqrt((1/15 + 1/4)/(1/15))."""
    n, rho = 15, 1.0 / 4.0
    factor = np.sqrt((1.0 / n + rho) / (1.0 / n))
    assert factor == pytest.approx(2.1794, abs=1e-4)
    assert factor == pytest.approx(np.sqrt(4.75))


def test_the_corrected_interval_is_exactly_that_much_wider(recompute):
    rng = np.random.default_rng(7)
    differences = rng.normal(0.05, 0.06, size=15)

    result = recompute.corrected_interval(differences, rho=0.25)

    naive_width = result["ci95_naive_high"] - result["ci95_naive_low"]
    corrected_width = result["ci95_corrected_high"] - result["ci95_corrected_low"]
    assert corrected_width / naive_width == pytest.approx(2.1794, abs=1e-4)


def test_rho_zero_recovers_the_naive_interval(recompute):
    """The two intervals share one code path so they cannot drift apart."""
    rng = np.random.default_rng(9)
    differences = rng.normal(0.02, 0.05, size=15)
    result = recompute.corrected_interval(differences, rho=0.0)
    assert result["ci95_corrected_low"] == pytest.approx(result["ci95_naive_low"])
    assert result["ci95_corrected_high"] == pytest.approx(result["ci95_naive_high"])
    assert result["p_corrected"] == pytest.approx(result["p_naive"])


def test_the_correction_is_conservative_never_narrower(recompute):
    rng = np.random.default_rng(13)
    for _ in range(20):
        differences = rng.normal(rng.normal(0, 0.05), 0.08, size=15)
        result = recompute.corrected_interval(differences, rho=0.25)
        assert result["se_corrected"] >= result["se_naive"]
        assert result["p_corrected"] >= result["p_naive"]


def test_a_significant_naive_result_can_lose_significance(recompute):
    """The expected outcome for three of the six pairs. Not a bug."""
    # t_naive must sit between the critical value and 2.1794x it: significant
    # before the correction, not after. t_naive = 3 is comfortably inside.
    spread = np.linspace(-1.0, 1.0, 15)
    spread = spread / spread.std(ddof=1)
    differences = 0.04 + spread * (0.04 / 3.0) * np.sqrt(15)
    result = recompute.corrected_interval(differences, rho=0.25)

    assert result["naive_excludes_zero"] is True
    assert result["corrected_excludes_zero"] is False
    assert result["p_naive"] < 0.05 < result["p_corrected"]


def test_win_loss_tie_counts(recompute):
    differences = np.array([0.1, 0.2, -0.1, 0.0, 0.0])
    result = recompute.corrected_interval(differences, rho=0.25)
    assert (result["a_wins"], result["b_wins"], result["ties"]) == (2, 1, 2)


# ------------------------------------------------------------------ k folds


def test_k_is_read_off_the_records_not_assumed(recompute):
    records = [{"fold": i % 5, "repeat": i // 5} for i in range(15)]
    assert recompute.folds_per_repeat(records) == 5


def test_a_gappy_fold_range_is_refused(recompute):
    """n_test/n_train is 1/(k-1) only for a complete partition."""
    with pytest.raises(SystemExit) as caught:
        recompute.folds_per_repeat([{"fold": 0}, {"fold": 1}, {"fold": 4}])
    assert "contiguous" in str(caught.value)


# --------------------------------------------------------- the 45-pair control


def test_there_are_exactly_45_single_pair_merges(recompute):
    import itertools

    assert len(list(itertools.combinations(CLASSES, 2))) == 45


def test_every_pair_is_ranked_within_its_arm(recompute):
    records = []
    rng = np.random.default_rng(5)
    for arm in ("a", "b"):
        for fold in range(3):
            records.append({
                "arm": arm,
                "repeat": 0,
                "fold": fold,
                "class_order": CLASSES,
                "confusion_matrix": rng.integers(0, 9, size=(10, 10)).tolist(),
                "f1_macro": 0.5,
            })

    frame = recompute.all_pair_merges(records)

    assert len(frame) == 90                       # 45 pairs x 2 arms
    for arm in ("a", "b"):
        block = frame[frame["arm"] == arm]
        assert sorted(block["rank_in_arm"]) == list(range(1, 46))
    assert set(frame[frame["hypothesised"] == "M1"]["arm"]) == {"a", "b"}


def test_records_disagreeing_on_class_order_are_refused(recompute):
    """Merging by name across two label spaces compares different quantities."""
    records = [
        {"arm": "a", "repeat": 0, "fold": 0, "class_order": CLASSES,
         "confusion_matrix": np.eye(10).tolist(), "f1_macro": 1.0},
        {"arm": "a", "repeat": 0, "fold": 1, "class_order": list(reversed(CLASSES)),
         "confusion_matrix": np.eye(10).tolist(), "f1_macro": 1.0},
    ]
    with pytest.raises(SystemExit) as caught:
        recompute.per_fold_merged_f1(records)
    assert "class_order" in str(caught.value)


def test_the_baseline_scenario_reproduces_the_recorded_metric(recompute):
    """The check that catches a whole class of error for free."""
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(21)
    truth = rng.integers(0, 10, size=134)
    predicted = np.where(rng.random(134) < 0.6, truth, rng.integers(0, 10, size=134))
    matrix = sklearn.confusion_matrix(truth, predicted, labels=list(range(10)))
    recorded = sklearn.f1_score(truth, predicted, labels=list(range(10)),
                                average="macro", zero_division=0)

    frame = recompute.per_fold_merged_f1([{
        "arm": "a", "repeat": 0, "fold": 0, "class_order": CLASSES,
        "confusion_matrix": matrix.tolist(), "f1_macro": recorded,
    }])
    baseline = frame[frame["scenario"] == "baseline"].iloc[0]
    assert baseline["macro_f1"] == pytest.approx(baseline["recorded_f1_macro"], abs=1e-12)


def test_m4_is_flagged_as_not_a_plausible_taxonomy(recompute):
    rng = np.random.default_rng(31)
    records = [{
        "arm": "a", "repeat": 0, "fold": f, "class_order": CLASSES,
        "confusion_matrix": rng.integers(0, 9, size=(10, 10)).tolist(),
        "f1_macro": 0.5,
    } for f in range(5)]

    folds = recompute.per_fold_merged_f1(records)
    summary = recompute.taxonomy_summary(folds, rho=0.25)

    m4 = summary[summary["scenario"] == "M4"].iloc[0]
    assert not m4["plausible_taxonomy"]
    assert "upper bound" in m4["note"]
    for scenario in ("M1", "M2", "M3"):
        assert summary[summary["scenario"] == scenario].iloc[0]["plausible_taxonomy"]


# ------------------------------------------- every new artefact says what it is


def test_mixed_protocols_are_refused_not_labelled(recompute):
    """The legacy augmented runs and the uniform ones are not comparable, so a
    column saying which would only make the mixture look deliberate."""
    records = [
        {"extra": {"protocol": "uniform"}},
        {"extra": {"protocol": "legacy"}},
    ]
    with pytest.raises(SystemExit) as caught:
        recompute.protocol_of(records)
    assert "not comparable" in str(caught.value)


def test_a_record_with_no_protocol_is_refused(recompute):
    with pytest.raises(SystemExit):
        recompute.protocol_of([{"extra": {}}])


def test_one_protocol_is_returned(recompute):
    assert recompute.protocol_of([{"extra": {"protocol": "uniform"}}] * 3) == "uniform"


def test_every_output_carries_arm_and_protocol():
    """The rule, going forward: a table that does not say which model and which
    regime produced it cannot be checked against anything. artifacts/
    learning_curve.csv records neither, which is why a stale caption survived
    for weeks."""
    source = (REPO_ROOT / "scripts" / "08_recompute_from_registry.py").read_text(
        encoding="utf-8"
    )
    assert 'frame["protocol"] = protocol' in source, (
        "script 08 must stamp every frame it writes with the protocol"
    )


@pytest.mark.parametrize("frame_name", ["summary", "pairs", "folds"])
def test_the_taxonomy_frames_all_carry_an_arm_column(recompute, frame_name):
    rng = np.random.default_rng(41)
    records = [{
        "arm": "a", "repeat": 0, "fold": f, "class_order": CLASSES,
        "confusion_matrix": rng.integers(0, 9, size=(10, 10)).tolist(),
        "f1_macro": 0.5, "extra": {"protocol": "uniform"},
    } for f in range(5)]

    folds = recompute.per_fold_merged_f1(records)
    frames = {
        "folds": folds,
        "summary": recompute.taxonomy_summary(folds, rho=0.25),
        "pairs": recompute.all_pair_merges(records),
    }
    assert "arm" in frames[frame_name].columns
