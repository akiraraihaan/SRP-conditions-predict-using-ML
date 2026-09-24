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


# ============================================================================
# M5 and the rank-concentration control.
#
# M1 and M2 were proposed as two separate confusable pairs. All six pairs
# drawn from {vibration, severe_vibration, pump_leakage, natural_flowing}
# occupy ranks 1-6 of 45 for mobilenetv3_small, resnet18 and yolo26m, and
# 1,2,4,5,6,7 for yolo26n and yolo26s. Those four classes are ONE error
# family, and that is a stronger result than the hypothesis.
# ============================================================================


def test_the_family_is_the_four_classes(recompute):
    assert set(recompute.ERROR_FAMILY) == {
        "vibration", "severe_vibration", "pump_leakage", "natural_flowing",
    }


def test_m5_merges_the_family_into_one_class(recompute):
    groups = recompute.scenario_groups(CLASSES)
    assert groups["M5"] == [recompute.ERROR_FAMILY]

    matrix = np.eye(10) * 5
    merged, names = recompute.merge_confusion(matrix, CLASSES, groups["M5"])
    assert merged.shape == (7, 7), "10 classes minus 4 plus 1 is 7"
    assert names[0] == "+".join(sorted(recompute.ERROR_FAMILY))


def test_m5_is_a_plausible_taxonomy_unlike_m4(recompute):
    """M4 fuses unrelated physics and is an upper bound. M5 fuses one severity
    continuum with the collapsed-card pair, which is a different claim."""
    rng = np.random.default_rng(77)
    records = [{
        "arm": "a", "repeat": 0, "fold": f, "class_order": CLASSES,
        "confusion_matrix": rng.integers(0, 9, size=(10, 10)).tolist(),
        "f1_macro": 0.5, "extra": {"protocol": "uniform"},
    } for f in range(5)]

    summary = recompute.taxonomy_summary(
        recompute.per_fold_merged_f1(records), rho=0.25
    )
    m5 = summary[summary["scenario"] == "M5"].iloc[0]
    m4 = summary[summary["scenario"] == "M4"].iloc[0]

    assert m5["plausible_taxonomy"]
    assert not m4["plausible_taxonomy"]
    assert m5["n_classes"] == 7


def test_the_m5_note_says_observation_not_recommendation(recompute):
    note = recompute.SCENARIO_NOTES["M5"]
    assert "OBSERVATION ABOUT ERROR STRUCTURE" in note
    assert "not a recommendation" in note


# ------------------------------------------------------- the exact statistic


def test_six_of_45_in_the_top_six_is_one_in_8_145_060(recompute):
    """Your figure, recomputed rather than quoted."""
    from math import comb

    assert comb(45, 6) == 8145060

    result = recompute.rank_concentration([1, 2, 3, 4, 5, 6], n_items=45)

    assert result["rank_sum"] == 21 == result["rank_sum_min_possible"]
    assert result["occupies_top_k"] is True
    assert result["p_top_k"] == pytest.approx(1 / 8145060)
    assert result["p_rank_sum"] == pytest.approx(1 / 8145060)


def test_the_graded_version_covers_a_block_that_is_not_flush(recompute):
    """yolo26n and yolo26s sit at 1,2,4,5,6,7. p_top_k is undefined there and
    the rank sum is what should be quoted."""
    result = recompute.rank_concentration([1, 2, 4, 5, 6, 7], n_items=45)

    assert result["rank_sum"] == 25
    assert result["occupies_top_k"] is False
    assert result["p_top_k"] is None
    assert result["p_rank_sum"] == pytest.approx(1.473e-06, rel=1e-3)


def test_the_distribution_is_exact_and_complete(recompute):
    """Counted, not sampled: the claim lives far out in the tail, which is
    where a normal approximation is least trustworthy."""
    from math import comb

    distribution = recompute.rank_sum_distribution(45, 6)
    assert sum(distribution.values()) == comb(45, 6)
    assert min(distribution) == 21          # 1+2+3+4+5+6
    assert max(distribution) == 255         # 40+41+42+43+44+45


def test_a_small_case_can_be_checked_by_hand(recompute):
    """C(5,2) = 10 subsets. Four sum to 5 or less: {1,2}=3, {1,3}=4, {1,4}=5
    and {2,3}=5. Enumerated here so the DP is checked against something that
    does not share its implementation."""
    from itertools import combinations

    by_hand = sum(1 for pair in combinations(range(1, 6), 2) if sum(pair) <= 5)
    assert by_hand == 4

    result = recompute.rank_concentration([1, 4], n_items=5)
    assert result["rank_sum"] == 5
    assert result["p_rank_sum"] == pytest.approx(by_hand / 10)


def test_an_unremarkable_position_is_not_significant(recompute):
    """The control has to be able to say no, or it is not a control."""
    result = recompute.rank_concentration([20, 21, 22, 23, 24, 25], n_items=45)
    assert result["p_rank_sum"] > 0.05


# ----------------------------------------------------- the per-arm table


def test_the_concentration_table_covers_every_arm(recompute):
    rng = np.random.default_rng(101)
    records = []
    for arm in ("a", "b"):
        for f in range(3):
            records.append({
                "arm": arm, "repeat": 0, "fold": f, "class_order": CLASSES,
                "confusion_matrix": rng.integers(0, 9, size=(10, 10)).tolist(),
                "f1_macro": 0.5, "extra": {"protocol": "uniform"},
            })

    pairs = recompute.all_pair_merges(records)
    concentration = recompute.family_rank_concentration(pairs)

    assert sorted(concentration["arm"]) == ["a", "b"]
    assert set(concentration["n_family_pairs"]) == {6}
    assert set(concentration["n_pairs_total"]) == {45}


def test_exactly_six_of_the_45_pairs_are_within_family(recompute):
    rng = np.random.default_rng(103)
    records = [{
        "arm": "a", "repeat": 0, "fold": f, "class_order": CLASSES,
        "confusion_matrix": rng.integers(0, 9, size=(10, 10)).tolist(),
        "f1_macro": 0.5, "extra": {"protocol": "uniform"},
    } for f in range(3)]

    pairs = recompute.all_pair_merges(records)

    assert pairs["within_family"].sum() == 6
    family = set(recompute.ERROR_FAMILY)
    for row in pairs[pairs["within_family"]].itertuples():
        assert {row.class_a, row.class_b} <= family
    # the two hypothesised pairs are a SUBSET of the family, not the whole of it
    assert pairs[pairs["hypothesised"] != ""]["within_family"].all()


# ------------------------------------------------------------- the figure


def test_the_figure_distinguishes_family_from_hypothesised():
    """Merging the two categories into one colour would hide the actual result:
    four pairs nobody proposed in advance arriving at the top alongside the two
    that were."""
    source = (REPO_ROOT / "src" / "srpcard" / "figures.py").read_text(encoding="utf-8")
    assert "within_family" in source
    assert "tab:red" in source and "tab:orange" in source
    assert "hypothesised in advance" in source


def test_the_preselection_caveat_travels_with_the_number(recompute):
    """Only M1 and M2 were pre-specified. The other four within-family pairs
    were discovered in the same ranking that scores them, so p_top_k applied to
    the family is circular -- selection and test share the numbers. The number
    must not stand anywhere without that sentence beside it."""
    header = " ".join(recompute.CONCENTRATION_HEADER)

    assert "Only TWO of the six pairs were pre-specified" in header
    assert "circular" in header
    assert "DESCRIPTIVE, PLUS" in header or "descriptive" in header.lower()
    assert "must not be multiplied" in header

    source = (REPO_ROOT / "scripts" / "08_recompute_from_registry.py").read_text(
        encoding="utf-8"
    )
    assert "CAVEAT. Only M1 and M2 were PRE-SPECIFIED." in source, (
        "the printed summary shows p_top_k, so it must carry the caveat too"
    )


def test_the_caveat_is_in_the_handover_too():
    doc = (REPO_ROOT / "HANDOVER.md").read_text(encoding="utf-8")
    section = doc[doc.index("## 4.14"):]
    assert "Only M1 and M2 were pre-specified" in section
    assert "circular" in section
    assert "must not be multiplied" in section
    assert "not as \"p < 1e-6" in section, (
        "the caveat should say what to write instead, not only what is wrong"
    )
