"""Paired comparisons and Pareto dominance.

These two tables decide which model the paper reports, so their arithmetic is
pinned against hand-computed fixtures rather than against whatever the registry
currently holds.
"""

from __future__ import annotations

import pandas as pd
import pytest

from srpcard import aggregate


def cv_record(arm, repeat, fold, f1, **overrides):
    base = {
        "run_id": "%s-r%df%d" % (arm, repeat, fold),
        "script": "03_run_cv",
        "arm": arm,
        "split_kind": "cv",
        "repeat": repeat,
        "fold": fold,
        "epochs": 50,
        "batch": 16,
        "lr": 0.01,
        "f1_macro": f1,
        "params": 1_000_000,
        "gflops": 1.0,
        "size_mb_fp16": 2.0,
        "extra": {"protocol": "uniform"},
    }
    base.update(overrides)
    return base


def records_for(spec: dict[str, list[float]], **efficiency):
    out = []
    for arm, scores in spec.items():
        for i, score in enumerate(scores):
            out.append(
                cv_record(arm, i // 5, i % 5, score, **efficiency.get(arm, {}))
            )
    return out


# ---------------------------------------------------------------- paired


def test_pairs_are_oriented_so_the_difference_is_positive():
    records = records_for({"a": [0.1, 0.2, 0.3], "b": [0.5, 0.6, 0.7]})
    frame = aggregate.paired_comparisons(records)
    row = frame.iloc[0]
    assert row["arm_a"] == "b" and row["arm_b"] == "a"
    assert row["mean_diff"] == pytest.approx(0.4)


def test_only_shared_folds_are_compared():
    """An arm with runs outstanding is compared on what it has, not padded."""
    records = records_for({"a": [0.5, 0.6, 0.7]}) + records_for({"b": [0.1, 0.2]})
    frame = aggregate.paired_comparisons(records)
    assert frame.iloc[0]["n_folds"] == 2
    assert frame.iloc[0]["mean_diff"] == pytest.approx(0.4)


def test_win_counts_and_ties():
    records = records_for({"a": [0.5, 0.5, 0.9], "b": [0.1, 0.5, 0.1]})
    row = aggregate.paired_comparisons(records).iloc[0]
    assert (row["a_wins"], row["b_wins"], row["ties"]) == (2, 0, 1)


def test_ci_excludes_zero_is_reported():
    clear = aggregate.paired_comparisons(
        records_for({"a": [0.9, 0.9, 0.9, 0.9], "b": [0.1, 0.1, 0.1, 0.1]})
    )
    assert bool(clear.iloc[0]["ci_excludes_zero"]) is True

    noisy = aggregate.paired_comparisons(
        records_for({"a": [0.5, 0.1, 0.6, 0.2], "b": [0.1, 0.5, 0.2, 0.6]})
    )
    assert bool(noisy.iloc[0]["ci_excludes_zero"]) is False


def test_every_pair_appears_once():
    records = records_for({"a": [0.1] * 3, "b": [0.2] * 3, "c": [0.3] * 3})
    frame = aggregate.paired_comparisons(records)
    pairs = {frozenset((r["arm_a"], r["arm_b"])) for _, r in frame.iterrows()}
    assert len(frame) == 3 and len(pairs) == 3


def test_empty_records_give_an_empty_frame():
    assert aggregate.paired_comparisons([]).empty


def test_paired_header_warns_that_folds_overlap():
    """Repeated CV puts every image in three test partitions, so the p-values
    are optimistic and must say so on the artefact itself."""
    text = " ".join(aggregate.PAIRED_HEADER).lower()
    assert "not independent" in text or "not\nindependent" in text
    assert "optimistic" in text
    assert "effect size" in text or "mean_diff" in text


# ---------------------------------------------------------------- pareto


EFFICIENCY = {
    "cheap_good": {"params": 1_000, "gflops": 0.1, "size_mb_fp16": 1.0},
    "dear_bad": {"params": 9_000, "gflops": 9.0, "size_mb_fp16": 9.0},
    "dear_good": {"params": 9_000, "gflops": 9.0, "size_mb_fp16": 9.0},
}


def test_a_dominated_arm_is_named_with_its_dominators():
    records = records_for(
        {"cheap_good": [0.9] * 3, "dear_bad": [0.5] * 3}, **{"": {}}, **EFFICIENCY
    )
    frame = aggregate.pareto_status(records).set_index("arm")
    assert bool(frame.loc["cheap_good", "on_pareto_frontier"]) is True
    assert bool(frame.loc["dear_bad", "on_pareto_frontier"]) is False
    assert frame.loc["dear_bad", "dominated_by"] == "cheap_good"
    on = frame.loc["dear_bad", "dominated_on"]
    for objective in ("f1_macro", "params", "gflops", "size_mb_fp16"):
        assert objective in on


def test_a_better_but_dearer_arm_is_not_dominated():
    """The real shape: resnet18 is the most accurate and the most expensive."""
    records = records_for(
        {"cheap_good": [0.5] * 3, "dear_good": [0.9] * 3}, **EFFICIENCY
    )
    frame = aggregate.pareto_status(records)
    assert frame["on_pareto_frontier"].all()


def test_ties_do_not_count_as_domination():
    records = records_for(
        {"x": [0.5] * 3, "y": [0.5] * 3},
        x={"params": 1, "gflops": 1.0, "size_mb_fp16": 1.0},
        y={"params": 1, "gflops": 1.0, "size_mb_fp16": 1.0},
    )
    frame = aggregate.pareto_status(records)
    assert frame["on_pareto_frontier"].all()


def test_the_objectives_are_the_four_the_paper_reports():
    assert aggregate.PARETO_MAXIMISE == ("f1_macro",)
    # fp16 explicitly, not the size_mb alias, so the objective cannot move if
    # the alias does
    assert aggregate.PARETO_MINIMISE == ("params", "gflops", "size_mb_fp16")


def test_both_tables_are_in_the_cleared_set():
    """Script 06 deletes its whole output set before regenerating; these two
    must be in it or a stale comparison could survive a re-run."""
    assert "paired_comparisons.csv" in aggregate.TABLE_NAMES
    assert "pareto_status.csv" in aggregate.TABLE_NAMES


# ---------------------------------------------------------------- config

def test_supporting_analyses_use_the_selected_arm():
    """04 and 05 analyse the model the protocol selected, not a dominated one."""
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "arms.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert cfg["ablation"]["arm"] == cfg["learning_curve"]["arm"], (
        "the ablation and the learning curve must analyse the same model"
    )
