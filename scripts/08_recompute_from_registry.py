#!/usr/bin/env python
"""08 -- two reviewer findings answered from the registry alone.

    python scripts/08_recompute_from_registry.py
    python scripts/08_recompute_from_registry.py --dry-run
    python scripts/08_recompute_from_registry.py --verify-reference

No training, no images, no GPU, no DATA_ROOT. Every number below is recomputed
from `artifacts/registry.jsonl`, because each 03_run_cv record already carries
the per-fold `f1_macro` AND the full 10x10 `confusion_matrix` with its
`class_order`. Nothing existing is rewritten.

A1 -- NADEAU-BENGIO CORRECTED INTERVALS

`artifacts/paired_comparisons.csv` divides the fold-to-fold standard deviation
by sqrt(n). That is the standard error of a mean of INDEPENDENT observations,
and these folds are not independent: 5-fold repeated 3 times puts every image
into three different test partitions, so the training sets of any two folds
overlap heavily and their errors are correlated. The naive interval is
therefore too narrow and the p-value too small.

Nadeau & Bengio (2003) give the correction for exactly this design: the
variance of the mean difference across resampled train/test splits is

    Var_corrected = ( 1/n + n_test/n_train ) * sd^2

rather than (1/n) * sd^2. For 5-fold, n_test/n_train = 1/4, and with n = 15
the interval widens by

    sqrt( (1/15 + 1/4) / (1/15) ) = sqrt(4.75) = 2.1794

Bouckaert & Frank (2004) evaluate this against the alternatives and find it the
best-behaved of the practical corrected tests. It is CONSERVATIVE: it is known
to under-reject rather than over-reject, so a comparison that survives it is
solid and one that does not is simply unproven, not disproven.

The deterministic cost objectives -- params, GFLOPs, model size, measured
latency -- carry NO interval at all. They are architectural constants measured
once, not estimates from a sample, and putting error bars on them would invent
uncertainty that does not exist.

A2 -- MERGED-TAXONOMY RECOMPUTATION

The paper argues the LABEL SPACE, not the model and not the corpus size, is the
binding constraint. That claim is currently indirect. It can be made direct at
zero compute cost: merging two classes inside a stored confusion matrix is
summing the corresponding row pair and column pair, and macro-F1 recomputes
from the result. No retraining, no images, same folds, same models.

THE CONTROL IS THE POINT. Fewer classes raises macro-F1 mechanically -- every
merge removes a class that could be confused with something. So the hypothesis
is only supported if the PROPOSED merges sit at the top of the ranking of all
45 possible single-pair merges while the MEDIAN merge gains almost nothing.
Both numbers are reported, and the median is the one that makes the result
publishable rather than circular.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srpcard import aggregate  # noqa: E402
from srpcard.aggregate import corrected_interval, folds_per_repeat  # noqa: E402
from srpcard.config import artifacts_dir, load_data_config  # noqa: E402

# --------------------------------------------------------------------------
# the hypothesised merges
# --------------------------------------------------------------------------

# Named by class label, never by index: the index order is whatever
# sorted(classes) happened to give, and a rename upstream would silently
# re-point a scenario at the wrong pair.
VIBRATION_PAIR = ("vibration", "severe_vibration")
COLLAPSED_CARD_PAIR = ("pump_leakage", "natural_flowing")

# Every class whose label names vibration. M4 merges all of them at once, which
# is NOT a taxonomy anybody would deploy -- "gas influence with vibration" and
# "pump collision with vibration" are different faults with different remedies.
# It is reported as an UPPER BOUND on what any vibration-side relabelling could
# buy, and labelled as such in the output.
VIBRATION_BEARING = (
    "collide_pump_and_vibration",
    "gas_influence_and_vibration",
    "insufficient_liquid_supply_and_vibration",
    "severe_vibration",
    "vibration",
)

SCENARIO_NOTES = {
    "baseline": "the 10-class taxonomy as trained and reported",
    "M1": "severity merge: vibration and severe_vibration are the same fault "
          "at two intensities, and the card shapes differ by degree",
    "M2": "collapsed-card pair: pump_leakage and natural_flowing both produce "
          "a near-degenerate card with little enclosed area",
    "M3": "M1 and M2 together -- the taxonomy this paper actually proposes",
    "M4": "NOT A PLAUSIBLE TAXONOMY. Every vibration-bearing class merged, "
          "plus M2. Reported as an upper bound only: it fuses faults with "
          "different physical causes and different remedies",
}


def scenario_groups(class_order: list[str]) -> dict[str, list[tuple[str, ...]]]:
    """{scenario: [group, ...]}, where each group is a tuple of class labels.

    Classes not named in any group stay on their own. Built against the
    record's own class_order so a label that does not exist fails loudly here
    rather than silently merging nothing.
    """
    known = set(class_order)
    for name in set(VIBRATION_PAIR) | set(COLLAPSED_CARD_PAIR) | set(VIBRATION_BEARING):
        if name not in known:
            raise SystemExit(
                "Class %r is named by a merge scenario but is not in the "
                "registry's class_order:\n  %s\n"
                "  The scenarios in this script are written against the label "
                "names, so a rename upstream must be reflected here rather than "
                "silently merging the wrong pair." % (name, ", ".join(class_order))
            )
    return {
        "baseline": [],
        "M1": [VIBRATION_PAIR],
        "M2": [COLLAPSED_CARD_PAIR],
        "M3": [VIBRATION_PAIR, COLLAPSED_CARD_PAIR],
        "M4": [VIBRATION_BEARING, COLLAPSED_CARD_PAIR],
    }


# --------------------------------------------------------------------------
# confusion-matrix arithmetic
# --------------------------------------------------------------------------


def macro_f1_from_confusion(matrix: np.ndarray) -> float:
    """Macro-F1 from a confusion matrix, rows true and columns predicted.

    A class with no support AND no predictions scores 0, which is what
    sklearn's `f1_score(average="macro", zero_division=0)` does over an
    explicit `labels` list -- and that is how the recorded f1_macro was
    computed, so the baseline scenario here must reproduce it exactly. The
    --verify-reference mode asserts that it does.
    """
    matrix = np.asarray(matrix, dtype=float)
    true_positive = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    denominator = predicted + actual
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denominator > 0, 2.0 * true_positive / denominator, 0.0)
    return float(f1.mean())


def merge_confusion(
    matrix: np.ndarray, class_order: list[str], groups: list[tuple[str, ...]]
) -> tuple[np.ndarray, list[str]]:
    """Sum the rows and columns of each group into one class.

    Merging classes i and j means every prediction of i for a true j (and the
    reverse) stops being an error, which is exactly what summing both the rows
    and the columns does. The total count is preserved; --verify-reference
    checks that too.
    """
    index = {name: position for position, name in enumerate(class_order)}
    assigned: dict[int, int] = {}
    merged_names: list[str] = []

    for group in groups:
        target = len(merged_names)
        for name in group:
            assigned[index[name]] = target
        merged_names.append("+".join(sorted(group)))

    for position, name in enumerate(class_order):
        if position not in assigned:
            assigned[position] = len(merged_names)
            merged_names.append(name)

    size = len(merged_names)
    matrix = np.asarray(matrix, dtype=float)
    out = np.zeros((size, size), dtype=float)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            out[assigned[i], assigned[j]] += matrix[i, j]
    return out, merged_names


# --------------------------------------------------------------------------
# A1 -- the correction
# --------------------------------------------------------------------------


def protocol_of(records: list[dict]) -> str:
    """The one protocol these records share, or a refusal.

    EVERY artefact written from here on carries `arm` and `protocol` as columns.
    The learning-curve figure captioned yolo26n for weeks after script 05 was
    retargeted at mobilenetv3_small, and nothing could catch it because
    artifacts/learning_curve.csv records neither -- a table that does not say
    which model and which regime produced it cannot be checked against anything.

    Mixing protocols in one table is refused rather than labelled, because the
    legacy augmented runs and the uniform ones are not comparable at all and a
    column would only make the mixture look deliberate.
    """
    protocols = sorted(
        {(r.get("extra") or {}).get("protocol") for r in records}
    )
    if len(protocols) != 1 or protocols[0] is None:
        raise SystemExit(
            "These records span %d protocol(s): %s\n"
            "  Runs under different protocols are not comparable -- see the "
            "hyperparameter\n  drift guard in src/srpcard/registry.py. Filter to one "
            "before aggregating." % (len(protocols), protocols)
        )
    return protocols[0]


def paired_corrected(
    records: list[dict], metrics: tuple[str, ...] = ("f1_macro", "accuracy")
) -> tuple[pd.DataFrame, float, float]:
    """Every pair of arms, every metric, naive and corrected side by side."""
    k = folds_per_repeat(records)
    rho = 1.0 / (k - 1)

    rows = []
    width_factor = float("nan")
    for metric in metrics:
        frame = pd.DataFrame(
            [
                {"arm": r["arm"], "fold": aggregate._fold_key(r), metric: r.get(metric)}
                for r in records
            ]
        ).dropna(subset=[metric])
        wide = frame.pivot(index="fold", columns="arm", values=metric)

        for arm_a, arm_b in itertools.combinations(sorted(wide.columns), 2):
            both = wide[[arm_a, arm_b]].dropna()
            if both.empty:
                continue
            differences = (both[arm_a] - both[arm_b]).to_numpy(dtype=float)
            if differences.mean() < 0:      # orient so mean_diff is positive
                arm_a, arm_b = arm_b, arm_a
                differences = -differences

            result = corrected_interval(differences, rho)
            n = result["n_folds"]
            width_factor = float(np.sqrt((1.0 / n + rho) / (1.0 / n)))
            rows.append(
                {
                    "metric": metric,
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    **result,
                    "k_folds": k,
                    "n_test_over_n_train": rho,
                    "width_factor": width_factor,
                }
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["metric", "mean_diff"], ascending=[True, False]
        ).reset_index(drop=True)
        for column in frame.columns:
            if frame[column].dtype.kind == "f":
                frame[column] = frame[column].round(6)
    return frame, rho, width_factor


# --------------------------------------------------------------------------
# A2 -- the merges
# --------------------------------------------------------------------------


def per_fold_merged_f1(records: list[dict]) -> pd.DataFrame:
    """One row per (arm, fold, scenario) with the merged macro-F1."""
    if not records:
        return pd.DataFrame()
    class_order = list(records[0]["class_order"])
    scenarios = scenario_groups(class_order)

    rows = []
    for record in records:
        if record.get("confusion_matrix") is None:
            continue
        if list(record["class_order"]) != class_order:
            raise SystemExit(
                "Records disagree on class_order. Merging by name across two "
                "different label spaces would compare different quantities.\n"
                "  %s\n  %s" % (class_order, record["class_order"])
            )
        matrix = np.asarray(record["confusion_matrix"], dtype=float)
        for name, groups in scenarios.items():
            merged, merged_names = merge_confusion(matrix, class_order, groups)
            rows.append(
                {
                    "arm": record["arm"],
                    "fold": aggregate._fold_key(record),
                    "repeat": record.get("repeat"),
                    "fold_index": record.get("fold"),
                    "scenario": name,
                    "n_classes": len(merged_names),
                    "macro_f1": macro_f1_from_confusion(merged),
                    "recorded_f1_macro": record.get("f1_macro"),
                    "support": float(matrix.sum()),
                }
            )
    return pd.DataFrame(rows)


def taxonomy_summary(folds: pd.DataFrame, rho: float) -> pd.DataFrame:
    """Scenario x arm: mean and sd over folds, plus the corrected gain interval.

    The gain interval is PAIRED -- the same fold under two taxonomies -- and it
    carries the same Nadeau-Bengio correction as A1, because it is computed over
    the same overlapping folds.
    """
    if folds.empty:
        return pd.DataFrame()

    baseline = folds[folds["scenario"] == "baseline"].set_index(["arm", "fold"])[
        "macro_f1"
    ]
    rows = []
    for (arm, scenario), block in folds.groupby(["arm", "scenario"], sort=True):
        values = block["macro_f1"].to_numpy(dtype=float)
        row = {
            "arm": arm,
            "scenario": scenario,
            "n_classes": int(block["n_classes"].iloc[0]),
            "n_folds": len(values),
            "macro_f1_mean": float(values.mean()),
            "macro_f1_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "plausible_taxonomy": scenario != "M4",
            "note": SCENARIO_NOTES.get(scenario, ""),
        }
        if scenario != "baseline":
            aligned = block.set_index(["arm", "fold"])["macro_f1"]
            gain = (aligned - baseline.reindex(aligned.index)).to_numpy(dtype=float)
            stats_block = corrected_interval(gain, rho)
            row.update(
                {
                    "gain_mean": stats_block["mean_diff"],
                    "gain_ci95_corrected_low": stats_block["ci95_corrected_low"],
                    "gain_ci95_corrected_high": stats_block["ci95_corrected_high"],
                    "gain_p_corrected": stats_block["p_corrected"],
                    "gain_corrected_excludes_zero":
                        stats_block["corrected_excludes_zero"],
                }
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    order = {"baseline": 0, "M1": 1, "M2": 2, "M3": 3, "M4": 4}
    frame["_order"] = frame["scenario"].map(order).fillna(99)
    frame = frame.sort_values(["arm", "_order"]).drop(columns="_order")
    for column in frame.columns:
        if frame[column].dtype.kind == "f":
            frame[column] = frame[column].round(6)
    return frame.reset_index(drop=True)


def all_pair_merges(records: list[dict]) -> pd.DataFrame:
    """Every one of the 45 possible single-pair merges, ranked per arm.

    This is the control. Merging any two classes raises macro-F1 for free, so
    the hypothesised pairs mean nothing until they are shown to sit above the
    rest of the distribution rather than inside it.
    """
    if not records:
        return pd.DataFrame()
    class_order = list(records[0]["class_order"])
    hypothesised = {
        frozenset(VIBRATION_PAIR): "M1",
        frozenset(COLLAPSED_CARD_PAIR): "M2",
    }

    by_arm: dict[str, list[dict]] = {}
    for record in records:
        if record.get("confusion_matrix") is not None:
            by_arm.setdefault(record["arm"], []).append(record)

    rows = []
    for arm, arm_records in sorted(by_arm.items()):
        matrices = [
            np.asarray(r["confusion_matrix"], dtype=float) for r in arm_records
        ]
        baseline = np.array(
            [macro_f1_from_confusion(m) for m in matrices], dtype=float
        )
        for left, right in itertools.combinations(class_order, 2):
            merged = np.array(
                [
                    macro_f1_from_confusion(
                        merge_confusion(m, class_order, [(left, right)])[0]
                    )
                    for m in matrices
                ],
                dtype=float,
            )
            gain = merged - baseline
            rows.append(
                {
                    "arm": arm,
                    "class_a": left,
                    "class_b": right,
                    "n_folds": len(merged),
                    "macro_f1_mean": float(merged.mean()),
                    "macro_f1_sd": float(merged.std(ddof=1)) if len(merged) > 1 else 0.0,
                    "gain_mean": float(gain.mean()),
                    "hypothesised": hypothesised.get(frozenset((left, right)), ""),
                }
            )

    frame = pd.DataFrame(rows)
    frame["rank_in_arm"] = (
        frame.groupby("arm")["gain_mean"].rank(ascending=False, method="min").astype(int)
    )
    frame = frame.sort_values(["arm", "gain_mean"], ascending=[True, False])
    for column in frame.columns:
        if frame[column].dtype.kind == "f":
            frame[column] = frame[column].round(6)
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

TOLERANCE = 1e-4

REFERENCE_WIDTH_FACTOR = 2.1794

REFERENCE_PAIRS = {
    # (arm_a, arm_b): (mean_diff, corrected_low, corrected_high)
    ("resnet18", "yolo26s"): (0.0981, 0.0257, 0.1705),
    ("resnet18", "yolo26n"): (0.0789, 0.0216, 0.1363),
    ("mobilenetv3_small", "yolo26s"): (0.0861, 0.0019, 0.1703),
    ("mobilenetv3_small", "yolo26n"): (0.0669, -0.0446, 0.1784),
    ("resnet18", "yolo26m"): (0.0719, -0.0082, 0.1520),
    ("mobilenetv3_small", "yolo26m"): (0.0599, -0.0392, 0.1591),
}

REFERENCE_SCENARIOS = {
    ("mobilenetv3_small", "baseline"): 0.5900,
    ("mobilenetv3_small", "M1"): 0.6263,
    ("mobilenetv3_small", "M2"): 0.6293,
    ("mobilenetv3_small", "M3"): 0.6751,
    ("mobilenetv3_small", "M4"): 0.7102,
}

REFERENCE_M3_GAIN = {
    "mobilenetv3_small": 0.0851,
    "resnet18": 0.0885,
    "yolo26m": 0.0682,
    "yolo26n": 0.0595,
    "yolo26s": 0.0842,
}

REFERENCE_PAIR_MEDIAN = {"mobilenetv3_small": 0.5931}


class VerificationFailed(SystemExit):
    pass


def _check(failures: list[str], label: str, expected: float, actual, tol=TOLERANCE):
    if actual is None or (isinstance(actual, float) and actual != actual):
        failures.append("%-52s expected %+.4f   actual MISSING" % (label, expected))
        return
    if abs(float(actual) - expected) > tol:
        failures.append(
            "%-52s expected %+.4f   actual %+.4f   diff %.2e"
            % (label, expected, float(actual), abs(float(actual) - expected))
        )


def verify(paired: pd.DataFrame, summary: pd.DataFrame, pairs: pd.DataFrame,
           folds: pd.DataFrame, width_factor: float) -> int:
    """Recompute the reference values and assert. Non-zero exit on any failure."""
    rule("VERIFY -- reference values recomputed from the registry")
    failures: list[str] = []

    _check(failures, "interval-width correction factor",
           REFERENCE_WIDTH_FACTOR, width_factor)

    f1_pairs = paired[paired["metric"] == "f1_macro"]
    for (arm_a, arm_b), (mean, low, high) in REFERENCE_PAIRS.items():
        row = f1_pairs[(f1_pairs["arm_a"] == arm_a) & (f1_pairs["arm_b"] == arm_b)]
        if row.empty:
            failures.append("%-52s PAIR ABSENT from the computed table"
                            % ("%s - %s" % (arm_a, arm_b)))
            continue
        row = row.iloc[0]
        label = "%s - %s" % (arm_a, arm_b)
        _check(failures, label + " mean", mean, row["mean_diff"])
        _check(failures, label + " corrected CI low", low, row["ci95_corrected_low"])
        _check(failures, label + " corrected CI high", high, row["ci95_corrected_high"])

    for (arm, scenario), expected in REFERENCE_SCENARIOS.items():
        row = summary[(summary["arm"] == arm) & (summary["scenario"] == scenario)]
        _check(failures, "%s %s macro-F1" % (arm, scenario), expected,
               None if row.empty else row.iloc[0]["macro_f1_mean"])

    for arm, expected in REFERENCE_M3_GAIN.items():
        row = summary[(summary["arm"] == arm) & (summary["scenario"] == "M3")]
        _check(failures, "%s M3 gain" % arm, expected,
               None if row.empty else row.iloc[0].get("gain_mean"))

    for arm, expected in REFERENCE_PAIR_MEDIAN.items():
        block = pairs[pairs["arm"] == arm]
        _check(failures, "%s median over all 45 merges" % arm, expected,
               None if block.empty else float(block["macro_f1_mean"].median()))
        if len(block) != 45:
            failures.append(
                "%-52s expected 45 pairs   actual %d" % ("%s pair count" % arm, len(block))
            )

    # The top six merges must be exactly the pairs drawn from the four
    # hypothesised classes. This is the claim; if it fails the paper is wrong,
    # not the code.
    for arm in REFERENCE_PAIR_MEDIAN:
        block = pairs[pairs["arm"] == arm].nsmallest(6, "rank_in_arm")
        four = set(VIBRATION_PAIR) | set(COLLAPSED_CARD_PAIR)
        outside = [
            "%s+%s" % (r.class_a, r.class_b)
            for r in block.itertuples()
            if not ({r.class_a, r.class_b} <= four)
        ]
        if outside:
            failures.append(
                "%-52s expected all 6 drawn from %s; found %s"
                % ("%s top-6 merges" % arm, sorted(four), outside)
            )

    # An internal consistency check that costs nothing and catches a whole class
    # of error: the baseline scenario must reproduce the RECORDED f1_macro.
    baseline = folds[folds["scenario"] == "baseline"]
    drift = (baseline["macro_f1"] - baseline["recorded_f1_macro"]).abs().max()
    if drift > 1e-9:
        failures.append(
            "%-52s expected  0.0000   actual %.2e"
            % ("baseline-from-confusion vs recorded f1_macro", drift)
        )

    if failures:
        print("\n  %d CHECK(S) FAILED:\n" % len(failures))
        for line in failures:
            print("    " + line)
        print(
            "\n  These reference values were computed independently from the same\n"
            "  registry. A mismatch means this implementation and that one do not\n"
            "  agree, and the difference must be understood before either number\n"
            "  reaches the manuscript. The tolerance is %g and is not negotiable."
            % TOLERANCE
        )
        return 1

    print("\n  All %d reference value(s) reproduced within %g."
          % (3 * len(REFERENCE_PAIRS) + len(REFERENCE_SCENARIOS)
             + len(REFERENCE_M3_GAIN) + 2 * len(REFERENCE_PAIR_MEDIAN) + 2,
             TOLERANCE))
    return 0


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


PAIRED_CORRECTED_HEADER = [
    "Paired per-fold differences with the NADEAU-BENGIO correction.",
    "",
    "THIS FILE SUPERSEDES paired_comparisons.csv FOR INFERENCE.",
    "",
    "The 15 folds are 5-fold CV repeated 3 times, so every image appears in",
    "three test partitions and the folds are not independent. The naive standard",
    "error sd/sqrt(n) assumes they are, and is therefore too small.",
    "",
    "Nadeau & Bengio (2003), 'Inference for the Generalization Error',",
    "Machine Learning 52(3):239-281, replace the factor 1/n by",
    "",
    "    1/n + n_test/n_train",
    "",
    "which for 5-fold (n_test/n_train = 1/4) and n = 15 widens every interval by",
    "sqrt((1/15 + 1/4)/(1/15)) = 2.1794. Bouckaert & Frank (2004), 'Evaluating",
    "the Replicability of Significance Tests for Comparing Learning Algorithms',",
    "PAKDD, find it the best-behaved of the practical corrected tests.",
    "",
    "The correction is CONSERVATIVE: it under-rejects rather than over-rejects.",
    "A comparison whose corrected interval excludes zero is solid; one whose",
    "interval now includes zero is UNPROVEN, not disproven.",
    "",
    "The deterministic cost objectives -- params, GFLOPs, model size and measured",
    "latency -- carry NO interval. They are architectural constants measured once,",
    "not estimates from a sample.",
]

TAXONOMY_HEADER = [
    "Macro-F1 recomputed after merging classes INSIDE the stored confusion",
    "matrices. No retraining: merging classes i and j sums rows i,j and columns",
    "i,j, which is exactly what it means for a confusion between them to stop",
    "being an error. Same folds, same models, same predictions.",
    "",
    "READ THIS WITH taxonomy_merge_pairs.csv. Fewer classes raises macro-F1",
    "mechanically, so a gain here means nothing on its own. The control is the",
    "ranking of all 45 possible single-pair merges: the claim holds only if the",
    "hypothesised pairs sit at the TOP of that ranking while the MEDIAN merge",
    "gains almost nothing.",
    "",
    "M4 is not a plausible taxonomy and is labelled as such -- it fuses faults",
    "with different physical causes. It is an upper bound, not a proposal.",
    "",
    "gain_* columns carry the Nadeau-Bengio correction, because baseline and",
    "merged are measured on the same overlapping folds.",
]

PAIRS_HEADER = [
    "All 45 single-pair merges, ranked by mean macro-F1 gain, per arm.",
    "",
    "THIS IS THE CONTROL for taxonomy_merge.csv. Merging any two of ten classes",
    "raises macro-F1 for free. The hypothesis is that the proposed pairs are not",
    "merely above zero but above the REST OF THIS DISTRIBUTION.",
    "",
    "`hypothesised` names the scenario a pair belongs to, and is blank for the",
    "other 43.",
]

FOLDS_HEADER = [
    "Per-fold merged macro-F1, the input to taxonomy_merge.csv.",
    "",
    "Kept so any scenario pair can be given a corrected paired interval without",
    "recomputing the merges. `recorded_f1_macro` is the value the run itself",
    "wrote; the baseline scenario must reproduce it exactly, and",
    "--verify-reference asserts that it does.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be written, write nothing")
    parser.add_argument("--verify-reference", action="store_true",
                        help="recompute the reference values and assert; "
                             "non-zero exit on any mismatch")
    parser.add_argument("--registry", default=None,
                        help="override artifacts/registry.jsonl")
    args = parser.parse_args()

    rule("08 -- recompute from the registry (no training, no images)")
    path = Path(args.registry) if args.registry else None
    records = aggregate.cv_records(path)
    if not records:
        print("  No 03_run_cv records. Nothing to do.")
        return 1

    out = artifacts_dir(load_data_config())
    arms = sorted({r["arm"] for r in records})
    print("  records : %d from 03_run_cv" % len(records))
    print("  arms    : %s" % ", ".join(arms))

    # ---- A1
    rule("A1 -- Nadeau-Bengio corrected intervals")
    paired, rho, width_factor = paired_corrected(records)
    k = folds_per_repeat(records)
    print("  k folds per repeat        : %d" % k)
    print("  n_test/n_train            : %.6f  (1/%d)" % (rho, k - 1))
    print("  interval-width factor     : %.4f" % width_factor)
    print()
    f1_rows = paired[paired["metric"] == "f1_macro"]
    print("  %-38s %9s %21s %21s" % ("comparison (f1_macro)", "mean", "naive 95% CI",
                                     "corrected 95% CI"))
    lost = 0
    for row in f1_rows.itertuples():
        changed = row.naive_excludes_zero and not row.corrected_excludes_zero
        lost += int(changed)
        print("  %-38s %+9.4f  [%+.4f, %+.4f]  [%+.4f, %+.4f] %s"
              % ("%s - %s" % (row.arm_a, row.arm_b), row.mean_diff,
                 row.ci95_naive_low, row.ci95_naive_high,
                 row.ci95_corrected_low, row.ci95_corrected_high,
                 "  <- loses significance" if changed else ""))
    print("\n  %d of %d comparisons that the naive interval called significant no"
          % (lost, len(f1_rows)))
    print("  longer exclude zero. That is the correction working, not a defect.")

    # ---- A2
    rule("A2 -- merged-taxonomy recomputation")
    folds = per_fold_merged_f1(records)
    summary = taxonomy_summary(folds, rho)
    pairs = all_pair_merges(records)

    print("  %-20s %-10s %8s %10s %12s" % ("arm", "scenario", "classes",
                                           "macro-F1", "gain"))
    for row in summary.itertuples():
        gain = getattr(row, "gain_mean", float("nan"))
        print("  %-20s %-10s %8d %10.4f %12s%s"
              % (row.arm, row.scenario, row.n_classes, row.macro_f1_mean,
                 "%+.4f" % gain if gain == gain else "--",
                 "   (upper bound only)" if row.scenario == "M4" else ""))

    print("\n  CONTROL -- all 45 single-pair merges")
    print("  %-20s %10s %10s %10s   %s" % ("arm", "baseline", "median", "median gain",
                                           "top-3 pairs"))
    for arm in arms:
        block = pairs[pairs["arm"] == arm]
        base = summary[(summary["arm"] == arm) & (summary["scenario"] == "baseline")]
        if block.empty or base.empty:
            continue
        baseline_value = float(base.iloc[0]["macro_f1_mean"])
        median = float(block["macro_f1_mean"].median())
        top = block.nsmallest(3, "rank_in_arm")
        print("  %-20s %10.4f %10.4f %+10.4f   %s"
              % (arm, baseline_value, median, median - baseline_value,
                 ", ".join("%s+%s" % (r.class_a, r.class_b) for r in top.itertuples())))

    if args.verify_reference:
        return verify(paired, summary, pairs, folds, width_factor)

    # Every artefact carries the protocol it was built under, beside the arm it
    # describes. See protocol_of().
    protocol = protocol_of(records)
    targets = {
        "paired_comparisons_corrected.csv": (paired, PAIRED_CORRECTED_HEADER),
        "taxonomy_merge.csv": (summary, TAXONOMY_HEADER),
        "taxonomy_merge_pairs.csv": (pairs, PAIRS_HEADER),
        "taxonomy_merge_folds.csv": (folds, FOLDS_HEADER),
    }
    for frame, _ in targets.values():
        if not frame.empty:
            frame["protocol"] = protocol

    rule("DRY RUN -- nothing written" if args.dry_run else "WRITING")
    stamp = aggregate.provenance(records)
    aggregate.assert_provenance_covers(stamp, records)
    for name, (frame, header) in targets.items():
        if args.dry_run:
            print("  would write %-38s %d row(s)" % (name, len(frame)))
            continue
        aggregate.write_csv_with_provenance(frame, out / name, stamp,
                                            extra_header=header)
        print("  wrote %-42s %d row(s)" % (name, len(frame)))

    if not args.dry_run:
        print("\n  Run again with --verify-reference to assert the reference values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
