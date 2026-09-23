#!/usr/bin/env python
"""11 -- was YOLO26 handicapped? The epoch budget, the optimizer, the recipe.

    python scripts/11_recipe_check.py
    python scripts/11_recipe_check.py --dry-run

CPU only. No training, no images. Reads the registry and writes two tables.

THE CONFOUND THIS EXISTS TO BREAK
---------------------------------
configs/arms.yaml sets `optimizer: MuSGD` on all three YOLO arms and `SGD` on
both baselines. So across the published comparison the optimizer is CONFOUNDED
with the architecture family: every YOLO run used MuSGD, every baseline used
SGD, and "YOLO26 loses" cannot be separated from "MuSGD loses" by looking at
those records alone.

Running YOLO with SGD does not break that. It only says what happens when the
YOLO family loses MuSGD. The cross has to go both ways:

                        MuSGD                    SGD
    yolo26n             published arm            contrast
    mobilenetv3_small   contrast                 published arm

Four cells on repeat 0's five folds, plus the native-recipe run as a fifth row.
If MobileNet under MuSGD still beats yolo26n under MuSGD, the confound is
answered directly and the negative finding stands. If it does not, that must be
known before submission rather than after.

PREPROCESSING IS NOT HELD CONSTANT ACROSS EVERY ROW, and the table says so in
its own column. The four uniform cells are `letterbox_224`; the native row is
`ultralytics_default`, because evaluating a natively trained model through our
letterbox would show it a transform it never trained on and manufacture the
result the run exists to test. Read the native row as "their recipe end to end",
never as one variable changed.

THE EPOCH BUDGET (C1) is a separate question with its own table. The grid
locked yolo26n at 25 epochs while every other arm got 50, and it is the only
arm whose selected epochs cluster near its budget. `yolo26n_ep50` is the same
arm with the budget raised, over all 15 folds.

Every interval carries the Nadeau-Bengio correction from
src/srpcard/aggregate.py, because these are the same overlapping folds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srpcard import aggregate  # noqa: E402
from srpcard.aggregate import corrected_interval, folds_per_repeat  # noqa: E402
from srpcard.config import artifacts_dir, load_data_config  # noqa: E402
from srpcard.registry import load_registry  # noqa: E402

PUBLISHED = "03_run_cv"
CONTRAST = "03b_contrast"
NATIVE = "03c_native_recipe"

LETTERBOX = "letterbox_224"
ULTRALYTICS = "ultralytics_default"

# The 2x2, plus the native row. Each cell names where its records come from.
CELLS = [
    {"cell": "yolo26n / MuSGD", "arm": "yolo26n", "optimizer": "musgd",
     "script": PUBLISHED, "override": None, "role": "published arm",
     "preprocessing": LETTERBOX},
    {"cell": "yolo26n / SGD", "arm": "yolo26n", "optimizer": "sgd",
     "script": CONTRAST, "override": "sgd", "role": "contrast",
     "preprocessing": LETTERBOX},
    {"cell": "mobilenetv3_small / SGD", "arm": "mobilenetv3_small",
     "optimizer": "sgd", "script": PUBLISHED, "override": None,
     "role": "published arm", "preprocessing": LETTERBOX},
    {"cell": "mobilenetv3_small / MuSGD", "arm": "mobilenetv3_small",
     "optimizer": "musgd", "script": CONTRAST, "override": "musgd",
     "role": "contrast", "preprocessing": LETTERBOX},
    {"cell": "yolo26n / native recipe", "arm": "yolo26n", "optimizer": "native",
     "script": NATIVE, "override": None, "role": "native recipe",
     "preprocessing": ULTRALYTICS},
]


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def _override_of(record: dict) -> str | None:
    value = (record.get("extra") or {}).get("run_id_optimizer")
    return str(value).lower() if value else None


def cell_records(records: list[dict], cell: dict, repeat: int) -> list[dict]:
    """The records for one cell of the 2x2, restricted to one repeat."""
    return [
        r for r in records
        if r.get("script") == cell["script"]
        and r.get("arm") == cell["arm"]
        and r.get("repeat") == repeat
        and _override_of(r) == cell["override"]
    ]


def fold_series(records: list[dict], metric: str = "f1_macro") -> dict[int, float]:
    return {
        int(r["fold"]): float(r[metric])
        for r in records
        if r.get("fold") is not None and r.get(metric) is not None
    }


# --------------------------------------------------------------------------
# the 2x2
# --------------------------------------------------------------------------


def recipe_table(records: list[dict], repeat: int, rho: float) -> pd.DataFrame:
    """One row per cell: per-fold values, mean, and how it compares."""
    series = {cell["cell"]: fold_series(cell_records(records, cell, repeat))
              for cell in CELLS}

    baseline = series.get("yolo26n / MuSGD") or {}
    rows = []
    for cell in CELLS:
        values = series[cell["cell"]]
        row = {
            "cell": cell["cell"],
            "arm": cell["arm"],
            "optimizer": cell["optimizer"],
            "role": cell["role"],
            "script": cell["script"],
            "preprocessing": cell["preprocessing"],
            "protocol": "native" if cell["script"] == NATIVE else "uniform",
            "repeat": repeat,
            "n_folds": len(values),
            "f1_macro_mean": float(np.mean(list(values.values()))) if values else None,
            "f1_macro_sd": (float(np.std(list(values.values()), ddof=1))
                            if len(values) > 1 else None),
        }
        for fold in range(5):
            row["fold%d" % fold] = values.get(fold)

        # against the published YOLO cell, on the folds both completed
        shared = sorted(set(values) & set(baseline))
        if shared and cell["cell"] != "yolo26n / MuSGD":
            difference = np.array([values[f] - baseline[f] for f in shared], dtype=float)
            stats = corrected_interval(difference, rho)
            row.update({
                "vs_yolo26n_musgd_mean": stats["mean_diff"],
                "vs_yolo26n_musgd_ci_low": stats["ci95_corrected_low"],
                "vs_yolo26n_musgd_ci_high": stats["ci95_corrected_high"],
                "vs_yolo26n_musgd_excludes_zero": stats["corrected_excludes_zero"],
                "vs_yolo26n_musgd_n_folds": len(shared),
            })
        rows.append(row)

    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if frame[column].dtype.kind == "f":
            frame[column] = frame[column].round(6)
    return frame


def confound_verdict(frame: pd.DataFrame) -> list[str]:
    """State the conclusion plainly, or say plainly that it cannot be stated."""
    def mean_of(cell):
        row = frame[frame["cell"] == cell]
        if row.empty:
            return None
        value = row.iloc[0]["f1_macro_mean"]
        return None if pd.isna(value) else float(value)

    yolo_musgd = mean_of("yolo26n / MuSGD")
    yolo_sgd = mean_of("yolo26n / SGD")
    mobile_sgd = mean_of("mobilenetv3_small / SGD")
    mobile_musgd = mean_of("mobilenetv3_small / MuSGD")

    if None in (yolo_musgd, mobile_musgd):
        return [
            "CANNOT CONCLUDE YET. The 2x2 is incomplete:",
            "  yolo26n / MuSGD            %s" % ("present" if yolo_musgd else "MISSING"),
            "  mobilenetv3_small / MuSGD  %s" % ("present" if mobile_musgd else "MISSING"),
            "",
            "  The cell that answers the confound is MobileNet under MuSGD. Without",
            "  it, running YOLO with SGD only says what happens when the YOLO family",
            "  loses MuSGD -- it does not separate architecture from optimizer.",
        ]

    lines = []
    if mobile_musgd > yolo_musgd:
        lines += [
            "THE NEGATIVE FINDING SURVIVES THE CONFOUND.",
            "",
            "  Held at the SAME optimizer (MuSGD), mobilenetv3_small scores %.4f"
            % mobile_musgd,
            "  against yolo26n's %.4f, a gap of %+.4f. The accuracy difference"
            % (yolo_musgd, mobile_musgd - yolo_musgd),
            "  therefore tracks the ARCHITECTURE, not the optimizer, and the paper",
            "  can say so directly instead of arguing it.",
        ]
    else:
        lines += [
            "THE NEGATIVE FINDING DOES NOT SURVIVE THE CONFOUND AS STATED.",
            "",
            "  Held at the SAME optimizer (MuSGD), mobilenetv3_small scores %.4f"
            % mobile_musgd,
            "  against yolo26n's %.4f. MobileNet's published advantage does not"
            % yolo_musgd,
            "  reproduce once the optimizer is held constant, which means the",
            "  comparison was measuring the optimizer at least in part.",
            "  THIS NEEDS TO BE IN THE MANUSCRIPT BEFORE SUBMISSION.",
        ]

    if yolo_sgd is not None:
        lines += [
            "",
            "  yolo26n loses %+.4f by giving up MuSGD (%.4f -> %.4f), so MuSGD is"
            % (yolo_sgd - yolo_musgd, yolo_musgd, yolo_sgd),
            "  doing real work for it -- which is why removing it alone would not",
            "  have answered anything.",
        ]
    if mobile_sgd is not None and mobile_musgd is not None:
        lines += [
            "  mobilenetv3_small moves %+.4f under MuSGD (%.4f -> %.4f)."
            % (mobile_musgd - mobile_sgd, mobile_sgd, mobile_musgd),
        ]
    return lines


# --------------------------------------------------------------------------
# the epoch budget (C1)
# --------------------------------------------------------------------------


def epoch_budget_table(records: list[dict], rho: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """yolo26n_ep50 against yolo26n and mobilenetv3_small, plus appendix C."""
    ep50 = [r for r in records
            if r.get("script") == CONTRAST and r.get("arm") == "yolo26n_ep50"]
    if not ep50:
        return pd.DataFrame(), pd.DataFrame()

    def keyed(rows):
        return {(r.get("repeat"), r.get("fold")): float(r["f1_macro"])
                for r in rows if r.get("f1_macro") is not None}

    subject = keyed(ep50)
    rows = []
    for other in ("yolo26n", "mobilenetv3_small"):
        reference = keyed([r for r in records
                           if r.get("script") == PUBLISHED and r.get("arm") == other])
        shared = sorted(set(subject) & set(reference))
        if not shared:
            continue
        difference = np.array([subject[k] - reference[k] for k in shared], dtype=float)
        stats = corrected_interval(difference, rho)
        rows.append({
            "arm_a": "yolo26n_ep50",
            "arm_b": other,
            "protocol": "uniform",
            "n_folds": len(shared),
            "mean_diff": stats["mean_diff"],
            "ci95_corrected_low": stats["ci95_corrected_low"],
            "ci95_corrected_high": stats["ci95_corrected_high"],
            "p_corrected": stats["p_corrected"],
            "corrected_excludes_zero": stats["corrected_excludes_zero"],
            "a_wins": stats["a_wins"],
            "b_wins": stats["b_wins"],
        })

    # appendix C form, for both budgets side by side
    epochs = aggregate.selected_epoch_distribution(
        [r for r in records
         if (r.get("script") == PUBLISHED and r.get("arm") == "yolo26n")
         or (r.get("script") == CONTRAST and r.get("arm") == "yolo26n_ep50")]
    )
    summary = pd.DataFrame()
    if not epochs.empty:
        grouped = epochs.groupby("arm")["fraction_of_budget"]
        summary = pd.DataFrame({
            "arm": grouped.mean().index,
            "protocol": "uniform",
            "epoch_budget": epochs.groupby("arm")["epoch_budget"].max().values,
            "n_folds": grouped.count().values,
            "mean_fraction_of_budget": grouped.mean().round(4).values,
            "median_fraction_of_budget": grouped.median().round(4).values,
            "folds_at_or_above_96pct": epochs.assign(
                hit=epochs["fraction_of_budget"] >= 0.96
            ).groupby("arm")["hit"].sum().values,
        })

    return pd.DataFrame(rows), summary


# --------------------------------------------------------------------------

RECIPE_HEADER = [
    "Was YOLO26 handicapped? The 2x2 that breaks the optimizer/architecture",
    "confound, plus the native-recipe run.",
    "",
    "configs/arms.yaml sets optimizer: MuSGD on all three YOLO arms and SGD on",
    "both baselines, so in the published comparison the optimizer is confounded",
    "with the architecture family. Running YOLO with SGD does not break that --",
    "it only says what happens when YOLO loses MuSGD. The cross goes both ways:",
    "",
    "                        MuSGD                SGD",
    "    yolo26n             published            contrast",
    "    mobilenetv3_small   contrast             published",
    "",
    "PREPROCESSING IS NOT CONSTANT ACROSS EVERY ROW. The four uniform cells are",
    "letterbox_224; the native row is ultralytics_default, because evaluating a",
    "natively trained model through our letterbox would show it a transform it",
    "never trained on and manufacture the result that run exists to test. The",
    "native row is 'their recipe end to end', NOT one variable changed. Do not",
    "read this table as an apples-to-apples preprocessing comparison.",
    "",
    "Intervals carry the Nadeau-Bengio correction; these are overlapping folds.",
]

BUDGET_HEADER = [
    "C1: yolo26n with the epoch budget raised from 25 to 50.",
    "",
    "The uniform grid locked nano at 25 epochs while every other arm got 50, and",
    "nano is the only arm whose selected epochs cluster near its budget. This",
    "separates 'YOLO26 loses' from 'YOLO26 was stopped early'.",
    "",
    "yolo26n_ep50 is NOT a grid winner and must not be described as one: it is",
    "yolo26n with one number changed. The locked arm stays locked, because",
    "raising the budget after seeing the CV result would be selection on test.",
    "",
    "If ep50 STILL saturates its budget -- a high mean fraction and many folds at",
    ">= 96 % -- that is a different finding from '25 was simply too few', and the",
    "manuscript needs to say which.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=0,
                        help="which repeat the 2x2 is computed over (default: 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is present and exit, writing nothing")
    args = parser.parse_args()

    rule("11 -- recipe and epoch-budget checks")
    records = load_registry()
    out = artifacts_dir(load_data_config())

    cv = [r for r in records if r.get("script") == PUBLISHED]
    rho = 1.0 / (folds_per_repeat(cv) - 1) if cv else 0.25
    print("  registry : %d record(s)" % len(records))
    print("  rho      : %.6f  (n_test/n_train)" % rho)

    rule("the 2x2, repeat %d" % args.repeat)
    recipe = recipe_table(records, args.repeat, rho)
    print("  %-30s %-20s %6s %10s %14s"
          % ("cell", "preprocessing", "folds", "macro-F1", "vs yolo/MuSGD"))
    for row in recipe.itertuples():
        delta = getattr(row, "vs_yolo26n_musgd_mean", None)
        print("  %-30s %-20s %6d %10s %14s"
              % (row.cell, row.preprocessing, row.n_folds,
                 "%.4f" % row.f1_macro_mean if row.f1_macro_mean == row.f1_macro_mean else "--",
                 "%+.4f" % delta if delta is not None and delta == delta else "--"))

    print()
    for line in confound_verdict(recipe):
        print("  " + line)

    rule("C1 -- the epoch budget")
    budget, epochs_summary = epoch_budget_table(records, rho)
    if budget.empty:
        print("  No yolo26n_ep50 records yet. Run:")
        print("    python scripts/03_run_cv.py --contrast --arms yolo26n_ep50")
    else:
        for row in budget.itertuples():
            print("  %-16s vs %-20s %+.4f  [%+.4f, %+.4f]  %s"
                  % (row.arm_a, row.arm_b, row.mean_diff,
                     row.ci95_corrected_low, row.ci95_corrected_high,
                     "excludes zero" if row.corrected_excludes_zero else "includes zero"))
        if not epochs_summary.empty:
            print("\n  appendix C form -- where best-weight selection landed:")
            print("  %-16s %8s %8s %12s %14s"
                  % ("arm", "budget", "folds", "mean frac", ">= 96 % of budget"))
            for row in epochs_summary.itertuples():
                print("  %-16s %8d %8d %12.4f %14d"
                      % (row.arm, row.epoch_budget, row.n_folds,
                         row.mean_fraction_of_budget, row.folds_at_or_above_96pct))
            saturating = epochs_summary[
                epochs_summary["arm"] == "yolo26n_ep50"
            ]
            if not saturating.empty:
                hits = int(saturating.iloc[0]["folds_at_or_above_96pct"])
                print()
                if hits >= 5:
                    print("  ep50 STILL saturates its budget in %d of %d folds. '25 was too"
                          % (hits, int(saturating.iloc[0]["n_folds"])))
                    print("  few' does not explain it; the arm wants more than 50 as well.")
                else:
                    print("  ep50 does NOT saturate (%d fold(s) at >= 96 %%), so 25 epochs"
                          % hits)
                    print("  was genuinely constraining and the grid's budget was the limit.")

    if args.dry_run:
        print("\n  --dry-run: nothing written.")
        return 0

    stamp = aggregate.provenance(records)
    written = []
    if not recipe.empty:
        written.append(aggregate.write_csv_with_provenance(
            recipe, out / "yolo_recipe_check.csv", stamp, extra_header=RECIPE_HEADER))
    if not budget.empty:
        written.append(aggregate.write_csv_with_provenance(
            budget, out / "epoch_budget_check.csv", stamp, extra_header=BUDGET_HEADER))
    if not epochs_summary.empty:
        written.append(aggregate.write_csv_with_provenance(
            epochs_summary, out / "epoch_budget_selection.csv", stamp,
            extra_header=BUDGET_HEADER))

    rule("DONE")
    for path in written:
        print("[artifacts] wrote %s" % path.name)
    if not written:
        print("  Nothing to write yet -- no contrast records in the registry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
