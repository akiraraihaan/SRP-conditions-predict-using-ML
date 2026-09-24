#!/usr/bin/env python
"""11 -- the recipe comparison, and what the environment did to it.

    python scripts/11_recipe_check.py
    python scripts/11_recipe_check.py --weights exported
    python scripts/11_recipe_check.py --dry-run

CPU only. No training, no images. Reads the registry and writes two tables.

THERE IS NO OPTIMIZER AXIS, AND THIS SCRIPT NO LONGER PRETENDS THERE IS.

It used to print a 2x2 crossing architecture against optimizer, on the belief
that the three YOLO arms trained with MuSGD and the two baselines with SGD --
which is what `configs/arms.yaml` said. It was wrong. ultralytics' MuSGD takes
`use_muon: bool = False`, and src/srpcard/train.py builds it from a flat
parameter list without passing it, so no Muon update was ever applied: the
object is bitwise identical to torch.optim.SGD (197/197 and 210/210 tensors
identical after five steps on yolo26n and mobilenetv3_small).

Every run in the registry trained with SGD. The config now says so.

That leaves two things worth reporting, and they are different questions, so
they get separate tables rather than one grid with an axis that collapsed.

TABLE 1 -- RECIPE. yolo26n and mobilenetv3_small under the common uniform
protocol, and yolo26n under Ultralytics' own recipe end to end. Preprocessing
is NOT constant across the third row and the table says so in its own column.

TABLE 2 -- REPRODUCTION, which was not planned and is the more interesting
result. Once MuSGD is SGD, the "optimizer contrast" runs are exact re-runs of
published configurations in a later session, which turns them into the only
measurement here of how reproducible a fold actually is.

It reproduces EXACTLY within a session and does not between sessions, on the
same GPU model and the same torch version. THE CAUSE IS UNIDENTIFIED and this
file names no mechanism -- two arms are unaffected and three are not, which
rules out a uniform numerical shift and rules in nothing.
"""

from __future__ import annotations

import argparse
import json
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

OURS = "validation macro-F1, tie-break validation loss"
THEIRS = "ultralytics fitness (top-1/top-5 accuracy)"

# Table 1. The reference row is named once so the whole table orients on it.
REFERENCE_ROW = "yolo26n / uniform"

ROWS = [
    {"row": REFERENCE_ROW, "arm": "yolo26n", "script": PUBLISHED,
     "protocol": "uniform", "preprocessing": LETTERBOX, "optimizer": "SGD",
     "selection": OURS},
    {"row": "mobilenetv3_small / uniform", "arm": "mobilenetv3_small",
     "script": PUBLISHED, "protocol": "uniform", "preprocessing": LETTERBOX,
     "optimizer": "SGD", "selection": OURS},
    {"row": "yolo26n / native recipe", "arm": "yolo26n", "script": NATIVE,
     "protocol": "native", "preprocessing": ULTRALYTICS, "optimizer": "theirs",
     "selection": THEIRS},
]

# WHAT REPRODUCTION ACTUALLY LOOKS LIKE, observed across sessions.
#
# Re-running a completed fold reproduces EXACTLY within a session: two
# --emit-weights runs of mobilenetv3_small r0f0 in one session both gave
# 0.673445, epoch for epoch. Across sessions on the same GPU model and the same
# torch version it does not: that fold has been observed at 0.671546 (the
# registry), 0.654810 (an earlier session) and 0.673445 (a later one).
#
# THE CAUSE IS UNIDENTIFIED. What is known is what was held constant -- GPU
# model, torch version, seeds, fold partition -- and that two arms are
# unaffected while three are not. Nothing here names a mechanism, because
# nothing here has tested one.
#
# These are OBSERVED values, reported from the runs that produced them. They
# are constants because this script cannot recompute them: the comparison needs
# two sessions, and a script runs in one.
BETWEEN_SESSION_SPREAD = {
    # arm: {metric: max observed spread across sessions}
    "mobilenetv3_small": {"f1_macro": 1.9e-2, "precision_macro": 7.3e-2},
    "yolo26s": {"f1_macro": 1.04e-2},
    "resnet18": {"f1_macro": 6.8e-3},
    "yolo26n": {"f1_macro": 0.0},
    "yolo26m": {"f1_macro": 0.0},
}

WITHIN_SESSION = (
    "exact: two re-runs of mobilenetv3_small r0f0 in one session both gave "
    "0.673445, epoch for epoch"
)


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def _override_of(record: dict) -> str | None:
    value = (record.get("extra") or {}).get("run_id_optimizer")
    return str(value).lower() if value else None


def fold_series(records, arm, script, repeat, metric="f1_macro"):
    return {
        int(r["fold"]): float(r[metric])
        for r in records
        if r.get("script") == script and r.get("arm") == arm
        and r.get("repeat") == repeat and r.get(metric) is not None
        and r.get("fold") is not None
    }


# --------------------------------------------------------------------------
# Table 1 -- the recipe comparison
# --------------------------------------------------------------------------


def recipe_table(records, repeat: int, rho: float) -> pd.DataFrame:
    series = {row["row"]: fold_series(records, row["arm"], row["script"], repeat)
              for row in ROWS}
    reference = series.get(REFERENCE_ROW) or {}

    out = []
    for row in ROWS:
        values = series[row["row"]]
        block = dict(row)
        native = [r for r in records
                  if r.get("script") == row["script"] and r.get("arm") == row["arm"]
                  and r.get("repeat") == repeat]
        block.update({
            "repeat": repeat,
            "epochs": native[0].get("epochs") if native else None,
            "n_folds": len(values),
            "f1_macro_mean": float(np.mean(list(values.values()))) if values else None,
            "f1_macro_sd": (float(np.std(list(values.values()), ddof=1))
                            if len(values) > 1 else None),
        })
        for fold in sorted(values):
            block["fold%d" % fold] = values[fold]

        shared = sorted(set(values) & set(reference))
        if shared and row["row"] != REFERENCE_ROW:
            difference = np.array([values[f] - reference[f] for f in shared], float)
            stats = corrected_interval(difference, rho)
            block.update({
                "vs_reference_mean": stats["mean_diff"],
                "vs_reference_ci_low": stats["ci95_corrected_low"],
                "vs_reference_ci_high": stats["ci95_corrected_high"],
                "vs_reference_excludes_zero": stats["corrected_excludes_zero"],
            })
        out.append(block)

    frame = pd.DataFrame(out)
    for column in frame.columns:
        if frame[column].dtype.kind == "f":
            frame[column] = frame[column].round(6)
    return frame


def recipe_conclusion(frame: pd.DataFrame) -> list[str]:
    """The one conclusion this table supports. Both halves, never one alone."""
    def mean_of(name):
        row = frame[frame["row"] == name]
        if row.empty or pd.isna(row.iloc[0]["f1_macro_mean"]):
            return None
        return float(row.iloc[0]["f1_macro_mean"])

    yolo = mean_of(REFERENCE_ROW)
    mobile = mean_of("mobilenetv3_small / uniform")
    native = mean_of("yolo26n / native recipe")

    if None in (yolo, mobile, native):
        missing = [name for name, value in
                   ((REFERENCE_ROW, yolo), ("mobilenetv3_small / uniform", mobile),
                    ("yolo26n / native recipe", native)) if value is None]
        return [
            "INCOMPLETE -- cannot state the conclusion. Missing: %s"
            % ", ".join(missing),
            "",
            "  Both halves of the finding need all three rows. Stating either one",
            "  alone is the thing this table exists to prevent.",
        ]

    return [
        "UNDER A COMMON RECIPE the architecture gap is %+.4f in mobilenet's" % (mobile - yolo),
        "favour (%.4f against %.4f), and yolo26n recovers %+.4f of it when given"
        % (mobile, yolo, native - yolo),
        "its own recipe (%.4f), landing %+.4f from mobilenet." % (native, native - mobile),
        "",
        "  STATE BOTH. 'YOLO26 loses under a common protocol' and 'most of that",
        "  gap is the protocol, not the architecture' are both true, and either",
        "  on its own misrepresents the result.",
        "",
        "  The third row is NOT one variable changed: its preprocessing,",
        "  augmentation, schedule, optimizer and checkpoint criterion are all",
        "  Ultralytics'. It is 'their recipe end to end', and the preprocessing",
        "  column says so.",
    ]


# --------------------------------------------------------------------------
# Table 2 -- environment replication
# --------------------------------------------------------------------------

def environment_finding(frame: pd.DataFrame) -> list[str]:
    """What reproduction looks like here. No mechanism is named."""
    if frame.empty:
        return ["No replication rows yet -- nothing to state."]

    exact = sorted(frame.loc[frame["reproduces_exactly"].astype(bool), "arm"])
    moved = frame[~frame["reproduces_exactly"].astype(bool)]

    lines = [
        "WITHIN a session, re-running a completed fold reproduces EXACTLY.",
        "  %s" % WITHIN_SESSION,
        "",
        "ACROSS sessions, on the same GPU model and the same torch version, it",
        "does not:",
    ]
    for row in moved.sort_values("f1_macro_spread", ascending=False).itertuples():
        extra = ("  and %.1e precision_macro" % row.precision_macro_spread
                 if row.precision_macro_spread == row.precision_macro_spread
                 and row.precision_macro_spread else "")
        lines.append("  %-22s up to %.2e macro-F1%s"
                     % (row.arm, row.f1_macro_spread, extra))
    if exact:
        lines += ["", "  %s reproduce EXACTLY across sessions." % ", ".join(exact)]

    lines += [
        "",
        "  THE CAUSE IS UNIDENTIFIED. Held constant: GPU model, torch version,",
        "  seeds, fold partition. %d of the %d arm(s) are unaffected and %d are"
        % (len(exact), len(frame), len(moved)),
        "  not, so whatever it is, it is not a uniform numerical shift. No",
        "  mechanism is named here because none has been tested.",
        "",
        "  WHAT FOLLOWS FOR THE MANUSCRIPT: a result quoted to more precision",
        "  than the between-session spread is not reproducible at that",
        "  precision. The 15-fold mean is what should be quoted; a single fold",
        "  is not, and neither is a difference smaller than the spread above.",
    ]
    return lines


def environment_replication(records, repeat: int,
                            weights_dir: Path | None = None) -> pd.DataFrame:
    """One row per arm: does it reproduce, and by how much does it move.

    Within-session reproduction is exact and is stated once, not per arm. What
    varies between arms is the BETWEEN-session spread, and that is what the
    table carries.

    Per-fold rows are added for any arm where the registry holds both a
    published run and a later re-run of the same configuration, since those are
    computed rather than reported.
    """
    rows = []
    for arm, spread in sorted(BETWEEN_SESSION_SPREAD.items()):
        f1_spread = spread.get("f1_macro", 0.0)
        rows.append({
            "kind": "between_session",
            "arm": arm,
            "within_session": "exact",
            "f1_macro_spread": f1_spread,
            "precision_macro_spread": spread.get("precision_macro"),
            "reproduces_exactly": f1_spread == 0.0,
            "cause": "unidentified",
            "held_constant": "GPU model, torch version, seeds, fold partition",
            "source": "observed across sessions; reported, not recomputed here",
        })

    # Computed rows, where the registry happens to hold both runs.
    for arm in sorted({r["arm"] for r in records if r.get("script") == CONTRAST}):
        published = fold_series(records, arm, PUBLISHED, repeat)
        rerun = fold_series(records, arm, CONTRAST, repeat)
        shared = sorted(set(published) & set(rerun))
        for fold in shared:
            rows.append({
                "kind": "per_fold",
                "arm": arm,
                "repeat": repeat,
                "fold": fold,
                "published": round(published[fold], 6),
                "rerun": round(rerun[fold], 6),
                "delta": round(rerun[fold] - published[fold], 6),
                "cause": "unidentified",
                "source": "registry: %s vs %s" % (PUBLISHED, CONTRAST),
            })

    # And from the checkpoint sidecars, if they are there.
    if weights_dir:
        for sidecar in sorted(Path(weights_dir).glob("*.json")):
            blob = json.loads(sidecar.read_text(encoding="utf-8"))
            if not blob.get("recorded"):
                continue
            rows.append({
                "kind": "checkpoint_sidecar",
                "arm": blob.get("arm"),
                "published": (blob.get("recorded") or {}).get("f1_macro"),
                "rerun": (blob.get("measured") or {}).get("f1_macro"),
                "delta": blob.get("measured_minus_recorded"),
                "cause": "unidentified",
                "source": "sidecar %s" % sidecar.name,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# C1 -- the epoch budget. Orthogonal to everything above and still owed.
# --------------------------------------------------------------------------


def epoch_budget(records, rho: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """yolo26n_ep50 against yolo26n and mobilenetv3_small, plus appendix C form.

    The grid locked nano at 25 epochs while every other arm got 50, and nano is
    the only arm whose selected epochs cluster near its budget. This separates
    "YOLO26 loses" from "YOLO26 was stopped early".
    """
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
        difference = np.array([subject[k] - reference[k] for k in shared], float)
        stats = corrected_interval(difference, rho)
        rows.append({
            "arm_a": "yolo26n_ep50", "arm_b": other, "protocol": "uniform",
            "n_folds": len(shared),
            "mean_diff": stats["mean_diff"],
            "ci95_corrected_low": stats["ci95_corrected_low"],
            "ci95_corrected_high": stats["ci95_corrected_high"],
            "p_corrected": stats["p_corrected"],
            "corrected_excludes_zero": stats["corrected_excludes_zero"],
            "a_wins": stats["a_wins"], "b_wins": stats["b_wins"],
        })

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
# the native recipe, read back from the run
# --------------------------------------------------------------------------


def native_recipe_rows(records) -> pd.DataFrame:
    """What Ultralytics actually did, from the trainer -- not from the docs."""
    native = [r for r in records if r.get("script") == NATIVE]
    rows = []
    for record in native:
        extra = record.get("extra") or {}
        recipe = extra.get("native_recipe") or {}
        row = {
            "arm": record.get("arm"),
            "repeat": record.get("repeat"),
            "fold": record.get("fold"),
            "protocol": extra.get("protocol"),
            "preprocessing": extra.get("preprocessing"),
            "optimizer_requested": recipe.get("optimizer_requested"),
            "optimizer_used": recipe.get("optimizer_used"),
            "epochs_run": extra.get("epochs_run_native"),
            "selected_epoch": extra.get("selected_epoch_native"),
            "selection_criterion": extra.get("selection_criterion_native"),
            "unreadable_keys": ", ".join(recipe.get("unreadable_keys") or []),
        }
        for key, value in (recipe.get("augmentation") or {}).items():
            row["aug_%s" % key] = value
        for key, value in (recipe.get("schedule") or {}).items():
            row["sched_%s" % key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def print_native_recipe(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("  No 03c_native_recipe records yet. Run:")
        print("    python scripts/03c_native_recipe.py --data-root $DATA_ROOT")
        return

    first = frame.iloc[0]
    print("  optimizer requested : %s" % first.get("optimizer_requested"))
    print("  optimizer that ran  : %s" % first.get("optimizer_used"))
    print("  epochs run          : %s" % first.get("epochs_run"))
    print("  checkpoint selected : epoch %s by %s"
          % (first.get("selected_epoch"), first.get("selection_criterion")))
    print("  preprocessing       : %s" % first.get("preprocessing"))

    aug = {c[4:]: first[c] for c in frame.columns if c.startswith("aug_")}
    active = {k: v for k, v in aug.items() if v not in (0, 0.0, False, None, "")}
    print("\n  augmentation APPLIED, at the strength that ran:")
    for key, value in sorted(active.items()):
        print("      %-16s %s" % (key, value))
    if not active:
        print("      none")
    off = sorted(set(aug) - set(active))
    if off:
        print("  augmentation off    : %s" % ", ".join(off))

    sched = {c[6:]: first[c] for c in frame.columns if c.startswith("sched_")}
    print("\n  schedule:")
    for key, value in sorted(sched.items()):
        print("      %-16s %s" % (key, value))

    unreadable = str(first.get("unreadable_keys") or "")
    if unreadable:
        print("\n  NOT READABLE from the trainer, and therefore NOT reported:")
        print("      %s" % unreadable)
        print("      These are gaps, not defaults. A documented default that was")
        print("      not in force would read as a measurement.")


# --------------------------------------------------------------------------

RECIPE_HEADER = [
    "Recipe comparison, repeat 0. THERE IS NO OPTIMIZER AXIS.",
    "",
    "This table used to be a 2x2 crossing architecture against optimizer, on the",
    "belief that the YOLO arms trained with MuSGD and the baselines with SGD --",
    "which is what configs/arms.yaml said. It was wrong: ultralytics' MuSGD takes",
    "use_muon=False and was built from a flat parameter list, so it is bitwise",
    "torch.optim.SGD. EVERY run in the registry trained with SGD.",
    "",
    "PREPROCESSING IS NOT CONSTANT ACROSS ROWS, and the column says so. The",
    "native row is 'their recipe end to end' -- their preprocessing,",
    "augmentation, schedule, optimizer and checkpoint criterion -- not one",
    "variable changed. Evaluating a natively trained model through our letterbox",
    "would show it a transform it never trained on and manufacture the result.",
    "",
    "THE CONCLUSION THIS TABLE SUPPORTS, AND THE ONLY ONE: under a common recipe",
    "the architecture gap favours mobilenetv3_small, AND yolo26n recovers most of",
    "that gap when given its own recipe. State both. Either alone misrepresents",
    "it.",
    "",
    "Intervals carry the Nadeau-Bengio correction; these are overlapping folds.",
]

ENVIRONMENT_HEADER = [
    "Reproduction: exact within a session, not exact between sessions.",
    "",
    "Re-running a completed fold reproduces EXACTLY within a session. Two",
    "--emit-weights runs of mobilenetv3_small r0f0 in one session both gave",
    "0.673445, epoch for epoch.",
    "",
    "A re-run in a LATER session, on the same GPU model and the same torch",
    "version, differs. The same fold has been observed at 0.671546 (the",
    "registry), 0.654810 and 0.673445. Between-session spread by arm:",
    "",
    "    mobilenetv3_small   up to 1.9e-2 macro-F1, 7.3e-2 precision_macro",
    "    yolo26s             up to 1.04e-2 macro-F1",
    "    resnet18            up to 6.8e-3 macro-F1",
    "    yolo26n             exact",
    "    yolo26m             exact",
    "",
    "THE CAUSE IS UNIDENTIFIED. What is known is what was held constant -- GPU",
    "model, torch version, seeds, fold partition -- and that two arms are",
    "unaffected while three are not, so it is not a uniform numerical shift.",
    "No mechanism is named in this file, because none has been tested. Do not",
    "let one be inferred from the fact that a library was reinstalled at some",
    "point: that is a coincidence in time, not evidence.",
    "",
    "WHAT FOLLOWS FOR THE MANUSCRIPT: a result quoted to more precision than",
    "the between-session spread is not reproducible at that precision. Quote",
    "the 15-fold mean. A single fold is not reproducible to three decimals, and",
    "neither is a between-arm difference smaller than the spread above.",
    "",
    "kind=between_session rows are OBSERVED and reported -- this script runs in",
    "one session and cannot recompute them. kind=per_fold and",
    "kind=checkpoint_sidecar rows are computed from the registry and from the",
    "checkpoint sidecars respectively.",
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

NATIVE_HEADER = [
    "What Ultralytics' own recipe actually consisted of, READ BACK FROM THE",
    "TRAINER that ran -- never quoted from documentation.",
    "",
    "ultralytics resolves `optimizer: auto` and several augmentation strengths",
    "at runtime, so the documented defaults are not necessarily the values that",
    "were in force. optimizer_used is the class of the optimizer OBJECT, which",
    "is the distinction that hid MuSGD == SGD in our own loop for weeks.",
    "",
    "Anything that could not be read from the trainer is listed by name in",
    "unreadable_keys and left empty. A default that was not in force would read",
    "as a measurement.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=0,
                        help="which repeat these tables cover (default: 0)")
    parser.add_argument("--weights", default=None,
                        help="checkpoint directory, to read reproduction deltas "
                             "from the sidecars instead of using reported values")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what is present and exit, writing nothing")
    args = parser.parse_args()

    rule("11 -- recipe comparison and environment replication")
    records = load_registry()
    out = artifacts_dir(load_data_config())
    weights_dir = Path(args.weights) if args.weights else None

    cv = [r for r in records if r.get("script") == PUBLISHED]
    rho = 1.0 / (folds_per_repeat(cv) - 1) if cv else 0.25
    print("  registry : %d record(s)" % len(records))
    print("  optimizer: SGD throughout -- MuSGD was never applied (use_muon=False)")

    # ---- Table 1
    rule("TABLE 1 -- recipe, repeat %d" % args.repeat)
    recipe = recipe_table(records, args.repeat, rho)
    print("  %-30s %-10s %-20s %-8s %6s %10s %12s"
          % ("row", "protocol", "preprocessing", "optim", "folds", "macro-F1",
             "vs yolo/unif"))
    for row in recipe.itertuples():
        delta = getattr(row, "vs_reference_mean", None)
        print("  %-30s %-10s %-20s %-8s %6d %10s %12s"
              % (row.row, row.protocol, row.preprocessing, row.optimizer,
                 row.n_folds,
                 "%.4f" % row.f1_macro_mean if row.f1_macro_mean == row.f1_macro_mean else "--",
                 "%+.4f" % delta if delta is not None and delta == delta else "--"))
    print()
    for line in recipe_conclusion(recipe):
        print("  " + line)

    # ---- Table 2
    rule("TABLE 2 -- reproduction within and between sessions")
    environment = environment_replication(records, args.repeat, weights_dir)
    between = environment[environment["kind"] == "between_session"]
    if between.empty:
        print("  nothing observed yet")
    else:
        print("  %-22s %-14s %16s %20s"
              % ("arm", "within session", "between: f1_macro", "between: precision"))
        for row in between.itertuples():
            precision = (
                "%.2e" % row.precision_macro_spread
                if row.precision_macro_spread == row.precision_macro_spread
                and row.precision_macro_spread else "--"
            )
            print("  %-22s %-14s %16s %20s"
                  % (row.arm, row.within_session,
                     "exact" if row.reproduces_exactly else "%.2e" % row.f1_macro_spread,
                     precision))

        computed = environment[environment["kind"].isin(
            ["per_fold", "checkpoint_sidecar"])]
        if not computed.empty:
            print("\n  computed from the registry and the sidecars:")
            for row in computed.itertuples():
                label = "%s %s" % (row.arm, ("fold %d" % row.fold
                                             if row.kind == "per_fold" else "checkpoint"))
                print("      %-28s %.6f -> %.6f   %+.4f"
                      % (label, row.published, row.rerun, row.delta))

        print()
        for line in environment_finding(between):
            print("  " + line)

    # ---- C1
    rule("C1 -- the epoch budget")
    budget, budget_epochs = epoch_budget(records, rho)
    if budget.empty:
        print("  No yolo26n_ep50 records yet. Run:")
        print("    python scripts/03_run_cv.py --contrast --arms yolo26n_ep50")
    else:
        for row in budget.itertuples():
            print("  %-16s vs %-20s %+.4f  [%+.4f, %+.4f]  %s"
                  % (row.arm_a, row.arm_b, row.mean_diff,
                     row.ci95_corrected_low, row.ci95_corrected_high,
                     "excludes zero" if row.corrected_excludes_zero else "includes zero"))
        if not budget_epochs.empty:
            print("\n  appendix C form -- where best-weight selection landed:")
            print("  %-16s %8s %8s %12s %14s"
                  % ("arm", "budget", "folds", "mean frac", ">= 96 % of budget"))
            for row in budget_epochs.itertuples():
                print("  %-16s %8d %8d %12.4f %14d"
                      % (row.arm, row.epoch_budget, row.n_folds,
                         row.mean_fraction_of_budget, row.folds_at_or_above_96pct))
            hit = budget_epochs[budget_epochs["arm"] == "yolo26n_ep50"]
            if not hit.empty:
                saturating = int(hit.iloc[0]["folds_at_or_above_96pct"])
                total = int(hit.iloc[0]["n_folds"])
                print()
                if saturating >= 5:
                    print("  ep50 STILL saturates its budget in %d of %d folds, so '25 was"
                          % (saturating, total))
                    print("  too few' does not explain it -- the arm wants more than 50 too.")
                else:
                    print("  ep50 does NOT saturate (%d of %d folds at >= 96 %%), so 25"
                          % (saturating, total))
                    print("  epochs was genuinely constraining.")

    # ---- the native recipe
    rule("THE NATIVE RECIPE, read back from the trainer")
    native = native_recipe_rows(records)
    print_native_recipe(native)

    if args.dry_run:
        print("\n  --dry-run: nothing written.")
        return 0

    stamp = aggregate.provenance(records)
    written = []
    for frame, name, header in (
        (recipe, "yolo_recipe_check.csv", RECIPE_HEADER),
        (environment, "environment_replication.csv", ENVIRONMENT_HEADER),
        (native, "native_recipe_settings.csv", NATIVE_HEADER),
        (budget, "epoch_budget_check.csv", BUDGET_HEADER),
        (budget_epochs, "epoch_budget_selection.csv", BUDGET_HEADER),
    ):
        if not frame.empty:
            written.append(aggregate.write_csv_with_provenance(
                frame, out / name, stamp, extra_header=header))

    rule("DONE")
    for path in written:
        print("[artifacts] wrote %s" % path.name)
    if not written:
        print("  Nothing to write yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
