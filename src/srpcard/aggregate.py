"""Aggregate the registry into manuscript-ready tables.

    artifacts/summary_cv.csv        mean and std of every metric per arm,
                                    across the 15 folds
    artifacts/summary_per_class.csv per-class F1 and recall, mean and std
    artifacts/selected_epochs.csv   distribution of selected_epoch per arm

Only `03_run_cv` records enter summary_cv.csv. Development-split runs (scripts
01, 02), the ablation (04) and the learning curve (05) are excluded: mixing them
would average across different protocols and different corpora.

For the same reason an arm whose own records are not unanimous on epochs, batch
and lr is REFUSED rather than summarised. Those three feed the run_id hash, so an
arm can accumulate two hyperparameter regimes without any run being skipped --
the failure `assert_hyperparameters_unanimous` exists to catch, and the one thing
a mean across folds would hide most effectively.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RUN_DEFINING_HYPERPARAMETERS, artifacts_dir, load_data_config
from .registry import load_registry

CV_SCRIPT = "03_run_cv"

# Everything write_all() produces. Script 06 clears these before regenerating,
# so an interrupted run cannot leave a mix of fresh and stale tables.
TABLE_NAMES = ("summary_cv.csv", "summary_per_class.csv", "selected_epochs.csv")

SCALAR_METRICS = [
    "f1_macro",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "params",
    "gflops",
    # Every size measurement reaches the summary table as its own column, so the
    # manuscript and the Pareto analysis can quote the same number without
    # anyone recomputing anything. fp16 is primary; see efficiency.py.
    "size_mb",
    "size_mb_fp16",
    "size_mb_fp32",
    "size_mb_fp16_payload",
    "latency_ms_mean",
    "wall_time_s",
]


class MixedHyperparametersError(RuntimeError):
    """An arm's completed runs disagree on the hyperparameters that define them."""


def selection_margins(
    table: "pd.DataFrame",
    *,
    val_n: int,
    k: int = 3,
    group: str = "arm",
    score: str = "f1_macro_val",
    secondary: str | None = "f1_macro_dev_test",
) -> "pd.DataFrame":
    """Top-k configurations per group, with how close they are to the winner.

    Shared by the two selection scripts -- 01b's grid and 02's lr sweep -- so a
    margin means the same thing in both artefacts.

    Selection is and remains ARGMAX. Changing the rule after seeing the results
    would be post-hoc. What this records is how much the argmax actually won by,
    in a unit that says whether the margin is a result or a rounding difference:

      margin_vs_winner    score minus the winner's score (0 for the winner)
      images_equivalent   that margin expressed in validation images, from the
                          ACTUAL size of the validation partition
      `secondary`_rank    where the row ranks on the secondary score, so a
                          winner that loses on held-out data is visible

    The secondary column exists because selection on validation can pick a
    configuration that is not the best on the development test partition. That
    is not a bug -- selecting on the test partition would be the bug -- but it
    belongs on the record rather than in an argument afterwards.
    """
    rows = []
    for name, raw in table.groupby(group):
        ranked = raw.sort_values(score, ascending=False)
        best = float(ranked[score].iloc[0])
        secondary_order = None
        if secondary and secondary in ranked:
            secondary_order = list(
                ranked.sort_values(secondary, ascending=False)[score]
            )
        for rank, record in enumerate(ranked.head(k).to_dict("records"), 1):
            margin = float(record[score]) - best
            row = {
                group: name,
                "rank": rank,
                "margin_vs_winner": round(margin, 6),
                "images_equivalent": round(abs(margin) * val_n, 3),
                "selected": rank == 1,
            }
            row.update({key: value for key, value in record.items() if key != group})
            if secondary_order is not None:
                row["%s_rank" % secondary] = secondary_order.index(record[score]) + 1
            rows.append(row)
    frame = pd.DataFrame(rows)
    lead = [group, "rank", "selected", "margin_vs_winner", "images_equivalent"]
    ordered = lead + [c for c in frame.columns if c not in lead]
    return frame[ordered]


def print_selection_margins(frame: "pd.DataFrame", *, val_n: int, group: str = "arm",
                            label: str = "key") -> None:
    print(
        "\n  %-20s %-4s %-22s %14s %10s %9s"
        % (group, "rank", label, "f1_macro_val", "margin", "~images")
    )
    for name, raw in frame.groupby(group):
        for row in raw.sort_values("rank").to_dict("records"):
            print(
                "  %-20s %-4d %-22s %14.4f %10s %9s"
                % (
                    name if row["rank"] == 1 else "",
                    row["rank"],
                    row.get(label, ""),
                    row["f1_macro_val"],
                    "--" if row["rank"] == 1 else "%+.4f" % row["margin_vs_winner"],
                    "--" if row["rank"] == 1 else "%.2f" % row["images_equivalent"],
                )
            )
    print(
        "\n  Selection is argmax on validation macro-F1 and stays argmax. '~images' is\n"
        "  the margin in validation images (%d in this partition): the unit that says\n"
        "  whether a margin is a result or a rounding difference." % val_n
    )


def provenance(records: list[dict[str, Any]], path: Path | None = None) -> dict[str, Any]:
    """What a generated artefact was built from.

    Stamped into every table and figure script 06 writes. A stale artefact then
    announces itself -- "built from 1 record" against a registry holding 75 --
    instead of waiting for someone to notice that the numbers describe a run
    that no longer exists.
    """
    from .registry import registry_path

    path = Path(path) if path is not None else registry_path()
    digest = "absent"
    if path.exists():
        digest = hashlib.sha1(path.read_bytes()).hexdigest()[:16]  # noqa: S324

    fingerprints = sorted(
        {
            (r.get("corpus_fingerprint") or {}).get("sha1_of_sorted_included_sha1s")
            for r in records
        }
        - {None}
    )
    return {
        "n_records": len(records),
        "arms": sorted({r.get("arm") for r in records} - {None}),
        "scripts": sorted({r.get("script") for r in records} - {None}),
        "corpus_fingerprint": fingerprints[0] if len(fingerprints) == 1 else fingerprints,
        "registry_sha1": digest,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def provenance_lines(block: dict[str, Any]) -> list[str]:
    """The provenance stamp as comment-free `key: value` strings."""
    return [
        "built from %d registry record(s)" % block["n_records"],
        "arms: %s" % (", ".join(block["arms"]) or "none"),
        "scripts: %s" % (", ".join(block["scripts"]) or "none"),
        "corpus: %s" % block["corpus_fingerprint"],
        "registry sha1: %s" % block["registry_sha1"],
        "generated: %s" % block["generated_at"],
    ]


def provenance_caption(block: dict[str, Any]) -> str:
    """One line, for a figure caption strip."""
    return (
        "%d record(s) | arms: %s | corpus %s | registry %s | %s"
        % (
            block["n_records"],
            ",".join(block["arms"]) or "none",
            block["corpus_fingerprint"],
            block["registry_sha1"],
            block["generated_at"],
        )
    )


def write_csv_with_provenance(
    frame: pd.DataFrame, target: Path, block: dict[str, Any]
) -> Path:
    """Write a table with the provenance stamp as leading `#` comment lines.

    pandas.read_csv(comment="#") skips them, and every consumer in this
    repository reads these files with pandas.
    """
    header = "".join("# %s\n" % line for line in provenance_lines(block))
    target.write_text(
        header + frame.to_csv(index=False, lineterminator="\n"),
        encoding="utf-8",
        newline="\n",
    )
    return target


def cv_records(path: Path | None = None, script: str = CV_SCRIPT) -> list[dict[str, Any]]:
    return [r for r in load_registry(path) if r.get("script") == script]


def hyperparameter_groups(records: list[dict[str, Any]]) -> dict[str, dict[tuple, list[str]]]:
    """{arm: {(epochs, batch, lr): [run_id, ...]}}. More than one key is drift."""
    groups: dict[str, dict[tuple, list[str]]] = {}
    for record in records:
        key = tuple(record.get(field) for field in RUN_DEFINING_HYPERPARAMETERS)
        groups.setdefault(record.get("arm", "?"), {}).setdefault(key, []).append(
            record.get("run_id", "?")
        )
    return groups


def assert_hyperparameters_unanimous(records: list[dict[str, Any]]) -> None:
    """Refuse to summarise an arm whose records were not all trained the same way.

    epochs, batch and lr are part of the run_id, so two regimes for one arm means
    two sets of run_ids and nothing skipped -- the folds were simply trained twice
    under different settings. Averaging across them silently reports a number that
    describes no configuration that was actually run.
    """
    offenders = {
        arm: keys for arm, keys in hyperparameter_groups(records).items() if len(keys) > 1
    }
    if not offenders:
        return

    lines = [
        "MIXED HYPERPARAMETERS in the registry -- refusing to summarise.",
        "",
        "  These arms have completed runs trained under more than one setting of",
        "  epochs/batch/lr. Those fields define the run_id, so both regimes sit in",
        "  the registry as separate runs and a mean across folds would average them",
        "  together without any indication that it had.",
        "",
    ]
    for arm, keys in sorted(offenders.items()):
        lines.append("  arm %s -- %d regimes:" % (arm, len(keys)))
        for key, run_ids in sorted(keys.items(), key=lambda kv: -len(kv[1])):
            described = "  ".join(
                "%s %s" % (field, value)
                for field, value in zip(RUN_DEFINING_HYPERPARAMETERS, key)
            )
            lines.append("    %-40s %d run(s)" % (described, len(run_ids)))
            for run_id in run_ids[:20]:
                lines.append("      %s" % run_id)
            if len(run_ids) > 20:
                lines.append("      ... and %d more" % (len(run_ids) - 20))
    lines += [
        "",
        "  Usually a resolved configuration was lost between sessions: scripts 01",
        "  and 02 rewrite configs/arms.yaml, and a clone that reverted to the",
        "  committed values retrains the same folds under the provisional ones.",
        "",
        "  Decide which regime is the real one, delete the other's run_ids from",
        "  artifacts/registry.jsonl, and restore the intended config with:",
        "      python scripts/restore_arms.py",
    ]
    raise MixedHyperparametersError("\n".join(lines))


def summarise_cv(
    records: list[dict[str, Any]] | None = None,
    data_cfg: dict[str, Any] | None = None,
    expected_folds: int = 15,
) -> pd.DataFrame:
    """Mean and std of every scalar metric per arm, ready to paste into a table."""
    data_cfg = data_cfg or load_data_config()
    records = records if records is not None else cv_records()
    if not records:
        return pd.DataFrame()
    assert_hyperparameters_unanimous(records)

    frame = pd.DataFrame(
        [{"arm": r["arm"], "architecture": r["architecture"],
          **{m: r.get(m) for m in SCALAR_METRICS},
          "selected_epoch": r.get("selected_epoch"),
          "epochs": r.get("epochs")}
         for r in records]
    )

    rows = []
    for arm, group in frame.groupby("arm"):
        row: dict[str, Any] = {
            "arm": arm,
            "architecture": group["architecture"].iloc[0],
            "n_folds": len(group),
            "complete": len(group) == expected_folds,
        }
        for metric in SCALAR_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row["%s_mean" % metric] = float(values.mean()) if len(values) else np.nan
            row["%s_std" % metric] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        epochs = pd.to_numeric(group["selected_epoch"], errors="coerce").dropna()
        if len(epochs):
            row["selected_epoch_mean"] = float(epochs.mean()) + 1
            row["selected_epoch_min"] = int(epochs.min()) + 1
            row["selected_epoch_max"] = int(epochs.max()) + 1
            row["epoch_budget"] = int(group["epochs"].iloc[0])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)


def summarise_per_class(
    records: list[dict[str, Any]] | None = None, data_cfg: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Per-class F1 and recall, mean and std across folds, rarest class first."""
    data_cfg = data_cfg or load_data_config()
    records = records if records is not None else cv_records()
    if not records:
        return pd.DataFrame()
    assert_hyperparameters_unanimous(records)

    classes = list(data_cfg["classes"])
    sizes = dict(data_cfg["clean_corpus"]["expected_counts"])
    order = sorted(classes, key=lambda c: sizes.get(c, 0))

    rows = []
    for arm in sorted({r["arm"] for r in records}):
        subset = [r for r in records if r["arm"] == arm]
        for name in order:
            f1_values = [r["f1_per_class"][name] for r in subset if r.get("f1_per_class")]
            recall_values = [r["recall_per_class"][name] for r in subset if r.get("recall_per_class")]
            rows.append(
                {
                    "arm": arm,
                    "class": name,
                    "n_clean": sizes.get(name, 0),
                    "f1_mean": float(np.mean(f1_values)) if f1_values else np.nan,
                    "f1_std": float(np.std(f1_values, ddof=1)) if len(f1_values) > 1 else 0.0,
                    "recall_mean": float(np.mean(recall_values)) if recall_values else np.nan,
                    "recall_std": float(np.std(recall_values, ddof=1)) if len(recall_values) > 1 else 0.0,
                    "n_folds": len(f1_values),
                }
            )
    return pd.DataFrame(rows)


def selected_epoch_distribution(
    records: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Where the selection criterion landed, per arm.

    If this clusters near the epoch budget, the budget is too short and the
    locked epoch counts need revisiting before the methods section is written.
    """
    records = records if records is not None else cv_records()
    rows = []
    for r in records:
        if r.get("selected_epoch") is None:
            continue
        rows.append(
            {
                "arm": r["arm"],
                "repeat": r.get("repeat"),
                "fold": r.get("fold"),
                "selected_epoch": r["selected_epoch"] + 1,
                "epoch_budget": r.get("epochs"),
                "fraction_of_budget": (r["selected_epoch"] + 1) / max(r.get("epochs") or 1, 1),
                "min_val_loss_epoch": (r.get("min_val_loss_epoch", -1) or -1) + 1,
                "best_val_f1": r.get("best_val_f1"),
            }
        )
    return pd.DataFrame(rows)


def mean_confusion_matrix(
    arm: str, records: list[dict[str, Any]] | None = None
) -> np.ndarray | None:
    """Summed confusion matrix over an arm's folds, in canonical order."""
    records = records if records is not None else cv_records()
    matrices = [
        np.asarray(r["confusion_matrix"]) for r in records if r["arm"] == arm and r.get("confusion_matrix")
    ]
    if not matrices:
        return None
    return np.sum(matrices, axis=0)


def write_all(data_cfg: dict[str, Any] | None = None, path: Path | None = None) -> dict[str, Path]:
    """Write every summary table. Returns {name: path}."""
    data_cfg = data_cfg or load_data_config()
    out = artifacts_dir(data_cfg)
    records = cv_records(path)

    written: dict[str, Path] = {}
    stamp = provenance(records)

    summary = summarise_cv(records, data_cfg)
    if not summary.empty:
        written["summary_cv"] = write_csv_with_provenance(
            summary, out / "summary_cv.csv", stamp
        )

    per_class = summarise_per_class(records, data_cfg)
    if not per_class.empty:
        written["summary_per_class"] = write_csv_with_provenance(
            per_class, out / "summary_per_class.csv", stamp
        )

    epochs = selected_epoch_distribution(records)
    if not epochs.empty:
        written["selected_epochs"] = write_csv_with_provenance(
            epochs, out / "selected_epochs.csv", stamp
        )

    return written
