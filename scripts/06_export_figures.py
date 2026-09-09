#!/usr/bin/env python
"""06 -- export every publication figure. Vector PDF plus high-resolution PNG.

    python scripts/06_export_figures.py
    python scripts/06_export_figures.py --out-dir artifacts/figures

Matplotlib only; seaborn is not a dependency of this repository.

Also refreshes the manuscript tables via src/srpcard/aggregate.py:
  artifacts/summary_cv.csv, summary_per_class.csv, selected_epochs.csv

Figures whose inputs are missing are skipped with a message naming the script
that produces them, so a partial run still emits everything it can.

TWO RULES AGAINST STALE OUTPUT.

1. The whole output set is DELETED before anything is regenerated. A partial or
   interrupted run then leaves fewer files, never a mix of fresh and stale ones.
   Before this, a figure from a run whose registry records had since been removed
   sat in artifacts/figures/ looking exactly like a current one.

2. Everything written carries a PROVENANCE stamp: how many registry records it
   was built from, which arms and scripts, the corpus fingerprint, the registry
   file's sha1 and a UTC timestamp. Tables get it as leading `#` comment lines;
   figures get a strip along the bottom and the same text in the PDF metadata,
   so it survives being cropped into a manuscript. A stale artefact announces
   itself -- "built from 1 record" against a registry holding 75.

   The stamp is PER ARTEFACT, not run-wide. A single stamp reused everywhere
   said "75 records, scripts: 03_run_cv" on the learning-curve and ablation
   figures, which are built from script 05 and 04 records -- naming runs those
   figures never saw and omitting the ones they did. Every figure below is
   stamped with exactly the records that fed it, and
   `aggregate.assert_provenance_covers` refuses a stamp that disagrees.

WHICH ARM IS REPORTED IN DETAIL

`configs/arms.yaml:reporting.detailed_arm`, never "the highest mean F1" and
never a hardcoded name. The two are not the same model here: resnet18 has the
higher mean (0.5997 vs 0.5900) by an amount indistinguishable from zero, at
seven times the size and thirty times the compute. The confusion matrix and the
per-class table belong to the model the paper recommends. A matrix is also
emitted for every non-dominated arm, so both frontier points are available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from srpcard import aggregate, figures  # noqa: E402
from srpcard import data as srp_data  # noqa: E402
from srpcard.config import artifacts_dir, load_arms_config, load_data_config  # noqa: E402
from srpcard.registry import load_registry  # noqa: E402


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def stamped(records, *, sources=None):
    """Install the provenance for the figure about to be drawn, and verify it.

    Every call is checked against the records actually passed, so a figure can
    never carry a stamp naming runs it did not consume.
    """
    block = aggregate.provenance(records, sources=sources)
    aggregate.assert_provenance_covers(block, records)
    figures.set_provenance(block)
    return block


def clear_outputs(artifacts: Path, out_dir: Path) -> int:
    """Delete everything this script generates, before regenerating any of it.

    Scoped deliberately: the three tables aggregate.write_all() produces, and the
    figure files in out_dir. Nothing else in artifacts/ is touched -- the frozen
    inputs, the registry and the other scripts' outputs are not this script's to
    remove.
    """
    removed = 0
    for name in aggregate.TABLE_NAMES:
        target = artifacts / name
        if target.exists():
            target.unlink()
            removed += 1
    if out_dir.exists():
        for target in sorted(out_dir.iterdir()):
            if target.is_file() and target.suffix.lower() in {".pdf", ".png"}:
                target.unlink()
                removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None, help="default: artifacts/figures")
    parser.add_argument(
        "--keep-stale",
        action="store_true",
        help="do NOT clear the output set first (debugging only; can leave a mix "
             "of fresh and stale files, which is the failure this guards against)",
    )
    args = parser.parse_args()

    data_cfg = load_data_config()
    arms_cfg = load_arms_config()
    out_dir = Path(args.out_dir) if args.out_dir else artifacts_dir(data_cfg) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rule("06 -- publication figures")
    print("[out] %s" % out_dir)

    # ---- provenance is per artefact; this is only the cv baseline ----
    records = aggregate.cv_records()
    print("[provenance] cross-validation records (03_run_cv)")
    for line in aggregate.provenance_lines(aggregate.provenance(records)):
        print("    %s" % line)
    print("    Each figure below is stamped with the records that fed IT, not this.")
    if not records:
        print(
            "\n  NOTE: no 03_run_cv records. Tables and the figures that read them\n"
            "        will be skipped, and anything left from a previous run has\n"
            "        just been cleared rather than left to look current."
        )

    # ---- clear the whole output set BEFORE regenerating ----
    if args.keep_stale:
        print("\n[clear] skipped -- --keep-stale")
    else:
        removed = clear_outputs(artifacts_dir(data_cfg), out_dir)
        print(
            "\n[clear] removed %d previously generated file(s); a partial run below\n"
            "        leaves fewer files, never a mix of fresh and stale ones" % removed
        )

    written: list[Path] = []
    skipped: list[str] = []

    # ---- tables first: the figures read them ----
    tables = aggregate.write_all(data_cfg)
    for name, path in tables.items():
        print("[table] %s -> %s" % (name, path.name))
    if not tables:
        print("[table] no 03_run_cv records yet; tables skipped")

    # ---- 1. class distribution ----
    # Built from the image index, not from any run: it gets an explicit source
    # rather than a record count that would be zero and unexplained.
    try:
        index = srp_data.load_image_index()
        stamped([], sources=["artifacts/image_index.csv"])
        written += figures.figure_class_distribution(index, data_cfg, out_dir)
        print("[fig] class distribution")
    except Exception as exc:  # noqa: BLE001
        skipped.append("class distribution: %s" % exc)

    records = aggregate.cv_records()

    # ---- 2. per-arm macro-F1 boxplot ----
    if records:
        stamped(records)
        written += figures.figure_cv_boxplot(records, out_dir)
        print("[fig] cross-validated macro-F1 by arm")
    else:
        skipped.append("cv boxplot: no 03_run_cv records (run scripts/03_run_cv.py)")

    # ---- 3. Pareto ----
    summary_path = artifacts_dir(data_cfg) / "summary_cv.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, comment="#")
        if summary["gflops_mean"].notna().any():
            stamped(records, sources=["artifacts/summary_cv.csv"])
            written += figures.figure_pareto(summary, out_dir)
            print("[fig] Pareto frontier (compute)")
        else:
            skipped.append("pareto: gflops missing from summary_cv.csv")

        size_column = (
            "size_mb_fp16_mean" if "size_mb_fp16_mean" in summary else "size_mb_mean"
        )
        if size_column in summary and summary[size_column].notna().any():
            stamped(records, sources=["artifacts/summary_cv.csv"])
            written += figures.figure_pareto_size(summary, out_dir)
            print("[fig] Pareto frontier (size, fp16 weights)")
        else:
            skipped.append("pareto (size): size_mb_fp16 missing from summary_cv.csv")
    else:
        skipped.append("pareto: artifacts/summary_cv.csv (run scripts/03_run_cv.py)")

    # ---- 4. confusion matrices ----
    #
    # The DETAILED arm comes from configs/arms.yaml:reporting.detailed_arm, not
    # from argmax of mean F1 -- those are different models here, and the paper's
    # detailed evaluation is of the one it recommends. Every non-dominated arm
    # gets one too, so both frontier points are available.
    if records:
        detailed_arm = (arms_cfg.get("reporting") or {}).get("detailed_arm")
        if not detailed_arm:
            skipped.append(
                "confusion matrices: configs/arms.yaml has no reporting.detailed_arm"
            )
        else:
            frontier = []
            pareto = aggregate.pareto_status(records)
            if not pareto.empty:
                frontier = pareto.loc[
                    pareto["on_pareto_frontier"], "arm"
                ].tolist()
            wanted = [detailed_arm] + [a for a in frontier if a != detailed_arm]

            for arm in wanted:
                matrix, used = aggregate.summed_confusion_matrix(arm, records)
                if matrix is None:
                    skipped.append("confusion matrix (%s): no records" % arm)
                    continue
                total, expected, note = aggregate.check_confusion_total(
                    matrix, arm, len(used), data_cfg
                )
                role = "detailed" if arm == detailed_arm else "frontier"
                stamped(used)
                written += figures.figure_confusion(
                    matrix,
                    list(data_cfg["classes"]),
                    out_dir,
                    # the arm is in the FILENAME: a confusion matrix that could
                    # be mistaken for another model's is worse than none
                    "fig_confusion_%s" % arm,
                    "Confusion matrix, %s (%s) -- %d predictions %s"
                    % (arm, role, total, note),
                )
                print(
                    "[fig] confusion matrix (%s, %s): %d predictions, %s"
                    % (arm, role, total, note)
                )

    # ---- 5. learning curve ----
    lc_path = artifacts_dir(data_cfg) / "learning_curve.csv"
    if lc_path.exists():
        lc_records = aggregate.records_for_script("05_learning_curve")
        stamped(lc_records, sources=["artifacts/learning_curve.csv"])
        written += figures.figure_learning_curve(pd.read_csv(lc_path, comment="#"), out_dir)
        print("[fig] learning curve (%d records from 05)" % len(lc_records))
    else:
        skipped.append("learning curve: artifacts/learning_curve.csv (run scripts/05_learning_curve.py)")

    # ---- 6. ablation ----
    paired_path = artifacts_dir(data_cfg) / "ablation_paired.csv"
    per_class_path = artifacts_dir(data_cfg) / "ablation_per_class.csv"
    if paired_path.exists() and per_class_path.exists():
        ablation_records = aggregate.records_for_script("04_run_ablation")
        stamped(
            ablation_records,
            sources=["artifacts/ablation_paired.csv", "artifacts/ablation_per_class.csv"],
        )
        written += figures.figure_ablation(
            pd.read_csv(paired_path, comment="#"),
            pd.read_csv(per_class_path, comment="#"),
            out_dir,
        )
        print("[fig] class-weight ablation (%d records from 04)" % len(ablation_records))
    else:
        skipped.append(
            "ablation: artifacts/ablation_paired.csv + ablation_per_class.csv "
            "(run scripts/04_run_ablation.py)"
        )

    # ---- 7. selected-epoch distribution ----
    epochs_path = artifacts_dir(data_cfg) / "selected_epochs.csv"
    if epochs_path.exists():
        epochs = pd.read_csv(epochs_path, comment="#")
        if not epochs.empty:
            stamped(records, sources=["artifacts/selected_epochs.csv"])
            written += figures.figure_selected_epochs(epochs, out_dir)
            print("[fig] selected-epoch distribution")
    else:
        skipped.append("selected epochs: artifacts/selected_epochs.csv (run scripts/03_run_cv.py)")

    rule("DONE")
    print("[figures] wrote %d file(s) (%d figures, PDF + PNG each)"
          % (len(written), len(written) // 2))
    for path in written:
        print("    %s" % path.name)
    if skipped:
        print("\n[skipped] %d figure(s) whose inputs are not present yet:" % len(skipped))
        for reason in skipped:
            print("    %s" % reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
