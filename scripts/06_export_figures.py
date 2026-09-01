#!/usr/bin/env python
"""06 -- export every publication figure. Vector PDF plus high-resolution PNG.

    python scripts/06_export_figures.py
    python scripts/06_export_figures.py --out-dir artifacts/figures

Matplotlib only; seaborn is not a dependency of this repository.

Also refreshes the manuscript tables via src/srpcard/aggregate.py:
  artifacts/summary_cv.csv, summary_per_class.csv, selected_epochs.csv

Figures whose inputs are missing are skipped with a message naming the script
that produces them, so a partial run still emits everything it can.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from srpcard import aggregate, figures  # noqa: E402
from srpcard import data as srp_data  # noqa: E402
from srpcard.config import artifacts_dir, load_data_config  # noqa: E402


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None, help="default: artifacts/figures")
    args = parser.parse_args()

    data_cfg = load_data_config()
    out_dir = Path(args.out_dir) if args.out_dir else artifacts_dir(data_cfg) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rule("06 -- publication figures")
    print("[out] %s" % out_dir)

    written: list[Path] = []
    skipped: list[str] = []

    # ---- tables first: the figures read them ----
    tables = aggregate.write_all(data_cfg)
    for name, path in tables.items():
        print("[table] %s -> %s" % (name, path.name))
    if not tables:
        print("[table] no 03_run_cv records yet; tables skipped")

    # ---- 1. class distribution ----
    try:
        index = srp_data.load_image_index()
        written += figures.figure_class_distribution(index, data_cfg, out_dir)
        print("[fig] class distribution")
    except Exception as exc:  # noqa: BLE001
        skipped.append("class distribution: %s" % exc)

    records = aggregate.cv_records()

    # ---- 2. per-arm macro-F1 boxplot ----
    if records:
        written += figures.figure_cv_boxplot(records, out_dir)
        print("[fig] cross-validated macro-F1 by arm")
    else:
        skipped.append("cv boxplot: no 03_run_cv records (run scripts/03_run_cv.py)")

    # ---- 3. Pareto ----
    summary_path = artifacts_dir(data_cfg) / "summary_cv.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if summary["gflops_mean"].notna().any():
            written += figures.figure_pareto(summary, out_dir)
            print("[fig] Pareto frontier")
        else:
            skipped.append("pareto: gflops missing from summary_cv.csv")
    else:
        skipped.append("pareto: artifacts/summary_cv.csv (run scripts/03_run_cv.py)")

    # ---- 4. confusion matrix of the best arm ----
    if records:
        summary = aggregate.summarise_cv(records, data_cfg)
        best_arm = summary.iloc[0]["arm"]
        matrix = aggregate.mean_confusion_matrix(best_arm, records)
        if matrix is not None:
            written += figures.figure_confusion(
                matrix,
                list(data_cfg["classes"]),
                out_dir,
                "fig_confusion_%s" % best_arm,
                "Confusion matrix, %s, summed over 15 folds" % best_arm,
            )
            print("[fig] confusion matrix (%s)" % best_arm)

    # ---- 5. learning curve ----
    lc_path = artifacts_dir(data_cfg) / "learning_curve.csv"
    if lc_path.exists():
        written += figures.figure_learning_curve(pd.read_csv(lc_path), out_dir)
        print("[fig] learning curve")
    else:
        skipped.append("learning curve: artifacts/learning_curve.csv (run scripts/05_learning_curve.py)")

    # ---- 6. ablation ----
    paired_path = artifacts_dir(data_cfg) / "ablation_paired.csv"
    per_class_path = artifacts_dir(data_cfg) / "ablation_per_class.csv"
    if paired_path.exists() and per_class_path.exists():
        written += figures.figure_ablation(
            pd.read_csv(paired_path), pd.read_csv(per_class_path), out_dir
        )
        print("[fig] class-weight ablation")
    else:
        skipped.append(
            "ablation: artifacts/ablation_paired.csv + ablation_per_class.csv "
            "(run scripts/04_run_ablation.py)"
        )

    # ---- 7. selected-epoch distribution ----
    epochs_path = artifacts_dir(data_cfg) / "selected_epochs.csv"
    if epochs_path.exists():
        epochs = pd.read_csv(epochs_path)
        if not epochs.empty:
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
