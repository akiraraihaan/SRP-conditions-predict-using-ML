"""Publication figures. Matplotlib only -- seaborn is not a dependency.

Every figure is written twice: a vector PDF for the manuscript and a
high-resolution PNG fallback. Colours come from matplotlib's default cycle so
the set stays internally consistent, and every figure carries axis labels and
units.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DPI = 300
FIGURE_DIR_NAME = "figures"


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,   # embed TrueType, not Type 3 -- required by many journals
            "ps.fonttype": 42,
        }
    )


# Set by script 06 before it draws anything: the provenance stamp that goes on
# every figure. A stale figure then says so on its own face rather than waiting
# to be noticed. None means "no stamp", which is what a caller outside 06 gets.
PROVENANCE: dict | None = None


def set_provenance(block: dict | None) -> None:
    """Install the stamp every subsequent save() applies."""
    global PROVENANCE
    PROVENANCE = block


def _stamp(fig) -> None:
    """Draw the provenance strip along the bottom of the figure."""
    if not PROVENANCE:
        return
    from .aggregate import provenance_caption

    fig.text(
        0.005,
        0.004,
        provenance_caption(PROVENANCE),
        fontsize=4.5,
        color="#888888",
        ha="left",
        va="bottom",
    )


def save(fig, out_dir: Path, name: str) -> list[Path]:
    """Write `name`.pdf and `name`.png, both carrying the provenance stamp.

    The stamp goes in two places: a small strip along the bottom of the image,
    and the PDF's own metadata, so it survives being cropped into a manuscript.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _stamp(fig)
    metadata = {}
    if PROVENANCE:
        from .aggregate import provenance_caption

        metadata = {
            "Title": name,
            "Subject": provenance_caption(PROVENANCE),
            "Creator": "srpcard/figures.py",
            "Keywords": "records=%d arms=%s registry=%s"
            % (
                PROVENANCE["n_records"],
                ",".join(PROVENANCE["arms"]),
                PROVENANCE["registry_sha1"],
            ),
        }
    written = []
    for suffix in ("pdf", "png"):
        target = out_dir / ("%s.%s" % (name, suffix))
        if suffix == "pdf" and metadata:
            fig.savefig(target, format=suffix, metadata=metadata)
        else:
            fig.savefig(target, format=suffix)
        written.append(target)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return written


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def figure_class_distribution(index, data_cfg: dict[str, Any], out_dir: Path) -> list[Path]:
    """Per-class counts before and after the conflict-group exclusion."""
    import matplotlib.pyplot as plt

    _style()
    classes = list(data_cfg["classes"])
    before = [int((index["class"] == c).sum()) for c in classes]
    kept = [
        int(((index["class"] == c) & (~index["excluded"].astype(bool))).sum()) for c in classes
    ]
    dropped = [b - k for b, k in zip(before, kept)]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    positions = np.arange(len(classes))
    ax.bar(positions, kept, label="retained (n=%d)" % sum(kept))
    ax.bar(positions, dropped, bottom=kept, label="excluded (n=%d)" % sum(dropped))
    ax.set_xticks(positions)
    ax.set_xticklabels([c.replace("_", " ") for c in classes], rotation=40, ha="right")
    ax.set_ylabel("images")
    ax.set_title("Class distribution before and after duplicate-label exclusion")
    ax.legend()
    return save(fig, out_dir, "fig_class_distribution")


def figure_cv_boxplot(records: list[dict], out_dir: Path) -> list[Path]:
    """Per-arm distribution of test macro-F1 across the 15 folds."""
    import matplotlib.pyplot as plt

    _style()
    arms = sorted({r["arm"] for r in records})
    data = [[r["f1_macro"] for r in records if r["arm"] == arm] for arm in arms]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.boxplot(data, tick_labels=[a.replace("_", "\n") for a in arms], showmeans=True)
    for position, values in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.045, len(values))
        ax.plot(position + jitter, values, ".", alpha=0.5, markersize=4)
    ax.set_ylabel("test macro-F1")
    ax.set_title("Cross-validated macro-F1 by architecture (5x3 folds)")
    return save(fig, out_dir, "fig_cv_macro_f1")


def figure_pareto(summary, out_dir: Path) -> list[Path]:
    """Accuracy against cost, with the Pareto frontier marked."""
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    x = summary["gflops_mean"].to_numpy(dtype=float)
    y = summary["f1_macro_mean"].to_numpy(dtype=float)
    err = summary["f1_macro_std"].to_numpy(dtype=float)
    names = summary["arm"].tolist()

    ax.errorbar(x, y, yerr=err, fmt="o", capsize=3, markersize=6)
    for xi, yi, name in zip(x, y, names):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Pareto: no other point has both lower GFLOPs and higher macro-F1
    optimal = [
        i
        for i in range(len(x))
        if not any((x[j] <= x[i]) and (y[j] >= y[i]) and (j != i) for j in range(len(x)))
    ]
    if optimal:
        order = np.argsort(x[optimal])
        ax.plot(
            x[np.array(optimal)][order],
            y[np.array(optimal)][order],
            "--",
            linewidth=1,
            label="Pareto frontier",
        )
        ax.legend()

    ax.set_xlabel("GFLOPs per inference")
    ax.set_ylabel("test macro-F1 (mean $\\pm$ s.d. over 15 folds)")
    ax.set_title("Accuracy against computational cost")
    return save(fig, out_dir, "fig_pareto")


def figure_pareto_size(summary, out_dir: Path) -> list[Path]:
    """Accuracy against MODEL SIZE, with the Pareto frontier marked.

    The size axis is fp16 -- half-precision weights, which is what the framework
    deploys -- and the axis label says so. An fp32 state_dict is twice the size
    and is never the artefact that reaches the device; a reader who assumes the
    wrong precision misreads the deployment cost by a factor of two, so the
    precision is named on the axis rather than left to the caption.
    """
    import matplotlib.pyplot as plt

    column = "size_mb_fp16_mean" if "size_mb_fp16_mean" in summary else "size_mb_mean"

    _style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    x = summary[column].to_numpy(dtype=float)
    y = summary["f1_macro_mean"].to_numpy(dtype=float)
    err = summary["f1_macro_std"].to_numpy(dtype=float)
    names = summary["arm"].tolist()

    ax.errorbar(x, y, yerr=err, fmt="o", capsize=3, markersize=6)
    for xi, yi, name in zip(x, y, names):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8)

    optimal = [
        i
        for i in range(len(x))
        if not any((x[j] <= x[i]) and (y[j] >= y[i]) and (j != i) for j in range(len(x)))
    ]
    if optimal:
        order = np.argsort(x[optimal])
        ax.plot(
            x[np.array(optimal)][order],
            y[np.array(optimal)][order],
            "--",
            linewidth=1,
            label="Pareto frontier",
        )
        ax.legend()

    ax.set_xlabel("model size (MB, fp16 weights as deployed)")
    ax.set_ylabel("test macro-F1 (mean $\\pm$ s.d. over 15 folds)")
    ax.set_title("Accuracy against model size")
    return save(fig, out_dir, "fig_pareto_size")


def figure_confusion(matrix, classes, out_dir: Path, name: str, title: str) -> list[Path]:
    """Row-normalised confusion matrix. Matplotlib imshow, no seaborn."""
    import matplotlib.pyplot as plt

    _style()
    matrix = np.asarray(matrix, dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([c.replace("_", " ") for c in classes], rotation=40, ha="right")
    ax.set_yticklabels([c.replace("_", " ") for c in classes])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    ax.grid(False)

    threshold = 0.55
    for i in range(len(classes)):
        for j in range(len(classes)):
            if matrix[i, j] > 0:
                ax.text(
                    j, i, "%d" % matrix[i, j],
                    ha="center", va="center", fontsize=7,
                    color="white" if normalised[i, j] > threshold else "black",
                )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="row-normalised rate")
    return save(fig, out_dir, name)


def figure_learning_curve(summary, out_dir: Path) -> list[Path]:
    """Mean +/- s.d. macro-F1 against training-set size."""
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = summary["n_train_mean"].to_numpy(dtype=float)
    y = summary["f1_macro_mean"].to_numpy(dtype=float)
    err = summary["f1_macro_std"].to_numpy(dtype=float)

    ax.errorbar(x, y, yerr=err, marker="o", capsize=3, linewidth=1.5)
    ax.fill_between(x, y - err, y + err, alpha=0.15)
    ax.set_xlabel("training images per fold")
    ax.set_ylabel("test macro-F1 (mean $\\pm$ s.d.)")
    ax.set_title("Learning curve under the locked yolo26n configuration")
    return save(fig, out_dir, "fig_learning_curve")


def figure_ablation(paired, per_class, out_dir: Path) -> list[Path]:
    """Paired per-fold deltas, and per-class recall deltas rarest-first."""
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    ax = axes[0]
    labels = ["r%df%d" % (r, f) for r, f in zip(paired["repeat"], paired["fold"])]
    deltas = paired["delta"].to_numpy(dtype=float)
    colours = ["tab:blue" if d >= 0 else "tab:red" for d in deltas]
    ax.bar(range(len(deltas)), deltas, color=colours)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(deltas.mean(), linestyle="--", linewidth=1,
               label="mean %+0.4f" % deltas.mean())
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("macro-F1: weighted $-$ unweighted")
    ax.set_title("Paired per-fold difference")
    ax.legend()

    ax = axes[1]
    positions = np.arange(len(per_class))
    ax.barh(positions, per_class["delta_mean"].to_numpy(dtype=float),
            xerr=per_class["delta_std"].to_numpy(dtype=float), capsize=2)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(
        ["%s (n=%d)" % (c.replace("_", " "), n)
         for c, n in zip(per_class["class"], per_class["n_clean"])],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_xlabel("recall: weighted $-$ unweighted")
    ax.set_title("Per-class recall delta (rarest class first)")

    fig.tight_layout()
    return save(fig, out_dir, "fig_ablation")


def figure_selected_epochs(epochs, out_dir: Path) -> list[Path]:
    """Where the selection criterion landed, relative to the epoch budget."""
    import matplotlib.pyplot as plt

    _style()
    arms = sorted(epochs["arm"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    data = [epochs.loc[epochs["arm"] == arm, "fraction_of_budget"].to_numpy() for arm in arms]
    ax.boxplot(data, tick_labels=[a.replace("_", "\n") for a in arms], showmeans=True)
    ax.axhline(1.0, linestyle="--", linewidth=1, color="tab:red",
               label="epoch budget exhausted")
    ax.set_ylabel("selected epoch / epoch budget")
    ax.set_ylim(0, 1.08)
    ax.set_title("Where best-weight selection landed")
    ax.legend()
    return save(fig, out_dir, "fig_selected_epochs")
