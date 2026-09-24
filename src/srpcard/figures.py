"""Publication figures. Matplotlib only -- seaborn is not a dependency.

Every figure is written twice: a vector PDF for the manuscript and a
high-resolution PNG fallback. Colours come from matplotlib's default cycle so
the set stays internally consistent, and every figure carries axis labels and
units.
"""

from __future__ import annotations

from contextlib import contextmanager
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

# Whether the stamp is DRAWN on the figure. It is always written into the file's
# metadata; this only controls the visible strip. A typeset manuscript does not
# want the strip, but a figure with no provenance at all cannot be identified
# later -- so the two are separated rather than one being traded for the other.
RENDER_PROVENANCE: bool = True


def set_provenance(block: dict | None) -> None:
    """Install the stamp every subsequent save() applies."""
    global PROVENANCE
    PROVENANCE = block


def set_render_provenance(enabled: bool) -> None:
    """Draw the provenance strip on the figure, or keep it to the metadata only."""
    global RENDER_PROVENANCE
    RENDER_PROVENANCE = bool(enabled)


@contextmanager
def provenance(block: dict | None, *, render: bool | None = None):
    """Install a stamp for the duration of one figure, then put it back.

    PROVENANCE and RENDER_PROVENANCE are module-level globals, which is what
    makes `save()` able to stamp without every figure function taking a
    provenance argument it never reads. The cost is that a caller which sets one
    and raises leaves it installed, and the NEXT figure is then stamped with the
    previous figure's record set -- the precise failure the per-artefact stamp
    was introduced to prevent, reintroduced by the mechanism that implements it.

    A test that sets the global and fails leaks it into every test after it in
    the same process, which is how this was found.

    Use this rather than set_provenance() wherever the stamp is meant to apply
    to a bounded piece of work:

        with figures.provenance(stamp):
            figures.figure_pareto(summary, out_dir)
    """
    previous_block, previous_render = PROVENANCE, RENDER_PROVENANCE
    set_provenance(block)
    if render is not None:
        set_render_provenance(render)
    try:
        yield block
    finally:
        set_provenance(previous_block)
        set_render_provenance(previous_render)


def _content_sha1_of(path: Path) -> str | None:
    """The content key recorded inside an already-written figure, if any."""
    import re

    try:
        raw = path.read_bytes()
    except OSError:
        return None
    match = re.search(rb"content=([0-9a-f]{6,40})", raw)
    return match.group(1).decode("ascii") if match else None


def _stamp(fig) -> None:
    """Draw the provenance strip along the bottom of the figure."""
    if not PROVENANCE or not RENDER_PROVENANCE:
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
    and the file's own metadata, so it survives being cropped into a manuscript.
    `set_render_provenance(False)` drops the visible strip and keeps the
    metadata, which is what `--for-publication` does.
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
            # scripts belongs here, not only in the Subject line: it is the
            # claim a reader checks -- "which runs is this figure of?"
            "Keywords": "records=%d scripts=%s arms=%s sources=%s registry=%s content=%s"
            % (
                PROVENANCE["n_records"],
                ",".join(PROVENANCE.get("scripts") or []) or "none",
                ",".join(PROVENANCE["arms"]) or "none",
                ",".join(PROVENANCE.get("sources") or []) or "none",
                PROVENANCE["registry_sha1"],
                PROVENANCE.get("content_sha1", "none"),
            ),
        }
    # PNG carries the same text, under the keys the PNG spec allows, so a
    # publication figure is identifiable in either format.
    png_metadata = (
        {
            "Title": metadata["Title"],
            "Description": metadata["Subject"],
            "Comment": metadata["Keywords"],
            "Software": metadata["Creator"],
        }
        if metadata
        else {}
    )

    written = [out_dir / ("%s.%s" % (name, suffix)) for suffix in ("pdf", "png")]

    # Only rewrite when the content changed. Every save embeds a timestamp, so
    # re-exporting an unchanged figure produced a diff on every run: review noise
    # that the stamp was not buying anything for. The record count, corpus
    # fingerprint and registry sha1 still identify a stale figure.
    wanted = (PROVENANCE or {}).get("content_sha1")
    if wanted and all(p.exists() for p in written):
        if all(_content_sha1_of(p) == wanted for p in written):
            return written

    for target in written:
        chosen = metadata if target.suffix == ".pdf" else png_metadata
        if chosen:
            fig.savefig(target, format=target.suffix.lstrip("."), metadata=chosen)
        else:
            fig.savefig(target, format=target.suffix.lstrip("."))
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


def sole_arm(records, *, fallback: str = "the locked") -> str:
    """The one arm a set of registry records describes, else the fallback.

    Figure titles must name the arm they were BUILT from. Hardcoding it survives
    a retarget silently: the analysis follows configs/arms.yaml, the caption
    does not, and the figure then states something the data never said.
    """
    arms = sorted({r.get("arm") for r in (records or []) if r.get("arm")})
    return arms[0] if len(arms) == 1 else fallback


def figure_learning_curve(summary, out_dir: Path, arm: str | None = None) -> list[Path]:
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
    # The arm is passed in from the RECORDS, never named in the string. This
    # title said "yolo26n" for weeks after script 05 was retargeted at
    # mobilenetv3_small (configs/arms.yaml:learning_curve.arm), so the figure
    # captioned the wrong model while the numbers underneath were correct.
    # artifacts/learning_curve.csv carries no arm column, which is why nothing
    # caught it; the caller reads it from the registry instead.
    ax.set_title(
        "Learning curve under the locked %s configuration" % arm
        if arm else "Learning curve under the locked configuration"
    )
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


def figure_taxonomy_pairs(pairs, arm: str, out_dir: Path) -> list[Path]:
    """All 45 single-pair merges for one arm, ranked, with the control visible.

    The point of the figure is the CONTROL, not the hypothesis. Merging any two
    of ten classes raises macro-F1 for free, so a bar chart of the proposed
    merges alone would prove nothing. Drawing all 45 with the baseline and the
    median marked shows whether the hypothesised pairs sit above the
    distribution or inside it -- which is the whole claim.
    """
    import matplotlib.pyplot as plt

    _style()
    block = pairs[pairs["arm"] == arm].sort_values("gain_mean", ascending=True)
    if block.empty:
        return []

    values = block["macro_f1_mean"].to_numpy(dtype=float)
    labels = [
        "%s + %s" % (a.replace("_", " "), b.replace("_", " "))
        for a, b in zip(block["class_a"], block["class_b"])
    ]
    hypothesised = [bool(h) for h in block["hypothesised"].fillna("")]
    # Three colours, not two. The two HYPOTHESISED pairs (M1, M2) were proposed
    # in advance; the other four within-family pairs were not, and their arrival
    # at the top of the ranking alongside them is the actual result -- those
    # four classes are one error family rather than two separate pairs. Merging
    # the two categories into one colour would hide exactly that.
    if "within_family" in block.columns:
        family = [bool(f) for f in block["within_family"].fillna(False)]
    else:
        family = list(hypothesised)

    fig, ax = plt.subplots(figsize=(7.2, 9.0))
    positions = np.arange(len(values))
    colours = [
        "tab:orange" if h else ("tab:red" if f else "tab:blue")
        for h, f in zip(hypothesised, family)
    ]
    ax.barh(positions, values, color=colours, height=0.78)

    # gain = merged - baseline, and the baseline is the same for every row, so
    # any row recovers it. Taking it from one row rather than from two separate
    # minima keeps that obvious.
    baseline = float(block["macro_f1_mean"].iloc[0] - block["gain_mean"].iloc[0])
    median = float(np.median(values))
    ax.axvline(baseline, color="black", linewidth=1.2,
               label="baseline, 10 classes (%.4f)" % baseline)
    ax.axvline(median, color="tab:red", linestyle="--", linewidth=1.2,
               label="median of all 45 merges (%.4f)" % median)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_ylim(-0.7, len(values) - 0.3)
    ax.set_xlim(left=min(baseline, values.min()) - 0.004)
    ax.set_xlabel("macro-F1 after merging the pair (mean over 15 folds)")
    ax.set_title(
        "Every possible single-pair merge, %s\n"
        "the four-class error family occupies the top of the ranking"
        % arm.replace("_", " ")
    )

    from matplotlib.patches import Patch

    n_family = sum(family)
    ax.legend(
        handles=[
            Patch(facecolor="tab:orange", label="hypothesised in advance (M1, M2)"),
            Patch(facecolor="tab:red",
                  label="other within-family pairs (%d)" % max(n_family - 2, 0)),
            Patch(facecolor="tab:blue", label="the remaining %d" % (len(values) - n_family)),
        ] + ax.get_legend_handles_labels()[0],
        fontsize=7, loc="lower right",
    )
    fig.tight_layout()
    return save(fig, out_dir, "fig_taxonomy_pairs_%s" % arm)


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
