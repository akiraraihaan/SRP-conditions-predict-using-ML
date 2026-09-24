"""Every artefact a script writes must be on the right side of .gitignore.

`git add` on an ignored path fails **silently**, so an artefact with no exception
is simply never committed and nobody finds out until it is needed. That has now
happened three times: legacy_grid_metrics.csv, folds_report.md, and then every
output of scripts 01b and 02-07 at once.

`git check-ignore` works on paths that do not exist yet, so this runs before
scripts 02-06 have ever produced their outputs. It reads nothing and writes
nothing -- it only asks git how it would treat a path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# (path, produced by, why it is committed)
COMMITTED = [
    ("artifacts/image_index.csv", "00", "frozen input; every artefact indexes by its idx"),
    ("artifacts/folds.json", "00", "frozen input; the reporting protocol"),
    ("artifacts/dev_split.json", "00", "Route A cannot re-run off-repo"),
    ("artifacts/excluded_images.csv", "00", "audit trail for the 27 exclusions"),
    ("artifacts/folds_report.md", "00", "reviewed artefact"),
    ("artifacts/legacy_contamination.json", "00", "the three legacy cross-references"),
    ("artifacts/legacy_grid_metrics.csv", "00/01", "the 46 legacy configs"),
    ("artifacts/legacy_test_predictions.csv", "00", "per-image predictions, quoted in the paper"),
    ("artifacts/registry.jsonl", "01-05", "the resume state"),
    ("artifacts/resolved_arms.yaml", "01/01b/02", "resolved hyperparameters; recovery path"),
    ("artifacts/medium_grid_complete.csv", "01", "the 18-config legacy grid table"),
    ("artifacts/uniform_grid.csv", "01b", "the 54-run uniform grid; selects the locked configs"),
    ("artifacts/uniform_grid_topk.csv", "01b", "the selection margin, reported in the manuscript"),
    ("artifacts/baseline_lr_sweep.csv", "02", "the 6-run baseline sweep"),
    ("artifacts/baseline_lr_sweep_topk.csv", "02", "the lr selection margin, at noise level"),
    ("artifacts/ablation_paired.csv", "04", "manuscript table"),
    ("artifacts/ablation_per_class.csv", "04", "manuscript table"),
    ("artifacts/learning_curve.csv", "05", "manuscript table"),
    ("artifacts/summary_cv.csv", "06", "manuscript table"),
    ("artifacts/summary_per_class.csv", "06", "manuscript table"),
    ("artifacts/selected_epochs.csv", "06", "manuscript table"),
    ("artifacts/paired_comparisons.csv", "06", "every pair of arms, fold by fold"),
    ("artifacts/pareto_status.csv", "06", "Pareto frontier and domination"),
    ("artifacts/edge_benchmark.json", "07", "Raspberry Pi results"),
    ("artifacts/pareto_status_device.csv", "07", "frontier on measured latency"),
    # 07 is run once per BOARD, with --out pointing at a subdirectory so several
    # devices can sit side by side. `artifacts/*` matches that directory and
    # stops git descending into it, so each board needs its own line in
    # .gitignore. Without one, `git add` fails silently and the results of a run
    # that took ten minutes on the hardware simply never reach the repository.
    ("artifacts/raspberry-pi-result/edge_benchmark.json", "07",
     "Raspberry Pi 3 results, --out subdirectory"),
    ("artifacts/raspberry-pi-result/pareto_status_device.csv", "07",
     "Raspberry Pi 3 measured-latency frontier"),
    ("artifacts/paired_comparisons_corrected.csv", "08",
     "Nadeau-Bengio corrected intervals; supersedes paired_comparisons.csv"),
    ("artifacts/taxonomy_merge.csv", "08", "macro-F1 under merged taxonomies"),
    ("artifacts/taxonomy_merge_pairs.csv", "08",
     "the control: all 45 single-pair merges ranked"),
    ("artifacts/taxonomy_merge_folds.csv", "08", "per-fold merged values"),
    ("artifacts/taxonomy_family_concentration.csv", "08",
     "where the six within-family pairs rank among all 45"),
    ("artifacts/quantisation.csv", "10",
     "dynamic vs static PTQ: size and accuracy, no latency"),
    ("artifacts/yolo_recipe_check.csv", "11",
     "the 2x2 breaking the optimizer/architecture confound"),
    ("artifacts/epoch_budget_check.csv", "11", "yolo26n at 50 epochs"),
    ("artifacts/epoch_budget_selection.csv", "11", "appendix C form"),
    ("artifacts/near_duplicates.csv", "09", "near-duplicate clusters"),
    ("artifacts/near_duplicate_summary.json", "09",
     "distance histogram, pixel evidence and the verdict"),
    ("artifacts/near_duplicate_contact_sheet.png", "09",
     "the ten largest clusters as thumbnails"),
    ("artifacts/pixel_duplicates.csv", "09",
     "exact decoded-pixel duplicate groups"),
    ("artifacts/figures/fig_taxonomy_pairs_mobilenetv3_small.pdf", "06",
     "the 45-pair merge ranking for the detailed arm"),
    ("artifacts/figures/fig_pareto.pdf", "06", "publication figure"),
    ("artifacts/figures/fig_pareto.png", "06", "publication figure"),
    ("artifacts/figures/fig_pareto_size.pdf", "06", "publication figure"),
    ("artifacts/figures/fig_pareto_size.png", "06", "publication figure"),
    ("artifacts/figures/fig_cv_box.pdf", "06", "publication figure"),
    ("artifacts/figures/fig_confusion_mobilenetv3_small.pdf", "06", "publication figure"),
    ("artifacts/figures/fig_confusion_resnet18.pdf", "06", "publication figure"),
    ("artifacts/figures/fig_ablation.pdf", "06", "publication figure"),
    ("artifacts/figures/fig_class_distribution.pdf", "06", "publication figure"),
    ("artifacts/figures/fig_cv_macro_f1.pdf", "06", "publication figure"),
    ("artifacts/figures/fig_learning_curve.pdf", "06", "publication figure"),
    ("artifacts/figures/fig_selected_epochs.pdf", "06", "publication figure"),
    ("artifacts/figures_pub/fig_confusion_mobilenetv3_small.pdf", "06", "typesetting set"),
    ("artifacts/figures_pub/fig_pareto.pdf", "06", "typesetting set"),
    ("artifacts/figures_pub/fig_pareto.png", "06", "typesetting set"),
]

# Written under artifacts/ but deliberately NOT committed.
IGNORED = [
    ("artifacts/registry.jsonl.20260101T000000Z.bak", "backfill_efficiency", "local safety copy"),
]


def is_ignored(path: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", path],
        capture_output=True,
    ).returncode == 0


@pytest.mark.parametrize("path,script,why", COMMITTED, ids=[c[0] for c in COMMITTED])
def test_committed_artefacts_are_not_ignored(path, script, why):
    assert not is_ignored(path), (
        "%s (written by script %s -- %s) is ignored, so `git add` would refuse it "
        "SILENTLY. Add an exception to .gitignore." % (path, script, why)
    )


@pytest.mark.parametrize("path,script,why", IGNORED, ids=[c[0] for c in IGNORED])
def test_uncommitted_artefacts_stay_ignored(path, script, why):
    assert is_ignored(path), "%s (%s) should not be committable" % (path, why)


def test_model_weights_and_runs_stay_ignored():
    for path in ("runs/classify/x/weights/best.pt", "weights/yolo26n-cls.pt",
                 "yolo26n-cls.pt", "artifacts/figures/scratch.pt"):
        assert is_ignored(path), "%s must never be committable" % path


def test_gitignore_has_no_inline_comments():
    """`#` only starts a comment at column 0. An inline one silently becomes part
    of the pattern, which is how ten exceptions were written and did nothing."""
    offenders = []
    for number, line in enumerate(
        (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "#" in line:
            offenders.append("%d: %s" % (number, line))
    assert not offenders, (
        "gitignore patterns with an inline comment, which git treats as part of "
        "the pattern:\n  " + "\n  ".join(offenders)
    )


def test_every_artefact_the_scripts_write_is_listed_here():
    """Guard against this test drifting behind the code.

    Any `artifacts_dir(...) / "name"` in the source must appear in COMMITTED or
    IGNORED, so adding an artefact without deciding its fate fails here.
    """
    import re

    known = {Path(p).name for p, _, _ in COMMITTED + IGNORED}
    known |= {"figures", "figures_pub"}  # directories, covered by the figure entries

    pattern = re.compile(r'artifacts_dir\([^)]*\)\s*/\s*"([^"]+)"')
    found: set[str] = set()
    for source in list((REPO_ROOT / "src").rglob("*.py")) + list(
        (REPO_ROOT / "scripts").rglob("*.py")
    ):
        found |= set(pattern.findall(source.read_text(encoding="utf-8")))

    missing = sorted(found - known)
    assert not missing, (
        "these artefacts are written by the scripts but not listed in this test, "
        "so nothing checks whether they are committable: %s" % missing
    )
