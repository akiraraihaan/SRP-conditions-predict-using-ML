"""Cross-references between the conflict-group exclusion and the old results.

Three questions, answered from artefacts in the read-only legacy directory plus
one re-run of INFERENCE (not training) with the already-trained selected weights.
Nothing is written to the legacy directory.

  1. How many excluded images fell inside the legacy dev split's val and test
     partitions -- i.e. how much of the reported test F1-macro was contaminated.
  2. Do the confusion-matrix errors the manuscript attributes to inter-class
     visual similarity involve excluded images? If they do, that paragraph is
     wrong.
  3. Which split did the within-nano Friedman test actually run on?

This module is an addition to the module list in the brief; it exists so these
answers are reproducible rather than ad hoc.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_data_config
from .data import normalize_class_name
from .legacy_split import SPLIT_NAMES, legacy_dir

# The off-diagonal cells the manuscript explains as inter-class visual similarity.
MANUSCRIPT_SIMILARITY_CELLS = [
    ("natural_flowing", "pump_leakage"),
    ("pump_leakage", "natural_flowing"),
    ("natural_flowing", "severe_vibration"),
]


def collapse_confusion_matrix(
    matrix: list[list[int]], class_names: list[str], canonical: list[str], pattern: str = r"^\d+_"
) -> np.ndarray:
    """Collapse a legacy confusion matrix to 10x10 in canonical order.

    The old matrices are 14x14 because evaluation unioned prefixed TRUE labels
    with unprefixed PREDICTED labels. The trained models had 10 outputs -- they
    were never 14-class models. Normalise the names and sum each cell into its
    canonical class.
    """
    source = np.asarray(matrix, dtype=int)
    position = {name: i for i, name in enumerate(canonical)}
    collapsed = np.zeros((len(canonical), len(canonical)), dtype=int)
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            collapsed[
                position[normalize_class_name(true_name, pattern)],
                position[normalize_class_name(pred_name, pattern)],
            ] += source[i, j]
    if collapsed.sum() != source.sum():
        raise ValueError(
            f"collapse lost mass: {source.sum()} -> {collapsed.sum()}. Check class names."
        )
    return collapsed


def crossref_1_split_contamination(
    index: pd.DataFrame, dev_split: dict[str, list[int]]
) -> dict[str, Any]:
    """Where the excluded images landed in the legacy development split."""
    excluded = set(int(i) for i in index.loc[index["excluded"].astype(bool), "idx"])
    class_of = dict(zip(index["idx"].tolist(), index["class"].tolist()))
    sha_of = dict(zip(index["idx"].tolist(), index["sha1"].tolist()))
    where = {i: s for s in SPLIT_NAMES for i in dev_split[s]}

    result: dict[str, Any] = {"n_excluded_total": len(excluded), "by_split": {}}
    for split in SPLIT_NAMES:
        hits = sorted(excluded & set(dev_split[split]))
        result["by_split"][split] = {
            "n_partition": len(dev_split[split]),
            "n_excluded": len(hits),
            "fraction": round(len(hits) / len(dev_split[split]), 4),
            "images": [
                {"idx": i, "class": class_of[i], "sha1": sha_of[i]} for i in hits
            ],
        }

    # A conflict group with members on both sides of a partition boundary is
    # leakage: the model saw the same bytes in training under another label.
    leaks = []
    for sha, members in index.loc[index["excluded"].astype(bool)].groupby("sha1"):
        placement = [
            {"idx": int(i), "class": c, "split": where[int(i)]}
            for i, c in zip(members["idx"].tolist(), members["class"].tolist())
        ]
        splits_present = {p["split"] for p in placement}
        if splits_present != {"train"}:
            leaks.append({"sha1": sha, "members": placement})
    result["leaking_groups"] = leaks
    result["n_leaking_groups"] = len(leaks)
    result["n_groups_confined_to_train"] = int(
        index.loc[index["excluded"].astype(bool), "sha1"].nunique() - len(leaks)
    )
    return result


def crossref_2_confusion_errors(
    predictions: pd.DataFrame, index: pd.DataFrame, canonical: list[str]
) -> dict[str, Any]:
    """Do the 'inter-class similarity' errors involve excluded images?"""
    excluded = set(int(i) for i in index.loc[index["excluded"].astype(bool), "idx"])
    errors = predictions.loc[predictions["true"] != predictions["pred"]]

    cells: dict[str, Any] = {}
    total_asked = 0
    total_excluded = 0
    for true_name, pred_name in MANUSCRIPT_SIMILARITY_CELLS:
        subset = errors.loc[(errors["true"] == true_name) & (errors["pred"] == pred_name)]
        involved = [int(i) for i in subset["idx"] if int(i) in excluded]
        total_asked += len(subset)
        total_excluded += len(involved)
        cells[f"{true_name} -> {pred_name}"] = {
            "n_errors": len(subset),
            "idx": [int(i) for i in subset["idx"]],
            "n_involving_excluded": len(involved),
            "excluded_idx": involved,
        }

    contaminated_errors = [
        {
            "idx": int(row.idx),
            "true": row.true,
            "pred": row.pred,
        }
        for row in errors.itertuples()
        if int(row.idx) in excluded
    ]

    return {
        "manuscript_cells": cells,
        "n_errors_in_those_cells": total_asked,
        "n_of_those_involving_excluded": total_excluded,
        "manuscript_claim_survives": total_excluded == 0,
        "all_contaminated_errors": contaminated_errors,
        "n_errors_total": int(len(errors)),
    }


def crossref_3_friedman_split(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Report verbatim what the within-nano Friedman artefacts say about the split."""
    cfg = cfg or load_data_config()
    root = legacy_dir(cfg)
    within = root / "results" / "friedman_within_nano.json"
    summary = root / "results" / "friedman_summary.json"

    result: dict[str, Any] = {"file": str(within)}
    if not within.exists():
        result["available"] = False
        result["reason"] = f"not found: {within}"
        return result
    result["available"] = True

    with open(within, encoding="utf-8") as fh:
        payload = json.load(fh)

    result["top_level_keys"] = list(payload)
    result["effective_split_field_present"] = "effective_split" in payload
    result["effective_split_verbatim"] = payload.get(
        "effective_split", "<<field absent from this file>>"
    )
    result["per_split"] = {
        split: {
            "chi2": block.get("chi2"),
            "p_value": block.get("p_value"),
            "significant": block.get("significant"),
            "nemenyi_present": block.get("nemenyi") is not None,
            "n_blocks": block.get("n_blocks"),
            "n_treatments": block.get("n_treatments"),
        }
        for split, block in payload.items()
        if isinstance(block, dict) and "chi2" in block
    }
    result["nemenyi_csv_files_on_disk"] = sorted(
        p.name for p in (root / "results").glob("friedman_within_nano_nemenyi_*.csv")
    )

    if summary.exists():
        with open(summary, encoding="utf-8") as fh:
            result["friedman_summary_effective_split_verbatim"] = json.load(fh).get(
                "effective_split", "<<field absent>>"
            )
    return result


def legacy_test_predictions(
    index: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    cache: Path | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame | None:
    """Per-image predictions of the selected legacy model on the legacy test set.

    Re-runs INFERENCE only, with weights that already exist. This is not one of
    the 46 completed grid-search configurations being re-trained; nothing is
    written to the legacy directory. Returns None if the weights or ultralytics
    are unavailable.
    """
    cfg = cfg or load_data_config()
    if cache is not None and Path(cache).exists():
        if verbose:
            print(f"[crossref] using cached predictions: {cache}")
        return pd.read_csv(cache)

    root = legacy_dir(cfg)
    weights = root / cfg["legacy_reference"]["selected_weights"]
    test_dir = root / cfg["legacy_reference"]["split_dir"] / "test"
    if not weights.exists() or not test_dir.exists():
        if verbose:
            print(f"[crossref] unavailable -- missing {weights if not weights.exists() else test_dir}")
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        if verbose:
            print("[crossref] unavailable -- ultralytics not installed")
        return None

    pattern = cfg.get("class_prefix_pattern", r"^\d+_")
    paths, truth = [], []
    for class_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        name = normalize_class_name(class_dir.name, pattern)
        for path in sorted(p for p in class_dir.iterdir() if p.is_file()):
            paths.append(path)
            truth.append(name)

    model = YOLO(str(weights))
    names = {
        int(k): str(v)
        for k, v in (model.names.items() if isinstance(model.names, dict) else enumerate(model.names))
    }
    if verbose:
        print(
            f"[crossref] {weights.name}: model.names has {len(names)} entries "
            f"-- a 10-class model, not a 14-class one"
        )
    results = model.predict(source=[str(p) for p in paths], imgsz=224, verbose=False, stream=False)
    preds = [normalize_class_name(names[int(r.probs.top1)], pattern) for r in results]

    lookup = {
        (cls, Path(rel).name): int(i)
        for i, rel, cls in zip(index["idx"], index["relpath"], index["class"])
    }
    frame = pd.DataFrame(
        [
            {"idx": lookup[(t, p.name)], "true": t, "pred": q, "correct": t == q}
            for p, t, q in zip(paths, truth, preds)
        ]
    ).sort_values("idx")
    if cache is not None:
        frame.to_csv(cache, index=False, lineterminator="\n")
    return frame
