"""Recovery of the legacy 80:10:10 development split.

This is the DEVELOPMENT SPLIT. Its only job is comparability with the 46
completed grid-search runs, so it is reconstructed on the ORIGINAL 695 images
with their original labels -- the conflict-group exclusion is deliberately NOT
applied here. Those 46 runs were trained on contaminated data and cannot be
retro-fixed; making the dev split disagree with them would defeat its purpose.

Used by scripts 01 and 02 for hyperparameter selection only. Never for
reporting; reporting uses artifacts/folds.json, built on the clean 668.

Three recovery routes, tried in order:

  Route A (preferred)  Read the file lists the old run materialised on disk and
                       match them to the new index. Robust to the directory
                       renaming between the two corpora.
  Route B (fallback)   Replay the exact train_test_split calls.
  Route C              Neither worked: stop and report. Never substitute a
                       fresh split silently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import REPO_ROOT, MissingInputError, artifacts_dir, load_data_config, require_file
from .data import normalize_class_name

SPLIT_NAMES = ("train", "val", "test")

# The distribution this reconstruction must reproduce exactly.
EXPECTED_DISTRIBUTION: dict[str, tuple[int, int, int]] = {
    "collide_pump_and_vibration": (28, 4, 3),
    "full_load_production": (42, 5, 5),
    "gas_influence": (26, 4, 3),
    "gas_influence_and_vibration": (56, 7, 7),
    "insufficient_liquid_supply_and_vibration": (85, 10, 11),
    "natural_flowing": (74, 9, 10),
    "pump_leakage": (56, 7, 7),
    "severe_insufficient_liquid_supply": (42, 5, 5),
    "severe_vibration": (106, 13, 14),
    "vibration": (41, 5, 5),
}
EXPECTED_TOTALS = (556, 69, 70)


def legacy_dir(cfg: dict[str, Any] | None = None) -> Path:
    """Resolve the read-only legacy reference directory. Never written to."""
    import os

    cfg = cfg or load_data_config()
    env = os.environ.get("SRPCARD_LEGACY_DIR")
    path = Path(env) if env else Path(cfg["legacy_reference"]["dir"])
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


# --------------------------------------------------------------------------
# Route A -- match the materialised split
# --------------------------------------------------------------------------


def recover_route_a(
    index: pd.DataFrame, cfg: dict[str, Any] | None = None, *, verbose: bool = True
) -> tuple[dict[str, list[int]] | None, dict[str, Any]]:
    """Match the old materialised train/val/test directories to the new index.

    sha1 first, filename second. sha1 is expected to miss entirely: the old
    directories hold letterboxed re-encodings, whose bytes differ from the raw
    originals. Filename matching is scoped by normalised class, which keeps it
    unambiguous even for the conflict groups, whose members share a filename
    across different classes.
    """
    cfg = cfg or load_data_config()
    root = legacy_dir(cfg) / cfg["legacy_reference"]["split_dir"]
    report: dict[str, Any] = {"route": "A", "source": str(root)}

    if not root.exists():
        report["available"] = False
        report["reason"] = f"legacy split directory not found: {root}"
        if verbose:
            print(f"[route A] unavailable -- {report['reason']}")
        return None, report
    report["available"] = True

    pattern = cfg.get("class_prefix_pattern", r"^\d+_")
    by_sha = dict(zip(index["sha1"].tolist(), (int(i) for i in index["idx"])))
    by_class_name: dict[tuple[str, str], list[int]] = {}
    for idx_value, relpath, class_name in zip(
        index["idx"].tolist(), index["relpath"].tolist(), index["class"].tolist()
    ):
        by_class_name.setdefault((class_name, Path(relpath).name), []).append(int(idx_value))

    split_idx: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    matched_by = {"sha1": 0, "filename": 0}
    unmatched: list[str] = []
    ambiguous: list[str] = []
    seen: dict[int, str] = {}
    duplicated_assignment: list[str] = []
    raw_dirs: set[str] = set()

    for split in SPLIT_NAMES:
        split_root = root / split
        if not split_root.exists():
            report["available"] = False
            report["reason"] = f"missing split directory: {split_root}"
            if verbose:
                print(f"[route A] unavailable -- {report['reason']}")
            return None, report
        for class_dir in sorted(d for d in split_root.iterdir() if d.is_dir()):
            raw_dirs.add(class_dir.name)
            class_name = normalize_class_name(class_dir.name, pattern)
            for path in sorted(p for p in class_dir.rglob("*") if p.is_file()):
                found: int | None = None
                try:
                    from .data import sha1_of_file

                    digest = sha1_of_file(path)
                except OSError:
                    digest = None
                if digest is not None and digest in by_sha:
                    found = by_sha[digest]
                    matched_by["sha1"] += 1
                else:
                    candidates = by_class_name.get((class_name, path.name), [])
                    if len(candidates) == 1:
                        found = candidates[0]
                        matched_by["filename"] += 1
                    elif len(candidates) > 1:
                        ambiguous.append(f"{split}/{class_dir.name}/{path.name}")
                        continue
                if found is None:
                    unmatched.append(f"{split}/{class_dir.name}/{path.name}")
                    continue
                if found in seen:
                    duplicated_assignment.append(
                        f"idx {found}: {seen[found]} and {split}/{class_dir.name}/{path.name}"
                    )
                    continue
                seen[found] = f"{split}/{class_dir.name}/{path.name}"
                split_idx[split].append(found)

    report["legacy_directory_names"] = sorted(raw_dirs)
    report["n_files_on_disk"] = sum(len(v) for v in split_idx.values()) + len(unmatched) + len(
        ambiguous
    )
    report["matched_by"] = matched_by
    report["n_unmatched"] = len(unmatched)
    report["unmatched"] = unmatched[:20]
    report["n_ambiguous"] = len(ambiguous)
    report["ambiguous"] = ambiguous[:20]
    report["n_duplicated_assignment"] = len(duplicated_assignment)
    report["duplicated_assignment"] = duplicated_assignment[:20]
    report["n_index_rows_unassigned"] = int(len(index) - len(seen))

    if verbose:
        print(f"[route A] source: {root}")
        print(f"[route A] legacy directory names: {sorted(raw_dirs)}")
        print(
            f"[route A] matched {matched_by['sha1']} by sha1, "
            f"{matched_by['filename']} by filename (scoped to class)"
        )
        print(
            f"[route A] unmatched={len(unmatched)}  ambiguous={len(ambiguous)}  "
            f"double-assigned={len(duplicated_assignment)}  "
            f"index rows unassigned={report['n_index_rows_unassigned']}"
        )

    if unmatched or ambiguous or duplicated_assignment or report["n_index_rows_unassigned"]:
        report["ok"] = False
        return None, report

    report["ok"] = True
    return {name: sorted(split_idx[name]) for name in SPLIT_NAMES}, report


# --------------------------------------------------------------------------
# Route B -- replay the original calls
# --------------------------------------------------------------------------


def recover_route_b(
    index: pd.DataFrame, cfg: dict[str, Any] | None = None, *, verbose: bool = True
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Replay the exact train_test_split calls from code.ipynb cell 5 ("[Cell 6]").

    Reproduced verbatim, including the argument order:

        train_df, temp_df = train_test_split(
            df, test_size=0.2, stratify=df["class_name"], random_state=42)
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df["class_name"], random_state=42)

    Note test_size (not train_size) on the second call, so val_df is the FIRST
    return value. Both calls take random_state=42 and sklearn's default
    shuffle=True.

    The per-class counts this produces are fixed by the class counts and the
    random_state alone, so they are stable. WHICH image lands where is not: it
    depends on row order, and the old row order came from an os-dependent
    rglob over 14 directory names that no longer exist.
    """
    from sklearn.model_selection import train_test_split

    cfg = cfg or load_data_config()
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

    frame = pd.DataFrame({"idx": index["idx"].to_numpy(), "class_name": index["class"].to_numpy()})
    train_df, temp_df = train_test_split(
        frame,
        test_size=(ratios["val"] + ratios["test"]),
        stratify=frame["class_name"],
        random_state=42,
    )
    val_ratio_adj = ratios["val"] / (ratios["val"] + ratios["test"])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_adj),
        stratify=temp_df["class_name"],
        random_state=42,
    )
    result = {
        "train": sorted(int(i) for i in train_df["idx"]),
        "val": sorted(int(i) for i in val_df["idx"]),
        "test": sorted(int(i) for i in test_df["idx"]),
    }
    report = {
        "route": "B",
        "ok": True,
        "random_state": 42,
        "call_1": "train_test_split(df, test_size=0.2, stratify=class, random_state=42)",
        "call_2": "train_test_split(temp, test_size=0.5, stratify=class, random_state=42)",
        "note": "per-class counts are order-independent; image identity is not",
    }
    if verbose:
        print(f"[route B] replayed both calls, random_state=42 -> "
              f"{len(result['train'])}/{len(result['val'])}/{len(result['test'])}")
    return result, report


# --------------------------------------------------------------------------
# assertion
# --------------------------------------------------------------------------


def distribution_table(index: pd.DataFrame, split: dict[str, list[int]]) -> pd.DataFrame:
    """Per-class train/val/test counts, canonical class order."""
    class_of = dict(zip(index["idx"].tolist(), index["class"].tolist()))
    rows = []
    for name in EXPECTED_DISTRIBUTION:
        counts = [sum(1 for i in split[s] if class_of[i] == name) for s in SPLIT_NAMES]
        rows.append({"class": name, "train": counts[0], "val": counts[1], "test": counts[2]})
    return pd.DataFrame(rows)


def assert_distribution(index: pd.DataFrame, split: dict[str, list[int]]) -> None:
    """Assert the reconstruction reproduces the legacy table. Fails with a diff."""
    table = distribution_table(index, split)
    problems = []
    lines = [
        f"  {'class':<45s} {'train':>11s} {'val':>9s} {'test':>9s}",
        f"  {'':<45s} {'got/exp':>11s} {'got/exp':>9s} {'got/exp':>9s}",
    ]
    for record in table.to_dict("records"):
        name = record["class"]
        exp = EXPECTED_DISTRIBUTION[name]
        got = (record["train"], record["val"], record["test"])
        bad = got != exp
        if bad:
            problems.append(name)
        lines.append(
            f"  {name:<45s} {f'{got[0]}/{exp[0]}':>11s} "
            f"{f'{got[1]}/{exp[1]}':>9s} {f'{got[2]}/{exp[2]}':>9s}"
            + ("   <-- MISMATCH" if bad else "")
        )
    totals = (len(split["train"]), len(split["val"]), len(split["test"]))
    if totals != EXPECTED_TOTALS:
        problems.append("TOTAL")
    lines.append(
        f"  {'TOTAL':<45s} {f'{totals[0]}/{EXPECTED_TOTALS[0]}':>11s} "
        f"{f'{totals[1]}/{EXPECTED_TOTALS[1]}':>9s} {f'{totals[2]}/{EXPECTED_TOTALS[2]}':>9s}"
        + ("   <-- MISMATCH" if totals != EXPECTED_TOTALS else "")
    )

    overlap = (
        set(split["train"]) & set(split["val"])
        | set(split["train"]) & set(split["test"])
        | set(split["val"]) & set(split["test"])
    )
    if overlap:
        problems.append(f"overlapping idx between splits: {sorted(overlap)[:10]}")

    if problems:
        raise ValueError(
            "Legacy development split reconstruction does NOT reproduce the expected "
            "distribution.\n" + "\n".join(lines) + f"\n  offending: {problems}"
        )


def compare_routes(a: dict[str, list[int]], b: dict[str, list[int]]) -> dict[str, Any]:
    """How much two recoveries agree, image by image. Reported, never asserted."""
    where_a = {i: s for s, idxs in a.items() for i in idxs}
    where_b = {i: s for s, idxs in b.items() for i in idxs}
    common = set(where_a) & set(where_b)
    agree = sum(1 for i in common if where_a[i] == where_b[i])
    return {
        "n_compared": len(common),
        "n_same_split": agree,
        "fraction_same_split": round(agree / len(common), 4) if common else 0.0,
        "per_split_agreement": {
            s: sum(1 for i in common if where_a[i] == s and where_b[i] == s) for s in SPLIT_NAMES
        },
    }


# --------------------------------------------------------------------------
# recovery driver + persistence
# --------------------------------------------------------------------------


def recover_dev_split(
    index: pd.DataFrame, cfg: dict[str, Any] | None = None, *, verbose: bool = True
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Try Route A, then Route B. Route C (stop and report) is a raise."""
    cfg = cfg or load_data_config()
    attempts: list[dict[str, Any]] = []

    split_a, report_a = recover_route_a(index, cfg, verbose=verbose)
    attempts.append(report_a)

    if split_a is not None:
        try:
            assert_distribution(index, split_a)
        except ValueError as exc:
            report_a["ok"] = False
            report_a["distribution_error"] = str(exc)
            if verbose:
                print("[route A] matched files but the distribution assert failed")
            split_a = None

    split_b, report_b = recover_route_b(index, cfg, verbose=verbose)
    b_ok = True
    try:
        assert_distribution(index, split_b)
    except ValueError as exc:
        b_ok = False
        report_b["ok"] = False
        report_b["distribution_error"] = str(exc)
    attempts.append(report_b)

    if split_a is not None:
        chosen, route = split_a, "A"
        report_a["agreement_with_route_b"] = compare_routes(split_a, split_b) if b_ok else None
    elif b_ok:
        chosen, route = split_b, "B"
    else:
        raise RuntimeError(
            "ROUTE C: the legacy development split could not be recovered.\n"
            "  Route A: " + str(report_a.get("reason") or report_a.get("distribution_error")) + "\n"
            "  Route B: " + str(report_b.get("distribution_error")) + "\n"
            "  Not substituting a fresh split. Stopping, as instructed."
        )

    summary = {
        "route_used": route,
        "attempts": attempts,
        "corpus": "raw_695_with_original_labels",
        "purpose": "hyperparameter selection only (scripts 01, 02); never reporting",
    }
    return chosen, summary


def write_dev_split(
    split: dict[str, list[int]], summary: dict[str, Any], path: Path | None = None
) -> Path:
    path = Path(path) if path is not None else artifacts_dir() / "dev_split.json"
    payload = {
        "corpus": "raw_695",
        "note": (
            "Indexes into the 695-row artifacts/image_index.csv, INCLUDING conflict-group "
            "members. Development split: hyperparameter selection only. Reporting uses "
            "artifacts/folds.json, which is built on the clean 668."
        ),
        "recovery": summary,
        "counts": {name: len(split[name]) for name in SPLIT_NAMES},
        "train_idx": split["train"],
        "val_idx": split["val"],
        "test_idx": split["test"],
    }
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_dev_split(path: Path | None = None) -> dict[str, list[int]]:
    path = Path(path) if path is not None else artifacts_dir() / "dev_split.json"
    require_file(path, produced_by="python scripts/00_build_folds.py")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    missing = [k for k in ("train_idx", "val_idx", "test_idx") if k not in payload]
    if missing:
        raise MissingInputError(f"{path} is missing key(s) {missing}")
    return {
        "train": list(payload["train_idx"]),
        "val": list(payload["val_idx"]),
        "test": list(payload["test_idx"]),
    }
