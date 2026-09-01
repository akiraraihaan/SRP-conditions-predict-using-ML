"""Evaluation folds for reporting, built once on the clean 668-image corpus.

RepeatedStratifiedKFold, n_splits=5, n_repeats=3 -> 15 folds. Within each fold a
stratified 10% slice is carved out of the training indices for best-weight
selection during training; it is never used for reporting.

Seeds are pure functions of (repeat, fold) and IDENTICAL across every arm:

    run_seed = seeds.run_base + repeat * 100 + fold
    val_seed = run_seed + seeds.val_slice_offset

That is what makes every comparison paired, and for the class-weight ablation it
is what isolates the weighting effect.

artifacts/folds.json is built ONCE, committed, and never regenerated. It carries
a corpus fingerprint; every consumer asserts that fingerprint against the corpus
it just loaded and refuses to run on a mismatch. Count-level equality is not
enough to verify a partition -- see MIGRATION_NOTES.md section 13.4, where two
different reconstructions of the legacy split produced identical per-class counts
while disagreeing about 225 of 695 images.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    MissingInputError,
    artifacts_dir,
    library_versions,
    load_data_config,
    load_folds_config,
    require_file,
    run_seed_for,
    val_seed_for,
)
from .data import clean_index


# --------------------------------------------------------------------------
# corpus fingerprint
# --------------------------------------------------------------------------


def corpus_fingerprint(index: pd.DataFrame, index_path: Path | None = None) -> dict[str, Any]:
    """Fingerprint the clean corpus this fold file was built from.

    `sha1_of_sorted_included_sha1s` is the SHA-1 of the included images' own
    SHA-1 hex digests, sorted ascending and joined with newlines (no trailing
    newline), UTF-8 encoded. It identifies the exact set of image CONTENTS,
    independent of path, filename or row order.
    """
    included = clean_index(index)
    joined = "\n".join(sorted(included["sha1"].tolist()))
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()  # noqa: S324

    fingerprint: dict[str, Any] = {
        "n": int(len(included)),
        "sha1_of_sorted_included_sha1s": digest,
        "excluded_n": int(index["excluded"].astype(bool).sum()),
        "conflict_groups": int(
            index.loc[index["excluded"].astype(bool), "conflict_group"].nunique()
        ),
    }
    if index_path is not None and Path(index_path).exists():
        fingerprint["built_from_index"] = hashlib.sha1(  # noqa: S324
            Path(index_path).read_bytes()
        ).hexdigest()
    else:
        fingerprint["built_from_index"] = "unknown"
    return fingerprint


def assert_corpus_matches(folds_payload: dict[str, Any], index: pd.DataFrame,
                          index_path: Path | None = None) -> None:
    """Refuse to run when the loaded corpus is not the one the folds were built on.

    Called by every consumer of folds.json.
    """
    stored = folds_payload.get("corpus")
    if not stored:
        raise ValueError("artifacts/folds.json has no corpus fingerprint block. Refusing to run.")
    current = corpus_fingerprint(index, index_path)

    mismatches = [
        (key, stored.get(key), current.get(key))
        for key in ("n", "sha1_of_sorted_included_sha1s", "excluded_n", "conflict_groups")
        if stored.get(key) != current.get(key)
    ]
    if mismatches:
        lines = [
            "Corpus fingerprint mismatch: artifacts/folds.json was built on a DIFFERENT corpus.",
            "  Refusing to run. Every fold index would refer to the wrong images.",
            "  %-34s %-42s %s" % ("field", "in folds.json", "loaded now"),
        ]
        for key, want, got in mismatches:
            lines.append("  %-34s %-42s %s" % (key, want, got))
        if stored.get("built_from_index") != current.get("built_from_index"):
            lines.append(
                "  %-34s %-42s %s"
                % ("built_from_index", stored.get("built_from_index"), current.get("built_from_index"))
            )
        raise ValueError("\n".join(lines))


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


def build_folds(
    index: pd.DataFrame,
    data_cfg: dict[str, Any] | None = None,
    fold_cfg: dict[str, Any] | None = None,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """Build the 15 folds plus their validation slices. Pure; writes nothing."""
    from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

    data_cfg = data_cfg or load_data_config()
    fold_cfg = fold_cfg or load_folds_config()

    included = clean_index(index).reset_index(drop=True)
    idx_values = included["idx"].to_numpy()
    labels = included["class"].to_numpy()

    n_splits = int(fold_cfg["cv"]["n_splits"])
    n_repeats = int(fold_cfg["cv"]["n_repeats"])
    cv_seed = int(fold_cfg["seeds"]["cv_partition"])
    val_fraction = float(fold_cfg["val_slice"]["fraction"])

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=cv_seed
    )

    folds: list[dict[str, Any]] = []
    for position, (train_pos, test_pos) in enumerate(splitter.split(idx_values, labels)):
        repeat = position // n_splits
        fold = position % n_splits
        val_seed = val_seed_for(fold_cfg, repeat, fold)

        pool_idx = idx_values[train_pos]
        pool_labels = labels[train_pos]

        train_pool, val_pool = train_test_split(
            pool_idx,
            test_size=val_fraction,
            stratify=pool_labels,
            random_state=val_seed,
            shuffle=True,
        )
        folds.append(
            {
                "repeat": repeat,
                "fold": fold,
                "run_seed": run_seed_for(fold_cfg, repeat, fold),
                "val_seed": val_seed,
                "train_idx": sorted(int(i) for i in train_pool),
                "val_idx": sorted(int(i) for i in val_pool),
                "test_idx": sorted(int(i) for i in idx_values[test_pos]),
            }
        )

    return {
        "corpus": {
            **corpus_fingerprint(index, index_path),
            "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "tool_versions": library_versions(),
        },
        "protocol": {
            "splitter": "RepeatedStratifiedKFold",
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "cv_partition_seed": cv_seed,
            "val_slice_fraction": val_fraction,
            "run_seed_formula": "run_base + repeat*100 + fold",
            "val_seed_formula": "run_seed + val_slice_offset",
            "run_base": int(fold_cfg["seeds"]["run_base"]),
            "val_slice_offset": int(fold_cfg["seeds"]["val_slice_offset"]),
            "note": (
                "idx values are positions in the 695-row artifacts/image_index.csv. "
                "Only the 668 non-excluded rows appear here. Seeds are identical "
                "across every arm so all comparisons are paired."
            ),
        },
        "folds": folds,
    }


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def verify_folds(payload: dict[str, Any], index: pd.DataFrame) -> dict[str, Any]:
    """Structural checks over the fold file. Returns findings; raises on hard errors."""
    included = clean_index(index)
    all_idx = set(int(i) for i in included["idx"])
    sha_of = dict(zip(index["idx"].tolist(), index["sha1"].tolist()))
    class_of = dict(zip(index["idx"].tolist(), index["class"].tolist()))
    excluded_idx = set(int(i) for i in index.loc[index["excluded"].astype(bool), "idx"])

    n_repeats = int(payload["protocol"]["n_repeats"])
    errors: list[str] = []
    test_membership: dict[int, int] = {i: 0 for i in all_idx}
    sha_leaks: list[str] = []

    for entry in payload["folds"]:
        tag = "repeat %d fold %d" % (entry["repeat"], entry["fold"])
        train = set(entry["train_idx"])
        val = set(entry["val_idx"])
        test = set(entry["test_idx"])

        if train & val or train & test or val & test:
            errors.append("%s: train/val/test overlap" % tag)
        union = train | val | test
        if union != all_idx:
            errors.append(
                "%s: partitions cover %d of %d images" % (tag, len(union), len(all_idx))
            )
        leaked = union & excluded_idx
        if leaked:
            errors.append("%s: excluded images present: %s" % (tag, sorted(leaked)[:5]))

        for i in test:
            test_membership[i] = test_membership.get(i, 0) + 1

        # no sha1 on both sides of the fold
        fit_shas = {sha_of[i] for i in (train | val)}
        test_shas = {sha_of[i] for i in test}
        shared = fit_shas & test_shas
        if shared:
            sha_leaks.append("%s: %d sha1 on both sides: %s" % (tag, len(shared), sorted(shared)[:3]))

    wrong_membership = {i: c for i, c in test_membership.items() if c != n_repeats}
    if wrong_membership:
        errors.append(
            "%d image(s) do not appear in exactly %d test partitions (e.g. %s)"
            % (len(wrong_membership), n_repeats, list(wrong_membership.items())[:5])
        )
    if sha_leaks:
        errors.extend(sha_leaks)

    # thin test classes
    thin: list[dict[str, Any]] = []
    for entry in payload["folds"]:
        counts: dict[str, int] = {}
        for i in entry["test_idx"]:
            counts[class_of[i]] = counts.get(class_of[i], 0) + 1
        for name, n in sorted(counts.items()):
            if n < 5:
                thin.append(
                    {"repeat": entry["repeat"], "fold": entry["fold"], "class": name, "n_test": n}
                )

    thin_val: list[dict[str, Any]] = []
    for entry in payload["folds"]:
        counts: dict[str, int] = {}
        for i in entry["val_idx"]:
            counts[class_of[i]] = counts.get(class_of[i], 0) + 1
        for name in sorted(set(class_of[i] for i in all_idx)):
            n = counts.get(name, 0)
            if n < 3:
                thin_val.append(
                    {"repeat": entry["repeat"], "fold": entry["fold"], "class": name, "n_val": n}
                )

    findings = {
        "n_folds": len(payload["folds"]),
        "coverage_ok": not wrong_membership,
        "every_image_in_exactly_n_test_partitions": n_repeats,
        "sha1_leak_folds": sha_leaks,
        "no_sha1_across_fold_sides": not sha_leaks,
        "thin_test_classes": thin,
        "thin_val_classes": thin_val,
        "val_slice_size": len(payload["folds"][0]["val_idx"]) if payload["folds"] else 0,
        "errors": errors,
    }
    if errors:
        raise ValueError("folds.json failed verification:\n  " + "\n  ".join(errors))
    return findings


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def write_folds_report(
    payload: dict[str, Any],
    index: pd.DataFrame,
    findings: dict[str, Any],
    path: Path | None = None,
    data_cfg: dict[str, Any] | None = None,
) -> Path:
    """Fold-level CONTENT report -- per-class counts per fold, not just totals."""
    data_cfg = data_cfg or load_data_config()
    path = Path(path) if path is not None else artifacts_dir(data_cfg) / "folds_report.md"
    canonical = list(data_cfg["classes"])
    class_of = dict(zip(index["idx"].tolist(), index["class"].tolist()))
    corpus = payload["corpus"]
    protocol = payload["protocol"]

    def counts(idxs: list[int]) -> dict[str, int]:
        out = {name: 0 for name in canonical}
        for i in idxs:
            out[class_of[i]] += 1
        return out

    lines: list[str] = []
    add = lines.append

    add("# Fold report")
    add("")
    add("Generated by `scripts/00_build_folds.py` alongside `artifacts/folds.json`.")
    add("`folds.json` is a frozen input: built once, committed, never regenerated.")
    add("")
    add("## Corpus fingerprint")
    add("")
    add("| field | value |")
    add("| --- | --- |")
    add("| n | %d |" % corpus["n"])
    add("| sha1_of_sorted_included_sha1s | `%s` |" % corpus["sha1_of_sorted_included_sha1s"])
    add("| excluded_n | %d |" % corpus["excluded_n"])
    add("| conflict_groups | %d |" % corpus["conflict_groups"])
    add("| built_from_index | `%s` |" % corpus["built_from_index"])
    add("| built_at | %s |" % corpus["built_at"])
    add("")
    add("Every consumer of `folds.json` asserts this block against the corpus it just")
    add("loaded and refuses to run on a mismatch. Count-level equality is not enough to")
    add("verify a partition: see MIGRATION_NOTES.md section 13.4, where two")
    add("reconstructions of the legacy split produced identical per-class counts while")
    add("disagreeing about 225 of 695 images.")
    add("")
    add("## Protocol")
    add("")
    add("| field | value |")
    add("| --- | --- |")
    for key in (
        "splitter",
        "n_splits",
        "n_repeats",
        "cv_partition_seed",
        "val_slice_fraction",
        "run_seed_formula",
        "val_seed_formula",
    ):
        add("| %s | `%s` |" % (key, protocol[key]))
    add("")
    add("Seeds are pure functions of (repeat, fold) and identical across every arm.")
    add("")

    # ---- checks
    add("## Checks")
    add("")
    n_repeats = protocol["n_repeats"]
    add(
        "- **Coverage**: every one of the %d images appears in exactly %d test partitions "
        "across the %d repeats -- **%s**"
        % (corpus["n"], n_repeats, n_repeats, "PASS" if findings["coverage_ok"] else "FAIL")
    )
    add(
        "- **No sha1 on both sides of any fold** -- **%s**"
        % ("PASS" if findings["no_sha1_across_fold_sides"] else "FAIL")
    )
    add("- **Excluded images absent from every fold** -- PASS")
    add("- **train / val / test disjoint and exhaustive in every fold** -- PASS")
    add("")
    thin = findings["thin_test_classes"]
    if thin:
        add("### Warning: folds with fewer than 5 test images in a class")
        add("")
        add("| repeat | fold | class | n_test |")
        add("| ---: | ---: | --- | ---: |")
        for row in thin:
            add("| %d | %d | %s | %d |" % (row["repeat"], row["fold"], row["class"], row["n_test"]))
        add("")
    else:
        add("### Thin-class warning: none")
        add("")
        add("No class falls below 5 test images in any of the %d folds." % len(payload["folds"]))
        add("")

    thin_val = findings.get("thin_val_classes") or []
    add("### Validation slice")
    add("")
    add(
        "The 10 %% slice is %d images per fold, used only for best-weight selection "
        "during training." % findings.get("val_slice_size", 0)
    )
    add("")
    if thin_val:
        worst: dict[str, int] = {}
        for row in thin_val:
            worst[row["class"]] = min(worst.get(row["class"], 99), row["n_val"])
        add("Classes with fewer than 3 validation images in at least one fold:")
        add("")
        add("| class | min n_val across folds | folds affected |")
        add("| --- | ---: | ---: |")
        for name, n in sorted(worst.items(), key=lambda kv: kv[1]):
            affected = sum(1 for r in thin_val if r["class"] == name)
            add("| %s | %d | %d of %d |" % (name, n, affected, len(payload["folds"])))
        add("")
        add(
            "Best-weight selection on such a slice is noisy for these classes. This does "
            "not affect the reported test metrics, which are computed on the fold's test "
            "partition, but it does add variance to which epoch's weights are kept."
        )
    else:
        add("No class falls below 3 validation images in any fold.")
    add("")

    # ---- per-class test counts across folds
    add("## Test-partition size per class, per fold")
    add("")
    add("| class | " + " | ".join("r%df%d" % (e["repeat"], e["fold"]) for e in payload["folds"]) + " | min | max |")
    add("| --- |" + " ---: |" * (len(payload["folds"]) + 2))
    per_fold_counts = [counts(e["test_idx"]) for e in payload["folds"]]
    for name in canonical:
        row = [c[name] for c in per_fold_counts]
        add(
            "| %s | %s | %d | %d |"
            % (name, " | ".join(str(v) for v in row), min(row), max(row))
        )
    totals = [len(e["test_idx"]) for e in payload["folds"]]
    add("| **TOTAL** | %s | %d | %d |" % (" | ".join(str(v) for v in totals), min(totals), max(totals)))
    add("")

    # ---- per-fold detail
    add("## Per-fold detail")
    add("")
    for entry, test_counts in zip(payload["folds"], per_fold_counts):
        train_counts = counts(entry["train_idx"])
        val_counts = counts(entry["val_idx"])
        add(
            "### repeat %d, fold %d  (run_seed %d, val_seed %d)"
            % (entry["repeat"], entry["fold"], entry["run_seed"], entry["val_seed"])
        )
        add("")
        add("| class | train | val | test |")
        add("| --- | ---: | ---: | ---: |")
        for name in canonical:
            add("| %s | %d | %d | %d |" % (name, train_counts[name], val_counts[name], test_counts[name]))
        add(
            "| **TOTAL** | %d | %d | %d |"
            % (len(entry["train_idx"]), len(entry["val_idx"]), len(entry["test_idx"]))
        )
        add("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def write_folds(payload: dict[str, Any], path: Path | None = None) -> Path:
    path = Path(path) if path is not None else artifacts_dir() / "folds.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_folds(
    index: pd.DataFrame | None = None,
    path: Path | None = None,
    index_path: Path | None = None,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    """Load folds.json and, when an index is given, assert the corpus fingerprint."""
    path = Path(path) if path is not None else artifacts_dir() / "folds.json"
    require_file(path, produced_by="python scripts/00_build_folds.py")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if "folds" not in payload:
        raise MissingInputError("%s has no 'folds' key" % path)
    if verify and index is not None:
        assert_corpus_matches(payload, index, index_path)
    return payload
