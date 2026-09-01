"""Dataset discovery, class-name normalisation and the stable image index.

The dataset is RAW and labelled only by directory:

    <DATA_ROOT>/<class_dir>/*.png|*.jpg

There is no pre-existing train/val/test structure; all partitioning is done in
code, and every later artefact refers to an image by its integer `idx` in
artifacts/image_index.csv -- never by path.

CLASS-NAME NORMALISATION. A leading `^\\d+_` is stripped from every directory
name before use. The Kaggle copy is already clean, so this is expected to be a
no-op there; it is a guard, because the old data in ../FINAL-pipeline held 14
directories where four classes were duplicated under a prefix
("10_severe_vibration" beside "severe_vibration"). Left unnormalised that
collision silently deflated every metric in the old pipeline. After
normalisation there must be EXACTLY 10 distinct classes, and we fail with the
offending names otherwise.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import MissingInputError, artifacts_dir, load_data_config, require_file, resolve_data_root

IMAGE_INDEX_COLUMNS = ["idx", "relpath", "class", "sha1"]


# --------------------------------------------------------------------------
# class names
# --------------------------------------------------------------------------


def normalize_class_name(name: str, pattern: str = r"^\d+_") -> str:
    """Strip a leading numeric prefix: '10_severe_vibration' -> 'severe_vibration'."""
    return re.sub(pattern, "", name)


def discover_class_dirs(
    data_root: Path, cfg: dict[str, Any] | None = None, *, verbose: bool = True
) -> dict[str, list[Path]]:
    """Map normalised class name -> the on-disk directories that feed it.

    A normalised name maps to more than one directory exactly when the source
    carried both a prefixed and an unprefixed copy. Raises if the normalised set
    is not the 10 canonical classes, naming what is extra or missing.
    """
    cfg = cfg or load_data_config()
    pattern = cfg.get("class_prefix_pattern", r"^\d+_")
    canonical = list(cfg["classes"])

    raw_dirs = sorted(d for d in Path(data_root).iterdir() if d.is_dir())
    if not raw_dirs:
        raise MissingInputError(f"No class directories found under DATA_ROOT: {data_root}")

    grouped: dict[str, list[Path]] = {}
    renamed: list[tuple[str, str]] = []
    for directory in raw_dirs:
        normalised = normalize_class_name(directory.name, pattern)
        if normalised != directory.name:
            renamed.append((directory.name, normalised))
        grouped.setdefault(normalised, []).append(directory)

    if verbose:
        if renamed:
            print(f"[classes] normalised {len(renamed)} directory name(s) with pattern {pattern!r}:")
            for old, new in renamed:
                print(f"            {old!r} -> {new!r}")
        else:
            print(f"[classes] no directory name required normalisation (pattern {pattern!r})")
        merged = {k: v for k, v in grouped.items() if len(v) > 1}
        if merged:
            print(f"[classes] {len(merged)} class(es) merged from multiple directories:")
            for name, dirs in sorted(merged.items()):
                print(f"            {name}: {[d.name for d in dirs]}")

    found = set(grouped)
    expected = set(canonical)
    if found != expected:
        extra = sorted(found - expected)
        missing = sorted(expected - found)
        raise ValueError(
            "Class normalisation did not yield the 10 canonical classes.\n"
            f"  DATA_ROOT            : {data_root}\n"
            f"  raw directories ({len(raw_dirs):2d}) : {[d.name for d in raw_dirs]}\n"
            f"  after normalisation ({len(found):2d}): {sorted(found)}\n"
            f"  unexpected ({len(extra)})        : {extra}\n"
            f"  missing ({len(missing)})           : {missing}\n"
            "  Fix the directory names or configs/data.yaml:classes. Do not proceed."
        )
    return grouped


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------


def sha1_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-1 of the file's bytes.

    This is how stage (c) matches images against the old split despite the
    directory renaming between ../FINAL-pipeline and the Kaggle copy.
    """
    digest = hashlib.sha1()  # noqa: S324 - content addressing, not security
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_class_files(dirs: list[Path], extensions: set[str]) -> list[Path]:
    """Every image under the given directories, sorted by filename.

    Sorting is on the filename alone (not the full path), so the order does not
    depend on which directory a file came from when a class was merged from a
    prefixed and an unprefixed copy.
    """
    files = [
        path
        for directory in dirs
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(files, key=lambda p: (p.name, str(p)))


def build_image_index(
    data_root: Path | None = None,
    cfg: dict[str, Any] | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the stable image index.

    Files are sorted by filename within each class, then classes are concatenated
    in canonical order. `idx` is the row position in that concatenation and is the
    only identifier any later artefact may use.

    Columns: idx, relpath, class, sha1.
    """
    cfg = cfg or load_data_config()
    data_root = Path(data_root) if data_root is not None else resolve_data_root(cfg, verbose=verbose)
    extensions = {e.lower() for e in cfg["image_extensions"]}
    canonical = list(cfg["classes"])

    grouped = discover_class_dirs(data_root, cfg, verbose=verbose)

    rows = []
    for class_name in canonical:  # canonical order, not sorted() and not os order
        for path in list_class_files(grouped[class_name], extensions):
            rows.append(
                {
                    "idx": len(rows),
                    "relpath": path.relative_to(data_root).as_posix(),
                    "class": class_name,
                    "sha1": sha1_of_file(path),
                }
            )

    if not rows:
        raise MissingInputError(
            f"No images with extensions {sorted(extensions)} found under {data_root}"
        )

    index = pd.DataFrame(rows, columns=IMAGE_INDEX_COLUMNS)

    duplicates = index[index.duplicated("sha1", keep=False)]
    if not duplicates.empty:
        groups = duplicates.groupby("sha1")["relpath"].apply(list)
        print(f"[index] WARNING: {len(groups)} sha1 collision(s) -- byte-identical images:")
        for sha, paths in list(groups.items())[:10]:
            print(f"          {sha[:12]}: {paths}")
        print("        Filename fallback is used for these when matching the legacy split.")

    return index


def assert_counts(index: pd.DataFrame, cfg: dict[str, Any] | None = None) -> None:
    """Assert per-class totals and the grand total. Fails with a per-class diff."""
    cfg = cfg or load_data_config()
    expected: dict[str, int] = dict(cfg["expected_counts"])
    canonical = list(cfg["classes"])
    actual = index["class"].value_counts().to_dict()

    diffs = [
        (name, expected[name], int(actual.get(name, 0)))
        for name in canonical
        if int(actual.get(name, 0)) != expected[name]
    ]
    total_expected = int(cfg["expected_total"])
    total_actual = int(len(index))

    if diffs or total_actual != total_expected:
        lines = [
            "Image counts do not match the data contract in configs/data.yaml.",
            f"  {'class':<45s} {'expected':>9s} {'actual':>7s} {'diff':>6s}",
        ]
        for name, exp, act in diffs:
            lines.append(f"  {name:<45s} {exp:>9d} {act:>7d} {act - exp:>+6d}")
        lines.append(
            f"  {'TOTAL':<45s} {total_expected:>9d} {total_actual:>7d} "
            f"{total_actual - total_expected:>+6d}"
        )
        raise ValueError("\n".join(lines))


def write_image_index(index: pd.DataFrame, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else artifacts_dir() / "image_index.csv"
    index.to_csv(path, index=False, lineterminator="\n")
    return path


def load_image_index(path: Path | None = None) -> pd.DataFrame:
    """Read artifacts/image_index.csv, failing by name if it is absent."""
    path = Path(path) if path is not None else artifacts_dir() / "image_index.csv"
    require_file(path, produced_by="python scripts/00_build_folds.py")
    index = pd.read_csv(path, dtype={"idx": int, "relpath": str, "class": str, "sha1": str})
    missing_cols = [c for c in IMAGE_INDEX_COLUMNS if c not in index.columns]
    if missing_cols:
        raise ValueError(f"{path} is missing column(s) {missing_cols}")
    if list(index["idx"]) != list(range(len(index))):
        raise ValueError(f"{path}: idx column is not 0..{len(index) - 1} in order")
    return index


def labels_array(index: pd.DataFrame, cfg: dict[str, Any] | None = None):
    """Integer labels in canonical class order, aligned to `idx`."""
    cfg = cfg or load_data_config()
    lookup = {name: i for i, name in enumerate(cfg["classes"])}
    return index["class"].map(lookup).to_numpy()


def class_distribution(index: pd.DataFrame, idxs, cfg: dict[str, Any] | None = None) -> dict[str, int]:
    """Per-class counts for a subset of idx values, in canonical order."""
    cfg = cfg or load_data_config()
    subset = index.loc[index["idx"].isin(list(idxs)), "class"].value_counts().to_dict()
    return {name: int(subset.get(name, 0)) for name in cfg["classes"]}


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------


def letterbox_resize(img, target_size: int = 224):
    """Zero-pad to a square, then resize -- preserves the original aspect ratio.

    Carried over unchanged from the legacy pipeline (code.ipynb cell 5,
    "[Cell 6]"). Dynamometer cards are landscape; resizing straight to a square
    distorts the curve geometry that carries the discriminative signal.

      1. max_side = max(w, h)
      2. new RGB canvas max_side x max_side filled (0, 0, 0)
      3. paste the original centred
      4. resize to target_size x target_size, BILINEAR
    """
    from PIL import Image

    width, height = img.size
    max_side = max(width, height)
    padded = Image.new("RGB", (max_side, max_side), (0, 0, 0))
    padded.paste(img, ((max_side - width) // 2, (max_side - height) // 2))
    return padded.resize((target_size, target_size), Image.BILINEAR)


def load_letterboxed(path: Path, target_size: int = 224):
    """Open an image, convert to RGB, letterbox it. The full preprocessing path."""
    from PIL import Image

    with Image.open(path) as img:
        return letterbox_resize(img.convert("RGB"), target_size)
