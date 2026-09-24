#!/usr/bin/env python
"""09 -- NEAR-duplicate audit. Four checks, because one of them lies.

    python scripts/09_duplicate_audit.py --data-root /content/dataset
    python scripts/09_duplicate_audit.py --data-root ... --dry-run
    python scripts/09_duplicate_audit.py --data-root ... --threshold 2

CPU only. No training, no GPU, no registry writes, and it CHANGES NOTHING --
not folds.json, not the registry, not the dataset.

WHAT THIS IS NOT
----------------
Byte-level duplication is already settled and is NOT redone here: 13 sha1
groups covering 27 files, every group carrying conflicting labels, all members
excluded, leaving 668 unique sha1s, and 00_build_folds.py asserts that no sha1
appears on both sides of any fold.

WHAT IT IS, AND WHY THE HASH IS NOT ENOUGH
------------------------------------------
Two screenshots of the same card seconds apart differ by a pixel of noise, so
their sha1s are unrelated and every byte-level check passes. A perceptual hash
sees them. That is the leak this script is for.

But a perceptual hash is a DCT over an 8x8 reduction, and a dynamometer card is
a thin curve on a uniform white background. Almost all of the low-frequency
energy is identical across the entire corpus WHATEVER THE CLASS, so a Hamming
threshold of 5/64 -- sensible for photographs -- largely measures SHAPE FAMILY
here, not identity. Run on this corpus it flags 116 pairs, only ONE of them at
distance 0, and two thirds of the resulting clusters carry more than one label.
A true duplicate cannot have two labels: every conflicting-label duplicate was
removed at the sha1 stage. The largest cluster spans exactly the classes this
paper independently finds morphologically confusable -- the finding reappearing
as an artefact.

So the hash only PROPOSES candidates. Four independent checks decide:

  1. PIXELS. Every flagged pair is letterboxed with our own code and compared
     by SSIM and normalised RMSE. A genuine duplicate is SSIM > 0.99. The count
     above 0.98 -- not the Hamming count -- is what drives any decision.

  2. TIMESTAMPS. The filenames carry capture times. A re-capture of one card is
     seconds to minutes apart; two survey sessions are hours or days apart. The
     within-cluster spread is reported against a NULL: the same statistic for
     random same-class pairs. If flagged pairs are no closer in time than the
     null, they are not duplicates.

  3. WITHIN-CLASS vs BETWEEN-CLASS. A duplicate carries its original's label,
     so real duplication is overwhelmingly within-class. If flagged pairs are
     spread across classes at roughly the base rate, the hash is detecting
     class morphology and the audit is measuring the wrong thing.

  4. THRESHOLD SENSITIVITY. The verdict is re-reported at 0, 1, 2 and 5, so
     whether the conclusion belongs to the data or to an arbitrary number is
     visible rather than buried.

And a contact sheet of the ten largest clusters --
artifacts/near_duplicate_contact_sheet.png -- because if a 37-image cluster is
visibly 37 different cards, one glance settles it and no statistic is needed.

THE REMEDY SECTION ONLY FIRES when the pixel AND timestamp evidence agree that
duplicates are real AND a confirmed cluster straddles a fold. It reports how
many pairs survive all the checks, never how many the hash flagged.

The dev split gets the straddle check too, because the hyperparameters were
selected on it -- a leak there inflates the selection, not the reported score,
which is quieter and just as real.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srpcard.config import (  # noqa: E402
    artifacts_dir,
    library_versions,
    load_data_config,
    resolve_data_root,
)

SCRIPT = "09_duplicate_audit"

# imagehash returns a 64-bit hash at hash_size 8, so the distance runs 0..64.
HASH_SIZE = 8
DEFAULT_THRESHOLD = 5

# Which hash decides. The others corroborate; see the module docstring.
PRIMARY_HASH = "phash"
HASHES = ("phash", "dhash", "average_hash")


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def require_imagehash():
    try:
        import imagehash  # noqa: F401
    except ImportError:
        raise SystemExit(
            "imagehash is not installed.\n"
            "\n"
            "    pip install imagehash\n"
            "\n"
            "  It is in requirements.txt and in the runner notebooks' install\n"
            "  cell. This script refuses to fall back to a hand-rolled hash:\n"
            "  a near-duplicate finding is only as trustworthy as the hash\n"
            "  behind it, and an unvalidated one would be worse than none."
        )
    import imagehash

    return imagehash


def load_index(artifacts: Path) -> pd.DataFrame:
    """The clean corpus, exactly as the fold builder sees it."""
    path = artifacts / "image_index.csv"
    if not path.exists():
        raise SystemExit(
            "No %s.\n  Run scripts/00_build_folds.py first: this audit must use "
            "the SAME index\n  the folds were built from, or the indices below "
            "mean nothing." % path
        )
    index = pd.read_csv(path, comment="#")
    for column in ("idx", "relpath", "class", "sha1", "excluded"):
        if column not in index.columns:
            raise SystemExit(
                "%s has no %r column. This is not the index 00_build_folds.py "
                "writes." % (path, column)
            )
    return index


def resolve_images(index: pd.DataFrame, data_root: Path) -> list[tuple[int, Path]]:
    """(idx, absolute path) for every INCLUDED image, or a refusal naming misses."""
    wanted = index[~index["excluded"].astype(bool)]
    found, missing = [], []
    for row in wanted.itertuples():
        path = data_root / row.relpath
        (found if path.exists() else missing).append(
            (row.idx, path) if path.exists() else row.relpath
        )
    if missing:
        raise SystemExit(
            "DATA_ROOT does not contain %d of the %d indexed images.\n"
            "  --data-root : %s\n"
            "  first misses: %s\n"
            "\n"
            "  Either the wrong directory was given, or it is a different copy of\n"
            "  the corpus. An audit over a subset would report 'no near-duplicates'\n"
            "  for pairs it never compared, which is the most dangerous possible\n"
            "  wrong answer here."
            % (len(missing), len(wanted), data_root, ", ".join(missing[:5]))
        )
    return found


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------


def compute_hashes(images: list[tuple[int, Path]], *, quiet: bool = False) -> dict:
    """{idx: {hash_name: ImageHash}} for every image."""
    imagehash = require_imagehash()
    from PIL import Image

    functions = {
        "phash": imagehash.phash,
        "dhash": imagehash.dhash,
        "average_hash": imagehash.average_hash,
    }

    out: dict[int, dict] = {}
    for position, (idx, path) in enumerate(images, 1):
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            out[idx] = {
                name: function(image, hash_size=HASH_SIZE)
                for name, function in functions.items()
            }
        if not quiet and position % 100 == 0:
            print("  hashed %d/%d" % (position, len(images)))
    return out


def pairwise_distances(hashes: dict, name: str) -> dict[tuple[int, int], int]:
    """Hamming distance for every unordered pair, under one hash.

    668 images is 222,778 pairs -- trivial at 64 bits, so this is exhaustive
    rather than bucketed. An approximate neighbour search would trade the one
    guarantee that matters: that no pair went uncompared.
    """
    import itertools

    return {
        (a, b): int(hashes[a][name] - hashes[b][name])
        for a, b in itertools.combinations(sorted(hashes), 2)
    }


def distance_histogram(distances: dict[tuple[int, int], int]) -> dict[int, int]:
    """The full distribution, so the threshold is VISIBLE rather than assumed."""
    return dict(sorted(Counter(distances.values()).items()))


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------


def connected_components(pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find over the flagged pairs.

    Near-duplication is not transitive -- A close to B and B close to C does not
    make A close to C -- but for LEAK purposes it behaves as if it were: if any
    two members of a chain land on opposite sides of a fold, information crosses.
    Clustering by connected component is therefore the conservative choice, and
    the cluster sizes are reported so an implausibly long chain is visible.
    """
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: dict[int, list[int]] = defaultdict(list)
    for node in parent:
        groups[find(node)].append(node)
    return [sorted(members) for members in sorted(groups.values(), key=min)]


# --------------------------------------------------------------------------
# 0. DECODED-PIXEL HASH -- the check that was actually missing
# --------------------------------------------------------------------------
#
# File-level sha1 detects duplicate FILES. It does not detect duplicate IMAGES:
# re-encoding a PNG changes every byte while leaving the decoded pixels
# identical, so the same card saved twice slips through with two different
# sha1s and, if it was filed under two class directories, two different labels.
#
# A perceptual hash does not find these either -- not reliably, and on this
# corpus not usefully: a dynamometer card is a thin curve on uniform white, so
# phash largely measures shape family and floods the result with same-shape,
# different-card pairs.
#
# Hashing the DECODED RGB buffer is the right tool, and it is exact:
#   * zero false positives by construction -- identical bytes after decoding
#     means the images ARE the same image
#   * robust to re-encoding, which is precisely what file sha1 is not
#
# It runs over all 695 INDEXED images, not the clean 668, because a group whose
# members were already excluded is exactly the case worth distinguishing from a
# group nothing has caught.


def decoded_pixel_sha1(path: Path) -> str:
    """sha1 of the raw RGB buffer, before any letterbox or resize.

    Mode and size are folded in so two images that happen to serialise to the
    same byte length at different dimensions cannot collide.
    """
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
        digest = hashlib.sha1()  # noqa: S324 - content identity, not security
        digest.update(("%dx%d RGB" % image.size).encode("ascii"))
        digest.update(image.tobytes())
    return digest.hexdigest()


def pixel_duplicate_groups(index, data_root: Path, *, quiet: bool = False) -> dict:
    """{pixel_sha1: [idx, ...]} for every group of 2 or more, over ALL indexed images."""
    groups: dict[str, list[int]] = defaultdict(list)
    total = len(index)
    for position, (_, row) in enumerate(index.iterrows(), 1):
        path = data_root / row["relpath"]
        if not path.exists():
            continue
        groups[decoded_pixel_sha1(path)].append(int(row["idx"]))
        if not quiet and position % 200 == 0:
            print("  decoded %d/%d" % (position, total))
    return {digest: sorted(members)
            for digest, members in groups.items() if len(members) > 1}


def describe_pixel_group(members: list[int], by_idx, folds, dev) -> dict:
    """Labels, consistency, fold straddle, and whether file sha1 already had it.

    ALREADY-CAUGHT means every member shares one file-level sha1, so the
    existing conflict-group rule saw the group. NEW means the group spans more
    than one file sha1 -- a re-encode -- which is the case file hashing cannot
    reach by construction.
    """
    labels = sorted({by_idx[i]["class"] for i in members})
    file_sha1s = sorted({str(by_idx[i]["sha1"]) for i in members})
    excluded = [i for i in members if bool(by_idx[i]["excluded"])]
    straddle = straddle_report(members, folds)
    block = {
        "n_images": len(members),
        "idx": members,
        "labels": labels,
        "label_consistent": len(labels) == 1,
        "n_file_sha1s": len(file_sha1s),
        "already_caught_by_file_sha1": len(file_sha1s) == 1,
        "n_excluded_members": len(excluded),
        "all_members_excluded": len(excluded) == len(members),
        **straddle,
    }
    if dev:
        block.update(dev_split_straddle(members, dev))
    return block


# --------------------------------------------------------------------------
# 1. TIMESTAMP EVIDENCE -- the decisive check
# --------------------------------------------------------------------------

# 692 of the 695 filenames carry a capture time in one of two shapes. The other
# three are UUIDs and are COUNTED as unparseable rather than guessed at.
TIMESTAMP_PATTERNS = (
    (re.compile(r"Screenshot (\d{4}-\d{2}-\d{2}) (\d{6})"), "%Y-%m-%d %H%M%S"),
    (re.compile(r"IMG_(\d{8})_(\d{6})"), "%Y%m%d %H%M%S"),
)


def parse_timestamp(relpath: str):
    """The capture time in a filename, or None.

    This is the evidence that settles whether a flagged pair is a duplicate. A
    re-capture of the same card is SECONDS to minutes apart; two images from
    different survey sessions are hours or days apart. A perceptual hash cannot
    tell those apart -- the clock can.
    """
    name = Path(relpath).name
    for pattern, fmt in TIMESTAMP_PATTERNS:
        match = pattern.search(name)
        if match:
            try:
                return datetime.strptime(" ".join(match.groups()), fmt)
            except ValueError:
                return None
    return None


def cluster_time_spread(cluster: list[int], times: dict[int, object]) -> dict:
    """How far apart in time a cluster's members were captured."""
    stamps = sorted(t for t in (times.get(i) for i in cluster) if t is not None)
    if len(stamps) < 2:
        return {"n_timed": len(stamps), "spread_s": None, "min_gap_s": None}
    gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
    return {
        "n_timed": len(stamps),
        "spread_s": (stamps[-1] - stamps[0]).total_seconds(),
        "min_gap_s": min(gaps),
    }


def null_time_gaps(index, times: dict[int, object], n_samples: int = 2000,
                   seed: int = 0) -> list[float]:
    """Minimum time gap for RANDOM same-class pairs -- the null.

    Without this the timestamp numbers mean nothing: "these two were captured
    40 minutes apart" is only evidence if random same-class pairs are typically
    much further apart. If flagged pairs are no closer in time than the null,
    they are not duplicates.
    """
    import random

    rng = random.Random(seed)
    by_class: dict[str, list[int]] = {}
    for _, row in index.iterrows():
        if bool(row["excluded"]):
            continue
        idx = int(row["idx"])
        if times.get(idx) is not None:
            by_class.setdefault(row["class"], []).append(idx)

    eligible = [c for c, members in by_class.items() if len(members) > 1]
    if not eligible:
        return []
    gaps = []
    for _ in range(n_samples):
        members = by_class[rng.choice(eligible)]
        a, b = rng.sample(members, 2)
        gaps.append(abs((times[a] - times[b]).total_seconds()))
    return sorted(gaps)


# --------------------------------------------------------------------------
# 2. PIXEL-LEVEL CONFIRMATION
# --------------------------------------------------------------------------

# A genuine duplicate is essentially identical after letterboxing. These are the
# thresholds the VERDICT uses -- not the Hamming count, which only proposes
# candidates.
SSIM_DUPLICATE = 0.98
SSIM_CERTAIN = 0.99


def require_skimage():
    try:
        from skimage.metrics import structural_similarity  # noqa: F401
    except ImportError:
        raise SystemExit(
            "scikit-image is not installed.\n"
            "\n"
            "    pip install scikit-image\n"
            "\n"
            "  It is in requirements.txt. This script refuses to hand-roll\n"
            "  SSIM: the pixel check is what decides whether 165 GPU re-runs\n"
            "  happen, and an unvalidated implementation is not something to\n"
            "  decide that on."
        )
    from skimage.metrics import structural_similarity

    return structural_similarity


def pixel_similarity(path_a: Path, path_b: Path, image_size: int = 224) -> dict:
    """SSIM and normalised RMSE between two images, through OUR letterbox.

    Only flagged pairs are compared, so this stays cheap. Our own letterbox is
    used deliberately: it is the transform the model sees, so "identical after
    letterboxing" is the property that actually matters for a leak.
    """
    structural_similarity = require_skimage()
    from srpcard.data import load_letterboxed

    a = np.asarray(load_letterboxed(path_a, image_size).convert("L"), dtype=float)
    b = np.asarray(load_letterboxed(path_b, image_size).convert("L"), dtype=float)
    ssim = float(structural_similarity(a, b, data_range=255.0))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)) / 255.0)
    return {"ssim": round(ssim, 5), "nrmse": round(rmse, 5)}


# --------------------------------------------------------------------------
# 3. THE NULL: within-class against between-class
# --------------------------------------------------------------------------


def class_contingency(pairs, class_of: dict[int, str], index) -> dict:
    """Are flagged pairs within-class, or spread like the base rate?

    If the hash were detecting DUPLICATION, flagged pairs would be
    overwhelmingly within-class -- a duplicate of an image has that image's
    label. If they are spread across classes at roughly the base rate for
    class pairs, the hash is detecting CLASS MORPHOLOGY and the audit is
    measuring the wrong thing.
    """
    within = sum(1 for a, b in pairs if class_of[a] == class_of[b])
    between = len(pairs) - within

    counts = Counter(
        row["class"] for _, row in index.iterrows() if not bool(row["excluded"])
    )
    total = sum(counts.values())
    all_pairs = total * (total - 1) // 2
    within_pairs = sum(n * (n - 1) // 2 for n in counts.values())
    base_rate = within_pairs / all_pairs if all_pairs else 0.0

    observed = within / len(pairs) if pairs else 0.0
    return {
        "n_flagged": len(pairs),
        "within_class": within,
        "between_class": between,
        "observed_within_rate": round(observed, 4),
        "base_rate_within": round(base_rate, 4),
        "enrichment": round(observed / base_rate, 2) if base_rate else None,
    }


# --------------------------------------------------------------------------
# the decisive check
# --------------------------------------------------------------------------


def straddle_report(cluster: list[int], folds: list[dict]) -> dict:
    """In how many folds this cluster has members on BOTH sides of train/test.

    This is the number that decides whether anything must change. A cluster
    entirely inside the training set, or entirely inside the test set, leaks
    nothing however similar its members are.
    """
    members = set(cluster)
    straddled = []
    for entry in folds:
        train = members & set(entry["train_idx"])
        test = members & set(entry["test_idx"])
        # the validation slice is carved out of train, so it counts as train
        # for leak purposes but is recorded separately for the dev-split case
        if train and test:
            straddled.append(
                {
                    "repeat": entry["repeat"],
                    "fold": entry["fold"],
                    "n_train": len(train),
                    "n_test": len(test),
                }
            )
    return {
        "n_folds_straddled": len(straddled),
        "folds_straddled": straddled,
    }


def dev_split_straddle(cluster: list[int], dev: dict) -> dict:
    """The same question for the split the hyperparameters were chosen on.

    A leak here does not inflate the reported score -- it inflates the SELECTION,
    which is quieter and shows up as a configuration that looks better than it is.
    """
    members = set(cluster)
    parts = {
        name: members & set(dev.get("%s_idx" % name) or [])
        for name in ("train", "val", "test")
    }
    present = [name for name, hit in parts.items() if hit]
    return {
        "dev_parts": present,
        "dev_straddles": len(present) > 1,
        "dev_counts": {name: len(hit) for name, hit in parts.items() if hit},
    }



# --------------------------------------------------------------------------
# threshold sensitivity and the contact sheet
# --------------------------------------------------------------------------

SWEEP_THRESHOLDS = (0, 1, 2, 5)


def threshold_sweep(distances, folds, thresholds=SWEEP_THRESHOLDS) -> list[dict]:
    """The verdict at several thresholds, so its sensitivity is visible.

    5/64 is an arbitrary number borrowed from photographic near-duplicate work.
    A dynamometer card is a thin curve on a uniform white background, so almost
    all of the low-frequency energy a perceptual hash measures is identical
    across the whole corpus whatever the class -- which makes a threshold tuned
    for photographs far too loose here. Printing the conclusion at 0, 1, 2 and 5
    shows whether it rests on the data or on the number.
    """
    rows = []
    for threshold in thresholds:
        flagged = [pair for pair, d in distances.items() if d <= threshold]
        clusters = connected_components(sorted(flagged))
        straddling = [c for c in clusters if straddle_report(c, folds)["n_folds_straddled"]]
        rows.append({
            "threshold": threshold,
            "n_pairs": len(flagged),
            "n_clusters": len(clusters),
            "n_images": sum(len(c) for c in clusters),
            "largest_cluster": max((len(c) for c in clusters), default=0),
            "n_clusters_straddling": len(straddling),
        })
    return rows


def contact_sheet(clusters, index, data_root: Path, times, out_path: Path,
                  n_clusters: int = 10, per_row: int = 12, thumb: int = 110):
    """The ten largest clusters, one per row, labelled with class and time.

    The cheapest check of all, and the one a human can settle in a glance: if a
    37-image cluster is visibly 37 different cards, no statistic is needed.
    """
    from PIL import Image, ImageDraw

    biggest = sorted(clusters, key=len, reverse=True)[:n_clusters]
    if not biggest:
        return None

    by_idx = {int(row["idx"]): row for _, row in index.iterrows()}
    label_h, pad = 26, 4
    rows = len(biggest)
    cols = min(per_row, max(len(c) for c in biggest))
    width = cols * (thumb + pad) + pad + 200
    height = rows * (thumb + label_h + pad) + pad

    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for row_n, cluster in enumerate(biggest):
        y = pad + row_n * (thumb + label_h + pad)
        classes = sorted({by_idx[i]["class"] for i in cluster})
        draw.text(
            (pad, y + thumb // 2),
            "cluster %d\nn=%d\n%s" % (row_n + 1, len(cluster),
                                      "MIXED" if len(classes) > 1 else classes[0][:18]),
            fill="black",
        )
        for col_n, idx in enumerate(cluster[:cols]):
            row = by_idx[idx]
            x = 200 + pad + col_n * (thumb + pad)
            try:
                with Image.open(data_root / row["relpath"]) as handle:
                    tile = handle.convert("RGB").resize((thumb, thumb))
                sheet.paste(tile, (x, y))
            except Exception:  # noqa: BLE001 - a missing file must not lose the sheet
                draw.rectangle([x, y, x + thumb, y + thumb], outline="red")
            stamp = times.get(idx)
            draw.text(
                (x, y + thumb + 2),
                "%s\n%s" % (row["class"][:16],
                            stamp.strftime("%m-%d %H:%M:%S") if stamp else "no time"),
                fill="black",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


REMEDIES = """
  TWO REMEDIES, AND WHAT EACH COSTS

  1. DROP the near-duplicates and refreeze the folds.
     Keep one member of each cluster, exclude the rest, rebuild
     artifacts/folds.json, and re-run scripts 03, 04 and 05.
       cost  : 75 + 15 + 75 = 165 GPU runs, roughly 6-8 hours on a T4
       gives : a corpus with no near-duplicate leak by construction
       loses : every existing 03/04/05 record becomes a record of a DIFFERENT
               corpus. The registry is append-only, so they stay, but the
               corpus_fingerprint changes and nothing before and after can be
               pooled. The manuscript's numbers all move.

  2. KEEP every image and group it: StratifiedGroupKFold with the cluster as
     the group, so a cluster can never be split across the boundary.
       cost  : the same 165 re-runs -- the partitions change, so every fold is
               a different fold
       gives : the full corpus, and a leak that is impossible by construction
       loses : exact stratification. Group-aware splitting cannot hold the
               class balance as tightly, so per-fold class counts will vary
               more, and the rarest classes are the ones that will feel it.

  Both cost the same re-runs. The difference is whether 668 images or fewer,
  and whether stratification or grouping is the guarantee you would rather
  have. THIS SCRIPT HAS CHANGED NOTHING -- that decision is yours.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None,
                        help="the image corpus (default: SRPCARD_DATA_ROOT / configs)")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help="Hamming distance counted as a near-duplicate "
                             "(default: %d of 64 bits)" % DEFAULT_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the plan and exit, hashing nothing")
    parser.add_argument("--quiet", action="store_true", help="no progress lines")
    args = parser.parse_args()

    rule("09 -- near-duplicate audit (CPU, no training, changes nothing)")

    data_cfg = load_data_config()
    artifacts = artifacts_dir(data_cfg)
    index = load_index(artifacts)
    included = index[~index["excluded"].astype(bool)]

    folds_path = artifacts / "folds.json"
    if not folds_path.exists():
        raise SystemExit("No %s. Run scripts/00_build_folds.py first." % folds_path)
    folds_blob = json.loads(folds_path.read_text(encoding="utf-8"))
    folds = folds_blob["folds"]
    corpus = folds_blob.get("corpus", {})

    dev_path = artifacts / "dev_split.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8")) if dev_path.exists() else None

    print("  index     : %d rows, %d included, %d excluded"
          % (len(index), len(included), len(index) - len(included)))
    print("  folds     : %d" % len(folds))
    print("  corpus    : %s (n=%s)"
          % (corpus.get("sha1_of_sorted_included_sha1s", "?"), corpus.get("n")))
    print("  dev split : %s" % ("present" if dev else "ABSENT -- selection unchecked"))
    print("  threshold : Hamming <= %d of %d bits" % (args.threshold, HASH_SIZE ** 2))
    print("  hashes    : %s (primary: %s)" % (", ".join(HASHES), PRIMARY_HASH))

    if len(included) != corpus.get("n", len(included)):
        raise SystemExit(
            "The index has %d included images but folds.json was built over %d.\n"
            "  These are different corpora; the fold indices would not line up."
            % (len(included), corpus.get("n"))
        )

    if args.dry_run:
        pairs = len(included) * (len(included) - 1) // 2
        print("\n  --dry-run, nothing written:")
        print("    decoded-pixel sha1 (EXACT) over all %d indexed images" % len(index))
        print("    %d perceptual hash(es) over the %d included images"
              % (len(HASHES), len(included)))
        print("    %d pair(s) compared, then SSIM on whatever the hash flags" % pairs)
        return 0

    data_root = Path(args.data_root) if args.data_root else resolve_data_root(data_cfg)
    if not data_root.is_dir():
        raise SystemExit(
            "--data-root is not a directory: %s\n"
            "  This script needs the images themselves; a perceptual hash cannot "
            "be\n  recovered from the index." % data_root
        )
    print("  data root : %s" % data_root)

    by_idx = {int(row["idx"]): row for _, row in index.iterrows()}
    images = resolve_images(index, data_root)
    rule("hashing %d images" % len(images))
    hashes = compute_hashes(images, quiet=args.quiet)

    # ---- distances, per hash ----
    rule("distance distribution")
    distances = {name: pairwise_distances(hashes, name) for name in HASHES}
    histograms = {name: distance_histogram(distances[name]) for name in HASHES}

    print("  distance   %s" % "  ".join("%14s" % h for h in HASHES))
    for distance in range(0, 13):
        cells = "  ".join("%14d" % histograms[h].get(distance, 0) for h in HASHES)
        marker = "  <- threshold" if distance == args.threshold else ""
        print("  %8d   %s%s" % (distance, cells, marker))
    beyond = {h: sum(n for d, n in histograms[h].items() if d > 12) for h in HASHES}
    print("     >12     %s" % "  ".join("%14d" % beyond[h] for h in HASHES))

    # ---- flagged pairs ----
    flagged = {
        name: {pair for pair, d in distances[name].items() if d <= args.threshold}
        for name in HASHES
    }
    primary = flagged[PRIMARY_HASH]
    print("\n  pairs at or below the threshold:")
    for name in HASHES:
        print("      %-14s %d" % (name, len(flagged[name])))
    unanimous = set.intersection(*flagged.values()) if primary else set()
    print("      %-14s %d" % ("all three", len(unanimous)))

    # ---- clusters ----
    clusters = connected_components(sorted(primary))

    rows = []
    summary_clusters = []
    for number, cluster in enumerate(clusters, 1):
        labels = [by_idx[i]["class"] for i in cluster]
        straddle = straddle_report(cluster, folds)
        dev_block = dev_split_straddle(cluster, dev) if dev else {}
        block = {
            "cluster": number,
            "size": len(cluster),
            "idx": cluster,
            "labels": sorted(set(labels)),
            "label_consistent": len(set(labels)) == 1,
            **straddle,
            **dev_block,
        }
        summary_clusters.append(block)

        for member in cluster:
            row = by_idx[member]
            rows.append({
                "cluster": number,
                "cluster_size": len(cluster),
                "idx": member,
                "relpath": row["relpath"],
                "class": row["class"],
                "sha1": row["sha1"],
                "label_consistent": block["label_consistent"],
                "n_folds_straddled": block["n_folds_straddled"],
                "dev_straddles": block.get("dev_straddles"),
                "corpus_fingerprint": corpus.get("sha1_of_sorted_included_sha1s"),
            })

    # ---- 0. DECODED-PIXEL HASH, over all 695 indexed images ----
    #
    # Exact, and the right tool for the job phash was wrong for. File sha1
    # detects duplicate FILES; this detects duplicate IMAGES, which is what a
    # re-encode hides. Run over the FULL index, not the clean 668, so a group
    # the existing rule already excluded is distinguishable from one nothing
    # has caught.
    rule("decoded-pixel hash (exact) -- all %d indexed images" % len(index))
    pixel_groups = pixel_duplicate_groups(index, data_root, quiet=args.quiet)
    pixel_blocks = []
    for number, (digest, members) in enumerate(sorted(pixel_groups.items()), 1):
        block = describe_pixel_group(members, by_idx, folds, dev)
        block["group"] = number
        block["pixel_sha1"] = digest[:16]
        pixel_blocks.append(block)

    already = [b for b in pixel_blocks if b["already_caught_by_file_sha1"]]
    novel = [b for b in pixel_blocks if not b["already_caught_by_file_sha1"]]
    novel_conflicting = [b for b in novel if not b["label_consistent"]]
    novel_straddling = [b for b in novel_conflicting if b["n_folds_straddled"] > 0]

    print("  groups of pixel-identical images : %d" % len(pixel_blocks))
    print("  already caught by file sha1      : %d" % len(already))
    print("  NEW (re-encodes, >1 file sha1)   : %d" % len(novel))
    print("  of those, CONFLICTING labels     : %d" % len(novel_conflicting))
    print("  of those, straddling a fold      : %d" % len(novel_straddling))
    if novel:
        print()
        print("  %-7s %6s %6s %8s %8s  %s"
              % ("group", "n", "sha1s", "folds", "dev", "labels"))
        for block in novel:
            print("  %-7d %6d %6d %8d %8s  %s"
                  % (block["group"], block["n_images"], block["n_file_sha1s"],
                     block["n_folds_straddled"],
                     "yes" if block.get("dev_straddles") else "no",
                     ", ".join(block["labels"])))
            for member in block["idx"]:
                print("            idx %-5d %-22s %s"
                      % (member, by_idx[member]["class"], by_idx[member]["relpath"]))

    pixel_rows_out = []
    for block in pixel_blocks:
        for member in block["idx"]:
            pixel_rows_out.append({
                "group": block["group"],
                "pixel_sha1": block["pixel_sha1"],
                "idx": member,
                "relpath": by_idx[member]["relpath"],
                "class": by_idx[member]["class"],
                "file_sha1": by_idx[member]["sha1"],
                "excluded": bool(by_idx[member]["excluded"]),
                "group_size": block["n_images"],
                "label_consistent": block["label_consistent"],
                "n_file_sha1s": block["n_file_sha1s"],
                "already_caught_by_file_sha1": block["already_caught_by_file_sha1"],
                "n_folds_straddled": block["n_folds_straddled"],
                "dev_straddles": block.get("dev_straddles"),
                "corpus_fingerprint": corpus.get("sha1_of_sorted_included_sha1s"),
            })
    pd.DataFrame(pixel_rows_out).to_csv(
        artifacts / "pixel_duplicates.csv", index=False, lineterminator="\n"
    )
    print("\n[artifacts] wrote pixel_duplicates.csv (%d row(s))" % len(pixel_rows_out))

    # ---- the three independent checks, computed before any verdict ----
    #
    # The hash only PROPOSES candidates. On a thin curve over a uniform white
    # background a perceptual hash is largely measuring shape family, so a
    # Hamming threshold borrowed from photographic work flags cards that look
    # alike rather than cards that are the same. Everything below exists to
    # tell those apart before anyone is told to spend 165 GPU re-runs.

    rule("evidence")

    # 1. timestamps
    times = {int(row["idx"]): parse_timestamp(row["relpath"])
             for _, row in index.iterrows()}
    n_untimed = sum(1 for idx in {i for c in clusters for i in c}
                    if times.get(idx) is None)
    flagged_gaps = sorted(
        abs((times[a] - times[b]).total_seconds())
        for a, b in primary
        if times.get(a) is not None and times.get(b) is not None
    )
    null_gaps = null_time_gaps(index, times)
    print("  timestamps parsed : %d of %d filename(s)"
          % (sum(1 for v in times.values() if v is not None), len(times)))

    # 2. pixels -- flagged pairs only, so this stays cheap
    pixel_rows = []
    for position, (a, b) in enumerate(sorted(primary), 1):
        similarity = pixel_similarity(
            data_root / by_idx[a]["relpath"],
            data_root / by_idx[b]["relpath"],
            int(load_data_config().get("image_size", 224) or 224),
        )
        pixel_rows.append({"a": a, "b": b, **similarity})
        if not args.quiet and position % 25 == 0:
            print("  compared %d/%d flagged pair(s)" % (position, len(primary)))
    confirmed = [r for r in pixel_rows if r["ssim"] > SSIM_DUPLICATE]
    confirmed_pairs = {(r["a"], r["b"]) for r in confirmed}

    # 3. within-class against between-class
    class_of = {int(row["idx"]): row["class"] for _, row in index.iterrows()}
    contingency = class_contingency(sorted(primary), class_of, index)

    # 4. threshold sensitivity
    sweep = threshold_sweep(distances[PRIMARY_HASH], folds)

    mixed_share = (
        sum(1 for c in summary_clusters if not c["label_consistent"])
        / len(summary_clusters) if summary_clusters else 0.0
    )

    # attach the pixel evidence to each cluster and row
    ssim_of = {(r["a"], r["b"]): r["ssim"] for r in pixel_rows}
    for block in summary_clusters:
        members = set(block["idx"])
        inside = [(a, b) for a, b in primary if a in members and b in members]
        block["n_pairs"] = len(inside)
        block["confirmed_pairs"] = sum(1 for pair in inside if pair in confirmed_pairs)
        scores = [ssim_of[pair] for pair in inside if pair in ssim_of]
        block["max_ssim"] = max(scores) if scores else None
        spread = cluster_time_spread(block["idx"], times)
        block.update(spread)
    confirmed_by_cluster = {b["cluster"]: b["confirmed_pairs"] for b in summary_clusters}
    max_ssim_by_cluster = {b["cluster"]: b["max_ssim"] for b in summary_clusters}
    for row in rows:
        row["cluster_confirmed_pairs"] = confirmed_by_cluster.get(row["cluster"], 0)
        row["cluster_max_ssim"] = max_ssim_by_cluster.get(row["cluster"])
        row["timestamp"] = times.get(row["idx"])

    # 5. the contact sheet -- the cheapest check of all
    sheet_path = artifacts / "near_duplicate_contact_sheet.png"
    sheet_name = sheet_path.name
    try:
        contact_sheet(clusters, index, data_root, times, sheet_path)
        print("  contact sheet     : %s" % sheet_name)
    except Exception as exc:  # noqa: BLE001 - a missing sheet must not lose the audit
        print("  contact sheet     : FAILED (%s: %s)" % (type(exc).__name__, exc))

    # ---- the verdict ----
    rule("VERDICT")
    straddling = [c for c in summary_clusters if c["n_folds_straddled"] > 0]
    dev_straddling = [c for c in summary_clusters if c.get("dev_straddles")]

    print("  CANDIDATE clusters (hash)    : %d" % len(clusters))
    print("  images involved              : %d of %d"
          % (sum(c["size"] for c in summary_clusters), len(included)))
    print("  clusters with MIXED labels   : %d of %d"
          % (sum(1 for c in summary_clusters if not c["label_consistent"]),
             len(summary_clusters)))
    print("  pairs confirmed by pixels    : %d of %d  (SSIM > %.2f)"
          % (len(confirmed), len(primary), SSIM_DUPLICATE))
    print()

    if mixed_share > 0.5:
        print("  WARNING -- %.0f %% of clusters carry MORE THAN ONE LABEL." % (100 * mixed_share))
        print("  A true duplicate cannot have two labels: every conflicting-label")
        print("  duplicate was already removed at the sha1 stage, all 27 of them.")
        print("  Mixed-label clusters mean the hash is grouping cards that LOOK")
        print("  alike, not cards that ARE the same. On line art over a uniform")
        print("  background almost all of the low-frequency energy a perceptual")
        print("  hash measures is shared corpus-wide, so this is the expected")
        print("  failure mode rather than a surprising one.")
        print()

    if not clusters:
        print("  NO near-duplicate cluster was found at Hamming <= %d." % args.threshold)
    elif not straddling:
        print("  NO NEAR-DUPLICATE CLUSTER STRADDLES ANY FOLD BOUNDARY.")
        print("  %d cluster(s) exist, but every one of them sits entirely on one\n"
              "  side of the train/test split in all %d folds, so none of them\n"
              "  leaks anything into a reported score." % (len(clusters), len(folds)))
    else:
        print("  %d CLUSTER(S) STRADDLE A FOLD BOUNDARY, affecting %d image(s)."
              % (len(straddling), sum(c["size"] for c in straddling)))
        print("  Worst cluster straddles %d of %d folds."
              % (max(c["n_folds_straddled"] for c in straddling), len(folds)))
        print()
        print("  %-9s %5s %7s  %-11s %s" % ("cluster", "size", "folds", "labels ok", "classes"))
        for block in sorted(straddling, key=lambda c: -c["n_folds_straddled"])[:20]:
            print("  %-9d %5d %7d  %-11s %s"
                  % (block["cluster"], block["size"], block["n_folds_straddled"],
                     "yes" if block["label_consistent"] else "NO",
                     ", ".join(block["labels"])))

    # ---- the three independent checks ----
    rule("IS THIS DUPLICATION, OR CLASS MORPHOLOGY?")

    print("  1. PIXELS -- the check that should drive any decision")
    if pixel_rows:
        ssims = sorted(r["ssim"] for r in pixel_rows)
        print("     SSIM over %d flagged pair(s):" % len(pixel_rows))
        print("       min %.4f   p25 %.4f   median %.4f   p75 %.4f   max %.4f"
              % (ssims[0], ssims[len(ssims)//4], ssims[len(ssims)//2],
                 ssims[3*len(ssims)//4], ssims[-1]))
        print("       above %.2f : %d        above %.2f : %d"
              % (SSIM_DUPLICATE, len(confirmed), SSIM_CERTAIN,
                 sum(1 for r in pixel_rows if r["ssim"] > SSIM_CERTAIN)))
        print("       A genuine duplicate is SSIM > 0.99 and RMSE near zero.")
    else:
        print("     no flagged pairs to compare")

    print()
    print("  2. TIMESTAMPS -- a re-capture is seconds apart, a second survey is not")
    if null_gaps and flagged_gaps:
        def pct(values, q):
            return values[min(len(values) - 1, int(q * len(values)))]
        print("     minimum within-pair gap, seconds:")
        print("       %-22s median %10.0f   p10 %10.0f   min %8.0f"
              % ("flagged pairs", pct(flagged_gaps, 0.5), pct(flagged_gaps, 0.10),
                 flagged_gaps[0]))
        print("       %-22s median %10.0f   p10 %10.0f   min %8.0f"
              % ("random same-class (null)", pct(null_gaps, 0.5), pct(null_gaps, 0.10),
                 null_gaps[0]))
        ratio = pct(flagged_gaps, 0.5) / max(pct(null_gaps, 0.5), 1.0)
        print("       flagged/null median ratio: %.2f" % ratio)
        if ratio > 0.5:
            print("       Flagged pairs are NOT meaningfully closer in time than")
            print("       random same-class pairs. They are not duplicates.")
        else:
            print("       Flagged pairs ARE much closer in time -- consistent with")
            print("       re-captures of the same card.")
    else:
        print("     not enough parsed timestamps to compare")
    if n_untimed:
        print("     %d image(s) have no parseable timestamp and are excluded from"
              % n_untimed)
        print("     this check rather than guessed at.")

    print()
    print("  3. WITHIN-CLASS vs BETWEEN-CLASS -- the null that decides what is measured")
    print("     %-28s %8d" % ("flagged pairs, same class", contingency["within_class"]))
    print("     %-28s %8d" % ("flagged pairs, different class",
                              contingency["between_class"]))
    print("     %-28s %8.4f" % ("observed within-class rate",
                                contingency["observed_within_rate"]))
    print("     %-28s %8.4f" % ("base rate for class pairs",
                                contingency["base_rate_within"]))
    print("     %-28s %8s" % ("enrichment", contingency["enrichment"]))
    if contingency["enrichment"] is not None and contingency["enrichment"] < 2.0:
        print("     Barely enriched over the base rate. If the hash were detecting")
        print("     DUPLICATION, flagged pairs would be overwhelmingly within-class,")
        print("     because a duplicate of an image carries that image's label.")

    print()
    print("  4. THRESHOLD SENSITIVITY -- is the conclusion the data's, or the number's?")
    print("     %-10s %8s %10s %9s %12s %14s"
          % ("threshold", "pairs", "clusters", "images", "largest", "straddling"))
    for row in sweep:
        print("     %-10d %8d %10d %9d %12d %14d"
              % (row["threshold"], row["n_pairs"], row["n_clusters"],
                 row["n_images"], row["largest_cluster"], row["n_clusters_straddling"]))

    if dev:
        print()
        if dev_straddling:
            print("  DEV SPLIT: %d cluster(s) straddle it. Hyperparameters were\n"
                  "  selected on that split, so the SELECTION is inflated, which is\n"
                  "  quieter than an inflated score and just as real."
                  % len(dev_straddling))
        else:
            print("  DEV SPLIT: clean -- no cluster straddles it.")

    # ---- write ----
    frame = pd.DataFrame(rows)
    out_csv = artifacts / "near_duplicates.csv"
    out_json = artifacts / "near_duplicate_summary.json"

    summary = {
        "script": SCRIPT,
        "threshold": args.threshold,
        "hash_size": HASH_SIZE,
        "hash_bits": HASH_SIZE ** 2,
        "hashes": list(HASHES),
        "primary_hash": PRIMARY_HASH,
        "n_images": len(included),
        "n_pairs_compared": len(included) * (len(included) - 1) // 2,
        "distance_histograms": histograms,
        "n_pairs_flagged": {name: len(flagged[name]) for name in HASHES},
        "n_pairs_flagged_by_all_three": len(unanimous),
        "n_clusters": len(clusters),
        "n_clusters_straddling_a_fold": len(straddling),
        "n_clusters_straddling_dev": len(dev_straddling),
        "pixel_duplicate_groups": len(pixel_blocks),
        "pixel_duplicate_groups_already_caught": len(already),
        "pixel_duplicate_groups_new": len(novel),
        "pixel_duplicate_groups_new_conflicting": len(novel_conflicting),
        "pixel_duplicate_groups_new_straddling": len(novel_straddling),
        "pixel_duplicate_detail": pixel_blocks,
        "pixel_hash_note": (
            "sha1 over the decoded RGB buffer before any letterbox or resize. "
            "Exact: zero false positives by construction, and robust to "
            "re-encoding, which file-level sha1 is not. Run over all indexed "
            "images, not the clean subset."
        ),
        "n_pairs_confirmed_by_pixels": len(confirmed),
        "ssim_duplicate_threshold": SSIM_DUPLICATE,
        "ssim_certain_threshold": SSIM_CERTAIN,
        "pixel_pairs": pixel_rows,
        "mixed_label_cluster_share": round(mixed_share, 4),
        "class_contingency": contingency,
        "threshold_sweep": sweep,
        "timestamp_gaps_flagged_s": flagged_gaps[:500],
        "timestamp_gaps_null_s": null_gaps[:500],
        "n_images_without_timestamp": n_untimed,
        "contact_sheet": sheet_name,
        "verdict_rests_on": (
            "pixel SSIM and timestamp evidence, NOT Hamming distance. The hash "
            "proposes candidates; on line art over a uniform background it "
            "largely measures shape family, so a threshold borrowed from "
            "photographic work flags cards that look alike rather than cards "
            "that are the same."
        ),
        "clusters": summary_clusters,
        "corpus_fingerprint": corpus.get("sha1_of_sorted_included_sha1s"),
        "n_folds": len(folds),
        # This artefact is corpus-level, so it carries no `arm` and no
        # `protocol` -- there is no model and no training regime involved. The
        # corpus fingerprint is the identity that matters here, and it is on
        # every row of the CSV as well as in this block.
        "arm": None,
        "protocol": None,
        "scope": "corpus-level: no arm, no protocol, identified by corpus_fingerprint",
        "library_versions": library_versions(),
        "changed_nothing": True,
    }

    if not frame.empty:
        frame.to_csv(out_csv, index=False, lineterminator="\n")
    else:
        pd.DataFrame(columns=[
            "cluster", "cluster_size", "idx", "relpath", "class", "sha1",
            "label_consistent", "n_folds_straddled", "dev_straddles",
            "corpus_fingerprint",
        ]).to_csv(out_csv, index=False, lineterminator="\n")
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n[artifacts] wrote %s (%d row(s))" % (out_csv.name, len(frame)))
    print("[artifacts] wrote %s" % out_json.name)

    # The remedy fires ONLY when the pixel and timestamp evidence agree that
    # duplicates are real. Recommending 165 GPU re-runs on Hamming distance
    # alone -- on line art, at a threshold borrowed from photographic work --
    # is how a measurement of class morphology turns into a refrozen corpus.
    confirmed_straddling = [
        c for c in summary_clusters
        if c["n_folds_straddled"] > 0 and c.get("confirmed_pairs", 0) > 0
    ]
    timestamps_agree = bool(
        null_gaps and flagged_gaps
        and (sorted(flagged_gaps)[len(flagged_gaps) // 2]
             / max(sorted(null_gaps)[len(null_gaps) // 2], 1.0)) <= 0.5
    )

    rule("VERDICT")
    print("  EXACT decoded-pixel duplicate groups, NEW : %d" % len(novel))
    print("    of those with CONFLICTING labels        : %d" % len(novel_conflicting))
    print("    of those straddling a fold              : %d" % len(novel_straddling))
    print()
    print("  hash-flagged pairs                 : %d" % len(primary))
    print("  surviving the pixel check          : %d" % len(confirmed))
    print("  in clusters that straddle a fold   : %d cluster(s)"
          % len(confirmed_straddling))
    print("  timestamps consistent with re-capture: %s"
          % ("yes" if timestamps_agree else "NO"))
    print()

    if novel_conflicting:
        print("  ONE CARD, TWO LABELS -- found by the decoded-pixel hash, not by")
        print("  the perceptual one. This is the SAME defect the 13 conflict groups")
        print("  were excluded for; file-level sha1 missed it because a re-encode")
        print("  changes every byte while leaving the decoded image identical.")
        print()
        print("  DIRECTION OF THE BIAS: because the labels DIFFER, a straddling")
        print("  group teaches one label and tests the other, so it guarantees an")
        print("  error at test time. It DEPRESSES the reported numbers rather than")
        print("  inflating them -- the conservative direction.")
        print()
        print("  NO REMEDY IS RECOMMENDED HERE and nothing has been changed. The")
        print("  count is in artifacts/pixel_duplicates.csv; the decision is yours.")
        print()

    if not confirmed:
        print("  NO PAIR SURVIVES THE PIXEL CHECK. Nothing here is a duplicate.")
        print("  The hash flagged %d pair(s), and not one of them is two copies of" % len(primary))
        print("  the same card. On a thin curve over a uniform background a")
        print("  perceptual hash measures shape family, not identity.")
        print()
        print("  NO REMEDY IS RECOMMENDED. Do not refreeze the folds. Do not")
        print("  re-run anything. The cross-validation stands as published.")
        print()
        print("  Look at %s to confirm this by eye." % sheet_name)
        return 0

    if not confirmed_straddling:
        print("  %d pair(s) survive the pixel check, but no confirmed cluster" % len(confirmed))
        print("  straddles a fold boundary, so nothing leaks into a reported score.")
        print("  NO REMEDY IS RECOMMENDED.")
        return 0

    if not timestamps_agree:
        print("  %d confirmed pair(s) straddle a fold, BUT the timestamp evidence"
              % len(confirmed))
        print("  does not support them being re-captures: flagged pairs are no")
        print("  closer in time than random same-class pairs. Two of three checks")
        print("  disagree, so this is NOT settled and no remedy is recommended yet.")
        print("  Inspect %s before deciding anything." % sheet_name)
        return 1

    rule("STOPPING FOR A DECISION -- all three checks agree")
    print("  %d pair(s) are confirmed duplicates by pixels AND timestamps, and"
          % len(confirmed))
    print("  they straddle fold boundaries in %d cluster(s)." % len(confirmed_straddling))
    print(REMEDIES)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
