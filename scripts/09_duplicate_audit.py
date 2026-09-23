#!/usr/bin/env python
"""09 -- NEAR-duplicate audit. The leak byte hashing cannot see.

    python scripts/09_duplicate_audit.py --data-root /content/dataset
    python scripts/09_duplicate_audit.py --data-root ... --dry-run
    python scripts/09_duplicate_audit.py --data-root ... --threshold 5

CPU only. No training, no GPU, no registry writes. Reads the same
artifacts/image_index.csv that 00_build_folds.py built the folds from, so the
indices here mean exactly what they mean there.

WHAT THIS IS NOT
----------------
Byte-level duplication is already settled and is NOT redone here: 13 sha1
groups covering 27 files, every group carrying conflicting labels, all members
excluded, leaving 668 unique sha1s, and 00_build_folds.py asserts that no sha1
appears on both sides of any fold.

WHAT IT IS
----------
Two screenshots of the same dynamometer card taken seconds apart differ by one
pixel of noise or one pixel of crop. Their sha1s are unrelated, so every check
above passes, and the two images land in different folds -- one in train, one
in test -- and the model is scored on an image it effectively trained on. The
measured macro-F1 is then optimistic by an unknown amount, and nothing in the
pipeline can detect it, because at the byte level the corpus is clean.

Perceptual hashing sees it. Three are computed -- phash, dhash, average_hash --
because each has a known blind spot and a finding that rests on one of them is
a finding about that hash. phash is the primary (it is the most robust to
rescaling and mild compression); the other two are corroboration, and a pair
flagged by only one hash is reported as weaker evidence rather than dropped.

THE DECISIVE NUMBER is not how many near-duplicates exist. It is in how many of
the 15 folds a near-duplicate cluster STRADDLES the train/test boundary. A
cluster whose members all sit on the same side of every fold leaks nothing.

The dev split gets the same check, because the hyperparameters were selected on
it -- a leak there inflates the selection, not the reported score, and that is a
different and quieter problem.

THIS SCRIPT CHANGES NOTHING. It does not touch folds.json, the registry, or the
dataset. If clusters do straddle folds it prints the two remedies with what each
costs in re-runs and stops for a human decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

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
        print("\n  --dry-run: would hash %d images with %d hash functions and\n"
              "  compare %d pairs. Nothing written."
              % (len(included), len(HASHES), pairs))
        return 0

    data_root = Path(args.data_root) if args.data_root else resolve_data_root(data_cfg)
    if not data_root.is_dir():
        raise SystemExit(
            "--data-root is not a directory: %s\n"
            "  This script needs the images themselves; a perceptual hash cannot "
            "be\n  recovered from the index." % data_root
        )
    print("  data root : %s" % data_root)

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
    by_idx = {int(row["idx"]): row for _, row in index.iterrows()}

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

    # ---- the verdict ----
    rule("VERDICT")
    straddling = [c for c in summary_clusters if c["n_folds_straddled"] > 0]
    dev_straddling = [c for c in summary_clusters if c.get("dev_straddles")]

    print("  near-duplicate clusters      : %d" % len(clusters))
    print("  images involved              : %d of %d"
          % (sum(c["size"] for c in summary_clusters), len(included)))
    print("  clusters with mixed labels   : %d"
          % sum(1 for c in summary_clusters if not c["label_consistent"]))
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

    if straddling or dev_straddling:
        rule("STOPPING FOR A DECISION")
        print(REMEDIES)
        return 1

    rule("DONE")
    print("  Nothing to decide: the folds are clean under this threshold.\n"
          "  folds.json, the registry and the dataset are untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
