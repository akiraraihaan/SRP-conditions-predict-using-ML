#!/usr/bin/env python
"""00 -- build artifacts/image_index.csv, artifacts/dev_split.json, artifacts/folds.json.

Runs every assert in the data contract. Idempotent: re-running rebuilds the image
index and verifies it against the committed copy, and refuses to overwrite a
folds.json that already exists, because folds.json is a frozen input.

    python scripts/00_build_folds.py

Phase 1  image index, class normalisation, conflict groups, count asserts
Phase 2  legacy development split (raw 695) + distribution assert
Phase 2b legacy cross-references (skipped when the legacy directory is absent)
Phase 3  RepeatedStratifiedKFold folds on the clean 668 + per-fold val slices
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srpcard import data as srp_data  # noqa: E402
from srpcard import legacy_audit, legacy_split  # noqa: E402
from srpcard.config import artifacts_dir, load_data_config, resolve_data_root  # noqa: E402


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


# --------------------------------------------------------------------------
# phase 1
# --------------------------------------------------------------------------


def phase_1_image_index(cfg: dict, *, force: bool):
    rule("PHASE 1 -- image index")

    data_root = resolve_data_root(cfg)
    index = srp_data.build_image_index(data_root, cfg)

    print("\n[index] %d images across %d classes" % (len(index), index["class"].nunique()))

    srp_data.assert_counts(index, cfg)
    print("[assert] RAW per-class counts and total match configs/data.yaml  OK")

    srp_data.assert_clean_counts(index, cfg)
    print("[assert] clean-corpus conflict groups, exclusions and counts match  OK")

    clean = srp_data.clean_index(index)
    print(
        "\n  %-16s %-45s %6s %5s %6s"
        % ("idx range (raw)", "class", "before", "drop", "after")
    )
    start = 0
    for name in cfg["classes"]:
        before = int((index["class"] == name).sum())
        after = int((clean["class"] == name).sum())
        span = "%d-%d" % (start, start + before - 1)
        print("  %-16s %-45s %6d %5d %6d" % (span, name, before, before - after, after))
        start += before
    n_excluded = int(index["excluded"].astype(bool).sum())
    print("  %-16s %-45s %6d %5d %6d" % ("", "TOTAL", len(index), n_excluded, len(clean)))

    counts = clean["class"].value_counts()
    print(
        "\n[clean] imbalance ratio after exclusion: %d/%d = %.2f:1"
        % (counts.max(), counts.min(), counts.max() / counts.min())
    )

    excluded_out = artifacts_dir(cfg) / "excluded_images.csv"
    excluded = srp_data.excluded_table(index, data_root)
    excluded.to_csv(excluded_out, index=False, lineterminator="\n")
    print("[clean] wrote %s  (%d rows)" % (excluded_out, len(excluded)))

    out = artifacts_dir(cfg) / "image_index.csv"
    if out.exists() and not force:
        previous = srp_data.load_image_index(out)
        same = (
            len(previous) == len(index)
            and previous["relpath"].tolist() == index["relpath"].tolist()
            and previous["sha1"].tolist() == index["sha1"].tolist()
        )
        if same:
            print("\n[index] %s already present and identical -- left untouched" % out.name)
            return index
        print(
            "\n[index] WARNING: %s differs from what this DATA_ROOT produces.\n"
            "        image_index.csv is a COMMITTED, frozen artefact -- every other\n"
            "        artefact references images by its idx. Overwriting it invalidates\n"
            "        artifacts/folds.json, dev_split.json and registry.jsonl.\n"
            "        Re-run with --force only if you intend that." % out
        )
        raise SystemExit(1)

    srp_data.write_image_index(index, out)
    print("\n[index] wrote %s" % out)
    return index


# --------------------------------------------------------------------------
# phase 2
# --------------------------------------------------------------------------


def phase_2_dev_split(cfg: dict, index) -> dict:
    rule("PHASE 2 -- legacy development split (raw 695, original labels)")

    split, summary = legacy_split.recover_dev_split(index, cfg)
    print("\n[recovery] route used: %s" % summary["route_used"])

    legacy_split.assert_distribution(index, split)
    table = legacy_split.distribution_table(index, split)
    print("\n[assert] reconstruction reproduces the legacy distribution exactly  OK\n")
    print("  %-45s %6s %5s %5s" % ("class", "train", "val", "test"))
    for record in table.to_dict("records"):
        print(
            "  %-45s %6d %5d %5d"
            % (record["class"], record["train"], record["val"], record["test"])
        )
    print(
        "  %-45s %6d %5d %5d"
        % ("TOTAL", len(split["train"]), len(split["val"]), len(split["test"]))
    )

    for attempt in summary["attempts"]:
        if attempt["route"] == "A" and attempt.get("agreement_with_route_b"):
            agree = attempt["agreement_with_route_b"]
            print(
                "\n[cross-check] Route A and Route B agree on %d/%d images (%.1f%%).\n"
                "              Both satisfy the count table -- the counts cannot tell them\n"
                "              apart, the image identities can. Route A is the real split."
                % (
                    agree["n_same_split"],
                    agree["n_compared"],
                    100 * agree["fraction_same_split"],
                )
            )

    out = legacy_split.write_dev_split(split, summary, artifacts_dir(cfg) / "dev_split.json")
    print("\n[dev split] wrote %s" % out)
    return split


# --------------------------------------------------------------------------
# phase 2b
# --------------------------------------------------------------------------


def phase_2b_legacy_crossrefs(cfg: dict, index, dev_split: dict) -> None:
    """Cross-references that decide whether parts of the manuscript are false.

    Never fatal: skipped with a message when the read-only legacy directory is
    absent, as it is on Kaggle.
    """
    rule("PHASE 2b -- legacy cross-references")
    out: dict = {}

    ref1 = legacy_audit.crossref_1_split_contamination(index, dev_split)
    out["crossref_1_split_contamination"] = ref1
    print("[1] excluded images inside the legacy development split")
    for split in ("train", "val", "test"):
        block = ref1["by_split"][split]
        print(
            "      %-5s n=%3d  excluded=%2d  (%.1f%%)"
            % (split, block["n_partition"], block["n_excluded"], 100 * block["fraction"])
        )
        for image in block["images"]:
            print("              idx %3d  %s" % (image["idx"], image["class"]))
    print(
        "    %d conflict group(s) straddle a partition boundary (leakage);\n"
        "    %d sit entirely inside train (label noise only)"
        % (ref1["n_leaking_groups"], ref1["n_groups_confined_to_train"])
    )

    cache = artifacts_dir(cfg) / "legacy_test_predictions.csv"
    preds = legacy_audit.legacy_test_predictions(index, cfg, cache=cache)
    if preds is None:
        out["crossref_2_confusion_errors"] = {"available": False}
        print("\n[2] unavailable -- legacy weights or ultralytics not present")
    else:
        ref2 = legacy_audit.crossref_2_confusion_errors(preds, index, list(cfg["classes"]))
        out["crossref_2_confusion_errors"] = ref2
        print("\n[2] confusion-matrix errors (%d total)" % ref2["n_errors_total"])
        for cell, block in ref2["manuscript_cells"].items():
            print(
                "      %-48s n=%d  involving excluded: %d"
                % (cell, block["n_errors"], block["n_involving_excluded"])
            )
        print(
            "    -> %d of %d such errors involve an excluded image"
            % (ref2["n_of_those_involving_excluded"], ref2["n_errors_in_those_cells"])
        )
        print(
            "    -> the manuscript's inter-class-similarity explanation %s"
            % ("SURVIVES" if ref2["manuscript_claim_survives"] else "IS CONTRADICTED")
        )
        for err in ref2["all_contaminated_errors"]:
            print(
                "    contaminated error: idx %d  %s -> %s"
                % (err["idx"], err["true"], err["pred"])
            )

    ref3 = legacy_audit.crossref_3_friedman_split(cfg)
    out["crossref_3_friedman_split"] = ref3
    print("\n[3] friedman_within_nano.json")
    if ref3.get("available"):
        print("      effective_split field present : %s" % ref3["effective_split_field_present"])
        print("      effective_split verbatim      : %r" % ref3["effective_split_verbatim"])
        for split, block in ref3["per_split"].items():
            print(
                "      block %-8r chi2=%.4f p=%.4f significant=%s nemenyi=%s"
                % (
                    split,
                    block["chi2"],
                    block["p_value"],
                    block["significant"],
                    block["nemenyi_present"],
                )
            )
        print("      nemenyi CSVs on disk          : %s" % ref3["nemenyi_csv_files_on_disk"])
        print(
            "      friedman_summary.json effective_split: %r"
            % ref3.get("friedman_summary_effective_split_verbatim")
        )
    else:
        print("      unavailable -- %s" % ref3.get("reason"))

    path = artifacts_dir(cfg) / "legacy_contamination.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
    print("\n[crossref] wrote %s" % path)


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite artifacts/image_index.csv even if it differs (invalidates folds)",
    )
    parser.add_argument(
        "--skip-crossrefs",
        action="store_true",
        help="skip phase 2b even when the legacy reference directory is present",
    )
    args = parser.parse_args()

    cfg = load_data_config()
    index = phase_1_image_index(cfg, force=args.force)
    dev_split = phase_2_dev_split(cfg, index)

    if args.skip_crossrefs:
        rule("PHASE 2b -- legacy cross-references")
        print("skipped -- --skip-crossrefs")
    elif legacy_split.legacy_dir(cfg).exists():
        phase_2b_legacy_crossrefs(cfg, index, dev_split)
    else:
        rule("PHASE 2b -- legacy cross-references")
        print("skipped -- legacy reference not present at %s" % legacy_split.legacy_dir(cfg))

    rule("DONE -- phases 1, 2 and 2b")
    print("Next: phase 3 (folds.json, built on the clean 668).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
