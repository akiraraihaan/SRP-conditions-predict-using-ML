#!/usr/bin/env python
"""00 -- build artifacts/image_index.csv, artifacts/dev_split.json, artifacts/folds.json.

Runs every assert in the data contract. Idempotent: re-running rebuilds the image
index and verifies it against the committed copy, and refuses to overwrite a
folds.json that already exists, because folds.json is a frozen input.

    python scripts/00_build_folds.py
    python scripts/00_build_folds.py --preflight-only

Phase 0  preflight: environment, DATA_ROOT, committed artefacts, pretrained
         checkpoints for all five arms. Nothing here trains, and nothing here
         raises -- it reports, and the script exits non-zero if it failed.
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
from srpcard import folds as srp_folds  # noqa: E402
from srpcard import legacy_audit, legacy_split, models  # noqa: E402
from srpcard.config import (  # noqa: E402
    artifacts_dir,
    library_versions,
    load_data_config,
    load_folds_config,
    resolve_data_root,
    set_seed,
)


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


# --------------------------------------------------------------------------
# phase 0 -- preflight
# --------------------------------------------------------------------------

PREFLIGHT_SEED = 0


def _preflight_environment() -> tuple[bool, dict]:
    """GPU, torch/CUDA versions, and whether cudnn.deterministic actually took."""
    print("\n-- environment ------------------------------------------------------")
    versions = library_versions()
    print("  python             : %s" % versions.get("python"))
    print("  platform           : %s" % versions.get("platform"))
    for name in ("torch", "torchvision", "ultralytics", "numpy", "sklearn"):
        print("  %-18s : %s" % (name, versions.get(name)))
    print("  torch.version.cuda : %s" % versions.get("torch_cuda"))
    print("  cuda available     : %s" % versions.get("cuda_available"))
    print("  gpu                : %s" % versions.get("gpu", "-- none visible --"))

    status = set_seed(PREFLIGHT_SEED)
    print("\n  determinism probe (set_seed(%d)):" % PREFLIGHT_SEED)
    for key in (
        "cudnn_deterministic",
        "cudnn_benchmark",
        "use_deterministic_algorithms",
        "cublas_workspace_config",
    ):
        print("    %-30s %s" % (key, status.get(key, "n/a")))

    ok = True
    if versions.get("cuda_available") != "True":
        print(
            "\n  [WARNING] no CUDA device visible. Training falls back to CPU and takes\n"
            "            roughly an order of magnitude longer. On Kaggle:\n"
            "            Settings -> Accelerator -> GPU T4 x2 (or P100)."
        )
    if versions.get("torch") != "not-installed" and status.get("cudnn_deterministic") is not True:
        print("  [FAIL] cudnn.deterministic did NOT take effect")
        ok = False
    return ok, {"versions": versions, "determinism": status}


def _preflight_data_root(cfg: dict) -> tuple[bool, dict]:
    """Resolve DATA_ROOT and count what is actually on disk, per class."""
    print("\n-- DATA_ROOT --------------------------------------------------------")
    try:
        data_root = resolve_data_root(cfg)
    except Exception as exc:  # noqa: BLE001 - reported here, raised properly in phase 1
        print("  [FAIL] %s" % exc)
        return False, {"resolved": None, "error": str(exc)}

    print("  resolved to        : %s" % data_root)
    try:
        grouped = srp_data.discover_class_dirs(data_root, cfg, verbose=True)
    except Exception as exc:  # noqa: BLE001
        print("  [FAIL] %s" % exc)
        return False, {"resolved": str(data_root), "error": str(exc)}

    extensions = {e.lower() for e in cfg["image_extensions"]}
    expected = cfg["expected_counts"]
    found = {
        name: len(srp_data.list_class_files(grouped[name], extensions)) for name in cfg["classes"]
    }

    print("\n  %-45s %8s %9s %7s" % ("class", "found", "expected", "delta"))
    ok = True
    for name in cfg["classes"]:
        delta = found[name] - int(expected[name])
        if delta:
            ok = False
        print(
            "  %-45s %8d %9d %7s"
            % (name, found[name], expected[name], "" if delta == 0 else "%+d" % delta)
        )
    total, want_total = sum(found.values()), int(cfg["expected_total"])
    if total != want_total:
        ok = False
    print(
        "  %-45s %8d %9d %7s"
        % ("TOTAL", total, want_total, "" if total == want_total else "%+d" % (total - want_total))
    )
    if not ok:
        print(
            "\n  [FAIL] the images on disk do not match configs/data.yaml:expected_counts.\n"
            "         Phase 1 raises on this. Check that DATA_ROOT points at the 10\n"
            "         class directories and that the dataset upload is complete."
        )
    return ok, {"resolved": str(data_root), "found": found, "total": total}


def _preflight_artefacts(cfg: dict) -> tuple[bool, dict]:
    """Verify the COMMITTED artefacts against their fingerprints, not just presence.

    This is the check that catches a fresh `git clone` whose committed artefacts
    disagree with one another: a folds.json built from a different image_index.csv,
    or a dev_split.json indexing rows that no longer exist.
    """
    print("\n-- committed artefacts ----------------------------------------------")
    art = artifacts_dir(cfg)
    paths = {
        "image_index.csv": art / "image_index.csv",
        "dev_split.json": art / "dev_split.json",
        "folds.json": art / "folds.json",
    }
    ok = True
    for name, path in paths.items():
        present = path.exists()
        size = ("%9.1f KB" % (path.stat().st_size / 1024)) if present else ("%9s" % "-")
        print("  [%-7s] %-18s %s" % ("ok" if present else "MISSING", name, size))
        ok = ok and present
    if not ok:
        print("  [FAIL] a committed artefact is absent; fingerprints cannot be checked.")
        return False, {"present": False}

    detail: dict = {"present": True}
    try:
        index = srp_data.load_image_index(paths["image_index.csv"])
    except Exception as exc:  # noqa: BLE001
        print("  [FAIL] image_index.csv did not load: %s" % exc)
        return False, detail
    print(
        "\n  image_index.csv    : %d rows, %d excluded, %d classes"
        % (len(index), int(index["excluded"].astype(bool).sum()), index["class"].nunique())
    )

    current = srp_folds.corpus_fingerprint(index, paths["image_index.csv"])
    payload = srp_folds.load_folds(path=paths["folds.json"], verify=False)
    stored = payload.get("corpus") or {}
    print("\n  folds.json corpus fingerprint vs the committed image_index.csv:")
    print(
        "    %-32s %-42s %-42s %s"
        % ("field", "in folds.json", "computed now", "verdict")
    )
    for key in (
        "n",
        "excluded_n",
        "conflict_groups",
        "sha1_of_sorted_included_sha1s",
        "built_from_index",
    ):
        want, got = stored.get(key), current.get(key)
        agree = want == got
        ok = ok and agree
        print(
            "    %-32s %-42s %-42s %s"
            % (key, want, got, "ok" if agree else "MISMATCH")
        )
    print("    %-32s %d" % ("folds recorded", len(payload.get("folds", []))))

    try:
        split = legacy_split.load_dev_split(paths["dev_split.json"])
    except Exception as exc:  # noqa: BLE001
        print("  [FAIL] dev_split.json did not load: %s" % exc)
        return False, detail

    counts = {name: len(split[name]) for name in ("train", "val", "test")}
    all_idx = [int(i) for i in split["train"] + split["val"] + split["test"]]
    n_rows = len(index)
    out_of_range = [i for i in all_idx if not 0 <= i < n_rows]
    duplicated = len(all_idx) - len(set(all_idx))
    print(
        "\n  dev_split.json     : train %d  val %d  test %d  (total %d of %d index rows)"
        % (counts["train"], counts["val"], counts["test"], len(all_idx), n_rows)
    )
    if out_of_range:
        ok = False
        print("    [FAIL] %d index(es) outside 0..%d" % (len(out_of_range), n_rows - 1))
    if duplicated:
        ok = False
        print("    [FAIL] %d index(es) appear in more than one partition" % duplicated)
    if len(all_idx) != n_rows:
        ok = False
        print("    [FAIL] the split does not cover all %d rows of image_index.csv" % n_rows)
    if not out_of_range and not duplicated and len(all_idx) == n_rows:
        print("    partitions disjoint, exhaustive and in range  ok")

    detail.update({"fingerprint_stored": stored, "fingerprint_now": current, "dev": counts})
    return ok, detail


def _preflight_checkpoints(cfg: dict) -> tuple[bool, dict]:
    """Resolve -- and actually load -- the pretrained checkpoint of all five arms."""
    print("\n-- pretrained checkpoints -------------------------------------------")
    print(
        "  Loading each arm's declared checkpoint now, so a missing YOLO26 file\n"
        "  surfaces in minute one rather than after 40 runs. On Kaggle this triggers\n"
        "  the download, which is exactly what is being tested.\n"
    )
    results = models.preflight_pretrained(data_cfg=cfg)
    header = ("arm", "declared architecture", "acquired", "checkpoint resolved", "status")
    print("  %-18s %-22s %-11s %-30s %s" % header)
    ok = True
    for record in results:
        ok = ok and record["status"] == "ok"
        print(
            "  %-18s %-22s %-11s %-30s %s"
            % (
                record["arm"],
                record["architecture"],
                record["acquisition"],
                record["checkpoint_resolved"] or "-",
                record["status"],
            )
        )
    for record in [r for r in results if r["status"] != "ok"]:
        print("\n  [FAIL] %s (%s)" % (record["arm"], record["architecture"]))
        print("         expected : %s" % record["expected"])
        print("         error    : %s" % record["error"])
        if record["declared_fallback"]:
            print(
                "         configs/arms.yaml offers the fallback %s, which is a\n"
                "         DIFFERENT ARCHITECTURE. It is NOT taken automatically: every\n"
                "         script refuses unless --allow-pretrained-fallback is passed,\n"
                "         and flags pretrained_fallback_used in the registry when it is."
                % record["declared_fallback"]
            )
    return ok, {"arms": results}


def phase_0_preflight(cfg: dict, *, skip_checkpoints: bool = False) -> bool:
    """Everything that should fail in minute one rather than in hour six.

    Never raises: each section reports ok / FAIL and the verdict is returned, so
    one broken section still lets the rest of the report print. main() exits
    non-zero when the verdict is False.
    """
    rule("PHASE 0 -- preflight")
    verdicts: dict[str, bool] = {}
    verdicts["environment"], _ = _preflight_environment()
    verdicts["data_root"], _ = _preflight_data_root(cfg)
    verdicts["artefacts"], _ = _preflight_artefacts(cfg)
    if skip_checkpoints:
        print("\n-- pretrained checkpoints -------------------------------------------")
        print("  skipped -- --skip-checkpoint-preflight")
    else:
        verdicts["checkpoints"], _ = _preflight_checkpoints(cfg)

    print("\n-- preflight verdict ------------------------------------------------")
    for name, good in verdicts.items():
        print("  %-14s %s" % (name, "ok" if good else "FAIL"))
    passed = all(verdicts.values())
    if not passed:
        print(
            "\n  PREFLIGHT FAILED. The rest of this script still runs so you get the\n"
            "  full report, but it exits non-zero. Do not start scripts 01-05 until\n"
            "  every line above reads ok."
        )
    return passed


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


def phase_3_folds(cfg: dict, index, *, rebuild: bool) -> None:
    rule("PHASE 3 -- evaluation folds (clean 668)")

    out = artifacts_dir(cfg) / "folds.json"
    index_path = artifacts_dir(cfg) / "image_index.csv"

    if out.exists() and not rebuild:
        payload = srp_folds.load_folds(index, out, index_path)
        print("[folds] %s already exists -- treated as a FROZEN input" % out.name)
        print("[folds] corpus fingerprint verified against the loaded index  OK")
    else:
        if out.exists():
            print("[folds] --rebuild-folds given: regenerating a frozen artefact")
        payload = srp_folds.build_folds(index, cfg, load_folds_config(), index_path)
        srp_folds.write_folds(payload, out)
        print("[folds] wrote %s" % out)

    corpus = payload["corpus"]
    print(
        "[folds] corpus n=%d  excluded=%d  groups=%d"
        % (corpus["n"], corpus["excluded_n"], corpus["conflict_groups"])
    )
    print("[folds] sha1_of_sorted_included_sha1s = %s" % corpus["sha1_of_sorted_included_sha1s"])
    print("[folds] built_from_index              = %s" % corpus["built_from_index"])

    findings = srp_folds.verify_folds(payload, index)
    print(
        "\n[assert] %d folds; every image in exactly %d test partitions  OK"
        % (findings["n_folds"], findings["every_image_in_exactly_n_test_partitions"])
    )
    print("[assert] no sha1 appears on both sides of any fold  OK")
    print("[assert] excluded images absent; partitions disjoint and exhaustive  OK")

    thin = findings["thin_test_classes"]
    if thin:
        print("\n[WARNING] %d fold/class combination(s) have fewer than 5 test images:" % len(thin))
        for row in thin:
            print(
                "            repeat %d fold %d  %-45s n_test=%d"
                % (row["repeat"], row["fold"], row["class"], row["n_test"])
            )
    else:
        print("\n[check] no class falls below 5 test images in any fold")

    clean = srp_data.clean_index(index)
    rarest = clean["class"].value_counts().idxmin()
    class_of = dict(zip(index["idx"].tolist(), index["class"].tolist()))
    per_fold = [
        sum(1 for i in entry["test_idx"] if class_of[i] == rarest) for entry in payload["folds"]
    ]
    print(
        "[check] rarest class '%s' (n=%d): test partitions range %d-%d images"
        % (rarest, int(clean["class"].value_counts().min()), min(per_fold), max(per_fold))
    )

    report = srp_folds.write_folds_report(payload, index, findings, data_cfg=cfg)
    print("\n[folds] wrote %s" % report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite artifacts/image_index.csv even if it differs (invalidates folds)",
    )
    parser.add_argument(
        "--rebuild-folds",
        action="store_true",
        help="regenerate artifacts/folds.json, which is otherwise a frozen input",
    )
    parser.add_argument(
        "--skip-crossrefs",
        action="store_true",
        help="skip phase 2b even when the legacy reference directory is present",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip phase 0 entirely",
    )
    parser.add_argument(
        "--skip-checkpoint-preflight",
        action="store_true",
        help="run phase 0 but do not load the five pretrained checkpoints (no downloads)",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run phase 0 and stop -- the fastest way to validate a fresh Kaggle session",
    )
    args = parser.parse_args()

    cfg = load_data_config()

    preflight_ok = True
    if args.skip_preflight:
        rule("PHASE 0 -- preflight")
        print("skipped -- --skip-preflight")
    else:
        preflight_ok = phase_0_preflight(
            cfg, skip_checkpoints=args.skip_checkpoint_preflight
        )
    if args.preflight_only:
        rule("DONE -- phase 0 only")
        return 0 if preflight_ok else 1

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

    phase_3_folds(cfg, index, rebuild=args.rebuild_folds)

    rule("DONE -- phases 0, 1, 2, 2b and 3")
    print("artifacts/folds.json is frozen from here on. Next: registry, train, evaluate.")
    if not preflight_ok:
        print(
            "\nEXIT 1: phase 0 reported at least one failure. Scroll up to the\n"
            "        'preflight verdict' block. Do not start scripts 01-05 yet."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
