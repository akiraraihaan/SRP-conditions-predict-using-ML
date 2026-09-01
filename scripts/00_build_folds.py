#!/usr/bin/env python
"""00 -- build artifacts/image_index.csv, artifacts/dev_split.json, artifacts/folds.json.

Runs every assert in the data contract. Idempotent: re-running rebuilds the
image index and verifies it, and refuses to overwrite a folds.json that already
exists, because folds.json is a frozen input.

    python scripts/00_build_folds.py

Phase 1 (this stage) : image index + class normalisation + count asserts
Phase 2              : legacy development split + distribution assert
Phase 3              : RepeatedStratifiedKFold folds + per-fold val slices
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srpcard import data as srp_data  # noqa: E402
from srpcard.config import artifacts_dir, load_data_config, resolve_data_root  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def phase_1_image_index(cfg: dict, *, force: bool) -> "object":
    rule("PHASE 1 -- image index")

    data_root = resolve_data_root(cfg)
    index = srp_data.build_image_index(data_root, cfg)

    print(f"\n[index] {len(index)} images across {index['class'].nunique()} classes")

    srp_data.assert_counts(index, cfg)
    print("[assert] per-class counts and total match configs/data.yaml  OK")

    print(f"\n  {'idx range':<12s} {'class':<45s} {'n':>4s}")
    start = 0
    for name in cfg["classes"]:
        n = int((index["class"] == name).sum())
        print(f"  {f'{start}-{start + n - 1}':<12s} {name:<45s} {n:>4d}")
        start += n
    print(f"  {'':<12s} {'TOTAL':<45s} {len(index):>4d}")

    out = artifacts_dir(cfg) / "image_index.csv"
    if out.exists() and not force:
        previous = srp_data.load_image_index(out)
        same = (
            len(previous) == len(index)
            and (previous["relpath"].tolist() == index["relpath"].tolist())
            and (previous["sha1"].tolist() == index["sha1"].tolist())
        )
        if same:
            print(f"\n[index] {out.name} already present and identical -- left untouched")
            return index
        print(
            f"\n[index] WARNING: {out} differs from what this DATA_ROOT produces.\n"
            f"        image_index.csv is a COMMITTED, frozen artefact -- every existing\n"
            f"        artefact references images by its idx. Overwriting it invalidates\n"
            f"        artifacts/folds.json, dev_split.json and registry.jsonl.\n"
            f"        Re-run with --force only if you intend that."
        )
        raise SystemExit(1)

    srp_data.write_image_index(index, out)
    print(f"\n[index] wrote {out}")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite artifacts/image_index.csv even if it differs (invalidates folds)",
    )
    args = parser.parse_args()

    cfg = load_data_config()
    phase_1_image_index(cfg, force=args.force)

    rule("DONE -- phase 1")
    print("Next: phase 2 (legacy development split) and phase 3 (folds.json).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
