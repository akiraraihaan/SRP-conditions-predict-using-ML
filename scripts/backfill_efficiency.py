#!/usr/bin/env python
"""backfill_efficiency -- fill null efficiency fields in existing registry records.

    python scripts/backfill_efficiency.py --dry-run   # report, change nothing
    python scripts/backfill_efficiency.py             # write, after a .bak copy

`params`, `gflops`, `size_mb_fp32` and `size_mb_fp16` are pure functions of
(architecture, num_classes, image_size), so a record missing them can have them
derived rather than re-run. Leaving them null would force aggregate.py and
figures.py to special-case values that are perfectly well defined.

Two rounds of this have been needed:

  params / gflops   script 01's first eight records were written before it
                    profiled the trained module.
  size_mb_*         `size_mb` used to mean the ultralytics checkpoint file size
                    in script 01 and an fp32 state_dict everywhere else -- two
                    different quantities under one name, differing by ~2x. The
                    checkpoint value is preserved as `size_mb_checkpoint_file`,
                    and `size_mb` now aliases the fp16 figure, which is what the
                    framework actually deploys. The payload variants (raw tensor
                    bytes, no container) are filled at the same time.
                    See src/srpcard/efficiency.py.

This is the one place in the repository that REWRITES artifacts/registry.jsonl
rather than appending to it. It is therefore deliberately narrow:

  - it only ever fills a field that is currently null; nothing else is touched,
    and no record is added or removed;
  - `run_id` is not recomputed, and could not change if it were: params and
    gflops are outcomes, not identity (registry.RUN_ID_FIELDS);
  - a timestamped .bak copy is written first;
  - --dry-run prints exactly what would change.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srpcard import registry  # noqa: E402
from srpcard.config import load_arms_config, load_data_config  # noqa: E402
from srpcard.efficiency import profile  # noqa: E402
from srpcard.models import build_model  # noqa: E402

DERIVED_FIELDS = (
    "params",
    "gflops",
    "size_mb_fp32",
    "size_mb_fp16",
    "size_mb_fp16_payload",
    "size_mb_fp32_payload",
)

# size_mb aliases the PRIMARY measurement, which is fp16 (efficiency.py).
PRIMARY_SIZE_FIELD = "size_mb_fp16"

# Records written before size_mb was disambiguated: script 01 stored the .pt file
# size there. Move it to its own name and let size_mb become the fp32 figure.
LEGACY_SIZE_SCRIPTS = ("01_complete_medium_grid",)


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def measure(arm: str, arms_cfg, data_cfg, cache: dict) -> dict | None:
    """The derived efficiency fields for one arm, built once and reused."""
    if arm in cache:
        return cache[arm]
    try:
        bundle = build_model(arm, arms_cfg, data_cfg, with_efficiency=False)
        stats = profile(bundle.module, int(arms_cfg["shared"]["image_size"]), latency=False)
        cache[arm] = {field: stats[field] for field in DERIVED_FIELDS}
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        print("  [%s] could not build or profile: %s" % (arm, exc))
        cache[arm] = None
    return cache[arm]


def needs_work(record: dict, *, refresh: bool = False) -> bool:
    if refresh:
        return True
    if any(record.get(field) is None for field in DERIVED_FIELDS):
        return True
    # a pre-disambiguation record: size_mb still holds the checkpoint file size
    return (
        record.get("script") in LEGACY_SIZE_SCRIPTS
        and record.get("size_mb_checkpoint_file") is None
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "also RECOMPUTE derived fields that are already set. Needed when the "
            "definition of a measurement changes -- the fp16/fp32 sizes were once "
            "measured through a temp filename that altered torch's zip container, "
            "so values written before that fix are stale. Off by default."
        ),
    )
    args = parser.parse_args()

    rule("backfill null params/gflops in the registry")

    path = registry.registry_path()
    records = registry.load_registry(path)
    print("  registry : %s" % path)
    print("  records  : %d" % len(records))
    if not records:
        print("\n  Nothing to do.")
        return 0

    incomplete = [r for r in records if needs_work(r, refresh=args.refresh)]
    if not incomplete:
        print("\n  Every record already carries the derived efficiency fields.")
        return 0

    print("  missing  : %d record(s)" % len(incomplete))

    arms_cfg, data_cfg = load_arms_config(), load_data_config()
    cache: dict = {}

    print(
        "\n  %-16s %-18s %10s %8s %9s %9s %10s"
        % ("run_id", "arm", "params", "gflops", "fp32", "fp16", ".pt file")
    )
    filled = 0
    for record in incomplete:
        arm = record.get("arm")
        stats = measure(arm, arms_cfg, data_cfg, cache) if arm else None
        if stats is None:
            print("  %-16s %-18s   SKIPPED" % (record.get("run_id"), arm))
            continue

        changed = []
        # Move the old checkpoint-file size out of size_mb before size_mb is
        # redefined, so the published figure is preserved rather than replaced.
        if (
            record.get("script") in LEGACY_SIZE_SCRIPTS
            and record.get("size_mb_checkpoint_file") is None
        ):
            record["size_mb_checkpoint_file"] = record.get("size_mb")
            changed.append("size_mb_checkpoint_file")

        for field in DERIVED_FIELDS:
            if record.get(field) is None:
                record[field] = stats[field]
                changed.append(field)
            elif args.refresh and record[field] != stats[field]:
                changed.append("%s %s->%s" % (field, record[field], stats[field]))
                record[field] = stats[field]

        if record.get("size_mb") != stats[PRIMARY_SIZE_FIELD]:
            record["size_mb"] = stats[PRIMARY_SIZE_FIELD]
            changed.append("size_mb->fp16")

        if changed:
            filled += 1
        print(
            "  %-16s %-18s %10s %8.4f %9.3f %9.3f %10s"
            % (
                record.get("run_id"),
                arm,
                stats["params"],
                stats["gflops"],
                stats["size_mb_fp32"],
                stats["size_mb_fp16"],
                record.get("size_mb_checkpoint_file", "-"),
            )
        )
        print("      %s" % ", ".join(changed))

    if args.dry_run:
        print("\n  --dry-run: %d record(s) would be filled. Nothing written." % filled)
        return 0

    if not filled:
        print("\n  Nothing to write.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_suffix(".jsonl.%s.bak" % stamp)
    shutil.copy2(path, backup)
    print("\n  backup   : %s" % backup)

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=False, default=str) + "\n")
    print("  rewrote  : %s  (%d record(s) filled)" % (path, filled))

    reloaded = registry.load_registry(path)
    still_null = [r for r in reloaded if any(r.get(f) is None for f in DERIVED_FIELDS)]
    print(
        "  verify   : %d record(s) re-read, %d still carrying a null"
        % (len(reloaded), len(still_null))
    )
    if len(reloaded) != len(records):
        print("  [FAIL] record count changed -- restore from the backup above")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
