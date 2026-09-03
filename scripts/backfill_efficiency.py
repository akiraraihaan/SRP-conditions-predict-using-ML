#!/usr/bin/env python
"""backfill_efficiency -- fill null params/gflops in existing registry records.

    python scripts/backfill_efficiency.py --dry-run   # report, change nothing
    python scripts/backfill_efficiency.py             # write, after a .bak copy

`params` and `gflops` are a pure function of (architecture, num_classes,
image_size), so a record that is missing them can have them derived rather than
being re-run. Script 01's first eight records were written before it profiled the
trained module and carry nulls; leaving them would force aggregate.py and
figures.py to special-case a value that is perfectly well defined.

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

FIELDS = ("params", "gflops")


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def measure(arm: str, arms_cfg, data_cfg, cache: dict) -> dict | None:
    """params/gflops for one arm, built once and reused."""
    if arm in cache:
        return cache[arm]
    try:
        bundle = build_model(arm, arms_cfg, data_cfg, with_efficiency=False)
        stats = profile(bundle.module, int(arms_cfg["shared"]["image_size"]), latency=False)
        cache[arm] = {"params": stats["params"], "gflops": stats["gflops"]}
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        print("  [%s] could not build or profile: %s" % (arm, exc))
        cache[arm] = None
    return cache[arm]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    rule("backfill null params/gflops in the registry")

    path = registry.registry_path()
    records = registry.load_registry(path)
    print("  registry : %s" % path)
    print("  records  : %d" % len(records))
    if not records:
        print("\n  Nothing to do.")
        return 0

    incomplete = [r for r in records if any(r.get(f) is None for f in FIELDS)]
    if not incomplete:
        print("\n  Every record already carries params and gflops. Nothing to do.")
        return 0

    print("  missing  : %d record(s)" % len(incomplete))

    arms_cfg, data_cfg = load_arms_config(), load_data_config()
    cache: dict = {}

    print(
        "\n  %-16s %-24s %-18s %12s %12s"
        % ("run_id", "script", "arm", "params", "gflops")
    )
    filled = 0
    for record in incomplete:
        arm = record.get("arm")
        stats = measure(arm, arms_cfg, data_cfg, cache) if arm else None
        if stats is None:
            print(
                "  %-16s %-24s %-18s %12s %12s   SKIPPED"
                % (record.get("run_id"), record.get("script"), arm, "-", "-")
            )
            continue
        changed = []
        for field in FIELDS:
            if record.get(field) is None:
                record[field] = stats[field]
                changed.append(field)
        if changed:
            filled += 1
        print(
            "  %-16s %-24s %-18s %12s %12.4f   %s"
            % (
                record.get("run_id"),
                record.get("script"),
                arm,
                stats["params"],
                stats["gflops"],
                "+".join(changed) or "already set",
            )
        )

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
    still_null = [r for r in reloaded if any(r.get(f) is None for f in FIELDS)]
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
