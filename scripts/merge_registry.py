#!/usr/bin/env python
"""merge_registry -- combine two diverged copies of artifacts/registry.jsonl.

    python scripts/merge_registry.py A.jsonl B.jsonl --out C.jsonl
    python scripts/merge_registry.py A.jsonl B.jsonl --out C.jsonl --dry-run

Both inputs are ordinary local paths. This script never looks for the registry
anywhere but where you point it.

WHY THIS EXISTS

The registry legitimately lives in two places: committed in the repository, and
live in whatever durable storage a session writes to. They diverge as soon as one
is edited without the other. That happened here -- a backfill that added device
provenance was committed to the repository while a session appended six new runs
to the other copy -- and overwriting either direction loses work.

Merging is safe because a record is IMMUTABLE once written. `run_id` is a hash of
the parameters that define a run, so two records sharing an id describe the same
run and can differ only in how complete they are. The merge therefore:

  - takes every run_id present in either input;
  - where an id is in both, keeps the more populated record and fills its gaps
    from the other, overwriting nothing;
  - REFUSES the whole merge, writing nothing, if two records sharing an id
    disagree on any measured value -- f1_macro, accuracy, the per-class arrays,
    the confusion matrix, wall_time_s. That means the immutability assumption is
    false and one of the two would be silently discarded.

The output path is backed up to a timestamped .bak before being overwritten.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srpcard import registry  # noqa: E402


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", help="a registry.jsonl")
    parser.add_argument("second", help="another registry.jsonl")
    parser.add_argument("--out", required=True, help="where to write the merged registry")
    parser.add_argument(
        "--dry-run", action="store_true", help="report the merge and write nothing"
    )
    args = parser.parse_args()

    first, second, out = Path(args.first), Path(args.second), Path(args.out)
    for path in (first, second):
        if not path.exists():
            print("No such file: %s" % path)
            return 2

    rule("merge two registries by run_id")
    print("  A : %s" % first)
    print("  B : %s" % second)
    print("  -> %s" % out)
    print()

    records_a = registry.load_registry(first)
    records_b = registry.load_registry(second)

    try:
        merged, report = registry.merge_registries(
            records_a, records_b, label_a="A", label_b="B"
        )
    except registry.RegistryConflictError as exc:
        print(exc)
        return 1

    registry.print_merge_report(report)

    # what the merge produced, by script, so the totals are checkable at a glance
    by_script: dict[str, int] = {}
    for record in merged:
        key = record.get("script", "?")
        by_script[key] = by_script.get(key, 0) + 1
    print("\n  merged registry by script:")
    for script, count in sorted(by_script.items()):
        print("      %-28s %3d" % (script, count))

    incomplete = [r for r in merged if registry.missing_fields(r)]
    if incomplete:
        print(
            "\n  NOTE: %d merged record(s) still lack a required schema field.\n"
            "        The merge fills gaps from the other input; it cannot invent a\n"
            "        value neither copy had. Run scripts/backfill_efficiency.py."
            % len(incomplete)
        )

    if args.dry_run:
        print("\n  --dry-run: nothing written.")
        return 0

    if out.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = out.with_suffix(".jsonl.%s.bak" % stamp)
        shutil.copy2(out, backup)
        print("\n  backup : %s" % backup)

    registry.write_registry(merged, out)
    print("  wrote  : %s  (%d record(s))" % (out, len(merged)))

    reloaded = registry.load_registry(out)
    if len(reloaded) != len(merged):
        print("  [FAIL] re-read %d record(s), expected %d" % (len(reloaded), len(merged)))
        return 1
    print("  verify : re-read %d record(s)" % len(reloaded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
