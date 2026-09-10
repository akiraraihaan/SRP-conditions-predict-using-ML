#!/usr/bin/env python
"""backfill_efficiency -- fill DERIVABLE fields in existing registry records.

    python scripts/backfill_efficiency.py --dry-run   # report, change nothing
    python scripts/backfill_efficiency.py             # write, after a .bak copy

`params`, `gflops`, `size_mb_fp32` and `size_mb_fp16` are pure functions of
(architecture, num_classes, image_size), so a record missing them can have them
derived rather than re-run. Leaving them null would force aggregate.py and
figures.py to special-case values that are perfectly well defined.

Two rounds of this have been needed:

  params / gflops   script 01's first eight records were written before it
                    profiled the trained module.
  hardware          gpu / cuda_version / driver_version / device_kind were
                    promoted to the top level so a mixed-hardware check can
                    compare them across records. The values were already in
                    `library_versions`, just not queryable, so existing records
                    are recovered from there rather than being re-run.
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

# Promoted out of library_versions, where they were recorded but unqueryable.
HARDWARE_FIELDS = ("gpu", "gpu_count", "cuda_version", "driver_version",
                   "compute_capability", "device_kind")


def hardware_from_library_versions(record: dict) -> dict:
    """Recover the top-level hardware block from the record's own provenance.

    Lossless: `library_versions` already held the GPU name and CUDA version.
    Fields it never carried (driver, compute capability) stay None -- they were
    not recorded at the time and inventing them would be worse than a null.
    """
    versions = record.get("library_versions") or {}
    available = str(versions.get("cuda_available", "")).lower() == "true"
    return {
        "gpu": versions.get("gpu"),
        "gpu_count": 1 if versions.get("gpu") else 0,
        "cuda_version": None if versions.get("torch_cuda") in (None, "None") else versions.get("torch_cuda"),
        "driver_version": None,
        "compute_capability": None,
        "device_kind": "cuda" if available else "cpu",
    }


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def derived_from_siblings(records: list[dict]) -> tuple[dict, dict]:
    """Derived fields already recorded for each ARCHITECTURE, field by field.

    Preferred over measuring locally, and not scoped by script, arm or protocol:
    params, gflops and the size_mb family are functions of the architecture
    alone, so any record carrying that architecture is a valid source no matter
    which script wrote it. mobilenet_v3_small resolves from 02, 03 and 04
    together; yolo26n-cls from 01b and 03.

    Unanimity is required, and it is checked PER FIELD rather than over the whole
    tuple. Checking the tuple excluded an architecture entirely as soon as any
    one of its six fields disagreed, which is how yolo26m-cls -- 41 complete
    records, all of them agreeing that params is 10,366,026 and gflops 4.8512 --
    ended up needing a model rebuilt to recover two numbers the registry already
    stated unanimously. The two that genuinely disagree are the container-
    inclusive sizes, and those are the ones that should be excluded.

    A field whose records disagree is left out rather than arbitrated. A
    disagreement means the quantity is not settled, and picking one of the
    candidates here would settle it by accident.

    Returns (resolved, rejected): the unanimous value per architecture and field,
    and the competing values for every field that was excluded.
    """
    seen: dict[str, dict[str, set]] = {}
    for record in records:
        architecture = record.get("architecture")
        if not architecture:
            continue
        for field in DERIVED_FIELDS:
            value = record.get(field)
            if value is not None:
                seen.setdefault(architecture, {}).setdefault(field, set()).add(value)

    resolved: dict[str, dict] = {}
    rejected: dict[str, dict] = {}
    for architecture, fields in seen.items():
        for field, values in fields.items():
            if len(values) == 1:
                resolved.setdefault(architecture, {})[field] = next(iter(values))
            else:
                rejected.setdefault(architecture, {})[field] = sorted(values)
    return resolved, rejected


def print_sibling_table(resolved: dict, rejected: dict) -> None:
    """What the registry already knows, before anything is written.

    These are the figures the manuscript reports, so they are printed rather
    than merely used.
    """
    if not resolved and not rejected:
        print("  no architecture has a complete record to borrow from")
        return
    for architecture in sorted(set(resolved) | set(rejected)):
        print("      %s" % architecture)
        for field in DERIVED_FIELDS:
            if field in resolved.get(architecture, {}):
                print("          %-22s %s" % (field, resolved[architecture][field]))
        for field in DERIVED_FIELDS:
            if field in rejected.get(architecture, {}):
                print(
                    "          %-22s EXCLUDED -- records disagree: %s"
                    % (field, ", ".join(str(v) for v in rejected[architecture][field]))
                )


def measure(arm: str, arms_cfg, data_cfg, cache: dict) -> dict:
    """The derived efficiency fields for one arm, built once and reused.

    Raises on failure rather than returning None. A missed arm used to be printed
    as SKIPPED and the loop carried on, which is what made a partial write
    possible; the caller now resolves every arm it needs BEFORE anything is
    changed. See main().
    """
    if arm not in cache:
        bundle = build_model(arm, arms_cfg, data_cfg, with_efficiency=False)
        stats = profile(bundle.module, int(arms_cfg["shared"]["image_size"]), latency=False)
        cache[arm] = {field: stats[field] for field in DERIVED_FIELDS}
    return cache[arm]


def needs_derived(record: dict, *, refresh: bool = False) -> bool:
    """True when this record needs a MODEL built to fill it."""
    if refresh:
        return True
    return any(record.get(field) is None for field in DERIVED_FIELDS)


def needs_hardware(record: dict) -> bool:
    """True when the hardware block is still buried in library_versions.

    Costs no model: the values are inside the record already.
    """
    return "device_kind" not in record


def needs_legacy_size(record: dict) -> bool:
    """A pre-disambiguation record: size_mb still holds the checkpoint file size."""
    # Key PRESENCE, not truthiness: a record whose checkpoint size was recorded
    # as null has been handled, and re-planning it every run would make the
    # post-apply verification below refuse a write that was in fact complete.
    return (
        record.get("script") in LEGACY_SIZE_SCRIPTS
        and "size_mb_checkpoint_file" not in record
    )


def needs_work(record: dict, *, refresh: bool = False) -> bool:
    return (
        needs_derived(record, refresh=refresh)
        or needs_hardware(record)
        or needs_legacy_size(record)
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

    # ------------------------------------------------------------------ plan
    #
    # Resolve every arm this invocation needs BEFORE changing a single record.
    # An arm that cannot be built used to be printed as SKIPPED while the loop
    # carried on, so a machine whose ultralytics cannot load yolo26*-cls.pt would
    # fill the mobilenet records and leave the YOLO ones empty -- a half-restored
    # registry, which is a worse state than the one it started in because nothing
    # afterwards can tell which half is which. Either the whole plan is
    # satisfiable or nothing is written.
    #
    # Only records needing DERIVED fields need a model. Hardware recovery reads
    # the record's own library_versions, so a hardware-only backfill is always
    # satisfiable and does not depend on any framework being loadable here.
    siblings, disputed = (
        ({}, {}) if args.refresh else derived_from_siblings(records)
    )

    print("\n-- what the registry already states ---------------------------------")
    print_sibling_table(siblings, disputed)

    wanting = [r for r in incomplete if needs_derived(r, refresh=args.refresh)]

    def unresolved(record: dict) -> list[str]:
        """Fields this record still needs that no sibling can supply."""
        known = siblings.get(record.get("architecture"), {})
        return [
            field
            for field in DERIVED_FIELDS
            if (args.refresh or record.get(field) is None) and field not in known
        ]

    required_arms = sorted({r.get("arm") for r in wanting if unresolved(r)})
    print(
        "\n  to build : %s" % (", ".join(str(a) for a in required_arms) or "none")
    )
    if required_arms:
        for arm in required_arms:
            gaps = sorted({f for r in wanting if r.get("arm") == arm for f in unresolved(r)})
            print("             %-18s needs %s" % (arm, ", ".join(gaps)))

    failures: list[tuple[str, str]] = []
    for arm in required_arms:
        if not arm:
            failures.append(("<no arm recorded>", "record carries no arm"))
            continue
        try:
            measure(arm, arms_cfg, data_cfg, cache)
        except Exception as exc:  # noqa: BLE001 - collected, reported, then fatal
            failures.append((arm, "%s: %s" % (type(exc).__name__, exc)))

    if failures:
        rule("REFUSING TO WRITE -- the plan cannot be satisfied in full")
        blocked = [
            r for r in wanting if r.get("arm") in {name for name, _ in failures}
        ]
        print(
            "  %d of %d planned record(s) cannot be filled on this machine, so\n"
            "  NOTHING has been written -- not even the %d that could be.\n"
            % (len(blocked), len(incomplete), len(incomplete) - len(blocked))
        )
        for arm, reason in failures:
            n = sum(1 for r in blocked if r.get("arm") == arm)
            print("      %-20s %d record(s)" % (arm, n))
            print("      %-20s %s" % ("", reason))
        print(
            "\n  A partial backfill is a state nobody can reason about later: every\n"
            "  record still validates and the run_ids still match, so the half that\n"
            "  was filled is indistinguishable from the half that was not.\n"
            "\n  These values need no model at all -- params, gflops and the size_mb\n"
            "  family are functions of the architecture, and the hardware block is\n"
            "  inside each record already. If a committed copy of the registry has\n"
            "  them, recover them from git instead of rebuilding models here:\n"
            "\n      git log --oneline -- artifacts/registry.jsonl\n"
            "      git show <commit>:artifacts/registry.jsonl > /tmp/older.jsonl\n"
            "      python scripts/merge_registry.py artifacts/registry.jsonl\n"
            "          older.jsonl --out merged.jsonl --dry-run\n"
            "\n  merge_registry prefers the more populated record and refuses on any\n"
            "  measured-metric disagreement, so it cannot invent a value."
        )
        return 1

    print(
        "\n  %-16s %-18s %10s %8s %9s %9s %10s"
        % ("run_id", "arm", "params", "gflops", "fp32", "fp16", ".pt file")
    )
    filled = 0
    for record in incomplete:
        arm = record.get("arm")
        # Per field, not per record: an architecture can be unanimous on params
        # and gflops while its container-inclusive sizes disagree, and there is
        # no reason to discard the four that ARE settled along with the two that
        # are not.
        known = {} if args.refresh else siblings.get(record.get("architecture"), {})
        measured = cache.get(arm) or {}
        stats = {field: known.get(field, measured.get(field)) for field in DERIVED_FIELDS}
        if all(value is None for value in stats.values()):
            stats = None
        sources = set()

        changed = []
        if "device_kind" not in record:
            recovered = hardware_from_library_versions(record)
            for field in HARDWARE_FIELDS:
                record[field] = recovered[field]
            changed.append("hardware<-library_versions(%s)" % (recovered["gpu"] or "cpu"))

        # Move the old checkpoint-file size out of size_mb before size_mb is
        # redefined, so the published figure is preserved rather than replaced.
        if needs_legacy_size(record):
            record["size_mb_checkpoint_file"] = record.get("size_mb")
            changed.append("size_mb_checkpoint_file")

        # stats is None only for a record that needed nothing derived -- the
        # plan above guarantees every arm it DID need was resolved.
        if stats is not None:
            for field in DERIVED_FIELDS:
                value = stats[field]
                if value is None:
                    continue
                origin = "registry" if field in known else "measured here"
                if record.get(field) is None:
                    record[field] = value
                    changed.append(field)
                    sources.add(origin)
                elif args.refresh and record[field] != value:
                    changed.append("%s %s->%s" % (field, record[field], value))
                    record[field] = value
                    sources.add(origin)
            if sources:
                changed.append("<- %s" % " + ".join(sorted(sources)))

        # The alias comes from THIS RECORD's own fp16 measurement, never from a
        # fresh local one. Container overhead differs between torch versions, so
        # recomputing here overwrites what the run actually measured with what
        # this machine measures -- and leaves size_mb != size_mb_fp16 inside one
        # record. Only a record that has no fp16 figure at all falls back to the
        # derived value computed above.
        alias = record.get(PRIMARY_SIZE_FIELD)
        if alias is None and stats is not None:
            alias = stats.get(PRIMARY_SIZE_FIELD)
        if alias is not None and record.get("size_mb") != alias:
            changed.append("size_mb %s->%s" % (record.get("size_mb"), alias))
            record["size_mb"] = alias

        if changed:
            filled += 1
        def show(field, fmt="%s"):
            value = None if stats is None else stats.get(field)
            return "-" if value is None else fmt % value

        print(
            "  %-16s %-18s %10s %8s %9s %9s %10s"
            % (
                record.get("run_id"),
                arm,
                show("params"),
                show("gflops", "%.4f"),
                show("size_mb_fp32", "%.3f"),
                show("size_mb_fp16", "%.3f"),
                record.get("size_mb_checkpoint_file", "-"),
            )
        )
        print("      %s" % ", ".join(changed))

    # Every record the plan claimed is now filled, or the plan was wrong.
    unsatisfied = [
        r for r in incomplete
        if needs_derived(r) or needs_hardware(r) or needs_legacy_size(r)
    ]
    if unsatisfied:
        rule("REFUSING TO WRITE -- %d planned record(s) are still incomplete"
             % len(unsatisfied))
        for record in unsatisfied[:10]:
            still = [f for f in DERIVED_FIELDS if record.get(f) is None]
            print("      %-16s %-18s %s"
                  % (record.get("run_id"), record.get("arm"),
                     ", ".join(still) or "hardware block"))
        print(
            "\n  This is a bug in this script, not in the registry: it planned to fill\n"
            "  these and did not. Nothing has been written."
        )
        return 1

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
