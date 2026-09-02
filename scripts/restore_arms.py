#!/usr/bin/env python
"""restore_arms -- put artifacts/resolved_arms.yaml back over configs/arms.yaml.

    python scripts/restore_arms.py            # show the diff, then restore
    python scripts/restore_arms.py --check    # show the diff and exit, changing nothing

Scripts 01 and 02 rewrite `configs/arms.yaml` with the hyperparameters they
select, and snapshot the result to `artifacts/resolved_arms.yaml`. That snapshot
is the durable copy: on Colab `artifacts/` is a symlink into Drive, so it survives
a dropped session, while `configs/arms.yaml` lives in the clone and does not.

In a fresh clone whose `configs/arms.yaml` has reverted to the committed
(provisional) values, this script puts the resolved ones back.

Why it matters more than convenience: epochs, batch and lr feed the run_id hash.
Running script 03 against a reverted config does not resume -- it computes
different run_ids, retrains every fold under the old settings, and leaves the
registry holding two hyperparameter regimes for one arm. The run scripts refuse
to start when they detect that, and point here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srpcard.config import (  # noqa: E402
    RUN_DEFINING_HYPERPARAMETERS,
    arms_path,
    compare_arms_configs,
    load_yaml,
    resolved_arms_path,
    snapshot_body,
    unresolved_arms,
)


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def describe(hyperparameters: dict | None) -> str:
    if hyperparameters is None:
        return "(arm absent)"
    return "  ".join(
        "%s=%s" % (field, hyperparameters.get(field))
        for field in RUN_DEFINING_HYPERPARAMETERS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report the difference and exit without writing configs/arms.yaml",
    )
    args = parser.parse_args()

    rule("restore configs/arms.yaml from the resolved snapshot")

    snapshot = resolved_arms_path()
    live = arms_path()
    print("  snapshot : %s" % snapshot)
    print("  target   : %s" % live)

    if not snapshot.exists():
        print(
            "\n  No snapshot exists yet.\n"
            "  It is written by scripts/01_complete_medium_grid.py and\n"
            "  scripts/02_lr_sweep_baselines.py when they resolve hyperparameters.\n"
            "  If you expected one, check that artifacts/ points where you think it\n"
            "  does -- on Colab it should be a symlink into Drive."
        )
        return 1

    for line in snapshot.read_text(encoding="utf-8").splitlines():
        if line.startswith("# Written by") or line.startswith("# At") or line.startswith("# Git"):
            print("  %s" % line.lstrip("# ").rstrip())
        if not line.startswith("#"):
            break

    snapshot_cfg = load_yaml(snapshot)
    live_cfg = load_yaml(live)
    differences = compare_arms_configs(live_cfg, snapshot_cfg)

    print("\n  %-20s %-34s %s" % ("arm", "configs/arms.yaml (now)", "snapshot (would become)"))
    if not differences:
        print("  -- identical: every arm already carries the resolved hyperparameters --")
    for entry in differences:
        print(
            "  %-20s %-34s %s"
            % (entry["arm"], describe(entry["left"]), describe(entry["right"]))
        )

    pending = unresolved_arms(snapshot_cfg)
    if pending["lr_null"] or pending["provisional"]:
        print("\n  Note -- the snapshot itself is not fully resolved:")
        if pending["lr_null"]:
            print("    lr still null : %s  (run scripts/02_lr_sweep_baselines.py)" % pending["lr_null"])
        if pending["provisional"]:
            print(
                "    provisional   : %s  (run scripts/01_complete_medium_grid.py)"
                % pending["provisional"]
            )

    if args.check:
        print("\n  --check given: configs/arms.yaml left untouched.")
        return 1 if differences else 0

    if not differences:
        print("\n  Nothing to restore.")
        return 0

    live.write_text(snapshot_body(snapshot), encoding="utf-8", newline="\n")
    print("\n  restored %s from the snapshot" % live)
    print("  COMMIT configs/arms.yaml now -- the clone is what the next session gets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
