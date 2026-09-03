#!/usr/bin/env python
"""01b -- the full 2 x 3 x 3 grid for all three YOLO arms, under the UNIFORM protocol.

    python scripts/01b_uniform_grid.py --dry-run
    python scripts/01b_uniform_grid.py
    python scripts/01b_uniform_grid.py --arms yolo26n

54 runs: 3 arms x (2 epochs x 3 batch x 3 lr), on the DEVELOPMENT SPLIT. Each
arm's locked configuration is selected by argmax of validation macro-F1 over its
own 18 and written back into configs/arms.yaml.

=============================================================================
WHY THIS SCRIPT EXISTS, AND WHY IT DOES NOT REPLACE SCRIPT 01
=============================================================================

Script 01 reproduces the legacy grid: ultralytics' trainer with ultralytics'
defaults, on the raw 695 with contaminated labels and an unweighted loss. Its
`--control-rerun` re-ran one already-complete configuration on the current
machine, and the result did not reproduce:

    m_ep25_bs8_lr1e-02   legacy 0.7657   re-run 0.6235   difference -0.1423

seven times the +/-0.02 comparability tolerance. The trainer log explains it.
Ultralytics' augmentation defaults were live in all 46 legacy runs:

    auto_augment=randaugment, erasing=0.4, fliplr=0.5, hsv_h=0.015,
    hsv_s=0.7, hsv_v=0.4, scale=0.5, translate=0.1

The manuscript argues explicitly AGAINST geometric augmentation (it distorts
the curve morphology) and against photometric augmentation (useless on line
drawings), and configs/arms.yaml's own uniform_protocol comment notes that a
horizontal flip reverses the traversal direction of the load curve. The legacy
grid was therefore searched under a protocol the write-up rejects. See
MIGRATION_NOTES.md section 16.

That also explains the gap better than mixed precision does. RandAugment and
random erasing are stochastic, their RNG stream depends on the dataloader
worker count, and Colab supplied 2 workers against the 4 requested. The model
saw genuinely different images, not the same images at a different precision.
On a 69-image validation slice, seven images separate the two values.

So the legacy grid cannot select the locked configurations for a paper whose
methods section forbids augmentation. This script re-searches them under the
protocol that scripts 02-05 actually use:

    no augmentation, class weights APPLIED, selection on validation macro-F1
    tie-broken by unweighted validation loss, no early stopping, one device

Script 01 and its records are retained unchanged: they are the reproduction of
the thesis grid and the evidence for the augmentation and class-weight
findings. Both grids are reported. Records from here carry split_kind "dev" and
extra.protocol "uniform", so they can never be pooled with script 01's.
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from srpcard import data as srp_data  # noqa: E402
from srpcard import evaluate, registry  # noqa: E402
from srpcard import folds as srp_folds  # noqa: E402
from srpcard.config import (  # noqa: E402
    artifacts_dir,
    load_arms_config,
    load_data_config,
    resolve_data_root,
    snapshot_arms,
)
from srpcard.efficiency import profile  # noqa: E402
from srpcard.legacy_split import load_dev_split  # noqa: E402
from srpcard.models import add_fallback_argument, build_model  # noqa: E402
from srpcard.train import (  # noqa: E402
    ImageCache,
    TrainConfig,
    labels_by_idx_map,
    require_class_weights_verified,
    train_fold,
)

SCRIPT = "01b_uniform_grid"
# Tags every record, and scopes the grid-membership check. Script 01 writes
# "legacy_unweighted_ultralytics" for the same arms on the same split, and the
# two must never be pooled.
PROTOCOL = "uniform"
YOLO_ARMS = ["yolo26n", "yolo26s", "yolo26m"]
# Distinct from DEV_SEED in script 02 (20000) and from the CV run_seed space.
GRID_SEED = 30000

VARIANT = {"yolo26n": "n", "yolo26s": "s", "yolo26m": "m"}

# Fitted on the nine yolo26m records already in the registry: 8 at 50 epochs
# (216-303 s, mean 262) and the ep25 control re-run (192 s). Ultralytics with
# augmentation; the uniform loop should be no slower.
SECONDS_PER_EPOCH_PER_GFLOP = 0.5744
SECONDS_PER_EPOCH_FLOOR = 1.5
SECONDS_FIXED_PER_RUN = 35.0
GFLOPS = {"yolo26n": 0.3983, "yolo26s": 1.4864, "yolo26m": 4.8512}


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def config_key(arm: str, epochs: int, batch: int, lr: float) -> str:
    return "%s_ep%d_bs%d_lr%.0e" % (VARIANT[arm], epochs, batch, lr)


def project_seconds(arm: str, epochs: int) -> float:
    per_epoch = max(SECONDS_PER_EPOCH_FLOOR, SECONDS_PER_EPOCH_PER_GFLOP * GFLOPS[arm])
    return SECONDS_FIXED_PER_RUN + per_epoch * epochs


# --------------------------------------------------------------------------
# writing the winners back
# --------------------------------------------------------------------------


def arm_block_bounds(text: str, arm: str) -> tuple[int, int]:
    """Character offsets of one arm's block in configs/arms.yaml."""
    start = text.index("\n  %s:\n" % arm) + 1
    following = [
        text.index("\n  %s:\n" % other) + 1
        for other in (
            "yolo26n", "yolo26s", "yolo26m", "mobilenetv3_small", "resnet18",
        )
        if "\n  %s:\n" % other in text and text.index("\n  %s:\n" % other) + 1 > start
    ]
    end = min(following) if following else text.index("\n# --- The uniform training")
    return start, end


def write_back_winner(
    arm: str, epochs: int, batch: int, lr: float, key: str, f1_val: float,
    path: Path | None = None,
):
    """Replace one arm's epochs/batch/lr/locked/provisional/lr_source, in place.

    `lr_source` names BOTH the grid and the protocol, because two grids now
    exist for these arms and a value that does not say which one it came from is
    exactly the ambiguity this script was written to remove.

    `path` exists so tests can drive this against a copy; it defaults to the
    repository's own configs/arms.yaml.
    """
    path = path or Path(__file__).resolve().parents[1] / "configs" / "arms.yaml"
    text = path.read_text(encoding="utf-8")
    start, end = arm_block_bounds(text, arm)

    source = (
        '"uniform_grid:01b:%s (f1_macro_val=%.4f, 18/18 grid, dev split, '
        'UNIFORM protocol: no augmentation, class weights applied, '
        'val-macro-F1 selection; written by 01b_uniform_grid.py)"' % (key, f1_val)
    )

    seen = set()
    out = []
    for line in text[start:end].splitlines():
        stripped = line.strip()
        # Drop the stale PROVISIONAL commentary from the yolo26m block: it
        # describes a state two grids ago and would otherwise outlive it.
        if stripped.startswith("#") and arm == "yolo26m":
            continue
        if stripped.startswith("epochs:") and line.startswith("    "):
            out.append("    epochs: %d" % epochs)
            seen.add("epochs")
        elif stripped.startswith("batch:") and line.startswith("    "):
            out.append("    batch: %d" % batch)
            seen.add("batch")
        elif stripped.startswith("lr:") and line.startswith("    "):
            out.append("    lr: %s" % repr(lr))
            seen.add("lr")
        elif stripped.startswith("locked:"):
            out.append("    locked: true")
            seen.add("locked")
        elif stripped.startswith("provisional:"):
            out.append("    provisional: false")
            seen.add("provisional")
        elif stripped.startswith("lr_source:"):
            out.append("    lr_source: %s" % source)
            seen.add("lr_source")
        else:
            out.append(line)

    missing = {"epochs", "batch", "lr", "locked", "lr_source"} - seen
    if missing:
        raise SystemExit(
            "configs/arms.yaml: arm %r has no %s line to rewrite. Refusing to\n"
            "  guess where it belongs -- fix the file by hand." % (arm, sorted(missing))
        )

    updated = "\n".join(out) + ("\n" if text[start:end].endswith("\n") else "")
    path.write_text(text[:start] + updated + text[end:], encoding="utf-8", newline="\n")
    print("  [arms.yaml] %-10s epochs=%d batch=%d lr=%g  locked" % (arm, epochs, batch, lr))


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--arms", nargs="*", default=None, help="subset (default: all three)")
    parser.add_argument("--limit", type=int, default=None, help="stop after N runs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--no-write-back",
        action="store_true",
        help="do not touch configs/arms.yaml, even when the grid is complete",
    )
    add_fallback_argument(parser)
    args = parser.parse_args()

    data_cfg = load_data_config()
    arms_cfg = load_arms_config()
    arms = args.arms or YOLO_ARMS
    unknown = [a for a in arms if a not in YOLO_ARMS]
    if unknown:
        print("Unknown arm(s) %s. This script covers %s." % (unknown, YOLO_ARMS))
        return 2

    grid = arms_cfg["medium_grid"]
    epochs_values = list(grid["epochs"])
    batch_values = list(grid["batch"])
    lr_values = [float(v) for v in grid["lr"]]

    rule("01b -- uniform-protocol grid for the YOLO arms (DEVELOPMENT SPLIT)")
    print("  arms           : %s" % ", ".join(arms))
    print("  grid           : %s epochs x %s batch x %s lr = %d per arm"
          % (epochs_values, batch_values, lr_values,
             len(epochs_values) * len(batch_values) * len(lr_values)))
    print("  corpus         : raw 695, dev split (selection only)")
    print("  protocol       : UNIFORM -- no augmentation, class weights APPLIED,")
    print("                   val macro-F1 selection, tie-break unweighted val loss")
    print("  NOT comparable with script 01's records, which are the legacy protocol")

    index = srp_data.load_image_index()
    index_path = artifacts_dir(data_cfg) / "image_index.csv"
    corpus_fp = srp_folds.dev_corpus_fingerprint(index, index_path)
    print("\n[corpus] %s  n=%d  sha1=%s"
          % (corpus_fp["kind"], corpus_fp["n"], corpus_fp["sha1_of_sorted_included_sha1s"]))

    dev_split = load_dev_split()
    print("[dev split] train %d  val %d  test %d"
          % (len(dev_split["train"]), len(dev_split["val"]), len(dev_split["test"])))

    specs = []
    for arm in arms:
        for epochs in epochs_values:
            for batch in batch_values:
                for lr in lr_values:
                    spec = {
                        "arm": arm,
                        "architecture": arms_cfg["arms"][arm]["architecture"],
                        "script": SCRIPT,
                        "split_kind": "dev",
                        "repeat": None,
                        "fold": None,
                        "epochs": int(epochs),
                        "batch": int(batch),
                        "lr": float(lr),
                        "class_weights": arms_cfg["shared"]["class_weights"],
                        "run_seed": GRID_SEED,
                        "extra": "uniform_grid",
                    }
                    spec["run_id"] = registry.compute_run_id(**spec)
                    spec["key"] = config_key(arm, epochs, batch, lr)
                    specs.append(spec)

    todo, skipped = registry.plan_runs(specs)
    registry.print_plan(SCRIPT, todo, skipped)
    if args.limit:
        todo = todo[: args.limit]
        print("[limit] running only the first %d" % len(todo))

    projected = sum(project_seconds(s["arm"], s["epochs"]) for s in todo)
    print("\n[budget] %d run(s) remaining, projected %.0f s = %.2f h"
          % (len(todo), projected, projected / 3600))
    for arm in arms:
        arm_todo = [s for s in todo if s["arm"] == arm]
        if arm_todo:
            seconds = sum(project_seconds(arm, s["epochs"]) for s in arm_todo)
            print("           %-10s %2d run(s)  %6.0f s = %.2f h"
                  % (arm, len(arm_todo), seconds, seconds / 3600))
    print("           extrapolated from the yolo26m records already in the registry;")
    print("           treat it as an order of magnitude, not a promise.")

    if args.dry_run:
        print()
        for spec in todo:
            print("  TODO %s  %-22s epochs %2d batch %2d lr %.0e"
                  % (spec["run_id"], spec["key"], spec["epochs"], spec["batch"], spec["lr"]))
        return 0

    if todo:
        # NOT the locked-config drift guard: this script sweeps epochs/batch/lr
        # across 18 combinations per arm, so it can never match a single locked
        # value. See registry.SWEEP_SCRIPTS. What it asserts instead is that every
        # configuration it is about to run, and every one it has already run under
        # this same script and protocol, is a point of the grid declared in
        # configs/arms.yaml -- which catches the grid being edited between
        # sessions without firing on the sweep itself.
        registry.assert_sweep_within_grid(
            script=SCRIPT,
            protocol=PROTOCOL,
            arms=arms,
            grid_points={
                (int(e), int(b), float(lr))
                for e in epochs_values
                for b in batch_values
                for lr in lr_values
            },
            planned=[(s["epochs"], s["batch"], s["lr"]) for s in specs],
            split_kind="dev",
        )
        weights_proof = require_class_weights_verified(
            int(arms_cfg["shared"]["num_classes"]), script=SCRIPT
        )
        data_root = resolve_data_root(data_cfg)
        labels_by_idx = labels_by_idx_map(index, data_cfg)
        cache = ImageCache(index, data_root, int(arms_cfg["shared"]["image_size"]))
        cache.warm(dev_split["train"] + dev_split["val"] + dev_split["test"])
        print("[cache] %d images letterboxed into RAM" % len(cache._cache))

    for position, spec in enumerate(todo, 1):
        rule("run %d/%d  %s  %s" % (position, len(todo), spec["arm"], spec["key"]))
        bundle = build_model(
            spec["arm"],
            arms_cfg,
            data_cfg,
            with_efficiency=False,
            seed=GRID_SEED,
            allow_pretrained_fallback=args.allow_pretrained_fallback,
        )
        cfg = TrainConfig.from_arm(
            spec["arm"],
            arms_cfg,
            epochs=spec["epochs"],
            batch=spec["batch"],
            lr=spec["lr"],
        )
        started = time.perf_counter()
        result = train_fold(
            bundle,
            cache,
            dev_split["train"],
            dev_split["val"],
            labels_by_idx,
            cfg,
            seed=GRID_SEED,
            device=args.device,
            verbose=not args.quiet,
        )
        val_metrics = evaluate.evaluate_fold(
            bundle.module, cache, dev_split["val"], labels_by_idx, data_cfg
        )
        test_metrics = evaluate.evaluate_fold(
            bundle.module, cache, dev_split["test"], labels_by_idx, data_cfg
        )
        efficiency = profile(bundle.module, cfg.image_size, latency=True)
        wall = round(time.perf_counter() - started, 2)
        print("  -> selected epoch %d/%d  val f1_macro %.4f  (dev-test %.4f)  %.1fs"
              % (result.best_epoch + 1, cfg.epochs, val_metrics["f1_macro"],
                 test_metrics["f1_macro"], wall))

        registry.append_record(
            registry.build_record(
                run_id=spec["run_id"],
                script=SCRIPT,
                arm=spec["arm"],
                architecture=spec["architecture"],
                split_kind="dev",
                repeat=None,
                fold=None,
                epochs=spec["epochs"],
                batch=spec["batch"],
                lr=spec["lr"],
                class_weights=spec["class_weights"],
                run_seed=GRID_SEED,
                val_seed=None,
                checkpoint_resolved=bundle.checkpoint_resolved,
                pretrained_fallback_used=bundle.pretrained_fallback_used,
                class_weights_verified=weights_proof["passed"],
                class_weights_proof=weights_proof,
                corpus_fingerprint=corpus_fp,
                training=registry.training_outcome(result),
                metrics=test_metrics,
                efficiency=efficiency,
                wall_time_s=wall,
                determinism_status=result.determinism,
                extra={
                    "key": spec["key"],
                    # The two fields that keep this grid separable from script 01's.
                    "protocol": PROTOCOL,
                    "grid": "uniform_18",
                    "augmentation": "none",
                    "class_weights_applied": True,
                    "f1_macro_val": val_metrics["f1_macro"],
                    "accuracy_val": val_metrics["accuracy"],
                    "val_metrics": val_metrics,
                    "selection_metric": "val_f1_macro",
                    "selection_tiebreak": "val_loss_unweighted",
                    "device": result.device,
                },
            )
        )
        print("  [registry] appended %s" % spec["run_id"])

    # ---------------- winners ----------------
    rule("uniform-protocol winners, per arm")
    rows = [
        {
            "arm": r["arm"],
            "key": r["extra"]["key"],
            "epochs": r["epochs"],
            "batch": r["batch"],
            "lr": r["lr"],
            "f1_macro_val": r["extra"]["f1_macro_val"],
            "f1_macro_dev_test": r["f1_macro"],
        }
        for r in registry.load_registry()
        if r.get("script") == SCRIPT and (r.get("extra") or {}).get("f1_macro_val") is not None
    ]
    if not rows:
        print("No runs recorded yet.")
        return 1

    table = pd.DataFrame(rows).drop_duplicates("key")
    out_csv = artifacts_dir(data_cfg) / "uniform_grid.csv"
    table.sort_values(["arm", "f1_macro_val"], ascending=[True, False]).to_csv(
        out_csv, index=False, lineterminator="\n"
    )
    print("[artifacts] wrote %s  (%d rows)" % (out_csv, len(table)))

    expected = len(epochs_values) * len(batch_values) * len(lr_values)
    print("\n  %-10s %-22s %7s %6s %9s %14s" % ("arm", "key", "epochs", "batch", "lr", "f1_macro_val"))
    for arm in arms:
        subset = table[table["arm"] == arm]
        if subset.empty:
            continue
        winner = subset.loc[subset["f1_macro_val"].idxmax()]
        print("  %-10s %-22s %7d %6d %9.0e %14.4f%s"
              % (arm, winner["key"], winner["epochs"], winner["batch"], winner["lr"],
                 winner["f1_macro_val"],
                 "" if len(subset) == expected else "   (%d/%d only)" % (len(subset), expected)))

    # side by side with what arms.yaml carried before
    print("\n  old (legacy grid, augmented) vs new (uniform grid):")
    print("  %-10s %-34s %-34s" % ("arm", "OLD", "NEW"))
    changes = []
    for arm in arms:
        subset = table[table["arm"] == arm]
        if subset.empty or len(subset) != expected:
            continue
        winner = subset.loc[subset["f1_macro_val"].idxmax()]
        old = arms_cfg["arms"][arm]
        old_text = "epochs %2d batch %2d lr %-7g" % (old["epochs"], old["batch"], old["lr"])
        new_text = "epochs %2d batch %2d lr %-7g" % (
            int(winner["epochs"]), int(winner["batch"]), float(winner["lr"])
        )
        same = (
            int(old["epochs"]) == int(winner["epochs"])
            and int(old["batch"]) == int(winner["batch"])
            and abs(float(old["lr"]) - float(winner["lr"])) < 1e-12
        )
        print("  %-10s %-34s %-34s %s" % (arm, old_text, new_text,
                                          "unchanged" if same else "CHANGED"))
        changes.append((arm, winner))

    if args.no_write_back:
        print("\n  --no-write-back given: configs/arms.yaml left untouched.")
        return 0
    if not changes:
        print("\n  No arm has a complete %d-configuration grid yet; nothing written back."
              % expected)
        print("  Re-run until every arm is complete, then the winners are locked.")
        return 0

    rule("writing the locked configurations into configs/arms.yaml")
    for arm, winner in changes:
        write_back_winner(
            arm,
            int(winner["epochs"]),
            int(winner["batch"]),
            float(winner["lr"]),
            str(winner["key"]),
            float(winner["f1_macro_val"]),
        )
    snapshot = snapshot_arms(SCRIPT, data_cfg)
    print("\n[arms.yaml] snapshotted to %s" % snapshot)
    print("            recover it in a fresh clone with: python scripts/restore_arms.py")
    print("\nCOMMIT configs/arms.yaml now -- the clone is what the next session gets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
