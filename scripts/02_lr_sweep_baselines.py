#!/usr/bin/env python
"""02 -- learning-rate sweep for the two baseline architectures. 6 runs.

    python scripts/02_lr_sweep_baselines.py
    python scripts/02_lr_sweep_baselines.py --dry-run

mobilenetv3_small and resnet18 only, lr in {1e-4, 1e-3, 1e-2}, on the
DEVELOPMENT SPLIT. Batch size is fixed at 16 and is NOT swept. The winning lr
per arm is written back into configs/arms.yaml.

Unlike script 01, this one uses the UNIFORM protocol -- weighted training loss,
validation macro-F1 selection -- because these runs must be commensurable with
script 03, not with the legacy grid. The two baselines were never trained in the
old study, so there is nothing legacy to stay comparable with.

Selection is on the development split's VALIDATION partition. The development
split's test partition is not touched here.
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
from srpcard.config import (  # noqa: E402
    artifacts_dir,
    load_arms_config,
    load_data_config,
    resolve_data_root,
)
from srpcard.efficiency import profile  # noqa: E402
from srpcard.legacy_split import load_dev_split  # noqa: E402
from srpcard.models import build_model  # noqa: E402
from srpcard.train import ImageCache, TrainConfig, labels_by_idx_map, train_fold  # noqa: E402

SCRIPT = "02_lr_sweep_baselines"
DEV_SEED = 20000  # distinct from the CV run_seed space (10000 + repeat*100 + fold)


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def write_back_lr(arm: str, lr: float, f1_val: float) -> None:
    """Replace `lr: null` for one arm in configs/arms.yaml, in place."""
    path = Path(__file__).resolve().parents[1] / "configs" / "arms.yaml"
    text = path.read_text(encoding="utf-8")

    start = text.index("  %s:" % arm)
    following = [
        text.index("  %s:" % other)
        for other in ("yolo26n", "yolo26s", "yolo26m", "mobilenetv3_small", "resnet18")
        if text.index("  %s:" % other) > start
    ]
    end = min(following) if following else text.index("\n# --- The uniform training protocol")

    block = text[start:end]
    lines = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("lr:") and line.startswith("    "):
            lines.append("    lr: %g" % lr)
        elif stripped.startswith("locked:"):
            lines.append("    locked: true")
        elif stripped.startswith("lr_source:"):
            lines.append(
                '    lr_source: "lr_sweep:02 (f1_macro_val=%.4f, dev split, uniform protocol)"'
                % f1_val
            )
        else:
            lines.append(line)
    updated = "\n".join(lines) + "\n"

    path.write_text(text[:start] + updated + text[end:], encoding="utf-8", newline="\n")
    print("[arms.yaml] %s lr set to %g" % (arm, lr))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data_cfg = load_data_config()
    arms_cfg = load_arms_config()
    sweep = arms_cfg["baseline_lr_sweep"]

    rule("02 -- baseline learning-rate sweep (DEVELOPMENT SPLIT, uniform protocol)")
    print("  arms   : %s" % sweep["arms"])
    print("  lr     : %s" % sweep["lr"])
    print("  batch  : %d (fixed, not swept)" % sweep["batch"])
    print("  epochs : %d" % sweep["epochs"])

    index = srp_data.load_image_index()
    dev_split = load_dev_split()
    print(
        "\n[dev split] train %d  val %d  test %d"
        % (len(dev_split["train"]), len(dev_split["val"]), len(dev_split["test"]))
    )

    specs = []
    for arm in sweep["arms"]:
        for lr in sweep["lr"]:
            spec = {
                "arm": arm,
                "architecture": arms_cfg["arms"][arm]["architecture"],
                "script": SCRIPT,
                "split_kind": "dev",
                "repeat": None,
                "fold": None,
                "epochs": int(sweep["epochs"]),
                "batch": int(sweep["batch"]),
                "lr": float(lr),
                "class_weights": arms_cfg["shared"]["class_weights"],
                "run_seed": DEV_SEED,
                "extra": "lr_sweep",
            }
            spec["run_id"] = registry.compute_run_id(**spec)
            specs.append(spec)

    todo, skipped = registry.plan_runs(specs)
    registry.print_plan(SCRIPT, todo, skipped)
    if args.dry_run:
        for spec in todo:
            print("  TODO %s  %-18s lr %g" % (spec["run_id"], spec["arm"], spec["lr"]))
        return 0

    if todo:
        data_root = resolve_data_root(data_cfg)
        labels_by_idx = labels_by_idx_map(index, data_cfg)
        cache = ImageCache(index, data_root, int(arms_cfg["shared"]["image_size"]))
        cache.warm(dev_split["train"] + dev_split["val"] + dev_split["test"])
        print("[cache] %d images letterboxed into RAM" % len(cache._cache))

    for position, spec in enumerate(todo, 1):
        rule("run %d/%d  %s  lr %g" % (position, len(todo), spec["arm"], spec["lr"]))
        bundle = build_model(
            spec["arm"], arms_cfg, data_cfg, with_efficiency=False, seed=DEV_SEED
        )
        cfg = TrainConfig.from_arm(
            spec["arm"],
            arms_cfg,
            epochs=spec["epochs"],
            batch=spec["batch"],
            lr=spec["lr"],
        ) if arms_cfg["arms"][spec["arm"]].get("lr") is not None else TrainConfig(
            epochs=spec["epochs"],
            batch=spec["batch"],
            lr=spec["lr"],
            optimizer=arms_cfg["arms"][spec["arm"]].get("optimizer", "SGD"),
            class_weights=arms_cfg["shared"]["class_weights"],
            image_size=int(arms_cfg["shared"]["image_size"]),
            num_classes=int(arms_cfg["shared"]["num_classes"]),
            **{
                k: v
                for k, v in arms_cfg.get("uniform_protocol", {}).items()
                if k in {"momentum", "weight_decay", "warmup_epochs", "warmup_momentum",
                         "lrf", "cos_lr", "num_workers", "amp"}
            },
        )

        started = time.perf_counter()
        result = train_fold(
            bundle,
            cache,
            dev_split["train"],
            dev_split["val"],
            labels_by_idx,
            cfg,
            seed=DEV_SEED,
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
        print(
            "  -> selected epoch %d/%d  val f1_macro %.4f  (dev-test f1_macro %.4f)  %.1fs"
            % (
                result.best_epoch + 1,
                cfg.epochs,
                val_metrics["f1_macro"],
                test_metrics["f1_macro"],
                wall,
            )
        )

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
                run_seed=DEV_SEED,
                val_seed=None,
                metrics=test_metrics,
                efficiency=efficiency,
                wall_time_s=wall,
                determinism_status=result.determinism,
                extra={
                    "protocol": "uniform",
                    "selected_epoch": result.best_epoch,
                    "selection_metric": "val_f1_macro",
                    "selection_tiebreak": "val_loss_unweighted",
                    "f1_macro_val": val_metrics["f1_macro"],
                    "best_val_f1": result.best_val_f1,
                    "best_val_loss": result.best_val_loss,
                    "min_val_loss_epoch": result.min_val_loss_epoch,
                    "history": result.history,
                    "val_metrics": val_metrics,
                },
            )
        )
        print("  [registry] appended %s" % spec["run_id"])

    # ---- winners ----
    rule("baseline lr winners (by development-split validation macro-F1)")
    rows = [
        {
            "arm": r["arm"],
            "lr": r["lr"],
            "f1_macro_val": r["extra"]["f1_macro_val"],
            "f1_macro_dev_test": r["f1_macro"],
            "selected_epoch": r["extra"]["selected_epoch"] + 1,
        }
        for r in registry.load_registry()
        if r.get("script") == SCRIPT
    ]
    if not rows:
        print("No runs found.")
        return 1
    table = pd.DataFrame(rows).drop_duplicates(["arm", "lr"]).sort_values(["arm", "lr"])
    out_csv = artifacts_dir(data_cfg) / "baseline_lr_sweep.csv"
    table.to_csv(out_csv, index=False, lineterminator="\n")

    print("  %-20s %8s %14s %18s %8s" % ("arm", "lr", "f1_macro_val", "f1_macro_dev_test", "epoch"))
    for row in table.to_dict("records"):
        print(
            "  %-20s %8.0e %14.4f %18.4f %8d"
            % (row["arm"], row["lr"], row["f1_macro_val"], row["f1_macro_dev_test"],
               row["selected_epoch"])
        )

    for arm in sweep["arms"]:
        subset = table[table["arm"] == arm]
        if subset.empty:
            continue
        winner = subset.loc[subset["f1_macro_val"].idxmax()]
        print(
            "\n  WINNER %-20s lr %.0e  (f1_macro_val %.4f)"
            % (arm, winner["lr"], winner["f1_macro_val"])
        )
        write_back_lr(arm, float(winner["lr"]), float(winner["f1_macro_val"]))

    print("\n[artifacts] wrote %s" % out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
