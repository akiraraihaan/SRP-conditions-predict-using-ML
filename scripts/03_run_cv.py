#!/usr/bin/env python
"""03 -- the main experiment: every arm, every fold. 5 arms x 15 folds = 75 runs.

Resumable. Every run is identified by a deterministic run_id; runs already in
artifacts/registry.jsonl are skipped, and the registry is flushed after each one,
so a session killed at run 40 loses nothing -- rerun the same command.

    python scripts/03_run_cv.py                     # all arms, all 15 folds
    python scripts/03_run_cv.py --arms yolo26n      # one arm
    python scripts/03_run_cv.py --arms yolo26n --repeat 0 --fold 0   # smoke test
    python scripts/03_run_cv.py --dry-run           # print the plan, run nothing

Seeds come from artifacts/folds.json and are identical across arms, so every
comparison is paired.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srpcard import data as srp_data  # noqa: E402
from srpcard import evaluate, registry  # noqa: E402
from srpcard import folds as srp_folds  # noqa: E402
from srpcard.config import artifacts_dir, load_arms_config, load_data_config, resolve_data_root  # noqa: E402
from srpcard.efficiency import profile  # noqa: E402
from srpcard.models import ARM_NAMES, add_fallback_argument, build_model  # noqa: E402
from srpcard.train import ImageCache, TrainConfig, labels_by_idx_map, train_fold  # noqa: E402

SCRIPT = "03_run_cv"


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=None, help="subset of arms (default: all)")
    parser.add_argument("--repeat", type=int, default=None, help="only this repeat")
    parser.add_argument("--fold", type=int, default=None, help="only this fold")
    parser.add_argument("--limit", type=int, default=None, help="stop after N runs")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs (smoke tests)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--quiet", action="store_true", help="suppress per-epoch lines")
    add_fallback_argument(parser)
    args = parser.parse_args()

    data_cfg = load_data_config()
    arms_cfg = load_arms_config()

    arms = args.arms or [a for a in arms_cfg["arms"]]
    unknown = [a for a in arms if a not in ARM_NAMES]
    if unknown:
        print("Unknown arm(s) %s. Known: %s" % (unknown, ARM_NAMES))
        return 2

    rule("03 -- cross-validated experiment")

    index = srp_data.load_image_index()
    index_path = artifacts_dir(data_cfg) / "image_index.csv"
    payload = srp_folds.load_folds(index, index_path=index_path)
    print("[folds] corpus fingerprint verified  OK  (n=%d)" % payload["corpus"]["n"])

    selected = [
        entry
        for entry in payload["folds"]
        if (args.repeat is None or entry["repeat"] == args.repeat)
        and (args.fold is None or entry["fold"] == args.fold)
    ]
    if not selected:
        print("No folds match --repeat/--fold.")
        return 2

    # ---- plan ----
    specs = []
    for arm in arms:
        arm_cfg = arms_cfg["arms"][arm]
        if arm_cfg.get("lr") is None:
            print(
                "[skip] arm %r has no learning rate yet (%s). Run scripts/02_lr_sweep_baselines.py."
                % (arm, arm_cfg.get("lr_source"))
            )
            continue
        epochs = args.epochs or int(arm_cfg["epochs"])
        for entry in selected:
            spec = {
                "arm": arm,
                "architecture": arm_cfg["architecture"],
                "script": SCRIPT,
                "split_kind": "cv",
                "repeat": entry["repeat"],
                "fold": entry["fold"],
                "epochs": epochs,
                "batch": int(arm_cfg["batch"]),
                "lr": float(arm_cfg["lr"]),
                "class_weights": arms_cfg["shared"]["class_weights"],
                "run_seed": entry["run_seed"],
                "val_seed": entry["val_seed"],
                "extra": None,
                "_entry": entry,
            }
            spec["run_id"] = registry.compute_run_id(**spec)
            specs.append(spec)

    todo, skipped = registry.plan_runs(specs)
    registry.print_plan(SCRIPT, todo, skipped)
    if args.limit:
        todo = todo[: args.limit]
        print("[registry] --limit %d: running %d of them now" % (args.limit, len(todo)))
    if args.dry_run:
        for spec in todo:
            print(
                "  TODO %s  %-18s r%df%d  ep%d bs%d lr%g  seed %d"
                % (
                    spec["run_id"],
                    spec["arm"],
                    spec["repeat"],
                    spec["fold"],
                    spec["epochs"],
                    spec["batch"],
                    spec["lr"],
                    spec["run_seed"],
                )
            )
        return 0
    if not todo:
        print("[registry] nothing to do.")
        return 0

    # ---- data ----
    data_root = resolve_data_root(data_cfg)
    labels_by_idx = labels_by_idx_map(index, data_cfg)
    cache = ImageCache(index, data_root, int(arms_cfg["shared"]["image_size"]))
    warm_start = time.perf_counter()
    cache.warm(srp_data.clean_index(index)["idx"].tolist())
    print(
        "[cache] letterboxed %d images into RAM in %.1fs"
        % (len(cache._cache), time.perf_counter() - warm_start)
    )

    # ---- run ----
    completed = 0
    for position, spec in enumerate(todo, 1):
        entry = spec.pop("_entry")
        rule(
            "run %d/%d  %s  repeat %d fold %d  (run_id %s)"
            % (position, len(todo), spec["arm"], spec["repeat"], spec["fold"], spec["run_id"])
        )
        print(
            "  epochs %d  batch %d  lr %g  class_weights %s  run_seed %d  val_seed %d"
            % (
                spec["epochs"],
                spec["batch"],
                spec["lr"],
                spec["class_weights"],
                spec["run_seed"],
                spec["val_seed"],
            )
        )
        print(
            "  train %d  val %d  test %d"
            % (len(entry["train_idx"]), len(entry["val_idx"]), len(entry["test_idx"]))
        )

        bundle = build_model(
            spec["arm"],
            arms_cfg,
            data_cfg,
            with_efficiency=False,
            seed=spec["run_seed"],
            allow_pretrained_fallback=args.allow_pretrained_fallback,
        )
        print(
            "  checkpoint %s%s"
            % (
                bundle.checkpoint_resolved,
                "  [FALLBACK -- NOT %s]" % spec["architecture"]
                if bundle.pretrained_fallback_used
                else "",
            )
        )
        cfg = TrainConfig.from_arm(
            spec["arm"], arms_cfg, epochs=spec["epochs"], class_weights=spec["class_weights"]
        )
        started = time.perf_counter()
        result = train_fold(
            bundle,
            cache,
            entry["train_idx"],
            entry["val_idx"],
            labels_by_idx,
            cfg,
            seed=spec["run_seed"],
            device=args.device,
            verbose=not args.quiet,
        )
        metrics = evaluate.evaluate_fold(
            bundle.module, cache, entry["test_idx"], labels_by_idx, data_cfg
        )
        efficiency = profile(bundle.module, cfg.image_size, latency=True)
        wall = round(time.perf_counter() - started, 2)

        print(
            "  -> selected epoch %d/%d (val_f1 %.4f) | test f1_macro %.4f  acc %.4f  | %.1fs on %s"
            % (
                result.best_epoch + 1,
                spec["epochs"],
                result.best_val_f1,
                metrics["f1_macro"],
                metrics["accuracy"],
                wall,
                result.device,
            )
        )

        record = registry.build_record(
            run_id=spec["run_id"],
            script=SCRIPT,
            arm=spec["arm"],
            architecture=spec["architecture"],
            split_kind="cv",
            repeat=spec["repeat"],
            fold=spec["fold"],
            epochs=spec["epochs"],
            batch=spec["batch"],
            lr=spec["lr"],
            class_weights=spec["class_weights"],
            run_seed=spec["run_seed"],
            val_seed=spec["val_seed"],
            checkpoint_resolved=bundle.checkpoint_resolved,
            pretrained_fallback_used=bundle.pretrained_fallback_used,
            metrics=metrics,
            efficiency=efficiency,
            wall_time_s=wall,
            determinism_status=result.determinism,
            extra={
                "selected_epoch": result.best_epoch,
                "best_val_f1": result.best_val_f1,
                "best_val_loss": result.best_val_loss,
                "stopped_early": result.stopped_early,
                "epochs_run": len(result.history),
                "selection_metric": "val_f1_macro",
                "selection_tiebreak": "val_loss_unweighted",
                # what a val-loss criterion WOULD have picked -- recorded, not acted on
                "min_val_loss_epoch": result.min_val_loss_epoch,
                "min_val_loss": result.min_val_loss,
                "history": result.history,
                "class_weight_values": result.class_weights,
                "device": result.device,
                "model_notes": bundle.notes,
            },
        )
        registry.append_record(record)
        completed += 1
        print("  [registry] appended %s" % spec["run_id"])

    rule("DONE")
    summary = registry.summarise()
    print("[registry] %d record(s) total: %s" % (summary["n_records"], summary["by_arm"]))
    print("[registry] completed %d run(s) this session" % completed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
