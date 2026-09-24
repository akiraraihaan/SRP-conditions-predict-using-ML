#!/usr/bin/env python
"""03c -- YOLO26 trained by Ultralytics' own recipe, scored by our metric code.

    python scripts/03c_native_recipe.py --data-root /content/dataset
    python scripts/03c_native_recipe.py --data-root ... --dry-run
    python scripts/03c_native_recipe.py --data-root ... --work-dir /content/tmp

NEEDS A GPU. Five folds (repeat 0) of yolo26n-cls, roughly 10-15 minutes on a
T4. Records under script `03c_native_recipe` with `extra.protocol = "native"`,
so the hyperparameter-drift guard keeps these out of every uniform-protocol
comparison.

THE QUESTION
------------
The paper's central claim is negative: YOLO26-cls loses to MobileNetV3-Small
and ResNet18. A reviewer will ask whether YOLO26 was handicapped by being
pulled out of its own training recipe. This answers that by removing the
handicap entirely and seeing whether the result survives.

WHAT IS HELD COMMON, AND WHAT IS NOT
------------------------------------
Common, and only these three things:

  * the FOLD PARTITION -- the same train/val/test indices from folds.json
  * the CLASS-INDEX MAPPING -- resolved through the trained model's own
    `names`, never assumed to match ours by sort order
  * the METRIC CODE -- our macro-F1, over our test partition

Deliberately NOT common: the preprocessing. This is the correction that makes
the check worth running at all. If the native path trains under Ultralytics'
resize-and-crop and is then evaluated through OUR letterbox, the model sees a
transform it never trained on, scores badly, and the check MANUFACTURES the
very result it exists to test. So the native arm trains with their recipe AND
is evaluated with their own inference preprocessing, through `model.predict`.

Every row this produces is therefore labelled `preprocessing = ultralytics_default`
against `letterbox_224` for the uniform arms. The table is NOT an
apples-to-apples preprocessing comparison and must never be read as one.

THE VALIDATION SLICE is the same 10 % our loop carves out, with the same
`val_seed`, so checkpoint selection runs on the same images. Without that, the
comparison would silently include a different validation split on top of the
recipe difference.

CHECKPOINT SELECTION still differs, and is REPORTED rather than hidden:
Ultralytics selects `best.pt` by its own fitness metric (top-1/top-5 accuracy
for classification), while our loop selects on validation macro-F1. Both are
recorded per fold so the difference is visible in the output instead of being
folded silently into "the recipe".

DISK. The Ultralytics trainer wants a directory layout, `root/split/class/*`.
It is built under --work-dir, which defaults to a temporary directory on LOCAL
disk, never under --data-root, and is removed afterwards. Reading images
straight off a Drive FUSE mount is not merely a symlink problem: caching 668
letterboxed images from Drive took 290 seconds in an earlier session, and this
path reads them repeatedly. Copy the dataset to local disk first; --data-root
should already point there by the time this script runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srpcard import folds as srp_folds  # noqa: E402
from srpcard import registry  # noqa: E402
from srpcard.config import (  # noqa: E402
    artifacts_dir,
    load_arms_config,
    load_data_config,
    resolve_data_root,
)

SCRIPT = "03c_native_recipe"
PROTOCOL = "native"
PREPROCESSING = "ultralytics_default"

# What the uniform arms use, for the column that stops this table being read as
# an apples-to-apples preprocessing comparison.
UNIFORM_PREPROCESSING = "letterbox_224"

DEFAULT_ARM = "yolo26n"
DEFAULT_REPEAT = 0


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


# --------------------------------------------------------------------------
# the fold tree
# --------------------------------------------------------------------------


def assert_outside_data_root(work_dir: Path, data_root: Path) -> None:
    """The dataset is an input. Nothing here may write into it.

    Checked by resolved path, not by string prefix, because a symlink or a
    relative --work-dir could land inside it while looking as if it does not.
    """
    work = work_dir.resolve()
    root = data_root.resolve()
    if work == root or root in work.parents:
        raise SystemExit(
            "--work-dir is inside --data-root.\n"
            "  work-dir  : %s\n"
            "  data-root : %s\n"
            "\n"
            "  The dataset is an input and this script writes a whole directory\n"
            "  tree per fold. Point --work-dir at local scratch space." % (work, root)
        )


def build_fold_tree(
    work_dir: Path,
    entry: dict,
    index_rows: dict[int, dict],
    data_root: Path,
    classes: list[str],
) -> Path:
    """`root/train/<class>/*` and `root/val/<class>/*` for one fold.

    COPIES rather than symlinks. Colab's Drive mount does not support symlinks
    reliably, and a tree that half-works produces a training set quietly missing
    images -- which looks like a worse recipe rather than a broken one.

    The test partition is NOT written. Ultralytics never sees it: evaluation is
    done afterwards, by us, on our own indices.
    """
    root = work_dir / ("r%df%d" % (entry["repeat"], entry["fold"]))
    if root.exists():
        shutil.rmtree(root)

    for split, indices in (("train", entry["train_idx"]), ("val", entry["val_idx"])):
        for name in classes:
            (root / split / name).mkdir(parents=True, exist_ok=True)
        for idx in indices:
            row = index_rows[idx]
            source = data_root / row["relpath"]
            target = root / split / row["class"] / ("%06d%s" % (idx, source.suffix))
            shutil.copyfile(source, target)

    written = sum(1 for _ in root.rglob("*") if _.is_file())
    expected = len(entry["train_idx"]) + len(entry["val_idx"])
    if written != expected:
        raise SystemExit(
            "Fold tree has %d files, expected %d. A partial tree trains on a "
            "silently smaller set and reads as a worse recipe." % (written, expected)
        )
    return root


# --------------------------------------------------------------------------
# native evaluation
# --------------------------------------------------------------------------


def native_predictions(model, paths: list[Path], classes: list[str]) -> list[int]:
    """Predicted class INDEX per image, in OUR class order.

    The mapping goes through the trained model's own `names`, never by assuming
    its sort order matches ours. Ultralytics builds its class list from the
    directory names it found; if that ever diverged from `class_order`, an
    assumed mapping would silently score every prediction against the wrong
    label and report it as a recipe difference.

    Preprocessing is THEIRS, on purpose. `model.predict` applies the same
    transform the model was trained under.
    """
    names = model.names
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    unknown = set(names) - set(classes)
    if unknown:
        raise SystemExit(
            "The trained model has class names we do not know: %s\n"
            "  ours: %s\n"
            "  A mapping guessed here would score every prediction against the "
            "wrong label." % (sorted(unknown), classes)
        )
    to_ours = {position: classes.index(name) for position, name in enumerate(names)}

    predicted = []
    for path in paths:
        result = model.predict(source=str(path), verbose=False)[0]
        predicted.append(to_ours[int(result.probs.top1)])
    return predicted


# What "their recipe" concretely consisted of. Read back from the trainer that
# ran, never quoted from documentation: ultralytics resolves `optimizer: auto`
# and several augmentation strengths at runtime, so the defaults in the docs are
# not necessarily the values that were in force.
AUGMENTATION_KEYS = (
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix",
    "copy_paste", "auto_augment", "erasing", "crop_fraction",
)

SCHEDULE_KEYS = (
    "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
    "warmup_momentum", "warmup_bias_lr", "cos_lr", "epochs", "batch", "imgsz",
    "patience", "amp", "seed", "single_cls", "dropout", "label_smoothing",
)


def recipe_report(trainer) -> dict:
    """The augmentations, schedule and optimizer THAT RAN.

    Anything not readable from the trainer is listed by name under
    `unreadable_keys` rather than filled in from documentation. A default that
    was not in force is worse than a gap: it reads as a measurement.
    """
    args = getattr(trainer, "args", None)
    resolved = dict(vars(args)) if args is not None else {}

    augmentation, schedule, unreadable = {}, {}, []
    for key in AUGMENTATION_KEYS:
        if key in resolved:
            augmentation[key] = resolved[key]
        else:
            unreadable.append("augmentation.%s" % key)
    for key in SCHEDULE_KEYS:
        if key in resolved:
            schedule[key] = resolved[key]
        else:
            unreadable.append("schedule.%s" % key)

    # The optimizer OBJECT, not the request. `optimizer: auto` resolves at
    # runtime, so the string in args is not necessarily what stepped the
    # weights -- the same distinction that hid MuSGD == SGD in our own loop.
    optimizer = getattr(trainer, "optimizer", None)
    optimizer_used = type(optimizer).__name__ if optimizer is not None else None
    if optimizer is None:
        unreadable.append("optimizer_object")

    groups = []
    if optimizer is not None:
        for group in optimizer.param_groups:
            groups.append({
                key: group.get(key)
                for key in ("lr", "momentum", "weight_decay", "nesterov")
                if key in group
            })

    active = {k: v for k, v in augmentation.items()
              if v not in (0, 0.0, False, None, "")}
    return {
        "augmentation": augmentation,
        "augmentation_active": active,
        "augmentation_disabled": sorted(set(augmentation) - set(active)),
        "schedule": schedule,
        "optimizer_requested": resolved.get("optimizer"),
        "optimizer_used": optimizer_used,
        "optimizer_param_groups": groups,
        "unreadable_keys": unreadable,
        "read_back_from": "ultralytics trainer instance, not documentation",
    }


def selection_report(run_dir: Path) -> dict:
    """Which checkpoint Ultralytics kept, and on what criterion.

    Their trainer selects `best.pt` by its own fitness, which for classification
    is accuracy-based. Our loop selects on validation macro-F1. That is a second
    difference on top of the recipe, and it is reported rather than absorbed
    into it.
    """
    report = {
        "selection_criterion_native": "ultralytics fitness (top-1/top-5 accuracy)",
        "selection_criterion_ours": "validation macro-F1, tie-break validation loss",
        "criteria_differ": True,
        "best_checkpoint": None,
        "selected_epoch_native": None,
        "epochs_run_native": None,
    }
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        report["best_checkpoint"] = str(best)

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        import pandas as pd

        frame = pd.read_csv(results_csv)
        frame.columns = [c.strip() for c in frame.columns]
        report["epochs_run_native"] = int(len(frame))
        for column in ("metrics/accuracy_top1", "metrics/accuracy_top5"):
            if column in frame.columns:
                report["selected_epoch_native"] = int(frame[column].idxmax()) + 1
                report["selection_metric_column"] = column
                break
    return report


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None,
                        help="the image corpus, ON LOCAL DISK (not a Drive mount)")
    parser.add_argument("--work-dir", default=None,
                        help="where the per-fold trees are built. Defaults to a "
                             "temporary directory on local disk; never inside "
                             "--data-root")
    parser.add_argument("--arm", default=DEFAULT_ARM,
                        help="the arm whose architecture and checkpoint to use")
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT,
                        help="which repeat's five folds (default: 0)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the epoch budget (default: the arm's)")
    parser.add_argument("--keep-trees", action="store_true",
                        help="leave the per-fold directories in place for inspection")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit, training nothing")
    args = parser.parse_args()

    rule("03c -- native Ultralytics recipe, our metric, our folds")

    data_cfg = load_data_config()
    arms_cfg = load_arms_config()
    artifacts = artifacts_dir(data_cfg)

    if args.arm not in arms_cfg["arms"]:
        raise SystemExit("Unknown arm %r. Known: %s"
                         % (args.arm, ", ".join(sorted(arms_cfg["arms"]))))
    arm_cfg = arms_cfg["arms"][args.arm]
    if arm_cfg.get("backend") != "ultralytics":
        raise SystemExit(
            "Arm %r has backend %r. The native recipe only exists for "
            "ultralytics arms -- there is no 'their own trainer' for a "
            "torchvision model." % (args.arm, arm_cfg.get("backend"))
        )

    import pandas as pd

    index = pd.read_csv(artifacts / "image_index.csv", comment="#")
    # verify=True asserts the corpus fingerprint, so a fold file built over a
    # different corpus cannot be used here by accident.
    bundle = srp_folds.load_folds(index, path=artifacts / "folds.json")
    entries = [e for e in bundle["folds"] if e["repeat"] == args.repeat]
    if not entries:
        raise SystemExit("No folds with repeat %d." % args.repeat)

    index_rows = {int(r["idx"]): r for _, r in index.iterrows()}
    # The CANONICAL order from configs/data.yaml, which is what every recorded
    # confusion_matrix and class_order is in. Re-deriving it by sorting the
    # index would agree today and diverge the moment a class is renamed.
    classes = list(data_cfg["classes"])
    epochs = args.epochs or int(arm_cfg["epochs"])

    print("  arm          : %s (%s)" % (args.arm, arm_cfg["architecture"]))
    print("  folds        : repeat %d, folds %s"
          % (args.repeat, ", ".join(str(e["fold"]) for e in entries)))
    print("  epochs       : %d" % epochs)
    print("  classes      : %d" % len(classes))
    print("  protocol     : %s   preprocessing: %s" % (PROTOCOL, PREPROCESSING))
    print("  common       : fold partition, class-index mapping, metric code")
    print("  NOT common   : preprocessing, augmentation, schedule, optimizer,")
    print("                 checkpoint-selection criterion (all reported)")

    if args.dry_run:
        print("\n  --dry-run: %d fold(s) planned. Nothing trained, nothing written."
              % len(entries))
        return 0

    data_root = Path(args.data_root) if args.data_root else resolve_data_root(data_cfg)
    if not data_root.is_dir():
        raise SystemExit("--data-root is not a directory: %s" % data_root)

    temporary = args.work_dir is None
    work_dir = Path(args.work_dir) if args.work_dir else Path(
        tempfile.mkdtemp(prefix="srpcard_native_")
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    assert_outside_data_root(work_dir, data_root)
    print("  data root    : %s" % data_root)
    print("  work dir     : %s%s" % (work_dir, "  (temporary)" if temporary else ""))

    from ultralytics import YOLO

    from srpcard import evaluate
    from srpcard.train import labels_by_idx_map

    labels_by_idx = labels_by_idx_map(index, data_cfg)
    corpus_fp = bundle.get("corpus", {})

    specs = []
    for entry in entries:
        spec = {
            "arm": args.arm,
            "architecture": arm_cfg["architecture"],
            "script": SCRIPT,
            "split_kind": "cv",
            "repeat": entry["repeat"],
            "fold": entry["fold"],
            "epochs": epochs,
            "batch": int(arm_cfg["batch"]),
            "lr": float(arm_cfg["lr"]),
            "class_weights": "native_ultralytics_default",
            "run_seed": entry["run_seed"],
            "optimizer": None,          # theirs, not an override of ours
            "extra": "native_recipe",
            "_entry": entry,
        }
        spec["run_id"] = registry.compute_run_id(**spec)
        specs.append(spec)

    todo, skipped = registry.plan_runs(specs)
    registry.print_plan(SCRIPT, todo, skipped)
    if not todo:
        print("  Nothing to run.")
        return 0

    try:
        for spec in todo:
            entry = spec["_entry"]
            rule("r%df%d" % (entry["repeat"], entry["fold"]))

            root = build_fold_tree(work_dir, entry, index_rows, data_root, classes)
            print("  tree: %d train, %d val -> %s"
                  % (len(entry["train_idx"]), len(entry["val_idx"]), root))

            started = time.perf_counter()
            model = YOLO("%s.pt" % arm_cfg["architecture"])
            model.train(
                data=str(root),
                epochs=epochs,
                batch=spec["batch"],
                imgsz=int(arms_cfg["shared"]["image_size"]),
                seed=spec["run_seed"],
                project=str(work_dir / "runs"),
                name="r%df%d" % (entry["repeat"], entry["fold"]),
                exist_ok=True,
                verbose=False,
                # Everything else deliberately left at THEIR defaults: their
                # augmentation, their schedule, their optimizer, their warmup.
                # That is the entire point of this run.
            )
            wall = round(time.perf_counter() - started, 2)

            run_dir = Path(model.trainer.save_dir)
            selection = selection_report(run_dir)
            recipe = recipe_report(model.trainer)
            best = run_dir / "weights" / "best.pt"
            scored = YOLO(str(best)) if best.exists() else model

            test_paths = [data_root / index_rows[i]["relpath"] for i in entry["test_idx"]]
            truth = [labels_by_idx[i] for i in entry["test_idx"]]
            predicted = native_predictions(scored, test_paths, classes)
            metrics = evaluate.metrics_from_predictions(truth, predicted, data_cfg)

            print("  -> test f1_macro %.4f  acc %.4f  (%.1fs)"
                  % (metrics["f1_macro"], metrics["accuracy"], wall))
            print("     their best epoch %s of %s by %s"
                  % (selection.get("selected_epoch_native"),
                     selection.get("epochs_run_native"),
                     selection.get("selection_metric_column", "fitness")))
            print("     optimizer: requested %r -> %s ran"
                  % (recipe["optimizer_requested"], recipe["optimizer_used"]))
            active = recipe["augmentation_active"]
            print("     augmentation ON  : %s"
                  % (", ".join("%s=%s" % kv for kv in sorted(active.items())) or "none"))
            print("     augmentation OFF : %s"
                  % (", ".join(recipe["augmentation_disabled"]) or "none"))
            if recipe["unreadable_keys"]:
                print("     NOT READABLE from the trainer (not guessed at): %s"
                      % ", ".join(recipe["unreadable_keys"]))

            registry.append_record(
                registry.build_record(
                    run_id=spec["run_id"],
                    script=SCRIPT,
                    arm=spec["arm"],
                    architecture=spec["architecture"],
                    split_kind="cv",
                    repeat=entry["repeat"],
                    fold=entry["fold"],
                    epochs=epochs,
                    batch=spec["batch"],
                    lr=spec["lr"],
                    class_weights=spec["class_weights"],
                    run_seed=spec["run_seed"],
                    val_seed=entry["val_seed"],
                    checkpoint_resolved="%s.pt" % arm_cfg["architecture"],
                    pretrained_fallback_used=False,
                    class_weights_verified=None,
                    class_weights_proof={
                        "not_applicable": True,
                        "reason": "the native recipe uses ultralytics' own loss "
                                  "and sampling; our balanced weights are "
                                  "deliberately not applied, because applying "
                                  "them would make this no longer their recipe",
                    },
                    corpus_fingerprint=corpus_fp,
                    training=registry.training_outcome_absent(
                        "trained by ultralytics model.train() with its own "
                        "defaults; per-epoch history and best-epoch selection "
                        "are theirs and are not comparable with scripts 02-05",
                        epochs_run=selection.get("epochs_run_native"),
                    ),
                    metrics=metrics,
                    efficiency={},
                    wall_time_s=wall,
                    run_id_extra=spec["extra"],
                    # Theirs, not ours -- reported from the trainer below.
                    optimizer_used="ultralytics_default (native recipe)",
                    extra={
                        "protocol": PROTOCOL,
                        "preprocessing": PREPROCESSING,
                        "preprocessing_note": (
                            "trained AND evaluated under ultralytics' own "
                            "transform. Evaluating this model through our "
                            "letterbox would show it a transform it never "
                            "trained on and manufacture the result this run "
                            "exists to test."
                        ),
                        "held_common": ["fold_partition", "class_index_mapping",
                                        "metric_code"],
                        "not_held_common": ["preprocessing", "augmentation",
                                            "schedule", "optimizer",
                                            "checkpoint_selection"],
                        # The recipe THAT RAN, for the manuscript to describe
                        # without quoting documentation at it.
                        "native_recipe": recipe,
                        **selection,
                    },
                )
            )
            if not args.keep_trees:
                shutil.rmtree(root, ignore_errors=True)
    finally:
        if temporary and not args.keep_trees:
            shutil.rmtree(work_dir, ignore_errors=True)
            print("\n[cleanup] removed %s" % work_dir)
        elif args.keep_trees:
            print("\n[cleanup] --keep-trees: left %s in place" % work_dir)

    rule("DONE")
    print("  %d record(s) under script %s, protocol %s." % (len(todo), SCRIPT, PROTOCOL))
    print("  Compare with scripts/11_recipe_check.py, which reports the 2x2 and")
    print("  labels this row %s against %s for the uniform arms."
          % (PREPROCESSING, UNIFORM_PREPROCESSING))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
