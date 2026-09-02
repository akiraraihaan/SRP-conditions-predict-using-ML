#!/usr/bin/env python
"""01 -- complete the yolo26m grid: run the 8 configurations that are missing.

    python scripts/01_complete_medium_grid.py
    python scripts/01_complete_medium_grid.py --dry-run

=============================================================================
TWO CONSTRAINTS THAT MUST NOT DRIFT. THIS SCRIPT IS DELIBERATELY DIFFERENT
FROM EVERY OTHER SCRIPT IN THE REPOSITORY.
=============================================================================

1. IT RUNS ON THE DEVELOPMENT SPLIT (artifacts/dev_split.json), the recovered
   legacy 80:10:10 partition over the RAW 695 images WITH THEIR ORIGINAL,
   CONTAMINATED LABELS. Not the folds. Not the clean 668 corpus. The
   development split exists for exactly this purpose and no other.

2. IT REPLICATES THE LEGACY PROTOCOL, INCLUDING THE LEGACY BUG: class weights
   are computed and then NOT applied to the loss, because that is what the
   original pipeline did (MIGRATION_NOTES.md section 5.4). It trains through
   ultralytics `model.train()` with the original arguments and ultralytics'
   own augmentation and best-weight selection -- NOT the uniform protocol in
   src/srpcard/train.py.

Why: the 8 runs here must be commensurable with the 46 grid-search runs that
are already complete. Those 46 were trained on contaminated data, with an
unweighted loss, under ultralytics' defaults, and they cannot be re-run. A
configuration trained under the new, better protocol cannot be compared with
them on validation macro-F1 -- the comparison would measure the protocol
change, not the hyperparameters.

THIS IS THE ONLY SCRIPT IN THE REPOSITORY THAT DELIBERATELY REPRODUCES THE
LEGACY BUG. Every number it produces belongs to the legacy, unweighted
protocol and must be labelled as such wherever it is reported. Do not copy
this file as a template for anything else.
=============================================================================

Output: the recomputed yolo26m winner, written back into configs/arms.yaml.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from srpcard import data as srp_data  # noqa: E402
from srpcard import folds as srp_folds  # noqa: E402
from srpcard import registry  # noqa: E402
from srpcard.config import (  # noqa: E402
    MissingInputError,
    artifacts_dir,
    snapshot_arms,
    load_arms_config,
    load_data_config,
    require_file,
    resolve_data_root,
    runs_dir,
    set_seed,
)
from srpcard.legacy_split import load_dev_split  # noqa: E402
from srpcard.models import (  # noqa: E402
    _resolve_pretrained,
    add_fallback_argument,
    assert_checkpoint_matches_architecture,
    warn_fallback_banner,
)

SCRIPT = "01_complete_medium_grid"
LEGACY_SEED = 42
LEGACY_IMG_SIZE = 224
LEGACY_PATIENCE = 20
LEGACY_OPTIMIZER = "MuSGD"
ARM = "yolo26m"
VARIANT = "m"


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def config_key(epochs: int, batch: int, lr: float) -> str:
    return "%s_ep%d_bs%d_lr%.0e" % (VARIANT, epochs, batch, lr)


def materialise_legacy_dataset(index, dev_split, data_root: Path, out_root: Path) -> Path:
    """Write the development split out as letterboxed directory trees.

    Exactly what code.ipynb cell 5 ("[Cell 6]") did: letterbox each image to
    224x224 and copy it into <out>/<split>/<class>/. Ultralytics classification
    training needs a directory layout, so this materialisation is unavoidable.

    Class directory names are the CANONICAL normalised ones. The legacy run used
    the prefixed names; that difference is what broke its evaluation, and it is
    corrected here because it never affected training -- only the string
    comparison at evaluation time (MIGRATION_NOTES.md section 3).
    """
    if out_root.exists():
        shutil.rmtree(out_root)
    relpath = dict(zip(index["idx"].tolist(), index["relpath"].tolist()))
    class_of = dict(zip(index["idx"].tolist(), index["class"].tolist()))

    for split_name, idxs in dev_split.items():
        for idx in idxs:
            destination = out_root / split_name / class_of[idx]
            destination.mkdir(parents=True, exist_ok=True)
            image = srp_data.load_letterboxed(data_root / relpath[idx], LEGACY_IMG_SIZE)
            image.save(destination / ("%06d.png" % idx))
    return out_root


def evaluate_legacy(weights: Path, split_dir: Path, classes: list[str]) -> dict:
    """Blind evaluation in the legacy style, but with the name bug fixed.

    Names are normalised on both sides before comparison, and the matrix is
    always 10x10 in canonical order -- the fix cell 20 ("[Cell 19c]") applied
    after the fact.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    from ultralytics import YOLO

    paths, y_true = [], []
    for class_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
        name = srp_data.normalize_class_name(class_dir.name)
        for path in sorted(p for p in class_dir.iterdir() if p.is_file()):
            paths.append(str(path))
            y_true.append(classes.index(name))

    model = YOLO(str(weights))
    names = {
        int(k): str(v)
        for k, v in (model.names.items() if isinstance(model.names, dict) else enumerate(model.names))
    }
    results = model.predict(source=paths, imgsz=LEGACY_IMG_SIZE, verbose=False, stream=False)
    y_pred = [
        classes.index(srp_data.normalize_class_name(names[int(r.probs.top1)])) for r in results
    ]

    labels = list(range(len(classes)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_per_class": {c: float(v) for c, v in zip(classes, per_class[2])},
        "recall_per_class": {c: float(v) for c, v in zip(classes, per_class[1])},
        "precision_per_class": {c: float(v) for c, v in zip(classes, per_class[0])},
        "support_per_class": {c: int(v) for c, v in zip(classes, per_class[3])},
        "confusion_matrix": matrix.tolist(),
        "class_order": classes,
        "n_images": len(paths),
    }


def write_back_winner(epochs: int, batch: int, lr: float, key: str, f1_val: float) -> None:
    """Replace the provisional yolo26m block in configs/arms.yaml, in place."""
    path = Path(__file__).resolve().parents[1] / "configs" / "arms.yaml"
    text = path.read_text(encoding="utf-8")
    original = text

    start = text.index("  yolo26m:")
    end = text.index("  mobilenetv3_small:")
    block = text[start:end]

    updated = block
    for field, value in (("epochs", str(epochs)), ("batch", str(batch)), ("lr", repr(lr))):
        lines = []
        for line in updated.splitlines():
            stripped = line.strip()
            if stripped.startswith("%s:" % field) and line.startswith("    "):
                lines.append("    %s: %s" % (field, value))
            else:
                lines.append(line)
        updated = "\n".join(lines) + ("\n" if updated.endswith("\n") else "")

    lines = []
    for line in updated.splitlines():
        stripped = line.strip()
        if stripped.startswith("locked:"):
            lines.append("    locked: true")
        elif stripped.startswith("provisional:"):
            lines.append("    provisional: false")
        elif stripped.startswith("lr_source:"):
            lines.append(
                '    lr_source: "grid_search:%s (f1_macro_val=%.4f, 18/18 grid, '
                'legacy unweighted protocol, written by 01_complete_medium_grid.py)"'
                % (key, f1_val)
            )
        else:
            lines.append(line)
    updated = "\n".join(lines) + "\n"

    path.write_text(original[:start] + updated + original[end:], encoding="utf-8", newline="\n")
    print("[arms.yaml] yolo26m updated: epochs=%d batch=%d lr=%g" % (epochs, batch, lr))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: ultralytics auto)")
    parser.add_argument("--keep-dataset", action="store_true", help="do not delete the temp tree")
    add_fallback_argument(parser)
    args = parser.parse_args()

    data_cfg = load_data_config()
    arms_cfg = load_arms_config()
    classes = list(data_cfg["classes"])

    rule("01 -- complete the yolo26m grid (LEGACY PROTOCOL, DEVELOPMENT SPLIT)")
    print("  corpus         : raw 695, ORIGINAL contaminated labels")
    print("  split          : artifacts/dev_split.json (legacy 80:10:10)")
    print("  class weights  : NOT APPLIED -- replicating the legacy bug on purpose")
    print("  trainer        : ultralytics model.train(), optimizer %s" % LEGACY_OPTIMIZER)
    print("  selection      : ultralytics' own best.pt, NOT the uniform criterion")
    print("  -> every number below belongs to the LEGACY UNWEIGHTED PROTOCOL")

    index = srp_data.load_image_index()
    index_path = artifacts_dir(data_cfg) / "image_index.csv"
    # The corpus THIS script runs on: the raw 695 with original labels, not the
    # clean 668 the CV folds use. Recorded on every record so each result carries
    # the corpus it was produced on.
    corpus_fp = srp_folds.dev_corpus_fingerprint(index, index_path)
    print(
        "[corpus] %s  n=%d  sha1=%s"
        % (corpus_fp["kind"], corpus_fp["n"], corpus_fp["sha1_of_sorted_included_sha1s"])
    )
    dev_split = load_dev_split()
    print(
        "\n[dev split] train %d  val %d  test %d"
        % (len(dev_split["train"]), len(dev_split["val"]), len(dev_split["test"]))
    )

    legacy_metrics_path = artifacts_dir(data_cfg) / "legacy_grid_metrics.csv"
    require_file(
        legacy_metrics_path,
        produced_by="extracted from FINAL-pipeline/results/metrics_canonical.csv; committed",
    )
    legacy = pd.read_csv(legacy_metrics_path)
    completed = legacy[legacy["variant"] == VARIANT].copy()
    print(
        "[legacy] %d of 18 medium configurations already complete (from %s)"
        % (len(completed), legacy_metrics_path.name)
    )

    grid = arms_cfg["medium_grid"]
    all_configs = [
        (e, b, lr) for e in grid["epochs"] for b in grid["batch"] for lr in grid["lr"]
    ]
    done_keys = set(completed["key"])
    missing = [(e, b, lr) for e, b, lr in all_configs if config_key(e, b, lr) not in done_keys]
    print("[grid] %d total, %d done, %d missing" % (len(all_configs), len(done_keys), len(missing)))
    for e, b, lr in missing:
        print("         MISSING %s" % config_key(e, b, lr))

    expected_missing = set(grid["missing"])
    if {config_key(e, b, lr) for e, b, lr in missing} != expected_missing:
        raise SystemExit(
            "Missing set %s does not match configs/arms.yaml:medium_grid.missing %s"
            % (sorted(config_key(e, b, lr) for e, b, lr in missing), sorted(expected_missing))
        )

    specs = []
    for epochs, batch, lr in missing:
        spec = {
            "arm": ARM,
            "architecture": arms_cfg["arms"][ARM]["architecture"],
            "script": SCRIPT,
            "split_kind": "dev",
            "repeat": None,
            "fold": None,
            "epochs": epochs,
            "batch": batch,
            "lr": float(lr),
            "class_weights": "none_legacy_bug",
            "run_seed": LEGACY_SEED,
            "extra": "legacy_protocol",
        }
        spec["run_id"] = registry.compute_run_id(**spec)
        spec["key"] = config_key(epochs, batch, lr)
        specs.append(spec)

    todo, skipped = registry.plan_runs(specs)
    registry.print_plan(SCRIPT, todo, skipped)
    if args.dry_run:
        for spec in todo:
            print("  TODO %s  %s" % (spec["run_id"], spec["key"]))
        return 0

    dataset_root = runs_dir(data_cfg) / "legacy_dev_dataset"
    if todo:
        data_root = resolve_data_root(data_cfg)
        print("\n[dataset] materialising the development split (letterboxed) ...")
        materialise_legacy_dataset(index, dev_split, data_root, dataset_root)
        print("[dataset] %s" % dataset_root)

    from ultralytics import YOLO

    for position, spec in enumerate(todo, 1):
        rule("run %d/%d  %s" % (position, len(todo), spec["key"]))
        set_seed(LEGACY_SEED)
        run_name = "yolo26_%s_cls_ep%d_bs%d_lr%.0e" % (
            VARIANT,
            spec["epochs"],
            spec["batch"],
            spec["lr"],
        )
        architecture = spec["architecture"]
        checkpoint_filename = "%s.pt" % architecture
        weights_name = _resolve_pretrained(checkpoint_filename, data_cfg)
        fallback_used = False
        started = time.perf_counter()

        # Same guard as src/srpcard/models.py: this script loads the checkpoint
        # directly rather than through build_model, so the check is repeated here
        # instead of being inherited. A YOLO11 checkpoint standing in for a YOLO26
        # arm would change the architecture behind every number in the grid table.
        try:
            model = YOLO(weights_name)
        except Exception as exc:  # noqa: BLE001
            fallback = arms_cfg["arms"][ARM].get("pretrained_fallback")
            if not fallback or not args.allow_pretrained_fallback:
                raise MissingInputError(
                    "Pretrained checkpoint %r for arm %r could not be loaded: %s\n"
                    "  declared architecture : %s\n"
                    "  configured fallback   : %s\n"
                    "  Refusing to fall back: it would silently change the\n"
                    "  architecture of every configuration in the medium grid.\n"
                    "  Make %s available, or pass --allow-pretrained-fallback."
                    % (weights_name, ARM, exc, architecture, fallback,
                       checkpoint_filename)
                ) from exc
            weights_name = _resolve_pretrained(fallback, data_cfg)
            model = YOLO(weights_name)
            fallback_used = True
            print(warn_fallback_banner(ARM, architecture, weights_name))
        else:
            assert_checkpoint_matches_architecture(ARM, architecture, weights_name)

        model.train(
            data=str(dataset_root),
            epochs=spec["epochs"],
            imgsz=LEGACY_IMG_SIZE,
            batch=spec["batch"],
            lr0=spec["lr"],
            patience=LEGACY_PATIENCE,
            seed=LEGACY_SEED,
            workers=4,
            optimizer=LEGACY_OPTIMIZER,
            project=str(runs_dir(data_cfg) / "classify"),
            name=run_name,
            exist_ok=True,
            device=args.device,
            # NOTE: no class-weight argument. Deliberate. See the header.
        )

        run_dir = runs_dir(data_cfg) / "classify" / run_name
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        weights = best if best.exists() else last
        if not weights.exists():
            raise MissingInputError("No weights produced at %s" % run_dir)

        val_metrics = evaluate_legacy(weights, dataset_root / "val", classes)
        test_metrics = evaluate_legacy(weights, dataset_root / "test", classes)
        wall = round(time.perf_counter() - started, 2)
        print(
            "  f1_macro_val %.4f   f1_macro_test %.4f   (%.1fs)"
            % (val_metrics["f1_macro"], test_metrics["f1_macro"], wall)
        )

        size_mb = round(weights.stat().st_size / (1024**2), 3)
        record = registry.build_record(
            run_id=spec["run_id"],
            script=SCRIPT,
            arm=ARM,
            architecture=spec["architecture"],
            split_kind="dev",
            repeat=None,
            fold=None,
            epochs=spec["epochs"],
            batch=spec["batch"],
            lr=spec["lr"],
            class_weights="none_legacy_bug",
            run_seed=LEGACY_SEED,
            val_seed=None,
            checkpoint_resolved=Path(weights_name).name,
            pretrained_fallback_used=fallback_used,
            # Deliberately None, not False: this script reproduces the LEGACY
            # unweighted protocol, so there are no class weights to verify here.
            # False would mean "measured and failed". See the header.
            class_weights_verified=None,
            class_weights_proof={
                "not_applicable": True,
                "reason": (
                    "01 replicates the legacy unweighted ultralytics protocol; "
                    "class weights are deliberately not applied (MIGRATION_NOTES 5.4)"
                ),
            },
            corpus_fingerprint=corpus_fp,
            training=registry.training_outcome_absent(
                "trained by ultralytics model.train(), not the uniform loop in "
                "src/srpcard/train.py; per-epoch history and best-epoch selection "
                "are ultralytics' own and are not comparable with scripts 02-05",
                epochs_run=spec["epochs"],
            ),
            metrics=test_metrics,
            efficiency={"size_mb": size_mb},
            wall_time_s=wall,
            determinism_status={"note": "ultralytics trainer; determinism not enforced"},
            extra={
                "key": spec["key"],
                "protocol": "legacy_unweighted_ultralytics",
                "class_weights_applied": False,
                "f1_macro_val": val_metrics["f1_macro"],
                "accuracy_val": val_metrics["accuracy"],
                "val_metrics": val_metrics,
                "weights": str(weights),
            },
        )
        registry.append_record(record)
        print("  [registry] appended %s" % spec["run_id"])

    # ---- recompute the winner over all 18 ----
    rule("yolo26m winner, recomputed over all 18 configurations")
    rows = [
        {"key": r["key"], "epochs": int(r["epochs"]), "batch": int(r["batch"]),
         "lr": float(r["lr"]), "f1_macro_val": float(r["f1_macro_val"]), "source": "legacy_registry"}
        for _, r in completed.iterrows()
    ]
    for record in registry.load_registry():
        if record.get("script") == SCRIPT and record.get("extra", {}).get("f1_macro_val") is not None:
            rows.append(
                {
                    "key": record["extra"]["key"],
                    "epochs": record["epochs"],
                    "batch": record["batch"],
                    "lr": record["lr"],
                    "f1_macro_val": record["extra"]["f1_macro_val"],
                    "source": "this_script",
                }
            )
    table = pd.DataFrame(rows).drop_duplicates("key").sort_values("f1_macro_val", ascending=False)
    out_csv = artifacts_dir(data_cfg) / "medium_grid_complete.csv"
    table.to_csv(out_csv, index=False, lineterminator="\n")

    print("  NOTE: this table is the LEGACY UNWEIGHTED PROTOCOL, on the development split.")
    print("        It is not comparable with any number from scripts 02-05.\n")
    print("  %-22s %7s %6s %8s %14s %s" % ("key", "epochs", "batch", "lr", "f1_macro_val", "source"))
    for row in table.to_dict("records"):
        print(
            "  %-22s %7d %6d %8.0e %14.4f %s"
            % (row["key"], row["epochs"], row["batch"], row["lr"], row["f1_macro_val"], row["source"])
        )

    if len(table) != 18:
        print("\n[WARN] %d of 18 configurations present; winner is provisional." % len(table))

    old = arms_cfg["arms"][ARM]
    winner = table.iloc[0]
    print("\n  %-14s %-22s epochs %2d  batch %2d  lr %.0e  f1_macro_val %s"
          % ("OLD (10/18):", "m_ep%d_bs%d_lr%.0e" % (old["epochs"], old["batch"], old["lr"]),
             old["epochs"], old["batch"], old["lr"], "0.7657"))
    print("  %-14s %-22s epochs %2d  batch %2d  lr %.0e  f1_macro_val %.4f"
          % ("NEW (18/18):", winner["key"], winner["epochs"], winner["batch"],
             winner["lr"], winner["f1_macro_val"]))
    changed = (
        int(winner["epochs"]) != int(old["epochs"])
        or int(winner["batch"]) != int(old["batch"])
        or abs(float(winner["lr"]) - float(old["lr"])) > 1e-12
    )
    print("  winner %s" % ("CHANGED" if changed else "unchanged"))

    write_back_winner(
        int(winner["epochs"]), int(winner["batch"]), float(winner["lr"]),
        winner["key"], float(winner["f1_macro_val"]),
    )
    # Snapshot the resolved config into artifacts/, which on Colab is a symlink
    # into Drive. configs/arms.yaml lives in the clone and dies with the session;
    # this copy is what scripts/restore_arms.py puts back in a fresh clone.
    snapshot = snapshot_arms(SCRIPT, data_cfg)
    print("[arms.yaml] snapshotted to %s" % snapshot)
    print("            recover it in a fresh clone with: python scripts/restore_arms.py")
    print("[artifacts] wrote %s" % out_csv)

    if todo and not args.keep_dataset and dataset_root.exists():
        shutil.rmtree(dataset_root)
        print("[dataset] removed %s" % dataset_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
