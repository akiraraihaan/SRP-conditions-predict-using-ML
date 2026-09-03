#!/usr/bin/env python
"""05 -- learning curve under the FINAL locked yolo26n configuration.

    python scripts/05_learning_curve.py
    python scripts/05_learning_curve.py --dry-run

Fractions 20/40/60/80/100 % of each fold's training set, drawn stratified.
ONE draw per (fold, fraction): 5 fractions x 15 folds = 75 runs, giving 15
estimates per fraction. Reports mean and standard deviation of test macro-F1
at each fraction.

There is deliberately no second `repeats` dimension. The 15 folds already ARE
3 repeats of 5-fold CV, so a `learning_curve.repeats: 3` multiplied that to 225
runs -- three times the size of the main experiment (75) for a supporting
analysis, and the extra draws resample the same 15 partitions rather than
adding independent information. See configs/arms.yaml:learning_curve.

This is NOT a re-run of the old learning curve. The legacy one (code.ipynb cell
18, "[Cell 19]") used batch 16 and **lr 1e-3** -- it indexed the option lists
positionally and landed on the wrong learning rate despite a comment claiming
"best hyperparams". The locked nano winner is lr 1e-2. The old curve therefore
never described the model that was actually reported, and its cached numbers
additionally carry the 14-class evaluation deflation. See MIGRATION_NOTES.md
section 7.

Subsample seeds are derived from (repeat, fold, fraction) so the curve is
reproducible and every arm would see the same subsets if it were ever extended.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from srpcard import data as srp_data  # noqa: E402
from srpcard import evaluate, registry  # noqa: E402
from srpcard import folds as srp_folds  # noqa: E402
from srpcard.config import (  # noqa: E402
    artifacts_dir,
    load_arms_config,
    load_data_config,
    resolve_data_root,
)
from srpcard.models import add_fallback_argument, build_model  # noqa: E402
from srpcard.train import (  # noqa: E402
    ImageCache,
    TrainConfig,
    labels_by_idx_map,
    require_class_weights_verified,
    train_fold,
)

SCRIPT = "05_learning_curve"
SUBSAMPLE_SEED_BASE = 900000


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def subsample_seed(repeat: int, fold: int, fraction: float) -> int:
    return SUBSAMPLE_SEED_BASE + repeat * 10000 + fold * 1000 + int(fraction * 100)


def stratified_subset(idxs, labels_by_idx, fraction: float, seed: int):
    """A stratified `fraction` of idxs. Returns all of them when fraction >= 1."""
    from sklearn.model_selection import train_test_split

    if fraction >= 1.0:
        return list(idxs)
    labels = [labels_by_idx[int(i)] for i in idxs]
    subset, _ = train_test_split(
        list(idxs), train_size=fraction, stratify=labels, random_state=seed, shuffle=True
    )
    return sorted(subset)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--folds", type=int, default=None, help="use only the first N folds")
    add_fallback_argument(parser)
    args = parser.parse_args()

    data_cfg = load_data_config()
    arms_cfg = load_arms_config()
    lc_cfg = arms_cfg["learning_curve"]
    arm = lc_cfg["arm"]
    arm_cfg = arms_cfg["arms"][arm]
    fractions = list(lc_cfg["fractions"])
    if "repeats" in lc_cfg:
        raise ValueError(
            "configs/arms.yaml:learning_curve carries a 'repeats' key (%r).\n"
            "  It was removed deliberately: the 15 folds already are 3 repeats of\n"
            "  5-fold CV, so multiplying by it produced %d runs -- three times the\n"
            "  size of the main experiment -- for a supporting analysis, and the\n"
            "  extra draws resample the same 15 partitions rather than adding\n"
            "  independent information. Delete the key.\n"
            "  The curve is %d fractions x 15 folds = %d runs, 15 per fraction."
            % (
                lc_cfg["repeats"],
                len(fractions) * 15 * int(lc_cfg["repeats"]),
                len(fractions),
                len(fractions) * 15,
            )
        )

    rule("05 -- learning curve (%s, FINAL locked configuration)" % arm)
    print(
        "  locked config : epochs %d  batch %d  lr %g   (NOT the legacy lr 1e-3)"
        % (arm_cfg["epochs"], arm_cfg["batch"], arm_cfg["lr"])
    )
    print("  fractions     : %s" % fractions)
    print("  draws         : 1 per (fold, fraction)")

    index = srp_data.load_image_index()
    index_path = artifacts_dir(data_cfg) / "image_index.csv"
    payload = srp_folds.load_folds(index, index_path=index_path)
    entries = payload["folds"][: args.folds] if args.folds else payload["folds"]
    print("[folds] using %d fold(s)" % len(entries))
    corpus_fp = srp_folds.cv_corpus_fingerprint(payload)

    labels_by_idx = labels_by_idx_map(index, data_cfg)

    specs = []
    for entry in entries:
        for fraction in fractions:
            spec = {
                "arm": arm,
                "architecture": arm_cfg["architecture"],
                "script": SCRIPT,
                "split_kind": "cv",
                "repeat": entry["repeat"],
                "fold": entry["fold"],
                "epochs": int(arm_cfg["epochs"]),
                "batch": int(arm_cfg["batch"]),
                "lr": float(arm_cfg["lr"]),
                "class_weights": arms_cfg["shared"]["class_weights"],
                "run_seed": entry["run_seed"],
                "extra": "lc_frac%.2f" % fraction,
            }
            spec["run_id"] = registry.compute_run_id(**spec)
            spec["_entry"] = entry
            spec["fraction"] = fraction
            spec["val_seed"] = entry["val_seed"]
            specs.append(spec)

    todo, skipped = registry.plan_runs(specs)
    registry.print_plan(SCRIPT, todo, skipped)
    if args.dry_run:
        print("  %d runs = %d folds x %d fractions (1 draw each)"
              % (len(specs), len(entries), len(fractions)))
        return 0

    registry.assert_arms_match_registry(
        script=SCRIPT, arms=[arm], arms_cfg=arms_cfg, split_kind="cv"
    )

    weights_proof = require_class_weights_verified(
        int(arms_cfg["shared"]["num_classes"]), script=SCRIPT
    )

    if todo:
        data_root = resolve_data_root(data_cfg)
        cache = ImageCache(index, data_root, int(arms_cfg["shared"]["image_size"]))
        cache.warm(srp_data.clean_index(index)["idx"].tolist())
        print("[cache] %d images letterboxed into RAM" % len(cache._cache))

    for position, spec in enumerate(todo, 1):
        entry = spec.pop("_entry")
        seed = subsample_seed(spec["repeat"], spec["fold"], spec["fraction"])
        subset = stratified_subset(entry["train_idx"], labels_by_idx, spec["fraction"], seed)
        rule(
            "run %d/%d  r%df%d  frac %.0f%%  (n_train %d)"
            % (position, len(todo), spec["repeat"], spec["fold"],
               spec["fraction"] * 100, len(subset))
        )

        bundle = build_model(
            arm,
            arms_cfg,
            data_cfg,
            with_efficiency=False,
            seed=spec["run_seed"],
            allow_pretrained_fallback=args.allow_pretrained_fallback,
        )
        cfg = TrainConfig.from_arm(arm, arms_cfg)
        started = time.perf_counter()
        result = train_fold(
            bundle, cache, subset, entry["val_idx"], labels_by_idx, cfg,
            seed=spec["run_seed"], device=args.device, verbose=not args.quiet,
        )
        metrics = evaluate.evaluate_fold(
            bundle.module, cache, entry["test_idx"], labels_by_idx, data_cfg
        )
        wall = round(time.perf_counter() - started, 2)
        print("  -> test f1_macro %.4f  (%.1fs)" % (metrics["f1_macro"], wall))

        registry.append_record(
            registry.build_record(
                run_id=spec["run_id"],
                script=SCRIPT,
                arm=arm,
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
                class_weights_verified=weights_proof["passed"],
                class_weights_proof=weights_proof,
                corpus_fingerprint=corpus_fp,
                training=registry.training_outcome(result),
                metrics=metrics,
                efficiency={},
                wall_time_s=wall,
                determinism_status=result.determinism,
                extra={
                    "protocol": "uniform",
                    "learning_curve": True,
                    "fraction": spec["fraction"],
                    "n_train": len(subset),
                    "subsample_seed": seed,
                },
            )
        )

    # ---------------- summary ----------------
    rule("learning curve summary")
    rows = [
        {
            "fraction": r["extra"]["fraction"],
            "n_train": r["extra"]["n_train"],
            "f1_macro": r["f1_macro"],
            "accuracy": r["accuracy"],
        }
        for r in registry.load_registry()
        if r.get("script") == SCRIPT
    ]
    if not rows:
        print("No learning-curve runs in the registry yet.")
        return 1

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("fraction")
        .agg(
            n_runs=("f1_macro", "size"),
            n_train_mean=("n_train", "mean"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
        .reset_index()
        .fillna(0.0)
    )
    out_csv = artifacts_dir(data_cfg) / "learning_curve.csv"
    summary.to_csv(out_csv, index=False, lineterminator="\n")

    print("  %8s %6s %10s %16s %14s" % ("fraction", "runs", "n_train", "f1_macro mean", "std"))
    for row in summary.to_dict("records"):
        print(
            "  %7.0f%% %6d %10.0f %16.4f %14.4f"
            % (row["fraction"] * 100, row["n_runs"], row["n_train_mean"],
               row["f1_macro_mean"], row["f1_macro_std"])
        )
    print("\n[artifacts] wrote %s" % out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
