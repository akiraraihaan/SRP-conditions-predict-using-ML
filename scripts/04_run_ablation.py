#!/usr/bin/env python
"""04 -- class-weight ablation on yolo26n. 15 runs.

    python scripts/04_run_ablation.py
    python scripts/04_run_ablation.py --dry-run

Identical to the yolo26n runs of script 03 in every respect -- same folds, same
run_seed and val_seed, same epochs, batch and lr, same selection criterion --
EXCEPT `class_weights: none`. The seeds are pure functions of (repeat, fold), so
the weighted and unweighted arms share their head initialisation, their batch
order and their validation slice. That is what isolates the weighting effect.

Also emits:
  artifacts/ablation_paired.csv     repeat, fold, f1_macro_weighted,
                                    f1_macro_unweighted, delta
  artifacts/ablation_per_class.csv  per-class recall delta, rarest class first
plus a paired Wilcoxon signed-rank test over the 15 pairs, with the mean delta
and a confidence interval -- not just a p-value.
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
from srpcard.efficiency import profile  # noqa: E402
from srpcard.models import add_fallback_argument, build_model  # noqa: E402
from srpcard.train import (  # noqa: E402
    ImageCache,
    TrainConfig,
    labels_by_idx_map,
    require_class_weights_verified,
    train_fold,
)

SCRIPT = "04_run_ablation"
CV_SCRIPT = "03_run_cv"


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 12345):
    """Percentile bootstrap CI for the mean of the paired differences."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--analyse-only", action="store_true", help="skip training, just report")
    add_fallback_argument(parser)
    args = parser.parse_args()

    data_cfg = load_data_config()
    arms_cfg = load_arms_config()
    ablation_cfg = arms_cfg["ablation"]
    arm = ablation_cfg["arm"]
    arm_cfg = arms_cfg["arms"][arm]

    rule("04 -- class-weight ablation (%s, class_weights: none)" % arm)
    print("  everything except class_weights is identical to this arm's 03 runs")

    index = srp_data.load_image_index()
    index_path = artifacts_dir(data_cfg) / "image_index.csv"
    payload = srp_folds.load_folds(index, index_path=index_path)
    print("[folds] corpus fingerprint verified  OK  (n=%d)" % payload["corpus"]["n"])
    corpus_fp = srp_folds.cv_corpus_fingerprint(payload)

    specs = []
    for entry in payload["folds"]:
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
            "class_weights": ablation_cfg["class_weights"],
            "run_seed": entry["run_seed"],
            "val_seed": entry["val_seed"],
            "extra": None,
            "_entry": entry,
        }
        spec["run_id"] = registry.compute_run_id(**spec)
        specs.append(spec)

    todo, skipped = registry.plan_runs(specs)
    registry.print_plan(SCRIPT, todo, skipped)
    if args.dry_run:
        for spec in todo:
            print("  TODO %s  r%df%d" % (spec["run_id"], spec["repeat"], spec["fold"]))
        return 0

    registry.assert_arms_match_registry(
        script=SCRIPT, arms=[arm], arms_cfg=arms_cfg, split_kind="cv"
    )

    # Not fatal: folds on different cards are valid runs, and free-tier compute
    # moves. But wall-time and latency stop being comparable across them, and it
    # belongs in the methods section rather than being found after submission.
    registry.warn_if_mixed_hardware([arm])

    # Run even though THIS script trains unweighted: the ablation only means
    # something if the weighting it removes demonstrably works in the first place.
    weights_proof = require_class_weights_verified(
        int(arms_cfg["shared"]["num_classes"]), script=SCRIPT
    )

    if todo and not args.analyse_only:
        data_root = resolve_data_root(data_cfg)
        labels_by_idx = labels_by_idx_map(index, data_cfg)
        cache = ImageCache(index, data_root, int(arms_cfg["shared"]["image_size"]))
        cache.warm(srp_data.clean_index(index)["idx"].tolist())
        print("[cache] %d images letterboxed into RAM" % len(cache._cache))

        for position, spec in enumerate(todo, 1):
            entry = spec.pop("_entry")
            rule(
                "run %d/%d  %s UNWEIGHTED  repeat %d fold %d"
                % (position, len(todo), arm, spec["repeat"], spec["fold"])
            )
            bundle = build_model(
                arm,
                arms_cfg,
                data_cfg,
                with_efficiency=False,
                seed=spec["run_seed"],
                allow_pretrained_fallback=args.allow_pretrained_fallback,
            )
            cfg = TrainConfig.from_arm(arm, arms_cfg, class_weights="none")
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
                "  -> selected epoch %d/%d  test f1_macro %.4f  (%.1fs)"
                % (result.best_epoch + 1, cfg.epochs, metrics["f1_macro"], wall)
            )

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
                    class_weights="none",
                    run_seed=spec["run_seed"],
                    val_seed=spec["val_seed"],
                    checkpoint_resolved=bundle.checkpoint_resolved,
                    pretrained_fallback_used=bundle.pretrained_fallback_used,
                    class_weights_verified=weights_proof["passed"],
                    class_weights_proof=weights_proof,
                    corpus_fingerprint=corpus_fp,
                    training=registry.training_outcome(result),
                    metrics=metrics,
                    efficiency=efficiency,
                    wall_time_s=wall,
                    determinism_status=result.determinism,
                    extra={
                        "protocol": "uniform",
                        "ablation": "class_weights_none",
                        "selection_metric": "val_f1_macro",
                    },
                )
            )
            print("  [registry] appended %s" % spec["run_id"])

    # ---------------- paired analysis ----------------
    rule("paired analysis: weighted (03) vs unweighted (04)")
    records = registry.load_registry()
    weighted = {
        (r["repeat"], r["fold"]): r
        for r in records
        if r.get("script") == CV_SCRIPT and r.get("arm") == arm
    }
    unweighted = {
        (r["repeat"], r["fold"]): r for r in records if r.get("script") == SCRIPT
    }
    common = sorted(set(weighted) & set(unweighted))
    if not common:
        print(
            "No paired folds yet. The weighted side comes from scripts/03_run_cv.py "
            "for arm %r.\n  weighted folds present: %d   unweighted: %d"
            % (arm, len(weighted), len(unweighted))
        )
        return 1
    if len(common) < len(payload["folds"]):
        print(
            "[WARN] only %d of %d folds are paired so far; the test below uses those %d."
            % (len(common), len(payload["folds"]), len(common))
        )

    rows = []
    for repeat, fold in common:
        w = weighted[(repeat, fold)]
        u = unweighted[(repeat, fold)]
        rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "f1_macro_weighted": w["f1_macro"],
                "f1_macro_unweighted": u["f1_macro"],
                "delta": w["f1_macro"] - u["f1_macro"],
            }
        )
    paired = pd.DataFrame(rows).sort_values(["repeat", "fold"])
    paired_path = artifacts_dir(data_cfg) / "ablation_paired.csv"
    paired.to_csv(paired_path, index=False, lineterminator="\n")

    print("  %6s %5s %18s %20s %10s" % ("repeat", "fold", "f1_macro_weighted",
                                        "f1_macro_unweighted", "delta"))
    for row in paired.to_dict("records"):
        print(
            "  %6d %5d %18.4f %20.4f %+10.4f"
            % (row["repeat"], row["fold"], row["f1_macro_weighted"],
               row["f1_macro_unweighted"], row["delta"])
        )

    deltas = paired["delta"].to_numpy()
    mean_delta = float(deltas.mean())
    low, high = bootstrap_ci(deltas)
    print(
        "\n  mean delta (weighted - unweighted) = %+0.4f   95%% CI [%+0.4f, %+0.4f]"
        % (mean_delta, low, high)
    )
    print(
        "  weighted better on %d of %d folds"
        % (int((deltas > 0).sum()), len(deltas))
    )

    try:
        from scipy.stats import wilcoxon

        if np.allclose(deltas, 0):
            print("  Wilcoxon: all differences are zero; test not defined.")
            stat, pvalue = float("nan"), 1.0
        else:
            stat, pvalue = wilcoxon(
                paired["f1_macro_weighted"], paired["f1_macro_unweighted"]
            )
            print(
                "  Wilcoxon signed-rank (paired, n=%d): W = %.1f, p = %.4f  -> %s at alpha=0.05"
                % (len(deltas), stat, pvalue,
                   "SIGNIFICANT" if pvalue < 0.05 else "not significant")
            )
    except ImportError:
        stat, pvalue = float("nan"), float("nan")
        print("  scipy not installed; Wilcoxon skipped.")

    # ---------------- per-class recall delta, rarest first ----------------
    clean = srp_data.clean_index(index)
    sizes = clean["class"].value_counts().to_dict()
    order = sorted(data_cfg["classes"], key=lambda c: sizes.get(c, 0))

    per_class_rows = []
    for name in order:
        w_values = [weighted[k]["recall_per_class"][name] for k in common]
        u_values = [unweighted[k]["recall_per_class"][name] for k in common]
        diffs = np.asarray(w_values) - np.asarray(u_values)
        per_class_rows.append(
            {
                "class": name,
                "n_clean": sizes.get(name, 0),
                "recall_weighted_mean": float(np.mean(w_values)),
                "recall_weighted_std": float(np.std(w_values, ddof=1)) if len(w_values) > 1 else 0.0,
                "recall_unweighted_mean": float(np.mean(u_values)),
                "recall_unweighted_std": float(np.std(u_values, ddof=1)) if len(u_values) > 1 else 0.0,
                "delta_mean": float(diffs.mean()),
                "delta_std": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
            }
        )
    per_class = pd.DataFrame(per_class_rows)
    per_class_path = artifacts_dir(data_cfg) / "ablation_per_class.csv"
    per_class.to_csv(per_class_path, index=False, lineterminator="\n")

    print("\n  per-class recall delta, RAREST CLASS FIRST (weighted - unweighted)")
    print("  %-45s %6s %16s %18s %10s" % ("class", "n", "recall_weighted",
                                          "recall_unweighted", "delta"))
    for row in per_class.to_dict("records"):
        print(
            "  %-45s %6d %16.4f %18.4f %+10.4f"
            % (row["class"], row["n_clean"], row["recall_weighted_mean"],
               row["recall_unweighted_mean"], row["delta_mean"])
        )

    print("\n[artifacts] wrote %s" % paired_path)
    print("[artifacts] wrote %s" % per_class_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
