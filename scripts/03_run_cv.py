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
from srpcard.config import (  # noqa: E402
    artifacts_dir,
    hardware,
    library_versions,
    load_arms_config,
    load_data_config,
    resolve_data_root,
)
from srpcard.efficiency import profile  # noqa: E402
from srpcard.models import ARM_NAMES, add_fallback_argument, build_model  # noqa: E402
from srpcard.train import (  # noqa: E402
    ImageCache,
    TrainConfig,
    labels_by_idx_map,
    require_class_weights_verified,
    train_fold,
)

SCRIPT = "03_run_cv"


def benchmark_fold(arms_cfg) -> tuple[int, int]:
    """The ONE fold whose weights are kept, from configs/arms.yaml.

    Fixed and pre-declared, identical across architectures, rather than "the
    fold with the highest test macro-F1". Selecting on test data is the exact
    circularity this project spent weeks removing: it is harmless for latency,
    which the architecture decides, but it contaminates the INT8 accuracy delta
    and cannot be described in a methods section without a caveat.
    """
    block = (arms_cfg.get("reporting") or {}).get("benchmark_fold") or {}
    return int(block.get("repeat", 0)), int(block.get("fold", 0))


def save_fold_weights(out_dir: Path, spec, bundle, result, f1_macro: float) -> None:
    """Write the benchmark fold's weights for one arm.

    No comparison against what is already there: the fold is fixed, so there is
    nothing to choose between. The file is the {'arm', 'state_dict'} form 07's
    loader documents.
    """
    import json

    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = out_dir / ("%s.json" % spec["arm"])
    target = out_dir / ("%s.pt" % spec["arm"])
    torch.save(
        {
            "arm": spec["arm"],
            "architecture": spec["architecture"],
            "state_dict": result.best_state,
            "run_id": spec["run_id"],
            "repeat": spec["repeat"],
            "fold": spec["fold"],
            "f1_macro": f1_macro,
            "checkpoint_resolved": bundle.checkpoint_resolved,
        },
        target,
    )
    sidecar.write_text(
        json.dumps(
            {
                "arm": spec["arm"],
                "run_id": spec["run_id"],
                "repeat": spec["repeat"],
                "fold": spec["fold"],
                "f1_macro": f1_macro,
                "weights": target.name,
                "selection": (
                    "fixed benchmark fold from configs/arms.yaml:"
                    "reporting.benchmark_fold -- not chosen on test performance"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    print("  [weights] %s r%df%d (f1 %.4f) -> %s"
          % (spec["arm"], spec["repeat"], spec["fold"], f1_macro, target))


VERIFIED_METRICS = ("f1_macro", "accuracy", "precision_macro", "recall_macro")


def emit_weights_for_completed(out_dir, specs, *, tolerance, arms_cfg, data_cfg,
                               index, cache, device, quiet, allow_fallback) -> int:
    """Re-run already-completed benchmark-fold runs purely to write their weights.

    --save-weights cannot do this. Weights are not part of the run_id, so once a
    run is in the registry it is skipped and no file is ever written: the plan
    prints "already complete (skipped), 0 remaining" and exits successfully
    having done nothing at all.

    Nothing is appended to the registry here -- the run already happened and its
    record stands. What IS checked is that the reproduction lands on the recorded
    numbers. run_seed and val_seed are pure functions of (repeat, fold), so a
    divergence means the run is not reproducible, which is worth knowing on its
    own and is a reason to stop rather than to ship a checkpoint that does not
    correspond to the published metrics.
    """
    from srpcard import evaluate
    from srpcard.train import TrainConfig, labels_by_idx_map

    recorded = {r.get("run_id"): r for r in registry.load_registry()}
    wanted = [spec for spec in specs if spec["run_id"] in recorded]
    for spec in specs:
        if spec["run_id"] not in recorded:
            print("  [skip] %s r%df%d is not in the registry -- run it normally "
                  "with --save-weights" % (spec["arm"], spec["repeat"], spec["fold"]))
    if not wanted:
        print("  nothing to reproduce")
        return 0

    labels_by_idx = labels_by_idx_map(index, data_cfg)
    written = 0
    for position, spec in enumerate(wanted, 1):
        entry = spec["_entry"]
        record = recorded[spec["run_id"]]
        rule("reproduce %d/%d  %s  r%df%d  (run_id %s)"
             % (position, len(wanted), spec["arm"], spec["repeat"], spec["fold"],
                spec["run_id"]))
        bundle = build_model(spec["arm"], arms_cfg, data_cfg, with_efficiency=False,
                             seed=spec["run_seed"],
                             allow_pretrained_fallback=allow_fallback)
        cfg = TrainConfig.from_arm(spec["arm"], arms_cfg, epochs=spec["epochs"],
                                   class_weights=spec["class_weights"])
        started = time.perf_counter()
        result = train_fold(bundle, cache, entry["train_idx"], entry["val_idx"],
                            labels_by_idx, cfg, seed=spec["run_seed"],
                            device=device, verbose=not quiet)
        metrics = evaluate.evaluate_fold(bundle.module, cache, entry["test_idx"],
                                         labels_by_idx, data_cfg)
        elapsed = time.perf_counter() - started

        drift = []
        for field in VERIFIED_METRICS:
            was, now = record.get(field), metrics.get(field)
            if was is None or now is None:
                continue
            if abs(float(was) - float(now)) > tolerance:
                drift.append((field, float(was), float(now)))

        print("  recorded f1_macro %.6f   reproduced %.6f   (%.1fs)"
              % (record.get("f1_macro", float("nan")), metrics["f1_macro"], elapsed))
        if drift:
            here = hardware()
            lines = [
                "REPRODUCTION MISMATCH for %s r%df%d (run_id %s)."
                % (spec["arm"], spec["repeat"], spec["fold"], spec["run_id"]),
                "",
                "  %-18s %18s %18s %13s" % ("metric", "recorded", "reproduced", "difference"),
            ]
            for field, was, now in drift:
                lines.append("  %-18s %18.9f %18.9f %13.2e"
                             % (field, was, now, abs(was - now)))
            lines += [
                "",
                "  run_seed and val_seed are pure functions of (repeat, fold), so the",
                "  same fold under the same code on the same hardware reproduces exactly.",
                "",
                "  recorded on : %s, torch %s"
                % (record.get("gpu") or record.get("device_kind"),
                   (record.get("library_versions") or {}).get("torch")),
                "  running on  : %s, torch %s"
                % (here.get("gpu") or here.get("device_kind"),
                   library_versions().get("torch")),
                "",
                "  Different hardware or a different torch build explains a small",
                "  difference and is not a reproducibility failure -- re-run with a",
                "  larger --reproduce-tolerance if that is the case. The SAME machine",
                "  diverging is a real problem, and no checkpoint should be shipped",
                "  from it: it would not correspond to the published metrics.",
            ]
            raise SystemExit("\n".join(lines))

        save_fold_weights(out_dir, spec, bundle, result, metrics["f1_macro"])
        written += 1
    return written


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
    parser.add_argument(
        "--save-weights",
        default=None,
        metavar="DIR",
        help=(
            "when a run is EXECUTED, also write the benchmark fold's weights to "
            "DIR/<arm>.pt for scripts/07_bench_edge.py. Off by default: none of "
            "the reported metrics need weights. Only the pre-declared benchmark "
            "fold is kept -- see configs/arms.yaml:reporting.benchmark_fold."
        ),
    )
    parser.add_argument(
        "--emit-weights",
        default=None,
        metavar="DIR",
        help=(
            "REPRODUCE already-completed benchmark-fold runs purely to write their "
            "weights, appending nothing to the registry. Needed because --save-weights "
            "cannot help once a run is complete: weights are not part of the run_id, "
            "so the run is skipped and no file is written. The reproduced metrics are "
            "checked against the recorded ones and a mismatch aborts."
        ),
    )
    parser.add_argument(
        "--reproduce-tolerance",
        type=float,
        default=1e-6,
        metavar="EPS",
        help=(
            "how far a reproduced metric may fall from the recorded one before "
            "--emit-weights aborts. The default is effectively exact; relax it only "
            "when reproducing on different hardware from the original run."
        ),
    )
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
    # Verified above; recorded on every record below, so each result carries
    # the corpus it was produced on rather than only having been checked once.
    corpus_fp = srp_folds.cv_corpus_fingerprint(payload)

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

    bench_fold = benchmark_fold(arms_cfg)
    todo, skipped = registry.plan_runs(specs)
    registry.print_plan(SCRIPT, todo, skipped)
    if args.save_weights or args.emit_weights:
        print("[weights] benchmark fold is repeat %d fold %d "
              "(configs/arms.yaml:reporting.benchmark_fold)" % bench_fold)
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
    emit_specs = [
        spec for spec in specs
        if (spec["repeat"], spec["fold"]) == bench_fold
    ] if args.emit_weights else []

    if not todo and not emit_specs:
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

    # Before any training: refuse to add runs under hyperparameters that disagree
    # with the ones completed runs of the same arm were trained under. epochs,
    # batch and lr feed the run_id hash, so drift does not resume -- it duplicates.
    registry.assert_arms_match_registry(
        script=SCRIPT, arms=arms, arms_cfg=arms_cfg, split_kind="cv"
    )

    # Not fatal: folds on different cards are valid runs, and free-tier compute
    # moves. But wall-time and latency stop being comparable across them, and it
    # belongs in the methods section rather than being found after submission.
    registry.warn_if_mixed_hardware(arms)

    # Once per invocation, before any training. Aborts if the balanced weights
    # do not actually reach the loss; the compact proof goes into every record.
    weights_proof = require_class_weights_verified(
        int(arms_cfg["shared"]["num_classes"]), script=SCRIPT
    )

    # ---- reproduce completed runs, purely to write their weights ----
    if emit_specs:
        rule("--emit-weights: reproducing %d completed run(s), appending nothing"
             % len(emit_specs))
        written = emit_weights_for_completed(
            Path(args.emit_weights), emit_specs,
            tolerance=args.reproduce_tolerance, arms_cfg=arms_cfg,
            data_cfg=data_cfg, index=index, cache=cache, device=args.device,
            quiet=args.quiet, allow_fallback=args.allow_pretrained_fallback,
        )
        rule("DONE -- %d checkpoint(s) written, registry untouched" % written)
        print("[registry] %d record(s), unchanged" % len(registry.load_registry()))
        if not todo:
            return 0

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
            class_weights_verified=weights_proof["passed"],
            class_weights_proof=weights_proof,
            corpus_fingerprint=corpus_fp,
            training=registry.training_outcome(result),
            metrics=metrics,
            efficiency=efficiency,
            wall_time_s=wall,
            determinism_status=result.determinism,
            extra={
                # Scopes the drift guard: records from script 01 (legacy
                # augmented protocol) must never be compared against these.
                "protocol": "uniform",
                "selection_metric": "val_f1_macro",
                "selection_tiebreak": "val_loss_unweighted",
                # what a val-loss criterion WOULD have picked -- recorded, not acted on
                "min_val_loss": result.min_val_loss,
                "class_weight_values": result.class_weights,
                "device": result.device,
                "model_notes": bundle.notes,
            },
        )
        if args.save_weights and (spec["repeat"], spec["fold"]) == bench_fold:
            save_fold_weights(
                Path(args.save_weights), spec, bundle, result, metrics["f1_macro"]
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
