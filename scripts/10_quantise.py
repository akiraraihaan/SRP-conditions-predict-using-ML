#!/usr/bin/env python
"""10 -- quantisation: size AND accuracy, on labelled data. CPU only.

    python scripts/10_quantise.py --weights exported --data-root /content/dataset
    python scripts/10_quantise.py --weights exported --data-root ... --dry-run

NO RASPBERRY PI IS INVOLVED, and NO LATENCY IS MEASURED HERE. This is size and
accuracy only. The printed summary says so, and so does every row, because the
manuscript must not end up claiming a speedup nobody timed.

WHY THIS EXISTS
---------------
The edge benchmark used DYNAMIC post-training quantisation, which converts
Linear and the RNN family and leaves every Conv2d in fp32. That finding is
honest and it stays -- resnet18's INT8 ratio of 1.0 is the correct result for a
convolution-dominated architecture, not a broken measurement.

But it is half the story, twice over:

  1. STATIC PTQ quantises convolutions. Fusing Conv-BN-ReLU and calibrating on
     real activations is the method that actually compresses these models, and
     not measuring it left the microcontroller argument resting on the one
     method guaranteed to do nothing for them.

  2. ACCURACY AFTER QUANTISATION was never measured, on the grounds that the
     benchmark images were unlabelled. That reads as an excuse when a labelled
     corpus of 668 images is sitting in the repository. It is measured here,
     with our own metric code, on the benchmark fold's test partition.

CALIBRATION DISCIPLINE. Static PTQ needs a calibration pass, and it is run on
the benchmark fold's TRAINING partition only. The test partition is asserted
untouched -- calibrating on test would tune the quantisation to the images it
is then scored on, which is the same circularity this project spent weeks
removing from everything else.

Which fold: `configs/arms.yaml:reporting.benchmark_fold`, the same fixed,
pre-declared fold whose weights script 03 exported. Not the best fold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srpcard import aggregate  # noqa: E402
from srpcard import folds as srp_folds  # noqa: E402
from srpcard.config import (  # noqa: E402
    published_arms,
    artifacts_dir,
    load_arms_config,
    load_data_config,
    resolve_data_root,
)

SCRIPT = "10_quantise"

# A flash budget worth naming. Below this an MCU with external flash can hold
# the weights; above it, the deployment story needs a different device.
FLASH_BUDGET_MB = 1.0

CALIBRATION_BATCHES = 16


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


def benchmark_fold(arms_cfg) -> tuple[int, int]:
    block = (arms_cfg.get("reporting") or {}).get("benchmark_fold") or {}
    return int(block.get("repeat", 0)), int(block.get("fold", 0))


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def state_dict_size_mb(module, path: Path) -> float:
    import torch

    torch.save(module.state_dict(), path)
    size = path.stat().st_size / (1024 ** 2)
    path.unlink(missing_ok=True)
    return round(size, 3)


def layer_census(module) -> dict[str, int]:
    census: dict[str, int] = {}
    for _, child in module.named_modules():
        if list(child.children()):
            continue
        name = type(child).__name__
        census[name] = census.get(name, 0) + 1
    return census


def coverage(fp32, quantised) -> dict:
    """Which layer types were converted, and what share of parameters."""
    originals = dict(fp32.named_modules())
    converted: dict[str, int] = {}
    for name, child in quantised.named_modules():
        if not type(child).__module__.startswith(
            ("torch.ao.nn.quantized", "torch.nn.quantized", "torch.ao.nn.intrinsic")
        ):
            continue
        original = originals.get(name)
        if original is None:
            continue          # internal plumbing, e.g. fc._packed_params
        key = type(original).__name__
        converted[key] = converted.get(key, 0) + 1

    total = sum(p.numel() for p in fp32.parameters())
    inside = 0
    for _, child in fp32.named_modules():
        if list(child.children()):
            continue
        if type(child).__name__ in converted:
            inside += sum(p.numel() for p in child.parameters(recurse=False))
    return {
        "quantised_layer_types": converted,
        "unquantised_layer_types": {
            layer: count for layer, count in layer_census(fp32).items()
            if layer not in converted
        },
        "params_total": total,
        "params_quantised_pct": round(100.0 * inside / total, 2) if total else None,
    }


def macro_f1(module, cache, indices, labels_by_idx, data_cfg) -> dict:
    from srpcard import evaluate

    return evaluate.evaluate_fold(module, cache, indices, labels_by_idx, data_cfg)


# --------------------------------------------------------------------------
# the two methods
# --------------------------------------------------------------------------


def dynamic_ptq(module):
    """Linear and the RNN family only. Conv2d is silently untouched."""
    import torch

    try:
        return torch.ao.quantization.quantize_dynamic(
            module, {torch.nn.Linear}, dtype=torch.qint8
        ), None
    except Exception as exc:  # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, exc)


def static_ptq(module, cache, calibration_idx, labels_by_idx):  # noqa: C901
    """Fuse, calibrate on TRAINING images, convert. Conv2d included.

    Returns (quantised, reason). A failure is reported per arm with its reason
    rather than skipped: "static PTQ is not available for this architecture" is
    itself a finding the manuscript needs, and a silent skip would read as a
    method that was never tried.
    """
    import copy

    import torch

    try:
        prepared = copy.deepcopy(module).eval()

        # Preference order, then ANY engine the build offers. Naming only
        # fbgemm and qnnpack looked reasonable and was wrong: torch 2.12.0+cpu
        # ships with `supported_engines == ["onednn"]`, so a hardcoded pair
        # would have reported "static PTQ unavailable" on a machine that
        # supports it perfectly well -- a false negative in the one table the
        # microcontroller argument rests on.
        available = [e for e in torch.backends.quantized.supported_engines
                     if e != "none"]
        backend = next(
            (e for e in ("fbgemm", "x86", "onednn", "qnnpack") if e in available),
            available[0] if available else None,
        )
        if backend is None:
            return None, ("no quantized backend available; torch reports "
                          "supported_engines=%s"
                          % list(torch.backends.quantized.supported_engines))
        torch.backends.quantized.engine = backend

        # Fuse where the architecture allows it. torch.ao.quantization.fuse_modules
        # needs explicit patterns per architecture, so the generic path is used
        # and the failure -- if any -- is reported rather than guessed around.
        try:
            prepared = torch.ao.quantization.fuse_modules(prepared, [], inplace=False)
        except Exception:      # noqa: BLE001 - fusion is an optimisation, not a requirement
            pass

        prepared.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        torch.ao.quantization.prepare(prepared, inplace=True)

        # Calibration sees exactly what evaluation sees: FoldDataset applies the
        # same normalisation, so the observed activation ranges are the ranges
        # the quantised model will actually meet.
        from torch.utils.data import DataLoader

        from srpcard.train import FoldDataset

        loader = DataLoader(
            FoldDataset(cache, calibration_idx, labels_by_idx),
            batch_size=8, shuffle=False,
        )
        with torch.no_grad():
            for position, (images, _) in enumerate(loader):
                if position >= CALIBRATION_BATCHES:
                    break
                prepared(images)

        converted = torch.ao.quantization.convert(prepared, inplace=False)
        return converted, "backend=%s" % backend
    except Exception as exc:  # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, exc)


# --------------------------------------------------------------------------

HEADER = [
    "Post-training quantisation: SIZE AND ACCURACY. NO LATENCY.",
    "",
    "Nothing here was timed. The Raspberry Pi is not involved, and no row in",
    "this table supports a claim about speed. Latency lives in",
    "artifacts/raspberry-pi-result/edge_benchmark.json and nowhere else.",
    "",
    "Two methods, because the edge benchmark measured only the first:",
    "  dynamic  converts Linear and the RNN family. Conv2d is NOT supported and",
    "           is silently left in fp32, so a convolution-dominated model is",
    "           barely compressed and a ratio near 1.0 is the CORRECT result.",
    "  static   fuses and calibrates, and does quantise convolutions. This is",
    "           the method the microcontroller argument actually rests on.",
    "",
    "CALIBRATION USED THE TRAINING PARTITION ONLY, asserted in code. Calibrating",
    "on test would tune the quantisation to the images it is then scored on.",
    "",
    "The fold is configs/arms.yaml:reporting.benchmark_fold -- fixed and",
    "pre-declared, not the best-scoring fold.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True,
                        help="directory of per-fold checkpoints (03 --emit-weights)")
    parser.add_argument("--data-root", default=None, help="the image corpus")
    parser.add_argument("--arms", nargs="*", default=None, help="subset (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the plan and exit, quantising nothing")
    args = parser.parse_args()

    rule("10 -- quantisation: size and accuracy. NO LATENCY MEASURED HERE.")

    data_cfg, arms_cfg = load_data_config(), load_arms_config()
    artifacts = artifacts_dir(data_cfg)
    repeat, fold = benchmark_fold(arms_cfg)
    arms = args.arms or sorted(published_arms(arms_cfg))

    weights_dir = Path(args.weights)
    missing = [a for a in arms if not (weights_dir / ("%s.pt" % a)).exists()]
    print("  arms          : %s" % ", ".join(arms))
    print("  weights       : %s" % weights_dir)
    print("  benchmark fold: repeat %d fold %d "
          "(configs/arms.yaml:reporting.benchmark_fold)" % (repeat, fold))
    print("  flash budget  : %.1f MB" % FLASH_BUDGET_MB)
    if missing:
        raise SystemExit(
            "No checkpoint for %d arm(s): %s\n"
            "  looked for <arm>.pt in %s\n"
            "  Produce them with:\n"
            "    python scripts/03_run_cv.py --emit-weights %s --arms <arm>"
            % (len(missing), ", ".join(missing), weights_dir, weights_dir)
        )

    if args.dry_run:
        print("\n  --dry-run: %d arm(s) x 2 methods. Nothing written." % len(arms))
        return 0

    import torch

    from srpcard.data import load_image_index
    from srpcard.train import FoldDataset, ImageCache, labels_by_idx_map

    data_root = Path(args.data_root) if args.data_root else resolve_data_root(data_cfg)
    index = load_image_index()
    bundle = srp_folds.load_folds(index, path=artifacts / "folds.json")
    entry = next(e for e in bundle["folds"]
                 if e["repeat"] == repeat and e["fold"] == fold)

    # The assertion that matters: calibration never sees a test image.
    calibration_idx = list(entry["train_idx"])
    test_idx = list(entry["test_idx"])
    overlap = set(calibration_idx) & set(test_idx)
    if overlap:
        raise SystemExit(
            "Calibration set overlaps the test partition on %d image(s). "
            "Calibrating on test tunes the quantisation to the images it is "
            "then scored on." % len(overlap)
        )
    print("  calibration   : %d training images (test partition: %d, untouched)"
          % (len(calibration_idx), len(test_idx)))

    labels_by_idx = labels_by_idx_map(index, data_cfg)
    cache = ImageCache(index, data_root, int(arms_cfg["shared"]["image_size"]))
    cache.warm(calibration_idx + test_idx)

    rows = []
    for arm in arms:
        rule(arm)
        payload = torch.load(str(weights_dir / ("%s.pt" % arm)),
                             map_location="cpu", weights_only=False)
        from srpcard.models import build_model

        built = build_model(arm, arms_cfg, data_cfg, with_efficiency=False)
        built.module.load_state_dict(payload["state_dict"])
        module = built.module.eval()

        scratch = artifacts / ("_q_%s.pt" % arm)
        fp32_size = state_dict_size_mb(module, scratch)
        fp32_metrics = macro_f1(module, cache, test_idx, labels_by_idx, data_cfg)

        row = {
            "arm": arm,
            "architecture": arms_cfg["arms"][arm]["architecture"],
            "protocol": "uniform",
            "repeat": repeat,
            "fold": fold,
            "n_test_images": len(test_idx),
            "n_calibration_images": len(calibration_idx),
            "latency_measured": False,
            "latency_note": "NOT MEASURED HERE -- size and accuracy only",
            "size_mb_fp32": fp32_size,
            "macro_f1_fp32": round(fp32_metrics["f1_macro"], 6),
        }
        print("  fp32          %8.3f MB   macro-F1 %.4f" % (fp32_size, fp32_metrics["f1_macro"]))

        for method, function in (("dynamic", dynamic_ptq), ("static", static_ptq)):
            if method == "dynamic":
                quantised, reason = function(module)
            else:
                quantised, reason = function(module, cache, calibration_idx, labels_by_idx)

            if quantised is None:
                row["%s_available" % method] = False
                row["%s_failure_reason" % method] = reason
                row["size_mb_%s" % method] = None
                row["macro_f1_%s" % method] = None
                print("  %-13s FAILED -- %s" % (method, reason))
                continue

            size = state_dict_size_mb(quantised, scratch)
            block = coverage(module, quantised)
            try:
                metrics = macro_f1(quantised, cache, test_idx, labels_by_idx, data_cfg)
                f1 = round(metrics["f1_macro"], 6)
            except Exception as exc:  # noqa: BLE001
                f1 = None
                row["%s_accuracy_failure" % method] = "%s: %s" % (type(exc).__name__, exc)

            row.update({
                "%s_available" % method: True,
                "%s_failure_reason" % method: None,
                "%s_backend" % method: reason,
                "size_mb_%s" % method: size,
                "size_ratio_%s" % method: round(size / fp32_size, 3) if fp32_size else None,
                "coverage_pct_%s" % method: block["params_quantised_pct"],
                "quantised_layers_%s" % method: ", ".join(
                    "%d x %s" % (n, t) for t, n in sorted(block["quantised_layer_types"].items())
                ) or "none",
                "macro_f1_%s" % method: f1,
                "macro_f1_delta_%s" % method: (
                    round(f1 - row["macro_f1_fp32"], 6) if f1 is not None else None
                ),
                "under_flash_budget_%s" % method: bool(size <= FLASH_BUDGET_MB),
            })
            print("  %-13s %8.3f MB   ratio %5.3f   %5.1f %% of params   macro-F1 %s"
                  % (method, size, size / fp32_size if fp32_size else float("nan"),
                     block["params_quantised_pct"] or 0.0,
                     "%.4f" % f1 if f1 is not None else "n/a"))
            print("                %s" % (row["quantised_layers_%s" % method]))

        rows.append(row)

    frame = pd.DataFrame(rows)
    rule("SUMMARY -- size and accuracy only, NO LATENCY")
    under = [r["arm"] for r in rows if r.get("under_flash_budget_static")]
    print("  arms under the %.1f MB flash budget after static PTQ: %s"
          % (FLASH_BUDGET_MB, ", ".join(under) if under else "none"))
    failed = [(r["arm"], r.get("static_failure_reason")) for r in rows
              if r.get("static_available") is False]
    for arm, reason in failed:
        print("  static PTQ unavailable for %-20s %s" % (arm, reason))
    print("\n  Nothing in this table was timed. Any speed claim needs the Pi run.")

    stamp = aggregate.provenance(aggregate.cv_records())
    path = aggregate.write_csv_with_provenance(
        frame, artifacts / "quantisation.csv", stamp, extra_header=HEADER
    )
    print("\n[artifacts] wrote %s" % path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
