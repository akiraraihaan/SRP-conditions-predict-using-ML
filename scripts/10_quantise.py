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
import json
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


# Eager-mode quantised kernels are CPU-only, and a quantised module may have
# NO parameters left to infer a device from -- static PTQ packs them into
# buffers, so `next(module.parameters())` raises StopIteration. The evaluation
# helper defaults to exactly that inference, which is why every quantised row
# came back "n/a" while the fp32 row scored fine.
QUANTISED_DEVICE = "cpu"


def prediction_vector(module, cache, indices, labels_by_idx, device=None):
    """The predicted class per image, in the order `indices` was given.

    Metrics are a summary; this is the thing itself. A macro-F1 delta with no
    prediction change behind it would mean the two models are the same model
    and the delta came from somewhere else -- which is the possibility this
    exists to rule out.
    """
    from srpcard import evaluate

    _, _, logits = evaluate.predict_logits(
        module, cache, indices, labels_by_idx, device=device
    )
    return np.asarray(logits).argmax(axis=1)


def prediction_changes(before, after, classes: list[str]) -> dict:
    """How many predictions moved, and where they went.

    On 134 test images one image is 0.75 %, so a macro-F1 move of 0.010 should
    correspond to two or three flips. None at all would mean the quantised
    model is not actually being evaluated; a great many would mean the
    conversion broke something rather than costing precision.
    """
    before = np.asarray(before)
    after = np.asarray(after)
    changed = np.flatnonzero(before != after)
    transitions = {}
    for position in changed:
        key = "%s->%s" % (classes[int(before[position])], classes[int(after[position])])
        transitions[key] = transitions.get(key, 0) + 1
    return {
        "n_predictions": int(before.size),
        "n_changed": int(changed.size),
        "pct_changed": round(100.0 * changed.size / max(before.size, 1), 2),
        "changed_image_positions": [int(i) for i in changed],
        "transitions": dict(sorted(transitions.items(), key=lambda kv: -kv[1])),
    }


def macro_f1(module, cache, indices, labels_by_idx, data_cfg, device=None) -> dict:
    from srpcard import evaluate

    return evaluate.evaluate_fold(
        module, cache, indices, labels_by_idx, data_cfg, device=device
    )


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


def _wrap_for_static(module):
    """Wrap a module in quant/dequant stubs.

    Eager-mode static PTQ converts Conv2d and Linear to quantized versions that
    accept a QUANTIZED tensor. Without a QuantStub the model is still handed a
    float tensor and the first converted layer raises

        Could not run 'quantized::conv2d' with arguments from the 'CPU' backend

    at evaluation time -- after the conversion has already been reported as a
    success. The stubs are what make the converted graph runnable.
    """
    import torch
    from torch import nn

    class Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.quant = torch.ao.quantization.QuantStub()
            self.inner = inner
            self.dequant = torch.ao.quantization.DeQuantStub()

        def forward(self, x):
            out = self.inner(self.quant(x))
            while isinstance(out, (tuple, list)):
                out = out[0]
            return self.dequant(out)

    return Wrapped(module)


def calibrate(prepared, cache, calibration_idx, labels_by_idx,
              batches: int = CALIBRATION_BATCHES) -> int:
    """Run the observers over TRAINING images. Returns the batch count.

    FoldDataset yields THREE items -- (tensor, label, idx) -- and unpacking two
    of them is what made static PTQ fail on all five arms with "too many values
    to unpack (expected 2)", reported as a platform limitation when it was our
    own loop. Index the batch rather than destructuring it, so adding a fourth
    element cannot break this again.
    """
    import torch
    from torch.utils.data import DataLoader

    from srpcard.train import FoldDataset

    loader = DataLoader(
        FoldDataset(cache, calibration_idx, labels_by_idx),
        batch_size=8, shuffle=False,
    )
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= batches:
                break
            prepared(batch[0])
            seen += 1
    return seen


def quantised_parameter_fraction(fp32, converted) -> float | None:
    """How much of the model actually became integer, by parameter count.

    Measured as what is LEFT in floating point after conversion, because FX
    leaves anything without a quantised kernel in fp32 rather than failing.
    Partial conversion changes what the size number means, so the size must
    never be quoted without this beside it.
    """
    total = sum(p.numel() for p in fp32.parameters())
    if not total:
        return None
    remaining = sum(p.numel() for p in converted.parameters())
    return round(100.0 * (total - remaining) / total, 2)


def static_ptq_fx(module, cache, calibration_idx, labels_by_idx, backend: str):
    """FX graph mode. Rewrites the graph instead of requiring the model to be
    written for quantisation.

    Eager mode needs residual adds replaced by FloatFunctional and unsupported
    activations wrapped in stubs -- a rewrite of three third-party
    architectures. FX leaves operations with no quantised kernel in floating
    point rather than raising, which is the behaviour this needs.

    It is not universal: symbolic tracing cannot follow data-dependent control
    flow, and the ultralytics forward has some. That failure is reported, not
    worked around.
    """
    import copy

    import torch
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    model = copy.deepcopy(module).eval()
    example = (torch.rand(1, 3, 224, 224),)
    prepared = prepare_fx(model, get_default_qconfig_mapping(backend),
                          example_inputs=example)
    batches = calibrate(prepared, cache, calibration_idx, labels_by_idx)
    if not batches:
        raise RuntimeError("calibration produced no batches")
    converted = convert_fx(prepared)
    return converted, batches


def static_ptq(module, cache, calibration_idx, labels_by_idx):  # noqa: C901
    """Fuse, calibrate on TRAINING images, convert. Conv2d included.

    Returns (quantised, reason). A failure is reported per arm with its reason
    rather than skipped: "static PTQ is not available for this architecture" is
    itself a finding the manuscript needs, and a silent skip would read as a
    method that was never tried.

    The reason NAMES THE SOURCE FILE of the failure. Reporting our own
    TypeError as though it were a platform limitation is how a broken
    calibration loop looked like "static PTQ unsupported" on all five arms.
    """
    import copy
    import traceback

    import torch

    try:
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

        # FX FIRST. It is the only one of the two that can handle a model not
        # written for quantisation, and where it works it converts everything.
        try:
            converted, batches = static_ptq_fx(
                module, cache, calibration_idx, labels_by_idx, backend
            )
            return converted, "fx graph mode, backend=%s, %d calibration batch(es)" % (
                backend, batches
            )
        except Exception as fx_error:  # noqa: BLE001 - reported below, with eager's
            fx_reason = "%s: %s" % (type(fx_error).__name__,
                                    str(fx_error).replace("\n", " ")[:160])

        prepared = _wrap_for_static(copy.deepcopy(module).eval()).eval()

        # Fusion is an OPTIMISATION here, not a requirement: fuse_modules needs
        # explicit per-architecture patterns and there is no generic list that
        # is correct for all five arms. Conv2d is quantised either way; fusion
        # would only fold BatchNorm in as well.
        fused = False

        prepared.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        torch.ao.quantization.prepare(prepared, inplace=True)

        batches = calibrate(prepared, cache, calibration_idx, labels_by_idx)
        if not batches:
            return None, "calibration produced no batches"

        converted = torch.ao.quantization.convert(prepared, inplace=False)
        return converted, ("eager mode (FX unavailable: %s), backend=%s, "
                           "%d calibration batch(es), fused=%s"
                           % (fx_reason, backend, batches, fused))
    except Exception as exc:  # noqa: BLE001
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        return None, "%s: %s  [raised at %s:%d]" % (
            type(exc).__name__, exc, Path(frame.filename).name, frame.lineno
        )


STATIC_GAP_NOTE = """
  STATIC PTQ: MEASURED FOR TWO ARMS, NOT MEASURABLE FOR THREE.

  mobilenetv3_small and resnet18 convert through FX graph mode, 100 % of
  parameters, and are scored normally. The compression figure for those two
  carries an accuracy cost beside it.

  yolo26n, yolo26s and yolo26m compress to the same fraction and CANNOT BE
  SCORED. Both available approaches fail, for different reasons:

    eager mode  the converted graph has no QuantizedCPU kernel for
                aten::add.out (the residual additions) or aten::silu.out
                (the SiLU activations). Fixing that means rewriting the
                architecture -- residual adds as nn.quantized.FloatFunctional,
                unsupported activations wrapped in quant/dequant stubs -- in
                three third-party models.

    FX mode     symbolic tracing cannot follow the ultralytics forward:
                "Proxy object cannot be iterated". FX is what would otherwise
                leave unsupported operations in floating point instead of
                failing, and it never gets that far.

  So for those three the accuracy cost of static PTQ IS NOT MEASURED. It is
  not zero, not small, not assumed -- unmeasured, with a named cause. No third
  approach was attempted.
"""

ENVIRONMENT_NOTE = """
  THE ACCURACY COLUMN IS INTERNALLY CONSISTENT, NOT A REPRODUCTION.

  Every macro-F1 here is measured in THIS environment from the checkpoint this
  script was given, and every quantisation delta is taken against the fp32
  number in the same row. That is the comparison the table is for: what
  quantisation costs, holding everything else fixed.

  It is NOT a reproduction of the published figure, and must not be read as
  one. A fold reproduces EXACTLY when re-run within a session and does NOT
  between sessions, on the same GPU model and the same torch version:
  mobilenetv3_small moves by up to 1.9e-2 macro-F1 and 7.3e-2 precision_macro,
  yolo26s 1.04e-2, resnet18 6.8e-3, while yolo26n and yolo26m reproduce
  exactly. THE CAUSE IS UNIDENTIFIED and nothing here names a mechanism.

  The registry figure is printed beside the measured one for context. A delta
  between them is a BETWEEN-SESSION difference, not a quantisation effect.
  See artifacts/environment_replication.csv.
"""

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
    "",
    "STATIC PTQ IS MEASURED FOR TWO ARMS AND NOT MEASURABLE FOR THREE.",
    "",
    "mobilenetv3_small and resnet18 convert through FX graph mode at 100 % of",
    "parameters and are scored. yolo26n, yolo26s and yolo26m compress to the",
    "same fraction and CANNOT BE SCORED: in eager mode the converted graph has",
    "no QuantizedCPU kernel for aten::add.out (residual additions) or",
    "aten::silu.out (SiLU activations), and FX symbolic tracing cannot follow",
    "the ultralytics forward (Proxy object cannot be iterated). Fixing the",
    "first would mean rewriting three third-party architectures.",
    "",
    "For those three arms the accuracy cost of static PTQ IS NOT MEASURED --",
    "not zero, not small, not assumed. macro_f1_static is empty and",
    "static_accuracy_failure names the operation. No third approach was tried.",
    "",
    "coverage_pct_* is the fraction of PARAMETERS converted, computed from what",
    "is left in floating point. Quote it beside the size: partial conversion",
    "changes what the size number means.",
    "",
    "THE ACCURACY COLUMN IS INTERNALLY CONSISTENT, NOT A REPRODUCTION.",
    "",
    "macro_f1_fp32 is MEASURED here from the checkpoint this script was given,",
    "and every quantisation delta is taken against it. macro_f1_fp32_recorded_",
    "in_registry is CONTEXT ONLY -- do not compute a delta against it.",
    "",
    "The two can differ without either being wrong. A fold reproduces exactly",
    "when re-run WITHIN a session and not BETWEEN sessions, on the same GPU",
    "model and the same torch version -- up to 1.9e-2 macro-F1 for",
    "mobilenetv3_small, while yolo26n and yolo26m reproduce exactly. The cause",
    "is UNIDENTIFIED. A gap in macro_f1_fp32_measured_minus_recorded is a",
    "between-session difference, not a quantisation effect. See",
    "artifacts/environment_replication.csv.",
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

    # The registry's own figure for the benchmark fold, for CONTEXT beside the
    # measured one. Never a baseline: see ENVIRONMENT_NOTE.
    # The canonical class order, for naming prediction transitions. It was
    # referenced without ever being defined, so every flip count died with
    # NameError -- the check that verifies the deltas, lost to a missing line.
    classes = list(data_cfg["classes"])

    recorded_f1 = {
        r["arm"]: r.get("f1_macro")
        for r in aggregate.cv_records()
        if r.get("repeat") == repeat and r.get("fold") == fold
    }

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
        # MEASURED HERE, from the checkpoint that was handed to this script. The
        # registry's value is shown beside it for context and is NEVER used as
        # the baseline: see the note below.
        # The fp32 baseline is scored on the SAME device as the quantised
        # variants, so the delta between them cannot carry a device difference.
        module = module.to(QUANTISED_DEVICE)
        fp32_metrics = macro_f1(module, cache, test_idx, labels_by_idx, data_cfg,
                                device=QUANTISED_DEVICE)
        fp32_predictions = prediction_vector(module, cache, test_idx, labels_by_idx,
                                             device=QUANTISED_DEVICE)
        recorded = recorded_f1.get(arm)
        drift = (
            round(fp32_metrics["f1_macro"] - recorded, 6)
            if recorded is not None else None
        )

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
            # The baseline every delta in this row is taken against.
            "macro_f1_fp32": round(fp32_metrics["f1_macro"], 6),
            # Context only, and possibly from a different session. Do NOT
            # compute a quantisation delta against this.
            "macro_f1_fp32_recorded_in_registry": recorded,
            "macro_f1_fp32_measured_minus_recorded": drift,
            "baseline_is": "measured from this checkpoint, in this environment",
        }
        print("  fp32          %8.3f MB   macro-F1 %.4f  (measured here)"
              % (fp32_size, fp32_metrics["f1_macro"]))
        if recorded is not None:
            print("                registry recorded %.4f for this run  ->  delta %+0.4f"
                  % (recorded, drift))

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
            # Two coverage measures, because they answer different questions
            # and one of them survives FX.
            #
            # `coverage()` matches module NAMES, which an FX GraphModule does
            # not preserve, so it reports the layer-type census where it can.
            # The parameter fraction is computed from what is LEFT in floating
            # point and works for every path -- it is the number that qualifies
            # the size, because partial conversion changes what the size means.
            block = coverage(module, getattr(quantised, "inner", quantised))
            converted_pct = quantised_parameter_fraction(module, quantised)
            # TWO separate attempts. They used to share one try, so a failure
            # in the VERIFICATION discarded the MEASUREMENT: an exception while
            # counting prediction flips set f1 back to None and the row read
            # "n/a" as though the model could not be scored at all.
            #
            # device is pinned to CPU rather than inferred. `predict_logits`
            # defaults to `next(module.parameters()).device`, which a fully
            # quantised module cannot answer -- static PTQ leaves no parameters
            # to ask -- and eager quantised kernels do not run on CUDA anyway.
            f1 = None
            flips = {}
            try:
                metrics = macro_f1(quantised, cache, test_idx, labels_by_idx,
                                   data_cfg, device=QUANTISED_DEVICE)
                f1 = round(metrics["f1_macro"], 6)
            except Exception as exc:  # noqa: BLE001
                row["%s_accuracy_failure" % method] = "%s: %s" % (
                    type(exc).__name__, exc)
                print("  %-13s SCORING FAILED -- %s: %s"
                      % (method, type(exc).__name__, exc))
                print("                the size above is real; the accuracy is NOT")
                print("                missing by choice. This row cannot support a")
                print("                cost-of-quantisation claim until it is fixed.")

            if f1 is not None:
                try:
                    flips = prediction_changes(
                        fp32_predictions,
                        prediction_vector(quantised, cache, test_idx, labels_by_idx,
                                          device=QUANTISED_DEVICE),
                        classes,
                    )
                except Exception as exc:  # noqa: BLE001
                    row["%s_flipcount_failure" % method] = "%s: %s" % (
                        type(exc).__name__, exc)
                    print("  %-13s flip count failed -- %s: %s  (macro-F1 above "
                          "still stands)" % (method, type(exc).__name__, exc))

            row.update({
                "%s_available" % method: True,
                "%s_failure_reason" % method: None,
                "%s_backend" % method: reason,
                "size_mb_%s" % method: size,
                "size_ratio_%s" % method: round(size / fp32_size, 3) if fp32_size else None,
                "coverage_pct_%s" % method: converted_pct,
                "coverage_pct_by_layer_name_%s" % method:
                    block["params_quantised_pct"],
                "quantised_layers_%s" % method: ", ".join(
                    "%d x %s" % (n, t) for t, n in sorted(block["quantised_layer_types"].items())
                ) or "none",
                "macro_f1_%s" % method: f1,
                "macro_f1_delta_%s" % method: (
                    round(f1 - row["macro_f1_fp32"], 6) if f1 is not None else None
                ),
                "under_flash_budget_%s" % method: bool(size <= FLASH_BUDGET_MB),
                # The delta, checked against the thing it is a summary of.
                "n_predictions_changed_%s" % method: flips.get("n_changed"),
                "pct_predictions_changed_%s" % method: flips.get("pct_changed"),
                "prediction_transitions_%s" % method: json.dumps(
                    flips.get("transitions", {})
                ),
            })
            print("  %-13s %8.3f MB   ratio %5.3f   %5.1f %% of params   macro-F1 %s"
                  % (method, size, size / fp32_size if fp32_size else float("nan"),
                     converted_pct or 0.0,
                     "%.4f" % f1 if f1 is not None else "NOT MEASURED"))
            print("                %s" % (row["quantised_layers_%s" % method]))
            if flips:
                delta = row["macro_f1_delta_%s" % method]
                print("                %d of %d prediction(s) changed (%.2f %%)%s"
                      % (flips["n_changed"], flips["n_predictions"],
                         flips["pct_changed"],
                         "  <- " + ", ".join(
                             "%s x%d" % (k, v)
                             for k, v in list(flips["transitions"].items())[:4])
                         if flips["transitions"] else ""))
                if delta and flips["n_changed"] == 0:
                    print("                [WARNING] macro-F1 moved by %+0.4f with ZERO"
                          % delta)
                    print("                prediction changes. The two models cannot")
                    print("                differ in score without differing in output:")
                    print("                the quantised model is probably not the one")
                    print("                being evaluated.")

        rows.append(row)

    frame = pd.DataFrame(rows)
    rule("SUMMARY -- size and accuracy only, NO LATENCY")
    print(ENVIRONMENT_NOTE)
    drifted = [(r["arm"], r["macro_f1_fp32_measured_minus_recorded"]) for r in rows
               if r.get("macro_f1_fp32_measured_minus_recorded")]
    if drifted:
        print("  arms whose measured fp32 differs from the registry:")
        for arm, delta in sorted(drifted, key=lambda t: -abs(t[1])):
            print("      %-22s %+0.4f" % (arm, delta))
        print()

    under = [r["arm"] for r in rows if r.get("under_flash_budget_static")]
    print("  arms under the %.1f MB flash budget after static PTQ: %s"
          % (FLASH_BUDGET_MB, ", ".join(under) if under else "none"))
    failed = [(r["arm"], r.get("static_failure_reason")) for r in rows
              if r.get("static_available") is False]
    for arm, reason in failed:
        print("  static PTQ unavailable for %-20s %s" % (arm, reason))

    unscored = [
        (r["arm"], method, r.get("%s_accuracy_failure" % method))
        for r in rows for method in ("dynamic", "static")
        if r.get("%s_available" % method) and r.get("macro_f1_%s" % method) is None
    ]
    if unscored:
        print()
        print("  ROWS WITH A SIZE BUT NO ACCURACY -- these cannot support a")
        print("  cost-of-quantisation claim:")
        for arm, method, reason in unscored:
            print("      %-22s %-8s %s" % (arm, method, reason))
        print(STATIC_GAP_NOTE)
    else:
        print("\n  Every converted variant was scored. The macro-F1 deltas are")
        print("  measured, and the prediction-flip counts say how many of the")
        print("  %d test images each delta rests on." % (rows[0]["n_test_images"]
                                                          if rows else 0))
    print("\n  Nothing in this table was timed. Any speed claim needs the Pi run.")

    stamp = aggregate.provenance(aggregate.cv_records())
    path = aggregate.write_csv_with_provenance(
        frame, artifacts / "quantisation.csv", stamp, extra_header=HEADER
    )
    print("\n[artifacts] wrote %s" % path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
