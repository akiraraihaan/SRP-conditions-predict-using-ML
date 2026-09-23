#!/usr/bin/env python
"""07 -- edge benchmark, ALL FIVE ARMS. STANDALONE. Runs on a Raspberry Pi.

    python scripts/07_bench_edge.py --weights-dir exported --images data/test --cooling passive
    python scripts/07_bench_edge.py --weights-dir exported --images data/test --soak-minutes 0

Inference needs no training, so every arm is benchmarked, not only the selected
one. That gives a SECOND, directly measured cost axis to set against GFLOPs, and
the Pareto frontier is recomputed on it. FLOPs and latency need not agree --
depthwise-separable operators are cheap in FLOPs and often poorly served by CPU
kernels -- so this is a result in its own right, not a formality.

No training. No CUDA -- the script refuses to use a GPU even if one is visible,
because these numbers describe the deployment target. It does not import
srpcard.train, ultralytics' trainer, or anything that needs the full dependency
set: torch (CPU), torchvision, pillow, numpy, psutil and PyYAML are enough.

Times the FULL inference pipeline, exactly as deployed:

    file read -> letterbox -> normalise -> forward -> label

Batch size 1. 50 warm-up iterations, discarded. At least 200 timed iterations.
Reports median, inter-quartile range and p95, twice: with letterbox and without
(model forward only), so the preprocessing cost is separable.

Also runs a soak loop -- 10 minutes by default -- and reports whether the median
drifts between the first and last thirds, which is how thermal throttling shows
up on a passively cooled board.

Per arm it measures, at batch size 1, after 50 discarded warm-up iterations and
over at least 200 timed ones:

  - the FULL pipeline: file read -> letterbox -> forward -> label
  - the forward pass alone, so the letterbox cost is visible in BOTH absolute
    milliseconds and as a share. The share is a property of the host: on a
    slower CPU the forward pass grows more than the file read and the resize,
    so the same model shows a smaller share on a Pi than on a workstation.
    Report the share from the Pi run, never from a workstation rehearsal.
  - median, IQR and p95 -- NOT the mean. On an edge device the tail is what
    disrupts operations, and a mean hides it.
  - peak resident memory
  - INT8 size AND the macro-F1 change after quantisation, on the same images, so
    the microcontroller claim rests on a measurement rather than a ratio quoted
    from the literature
  - a sustained loop, reporting whether the median drifts between the first and
    last minute, with the CPU temperature trace where the board exposes it

Every measurement records which checkpoint file produced it. A missing
checkpoint stops the run: latency and size are decided by the architecture and
would look perfectly plausible from an untrained model, while the INT8 accuracy
delta would be meaningless, and nothing in the output would show which.

The device block -- CPU, cores, OS, kernel, python, torch, threads, governor,
COOLING, iteration counts, timestamp -- goes into edge_benchmark.json and is
printed. Latency without it is not interpretable.

Dependencies are deliberately small enough for a Pi: torch (CPU), torchvision,
pillow, numpy, psutil and PyYAML. No pandas, no ultralytics trainer, no CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

# CPU only, decided before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srpcard.config import (  # noqa: E402
    published_arms,
    artifacts_dir,
    load_arms_config,
    load_data_config,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
WARMUP = 50
MIN_TIMED = 200
IMAGE_SIZE = 224

# Above this the board is throttling and every number taken from that point on
# is a measurement of the cooling, not of the model. A Raspberry Pi 3 with no
# heatsink sits here within a minute of starting work.
THROTTLE_WARN_C = 80.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def rule(title: str) -> None:
    print("\n" + "=" * 74 + "\n" + title + "\n" + "=" * 74)


# --------------------------------------------------------------------------
# host provenance
# --------------------------------------------------------------------------


def cpu_model() -> str:
    """Best-effort CPU name. /proc/cpuinfo first -- that is what a Pi exposes."""
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        for key in ("model name", "Model", "Hardware"):
            for line in text.splitlines():
                if line.startswith(key):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def power_mode() -> dict[str, str]:
    """Governor, clock and throttle state. Absent keys simply mean "not a Pi"."""
    info: dict[str, str] = {}
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor.exists():
        info["scaling_governor"] = governor.read_text().strip()
    for name, path in (
        ("cpu_max_freq_khz", "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"),
        ("cpu_cur_freq_khz", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
    ):
        candidate = Path(path)
        if candidate.exists():
            info[name] = candidate.read_text().strip()
    for label, command in (
        ("vcgencmd_throttled", ["vcgencmd", "get_throttled"]),
        ("vcgencmd_temp", ["vcgencmd", "measure_temp"]),
    ):
        try:
            out = subprocess.run(command, capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                info[label] = out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def host_provenance(threads: int) -> dict:
    import torch

    info = {
        "cpu_model": cpu_model(),
        "machine": platform.machine(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_threads": threads,
        "cpu_count": os.cpu_count(),
        "power_mode": power_mode(),
    }
    for name in ("torchvision", "numpy", "PIL"):
        try:
            info[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            info[name] = "not-installed"
    try:
        import psutil

        info["ram_total_mb"] = round(psutil.virtual_memory().total / (1024**2), 1)
    except ImportError:
        info["ram_total_mb"] = None
    return info


def peak_rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024**2), 2)
    except ImportError:
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports kB, macOS bytes
            return round(peak / 1024, 2) if sys.platform != "darwin" else round(peak / (1024**2), 2)
        except Exception:  # noqa: BLE001
            return None


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------


def letterbox(img, target: int = IMAGE_SIZE):
    from PIL import Image

    width, height = img.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - width) // 2, (side - height) // 2))
    return canvas.resize((target, target), Image.BILINEAR)


def to_tensor(img):
    import torch

    array = np.asarray(img, dtype=np.uint8)
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def load_model(weights: Path, classes: list[str]):
    """Load an exported model. Accepts TorchScript or a state_dict + arm name."""
    import torch

    try:
        module = torch.jit.load(str(weights), map_location="cpu")
        module.eval()
        return module, "torchscript"
    except Exception:  # noqa: BLE001 - fall through to the state_dict path
        pass

    payload = torch.load(str(weights), map_location="cpu", weights_only=False)
    if hasattr(payload, "eval"):
        payload.eval()
        return payload, "pickled_module"

    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    arm = payload.get("arm") if isinstance(payload, dict) else None
    if arm is None:
        raise SystemExit(
            "Could not infer the architecture from %s.\n"
            "  Export either a TorchScript module, or a dict with keys "
            "{'arm': <arm name>, 'state_dict': ...}." % weights
        )
    from srpcard.models import build_model

    bundle = build_model(arm, num_classes=len(classes), with_efficiency=False)
    bundle.module.load_state_dict(state)
    bundle.module.eval()
    return bundle.module, "state_dict:%s" % arm


def stats(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=float)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "n": int(values.size),
        "median_ms": round(float(median), 3),
        "iqr_ms": round(float(q3 - q1), 3),
        "q1_ms": round(float(q1), 3),
        "q3_ms": round(float(q3), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "mean_ms": round(float(values.mean()), 3),
        "min_ms": round(float(values.min()), 3),
        "max_ms": round(float(values.max()), 3),
    }


def _predict(module, tensor) -> None:
    """One forward pass and an argmax, the work a deployment actually does."""
    logits = module(tensor)
    while isinstance(logits, (tuple, list)):
        logits = logits[0]
    int(logits.argmax(dim=1).item())


def bench_interleaved(
    module, image_paths: list[Path], iterations: int
) -> tuple[list[float], list[float]]:
    """Time both scopes on the SAME iterations, alternating between them.

    Measuring all of one scope and then all of the other makes the difference
    between them a measurement of whatever drifted in between. On a passively
    cooled board that drift is thermal, and it is larger than the quantity being
    measured: a Raspberry Pi 3 already at 80.6 C reported a full-pipeline median
    of 725.9 ms against a forward-only median of 768.2 ms -- a letterbox cost of
    -42.3 ms, which cannot happen, because the full pipeline CONTAINS the
    forward pass. Nothing was wrong with the model. The forward-only block
    simply ran second, on a chip that had climbed to 82.7 C.

    Interleaving gives both scopes the same thermal history, so the difference
    stays valid however hot the board gets, and it costs nothing: the same
    number of forward passes, in a different order.

    The ORDER alternates too. Whichever scope runs second inherits the other's
    cache and clock state, and always handing that position to the same scope
    would put a systematic bias back in by the back door.
    """
    import torch
    from PIL import Image

    with Image.open(image_paths[0]) as img:
        fixed = to_tensor(letterbox(img.convert("RGB")))

    full: list[float] = []
    forward: list[float] = []

    with torch.no_grad():
        for i in range(WARMUP + iterations):
            path = image_paths[i % len(image_paths)]

            def time_full():
                start = time.perf_counter()
                with Image.open(path) as img:                      # file read
                    tensor = to_tensor(letterbox(img.convert("RGB")))  # letterbox
                _predict(module, tensor)                           # forward
                return (time.perf_counter() - start) * 1000.0

            def time_forward():
                start = time.perf_counter()
                _predict(module, fixed)
                return (time.perf_counter() - start) * 1000.0

            if i % 2 == 0:
                full_ms = time_full()
                forward_ms = time_forward()
            else:
                forward_ms = time_forward()
                full_ms = time_full()

            if i >= WARMUP:                            # discard the warm-up
                full.append(full_ms)
                forward.append(forward_ms)
    return full, forward


def letterbox_cost(full: dict, forward: dict) -> dict:
    """The letterbox cost, or a refusal when the measurement is not physical.

    The full pipeline contains the forward pass, so a negative difference is not
    a fast letterbox: it is a failed measurement. Interleaving removes the cause
    that produced one here, and this is the check that the cause stayed removed.
    Reporting a failure costs a cell in a table; writing -42.3 ms into one costs
    the reader's trust in every other number beside it.
    """
    difference = full["median_ms"] - forward["median_ms"]
    if difference < 0:
        return {
            "ok": False,
            "letterbox_ms": None,
            "letterbox_share_pct": None,
            "reason": (
                "forward-only median (%.3f ms) EXCEEDS the full-pipeline median "
                "(%.3f ms) by %.3f ms. The full pipeline contains the forward "
                "pass, so this is a failed measurement, not a result. The scopes "
                "are interleaved, so a drift big enough to survive that means "
                "the host was unstable for another reason -- check the timed-run "
                "temperatures and re-run on a cooler board."
                % (forward["median_ms"], full["median_ms"], -difference)
            ),
        }
    return {
        "ok": True,
        "letterbox_ms": round(difference, 3),
        "letterbox_share_pct": round(100.0 * difference / full["median_ms"], 1),
        "reason": None,
    }


def soak(module, image_paths: list[Path], minutes: float) -> dict:
    """Run continuously and report whether the median drifts -- thermal throttling.

    Reported by MINUTE, first against last, because that is the comparison a
    reader can act on: it answers "does this board still hit its latency target
    after ten minutes of continuous inference". The temperature trace is sampled
    once a second where the board exposes it.
    """
    import torch
    from PIL import Image

    deadline = time.perf_counter() + minutes * 60.0
    samples: list[float] = []
    stamps: list[float] = []
    temperatures: list[float] = []
    temp_stamps: list[float] = []
    started = time.perf_counter()
    next_temp = 0.0
    index = 0
    with torch.no_grad():
        while time.perf_counter() < deadline:
            path = image_paths[index % len(image_paths)]
            index += 1
            start = time.perf_counter()
            with Image.open(path) as img:
                tensor = to_tensor(letterbox(img.convert("RGB")))
            logits = module(tensor)
            while isinstance(logits, (tuple, list)):
                logits = logits[0]
            int(logits.argmax(dim=1).item())
            now = time.perf_counter()
            samples.append((now - start) * 1000.0)
            elapsed = now - started
            stamps.append(elapsed)
            if elapsed >= next_temp:
                reading = read_temperature_c()
                if reading is not None:
                    temperatures.append(reading)
                    temp_stamps.append(round(elapsed, 1))
                next_temp = elapsed + 1.0

    if len(samples) < 30:
        return {"ran": False, "reason": "too few iterations (%d)" % len(samples)}

    def window(low, high):
        chosen = [s for s, t in zip(samples, stamps) if low <= t < high]
        return chosen or None

    # A first/last MINUTE comparison needs at least two minutes, or the two
    # windows overlap and the drift is 0.00 % by construction rather than by
    # measurement -- which would read as "no throttling".
    duration = stamps[-1]
    long_enough = duration >= 120.0
    first_minute = window(0.0, 60.0) if long_enough else None
    last_minute = window(duration - 60.0, duration + 1.0) if long_enough else None
    third = len(samples) // 3

    result = {
        "ran": True,
        "minutes": minutes,
        "iterations": len(samples),
        "overall": stats(samples),
        "first_third_median_ms": round(float(np.median(samples[:third])), 3),
        "last_third_median_ms": round(float(np.median(samples[-third:])), 3),
        "temperature_c": temperatures,
        "temperature_at_s": temp_stamps,
        "temperature_start_c": temperatures[0] if temperatures else None,
        "temperature_end_c": temperatures[-1] if temperatures else None,
        "temperature_peak_c": max(temperatures) if temperatures else None,
        "temperature_readable": bool(temperatures),
        "power_mode_after": power_mode(),
    }
    if first_minute and last_minute:
        first_median = float(np.median(first_minute))
        last_median = float(np.median(last_minute))
        result.update({
            "minute_windows_used": True,
            "first_minute_median_ms": round(first_median, 3),
            "last_minute_median_ms": round(last_median, 3),
            "drift_ms_minutes": round(last_median - first_median, 3),
            "drift_pct_minutes": round(100.0 * (last_median - first_median) / first_median, 2),
        })
    else:
        # a soak shorter than two minutes cannot support a first/last comparison
        result.update({
            "minute_windows_used": False,
            "first_minute_median_ms": result["first_third_median_ms"],
            "last_minute_median_ms": result["last_third_median_ms"],
            "drift_ms_minutes": round(
                result["last_third_median_ms"] - result["first_third_median_ms"], 3),
            "drift_pct_minutes": round(
                100.0 * (result["last_third_median_ms"] - result["first_third_median_ms"])
                / result["first_third_median_ms"], 2),
            "note": "soak shorter than 2 minutes; first/last THIRD used instead of minutes",
        })
    # 5 % is conservative; sustained throttling on a passively cooled Pi is larger
    result["throttling_suspected"] = bool(result["drift_pct_minutes"] > 5.0)
    return result


# --------------------------------------------------------------------------
# thermal
# --------------------------------------------------------------------------


def read_temperature_c() -> float | None:
    """CPU temperature in degrees C, or None where it is not readable.

    Tries the Linux thermal zones first, then the Raspberry Pi firmware tool.
    Never raises: a board that will not report its temperature still produces a
    valid benchmark, it just cannot support a thermal claim.
    """
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            value = int(zone.read_text().strip())
        except Exception:  # noqa: BLE001
            continue
        # millidegrees on every board that uses this interface
        return round(value / 1000.0, 2) if value > 1000 else float(value)
    try:
        out = subprocess.run(
            ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0 and "=" in out.stdout:
            return round(float(out.stdout.split("=")[1].split("'")[0]), 2)
    except Exception:  # noqa: BLE001
        pass
    return None


# --------------------------------------------------------------------------
# accuracy, so the INT8 claim is measured rather than quoted
# --------------------------------------------------------------------------


def labelled_images(image_dir: Path, classes: list[str]) -> list[tuple[Path, int]]:
    """(path, label) pairs when `image_dir` holds one subdirectory per class.

    A flat directory is valid too -- it just means latency only, with no accuracy
    and therefore no INT8 accuracy delta. That is reported, not silently skipped.
    """
    pairs: list[tuple[Path, int]] = []
    for index, name in enumerate(classes):
        folder = image_dir / name
        if not folder.is_dir():
            return []
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                pairs.append((path, index))
    return pairs


def macro_f1(module, pairs: list[tuple[Path, int]], n_classes: int) -> dict:
    """Macro-F1 over labelled images, one at a time, exactly as deployed."""
    import torch
    from PIL import Image

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for path, truth in pairs:
            with Image.open(path) as img:
                tensor = to_tensor(letterbox(img.convert("RGB")))
            logits = module(tensor)
            while isinstance(logits, (tuple, list)):
                logits = logits[0]
            confusion[truth, int(logits.argmax(dim=1).item())] += 1

    f1s = []
    for c in range(n_classes):
        tp = float(confusion[c, c])
        fp = float(confusion[:, c].sum() - tp)
        fn = float(confusion[c, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "macro_f1": round(float(np.mean(f1s)), 6),
        "accuracy": round(float(np.trace(confusion) / max(confusion.sum(), 1)), 6),
        "n_images": int(confusion.sum()),
    }


def quantise_int8(module):
    """A dynamically quantised INT8 copy, or None with the reason.

    `torch.nn.Conv2d` is passed and IGNORED. quantize_dynamic converts Linear
    and the RNN family only; naming Conv2d in the set neither works nor errors,
    it simply does nothing. It is kept here so that the day dynamic conv support
    lands, this picks it up -- and documented so nobody reads its presence as a
    claim that convolutions were quantised.
    """
    import torch

    try:
        return (
            torch.ao.quantization.quantize_dynamic(
                module, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def layer_census(module) -> dict[str, int]:
    """How many leaf modules of each type, e.g. {"Conv2d": 20, "Linear": 1}."""
    census: dict[str, int] = {}
    for _, child in module.named_modules():
        if list(child.children()):
            continue                      # containers, not layers
        name = type(child).__name__
        census[name] = census.get(name, 0) + 1
    return census


def quantisation_coverage(fp32, quantised) -> dict:
    """Which layer types quantize_dynamic actually converted, and which it left.

    The manuscript's microcontroller argument rests on quantisation, so a reader
    needs to know which architectures it can compress and which it cannot. Left
    unexplained, resnet18's size ratio of 1.0 reads as a broken measurement. It
    is not: dynamic quantisation converts Linear and the RNN family, ResNet18 is
    almost entirely Conv2d, and 11.18 M of its 11.18 M parameters are therefore
    untouched. A mobilenet head is a Linear, so it compresses a little. Nothing
    here compresses the way an all-Linear model would.
    """
    before = layer_census(fp32)
    converted: dict[str, int] = {}
    originals = dict(fp32.named_modules())
    for name, child in quantised.named_modules():
        if not type(child).__module__.startswith(("torch.ao.nn.quantized",
                                                  "torch.nn.quantized")):
            continue
        original = originals.get(name)
        if original is None:
            # Internal plumbing the conversion adds, e.g. fc._packed_params.
            # It has no counterpart in the fp32 module and is not a layer.
            continue
        converted[type(original).__name__] = converted.get(
            type(original).__name__, 0
        ) + 1

    quantised_params = 0
    for name, child in fp32.named_modules():
        if type(child).__name__ in converted and not list(child.children()):
            quantised_params += sum(p.numel() for p in child.parameters(recurse=False))
    total_params = sum(p.numel() for p in fp32.parameters())

    untouched = {
        layer: count
        for layer, count in before.items()
        if layer not in converted and count
    }
    return {
        "layer_census": before,
        "quantised_layer_types": converted or {},
        "unquantised_layer_types": untouched,
        "params_total": total_params,
        "params_in_quantised_layers": quantised_params,
        "params_quantised_pct": (
            round(100.0 * quantised_params / total_params, 2) if total_params else None
        ),
        "method": "torch.ao.quantization.quantize_dynamic(dtype=qint8)",
        "method_scope": (
            "Dynamic quantisation converts Linear and the RNN family only. "
            "Conv2d is NOT supported and is silently left in fp32, so a "
            "convolution-dominated architecture is barely compressed and a size "
            "ratio near 1.0 is the correct result, not a failed measurement."
        ),
    }


def _describe_coverage(coverage: dict) -> str:
    """One sentence naming what was quantised and what was not."""
    converted = coverage.get("quantised_layer_types") or {}
    untouched = coverage.get("unquantised_layer_types") or {}
    if not converted:
        return (
            "no layer was quantised (%s left in fp32)"
            % ", ".join("%d x %s" % (n, t) for t, n in sorted(untouched.items()))
        )
    return (
        "quantised %s, covering %s %% of parameters; left %s in fp32"
        % (
            ", ".join("%d x %s" % (n, t) for t, n in sorted(converted.items())),
            coverage.get("params_quantised_pct"),
            ", ".join("%d x %s" % (n, t) for t, n in sorted(untouched.items())) or "nothing",
        )
    )


def int8_report(module, out_path: Path, pairs, n_classes: int) -> dict:
    """INT8 size AND the macro-F1 change, on the same images.

    The microcontroller argument rests on both halves. A size ratio quoted from
    the literature is not a result for this model on this data; an accuracy that
    collapses under quantisation would make the size irrelevant. So both are
    measured here, on the same images, with the same pipeline.
    """
    import torch

    quantised, reason = quantise_int8(module)
    if quantised is None:
        return {"available": False, "reason": reason}

    torch.save(quantised.state_dict(), out_path)
    report = {
        "available": True,
        "int8_size_mb": round(out_path.stat().st_size / (1024**2), 3),
        "path": str(out_path),
    }
    fp32_path = out_path.with_name(out_path.stem + "_fp32.pt")
    torch.save(module.state_dict(), fp32_path)
    report["fp32_size_mb"] = round(fp32_path.stat().st_size / (1024**2), 3)
    report["size_ratio"] = (
        round(report["int8_size_mb"] / report["fp32_size_mb"], 3)
        if report["fp32_size_mb"]
        else None
    )
    fp32_path.unlink(missing_ok=True)

    report["coverage"] = quantisation_coverage(module, quantised)
    ratio = report["size_ratio"]
    if ratio is not None and ratio > 0.95:
        report["size_ratio_note"] = (
            "A ratio this close to 1.0 means quantisation achieved essentially "
            "nothing, and that is the CORRECT result for this architecture, not "
            "a bug: %s. %s"
            % (
                _describe_coverage(report["coverage"]),
                report["coverage"]["method_scope"],
            )
        )
    else:
        report["size_ratio_note"] = _describe_coverage(report["coverage"])

    if not pairs:
        report["accuracy_measured"] = False
        report["reason_no_accuracy"] = (
            "--images is a flat directory, so there are no labels. Point it at a "
            "directory with one subdirectory per class to measure the INT8 "
            "accuracy change."
        )
        return report

    before = macro_f1(module, pairs, n_classes)
    after = macro_f1(quantised, pairs, n_classes)
    report.update(
        {
            "accuracy_measured": True,
            "n_images": before["n_images"],
            "macro_f1_fp32": before["macro_f1"],
            "macro_f1_int8": after["macro_f1"],
            "macro_f1_delta": round(after["macro_f1"] - before["macro_f1"], 6),
            "accuracy_fp32": before["accuracy"],
            "accuracy_int8": after["accuracy"],
        }
    )
    return report


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------


def resolve_checkpoints(weights_dir: Path, arms: list[str]) -> dict[str, Path]:
    """One checkpoint per arm, or a refusal naming exactly what is missing.

    Benchmarking an untrained model would produce latency and size numbers that
    look entirely correct -- architecture decides those -- while the INT8
    accuracy delta would be meaningless noise. There is no way to tell from the
    output afterwards, so a missing checkpoint stops the run.
    """
    found: dict[str, Path] = {}
    missing: list[str] = []
    for arm in arms:
        candidates = [
            weights_dir / ("%s.pt" % arm),
            weights_dir / ("%s.pth" % arm),
            weights_dir / arm / "best.pt",
        ]
        match = next((c for c in candidates if c.exists()), None)
        if match is None:
            missing.append(arm)
        else:
            found[arm] = match
    if missing:
        raise SystemExit(
            "No checkpoint for %d arm(s): %s\n"
            "  looked in %s for <arm>.pt, <arm>.pth or <arm>/best.pt\n"
            "\n"
            "  Refusing to benchmark an untrained model. Latency, memory and INT8\n"
            "  size are decided by the architecture and would look perfectly\n"
            "  plausible; the INT8 accuracy delta would be meaningless, and\n"
            "  nothing in the output would show which.\n"
            "\n"
            "  NOTE: scripts 03-05 do not persist weights -- train_fold returns\n"
            "  best_state in memory and the run records only metrics. Re-run one\n"
            "  fold per arm with 03_run_cv.py --save-weights <dir> to produce\n"
            "  them, or pass --arms to benchmark only the arms you have."
            % (len(missing), ", ".join(missing), weights_dir)
        )
    return found


def file_sha1(path: Path) -> str:
    import hashlib

    digest = hashlib.sha1()  # noqa: S324 - content addressing, not security
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------
# the Pareto frontier, recomputed on MEASURED cost
# --------------------------------------------------------------------------


def registry_mean_f1(registry_path: Path, script: str = "03_run_cv") -> dict[str, float]:
    """Mean test macro-F1 per arm, straight from the registry.

    Read with the json module rather than pandas: this script has to install on a
    Raspberry Pi, and its dependency set is deliberately torch, torchvision,
    pillow, numpy, psutil and PyYAML.
    """
    if not registry_path.exists():
        return {}
    totals: dict[str, list[float]] = {}
    with open(registry_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("script") != script or record.get("f1_macro") is None:
                continue
            totals.setdefault(record["arm"], []).append(float(record["f1_macro"]))
    return {arm: sum(v) / len(v) for arm, v in totals.items() if v}


def dominance(rows: dict[str, dict], maximise: tuple, minimise: tuple) -> dict[str, dict]:
    """Who dominates whom. Same rule as aggregate.pareto_status, on other axes."""
    status = {}
    for arm, values in rows.items():
        dominators = {}
        for other, theirs in rows.items():
            if other == arm:
                continue
            better = []
            worse = False
            for field in maximise:
                if theirs[field] < values[field]:
                    worse = True
                elif theirs[field] > values[field]:
                    better.append(field)
            for field in minimise:
                if theirs[field] > values[field]:
                    worse = True
                elif theirs[field] < values[field]:
                    better.append(field)
            if not worse and better:
                dominators[other] = better
        status[arm] = {
            "on_frontier": not dominators,
            "dominated_by": dominators,
        }
    return status


def write_device_pareto(results: dict, registry_path: Path, target: Path,
                        gflops_pareto: Path) -> dict:
    """artifacts/pareto_status_device.csv, and whether the frontier moved.

    The GFLOPs frontier is a MODELLED cost. This one is measured on the hardware
    the claim is about. FLOPs and latency disagree in general -- depthwise
    separable convolutions are the usual example, cheap in FLOPs and poorly
    served by most CPU kernels -- so agreement between the two is a result and
    disagreement is a bigger one.
    """
    import csv

    means = registry_mean_f1(registry_path)
    rows = {}
    for arm, payload in results.items():
        if arm not in means:
            continue
        int8 = payload.get("int8") or {}
        rows[arm] = {
            "f1_macro_mean": round(means[arm], 6),
            "params": payload.get("params"),
            "latency_median_ms": payload["full_pipeline"]["median_ms"],
            "int8_size_mb": int8.get("int8_size_mb"),
        }
    rows = {a: v for a, v in rows.items() if all(x is not None for x in v.values())}
    if not rows:
        return {"written": False, "reason": "no arm has both a registry F1 and a measurement"}

    status = dominance(
        rows,
        maximise=("f1_macro_mean",),
        minimise=("params", "latency_median_ms", "int8_size_mb"),
    )

    with open(target, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "arm", "f1_macro_mean", "params", "latency_median_ms", "int8_size_mb",
            "on_pareto_frontier", "dominated_by", "dominated_on",
        ])
        for arm in sorted(rows, key=lambda a: -rows[a]["f1_macro_mean"]):
            info = status[arm]
            writer.writerow([
                arm,
                rows[arm]["f1_macro_mean"],
                rows[arm]["params"],
                rows[arm]["latency_median_ms"],
                rows[arm]["int8_size_mb"],
                info["on_frontier"],
                "; ".join(sorted(info["dominated_by"])),
                "; ".join(
                    "%s:%s" % (k, "+".join(v))
                    for k, v in sorted(info["dominated_by"].items())
                ),
            ])

    measured = sorted(a for a, i in status.items() if i["on_frontier"])
    modelled_all = read_gflops_frontier(gflops_pareto)

    # The two frontiers are only comparable over the SAME arms. Benchmarking a
    # subset and comparing it against a frontier computed over all five reports a
    # difference caused by the arm set rather than by the cost axis.
    benchmarked = set(rows)
    partial = modelled_all is not None and not benchmarked.issuperset(set(modelled_all))
    modelled_here = (
        sorted(a for a in modelled_all if a in benchmarked)
        if modelled_all is not None
        else None
    )
    return {
        "written": True,
        "path": str(target),
        "arms_benchmarked": sorted(benchmarked),
        "frontier_measured": measured,
        "frontier_gflops_all_arms": modelled_all,
        "frontier_gflops_restricted": modelled_here,
        "comparison_is_partial": bool(partial),
        "frontier_agrees": (
            modelled_here is not None and not partial and modelled_here == measured
        ),
        "rows": rows,
        "status": {a: i["on_frontier"] for a, i in status.items()},
    }


def read_gflops_frontier(path: Path) -> list[str] | None:
    """The frontier script 06 computed from GFLOPs, for comparison."""
    import csv

    if not path.exists():
        return None
    frontier = []
    with open(path, encoding="utf-8", newline="") as fh:
        rows = [r for r in fh if not r.startswith("#")]
    for row in csv.DictReader(rows):
        if str(row.get("on_pareto_frontier", "")).strip().lower() == "true":
            frontier.append(row["arm"])
    return sorted(frontier)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir", required=True,
        help="directory holding one checkpoint per arm: <arm>.pt, <arm>.pth or <arm>/best.pt",
    )
    parser.add_argument(
        "--images", required=True,
        help="test images. One subdirectory per class enables the INT8 accuracy "
             "measurement; a flat directory gives latency only.",
    )
    parser.add_argument("--arms", nargs="*", default=None, help="subset (default: all five)")
    parser.add_argument("--iterations", type=int, default=MIN_TIMED,
                        help="timed iterations per measurement (minimum %d)" % MIN_TIMED)
    parser.add_argument("--soak-minutes", type=float, default=10.0,
                        help="sustained loop per arm; 0 disables it")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch thread count; 1 is the deployment default")
    parser.add_argument(
        "--cooling", default="unknown",
        choices=["passive", "active", "heatsink-only", "unknown"],
        help="how the board is cooled. Recorded verbatim: a thermal result "
             "without it is not interpretable.",
    )
    parser.add_argument("--out", default="artifacts/edge_benchmark.json")
    parser.add_argument("--device-pareto", default="artifacts/pareto_status_device.csv")
    args = parser.parse_args()

    import torch

    if torch.cuda.is_available():
        raise SystemExit(
            "CUDA is visible. These numbers describe the deployment target, so "
            "this script refuses to run on a GPU."
        )
    torch.set_num_threads(max(1, args.threads))
    iterations = max(MIN_TIMED, args.iterations)

    data_cfg = load_data_config()
    arms_cfg = load_arms_config()
    classes = list(data_cfg["classes"])
    all_arms = sorted(published_arms(arms_cfg))
    arms = args.arms or all_arms
    unknown = [a for a in arms if a not in all_arms]
    if unknown:
        raise SystemExit("Unknown arm(s) %s. Known: %s" % (unknown, all_arms))

    image_dir = Path(args.images)
    if not image_dir.is_dir():
        raise SystemExit("--images is not a directory: %s" % image_dir)
    pairs = labelled_images(image_dir, classes)
    image_paths = (
        [p for p, _ in pairs]
        if pairs
        else sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    )
    if not image_paths:
        raise SystemExit("No images under %s" % image_dir)

    rule("07 -- edge benchmark, all arms")
    checkpoints = resolve_checkpoints(Path(args.weights_dir), arms)

    device = host_provenance(args.threads)
    device.update({
        "cooling": args.cooling,
        "warmup_iterations": WARMUP,
        "timed_iterations": iterations,
        "soak_minutes": args.soak_minutes,
        "temperature_start_c": read_temperature_c(),
        "images_dir": str(image_dir),
        "n_images": len(image_paths),
        "labelled": bool(pairs),
        "arms": arms,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print("[device]")
    for key in sorted(device):
        print("    %-24s %s" % (key, device[key]))
    if args.cooling == "unknown":
        print(
            "\n  [WARNING] --cooling not given. A thermal result without the cooling\n"
            "            arrangement stated is not interpretable, and the soak\n"
            "            section below will say so."
        )
    if not pairs:
        print(
            "\n  [note] --images is flat, so there are no labels: latency, memory and\n"
            "         INT8 size are measured, the INT8 ACCURACY change is not."
        )

    results: dict[str, dict] = {}
    for arm in arms:
        checkpoint = checkpoints[arm]
        rule("%s  (%s)" % (arm, checkpoint.name))
        module, how = load_model(checkpoint, classes)
        module.eval()

        params = int(sum(p.numel() for p in module.parameters()))

        # The timed run is the part most at risk on a passively cooled board, so
        # it gets its own temperature bracket rather than borrowing the soak's.
        timed_start_c = read_temperature_c()
        if timed_start_c is not None and timed_start_c >= THROTTLE_WARN_C:
            print(
                "  [WARNING] %s starts its timed run at %.1f C, at or above the "
                "%.0f C\n            throttling threshold. Every latency below is "
                "taken under\n            throttling and understates what this board "
                "can do cool." % (arm, timed_start_c, THROTTLE_WARN_C)
            )

        full_samples, forward_samples = bench_interleaved(
            module, image_paths, iterations
        )
        timed_end_c = read_temperature_c()
        full = stats(full_samples)
        forward = stats(forward_samples)
        cost = letterbox_cost(full, forward)
        letterbox_ms = cost["letterbox_ms"]
        letterbox_share = cost["letterbox_share_pct"]

        out_dir = Path(args.out).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        int8 = int8_report(module, out_dir / ("int8_%s.pt" % arm), pairs, len(classes))

        print("  full pipeline   median %8.3f ms  IQR %7.3f  p95 %8.3f"
              % (full["median_ms"], full["iqr_ms"], full["p95_ms"]))
        print("  forward only    median %8.3f ms  IQR %7.3f  p95 %8.3f"
              % (forward["median_ms"], forward["iqr_ms"], forward["p95_ms"]))
        # Both the absolute cost and the share, because the share is a property
        # of THIS host and moves with it: on a slower CPU the forward pass grows
        # far more than a file read and a resize, so the same model shows a much
        # smaller letterbox share on a Pi than on a workstation. The milliseconds
        # are the portable number; the percentage is only meaningful next to the
        # device block.
        if cost["ok"]:
            print("  letterbox cost  %8.3f ms  = %.1f %% of the full-pipeline median"
                  % (letterbox_ms, letterbox_share))
        else:
            print("  letterbox cost  FAILED MEASUREMENT -- not reported")
            for line in textwrap.wrap(cost["reason"], width=70):
                print("      %s" % line)
        if timed_start_c is not None:
            print("  timed-run temp  %8.1f -> %.1f C  (%+.1f)"
                  % (timed_start_c, timed_end_c if timed_end_c is not None else float("nan"),
                     (timed_end_c - timed_start_c) if timed_end_c is not None else float("nan")))
        else:
            print("  timed-run temp  not readable on this host")
        print("  peak RSS        %8s MB" % peak_rss_mb())
        if int8.get("available"):
            print("  int8 size       %8.3f MB  (fp32 %.3f, ratio %s)"
                  % (int8["int8_size_mb"], int8.get("fp32_size_mb", float("nan")),
                     int8.get("size_ratio")))
            if int8.get("size_ratio_note"):
                for line in textwrap.wrap(int8["size_ratio_note"], width=70):
                    print("      %s" % line)
            if int8.get("accuracy_measured"):
                print("  int8 macro-F1   %8.4f -> %.4f  (delta %+.4f over %d images)"
                      % (int8["macro_f1_fp32"], int8["macro_f1_int8"],
                         int8["macro_f1_delta"], int8["n_images"]))

        soak_result = {"ran": False, "reason": "--soak-minutes 0"}
        if args.soak_minutes > 0:
            print("  soaking for %.1f minute(s) ..." % args.soak_minutes)
            soak_result = soak(module, image_paths, args.soak_minutes)
            if soak_result.get("ran"):
                print("  soak            first minute %.3f ms -> last minute %.3f ms "
                      "(%+.2f %%)%s"
                      % (soak_result["first_minute_median_ms"],
                         soak_result["last_minute_median_ms"],
                         soak_result["drift_pct_minutes"],
                         "  THROTTLING SUSPECTED" if soak_result["throttling_suspected"] else ""))
                trace = soak_result.get("temperature_c") or []
                if trace:
                    print("  temperature     %.1f -> %.1f C (peak %.1f)"
                          % (trace[0], trace[-1], max(trace)))
                else:
                    print("  temperature     not readable on this host")

        results[arm] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha1": file_sha1(checkpoint),
            "loaded_as": how,
            "params": params,
            "full_pipeline": full,
            "forward_only": forward,
            "letterbox_ms": letterbox_ms,
            "letterbox_share_pct": letterbox_share,
            "letterbox_measurement_ok": cost["ok"],
            "letterbox_failure_reason": cost["reason"],
            "scopes_interleaved": True,
            "letterbox_method_note": (
                "full-pipeline and forward-only are timed on the SAME iterations, "
                "alternating, so thermal drift affects both equally and the "
                "difference stays valid however hot the board gets. Measuring all "
                "of one scope and then all of the other made the difference a "
                "measurement of the drift between them."
            ),
            "timed_run_temperature_c": {
                "start": timed_start_c,
                "end": timed_end_c,
                "delta": (
                    round(timed_end_c - timed_start_c, 2)
                    if None not in (timed_start_c, timed_end_c)
                    else None
                ),
                "throttled_at_start": (
                    None if timed_start_c is None else timed_start_c >= THROTTLE_WARN_C
                ),
                "threshold_c": THROTTLE_WARN_C,
            },
            "letterbox_share_note": (
                "share of the full-pipeline median on THIS host; it falls on a "
                "slower CPU, where the forward pass grows more than the file read "
                "and resize. Quote letterbox_ms across devices, not the share."
            ),
            "peak_rss_mb": peak_rss_mb(),
            "int8": int8,
            "soak": soak_result,
        }

    # ---- the frontier, recomputed on measured cost ----
    rule("Pareto frontier on MEASURED latency")
    pareto = write_device_pareto(
        results,
        artifacts_dir(data_cfg) / "registry.jsonl",
        Path(args.device_pareto),
        artifacts_dir(data_cfg) / "pareto_status.csv",
    )
    if pareto.get("written"):
        print("  %-20s %12s %10s %14s %12s  %s"
              % ("arm", "f1_mean", "params", "median_ms", "int8_MB", "frontier"))
        for arm in sorted(pareto["rows"], key=lambda a: -pareto["rows"][a]["f1_macro_mean"]):
            row = pareto["rows"][arm]
            print("  %-20s %12.4f %10d %14.3f %12.3f  %s"
                  % (arm, row["f1_macro_mean"], row["params"],
                     row["latency_median_ms"], row["int8_size_mb"],
                     "YES" if pareto["status"][arm] else "no"))
        print("\n  measured-latency frontier : %s" % ", ".join(pareto["frontier_measured"]))
        if pareto["frontier_gflops_all_arms"] is None:
            print("  GFLOPs frontier           : artifacts/pareto_status.csv absent")
        elif pareto["comparison_is_partial"]:
            print("  GFLOPs frontier (all arms): %s"
                  % ", ".join(pareto["frontier_gflops_all_arms"]))
            print(
                "\n  NOT COMPARABLE. Only %d arm(s) were benchmarked (%s) while the\n"
                "  GFLOPs frontier covers all of them, so any difference between the\n"
                "  two would be caused by the arm set rather than by the cost axis.\n"
                "  Re-run without --arms to compare them."
                % (len(pareto["arms_benchmarked"]), ", ".join(pareto["arms_benchmarked"]))
            )
        elif pareto["frontier_agrees"]:
            print("  GFLOPs frontier           : %s"
                  % ", ".join(pareto["frontier_gflops_restricted"]))
            print(
                "\n  The frontier is UNCHANGED under a second, independent definition\n"
                "  of cost -- modelled FLOPs and measured wall-clock latency on the\n"
                "  deployment hardware select the same models. That agreement is the\n"
                "  stronger result: it is not an artefact of how cost was defined."
            )
        else:
            print("  GFLOPs frontier           : %s"
                  % ", ".join(pareto["frontier_gflops_restricted"]))
            print(
                "\n  The frontier DIFFERS between modelled and measured cost. FLOPs and\n"
                "  latency do not have to agree -- depthwise-separable operators are\n"
                "  cheap in FLOPs and often poorly served by CPU kernels -- and the\n"
                "  measured frontier is the one the deployment claim rests on."
            )
        print("\n[artifacts] wrote %s" % pareto["path"])
    else:
        print("  skipped: %s" % pareto.get("reason"))

    payload = {
        "device": device,
        "arms": results,
        "pareto_device": {k: v for k, v in pareto.items() if k != "rows"},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, default=str)
    rule("DONE")
    print("[artifacts] wrote %s" % out_path)
    if args.cooling == "unknown" and args.soak_minutes > 0:
        print("  Re-run with --cooling to make the thermal result reportable.")

    hot = [
        arm for arm, row in results.items()
        if (row["timed_run_temperature_c"] or {}).get("throttled_at_start")
    ]
    if hot:
        print(
            "\n  [WARNING] %s began its timed run at or above %.0f C. Those latencies\n"
            "            are throttled figures and understate the board when cool."
            % (", ".join(sorted(hot)), THROTTLE_WARN_C)
        )

    # An assertion that never fails is not an assertion. The JSON is written
    # first either way, so a failed scope comparison costs nothing already
    # measured -- but the run does not report success.
    failed = [arm for arm, row in results.items()
              if not row.get("letterbox_measurement_ok", True)]
    if failed:
        print(
            "\n  FAILED MEASUREMENT for %s: forward-only exceeded the full\n"
            "  pipeline, which cannot happen. letterbox_ms is null in the JSON for\n"
            "  these arms rather than negative. Everything else above stands."
            % ", ".join(sorted(failed))
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
