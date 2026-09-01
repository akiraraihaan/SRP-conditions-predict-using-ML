#!/usr/bin/env python
"""07 -- edge latency benchmark. STANDALONE. Runs on a Raspberry Pi, not on Kaggle.

    python scripts/07_bench_edge.py --weights exported/yolo26n_fold0.pt --images data/sample
    python scripts/07_bench_edge.py --weights model.pt --images data/sample --soak-minutes 10

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

Records CPU model, OS, runtime versions, thread count and power mode, plus peak
resident memory and the size of an INT8-quantised copy of the model.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# CPU only, decided before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

WARMUP = 50
MIN_TIMED = 200
IMAGE_SIZE = 224
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


def bench(module, image_paths: list[Path], iterations: int, with_letterbox: bool) -> list[float]:
    """Time the pipeline. `with_letterbox=False` times the forward pass only."""
    import torch
    from PIL import Image

    samples: list[float] = []
    if not with_letterbox:
        with Image.open(image_paths[0]) as img:
            fixed = to_tensor(letterbox(img.convert("RGB")))

    with torch.no_grad():
        for i in range(WARMUP + iterations):
            path = image_paths[i % len(image_paths)]
            start = time.perf_counter()
            if with_letterbox:
                with Image.open(path) as img:          # file read
                    tensor = to_tensor(letterbox(img.convert("RGB")))   # letterbox
            else:
                tensor = fixed
            logits = module(tensor)                     # forward
            while isinstance(logits, (tuple, list)):
                logits = logits[0]
            int(logits.argmax(dim=1).item())            # label
            elapsed = (time.perf_counter() - start) * 1000.0
            if i >= WARMUP:                             # discard the warm-up
                samples.append(elapsed)
    return samples


def soak(module, image_paths: list[Path], minutes: float) -> dict:
    """Run continuously and report whether the median drifts -- thermal throttling."""
    import torch
    from PIL import Image

    deadline = time.perf_counter() + minutes * 60.0
    samples: list[float] = []
    stamps: list[float] = []
    started = time.perf_counter()
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
            samples.append((time.perf_counter() - start) * 1000.0)
            stamps.append(time.perf_counter() - started)

    if len(samples) < 30:
        return {"ran": False, "reason": "too few iterations (%d)" % len(samples)}

    third = len(samples) // 3
    first_median = float(np.median(samples[:third]))
    last_median = float(np.median(samples[-third:]))
    drift_pct = 100.0 * (last_median - first_median) / first_median
    return {
        "ran": True,
        "minutes": minutes,
        "iterations": len(samples),
        "first_third_median_ms": round(first_median, 3),
        "last_third_median_ms": round(last_median, 3),
        "drift_ms": round(last_median - first_median, 3),
        "drift_pct": round(drift_pct, 2),
        # 5 % is a conservative flag; sustained throttling on a Pi is usually far larger
        "throttling_suspected": bool(drift_pct > 5.0),
        "overall": stats(samples),
        "power_mode_after": power_mode(),
    }


def int8_size_mb(module, out_path: Path) -> dict:
    """Dynamically quantise to INT8 and report the serialised size."""
    import torch

    try:
        quantised = torch.ao.quantization.quantize_dynamic(
            module, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        torch.save(quantised.state_dict(), out_path)
        return {
            "available": True,
            "int8_size_mb": round(out_path.stat().st_size / (1024**2), 3),
            "path": str(out_path),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="exported model file")
    parser.add_argument("--images", required=True, help="directory of sample images")
    parser.add_argument("--iterations", type=int, default=MIN_TIMED)
    parser.add_argument("--soak-minutes", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--out", default="artifacts/edge_benchmark.json")
    parser.add_argument("--classes", default=None, help="configs/data.yaml (default: repo copy)")
    args = parser.parse_args()

    if args.iterations < MIN_TIMED:
        raise SystemExit(
            "--iterations must be at least %d; got %d" % (MIN_TIMED, args.iterations)
        )

    import torch

    if torch.cuda.is_available():
        raise SystemExit(
            "CUDA is visible. This benchmark describes the edge target and must run on CPU.\n"
            "  Unset CUDA_VISIBLE_DEVICES interference or run on the device itself."
        )
    threads = args.threads or torch.get_num_threads()
    torch.set_num_threads(threads)

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit("Required input not found: %s" % weights)
    image_dir = Path(args.images)
    if not image_dir.exists():
        raise SystemExit("Required input not found: %s" % image_dir)
    image_paths = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    if not image_paths:
        raise SystemExit("No images found under %s" % image_dir)

    classes_path = Path(args.classes) if args.classes else (
        Path(__file__).resolve().parents[1] / "configs" / "data.yaml"
    )
    if not classes_path.exists():
        raise SystemExit("Required input not found: %s" % classes_path)
    import yaml

    classes = list(yaml.safe_load(classes_path.read_text(encoding="utf-8"))["classes"])

    rule("07 -- edge benchmark (CPU only)")
    provenance = host_provenance(threads)
    for key, value in provenance.items():
        print("  %-16s %s" % (key, value))
    print("  %-16s %d image(s) from %s" % ("images", len(image_paths), image_dir))

    module, kind = load_model(weights, classes)
    print("  %-16s %s (%s)" % ("model", weights.name, kind))

    rule("latency: full pipeline (file read -> letterbox -> forward -> label)")
    full = stats(bench(module, image_paths, args.iterations, with_letterbox=True))
    for key in ("n", "median_ms", "iqr_ms", "p95_ms", "mean_ms", "min_ms", "max_ms"):
        print("  %-12s %s" % (key, full[key]))

    rule("latency: forward pass only (no file read, no letterbox)")
    forward = stats(bench(module, image_paths, args.iterations, with_letterbox=False))
    for key in ("n", "median_ms", "iqr_ms", "p95_ms"):
        print("  %-12s %s" % (key, forward[key]))
    print(
        "\n  preprocessing overhead (median): %.3f ms  (%.1f%% of the full pipeline)"
        % (
            full["median_ms"] - forward["median_ms"],
            100.0 * (full["median_ms"] - forward["median_ms"]) / full["median_ms"],
        )
    )

    rule("thermal soak (%g minutes)" % args.soak_minutes)
    soak_result = soak(module, image_paths, args.soak_minutes) if args.soak_minutes > 0 else {"ran": False}
    if soak_result.get("ran"):
        print("  iterations            %d" % soak_result["iterations"])
        print("  first-third median    %.3f ms" % soak_result["first_third_median_ms"])
        print("  last-third median     %.3f ms" % soak_result["last_third_median_ms"])
        print("  drift                 %+.3f ms  (%+.2f%%)"
              % (soak_result["drift_ms"], soak_result["drift_pct"]))
        print("  throttling suspected  %s" % soak_result["throttling_suspected"])
    else:
        print("  skipped (%s)" % soak_result.get("reason", "--soak-minutes 0"))

    rule("footprint")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quant = int8_size_mb(module, out_path.with_name("model_int8.pt"))
    fp32_mb = round(weights.stat().st_size / (1024**2), 3)
    print("  fp32 model on disk    %.3f MB" % fp32_mb)
    if quant["available"]:
        print("  int8 quantised        %.3f MB  (%.1f%% of fp32)"
              % (quant["int8_size_mb"], 100.0 * quant["int8_size_mb"] / fp32_mb))
    else:
        print("  int8 quantised        unavailable: %s" % quant["reason"])
    peak = peak_rss_mb()
    print("  peak resident memory  %s MB" % peak)

    payload = {
        "host": provenance,
        "model": {"weights": str(weights), "kind": kind, "fp32_size_mb": fp32_mb},
        "protocol": {
            "batch_size": 1,
            "warmup_discarded": WARMUP,
            "timed_iterations": args.iterations,
            "image_size": IMAGE_SIZE,
            "device": "cpu",
        },
        "latency_full_pipeline": full,
        "latency_forward_only": forward,
        "preprocessing_overhead_ms": round(full["median_ms"] - forward["median_ms"], 3),
        "soak": soak_result,
        "quantisation": quant,
        "peak_rss_mb": peak,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    print("\n[artifacts] wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
