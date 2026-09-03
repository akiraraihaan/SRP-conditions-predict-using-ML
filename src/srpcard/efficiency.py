"""Efficiency statistics: parameters, GFLOPs, serialised size, latency.

Carries over the definitions used in the legacy pipeline (code.ipynb cell 8,
"[Cell 9]") so the numbers stay comparable:

  params  sum of numel() over all parameters
  gflops  2 * MACs from thop on a (1, 3, 224, 224) dummy

SIZE IS MEASURED TWICE, ON PURPOSE.

The legacy pipeline reported the size of the ultralytics `best.pt` on disk --
3.063 / 10.538 / 19.932 MB for nano / small / medium, the numbers that are in the
manuscript abstract and two of its tables. Ultralytics strips the optimizer and
casts the model to **half precision** before saving, so those are fp16 figures.
`state_dict()` here is fp32, so serialising it gives very nearly exactly TWICE
the legacy number (measured: 1.96x, 1.99x, 1.99x). Reporting one where the other
is expected would silently double every published model size.

So both are recorded, under names that say which is which:

  size_mb_fp32  fp32 state_dict on disk. Internally consistent across all five
                arms and every script; this is the Pareto objective.
  size_mb_fp16  the same weights cast to half. Reproduces the legacy checkpoint
                measurement to within ~0.04 MB (the residual is the ultralytics
                checkpoint's metadata: class names, train args, version, date).
                This is the number the manuscript reports.
  size_mb       == size_mb_fp32. Kept because existing code and figures read it;
                prefer the explicit names in anything new.

Script 01 additionally records `size_mb_checkpoint_file`, the actual `best.pt`
size, because it is the one script that produces such a file.

The latency helper here is the LIGHT one, recorded alongside every training run
for context. The publication latency numbers come from scripts/07_bench_edge.py,
which is far stricter (50 warm-up, >=200 timed, median/IQR/p95, thermal soak).
Do not quote the numbers from this module in the manuscript.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def count_parameters(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# thop instruments a module by REGISTERING BUFFERS on every submodule, and does
# not always remove them afterwards. They then appear in state_dict() and inflate
# every later size measurement -- on yolo26m, 248 extra buffers worth ~0.074 MB.
# The size of a model must not depend on whether its FLOPs were counted first.
THOP_BUFFERS = ("total_ops", "total_params")


def _strip_thop_buffers(module) -> int:
    """Remove thop's instrumentation buffers. Returns how many were removed."""
    removed = 0
    for submodule in module.modules():
        for name in THOP_BUFFERS:
            if name in getattr(submodule, "_buffers", {}):
                del submodule._buffers[name]
                removed += 1
    return removed


def count_gflops(module, image_size: int = 224) -> float | None:
    """2 * MACs / 1e9, via thop. Returns None if thop is unavailable.

    Leaves the module exactly as it found it: thop's buffers are stripped again
    on the way out, so a later serialised_size_mb() is not inflated by them.
    """
    import torch

    try:
        from thop import profile as thop_profile
    except ImportError:
        try:
            from ultralytics.thop import profile as thop_profile  # type: ignore
        except ImportError:
            return None

    was_training = module.training
    module.eval()
    device = next(module.parameters()).device
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    try:
        macs, _ = thop_profile(module, inputs=(dummy,), verbose=False)
        return float(2.0 * macs / 1e9)
    except Exception:  # noqa: BLE001 - profiling never blocks a run
        return None
    finally:
        _strip_thop_buffers(module)
        module.train(was_training)


def serialised_size_mb(module, *, half: bool = False) -> float:
    """Size of the state_dict written to disk, in MB.

    `half=True` casts floating-point tensors to fp16 first, which is what
    ultralytics does before saving a checkpoint. See the module docstring: the
    two differ by a factor of ~2 and the manuscript quotes the fp16 figures.
    """
    import torch

    # Belt and braces alongside _strip_thop_buffers: never let instrumentation
    # left behind by a FLOP count enter a published model size.
    state = {
        key: value
        for key, value in module.state_dict().items()
        if not key.rsplit(".", 1)[-1] in THOP_BUFFERS
    }
    if half:
        state = {
            key: (value.half() if value.is_floating_point() else value)
            for key, value in state.items()
        }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weights.pt"
        torch.save(state, path)
        return round(path.stat().st_size / (1024**2), 3)


def measure_latency(
    module, image_size: int = 224, warmup: int = 5, repeats: int = 50, device: str = "cpu"
) -> dict[str, float]:
    """Light single-image latency, CPU by default. Context only, not for publication."""
    import torch

    was_training = module.training
    original_device = next(module.parameters()).device
    module.eval().to(device)
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    times: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            module(dummy)
        for _ in range(repeats):
            start = time.perf_counter()
            module(dummy)
            times.append((time.perf_counter() - start) * 1000.0)

    module.to(original_device)
    module.train(was_training)
    return {
        "latency_ms_mean": round(float(np.mean(times)), 3),
        "latency_ms_std": round(float(np.std(times)), 3),
        "latency_device": device,
        "latency_repeats": repeats,
    }


def profile(module, image_size: int = 224, *, latency: bool = True) -> dict[str, Any]:
    """Every efficiency statistic in one call.

    Both size measurements are always returned; see the module docstring for why
    picking one would silently change the published model sizes.
    """
    # Sizes first, FLOPs second. count_gflops now cleans up after itself, but the
    # ordering makes the measurement independent of that as well.
    fp32 = serialised_size_mb(module)
    fp16 = serialised_size_mb(module, half=True)
    stats: dict[str, Any] = {
        "params": count_parameters(module),
        "gflops": count_gflops(module, image_size),
        "size_mb": fp32,
        "size_mb_fp32": fp32,
        "size_mb_fp16": fp16,
    }
    if latency:
        stats.update(measure_latency(module, image_size))
    return stats
