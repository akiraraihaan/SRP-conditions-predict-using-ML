"""Efficiency statistics: parameters, GFLOPs, serialised size, latency.

Carries over the definitions used in the legacy pipeline (code.ipynb cell 8,
"[Cell 9]") so the numbers stay comparable:

  params  sum of numel() over all parameters
  gflops  2 * MACs from thop on a (1, 3, 224, 224) dummy
  size_mb size of the serialised state_dict on disk

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


def count_gflops(module, image_size: int = 224) -> float | None:
    """2 * MACs / 1e9, via thop. Returns None if thop is unavailable."""
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
        module.train(was_training)


def serialised_size_mb(module) -> float:
    """Size of the state_dict written to disk, in MB."""
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weights.pt"
        torch.save(module.state_dict(), path)
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
    """Every efficiency statistic in one call."""
    stats: dict[str, Any] = {
        "params": count_parameters(module),
        "gflops": count_gflops(module, image_size),
        "size_mb": serialised_size_mb(module),
    }
    if latency:
        stats.update(measure_latency(module, image_size))
    return stats
