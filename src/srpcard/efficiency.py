"""Efficiency statistics: parameters, GFLOPs, serialised size, latency.

Carries over the definitions used in the legacy pipeline (code.ipynb cell 8,
"[Cell 9]") so the numbers stay comparable:

  params  sum of numel() over all parameters
  gflops  2 * MACs from thop on a (1, 3, 224, 224) dummy

SIZE IS MEASURED FOUR WAYS, ON PURPOSE, AND fp16 IS PRIMARY.

The legacy pipeline reported the size of the ultralytics `best.pt` on disk --
3.063 / 10.538 / 19.932 MB for nano / small / medium, the numbers that are in the
manuscript abstract and two of its tables. Ultralytics strips the optimizer and
casts the model to **half precision** before saving, so those are fp16 figures.
`state_dict()` here is fp32, so serialising it gives very nearly exactly TWICE
the legacy number (measured: 1.96x, 1.99x, 1.99x). Reporting one where the other
is expected would silently double every published model size.

fp16 is PRIMARY because it is what is actually deployed. The artefact copied
to a Raspberry Pi is the checkpoint the framework ships, and ultralytics ships
half precision; an fp32 state_dict is a form that is never deployed, so
reporting it as "model size" overstates the deployment cost by 2x. fp16 is also
computable uniformly for all five arms, unlike the checkpoint file size, which
only exists for the three ultralytics ones.

  size_mb_fp16          PRIMARY. Half-precision state_dict written by torch.save.
  size_mb_fp32          the same weights at full precision, for reference.
  size_mb_fp16_payload  raw fp16 tensor bytes, no container at all.
  size_mb_fp32_payload  raw fp32 tensor bytes.
  size_mb               == size_mb_fp16. Kept because existing code reads it;
                        prefer the explicit names in anything new.

The payload figures exist because the on-disk ones include torch's zip
container -- about 250 bytes per tensor, uniformly across both backends, so
0.03-0.07 MB depending on the model. That overhead is a property of the saving
convention rather than of the model, and the payload is what does not move when
the convention changes.

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


# torch.save writes a zip whose internal record names are prefixed with the
# ARCHIVE STEM, stored once in each local header and once in the central
# directory, with 64-byte alignment padding between records. The serialised size
# therefore depends on the filename it was written to: the same yolo26n weights
# measure 3.021 MB as "w.pt" and 3.039 MB as
# "a_very_long_temporary_filename.pt". A published model size must not depend on
# that, so the name is fixed here and never taken from the caller.
_ARCHIVE_NAME = "model.pt"


def _clean_state_dict(module, *, half: bool = False) -> dict:
    """state_dict without thop instrumentation, optionally cast to fp16."""
    state = {
        key: value
        for key, value in module.state_dict().items()
        if key.rsplit(".", 1)[-1] not in THOP_BUFFERS
    }
    if half:
        state = {
            key: (value.half() if value.is_floating_point() else value)
            for key, value in state.items()
        }
    return state


def payload_size_mb(module, *, half: bool = False) -> float:
    """Raw tensor bytes, in MB. The container-free weight payload.

    Sums numel * element_size over the state_dict, so it is independent of the
    serialisation format entirely -- no zip headers, no alignment padding, no
    filename. This is the quantity that does not move when anything about the
    saving convention changes, and the one to compare across frameworks:
    `size_mb_fp16` includes ultralytics' or torch's container, this does not.
    """
    state = _clean_state_dict(module, half=half)
    total = sum(value.numel() * value.element_size() for value in state.values())
    return round(total / (1024**2), 3)


def serialised_size_mb(module, *, half: bool = False) -> float:
    """Size of the state_dict written to disk by torch.save, in MB.

    `half=True` casts floating-point tensors to fp16 first, which is what
    ultralytics does before saving a checkpoint. See the module docstring: the
    two differ by a factor of ~2 and the manuscript quotes the fp16 figures.

    Includes torch's zip container overhead, which is a few KB and scales with
    the number of tensors. `payload_size_mb` is the same weights without it.
    """
    import torch

    state = _clean_state_dict(module, half=half)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / _ARCHIVE_NAME
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
    payload_fp16 = payload_size_mb(module, half=True)
    payload_fp32 = payload_size_mb(module)
    stats: dict[str, Any] = {
        "params": count_parameters(module),
        "gflops": count_gflops(module, image_size),
        # fp16 is the primary: it is what the framework actually ships to the
        # device. See the module docstring and HANDOVER.md section 7.
        "size_mb": fp16,
        "size_mb_fp32": fp32,
        "size_mb_fp16": fp16,
        "size_mb_fp16_payload": payload_fp16,
        "size_mb_fp32_payload": payload_fp32,
    }
    if latency:
        stats.update(measure_latency(module, image_size))
    return stats
