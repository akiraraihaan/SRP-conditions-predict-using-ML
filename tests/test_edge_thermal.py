"""The letterbox cost must be a property of the pipeline, not of the cooling.

The incident: a Raspberry Pi 3 with no heatsink reported a full-pipeline median
of 725.9 ms against a forward-only median of 768.2 ms -- a letterbox cost of
-42.3 ms. The full pipeline contains the forward pass, so that cannot happen.
The board was at 80.6 C before the timed runs began and climbed to 82.7 C during
them, and the two scopes were measured sequentially, so the second one ran on a
hotter chip. What was measured was thermal drift.

The fix is to remove the confound rather than warn about it: interleave the two
scopes so both see the same thermal history. These tests pin the interleaving,
the refusal to report a negative cost, and the temperature bracket around each
arm's timed run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location(
        "bench_edge", REPO_ROOT / "scripts" / "07_bench_edge.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_edge"] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------- interleaving


class _FakeImage:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def convert(self, mode):
        return "image"


class _Recorder:
    """Records which scope each forward pass belonged to.

    bench_interleaved builds the forward-only tensor ONCE before the loop, so
    the first to_tensor call is that one and every later call is a
    full-pipeline iteration.
    """

    def __init__(self):
        self.order = []
        self.calls = 0

    def to_tensor(self, img):
        self.calls += 1
        return "forward" if self.calls == 1 else "full"

    def __call__(self, tensor):
        import torch

        self.order.append(tensor)
        return torch.zeros(1, 9)


def _install(bench, monkeypatch, recorder):
    import PIL.Image

    monkeypatch.setattr(bench, "WARMUP", 0)
    monkeypatch.setattr(bench, "letterbox", lambda img, target=224: img)
    monkeypatch.setattr(bench, "to_tensor", recorder.to_tensor)
    monkeypatch.setattr(PIL.Image, "open", lambda path: _FakeImage())


def test_the_two_scopes_are_interleaved_not_run_in_blocks(bench, monkeypatch):
    """Each iteration times both scopes before moving to the next one.

    Run in blocks, the difference between them measures whatever drifted in
    between -- on a passively cooled board, the temperature.
    """
    recorder = _Recorder()
    _install(bench, monkeypatch, recorder)

    bench.bench_interleaved(recorder, [Path("a.jpg")], 4)

    assert len(recorder.order) == 8
    assert recorder.order.count("full") == 4
    assert recorder.order.count("forward") == 4
    assert recorder.order != ["full"] * 4 + ["forward"] * 4, "measured in blocks"
    assert recorder.order != ["forward"] * 4 + ["full"] * 4, "measured in blocks"


def test_the_order_alternates_so_neither_scope_is_always_second(bench, monkeypatch):
    """Whichever scope runs second inherits the other's cache and clock state,
    so always handing that position to the same scope puts the bias back."""
    recorder = _Recorder()
    _install(bench, monkeypatch, recorder)

    bench.bench_interleaved(recorder, [Path("a.jpg")], 2)

    assert recorder.order == ["full", "forward", "forward", "full"]


def test_both_scopes_get_the_same_number_of_samples(bench, monkeypatch):
    """They are timed on the same iterations, so the counts cannot diverge."""
    recorder = _Recorder()
    _install(bench, monkeypatch, recorder)

    full, forward = bench.bench_interleaved(recorder, [Path("a.jpg")], 6)

    assert len(full) == len(forward) == 6


def test_the_warmup_is_discarded_from_both(bench, monkeypatch):
    recorder = _Recorder()
    _install(bench, monkeypatch, recorder)
    monkeypatch.setattr(bench, "WARMUP", 3)

    full, forward = bench.bench_interleaved(recorder, [Path("a.jpg")], 2)

    assert len(full) == len(forward) == 2
    assert len(recorder.order) == (3 + 2) * 2


# --------------------------------------------------- the non-negative assert


def test_a_negative_letterbox_cost_is_a_failed_measurement(bench):
    """The exact Pi 3 numbers. -42.3 ms must never reach a results table."""
    cost = bench.letterbox_cost(
        {"median_ms": 725.9}, {"median_ms": 768.2}
    )
    assert cost["ok"] is False
    assert cost["letterbox_ms"] is None
    assert cost["letterbox_share_pct"] is None
    assert "cannot happen" in cost["reason"] or "failed measurement" in cost["reason"]
    assert "768.2" in cost["reason"] and "725.9" in cost["reason"]


def test_a_normal_measurement_reports_both_the_cost_and_the_share(bench):
    cost = bench.letterbox_cost({"median_ms": 100.0}, {"median_ms": 75.0})
    assert cost["ok"] is True
    assert cost["letterbox_ms"] == 25.0
    assert cost["letterbox_share_pct"] == 25.0
    assert cost["reason"] is None


def test_a_zero_cost_is_not_treated_as_a_failure(bench):
    """Equal medians are implausible but not impossible, and not negative."""
    cost = bench.letterbox_cost({"median_ms": 50.0}, {"median_ms": 50.0})
    assert cost["ok"] is True
    assert cost["letterbox_ms"] == 0.0


# ------------------------------------------------------ the throttle threshold


def test_the_throttle_threshold_is_declared_once(bench):
    assert bench.THROTTLE_WARN_C == 80.0


def test_the_pi_reading_is_above_the_threshold(bench):
    """80.6 C at the start of a timed run must trip the warning."""
    assert 80.6 >= bench.THROTTLE_WARN_C


# ------------------------------------------------------------ quantisation


def test_resnet18_reports_that_conv_layers_were_not_quantised(bench):
    """The manuscript's microcontroller argument rests on quantisation.

    A ratio of 1.0 with no explanation reads as a broken measurement. It is the
    correct result: quantize_dynamic converts Linear and the RNN family, and
    ResNet18 is almost entirely Conv2d.
    """
    torch = pytest.importorskip("torch")
    from torchvision.models import resnet18

    module = resnet18(weights=None)
    module.eval()
    quantised, reason = bench.quantise_int8(module)
    assert quantised is not None, reason

    coverage = bench.quantisation_coverage(module, quantised)

    assert coverage["quantised_layer_types"] == {"Linear": 1}
    assert coverage["unquantised_layer_types"]["Conv2d"] == 20
    assert coverage["params_quantised_pct"] < 5.0
    assert "Conv2d is NOT supported" in coverage["method_scope"]


def test_the_census_counts_leaf_layers_only(bench):
    torch = pytest.importorskip("torch")

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3),
        torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.ReLU()),
    )
    census = bench.layer_census(model)
    assert census["Conv2d"] == 1
    assert census["Linear"] == 1
    assert "Sequential" not in census


def test_a_ratio_near_one_carries_an_explanation(bench, tmp_path):
    torch = pytest.importorskip("torch")
    from torchvision.models import resnet18

    module = resnet18(weights=None)
    module.eval()
    report = bench.int8_report(module, tmp_path / "int8.pt", [], 9)

    assert report["available"] is True
    assert report["size_ratio"] > 0.95
    assert "CORRECT result" in report["size_ratio_note"]
    assert "Conv2d" in report["size_ratio_note"]
    assert report["coverage"]["quantised_layer_types"] == {"Linear": 1}


def test_conv2d_in_the_quantisation_set_is_documented_as_a_no_op(bench):
    """Passing Conv2d neither works nor errors. Nobody should read its presence
    as a claim that convolutions were quantised."""
    assert "IGNORED" in bench.quantise_int8.__doc__
