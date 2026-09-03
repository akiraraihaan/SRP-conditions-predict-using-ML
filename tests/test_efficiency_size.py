"""Model size is a published number, so its measurement is pinned by tests.

Three separate hazards, each of which silently changed a figure once:

  1. `size_mb` meant the ultralytics checkpoint file (fp16) in script 01 and an
     fp32 state_dict everywhere else -- a factor of ~2.
  2. thop registers buffers on every submodule when counting FLOPs and leaves
     them behind, inflating any size measured afterwards.
  3. the two are compounding: profile() used to count FLOPs before measuring.

None of these touch the filesystem outside tmp_path.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from srpcard import efficiency  # noqa: E402


@pytest.fixture(scope="module")
def net():
    """A small conv net -- enough for thop to instrument, cheap to build.

    Deliberately over a megabyte of weights: sizes are reported to three decimal
    places in MB, so a toy model rounds fp32 and fp16 to the same number and the
    factor-of-two property becomes untestable.
    """
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(4),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 4 * 4, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 10),
    )


def test_fp16_is_about_half_of_fp32(net):
    fp32 = efficiency.serialised_size_mb(net)
    fp16 = efficiency.serialised_size_mb(net, half=True)
    assert fp16 < fp32
    # small models carry proportionally more pickle overhead, so this is loose;
    # the real arms measure 1.96-1.99x (see HANDOVER.md section 8).
    assert 1.2 < fp32 / fp16 < 2.2


def test_counting_flops_does_not_change_the_size(net):
    before = efficiency.serialised_size_mb(net)
    efficiency.count_gflops(net, 32)
    assert efficiency.serialised_size_mb(net) == before


def test_thop_buffers_are_stripped(net):
    efficiency.count_gflops(net, 32)
    leftover = [
        key for key in net.state_dict()
        if key.rsplit(".", 1)[-1] in efficiency.THOP_BUFFERS
    ]
    assert leftover == []


def test_size_ignores_thop_buffers_even_if_present(net):
    """The measurement must not depend on cleanup having happened."""
    clean = efficiency.serialised_size_mb(net)
    for submodule in net.modules():
        submodule.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
        submodule.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))
    try:
        assert efficiency.serialised_size_mb(net) == clean
    finally:
        efficiency._strip_thop_buffers(net)


def test_profile_is_repeatable(net):
    runs = [efficiency.profile(net, 32, latency=False) for _ in range(3)]
    for key in ("params", "gflops", "size_mb", "size_mb_fp32", "size_mb_fp16",
                "size_mb_fp16_payload", "size_mb_fp32_payload"):
        assert len({run[key] for run in runs}) == 1, "%s is not repeatable" % key


def test_size_mb_is_the_fp16_alias(net):
    """fp16 is primary: it is what the framework deploys."""
    stats = efficiency.profile(net, 32, latency=False)
    assert stats["size_mb"] == stats["size_mb_fp16"]
    assert stats["size_mb_fp16"] < stats["size_mb_fp32"]


def test_payload_is_smaller_than_the_container(net):
    stats = efficiency.profile(net, 32, latency=False)
    assert stats["size_mb_fp16_payload"] <= stats["size_mb_fp16"]
    assert stats["size_mb_fp32_payload"] <= stats["size_mb_fp32"]


def test_payload_matches_the_raw_tensor_bytes(net):
    expected = sum(
        v.numel() * (2 if v.is_floating_point() else v.element_size())
        for v in net.state_dict().values()
    ) / 1024 ** 2
    assert efficiency.payload_size_mb(net, half=True) == round(expected, 3)


def test_size_does_not_depend_on_the_temp_filename(net, tmp_path, monkeypatch):
    """torch.save embeds the archive stem in every record name, twice, so the
    measured size moves with the filename. The name must be fixed, not the
    caller's or the tempdir's."""
    first = efficiency.serialised_size_mb(net, half=True)
    monkeypatch.setattr(efficiency, "_ARCHIVE_NAME", "a_very_long_archive_name.pt")
    inflated = efficiency.serialised_size_mb(net, half=True)
    monkeypatch.setattr(efficiency, "_ARCHIVE_NAME", "model.pt")
    assert efficiency.serialised_size_mb(net, half=True) == first
    # the hazard is real, which is why the name is pinned
    assert inflated >= first


def test_payload_is_immune_to_the_filename(net, monkeypatch):
    before = efficiency.payload_size_mb(net, half=True)
    monkeypatch.setattr(efficiency, "_ARCHIVE_NAME", "a_very_long_archive_name.pt")
    assert efficiency.payload_size_mb(net, half=True) == before
