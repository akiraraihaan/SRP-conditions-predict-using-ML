"""Every record in the real registry must still recompute to its own run_id.

Not one example -- all 234. A run_id that moves is not a cosmetic problem: the
resume path works by hash, so a moved id means the run stops being recognised,
the next dry run proposes repeating it, and an append-only registry ends up with
two records for one run under different names.

This became a live risk when `optimizer` joined RUN_ID_FIELDS. The first attempt
treated the DECLARED optimizer as part of the identity with "sgd" as the neutral
default, on the belief that every run so far used SGD. That belief was wrong:
configs/arms.yaml sets `optimizer: MuSGD` on yolo26n, yolo26s and yolo26m, so
165 of the 234 records are MuSGD runs and all of them would have been orphaned.
The identity now hashes an OVERRIDE of the arm's declared optimizer, never the
declared value, and this file is the proof over the real data rather than over a
constructed example.

WHAT THIS TEST ALSO DOCUMENTS: a record does NOT contain enough information to
reproduce its own run_id. `extra` is hashed as a short string marker chosen by
the script, while the record stores a rich dict under the same key. The markers
are listed below because they exist nowhere else in one place.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from srpcard import registry

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "artifacts" / "registry.jsonl"


# The value each script passes as `extra` to compute_run_id. It is NOT the
# `extra` dict the record stores -- see the module docstring.
#
#   scripts/01_complete_medium_grid.py:363,384
#   scripts/01b_uniform_grid.py:315
#   scripts/02_lr_sweep_baselines.py:143
#   scripts/03_run_cv.py, 04_run_ablation.py   -- None
#   scripts/05_learning_curve.py:157
def identity_extra_candidates(record: dict) -> list:
    extra = record.get("extra") or {}
    script = record.get("script")
    if script in ("03_run_cv", "04_run_ablation"):
        return [None]
    if script == "01_complete_medium_grid":
        return ["legacy_protocol", "control_rerun"]
    if script == "01b_uniform_grid":
        return ["uniform_grid"]
    if script == "02_lr_sweep_baselines":
        return ["lr_sweep"]
    if script == "05_learning_curve":
        return ["lc_frac%.2f" % extra["fraction"]] if "fraction" in extra else []
    return [None]


def recompute(record: dict, extra) -> str:
    fields = {
        key: (extra if key == "extra" else record.get(key))
        for key in registry.RUN_ID_FIELDS
    }
    return registry.compute_run_id(**fields)


def reproduces(record: dict) -> bool:
    return any(
        recompute(record, extra) == record.get("run_id")
        for extra in identity_extra_candidates(record)
    )


@pytest.fixture(scope="module")
def records():
    if not REGISTRY.exists():
        pytest.skip("no artifacts/registry.jsonl in this checkout")
    return [
        json.loads(line)
        for line in REGISTRY.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_every_record_recomputes_to_its_own_run_id(records):
    """The whole registry, not one example."""
    moved = [r for r in records if not reproduces(r)]
    assert not moved, (
        "%d of %d record(s) no longer recompute to their stored run_id. Each of "
        "these would stop resuming and be proposed for a re-run:\n  %s"
        % (
            len(moved),
            len(records),
            "\n  ".join(
                "%s  %s %s r%sf%s"
                % (r.get("run_id"), r.get("script"), r.get("arm"),
                   r.get("repeat"), r.get("fold"))
                for r in moved[:15]
            ),
        )
    )


def test_the_musgd_arms_are_actually_in_there(records):
    """Guards the test above from passing vacuously if the corpus of YOLO
    records ever disappeared -- they are the ones at risk."""
    musgd_arms = {"yolo26n", "yolo26s", "yolo26m"}
    count = sum(1 for r in records if r.get("arm") in musgd_arms)
    assert count >= 100, (
        "only %d MuSGD-arm records found; this test is meant to cover them" % count
    )


def test_an_override_would_move_the_hash(records):
    """The other half: if an override did NOT change the identity, the contrast
    run would resume as its twin and never happen."""
    sample = next(r for r in records if r.get("script") == "03_run_cv")
    without = recompute(sample, None)
    fields = {
        key: (None if key == "extra" else sample.get(key))
        for key in registry.RUN_ID_FIELDS
    }
    fields["optimizer"] = "sgd"
    assert registry.compute_run_id(**fields) != without


def test_the_recorded_extra_does_not_reproduce_the_hash(records):
    """Documents a real provenance gap rather than asserting it is fine.

    `extra` is hashed as a short marker and stored as a dict, so a record cannot
    verify its own run_id without reading the source of the script that wrote
    it. This test exists so that if someone later makes the record
    self-verifying, it fails and points at this file.
    """
    sample = next(r for r in records if r.get("script") == "03_run_cv")
    assert recompute(sample, sample.get("extra")) != sample["run_id"]


def test_the_registry_is_the_expected_size(records):
    """A tripwire: if this number changes, every count in the manuscript and in
    the tests above is describing a different registry."""
    assert len(records) == 234, (
        "registry holds %d records, not 234. If runs were added deliberately, "
        "update this number and check every count that depends on it."
        % len(records)
    )
