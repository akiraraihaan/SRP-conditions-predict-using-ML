"""A backfill either restores every record it planned to, or none of them.

The incident: the registry was overwritten by a copy that predated two backfills,
dropping the hardware block from 63 records and, separately, exposing that script
05 had never recorded efficiency figures at all. The obvious repair --
`backfill_efficiency.py` -- would have rebuilt every model to derive the values,
and on a machine whose ultralytics cannot load yolo26*-cls.pt it would have
filled the mobilenet records, printed SKIPPED for the YOLO ones, and written the
result anyway.

That half-restored registry is worse than the state it started from. Every record
still validates, every run_id still matches, so the half that was filled is
indistinguishable from the half that was not, and nothing downstream can tell.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from srpcard import registry  # noqa: E402


@pytest.fixture(scope="module")
def backfill():
    spec = importlib.util.spec_from_file_location(
        "backfill_efficiency", REPO_ROOT / "scripts" / "backfill_efficiency.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["backfill_efficiency"] = module
    spec.loader.exec_module(module)
    return module


DERIVED = {
    "params": 1528106,
    "gflops": 0.122928368,
    "size_mb_fp32": 5.94,
    "size_mb_fp16": 3.004,
    "size_mb_fp16_payload": 2.9,
    "size_mb_fp32_payload": 5.8,
}


def record(run_id, arm, architecture, *, derived=True, hardware=True,
           script="03_run_cv"):
    out = {
        "run_id": run_id,
        "script": script,
        "arm": arm,
        "architecture": architecture,
        "library_versions": {
            "gpu": "Tesla T4",
            "torch_cuda": "13.0",
            "cuda_available": "True",
        },
    }
    if derived:
        out.update(DERIVED)
        out["size_mb"] = DERIVED["size_mb_fp16"]
    else:
        out.update({field: None for field in DERIVED})
        out["size_mb"] = None
    if hardware:
        out.update({
            "gpu": "Tesla T4",
            "gpu_count": 1,
            "cuda_version": "13.0",
            "driver_version": None,
            "compute_capability": None,
            "device_kind": "cuda",
        })
    return out


def write(path, records):
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )


# ------------------------------------------------------------------ atomicity


def test_nothing_is_written_when_one_arm_cannot_be_built(
    backfill, registry_path, monkeypatch, capsys
):
    fillable = record("a1", "mobilenetv3_small", "mobilenet_v3_small", derived=False)
    blocked = record("b1", "yolo26m", "yolo26m-cls", derived=False)
    write(registry_path, [fillable, blocked])
    before = registry_path.read_bytes()

    def only_yolo_fails(arm, *args, **kwargs):
        if str(arm).startswith("yolo"):
            raise RuntimeError("Can't get attribute 'C3k2'")
        raise AssertionError("unreachable: the plan must abort before applying")

    monkeypatch.setattr(backfill, "build_model", only_yolo_fails)
    monkeypatch.setattr(backfill.registry, "registry_path", lambda: registry_path)
    monkeypatch.setattr(sys, "argv", ["backfill_efficiency.py"])

    assert backfill.main() == 1
    assert registry_path.read_bytes() == before, "a refused backfill still wrote"

    out = capsys.readouterr().out
    assert "REFUSING TO WRITE" in out
    assert "NOTHING has been written" in out
    assert "C3k2" in out, "the reason the arm failed must be named"


def test_a_refused_backfill_leaves_no_backup_either(
    backfill, registry_path, monkeypatch
):
    write(registry_path, [record("b1", "yolo26m", "yolo26m-cls", derived=False)])

    def fails(arm, *args, **kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(backfill, "build_model", fails)
    monkeypatch.setattr(backfill.registry, "registry_path", lambda: registry_path)
    monkeypatch.setattr(sys, "argv", ["backfill_efficiency.py"])

    assert backfill.main() == 1
    assert list(registry_path.parent.glob("*.bak")) == []


def test_hardware_only_recovery_needs_no_model(
    backfill, registry_path, monkeypatch, capsys
):
    """The 54 uniform-grid records need nothing but their own library_versions.

    Requiring a model for them would let an ultralytics failure block a repair
    that never depended on ultralytics.
    """
    write(registry_path, [record("h1", "yolo26n", "yolo26n-cls", hardware=False)])

    def never(arm, *args, **kwargs):
        raise AssertionError("built a model for a hardware-only record")

    monkeypatch.setattr(backfill, "build_model", never)
    monkeypatch.setattr(backfill.registry, "registry_path", lambda: registry_path)
    monkeypatch.setattr(sys, "argv", ["backfill_efficiency.py"])

    assert backfill.main() == 0
    restored = json.loads(registry_path.read_text(encoding="utf-8").splitlines()[0])
    assert restored["gpu"] == "Tesla T4"
    assert restored["device_kind"] == "cuda"
    assert "to build : none" in capsys.readouterr().out


# ------------------------------------------------------------ sibling records


def test_a_sibling_record_is_preferred_over_measuring_here(
    backfill, registry_path, monkeypatch, capsys
):
    """Two figures for one model, in one table, is the failure this prevents.

    torch's zip container costs a few kilobytes that differ between builds, so a
    size measured on this machine lands ~0.003 MB from the one the run recorded.
    """
    sibling = record("s1", "mobilenetv3_small", "mobilenet_v3_small")
    empty = record("s2", "mobilenetv3_small", "mobilenet_v3_small",
                   derived=False, script="05_learning_curve")
    write(registry_path, [sibling, empty])

    def never(arm, *args, **kwargs):
        raise AssertionError("measured locally when the registry already knew")

    monkeypatch.setattr(backfill, "build_model", never)
    monkeypatch.setattr(backfill.registry, "registry_path", lambda: registry_path)
    monkeypatch.setattr(sys, "argv", ["backfill_efficiency.py"])

    assert backfill.main() == 0
    filled = json.loads(registry_path.read_text(encoding="utf-8").splitlines()[1])
    assert filled["size_mb_fp16"] == DERIVED["size_mb_fp16"]
    assert filled["params"] == DERIVED["params"]
    assert "<- registry" in capsys.readouterr().out


def test_only_the_disagreeing_FIELD_is_excluded(backfill):
    """yolo26m-cls is the case that motivated this.

    41 complete records, unanimous that params is 10,366,026 and gflops 4.8512,
    disagreeing only on the two container-inclusive sizes. Checking unanimity
    over the whole tuple threw away all six, so the plan wanted a model rebuilt
    to recover two numbers the registry already stated without dissent.
    """
    a = record("x1", "yolo26m", "yolo26m-cls")
    b = record("x2", "yolo26m", "yolo26m-cls")
    b["size_mb_fp16"] = a["size_mb_fp16"] + 0.006

    resolved, rejected = backfill.derived_from_siblings([a, b])

    assert resolved["yolo26m-cls"]["params"] == a["params"]
    assert resolved["yolo26m-cls"]["gflops"] == a["gflops"]
    assert "size_mb_fp16" not in resolved["yolo26m-cls"]
    assert rejected["yolo26m-cls"]["size_mb_fp16"] == sorted(
        [a["size_mb_fp16"], b["size_mb_fp16"]]
    )


def test_a_disagreement_is_excluded_never_arbitrated(backfill):
    """Not the majority, not the first, not the newest -- excluded."""
    records = [record("x%d" % i, "yolo26m", "yolo26m-cls") for i in range(5)]
    records[4]["size_mb_fp32"] = records[0]["size_mb_fp32"] + 0.006

    resolved, rejected = backfill.derived_from_siblings(records)

    assert "size_mb_fp32" not in resolved["yolo26m-cls"], (
        "four records against one is still a disagreement"
    )
    assert len(rejected["yolo26m-cls"]["size_mb_fp32"]) == 2


def test_the_lookup_is_not_scoped_by_script(backfill):
    """params, gflops and the size_mb family are functions of the architecture
    alone, so the script that wrote the record is irrelevant."""
    grid = record("g1", "yolo26n", "yolo26n-cls", script="01b_uniform_grid")
    cv = record("c1", "yolo26n", "yolo26n-cls", script="03_run_cv")
    lc = record("l1", "yolo26n", "yolo26n-cls", script="05_learning_curve")

    resolved, rejected = backfill.derived_from_siblings([grid, cv, lc])

    assert rejected == {}
    assert resolved["yolo26n-cls"]["params"] == grid["params"]


def test_architecture_not_arm_is_the_key(backfill):
    """Two arms can share an architecture; the constants are the same either way."""
    one = record("a1", "yolo26n", "yolo26n-cls")
    two = record("a2", "yolo26n_alt", "yolo26n-cls")

    resolved, _ = backfill.derived_from_siblings([one, two])

    assert set(resolved) == {"yolo26n-cls"}


def test_a_partly_resolved_architecture_still_needs_a_model(backfill, registry_path,
                                                            monkeypatch, capsys):
    """Honest reporting: yolo26m resolves four of six fields from the registry
    and must still be built for the other two."""
    complete = record("k1", "yolo26m", "yolo26m-cls")
    other = record("k2", "yolo26m", "yolo26m-cls")
    other["size_mb_fp16"] = complete["size_mb_fp16"] + 0.006
    other["size_mb_fp32"] = complete["size_mb_fp32"] + 0.006
    empty = record("k3", "yolo26m", "yolo26m-cls", derived=False)
    write(registry_path, [complete, other, empty])

    def fails(arm, *args, **kwargs):
        raise RuntimeError("Can't get attribute 'C3k2'")

    monkeypatch.setattr(backfill, "build_model", fails)
    monkeypatch.setattr(backfill.registry, "registry_path", lambda: registry_path)
    monkeypatch.setattr(sys, "argv", ["backfill_efficiency.py"])

    assert backfill.main() == 1
    out = capsys.readouterr().out
    assert "needs size_mb_fp16, size_mb_fp32" in out
    assert "params" not in out.split("to build")[1].split("REFUSING")[0]


# --------------------------------------------------------- the grouped report


def test_the_audit_groups_missing_fields_by_script(registry_path):
    write(registry_path, [
        record("a", "yolo26n", "yolo26n-cls", hardware=False,
               script="01b_uniform_grid"),
        record("b", "yolo26n", "yolo26n-cls", hardware=False,
               script="01b_uniform_grid"),
        record("c", "mobilenetv3_small", "mobilenet_v3_small", derived=False,
               script="05_learning_curve"),
        record("d", "mobilenetv3_small", "mobilenet_v3_small"),
    ])
    audit = registry.derived_field_audit(registry_path)

    assert audit["n_records"] == 4
    assert audit["n_affected"] == 3
    assert audit["by_script"]["01b_uniform_grid"]["missing"]["gpu"] == 2
    assert audit["by_script"]["05_learning_curve"]["missing"]["params"] == 1
    assert audit["by_script"]["03_run_cv"]["missing"] == {}


def test_the_report_says_so_when_nothing_is_missing(registry_path, capsys):
    write(registry_path, [record("d", "mobilenetv3_small", "mobilenet_v3_small")])
    assert registry.print_derived_field_report(registry_path) is True
    assert "every recoverable field populated" in capsys.readouterr().out


def test_the_report_names_the_script_and_the_count(registry_path, capsys):
    """One line per script, not one line per record: 63 near-identical lines
    scroll the reason for them off the screen."""
    write(registry_path, [
        record("a%d" % i, "yolo26n", "yolo26n-cls", hardware=False,
               script="01b_uniform_grid")
        for i in range(54)
    ])
    assert registry.print_derived_field_report(registry_path) is False
    out = capsys.readouterr().out
    assert "01b_uniform_grid" in out
    assert "gpu x54" in out
    assert len(out.splitlines()) < 25, "meant to be readable at a glance"


def test_never_recorded_fields_are_not_called_missing():
    """driver_version was never captured by any run. A null there is the honest
    value, not a lost backfill, and reporting it would cry wolf forever."""
    for field in registry.NEVER_RECORDED_FIELDS:
        assert field not in registry.RECOVERABLE_FIELDS
