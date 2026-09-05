"""Merging two diverged copies of the registry.

The divergence that happened: the repository held 63 backfilled records with
device provenance, while the other copy held the same 63 without the backfill
plus 6 new runs from script 02. Neither was a superset, so overwriting either
direction lost work.

Everything here runs against tmp_path.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from srpcard import registry  # noqa: E402


def record(run_id, **overrides):
    base = {
        "run_id": run_id,
        "script": "01b_uniform_grid",
        "arm": "yolo26n",
        "split_kind": "dev",
        "repeat": None,
        "fold": None,
        "f1_macro": 0.6,
        "accuracy": 0.7,
        "wall_time_s": 42.0,
        "confusion_matrix": [[1, 0], [0, 1]],
    }
    base.update(overrides)
    return base


def write(path: Path, records) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


# ---------------------------------------------------------------- core


def test_union_by_run_id():
    a = [record("x"), record("y")]
    b = [record("y"), record("z")]
    merged, report = registry.merge_registries(a, b)
    assert {r["run_id"] for r in merged} == {"x", "y", "z"}
    assert report["n_merged"] == 3
    assert report["only_a"] == ["x"] and report["only_b"] == ["z"]
    assert report["shared"] == ["y"]


def test_the_more_populated_record_wins_and_gaps_are_filled():
    sparse = record("x", gpu=None, cuda_version=None)
    full = record("x", gpu="Tesla T4", cuda_version="13.0")
    merged, report = registry.merge_registries([sparse], [full])
    assert merged[0]["gpu"] == "Tesla T4"
    assert merged[0]["cuda_version"] == "13.0"
    assert report["reconciled"] == [] or "gpu" in report["reconciled"][0]["filled"]


def test_nothing_populated_is_overwritten():
    a = record("x", gpu="Tesla T4", accuracy=0.7)
    b = record("x", gpu=None)
    merged, _ = registry.merge_registries([a], [b])
    assert merged[0]["gpu"] == "Tesla T4"
    assert merged[0]["accuracy"] == 0.7


def test_a_measured_disagreement_refuses_the_whole_merge():
    a = [record("x"), record("y")]
    b = [record("x", f1_macro=0.99), record("y")]
    with pytest.raises(registry.RegistryConflictError) as exc:
        registry.merge_registries(a, b)
    message = str(exc.value)
    assert "REGISTRY CONFLICT" in message
    assert "f1_macro" in message and "x" in message
    assert "DIFFERENT RUNS wearing one id" in message


@pytest.mark.parametrize(
    "field,value",
    [
        ("f1_macro", 0.99),
        ("accuracy", 0.11),
        ("confusion_matrix", [[9, 9], [9, 9]]),
        ("wall_time_s", 1.0),
        ("f1_per_class", {"a": 0.1}),
    ],
)
def test_every_measured_field_is_guarded(field, value):
    a = [record("x", **{field: 0.5 if field != "confusion_matrix" else [[1, 0], [0, 1]]})]
    b = [record("x", **{field: value})]
    with pytest.raises(registry.RegistryConflictError):
        registry.merge_registries(a, b)


def test_absence_is_not_disagreement():
    """A field one copy lacks is exactly what merging fills, not a conflict."""
    a = [record("x", f1_macro=0.6)]
    b = [record("x", f1_macro=None)]
    merged, _ = registry.merge_registries(a, b)
    assert merged[0]["f1_macro"] == 0.6


def test_order_is_stable():
    a = [record("a1"), record("a2")]
    b = [record("a2"), record("b1")]
    merged, _ = registry.merge_registries(a, b)
    assert [r["run_id"] for r in merged] == ["a1", "a2", "b1"]


# ---------------------------------------------------------------- CLI


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "merge_registry.py"), *args],
        capture_output=True,
        text=True,
    )


def test_cli_merges_and_reports(tmp_path):
    """The real shape: 63 backfilled + (63 pre-backfill + 6 new) = 69."""
    backfilled = [record("r%02d" % i, gpu="Tesla T4") for i in range(63)]
    pre = [record("r%02d" % i, gpu=None) for i in range(63)]
    new = [record("s%02d" % i, script="02_lr_sweep_baselines") for i in range(6)]

    a = write(tmp_path / "A.jsonl", backfilled)
    b = write(tmp_path / "B.jsonl", pre + new)
    out = tmp_path / "C.jsonl"

    result = run_cli(str(a), str(b), "--out", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGED TOTAL                       69" in result.stdout
    assert len(registry.load_registry(out)) == 69
    assert all(r["gpu"] == "Tesla T4" for r in registry.load_registry(out)[:63])


def test_cli_dry_run_writes_nothing(tmp_path):
    a = write(tmp_path / "A.jsonl", [record("x")])
    b = write(tmp_path / "B.jsonl", [record("y")])
    out = tmp_path / "C.jsonl"
    result = run_cli(str(a), str(b), "--out", str(out), "--dry-run")
    assert result.returncode == 0
    assert not out.exists()


def test_cli_refuses_a_conflict_and_writes_nothing(tmp_path):
    a = write(tmp_path / "A.jsonl", [record("x", f1_macro=0.5)])
    b = write(tmp_path / "B.jsonl", [record("x", f1_macro=0.9)])
    out = tmp_path / "C.jsonl"
    result = run_cli(str(a), str(b), "--out", str(out))
    assert result.returncode == 1
    assert "REGISTRY CONFLICT" in result.stdout
    assert not out.exists()


def test_cli_backs_up_an_existing_output(tmp_path):
    a = write(tmp_path / "A.jsonl", [record("x")])
    b = write(tmp_path / "B.jsonl", [record("y")])
    out = write(tmp_path / "C.jsonl", [record("old")])
    result = run_cli(str(a), str(b), "--out", str(out))
    assert result.returncode == 0
    backups = list(tmp_path.glob("C.jsonl.*.bak"))
    assert len(backups) == 1
    assert [r["run_id"] for r in registry.load_registry(backups[0])] == ["old"]


def test_cli_rejects_a_missing_input(tmp_path):
    a = write(tmp_path / "A.jsonl", [record("x")])
    result = run_cli(str(a), str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "C.jsonl"))
    assert result.returncode == 2
