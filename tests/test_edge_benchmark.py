"""Script 07's arithmetic and its refusals.

Nothing here trains or benchmarks for real: the timing loops are the one part
that cannot be unit-tested meaningfully, so what is pinned instead is everything
a wrong number would flow through -- the statistics, the drift window, the
domination rule, the frontier comparison, and the refusal to benchmark a model
that was never trained.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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


# ---------------------------------------------------------------- statistics


def test_reports_median_iqr_and_p95_not_only_the_mean(bench):
    """On an edge device the tail is what disrupts operations."""
    samples = list(range(1, 101))
    out = bench.stats(samples)
    assert out["median_ms"] == pytest.approx(50.5)
    assert out["iqr_ms"] == pytest.approx(49.5)
    assert out["p95_ms"] == pytest.approx(95.05, abs=0.5)
    assert "mean_ms" in out and out["n"] == 100


def test_a_heavy_tail_moves_p95_but_not_the_median(bench):
    calm = bench.stats([10.0] * 99 + [10.0])
    spiky = bench.stats([10.0] * 99 + [900.0])
    assert spiky["median_ms"] == calm["median_ms"]
    assert spiky["p95_ms"] >= calm["p95_ms"]
    assert spiky["max_ms"] == 900.0


# ---------------------------------------------------------------- checkpoints


def test_a_missing_checkpoint_stops_the_run(bench, tmp_path):
    (tmp_path / "yolo26n.pt").write_bytes(b"x")
    with pytest.raises(SystemExit) as exc:
        bench.resolve_checkpoints(tmp_path, ["yolo26n", "resnet18", "yolo26m"])
    message = str(exc.value)
    assert "resnet18" in message and "yolo26m" in message
    assert "yolo26n" not in message.split("looked in")[0].split(":")[1]
    assert "Refusing to benchmark an untrained model" in message
    assert "do not persist weights" in message


def test_all_three_checkpoint_layouts_resolve(bench, tmp_path):
    (tmp_path / "a.pt").write_bytes(b"x")
    (tmp_path / "b.pth").write_bytes(b"x")
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "best.pt").write_bytes(b"x")
    found = bench.resolve_checkpoints(tmp_path, ["a", "b", "c"])
    assert found["a"].name == "a.pt"
    assert found["b"].name == "b.pth"
    assert found["c"].name == "best.pt"


def test_each_measurement_records_its_checkpoint(bench, tmp_path):
    target = tmp_path / "m.pt"
    target.write_bytes(b"some weights")
    assert len(bench.file_sha1(target)) == 16
    other = tmp_path / "n.pt"
    other.write_bytes(b"different weights")
    assert bench.file_sha1(target) != bench.file_sha1(other)


# ---------------------------------------------------------------- dominance


ROWS = {
    "cheap_good": {"f1_macro_mean": 0.59, "params": 1_500_000,
                   "latency_median_ms": 11.0, "int8_size_mb": 4.2},
    "dear_bad": {"f1_macro_mean": 0.52, "params": 10_000_000,
                 "latency_median_ms": 40.0, "int8_size_mb": 20.0},
    "dear_good": {"f1_macro_mean": 0.60, "params": 11_000_000,
                  "latency_median_ms": 45.0, "int8_size_mb": 21.0},
}
AXES = (("f1_macro_mean",), ("params", "latency_median_ms", "int8_size_mb"))


def test_domination_uses_measured_latency(bench):
    status = bench.dominance(ROWS, *AXES)
    assert status["cheap_good"]["on_frontier"] is True
    assert status["dear_good"]["on_frontier"] is True     # best F1, dearest
    assert status["dear_bad"]["on_frontier"] is False
    assert "cheap_good" in status["dear_bad"]["dominated_by"]
    assert "latency_median_ms" in status["dear_bad"]["dominated_by"]["cheap_good"]


def test_a_tie_is_not_domination(bench):
    same = {"x": dict(ROWS["cheap_good"]), "y": dict(ROWS["cheap_good"])}
    status = bench.dominance(same, *AXES)
    assert all(v["on_frontier"] for v in status.values())


# ---------------------------------------------------------------- frontier


def write_gflops_pareto(path: Path, frontier, others):
    lines = ["# a comment line the reader must skip",
             "arm,f1_macro_mean,on_pareto_frontier"]
    for arm in frontier:
        lines.append("%s,0.6,True" % arm)
    for arm in others:
        lines.append("%s,0.5,False" % arm)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_gflops_frontier_is_read_past_the_provenance_header(bench, tmp_path):
    path = tmp_path / "pareto_status.csv"
    write_gflops_pareto(path, ["a", "b"], ["c"])
    assert bench.read_gflops_frontier(path) == ["a", "b"]


def test_a_partial_arm_set_is_reported_as_not_comparable(bench, tmp_path, monkeypatch):
    """Comparing a 2-arm measured frontier against a 5-arm modelled one would
    report a difference caused by the arm set, not by the cost axis."""
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        "".join(
            json.dumps({"script": "03_run_cv", "arm": a, "f1_macro": f}) + "\n"
            for a, f in (("cheap_good", 0.59), ("dear_bad", 0.52))
        ),
        encoding="utf-8",
    )
    gflops = tmp_path / "pareto_status.csv"
    write_gflops_pareto(gflops, ["cheap_good", "resnet18"], ["dear_bad"])

    results = {
        arm: {"params": ROWS[arm]["params"],
              "full_pipeline": {"median_ms": ROWS[arm]["latency_median_ms"]},
              "int8": {"int8_size_mb": ROWS[arm]["int8_size_mb"]}}
        for arm in ("cheap_good", "dear_bad")
    }
    out = bench.write_device_pareto(results, registry, tmp_path / "d.csv", gflops)
    assert out["written"] is True
    assert out["comparison_is_partial"] is True
    assert out["frontier_agrees"] is False, "a partial set must never claim agreement"


def test_a_complete_arm_set_can_agree(bench, tmp_path):
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        "".join(
            json.dumps({"script": "03_run_cv", "arm": a, "f1_macro": ROWS[a]["f1_macro_mean"]}) + "\n"
            for a in ROWS
        ),
        encoding="utf-8",
    )
    gflops = tmp_path / "pareto_status.csv"
    write_gflops_pareto(gflops, ["cheap_good", "dear_good"], ["dear_bad"])

    results = {
        arm: {"params": ROWS[arm]["params"],
              "full_pipeline": {"median_ms": ROWS[arm]["latency_median_ms"]},
              "int8": {"int8_size_mb": ROWS[arm]["int8_size_mb"]}}
        for arm in ROWS
    }
    out = bench.write_device_pareto(results, registry, tmp_path / "d.csv", gflops)
    assert out["comparison_is_partial"] is False
    assert out["frontier_agrees"] is True
    assert out["frontier_measured"] == ["cheap_good", "dear_good"]

    text = (tmp_path / "d.csv").read_text(encoding="utf-8")
    assert "latency_median_ms" in text and "int8_size_mb" in text


def test_registry_mean_f1_averages_the_folds(bench, tmp_path):
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        "".join(
            json.dumps({"script": "03_run_cv", "arm": "a", "f1_macro": f}) + "\n"
            for f in (0.4, 0.6)
        )
        + json.dumps({"script": "05_learning_curve", "arm": "a", "f1_macro": 0.9}) + "\n",
        encoding="utf-8",
    )
    # the learning-curve record must not enter the cross-validation mean
    assert bench.registry_mean_f1(registry) == {"a": pytest.approx(0.5)}


# ---------------------------------------------------------------- labels


def test_a_flat_directory_gives_no_labels(bench, tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    assert bench.labelled_images(tmp_path, ["cat", "dog"]) == []


def test_class_subdirectories_give_labels(bench, tmp_path):
    for index, name in enumerate(("cat", "dog")):
        (tmp_path / name).mkdir()
        (tmp_path / name / ("%d.png" % index)).write_bytes(b"x")
    pairs = bench.labelled_images(tmp_path, ["cat", "dog"])
    assert [label for _, label in pairs] == [0, 1]


# ---------------------------------------------------------------- constraints


def test_the_script_refuses_cuda_and_never_trains():
    source = (REPO_ROOT / "scripts" / "07_bench_edge.py").read_text(encoding="utf-8")
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = ""' in source
    assert "torch.cuda.is_available()" in source
    # check IMPORTS, not prose: the docstring legitimately mentions both names
    import ast

    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            if node.module.startswith("srpcard"):
                imported.add(node.module)
    assert "srpcard.train" not in imported, "07 must not import the training loop"
    assert "pandas" not in imported, "pandas is not in the Pi dependency set"
    assert "ultralytics" not in imported


def test_cooling_is_recorded_and_unknown_is_flagged():
    source = (REPO_ROOT / "scripts" / "07_bench_edge.py").read_text(encoding="utf-8")
    assert '"--cooling"' in source
    assert "not interpretable" in source


def test_the_device_block_is_top_level(bench, tmp_path):
    source = (REPO_ROOT / "scripts" / "07_bench_edge.py").read_text(encoding="utf-8")
    assert '"device": device,' in source
    for field in ("cpu_model", "cpu_count", "os", "python", "torch",
                  "torch_threads", "power_mode"):
        assert field in source, field
