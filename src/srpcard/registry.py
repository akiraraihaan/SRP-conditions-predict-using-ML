"""Append-only run registry at artifacts/registry.jsonl -- one line per completed run.

`run_id` is a deterministic SHA-1 over the parameters that DEFINE a run, so a
script can ask "is this already done?" before spending a GPU hour on it. Every
script checks the registry first and prints complete / skipped / remaining.

Append-only and flushed per record: a session killed at run 40 loses nothing, and
re-running the same command resumes at 41.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import artifacts_dir, git_commit, library_versions

# The fields that DEFINE a run. Anything not listed here (metrics, timings,
# hardware) is an outcome, not an identity, and must not enter the hash.
RUN_ID_FIELDS = (
    "script",
    "arm",
    "architecture",
    "split_kind",
    "repeat",
    "fold",
    "epochs",
    "batch",
    "lr",
    "class_weights",
    "run_seed",
    "extra",
)


def compute_run_id(**params: Any) -> str:
    """Deterministic id from the run-defining parameters only."""
    payload = {key: params.get(key) for key in RUN_ID_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]  # noqa: S324


def registry_path(cfg: dict[str, Any] | None = None) -> Path:
    return artifacts_dir(cfg) / "registry.jsonl"


def load_registry(path: Path | None = None) -> list[dict[str, Any]]:
    """Read every record. A truncated final line (killed mid-write) is skipped."""
    path = Path(path) if path is not None else registry_path()
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    "[registry] WARNING: skipping malformed line %d of %s "
                    "(likely a session killed mid-write)" % (number, path)
                )
    return records


def completed_run_ids(path: Path | None = None) -> set[str]:
    return {r["run_id"] for r in load_registry(path) if "run_id" in r}


def append_record(record: dict[str, Any], path: Path | None = None) -> Path:
    """Append one record and flush it to disk immediately."""
    path = Path(path) if path is not None else registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, sort_keys=False, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return path


def build_record(
    *,
    run_id: str,
    script: str,
    arm: str,
    architecture: str,
    split_kind: str,
    repeat: int | None,
    fold: int | None,
    epochs: int,
    batch: int,
    lr: float,
    class_weights: str,
    run_seed: int,
    val_seed: int | None,
    metrics: dict[str, Any],
    efficiency: dict[str, Any],
    wall_time_s: float,
    determinism_status: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one registry record. Every field the brief asks for is present."""
    return {
        "run_id": run_id,
        "script": script,
        "arm": arm,
        "architecture": architecture,
        "split_kind": split_kind,
        "repeat": repeat,
        "fold": fold,
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "class_weights": class_weights,
        "run_seed": run_seed,
        "val_seed": val_seed,
        # --- quality ---
        "f1_macro": metrics.get("f1_macro"),
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_per_class": metrics.get("f1_per_class"),
        "recall_per_class": metrics.get("recall_per_class"),
        "precision_per_class": metrics.get("precision_per_class"),
        "support_per_class": metrics.get("support_per_class"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "class_order": metrics.get("class_order"),
        "n_test_images": metrics.get("n_images"),
        # --- efficiency ---
        "params": efficiency.get("params"),
        "gflops": efficiency.get("gflops"),
        "size_mb": efficiency.get("size_mb"),
        "latency_ms_mean": efficiency.get("latency_ms_mean"),
        "latency_ms_std": efficiency.get("latency_ms_std"),
        # --- provenance ---
        "wall_time_s": wall_time_s,
        "determinism_status": determinism_status or {},
        "git_commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_versions": library_versions(),
        "extra": extra or {},
    }


def plan_runs(
    specs: Iterable[dict[str, Any]], path: Path | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split planned runs into (todo, already_done) by run_id.

    Each spec must already carry a "run_id" key.
    """
    done = completed_run_ids(path)
    todo = [s for s in specs if s["run_id"] not in done]
    skipped = [s for s in specs if s["run_id"] in done]
    return todo, skipped


def print_plan(script: str, todo: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    total = len(todo) + len(skipped)
    print(
        "[registry] %s: %d run(s) planned -- %d already complete (skipped), %d remaining"
        % (script, total, len(skipped), len(todo))
    )


def summarise(path: Path | None = None) -> dict[str, Any]:
    """Counts by script, arm and split_kind. For the end-of-run summary."""
    records = load_registry(path)
    by_script: dict[str, int] = {}
    by_arm: dict[str, int] = {}
    for record in records:
        by_script[record.get("script", "?")] = by_script.get(record.get("script", "?"), 0) + 1
        by_arm[record.get("arm", "?")] = by_arm.get(record.get("arm", "?"), 0) + 1
    return {"n_records": len(records), "by_script": by_script, "by_arm": by_arm}
