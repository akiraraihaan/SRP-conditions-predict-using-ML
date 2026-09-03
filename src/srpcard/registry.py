"""Append-only run registry at artifacts/registry.jsonl -- one line per completed run.

`run_id` is a deterministic SHA-1 over the parameters that DEFINE a run, so a
script can ask "is this already done?" before spending a GPU hour on it. Every
script checks the registry first and prints complete / skipped / remaining.

Append-only and flushed per record: a session killed at run 40 loses nothing, and
re-running the same command resumes at 41.

That resume mechanism is also the reason this module polices its own schema.
A record written by an older version of the code still matches by `run_id`, so
the run it stands for is SKIPPED and never re-run -- the stale record is
inherited into the final results instead of being replaced. `REQUIRED_RECORD_FIELDS`
is the current schema: `append_record` refuses to write a record missing any of
them, and `audit_registry` (called from `print_plan`, once per script invocation)
warns loudly about any record already on disk that lacks them.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import (
    RUN_DEFINING_HYPERPARAMETERS,
    artifacts_dir,
    git_commit,
    library_versions,
)

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


# Every field a record must carry to be readable as CURRENT-schema. Presence is
# what is checked, not truthiness: a legitimately inapplicable value is None with
# the reason recorded beside it (script 01 has no uniform-protocol training
# outcome, for instance), and that is different from the field being absent
# because an older version of this code did not know about it.
REQUIRED_RECORD_FIELDS = (
    # identity -- these also feed run_id
    "run_id",
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
    "val_seed",
    # what was actually loaded and proved, rather than what was requested
    "checkpoint_resolved",
    "pretrained_fallback_used",
    "class_weights_verified",
    "class_weights_proof",
    "corpus_fingerprint",
    # training outcome
    "selected_epoch",
    "epochs_run",
    "stopped_early",
    "best_val_f1",
    "best_val_loss",
    "min_val_loss_epoch",
    "history",
    # quality
    "f1_macro",
    "accuracy",
    "confusion_matrix",
    "class_order",
    # provenance
    "wall_time_s",
    "determinism_status",
    "git_commit",
    "timestamp",
    "library_versions",
    "extra",
)

# The training-outcome block, filled from a TrainResult or explicitly absent.
TRAINING_FIELDS = (
    "selected_epoch",
    "epochs_run",
    "stopped_early",
    "best_val_f1",
    "best_val_loss",
    "min_val_loss_epoch",
    "history",
)


def training_outcome(result: Any) -> dict[str, Any]:
    """The uniform-protocol training outcome, from a `train.TrainResult`."""
    return {
        "selected_epoch": result.best_epoch,
        "epochs_run": len(result.history),
        "stopped_early": bool(result.stopped_early),
        "best_val_f1": result.best_val_f1,
        "best_val_loss": result.best_val_loss,
        "min_val_loss_epoch": result.min_val_loss_epoch,
        "history": result.history,
    }


def training_outcome_absent(reason: str, *, epochs_run: int | None = None) -> dict[str, Any]:
    """For a run trained outside the uniform loop -- script 01's legacy protocol.

    The fields are present and None, with the reason recorded, so the schema check
    distinguishes "does not apply here" from "written before this field existed".
    """
    block: dict[str, Any] = {field: None for field in TRAINING_FIELDS}
    block["epochs_run"] = epochs_run
    block["training_outcome_absent_reason"] = reason
    return block


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


def missing_fields(record: dict[str, Any]) -> list[str]:
    """Which REQUIRED_RECORD_FIELDS this record does not carry."""
    return [field for field in REQUIRED_RECORD_FIELDS if field not in record]


def audit_registry(path: Path | None = None) -> dict[str, Any]:
    """Find records on disk that predate the current schema.

    Returns the incomplete ones with the fields each is missing. Never raises:
    the caller decides whether a stale record is fatal.
    """
    path = Path(path) if path is not None else registry_path()
    records = load_registry(path)
    incomplete = []
    for position, record in enumerate(records, 1):
        missing = missing_fields(record)
        if missing:
            incomplete.append(
                {
                    "line": position,
                    "run_id": record.get("run_id", "?"),
                    "script": record.get("script", "?"),
                    "arm": record.get("arm", "?"),
                    "repeat": record.get("repeat"),
                    "fold": record.get("fold"),
                    "missing": missing,
                }
            )
    return {"path": str(path), "n_records": len(records), "incomplete": incomplete}


def warn_if_stale(path: Path | None = None) -> bool:
    """Print a loud block if any record on disk predates the current schema.

    Returns True when the registry is clean. Called once per script invocation
    from `print_plan`, because a stale record is not merely untidy: its run_id
    still matches, so the run it stands for is SKIPPED and the stale numbers are
    inherited into the final results.
    """
    audit = audit_registry(path)
    if not audit["incomplete"]:
        return True

    bar = "!" * 74
    print("\n" + bar)
    print(
        "REGISTRY SCHEMA DRIFT -- %d of %d record(s) predate the current schema"
        % (len(audit["incomplete"]), audit["n_records"])
    )
    print("  %s" % audit["path"])
    for entry in audit["incomplete"]:
        print(
            "  line %-4d %-16s %-22s %-12s r%sf%s"
            % (
                entry["line"],
                entry["run_id"],
                entry["script"],
                entry["arm"],
                entry["repeat"],
                entry["fold"],
            )
        )
        print("            missing: %s" % ", ".join(entry["missing"]))
    print(
        "  These runs are SKIPPED by run_id, so they will not be re-run and their\n"
        "  older numbers would be inherited into the final results. Delete the\n"
        "  offending line(s) from the registry and let the run happen again."
    )
    print(bar + "\n")
    return False


def append_record(record: dict[str, Any], path: Path | None = None) -> Path:
    """Append one record and flush it to disk immediately.

    Refuses a record that does not carry the current schema -- that is this code's
    own bug, and writing it would plant exactly the stale record `warn_if_stale`
    exists to catch.
    """
    missing = missing_fields(record)
    if missing:
        raise ValueError(
            "Refusing to append a registry record missing %d required field(s): %s\n"
            "  run_id %s (%s, %s)\n"
            "  Every field in registry.REQUIRED_RECORD_FIELDS must be present, even\n"
            "  if its value is None. See registry.training_outcome_absent()."
            % (
                len(missing),
                ", ".join(missing),
                record.get("run_id", "?"),
                record.get("script", "?"),
                record.get("arm", "?"),
            )
        )
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
    checkpoint_resolved: str,
    pretrained_fallback_used: bool,
    class_weights_verified: bool | None,
    class_weights_proof: dict[str, Any],
    corpus_fingerprint: dict[str, Any],
    training: dict[str, Any],
    metrics: dict[str, Any],
    efficiency: dict[str, Any],
    wall_time_s: float,
    determinism_status: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one registry record. Every field the brief asks for is present.

    Nothing here is optional. Each argument that could have been defaulted is one
    a future caller could forget, and a forgotten field is a stale record that
    still matches by run_id and is therefore never re-run:

    - `checkpoint_resolved` / `pretrained_fallback_used` -- which pretrained
      weights the run actually loaded, so a YOLO11 fallback standing in for a
      YOLO26 arm is visible in the record rather than only in a console line
      nobody kept (src/srpcard/models.py).
    - `class_weights_verified` / `class_weights_proof` -- the measured proof that
      the balanced weights reach the loss. The legacy pipeline computed them,
      printed them, charted them and never applied them; that proof has to travel
      with the results (src/srpcard/train.py:verify_class_weights_applied).
    - `corpus_fingerprint` -- the corpus the run was produced on. Verified at load
      time by every consumer of folds.json; recorded here so the result carries it.
    - `training` -- the uniform-protocol outcome, from `training_outcome(result)`
      or `training_outcome_absent(reason)`.
    """
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
        # --- what was actually loaded and proved, not what was requested ---
        "checkpoint_resolved": checkpoint_resolved,
        "pretrained_fallback_used": bool(pretrained_fallback_used),
        "class_weights_verified": class_weights_verified,
        "class_weights_proof": class_weights_proof,
        "corpus_fingerprint": corpus_fingerprint,
        # --- training outcome (None throughout when the uniform loop was not used) ---
        **{field: training.get(field) for field in TRAINING_FIELDS},
        "training_outcome_absent_reason": training.get("training_outcome_absent_reason"),
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
        # Three size measurements, never one. fp32 and fp16 differ by ~2x, and the
        # manuscript's published sizes are the fp16 ones. See efficiency.py.
        "size_mb": efficiency.get("size_mb"),
        "size_mb_fp32": efficiency.get("size_mb_fp32"),
        "size_mb_fp16": efficiency.get("size_mb_fp16"),
        "size_mb_checkpoint_file": efficiency.get("size_mb_checkpoint_file"),
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


def print_plan(
    script: str,
    todo: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    path: Path | None = None,
) -> None:
    total = len(todo) + len(skipped)
    print(
        "[registry] %s: %d run(s) planned -- %d already complete (skipped), %d remaining"
        % (script, total, len(skipped), len(todo))
    )
    # Once per script invocation, right where the skip count is printed: those
    # skips are exactly what a stale record buys you.
    warn_if_stale(path)


# --------------------------------------------------------------------------
# hyperparameter drift
# --------------------------------------------------------------------------
#
# epochs, batch and lr are in RUN_ID_FIELDS. Change one and the same fold of the
# same arm hashes to a DIFFERENT run_id, so nothing is skipped and the run
# happens again under the new settings -- leaving the registry holding two
# hyperparameter regimes for one arm, which aggregate.py would then average
# together without noticing.
#
# The realistic way that happens: scripts 01 and 02 rewrite configs/arms.yaml,
# the session dies before that file is committed, and the next clone carries the
# PROVISIONAL yolo26m config (or a baseline lr of null) again.


class HyperparameterDriftError(RuntimeError):
    """The loaded config disagrees with the config completed runs were trained under."""


def completed_hyperparameters(
    arm: str, split_kind: str = "cv", path: Path | None = None
) -> dict[tuple, list[dict[str, Any]]]:
    """Completed runs of one arm, grouped by their (epochs, batch, lr).

    More than one key means the registry already holds mixed regimes.
    """
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for record in load_registry(path):
        if record.get("arm") != arm or record.get("split_kind") != split_kind:
            continue
        key = tuple(record.get(field) for field in RUN_DEFINING_HYPERPARAMETERS)
        groups.setdefault(key, []).append(record)
    return groups


def _format_hyperparameters(key) -> str:
    return "  ".join(
        "%s %s" % (field, value)
        for field, value in zip(RUN_DEFINING_HYPERPARAMETERS, key)
    )


def assert_config_matches_registry(
    *,
    script: str,
    arm: str,
    epochs: int,
    batch: int,
    lr: float,
    split_kind: str = "cv",
    path: Path | None = None,
) -> None:
    """Refuse to add runs under hyperparameters that disagree with completed ones.

    Called by every script that writes cv-kind records, once per arm, BEFORE the
    run loop. Aborts rather than proceeding: a mismatch is never a new run, it is
    the same experiment about to be duplicated under different settings.
    """
    groups = completed_hyperparameters(arm, split_kind, path)
    if not groups:
        return

    wanted = (epochs, batch, lr)
    mismatched = {key: records for key, records in groups.items() if key != wanted}
    if not mismatched:
        return

    lines = [
        "HYPERPARAMETER DRIFT for arm %r -- refusing to run %s." % (arm, script),
        "",
        "  The registry already holds completed runs of this arm trained under",
        "  DIFFERENT hyperparameters than configs/arms.yaml currently specifies.",
        "  epochs, batch and lr feed the run_id hash, so proceeding would not skip",
        "  those folds -- it would retrain them under the loaded config and leave",
        "  two regimes in the registry for one arm. aggregate.py would then average",
        "  across both without noticing.",
        "",
        "  %-26s %s" % ("configs/arms.yaml (loaded)", _format_hyperparameters(wanted)),
    ]
    for key, records in sorted(mismatched.items(), key=lambda kv: -len(kv[1])):
        lines.append(
            "  %-26s %s   (%d completed run(s))"
            % ("registry", _format_hyperparameters(key), len(records))
        )
    lines.append("")
    lines.append("  Affected run_ids:")
    for key, records in sorted(mismatched.items(), key=lambda kv: -len(kv[1])):
        for record in records[:20]:
            lines.append(
                "    %-16s %-24s r%sf%s"
                % (
                    record.get("run_id", "?"),
                    record.get("script", "?"),
                    record.get("repeat"),
                    record.get("fold"),
                )
            )
        if len(records) > 20:
            lines.append("    ... and %d more" % (len(records) - 20))
    lines += [
        "",
        "  Almost always this means a resolved configuration was lost: scripts 01",
        "  and 02 rewrite configs/arms.yaml, and a session that ended before that",
        "  file was committed leaves the next clone with the provisional values.",
        "",
        "  Recover the resolved config, do not overwrite the runs:",
        "      python scripts/restore_arms.py",
        "",
        "  If instead you deliberately changed the hyperparameters, the completed",
        "  runs above belong to the old configuration and must be deleted from",
        "  artifacts/registry.jsonl before this arm is run again.",
    ]
    raise HyperparameterDriftError("\n".join(lines))


def assert_arms_match_registry(
    *,
    script: str,
    arms: list[str],
    arms_cfg: dict[str, Any],
    split_kind: str = "cv",
    path: Path | None = None,
) -> None:
    """`assert_config_matches_registry` for each arm a script is about to run."""
    for arm in arms:
        arm_cfg = (arms_cfg.get("arms") or {}).get(arm)
        if not arm_cfg or arm_cfg.get("lr") is None:
            continue  # an unresolved arm is skipped by the caller anyway
        assert_config_matches_registry(
            script=script,
            arm=arm,
            epochs=int(arm_cfg["epochs"]),
            batch=int(arm_cfg["batch"]),
            lr=float(arm_cfg["lr"]),
            split_kind=split_kind,
            path=path,
        )
    print(
        "[config] %d arm(s) checked against completed runs -- no hyperparameter drift"
        % len(arms)
    )


def summarise(path: Path | None = None) -> dict[str, Any]:
    """Counts by script, arm and split_kind. For the end-of-run summary."""
    records = load_registry(path)
    by_script: dict[str, int] = {}
    by_arm: dict[str, int] = {}
    for record in records:
        by_script[record.get("script", "?")] = by_script.get(record.get("script", "?"), 0) + 1
        by_arm[record.get("arm", "?")] = by_arm.get(record.get("arm", "?"), 0) + 1
    fallback = sorted(
        {record.get("arm", "?") for record in records if record.get("pretrained_fallback_used")}
    )
    incomplete = [r for r in records if missing_fields(r)]
    unverified = [
        r for r in records if r.get("class_weights_verified") is False
    ]
    return {
        "n_records": len(records),
        "by_script": by_script,
        "by_arm": by_arm,
        "n_pretrained_fallback": sum(
            1 for record in records if record.get("pretrained_fallback_used")
        ),
        "arms_with_pretrained_fallback": fallback,
        "n_incomplete_schema": len(incomplete),
        "n_class_weights_unverified": len(unverified),
        "corpus_fingerprints": sorted(
            {
                (r.get("corpus_fingerprint") or {}).get("sha1_of_sorted_included_sha1s", "?")
                for r in records
            }
        ),
    }
