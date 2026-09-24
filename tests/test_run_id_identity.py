"""Adding a field to a run's identity must not rewrite the past.

`compute_run_id` serialised every field in RUN_ID_FIELDS, absent ones included,
as an explicit null. Appending "optimizer" the obvious way would therefore have
changed the hash of all 234 records already in the registry -- every completed
fold would stop resuming, and the next run would silently propose repeating the
entire campaign.

The fix is to hash the OVERRIDE rather than the value. `optimizer=None` means
"whatever configs/arms.yaml declares for this arm", which is what every record
so far was hashed with, and it is omitted from the payload entirely.

That distinction is not cosmetic. yolo26n, yolo26s and yolo26m declare
`optimizer: MuSGD` and always have, so a scheme where "absent means SGD" would
have changed the identity of all 45 YOLO records in script 03 -- the very
failure this file exists to prevent, reintroduced by the fix for it. The arm's
own choice is already pinned by `arm` plus the resolved_arms.yaml snapshot.

Both halves are pinned here: the old hashes survive, and a deliberate override
still gets a distinct identity.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from pathlib import Path

from srpcard import registry

REPO_ROOT = Path(__file__).resolve().parents[1]


# The payload exactly as compute_run_id built it BEFORE optimizer existed:
# every field of the old RUN_ID_FIELDS, nothing else.
LEGACY_FIELDS = (
    "script", "arm", "architecture", "split_kind", "repeat", "fold",
    "epochs", "batch", "lr", "class_weights", "run_seed", "extra",
)


def legacy_run_id(**params):
    payload = {key: params.get(key) for key in LEGACY_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def a_run(**overrides):
    run = {
        "script": "03_run_cv",
        "arm": "yolo26n",
        "architecture": "yolo26n-cls",
        "split_kind": "cv",
        "repeat": 0,
        "fold": 0,
        "epochs": 25,
        "batch": 16,
        "lr": 0.001,
        "class_weights": "balanced",
        "run_seed": 10000,
        "extra": {"protocol": "uniform"},
    }
    run.update(overrides)
    return run


# ------------------------------------------------- the past must not move


def test_an_existing_run_id_is_unchanged():
    """The whole point. 234 records depend on this."""
    run = a_run()
    assert registry.compute_run_id(**run) == legacy_run_id(**run)


def test_no_override_means_the_hash_does_not_move():
    """None means "whatever the arm declares", which is what every record so far
    was hashed with."""
    run = a_run()
    assert registry.compute_run_id(**run, optimizer=None) == legacy_run_id(**run)


def test_the_musgd_arms_keep_their_identity():
    """THE regression this design exists for.

    yolo26n, yolo26s and yolo26m declare `optimizer: MuSGD` in
    configs/arms.yaml and always have. Treating the DECLARED optimizer as part
    of the identity -- rather than only an override of it -- would have changed
    the hash of all 45 YOLO records in script 03 and quietly proposed re-running
    them. The arm's own choice is already pinned by `arm`.
    """
    for arm, architecture in (("yolo26n", "yolo26n-cls"),
                              ("yolo26s", "yolo26s-cls"),
                              ("yolo26m", "yolo26m-cls")):
        run = a_run(arm=arm, architecture=architecture)
        assert registry.compute_run_id(**run, optimizer=None) == legacy_run_id(**run)


def test_an_override_back_to_sgd_is_a_different_run():
    """For the MuSGD arms the contrast run is --optimizer SGD, not the reverse."""
    run = a_run()
    assert registry.compute_run_id(**run, optimizer="SGD") != legacy_run_id(**run)
    assert (registry.compute_run_id(**run, optimizer="SGD")
            == registry.compute_run_id(**run, optimizer="sgd"))


@pytest.mark.parametrize("arm,fold,epochs", [
    ("mobilenetv3_small", 3, 50), ("resnet18", 4, 50), ("yolo26m", 0, 50),
])
def test_the_hash_is_stable_across_the_arms_actually_in_the_registry(arm, fold, epochs):
    run = a_run(arm=arm, fold=fold, epochs=epochs)
    assert registry.compute_run_id(**run) == legacy_run_id(**run)


# ------------------------------------------- a different run must be different


def test_an_override_gets_its_own_identity():
    """Without this the contrast run resumes as its twin and never happens."""
    run = a_run()
    assert (registry.compute_run_id(**run, optimizer="musgd")
            != registry.compute_run_id(**run, optimizer=None))
    assert (registry.compute_run_id(**run, optimizer="musgd")
            != registry.compute_run_id(**run, optimizer="sgd"))


def test_the_optimizer_name_is_case_insensitive():
    run = a_run()
    assert (registry.compute_run_id(**run, optimizer="MuSGD")
            == registry.compute_run_id(**run, optimizer="musgd"))


def test_the_epoch_budget_still_separates_runs():
    """C1 adds yolo26n at 50 epochs beside yolo26n at 25."""
    assert (registry.compute_run_id(**a_run(epochs=25))
            != registry.compute_run_id(**a_run(epochs=50)))


def test_the_protocol_separates_the_native_recipe_runs():
    uniform = a_run(extra={"protocol": "uniform"})
    native = a_run(script="03c_native_recipe", extra={"protocol": "native"})
    assert registry.compute_run_id(**uniform) != registry.compute_run_id(**native)


# ------------------------------------------------------------- the payload


def test_the_payload_omits_an_absent_override_rather_than_nulling_it():
    payload = registry.run_id_payload(**a_run(), optimizer=None)
    assert "optimizer" not in payload, "a null here would move every old hash"
    assert set(payload) == set(LEGACY_FIELDS)


def test_the_payload_keeps_an_sgd_override():
    payload = registry.run_id_payload(**a_run(), optimizer="SGD")
    assert payload["optimizer"] == "sgd"


def test_the_payload_keeps_a_non_default_optimizer():
    payload = registry.run_id_payload(**a_run(), optimizer="musgd")
    assert payload["optimizer"] == "musgd"


def test_explain_shows_the_fields_and_the_hash():
    text = registry.explain_run_id(**a_run(), optimizer="musgd")
    assert registry.compute_run_id(**a_run(), optimizer="musgd") in text
    assert "optimizer" in text and "musgd" in text
    assert "yolo26n-cls" in text


def test_explain_names_what_it_omitted_and_why():
    text = registry.explain_run_id(**a_run(), optimizer=None)
    assert "identity-neutral" in text
    assert "optimizer" in text
    assert "already pinned by" in text


# ------------------------------------------------------- extra.protocol


def test_a_record_without_extra_protocol_is_incomplete():
    """Promoted to required: the drift guard scopes on it, and a record without
    one is pooled with a regime it was never part of."""
    record = {field: None for field in registry.REQUIRED_RECORD_FIELDS}
    record["extra"] = {"selection_metric": "val_f1_macro"}
    assert "extra.protocol" in registry.missing_fields(record)


def test_a_record_with_extra_protocol_is_complete():
    record = {field: None for field in registry.REQUIRED_RECORD_FIELDS}
    record["extra"] = {"protocol": "uniform"}
    assert registry.missing_fields(record) == []


def test_a_non_dict_extra_is_reported_not_crashed_on():
    record = {field: None for field in registry.REQUIRED_RECORD_FIELDS}
    record["extra"] = "uniform"
    assert "extra.protocol" in registry.missing_fields(record)


def test_append_refuses_a_record_with_no_protocol(registry_path):
    record = {field: None for field in registry.REQUIRED_RECORD_FIELDS}
    record["extra"] = {}
    with pytest.raises(ValueError) as caught:
        registry.append_record(record, registry_path)
    assert "extra.protocol" in str(caught.value)


# ------------------------------------------------- the record verifies itself


def test_build_record_requires_what_the_hash_saw():
    """No default. A field that can be forgotten will be, and a record that
    cannot verify its own identity is not reproducible in any sense that
    matters. See docs/RUN_ID.md."""
    import inspect

    signature = inspect.signature(registry.build_record)
    parameter = signature.parameters["run_id_extra"]
    assert parameter.default is inspect.Parameter.empty


def test_the_record_carries_the_hashed_marker(registry_path):
    record = registry.build_record(
        run_id="abc", script="05_learning_curve", arm="mobilenetv3_small",
        architecture="mobilenet_v3_small", split_kind="cv", repeat=0, fold=0,
        epochs=50, batch=16, lr=0.01, class_weights="balanced", run_seed=1,
        val_seed=2, checkpoint_resolved="x", pretrained_fallback_used=False,
        class_weights_verified=True, class_weights_proof={}, corpus_fingerprint={},
        training={}, metrics={}, efficiency={}, wall_time_s=1.0,
        run_id_extra="lc_frac0.20", optimizer_used="SGD",
        extra={"protocol": "uniform", "fraction": 0.2},
    )
    assert record["extra"]["run_id_extra"] == "lc_frac0.20"
    assert record["extra"]["protocol"] == "uniform"      # not clobbered
    assert record["extra"]["run_id_optimizer"] is None


def test_an_override_is_recorded_beside_the_marker(registry_path):
    record = registry.build_record(
        run_id="abc", script="03_run_cv", arm="yolo26n", architecture="yolo26n-cls",
        split_kind="cv", repeat=0, fold=0, epochs=25, batch=16, lr=0.001,
        class_weights="balanced", run_seed=1, val_seed=2, checkpoint_resolved="x",
        pretrained_fallback_used=False, class_weights_verified=True,
        class_weights_proof={}, corpus_fingerprint={}, training={}, metrics={},
        efficiency={}, wall_time_s=1.0,
        run_id_extra=None, run_id_optimizer="SGD", optimizer_used="SGD",
        extra={"protocol": "uniform"},
    )
    assert record["extra"]["run_id_extra"] is None
    assert record["extra"]["run_id_optimizer"] == "SGD"


def test_a_record_written_now_can_verify_itself(registry_path):
    """The point of the whole exercise, end to end."""
    spec = {
        "script": "03_run_cv", "arm": "yolo26n", "architecture": "yolo26n-cls",
        "split_kind": "cv", "repeat": 0, "fold": 0, "epochs": 25, "batch": 16,
        "lr": 0.001, "class_weights": "balanced", "run_seed": 10000,
        "optimizer": None, "extra": None,
    }
    run_id = registry.compute_run_id(**spec)
    record = registry.build_record(
        run_id=run_id, val_seed=2, checkpoint_resolved="x",
        pretrained_fallback_used=False, class_weights_verified=True,
        class_weights_proof={}, corpus_fingerprint={}, training={}, metrics={},
        efficiency={}, wall_time_s=1.0,
        run_id_extra=spec["extra"], run_id_optimizer=spec["optimizer"],
        optimizer_used="MuSGD",
        extra={"protocol": "uniform"},
        **{k: v for k, v in spec.items() if k not in ("extra", "optimizer")},
    )

    # verification using nothing but the record
    recomputed = registry.compute_run_id(**{
        key: (record["extra"]["run_id_extra"] if key == "extra"
              else record["extra"]["run_id_optimizer"] if key == "optimizer"
              else record.get(key))
        for key in registry.RUN_ID_FIELDS
    })
    assert recomputed == record["run_id"]


def test_the_marker_table_is_documented_where_a_human_reads_it():
    """A test asserts the mapping; it does not tell the next person it exists."""
    doc = (REPO_ROOT / "docs" / "RUN_ID.md").read_text(encoding="utf-8")
    for marker in ("uniform_grid", "lr_sweep", "legacy_protocol", "lc_frac0.20"):
        assert marker in doc, "docs/RUN_ID.md does not list %r" % marker
    assert "MuSGD" in doc
    assert "not backfill" in doc.lower() or "not backfilled" in doc.lower()


# ------------------------------------------------- contrast runs stay separate


def _run_03(*args):
    """Invoke scripts/03_run_cv.py's argument handling without training."""
    import importlib.util
    import subprocess
    import sys as _sys

    return subprocess.run(
        [_sys.executable, "scripts/03_run_cv.py", *args],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_a_contrast_run_needs_an_explicit_arm_set():
    """A contrast is a question about one or two arms. Running it across the
    whole set would spend hours producing records nothing is waiting for."""
    result = _run_03("--contrast", "--dry-run")
    assert result.returncode == 2
    assert "--contrast requires --arms" in result.stdout


def test_an_optimizer_override_cannot_be_recorded_as_the_published_arm():
    """A run with a swapped optimizer is NOT the published arm, so it must not
    land under 03_run_cv where every summary table would pick it up."""
    result = _run_03("--optimizer", "sgd", "--arms", "yolo26n", "--dry-run")
    assert result.returncode == 2
    assert "--optimizer requires --contrast" in result.stdout


def test_a_contrast_run_is_recorded_under_its_own_script_name():
    """That name is the entire mechanism keeping contrast arms out of the
    published five-arm comparison: aggregate.cv_records() selects on
    script == '03_run_cv'."""
    source = (REPO_ROOT / "scripts" / "03_run_cv.py").read_text(encoding="utf-8")
    assert 'CONTRAST_SCRIPT = "03b_contrast"' in source
    assert "script = CONTRAST_SCRIPT if args.contrast else SCRIPT" in source
    assert '"script": script,' in source


def test_the_contrast_script_is_not_what_aggregate_reads():
    from srpcard import aggregate

    assert aggregate.CV_SCRIPT == "03_run_cv"
    assert aggregate.CV_SCRIPT != "03b_contrast"
