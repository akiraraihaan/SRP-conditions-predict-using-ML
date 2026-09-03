"""The registry's own guards: schema completeness and hyperparameter drift.

Every test writes to a tmp registry, never artifacts/registry.jsonl.
"""

from __future__ import annotations

import json

import pytest

from srpcard import aggregate, registry


def make_record(**overrides):
    """A schema-complete record, with only what a test cares about overridden."""
    record = registry.build_record(
        run_id="r0",
        script="03_run_cv",
        arm="yolo26m",
        architecture="yolo26m-cls",
        split_kind="cv",
        repeat=0,
        fold=0,
        epochs=50,
        batch=16,
        lr=0.001,
        class_weights="balanced",
        run_seed=10000,
        val_seed=60000,
        checkpoint_resolved="yolo26m-cls.pt",
        pretrained_fallback_used=False,
        class_weights_verified=True,
        class_weights_proof={"passed": True},
        corpus_fingerprint={"kind": "cv_clean_668", "n": 668},
        training=registry.training_outcome_absent("synthetic"),
        metrics={"f1_macro": 0.8, "accuracy": 0.8, "confusion_matrix": [], "class_order": []},
        efficiency={"params": 1, "gflops": 1.0, "size_mb": 1.0},
        wall_time_s=1.0,
    )
    record.update(overrides)
    return record


# ---------------------------------------------------------------- schema


def test_build_record_is_schema_complete():
    assert registry.missing_fields(make_record()) == []


def test_append_refuses_an_incomplete_record(registry_path):
    with pytest.raises(ValueError) as exc:
        registry.append_record({"run_id": "x", "script": "03_run_cv"}, registry_path)
    assert "checkpoint_resolved" in str(exc.value)
    assert registry_path.read_text(encoding="utf-8") == ""


def test_append_accepts_a_complete_record(registry_path):
    registry.append_record(make_record(), registry_path)
    assert len(registry.load_registry(registry_path)) == 1


def test_stale_record_is_reported(registry_path, capsys):
    stale = make_record()
    for field in ("checkpoint_resolved", "class_weights_verified", "corpus_fingerprint"):
        stale.pop(field)
    registry_path.write_text(json.dumps(stale) + "\n", encoding="utf-8")

    assert registry.warn_if_stale(registry_path) is False
    out = capsys.readouterr().out
    assert "REGISTRY SCHEMA DRIFT" in out
    assert "checkpoint_resolved" in out

    audit = registry.audit_registry(registry_path)
    assert len(audit["incomplete"]) == 1
    assert set(audit["incomplete"][0]["missing"]) >= {
        "checkpoint_resolved",
        "class_weights_verified",
        "corpus_fingerprint",
    }


def test_clean_registry_is_not_reported(registry_path):
    registry.append_record(make_record(), registry_path)
    assert registry.warn_if_stale(registry_path) is True


# ---------------------------------------------------------------- drift


def _seed_folds(path, *, epochs, batch, lr, n=3, prefix="a"):
    for fold in range(n):
        registry.append_record(
            make_record(
                run_id="%s%02d" % (prefix, fold),
                fold=fold,
                epochs=epochs,
                batch=batch,
                lr=lr,
            ),
            path,
        )


def test_matching_config_passes(registry_path):
    _seed_folds(registry_path, epochs=50, batch=16, lr=0.001)
    registry.assert_config_matches_registry(
        script="03_run_cv", arm="yolo26m", epochs=50, batch=16, lr=0.001,
        path=registry_path,
    )


def test_empty_registry_passes(registry_path):
    registry.assert_config_matches_registry(
        script="03_run_cv", arm="yolo26m", epochs=25, batch=8, lr=0.01,
        path=registry_path,
    )


def test_reverted_config_aborts(registry_path):
    """The real scenario: arms.yaml reverts to the provisional yolo26m config."""
    _seed_folds(registry_path, epochs=50, batch=16, lr=0.001)
    with pytest.raises(registry.HyperparameterDriftError) as exc:
        registry.assert_config_matches_registry(
            script="03_run_cv", arm="yolo26m",
            epochs=25, batch=8, lr=0.01,          # the provisional values
            path=registry_path,
        )
    message = str(exc.value)
    assert "HYPERPARAMETER DRIFT" in message
    assert "restore_arms" in message
    assert "a00" in message and "a01" in message and "a02" in message


def test_drift_is_scoped_to_the_arm_and_split(registry_path):
    _seed_folds(registry_path, epochs=50, batch=16, lr=0.001)
    # a different arm's records must not trip this arm's check
    registry.assert_config_matches_registry(
        script="03_run_cv", arm="yolo26n", epochs=50, batch=16, lr=0.01,
        path=registry_path,
    )
    # nor must a different split_kind
    registry.assert_config_matches_registry(
        script="01", arm="yolo26m", epochs=25, batch=8, lr=0.01,
        split_kind="dev", path=registry_path,
    )


def test_unresolved_arm_is_skipped(registry_path):
    _seed_folds(registry_path, epochs=50, batch=16, lr=0.001)
    arms_cfg = {"arms": {"resnet18": {"epochs": 50, "batch": 16, "lr": None}}}
    registry.assert_arms_match_registry(
        script="03_run_cv", arms=["resnet18"], arms_cfg=arms_cfg, path=registry_path,
    )


# ---------------------------------------------------------------- aggregate


def test_aggregate_refuses_mixed_regimes():
    records = [
        make_record(run_id="a%d" % i, fold=i, epochs=50, batch=16, lr=0.001)
        for i in range(3)
    ] + [
        make_record(run_id="b%d" % i, fold=i, epochs=25, batch=8, lr=0.01)
        for i in range(2)
    ]
    with pytest.raises(aggregate.MixedHyperparametersError) as exc:
        aggregate.assert_hyperparameters_unanimous(records)
    message = str(exc.value)
    assert "MIXED HYPERPARAMETERS" in message
    assert "a0" in message and "b0" in message


def test_aggregate_accepts_a_unanimous_arm():
    records = [make_record(run_id="a%d" % i, fold=i) for i in range(3)]
    aggregate.assert_hyperparameters_unanimous(records)


def test_summarise_cv_refuses_mixed_regimes():
    records = [
        make_record(run_id="a", fold=0, epochs=50, batch=16, lr=0.001),
        make_record(run_id="b", fold=1, epochs=25, batch=8, lr=0.01),
    ]
    with pytest.raises(aggregate.MixedHyperparametersError):
        aggregate.summarise_cv(records)
