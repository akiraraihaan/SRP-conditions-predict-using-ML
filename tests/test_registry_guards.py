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
        extra={"protocol": "uniform"},
    )
    record.update(overrides)
    return record


def legacy_record(**overrides):
    """A script 01 record: same arm, same dev split, DIFFERENT protocol."""
    return make_record(
        script="01_complete_medium_grid",
        split_kind="dev",
        class_weights="none_legacy_bug",
        extra={"protocol": "legacy_unweighted_ultralytics", "key": "m_ep25_bs8_lr1e-02"},
        **overrides,
    )


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


def _seed_folds(path, *, epochs, batch, lr, n=3, prefix="a", **overrides):
    for fold in range(n):
        registry.append_record(
            make_record(
                run_id="%s%02d" % (prefix, fold),
                fold=fold,
                epochs=epochs,
                batch=batch,
                lr=lr,
                **overrides,
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


# ---------------------------------------------------------------- scoping
#
# The false positive this scoping fixes: script 01 (legacy augmented protocol)
# and script 01b (uniform protocol) both write split_kind "dev" for yolo26m, so
# a guard filtering on arm and split_kind alone pooled them -- and then proposed
# deleting the nine script 01 records that evidence the augmentation finding.


def test_a_different_protocol_is_not_evidence(registry_path):
    """A legacy record and a uniform record for one arm must not collide."""
    registry.append_record(
        legacy_record(run_id="legacy0", epochs=25, batch=8, lr=0.01), registry_path
    )
    registry.append_record(
        make_record(run_id="uniform0", split_kind="dev", epochs=50, batch=16, lr=0.001),
        registry_path,
    )
    # asking about the uniform protocol must ignore the legacy record entirely
    registry.assert_config_matches_registry(
        script="03_run_cv", arm="yolo26m", epochs=50, batch=16, lr=0.001,
        split_kind="dev", scripts={"03_run_cv"}, path=registry_path,
    )


def test_legacy_records_do_not_trip_the_locked_config_guard(registry_path):
    """The exact false positive: 8 legacy yolo26m runs, then 03 starts."""
    for i, (epochs, batch, lr) in enumerate(
        [(50, 8, 0.001), (50, 8, 0.01), (50, 16, 0.0001), (50, 16, 0.001),
         (50, 16, 0.01), (50, 32, 0.0001), (50, 32, 0.001), (50, 32, 0.01)]
    ):
        registry.append_record(
            legacy_record(run_id="legacy%d" % i, epochs=epochs, batch=batch, lr=lr),
            registry_path,
        )
    arms_cfg = {"arms": {"yolo26m": {"epochs": 25, "batch": 8, "lr": 0.01}}}
    registry.assert_arms_match_registry(
        script="03_run_cv", arms=["yolo26m"], arms_cfg=arms_cfg,
        split_kind="dev", path=registry_path,
    )


def test_a_different_script_is_not_evidence(registry_path):
    _seed_folds(registry_path, epochs=50, batch=16, lr=0.001,
                script="02_lr_sweep_baselines", split_kind="dev")
    registry.assert_config_matches_registry(
        script="03_run_cv", arm="yolo26m", epochs=25, batch=8, lr=0.01,
        split_kind="dev", scripts={"03_run_cv"}, path=registry_path,
    )


def test_same_script_and_protocol_still_fires(registry_path):
    """Narrowing the scope must not disarm the guard where it belongs."""
    _seed_folds(registry_path, epochs=50, batch=16, lr=0.001)
    with pytest.raises(registry.HyperparameterDriftError) as exc:
        registry.assert_config_matches_registry(
            script="03_run_cv", arm="yolo26m", epochs=25, batch=8, lr=0.01,
            path=registry_path,
        )
    message = str(exc.value)
    assert "come from THIS script under THIS protocol" in message
    assert "restore_arms.py" in message


def test_message_states_what_it_compared(registry_path):
    _seed_folds(registry_path, epochs=50, batch=16, lr=0.001)
    with pytest.raises(registry.HyperparameterDriftError) as exc:
        registry.assert_config_matches_registry(
            script="03_run_cv", arm="yolo26m", epochs=25, batch=8, lr=0.01,
            path=registry_path,
        )
    message = str(exc.value)
    for field in ("arm", "split_kind", "protocol", "scripts"):
        assert field in message
    assert "03_run_cv" in message


def test_cross_protocol_conflict_never_proposes_deletion(registry_path):
    """If a caller mis-scopes anyway, the message must not say 'delete them'."""
    registry.append_record(
        legacy_record(run_id="legacy0", epochs=50, batch=8, lr=0.001), registry_path
    )
    # force the mis-scoped path: the defaults would correctly exclude this record,
    # so widen the scope deliberately to reach the branch under test
    with pytest.raises(registry.HyperparameterDriftError) as exc:
        registry.assert_config_matches_registry(
            script="03_run_cv", arm="yolo26m", epochs=25, batch=8, lr=0.01,
            split_kind="dev", protocol=None,
            scripts={"01_complete_medium_grid", "03_run_cv"},
            path=registry_path,
        )
    message = str(exc.value)
    assert "DO NOT delete those records" in message
    assert "SCOPING error" in message
    assert "01_complete_medium_grid" in message


# ---------------------------------------------------------------- sweeps


def test_sweep_scripts_are_exempt_from_the_locked_config_guard():
    assert "01_complete_medium_grid" in registry.SWEEP_SCRIPTS
    assert "01b_uniform_grid" in registry.SWEEP_SCRIPTS
    assert "02_lr_sweep_baselines" in registry.SWEEP_SCRIPTS
    assert not (registry.SWEEP_SCRIPTS & registry.LOCKED_CONFIG_SCRIPTS)


def test_calling_the_locked_guard_from_a_sweep_is_an_error(registry_path):
    for script in sorted(registry.SWEEP_SCRIPTS):
        with pytest.raises(ValueError, match="sweeps epochs/batch/lr"):
            registry.assert_config_matches_registry(
                script=script, arm="yolo26m", epochs=25, batch=8, lr=0.01,
                path=registry_path,
            )


GRID = {(e, b, lr) for e in (25, 50) for b in (8, 16, 32)
        for lr in (0.0001, 0.001, 0.01)}


def test_sweep_membership_passes_for_a_declared_grid(registry_path):
    registry.assert_sweep_within_grid(
        script="01b_uniform_grid", protocol="uniform", arms=["yolo26m"],
        grid_points=GRID, planned=sorted(GRID), path=registry_path,
    )


def test_sweep_membership_ignores_another_script(registry_path):
    """Script 01's legacy runs must not constrain 01b's grid check."""
    registry.append_record(
        legacy_record(run_id="legacy0", epochs=99, batch=99, lr=0.5), registry_path
    )
    registry.assert_sweep_within_grid(
        script="01b_uniform_grid", protocol="uniform", arms=["yolo26m"],
        grid_points=GRID, planned=sorted(GRID), path=registry_path,
    )


def test_sweep_membership_rejects_a_planned_point_outside_the_grid(registry_path):
    with pytest.raises(registry.HyperparameterDriftError, match="NOT points of the"):
        registry.assert_sweep_within_grid(
            script="01b_uniform_grid", protocol="uniform", arms=["yolo26m"],
            grid_points=GRID, planned=[(25, 8, 0.01), (75, 8, 0.01)],
            path=registry_path,
        )


def test_sweep_membership_detects_a_grid_edited_between_sessions(registry_path):
    """The real error a sweep can suffer: completed runs from an older grid."""
    registry.append_record(
        make_record(run_id="old0", script="01b_uniform_grid", split_kind="dev",
                    epochs=75, batch=8, lr=0.01),
        registry_path,
    )
    with pytest.raises(registry.HyperparameterDriftError) as exc:
        registry.assert_sweep_within_grid(
            script="01b_uniform_grid", protocol="uniform", arms=["yolo26m"],
            grid_points=GRID, planned=sorted(GRID), path=registry_path,
        )
    message = str(exc.value)
    assert "GRID CHANGED between sessions" in message
    assert "old0" in message


def test_default_scoping_excludes_a_legacy_record_entirely(registry_path):
    """The companion to the test above: with the defaults, the legacy record is
    not merely reported differently -- it is never compared."""
    registry.append_record(
        legacy_record(run_id="legacy0", epochs=50, batch=8, lr=0.001), registry_path
    )
    registry.assert_config_matches_registry(
        script="03_run_cv", arm="yolo26m", epochs=25, batch=8, lr=0.01,
        split_kind="dev", path=registry_path,
    )
    groups = registry.completed_hyperparameters(
        "yolo26m", "dev", registry_path,
        scripts=registry.LOCKED_CONFIG_SCRIPTS, protocol="uniform",
    )
    assert groups == {}
