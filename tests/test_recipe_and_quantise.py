"""Tasks C and D: the fairness checks and the quantisation table.

None of this needs a GPU, images or checkpoints. What is pinned is the logic
that decides what the manuscript may claim -- which records belong to which
cell of the 2x2, what the verdict says when a cell is missing, that the native
tree can never be written into the dataset, and that a quantisation row can
never be read as a latency measurement.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recipe():
    return _load("recipe_check", "11_recipe_check.py")


@pytest.fixture(scope="module")
def native():
    return _load("native_recipe", "03c_native_recipe.py")


@pytest.fixture(scope="module")
def quantise():
    return _load("quantise", "10_quantise.py")


def record(arm, script, fold, f1, override=None, repeat=0):
    return {
        "arm": arm, "script": script, "repeat": repeat, "fold": fold,
        "f1_macro": f1,
        "extra": {"protocol": "uniform", "run_id_optimizer": override},
    }


# ------------------------------------------------------ the 2x2 is gone
#
# The tests that lived here exercised a 2x2 crossing architecture against
# optimizer -- cell_records, _override_of, confound_verdict. That design was
# built on configs/arms.yaml declaring MuSGD for the YOLO arms, and MuSGD was
# SGD: ultralytics' MuSGD takes use_muon=False and was constructed from a flat
# parameter list, so the "optimizer" axis crossed one thing with itself.
#
# They are deleted rather than adapted. A test for a design that no longer
# exists passes for the wrong reason and keeps the wrong shape alive in the
# reader's head. What replaces them is at the end of this file.


# ------------------------------------------------- the native path's guards


def test_the_fold_tree_can_never_be_written_into_the_dataset(native, tmp_path):
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    inside = data_root / "scratch"
    with pytest.raises(SystemExit) as caught:
        native.assert_outside_data_root(inside, data_root)
    assert "inside --data-root" in str(caught.value)


def test_the_dataset_itself_is_refused_as_a_work_dir(native, tmp_path):
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    with pytest.raises(SystemExit):
        native.assert_outside_data_root(data_root, data_root)


def test_a_sibling_work_dir_is_allowed(native, tmp_path):
    data_root = tmp_path / "dataset"
    work = tmp_path / "scratch"
    data_root.mkdir()
    work.mkdir()
    native.assert_outside_data_root(work, data_root)     # must not raise


def test_class_names_are_mapped_through_the_models_own_list(native):
    """Never by assuming sort order matches. A guessed mapping would score
    every prediction against the wrong label and report it as a recipe
    difference."""
    class FakeResult:
        class probs:
            top1 = 0

    class FakeModel:
        names = {0: "vibration", 1: "gas_influence"}

        def predict(self, source, verbose=False):
            return [FakeResult()]

    classes = ["gas_influence", "vibration"]      # DIFFERENT order from names
    predicted = native.native_predictions(FakeModel(), [Path("a.png")], classes)
    assert predicted == [1], "index 0 in their order is 'vibration', ours index 1"


def test_an_unknown_class_name_is_refused(native):
    class FakeModel:
        names = {0: "something_else"}

    with pytest.raises(SystemExit) as caught:
        native.native_predictions(FakeModel(), [], ["vibration"])
    assert "class names we do not know" in str(caught.value)


def test_the_native_row_declares_its_own_preprocessing(native):
    assert native.PREPROCESSING == "ultralytics_default"
    assert native.UNIFORM_PREPROCESSING == "letterbox_224"
    assert native.PROTOCOL == "native"


# ---------------------------------------------------------------- task D


def test_quantisation_never_claims_a_latency(quantise):
    source = (REPO_ROOT / "scripts" / "10_quantise.py").read_text(encoding="utf-8")
    assert "NO LATENCY" in source
    assert "latency_measured" in source
    for forbidden in ("perf_counter", "time.time", "median_ms"):
        assert forbidden not in source, (
            "10_quantise must not time anything: found %r" % forbidden
        )


def test_the_benchmark_fold_is_read_from_config_not_chosen(quantise):
    assert quantise.benchmark_fold({"reporting": {"benchmark_fold":
                                                  {"repeat": 2, "fold": 3}}}) == (2, 3)
    assert quantise.benchmark_fold({}) == (0, 0)


def test_the_census_counts_leaf_layers_only(quantise):
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3),
        torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.ReLU()),
    )
    census = quantise.layer_census(model)
    assert census["Conv2d"] == 1 and census["Linear"] == 1
    assert "Sequential" not in census


def test_dynamic_ptq_leaves_convolutions_alone(quantise):
    """The honest finding from the edge benchmark, restated here so the two
    methods can be compared rather than conflated."""
    pytest.importorskip("torch")
    from torchvision.models import resnet18

    module = resnet18(weights=None).eval()
    quantised, reason = quantise.dynamic_ptq(module)
    assert quantised is not None, reason

    block = quantise.coverage(module, quantised)
    assert block["quantised_layer_types"] == {"Linear": 1}
    assert block["unquantised_layer_types"]["Conv2d"] == 20
    assert block["params_quantised_pct"] < 5.0


def test_a_static_failure_is_reported_with_its_reason(quantise, monkeypatch):
    """'Static PTQ is not available for this architecture' is itself a finding.
    A silent skip would read as a method nobody tried."""
    torch = pytest.importorskip("torch")

    def explode(*args, **kwargs):
        raise RuntimeError("Could not run 'quantized::conv2d' with this backend")

    monkeypatch.setattr(torch.ao.quantization, "prepare", explode)
    module = torch.nn.Linear(4, 2).eval()

    result, reason = quantise.static_ptq(module, None, [], {})

    assert result is None
    assert "RuntimeError" in reason
    assert "quantized::conv2d" in reason, "the reason must name what failed"


def test_any_engine_the_build_offers_is_accepted(quantise):
    """torch 2.12.0+cpu reports supported_engines == ['onednn']. Hardcoding
    fbgemm/qnnpack would report 'static PTQ unavailable' on a machine that
    supports it -- a false negative in the one table the microcontroller
    argument rests on."""
    import inspect

    source = inspect.getsource(quantise.static_ptq)
    assert "onednn" in source and "x86" in source
    assert "available[0] if available else None" in source


def test_a_missing_backend_is_reported_rather_than_crashed_on(quantise):
    """Read straight off torch rather than patched: supported_engines is a
    read-only property, so this documents the branch instead of forcing it."""
    import inspect

    source = inspect.getsource(quantise.static_ptq)
    assert "no quantized backend available" in source
    assert "supported_engines" in source


def test_the_flash_budget_is_declared_once(quantise):
    assert quantise.FLASH_BUDGET_MB == 1.0


# --------------------------------- a contrast arm never joins the comparison


def test_a_contrast_only_arm_is_not_in_the_default_set():
    """Every script that takes --arms falls back to 'all arms in the config'.
    Without this flag, adding yolo26n_ep50 to arms.yaml would turn the five-arm
    headline comparison into a six-arm one and grow paired_comparisons.csv from
    10 pairs to 15 -- as a side effect of asking a question about it."""
    from srpcard.config import contrast_arms, load_arms_config, published_arms

    arms_cfg = load_arms_config()
    published = published_arms(arms_cfg)

    assert set(published) == {"yolo26n", "yolo26s", "yolo26m",
                              "mobilenetv3_small", "resnet18"}
    assert "yolo26n_ep50" in contrast_arms(arms_cfg)
    assert "yolo26n_ep50" in arms_cfg["arms"], "it must still be runnable by name"


def test_the_contrast_arm_differs_from_its_parent_only_in_the_budget():
    """It is not a sixth model. Anything else differing would make the epoch
    budget uninterpretable as the cause."""
    from srpcard.config import load_arms_config

    arms = load_arms_config()["arms"]
    parent, child = arms["yolo26n"], arms["yolo26n_ep50"]
    differing = {
        key for key in set(parent) | set(child)
        if parent.get(key) != child.get(key)
    }
    assert differing == {"epochs", "lr_source", "contrast_only"}
    assert child["epochs"] == 50 and parent["epochs"] == 25


def test_the_published_scripts_use_the_published_set():
    for name in ("03_run_cv.py", "07_bench_edge.py", "10_quantise.py"):
        source = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "published_arms(arms_cfg)" in source, (
            "%s falls back to every arm in the config, so a contrast arm would "
            "join the comparison by default" % name
        )


# ============================================================ the optimizer lie


def test_no_arm_declares_musgd_any_more():
    """arms.yaml said MuSGD on three arms and the code never applied Muon.
    ultralytics' MuSGD takes use_muon=False by default and train.py builds it
    from a flat parameter list, so it is bitwise SGD. The config now says what
    runs; a config that names something the code does not do is a trap."""
    from srpcard.config import load_arms_config

    for name, arm in load_arms_config()["arms"].items():
        assert arm.get("optimizer") != "MuSGD", (
            "%s declares MuSGD, which this project never actually applies" % name
        )


def test_the_reason_is_recorded_in_the_config():
    text = (REPO_ROOT / "configs" / "arms.yaml").read_text(encoding="utf-8")
    assert "use_muon" in text
    assert "bitwise identical" in text or "bitwise" in text


def test_musgd_and_sgd_are_the_same_object_in_practice():
    """The measurement behind the claim, so it is checked rather than asserted.
    If ultralytics ever changes the default, this fails and the config note
    becomes wrong -- which is exactly when someone needs to know."""
    torch = pytest.importorskip("torch")
    from ultralytics.optim.muon import MuSGD

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.Flatten(),
                                torch.nn.LazyLinear(2))
    model(torch.rand(1, 3, 16, 16))          # materialise the lazy layer
    opt = MuSGD(list(model.parameters()), lr=0.01, momentum=0.9,
                weight_decay=1e-4, nesterov=True)
    assert opt.param_groups[0].get("use_muon") is False, (
        "MuSGD now enables Muon by default; configs/arms.yaml and HANDOVER "
        "describe the old behaviour and must be revisited"
    )


def test_the_effective_optimizer_is_read_back_from_the_object():
    """Not from the config. The config was wrong for weeks and nothing caught
    it, because no record could say what actually trained it."""
    torch = pytest.importorskip("torch")
    from srpcard.train import TrainConfig, build_optimizer_checked

    model = torch.nn.Linear(4, 2)
    cfg = TrainConfig(epochs=1, batch=2, lr=0.01, optimizer="musgd")
    _, fingerprint = build_optimizer_checked(model, cfg)

    assert fingerprint["class"] == "MuSGD"
    assert fingerprint["degenerate_to_sgd"] is True
    assert fingerprint["effective"] == "SGD (MuSGD with use_muon=False)"


def test_a_plain_sgd_is_not_flagged_as_degenerate():
    torch = pytest.importorskip("torch")
    from srpcard.train import TrainConfig, build_optimizer_checked

    cfg = TrainConfig(epochs=1, batch=2, lr=0.01, optimizer="SGD")
    _, fingerprint = build_optimizer_checked(torch.nn.Linear(4, 2), cfg)
    assert fingerprint["effective"] == "SGD"
    assert fingerprint["degenerate_to_sgd"] is False


def test_the_record_cannot_be_written_without_the_effective_optimizer():
    import inspect

    from srpcard import registry

    parameter = inspect.signature(registry.build_record).parameters["optimizer_used"]
    assert parameter.default is inspect.Parameter.empty


# ======================================================= the CUDA library stack


def test_the_cuda_stack_is_recorded_under_honest_names():
    """torch has no public cuBLAS version API, so the cuBLAS version comes from
    the installed nvidia-cublas-* distribution. Calling something else 'cublas'
    would repeat the mistake just removed from arms.yaml."""
    from srpcard.config import library_versions

    versions = library_versions()
    assert "cudnn" in versions
    assert "cudnn_enabled" in versions
    # whatever is reported must not be mislabelled
    assert "cublas" not in versions or versions["cublas"].startswith(
        tuple("0123456789")
    )

    source = (REPO_ROOT / "src" / "srpcard" / "config.py").read_text(encoding="utf-8")
    assert "nvidia-" in source
    assert "EVERY NAME HERE MEANS WHAT IT SAYS" in source


def test_quantisation_does_not_baseline_against_the_registry():
    """Three arms stopped reproducing their records after a torch reinstall
    moved cuDNN. The accuracy column is internally consistent; comparing it
    against the registry would report an environment change as a quantisation
    effect."""
    source = (REPO_ROOT / "scripts" / "10_quantise.py").read_text(encoding="utf-8")
    assert "INTERNALLY CONSISTENT, NOT A REPRODUCTION" in source
    assert "macro_f1_fp32_recorded_in_registry" in source
    assert "CONTEXT ONLY" in source or "Context only" in source
    assert "ENVIRONMENT difference, not a" in source


# ======================================================= 11, rebuilt


def test_there_is_no_optimizer_axis_any_more(recipe):
    """The 2x2 is gone. MuSGD was SGD, so crossing architecture against
    optimizer crossed one thing with itself."""
    assert not hasattr(recipe, "CELLS"), "the 2x2 must not come back"
    assert len(recipe.ROWS) == 3
    assert {r["optimizer"] for r in recipe.ROWS} == {"SGD", "theirs"}


def test_table_1_states_both_halves_or_neither(recipe):
    import pandas as pd

    full = pd.DataFrame([
        {"row": "yolo26n / uniform", "f1_macro_mean": 0.5233},
        {"row": "mobilenetv3_small / uniform", "f1_macro_mean": 0.5812},
        {"row": "yolo26n / native recipe", "f1_macro_mean": 0.5787},
    ])
    lines = " ".join(recipe.recipe_conclusion(full))
    assert "+0.0579" in lines, "the architecture gap under a common recipe"
    assert "+0.0554" in lines, "what the native recipe recovers"
    assert "STATE BOTH" in lines


def test_a_missing_row_refuses_to_conclude(recipe):
    import pandas as pd

    partial = pd.DataFrame([
        {"row": "yolo26n / uniform", "f1_macro_mean": 0.5233},
        {"row": "mobilenetv3_small / uniform", "f1_macro_mean": 0.5812},
        {"row": "yolo26n / native recipe", "f1_macro_mean": None},
    ])
    lines = " ".join(recipe.recipe_conclusion(partial))
    assert "INCOMPLETE" in lines
    assert "yolo26n / native recipe" in lines


def test_the_environment_finding_is_computed_not_written(recipe):
    """Hardcoding 0.147 would make the paragraph a claim rather than a result."""
    import pandas as pd

    frame = pd.DataFrame([
        {"kind": "summary", "arm": "mobilenetv3_small", "n_folds": 5,
         "delta": -0.0109, "max_abs_delta": 0.1470, "reproduces_exactly": False},
        {"kind": "summary", "arm": "yolo26n", "n_folds": 5,
         "delta": 0.0, "max_abs_delta": 0.0, "reproduces_exactly": True},
    ])
    lines = " ".join(recipe.environment_finding(frame))

    assert "0.147" in lines and "0.011" in lines
    assert "mobilenetv3_small" in lines and "yolo26n" in lines
    assert "ARCHITECTURE-DEPENDENT" in lines
    assert "1 of the 2" in lines


def test_causality_is_stated_as_plausible_not_proven(recipe):
    import pandas as pd

    frame = pd.DataFrame([
        {"kind": "summary", "arm": "a", "n_folds": 5, "delta": 0.01,
         "max_abs_delta": 0.1, "reproduces_exactly": False},
    ])
    lines = " ".join(recipe.environment_finding(frame))
    assert "CUDA library stack" in lines
    assert "PLAUSIBLE mechanism, not a proven one" in lines
    assert "cuDNN caused" not in lines


def test_the_emit_deltas_are_marked_as_reported_when_no_sidecar(recipe):
    frame = recipe.environment_replication([], 0, None)
    support = frame[frame["kind"] == "emit_weights_reproduction"]
    assert len(support) == 3
    assert all("REPORTED" in s for s in support["source"])


def test_a_sidecar_is_preferred_over_the_reported_value(recipe, tmp_path):
    import json

    (tmp_path / "resnet18.json").write_text(json.dumps({
        "measured": {"f1_macro": 0.60}, "recorded": {"f1_macro": 0.6068},
        "measured_minus_recorded": -0.0068,
    }), encoding="utf-8")

    frame = recipe.environment_replication([], 0, tmp_path)
    row = frame[(frame["kind"] == "emit_weights_reproduction")
                & (frame["arm"] == "resnet18")].iloc[0]
    assert "sidecar" in row["source"]
    assert row["delta"] == -0.0068


def test_the_native_recipe_is_read_back_not_quoted():
    source = (REPO_ROOT / "scripts" / "03c_native_recipe.py").read_text(encoding="utf-8")
    assert "read_back_from" in source
    assert "unreadable_keys" in source
    assert "not documentation" in source
    # the optimizer OBJECT, not the requested string
    assert 'getattr(trainer, "optimizer", None)' in source


def test_the_epoch_budget_table_survived_the_rebuild(recipe):
    assert hasattr(recipe, "epoch_budget"), (
        "C1's table is orthogonal to the optimizer collapse and is still owed"
    )
