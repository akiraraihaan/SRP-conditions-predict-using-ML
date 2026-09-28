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

    import torch.ao.quantization.quantize_fx as fx

    monkeypatch.setattr(fx, "prepare_fx", explode)
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
    assert "between-session difference, not a" in source
    assert "UNIDENTIFIED" in source


# ======================================================= 11, rebuilt


def test_there_is_no_optimizer_axis_any_more(recipe):
    """The 2x2 is gone. MuSGD was SGD, so crossing architecture against
    optimizer crossed one thing with itself."""
    assert not hasattr(recipe, "CELLS"), "the 2x2 must not come back"
    # three reported rows plus the v1 native replicate, which is carried but
    # marked not reported
    assert len(recipe.ROWS) == 4
    assert sum(1 for r in recipe.ROWS if r["reported"]) == 3
    assert {r["optimizer"] for r in recipe.ROWS} == {"SGD", "theirs"}


def test_table_1_states_both_halves_or_neither(recipe):
    import pandas as pd

    full = pd.DataFrame([
        {"row": "yolo26n / uniform", "f1_macro_mean": 0.5233},
        {"row": "mobilenetv3_small / uniform", "f1_macro_mean": 0.5812},
        {"row": recipe.NATIVE_ROW, "f1_macro_mean": 0.5787},
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
        {"row": recipe.NATIVE_ROW, "f1_macro_mean": None},
    ])
    lines = " ".join(recipe.recipe_conclusion(partial))
    assert "INCOMPLETE" in lines
    assert recipe.NATIVE_ROW in lines


def test_reproduction_is_exact_within_a_session(recipe):
    """Two --emit-weights runs of mobilenetv3_small r0f0 in one session both
    gave 0.673445, epoch for epoch. That is the control that makes the
    between-session spread mean something."""
    assert "0.673445" in recipe.WITHIN_SESSION
    frame = recipe.environment_replication([], 0, None)
    assert set(frame["within_session"]) == {"exact"}


def test_the_between_session_spread_is_per_arm(recipe):
    frame = recipe.environment_replication([], 0, None)
    by_arm = dict(zip(frame["arm"], frame["f1_macro_spread"]))
    assert by_arm["mobilenetv3_small"] == 1.9e-2
    assert by_arm["yolo26s"] == 1.04e-2
    assert by_arm["resnet18"] == 6.8e-3
    assert by_arm["yolo26n"] == 0.0
    assert by_arm["yolo26m"] == 0.0

    precision = dict(zip(frame["arm"], frame["precision_macro_spread"]))
    assert precision["mobilenetv3_small"] == 7.3e-2


def test_two_arms_reproduce_exactly_and_are_reported_as_such(recipe):
    frame = recipe.environment_replication([], 0, None)
    exact = set(frame.loc[frame["reproduces_exactly"], "arm"])
    assert exact == {"yolo26n", "yolo26m"}

    lines = " ".join(recipe.environment_finding(frame))
    assert "reproduce EXACTLY across sessions" in lines
    assert "2 of the 5" in lines


def test_the_cause_is_stated_as_unidentified_and_no_mechanism_is_named(recipe):
    """Earlier drafts blamed the CUDA library stack. The same GPU model and the
    same torch version were in force, so that is not supported -- and a
    reinstall coinciding in time is not evidence."""
    frame = recipe.environment_replication([], 0, None)
    lines = " ".join(recipe.environment_finding(frame))

    assert "CAUSE IS UNIDENTIFIED" in lines
    assert "none has been tested" in lines
    assert set(frame["cause"]) == {"unidentified"}

    source = (REPO_ROOT / "scripts" / "11_recipe_check.py").read_text(encoding="utf-8")
    for forbidden in ("cuDNN", "cudnn", "cuBLAS", "cublas"):
        assert forbidden not in source, (
            "11 must name no mechanism: found %r" % forbidden
        )


def test_the_finding_says_what_it_means_for_the_manuscript(recipe):
    """A spread is only useful if it says what may be quoted."""
    frame = recipe.environment_replication([], 0, None)
    lines = " ".join(recipe.environment_finding(frame))
    # the sentence wraps, so match on parts that survive the line break
    assert "more precision" in lines and "not reproducible at that" in lines
    assert "15-fold mean" in lines


def test_a_sidecar_row_is_computed_when_one_is_available(recipe, tmp_path):
    import json

    (tmp_path / "resnet18.json").write_text(json.dumps({
        "arm": "resnet18",
        "measured": {"f1_macro": 0.60}, "recorded": {"f1_macro": 0.6068},
        "measured_minus_recorded": -0.0068,
    }), encoding="utf-8")

    frame = recipe.environment_replication([], 0, tmp_path)
    row = frame[frame["kind"] == "checkpoint_sidecar"].iloc[0]
    assert row["arm"] == "resnet18"
    assert row["delta"] == -0.0068
    assert row["cause"] == "unidentified"


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


# ================================================= static PTQ, end to end
#
# Static PTQ failed on all five arms with "too many values to unpack
# (expected 2)" -- raised after prepare(), so it was our calibration loop, not
# a platform limitation. FoldDataset yields (tensor, label, idx) and the loop
# destructured two. It was reported as "static PTQ unavailable", which is how a
# bug in our code spent a run disguised as a finding about torch.
#
# This is the method the microcontroller argument rests on. Dynamic PTQ reaches
# one Linear layer and cannot support it.


def _toy_model():
    """Two layers with something for each quantisation method to reach: a
    Conv2d that only static PTQ converts, and a Linear that both do."""
    import torch

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.relu = torch.nn.ReLU()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(self.pool(self.relu(self.conv(x))).flatten(1))

    return Toy()


class _ToyCache16:
    """224x224, matching the FX example input."""

    def get(self, idx):
        import numpy as np

        rng = np.random.default_rng(idx)
        return (rng.random((224, 224, 3)) * 255).astype("uint8")


class _ToyCache:
    """The ImageCache interface, over deterministic noise."""

    def get(self, idx):
        import numpy as np

        rng = np.random.default_rng(idx)
        return (rng.random((16, 16, 3)) * 255).astype("uint8")


def test_static_ptq_runs_end_to_end_and_the_result_is_callable(quantise):
    """prepare -> calibrate -> convert -> forward. A conversion that reports
    success and then raises at evaluation time is not a success."""
    torch = pytest.importorskip("torch")

    model = _toy_model().eval()
    converted, reason = quantise.static_ptq(model, _ToyCache(), list(range(24)),
                                            {i: i % 4 for i in range(24)})

    assert converted is not None, reason
    assert "backend=" in reason and "calibration batch" in reason

    out = converted(torch.rand(2, 3, 16, 16))
    assert tuple(out.shape) == (2, 4)


def test_static_ptq_quantises_convolutions(quantise):
    """The whole reason it exists: dynamic PTQ leaves every Conv2d in fp32."""
    pytest.importorskip("torch")

    model = _toy_model().eval()
    converted, reason = quantise.static_ptq(model, _ToyCache(), list(range(24)),
                                            {i: i % 4 for i in range(24)})
    assert converted is not None, reason

    block = quantise.coverage(model, getattr(converted, "inner", converted))
    assert "Conv2d" in block["quantised_layer_types"]
    assert block["params_quantised_pct"] > 90.0


def test_the_calibration_loop_indexes_the_batch(quantise):
    """FoldDataset yields three items. Destructuring two is what broke it, and
    a fourth element must not break it again."""
    import inspect

    source = inspect.getsource(quantise.calibrate)
    assert "batch[0]" in source
    assert "for images, _ in" not in source
    assert "for position, (images, _)" not in source


def test_a_failure_names_the_file_it_came_from(quantise, monkeypatch):
    """Our TypeError reported as a platform limitation is what hid this bug."""
    torch = pytest.importorskip("torch")

    def explode(*args, **kwargs):
        raise ValueError("too many values to unpack (expected 2)")

    # FX is tried first and would succeed, so both paths must fail for the
    # eager locator to be the thing under test.
    import torch.ao.quantization.quantize_fx as fx

    monkeypatch.setattr(fx, "prepare_fx", explode)
    monkeypatch.setattr(torch.ao.quantization, "convert", explode)
    result, reason = quantise.static_ptq(_toy_model().eval(), _ToyCache(),
                                         list(range(8)), {i: 0 for i in range(8)})
    assert result is None
    # It names where the exception ACTUALLY came from -- here, this test file,
    # because that is where the patched function raised. In the real bug it
    # named 10_quantise.py, which is the point: "ValueError in our calibration
    # loop" and "static PTQ unsupported on this platform" must not look alike.
    assert "raised at" in reason
    assert "test_recipe_and_quantise.py" in reason
    assert "ValueError" in reason


def test_the_stubs_are_what_make_it_runnable(quantise):
    import inspect

    source = inspect.getsource(quantise._wrap_for_static)
    assert "QuantStub" in source and "DeQuantStub" in source


# ================================================= the delta, checked


def test_a_prediction_flip_is_counted_and_named(quantise):
    import numpy as np

    classes = ["a", "b", "c", "d"]
    before = np.array([0, 1, 2, 3, 0])
    after = np.array([0, 2, 2, 3, 0])

    result = quantise.prediction_changes(before, after, classes)

    assert result["n_changed"] == 1
    assert result["n_predictions"] == 5
    assert result["pct_changed"] == 20.0
    assert result["transitions"] == {"b->c": 1}
    assert result["changed_image_positions"] == [1]


def test_identical_predictions_report_zero(quantise):
    import numpy as np

    before = np.array([0, 1, 2, 3])
    result = quantise.prediction_changes(before, before, ["a", "b", "c", "d"])
    assert result["n_changed"] == 0
    assert result["transitions"] == {}


def test_a_score_delta_with_no_flips_is_flagged(quantise):
    """Two models cannot differ in macro-F1 without differing in output. If
    that is reported, the quantised model is not the one being evaluated."""
    source = (REPO_ROOT / "scripts" / "10_quantise.py").read_text(encoding="utf-8")
    assert "with ZERO" in source
    assert "not the one" in source


def test_one_image_of_134_is_three_quarters_of_a_percent(quantise):
    """The arithmetic behind reading the flip count: a macro-F1 move of ~0.010
    on 134 images should be two or three flips, not zero and not thirty."""
    import numpy as np

    before = np.zeros(134, dtype=int)
    after = before.copy()
    after[0] = 1
    result = quantise.prediction_changes(before, after, ["a", "b"])
    assert result["pct_changed"] == 0.75


# ============================================ capture version in the identity
#
# The v1 native-recipe records describe their result and nothing about the
# recipe that produced it. Re-running to capture it must NOT overwrite them:
# the registry is append-only and that property is worth more than a tidy
# table. Putting the capture version in what 03c hashes costs one string and
# keeps both sets.


def test_the_capture_version_is_part_of_what_is_hashed(native):
    assert native.CAPTURE_VERSION == 2
    assert native.RUN_ID_EXTRA == "native_recipe_v2"

    source = (REPO_ROOT / "scripts" / "03c_native_recipe.py").read_text(encoding="utf-8")
    assert '"extra": RUN_ID_EXTRA,' in source, (
        "the marker must reach the spec that is hashed, not just the record"
    )


def test_v2_cannot_collide_with_v1(native):
    """Different marker, different run_id -- so the re-run appends."""
    from srpcard import registry

    base = {
        "arm": "yolo26n", "architecture": "yolo26n-cls", "script": native.SCRIPT,
        "split_kind": "cv", "repeat": 0, "fold": 0, "epochs": 25, "batch": 16,
        "lr": 0.001, "class_weights": "native_ultralytics_default",
        "run_seed": 10000, "optimizer": None,
    }
    v1 = registry.compute_run_id(**base, extra="native_recipe")
    v2 = registry.compute_run_id(**base, extra=native.RUN_ID_EXTRA)
    assert v1 != v2


def test_a_record_reports_which_capture_wrote_it(recipe):
    v1 = {"extra": {"run_id_extra": "native_recipe"}}
    v2 = {"extra": {"run_id_extra": "native_recipe_v2", "capture_version": 2}}
    assert recipe.capture_version_of(v1) == 1
    assert recipe.capture_version_of(v2) == 2


def test_only_v2_feeds_the_settings_table(recipe):
    """A v1 row in that table would show empty augmentation columns, which
    reads as a measured 'no augmentation applied' rather than as a gap."""
    records = [
        {"script": recipe.NATIVE, "arm": "yolo26n", "repeat": 0, "fold": 0,
         "extra": {"run_id_extra": "native_recipe", "protocol": "native"}},
        {"script": recipe.NATIVE, "arm": "yolo26n", "repeat": 0, "fold": 1,
         "extra": {"run_id_extra": "native_recipe_v2", "capture_version": 2,
                   "protocol": "native", "preprocessing": "ultralytics_default",
                   "native_recipe": {"augmentation": {"fliplr": 0.5},
                                     "schedule": {"lr0": 0.01},
                                     "optimizer_used": "SGD",
                                     "unreadable_keys": []}}},
    ]
    frame = recipe.native_recipe_rows(records)
    assert len(frame) == 1
    assert frame.iloc[0]["fold"] == 1
    assert frame.iloc[0]["aug_fliplr"] == 0.5


def test_the_two_captures_are_compared_per_fold(recipe):
    records = [
        {"script": recipe.NATIVE, "arm": "yolo26n", "repeat": 0, "fold": f,
         "run_id": "v1_%d" % f, "f1_macro": 0.60 + f / 1000,
         "extra": {"run_id_extra": "native_recipe"}}
        for f in range(3)
    ] + [
        {"script": recipe.NATIVE, "arm": "yolo26n", "repeat": 0, "fold": f,
         "run_id": "v2_%d" % f, "f1_macro": 0.60 + f / 1000,
         "extra": {"run_id_extra": "native_recipe_v2", "capture_version": 2,
                   "native_recipe": {}}}
        for f in range(3)
    ]
    frame = recipe.native_capture_comparison(records, 0)

    assert len(frame) == 3
    assert list(frame["delta"]) == [0.0, 0.0, 0.0]
    assert list(frame["v1_run_id"]) == ["v1_0", "v1_1", "v1_2"]


def test_a_disagreement_is_reported_not_smoothed_over(recipe, capsys):
    records = [
        {"script": recipe.NATIVE, "arm": "yolo26n", "repeat": 0, "fold": 0,
         "run_id": "a", "f1_macro": 0.600, "extra": {"run_id_extra": "native_recipe"}},
        {"script": recipe.NATIVE, "arm": "yolo26n", "repeat": 0, "fold": 0,
         "run_id": "b", "f1_macro": 0.615,
         "extra": {"run_id_extra": "native_recipe_v2", "capture_version": 2}},
    ]
    recipe.print_capture_comparison(recipe.native_capture_comparison(records, 0))
    out = capsys.readouterr().out

    assert "THEY DIFFER" in out
    assert "BETWEEN-SESSION" in out
    assert "neither supersedes the other" in out


def test_agreement_says_the_description_attaches_to_published_numbers(recipe, capsys):
    records = [
        {"script": recipe.NATIVE, "arm": "yolo26n", "repeat": 0, "fold": 0,
         "run_id": "a", "f1_macro": 0.6, "extra": {"run_id_extra": "native_recipe"}},
        {"script": recipe.NATIVE, "arm": "yolo26n", "repeat": 0, "fold": 0,
         "run_id": "b", "f1_macro": 0.6,
         "extra": {"run_id_extra": "native_recipe_v2", "capture_version": 2}},
    ]
    recipe.print_capture_comparison(recipe.native_capture_comparison(records, 0))
    out = capsys.readouterr().out
    assert "IDENTICAL" in out
    assert "do not move" in out


# ==================================== the table and its footnote must agree


def _native(fold, f1, capture, run_id=None):
    extra = {"run_id_extra": "native_recipe" if capture == 1 else "native_recipe_v2",
             "protocol": "native", "preprocessing": "ultralytics_default"}
    if capture >= 2:
        extra["capture_version"] = capture
    return {"script": "03c_native_recipe", "arm": "yolo26n", "repeat": 0,
            "fold": fold, "f1_macro": f1, "epochs": 25,
            "run_id": run_id or "%s_%d" % (capture, fold), "extra": extra}


def _uniform(arm, fold, f1):
    return {"script": "03_run_cv", "arm": arm, "repeat": 0, "fold": fold,
            "f1_macro": f1, "epochs": 25, "run_id": "%s_%d" % (arm, fold),
            "extra": {"protocol": "uniform"}}


def _both_captures():
    v1 = [0.635422, 0.493089, 0.607743, 0.572695, 0.584331]
    v2 = [0.664768, 0.540807, 0.632015, 0.590258, 0.610108]
    yolo = [0.572968, 0.391518, 0.545736, 0.564640, 0.541523]
    mobile = [0.671546, 0.609770, 0.558935, 0.651321, 0.414627]
    records = []
    for fold in range(5):
        records.append(_native(fold, v1[fold], 1))
        records.append(_native(fold, v2[fold], 2))
        records.append(_uniform("yolo26n", fold, yolo[fold]))
        records.append(_uniform("mobilenetv3_small", fold, mobile[fold]))
    return records


def test_two_captures_no_longer_collapse_silently(recipe):
    """fold_series keyed on fold, so with both captures present the later
    record won and Table 1 reported v2 while its footnote said v1."""
    with pytest.raises(SystemExit) as caught:
        recipe.fold_series(_both_captures(), "yolo26n", recipe.NATIVE, 0)
    assert "no way to choose between them" in str(caught.value)


def test_asking_for_a_capture_resolves_it(recipe):
    v1 = recipe.fold_series(_both_captures(), "yolo26n", recipe.NATIVE, 0, capture=1)
    v2 = recipe.fold_series(_both_captures(), "yolo26n", recipe.NATIVE, 0, capture=2)
    assert v1[0] == 0.635422
    assert v2[0] == 0.664768


def test_the_table_carries_the_capture_version(recipe):
    frame = recipe.recipe_table(_both_captures(), 0, 0.25)
    native = frame[frame["row"] == recipe.NATIVE_ROW].iloc[0]
    replicate = frame[frame["row"] == recipe.NATIVE_REPLICATE_ROW].iloc[0]

    assert native["capture_version"] == 2
    assert bool(native["reported"]) is True
    assert replicate["capture_version"] == 1
    assert bool(replicate["reported"]) is False
    assert "capture_version" in frame.columns


def test_the_reported_native_mean_is_v2(recipe):
    """v1 puts the native row BELOW mobilenet by 0.0025; v2 puts it ABOVE by
    0.0264. Those are different sentences, so which one is reported cannot be
    an accident of dict ordering."""
    frame = recipe.recipe_table(_both_captures(), 0, 0.25)
    native = frame[frame["row"] == recipe.NATIVE_ROW].iloc[0]
    mobile = frame[frame["row"] == "mobilenetv3_small / uniform"].iloc[0]

    assert native["f1_macro_mean"] == pytest.approx(0.6076, abs=1e-4)
    assert native["f1_macro_mean"] - mobile["f1_macro_mean"] == pytest.approx(
        0.0264, abs=1e-4)


def test_the_conclusion_names_the_capture_it_used(recipe):
    frame = recipe.recipe_table(_both_captures(), 0, 0.25)
    lines = " ".join(recipe.recipe_conclusion(frame))

    assert "CAPTURE v2" in lines
    assert "NOT REPORTED" in lines
    assert "+0.0843" in lines          # native against its own uniform run
    assert "+0.0264" in lines          # native against mobilenet


def test_the_conclusion_names_all_five_axes(recipe):
    """+0.0843 is not an augmentation effect and must not be read as one."""
    frame = recipe.recipe_table(_both_captures(), 0, 0.25)
    lines = " ".join(recipe.recipe_conclusion(frame))

    assert "FIVE AXES" in lines
    for axis in ("preprocessing", "augmentation", "optimizer", "schedule",
                 "checkpoint"):
        assert axis in lines
    assert "AdamW" in lines and "top-1 accuracy" in lines
    assert "NOT measured here" in lines


def test_the_footnote_agrees_with_the_table(recipe, capsys):
    """The bug: the table showed v2 and the note said v1."""
    recipe.print_capture_comparison(
        recipe.native_capture_comparison(_both_captures(), 0))
    out = capsys.readouterr().out

    assert "TABLE 1 REPORTS v2" in out
    assert "reports v1" not in out
    assert "neither supersedes the other" in out


def test_the_spread_is_set_against_the_effect(recipe):
    """A +0.0843 effect beside a 0.0289 between-session spread is a direction,
    not a third decimal."""
    frame = recipe.recipe_table(_both_captures(), 0, 0.25)
    lines = " ".join(recipe.recipe_conclusion(frame))
    assert "between-session spread" in lines
    assert "quote the direction" in lines


# =============================== every converted variant must be SCORED
#
# Static PTQ started working and every quantised row went to "n/a" -- dynamic
# too, which had scored before. Three causes, and only the first was guessed:
#
#   * predict_logits defaults to next(module.parameters()).device, and a
#     statically quantised module has NO parameters left to ask -- they are
#     packed into buffers. That raises StopIteration.
#   * eager quantised kernels are CPU-only regardless.
#   * one try wrapped BOTH the macro-F1 and the flip count, so a failure in the
#     verification discarded the measurement.
#
# Size alone cannot answer the microcontroller question. "3.2x smaller" needs
# "at a cost of X macro-F1", and X is what these tests protect.


def test_a_statically_quantised_model_has_no_parameters_to_infer_from(quantise):
    """The root cause, stated as a fact about torch rather than a guess."""
    pytest.importorskip("torch")

    converted, reason = quantise.static_ptq(_toy_model().eval(), _ToyCache(),
                                            list(range(16)),
                                            {i: i % 4 for i in range(16)})
    assert converted is not None, reason
    with pytest.raises(StopIteration):
        next(converted.parameters())


def test_both_variants_score_with_the_device_pinned(quantise):
    """The fix, end to end: dynamic AND static produce a macro-F1."""
    pytest.importorskip("torch")

    model = _toy_model().eval()
    idxs = list(range(32))
    labels = {i: i % 4 for i in idxs}
    cfg = {"classes": ["a", "b", "c", "d"]}

    for name, converted in (
        ("dynamic", quantise.dynamic_ptq(model)[0]),
        ("static", quantise.static_ptq(model, _ToyCache(), idxs, labels)[0]),
    ):
        assert converted is not None, name
        result = quantise.macro_f1(converted, _ToyCache(), idxs, labels, cfg,
                                   device=quantise.QUANTISED_DEVICE)
        assert result["f1_macro"] is not None, name
        assert 0.0 <= result["f1_macro"] <= 1.0


def test_the_device_is_pinned_not_inferred(quantise):
    assert quantise.QUANTISED_DEVICE == "cpu"

    source = (REPO_ROOT / "scripts" / "10_quantise.py").read_text(encoding="utf-8")
    assert "device=QUANTISED_DEVICE" in source
    # the fp32 baseline must be scored on the same device, or the delta
    # between them carries a device difference
    assert "module = module.to(QUANTISED_DEVICE)" in source


def test_a_flip_count_failure_does_not_discard_the_macro_f1(quantise):
    """They shared one try. An exception while counting flips set f1 back to
    None and the row read n/a as though the model could not be scored."""
    source = (REPO_ROOT / "scripts" / "10_quantise.py").read_text(encoding="utf-8")
    assert "_flipcount_failure" in source
    assert "macro-F1 above" in source
    # the flip attempt is guarded by the score having succeeded
    assert "if f1 is not None:" in source


def test_an_unscored_row_is_reported_loudly(quantise):
    """A size with no accuracy cannot support a cost-of-quantisation claim,
    and must not be left as a quiet n/a in a column."""
    source = (REPO_ROOT / "scripts" / "10_quantise.py").read_text(encoding="utf-8")
    assert "SCORING FAILED" in source
    assert "ROWS WITH A SIZE BUT NO ACCURACY" in source
    assert "cannot support a" in source


def test_prediction_vector_accepts_a_device(quantise):
    import inspect

    assert "device" in inspect.signature(quantise.prediction_vector).parameters
    assert "device" in inspect.signature(quantise.macro_f1).parameters


# ================================= static PTQ: FX where it works, named gap
#
# Eager mode needs the model WRITTEN for quantisation -- residual adds as
# FloatFunctional, unsupported activations in stubs. FX rewrites the graph
# instead and leaves operations without a quantised kernel in floating point.
#
# Measured on the real architectures: FX converts mobilenetv3_small and
# resnet18 at 100 % of parameters and both score. It cannot trace the
# ultralytics forward at all, and eager converts those but produces a graph
# with no QuantizedCPU kernel for aten::add.out or aten::silu.out. For those
# three the accuracy cost is UNMEASURED, with a named cause. No third
# approach was attempted.


def test_fx_is_tried_before_eager(quantise):
    import inspect

    source = inspect.getsource(quantise.static_ptq)
    fx_at = source.index("static_ptq_fx")
    eager_at = source.index("_wrap_for_static")
    assert fx_at < eager_at, "FX handles models not written for quantisation"


def test_fx_converts_a_traceable_model_completely(quantise):
    """mobilenetv3_small and resnet18 reach 100 %. The toy model stands in for
    them here so the test needs no checkpoint."""
    pytest.importorskip("torch")

    model = _toy_model().eval()
    converted, reason = quantise.static_ptq(
        model, _ToyCache16(), list(range(24)), {i: i % 4 for i in range(24)}
    )
    assert converted is not None, reason
    assert "fx graph mode" in reason
    assert quantise.quantised_parameter_fraction(model, converted) == 100.0


def test_the_parameter_fraction_measures_what_is_left_in_fp32(quantise):
    """It must work for FX too, whose module names do not survive tracing --
    which is why the name-matching coverage cannot be the headline number."""
    torch = pytest.importorskip("torch")

    model = torch.nn.Linear(4, 2)
    assert quantise.quantised_parameter_fraction(model, model) == 0.0

    class NoParams(torch.nn.Module):
        pass

    assert quantise.quantised_parameter_fraction(model, NoParams()) == 100.0


def test_the_gap_names_both_operations_and_both_approaches(quantise):
    note = quantise.STATIC_GAP_NOTE
    assert "aten::add.out" in note and "aten::silu.out" in note
    assert "residual addition" in note and "SiLU" in note
    assert "Proxy object cannot be iterated" in note
    assert "IS NOT MEASURED" in note
    # the sentence wraps, so match on parts that survive the line break
    assert "No third" in note and "approach was attempted" in note


def test_the_gap_is_in_the_csv_header_too(quantise):
    header = " ".join(quantise.HEADER)
    assert "aten::add.out" in header and "aten::silu.out" in header
    assert "CANNOT BE SCORED" in header
    assert "IS NOT MEASURED" in header
    assert "No third approach was tried" in header


def test_an_unmeasured_row_says_so_rather_than_n_a(quantise):
    """'n/a' reads as a missing column. 'NOT MEASURED' is a finding."""
    source = (REPO_ROOT / "scripts" / "10_quantise.py").read_text(encoding="utf-8")
    assert '"NOT MEASURED"' in source
    assert "STATIC_GAP_NOTE" in source


def test_the_coverage_quoted_is_the_parameter_fraction(quantise):
    source = (REPO_ROOT / "scripts" / "10_quantise.py").read_text(encoding="utf-8")
    assert '"coverage_pct_%s" % method: converted_pct,' in source
    assert "coverage_pct_by_layer_name_" in source, (
        "the name-matched census is kept, but not as the headline number"
    )


def test_the_class_list_exists(quantise):
    """The flip count died with NameError: name 'classes' is not defined --
    the check that verifies the deltas, lost to a missing line."""
    source = (REPO_ROOT / "scripts" / "10_quantise.py").read_text(encoding="utf-8")
    assert 'classes = list(data_cfg["classes"])' in source
    assert source.index('classes = list(data_cfg["classes"])') < source.index(
        "                        classes,"
    )
