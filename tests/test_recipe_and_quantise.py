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


# ------------------------------------------------------ which cell is which


def test_an_override_run_is_not_mistaken_for_the_published_arm(recipe):
    """yolo26n/MuSGD and yolo26n/SGD differ only by the override, so a cell
    that ignored it would silently average the two together."""
    records = [
        record("yolo26n", recipe.PUBLISHED, 0, 0.50),
        record("yolo26n", recipe.CONTRAST, 0, 0.40, override="sgd"),
    ]
    published = next(c for c in recipe.CELLS if c["cell"] == "yolo26n / MuSGD")
    contrast = next(c for c in recipe.CELLS if c["cell"] == "yolo26n / SGD")

    assert len(recipe.cell_records(records, published, 0)) == 1
    assert recipe.cell_records(records, published, 0)[0]["f1_macro"] == 0.50
    assert recipe.cell_records(records, contrast, 0)[0]["f1_macro"] == 0.40


def test_the_override_is_read_case_insensitively(recipe):
    records = [record("yolo26n", recipe.CONTRAST, 0, 0.4, override="SGD")]
    contrast = next(c for c in recipe.CELLS if c["cell"] == "yolo26n / SGD")
    assert len(recipe.cell_records(records, contrast, 0)) == 1


def test_another_repeat_is_not_pulled_into_the_2x2(recipe):
    records = [
        record("yolo26n", recipe.PUBLISHED, 0, 0.50, repeat=0),
        record("yolo26n", recipe.PUBLISHED, 0, 0.90, repeat=1),
    ]
    published = next(c for c in recipe.CELLS if c["cell"] == "yolo26n / MuSGD")
    assert [r["f1_macro"] for r in recipe.cell_records(records, published, 0)] == [0.50]


def test_the_2x2_has_four_cells_plus_the_native_row(recipe):
    assert len(recipe.CELLS) == 5
    uniform = [c for c in recipe.CELLS if c["preprocessing"] == recipe.LETTERBOX]
    native = [c for c in recipe.CELLS if c["preprocessing"] == recipe.ULTRALYTICS]
    assert len(uniform) == 4 and len(native) == 1
    assert {(c["arm"], c["optimizer"]) for c in uniform} == {
        ("yolo26n", "musgd"), ("yolo26n", "sgd"),
        ("mobilenetv3_small", "sgd"), ("mobilenetv3_small", "musgd"),
    }


# ----------------------------------------------------------- the verdict


def build(recipe, **means):
    rows = []
    for cell in recipe.CELLS:
        rows.append({"cell": cell["cell"],
                     "f1_macro_mean": means.get(cell["cell"]),
                     "preprocessing": cell["preprocessing"]})
    return pd.DataFrame(rows)


def test_a_missing_musgd_mobilenet_cell_refuses_to_conclude(recipe):
    """Running YOLO with SGD alone does not break the confound, and the script
    must say so rather than imply the question was answered."""
    frame = build(recipe, **{"yolo26n / MuSGD": 0.52, "yolo26n / SGD": 0.48,
                             "mobilenetv3_small / SGD": 0.58})
    lines = "\n".join(recipe.confound_verdict(frame))
    assert "CANNOT CONCLUDE YET" in lines
    assert "MISSING" in lines
    assert "does not separate architecture from optimizer" in lines


def test_the_finding_survives_when_mobilenet_still_wins_at_equal_optimizer(recipe):
    frame = build(recipe, **{"yolo26n / MuSGD": 0.52, "yolo26n / SGD": 0.48,
                             "mobilenetv3_small / SGD": 0.58,
                             "mobilenetv3_small / MuSGD": 0.57})
    lines = "\n".join(recipe.confound_verdict(frame))
    assert "SURVIVES THE CONFOUND" in lines
    assert "tracks the ARCHITECTURE" in lines


def test_the_finding_is_called_out_when_it_does_not_survive(recipe):
    """The case that must reach the manuscript before submission, not after."""
    frame = build(recipe, **{"yolo26n / MuSGD": 0.60, "yolo26n / SGD": 0.48,
                             "mobilenetv3_small / SGD": 0.58,
                             "mobilenetv3_small / MuSGD": 0.55})
    lines = "\n".join(recipe.confound_verdict(frame))
    assert "DOES NOT SURVIVE" in lines
    assert "BEFORE SUBMISSION" in lines


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
