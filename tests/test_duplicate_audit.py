"""Task B: the leak byte hashing cannot see.

Nothing here needs images or imagehash. What is tested is the logic that turns
"these two pictures look alike" into "and therefore the reported macro-F1 is
optimistic" -- the clustering and the fold-straddle count. Those are what decide
whether the whole cross-validation has to be refrozen, so they are pinned
against hand-built folds where the right answer is known by construction.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location(
        "duplicate_audit", REPO_ROOT / "scripts" / "09_duplicate_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["duplicate_audit"] = module
    spec.loader.exec_module(module)
    return module


def fold(repeat, fold_index, train, test, val=()):
    return {
        "repeat": repeat,
        "fold": fold_index,
        "train_idx": list(train),
        "val_idx": list(val),
        "test_idx": list(test),
    }


# ------------------------------------------------------------- clustering


def test_a_chain_becomes_one_cluster(audit):
    """Near-duplication is not transitive, but for LEAK purposes it behaves as
    if it were: if any two members of a chain land on opposite sides,
    information crosses."""
    clusters = audit.connected_components([(1, 2), (2, 3), (10, 11)])
    assert clusters == [[1, 2, 3], [10, 11]]


def test_an_unpaired_image_forms_no_cluster(audit):
    clusters = audit.connected_components([(1, 2)])
    assert clusters == [[1, 2]]
    assert 99 not in [i for c in clusters for i in c]


def test_no_pairs_means_no_clusters(audit):
    assert audit.connected_components([]) == []


def test_clusters_come_back_sorted_and_stable(audit):
    a = audit.connected_components([(5, 1), (3, 5), (9, 8)])
    b = audit.connected_components([(8, 9), (5, 3), (1, 5)])
    assert a == b == [[1, 3, 5], [8, 9]]


# ------------------------------------------------- the number that decides


def test_a_cluster_inside_the_training_set_leaks_nothing(audit):
    """The whole point of counting straddles rather than duplicates: similar
    images that never cross the boundary cost nothing."""
    folds = [fold(0, 0, train=[1, 2, 3], test=[4, 5])]
    report = audit.straddle_report([1, 2], folds)
    assert report["n_folds_straddled"] == 0
    assert report["folds_straddled"] == []


def test_a_cluster_inside_the_test_set_leaks_nothing_either(audit):
    folds = [fold(0, 0, train=[1, 2], test=[3, 4])]
    assert audit.straddle_report([3, 4], folds)["n_folds_straddled"] == 0


def test_a_cluster_across_the_boundary_is_counted(audit):
    folds = [fold(0, 0, train=[1, 2], test=[3, 4])]
    report = audit.straddle_report([2, 3], folds)
    assert report["n_folds_straddled"] == 1
    assert report["folds_straddled"][0] == {
        "repeat": 0, "fold": 0, "n_train": 1, "n_test": 1
    }


def test_the_count_is_per_fold_not_per_pair(audit):
    """Repeated k-fold puts every image in three test partitions, so the same
    cluster can straddle several folds and each one is a separate leak."""
    folds = [
        fold(0, 0, train=[1, 2], test=[3]),
        fold(0, 1, train=[1, 3], test=[2]),
        fold(0, 2, train=[2, 3], test=[1]),
    ]
    assert audit.straddle_report([1, 2, 3], folds)["n_folds_straddled"] == 3


def test_a_member_outside_every_fold_does_not_invent_a_straddle(audit):
    folds = [fold(0, 0, train=[1, 2], test=[3, 4])]
    assert audit.straddle_report([1, 999], folds)["n_folds_straddled"] == 0


def test_the_validation_slice_counts_as_train(audit):
    """It is carved out of train, so an image there was still trained around."""
    folds = [fold(0, 0, train=[1, 2], test=[5], val=[2])]
    assert audit.straddle_report([2, 5], folds)["n_folds_straddled"] == 1


# ----------------------------------------------------------- the dev split


def test_a_cluster_inside_one_dev_part_is_clean(audit):
    dev = {"train_idx": [1, 2, 3], "val_idx": [4], "test_idx": [5]}
    report = audit.dev_split_straddle([1, 2], dev)
    assert report["dev_straddles"] is False
    assert report["dev_parts"] == ["train"]


def test_a_cluster_across_dev_train_and_val_is_flagged(audit):
    """Hyperparameters were selected on this split, so a leak here inflates the
    SELECTION rather than the reported score -- quieter, and just as real."""
    dev = {"train_idx": [1, 2], "val_idx": [3], "test_idx": [4]}
    report = audit.dev_split_straddle([2, 3], dev)
    assert report["dev_straddles"] is True
    assert set(report["dev_parts"]) == {"train", "val"}
    assert report["dev_counts"] == {"train": 1, "val": 1}


def test_a_missing_dev_part_is_not_a_crash(audit):
    assert audit.dev_split_straddle([1], {})["dev_straddles"] is False


# -------------------------------------------------------------- histogram


def test_the_histogram_shows_every_distance_present(audit):
    distances = {(1, 2): 0, (1, 3): 5, (2, 3): 5, (1, 4): 31}
    assert audit.distance_histogram(distances) == {0: 1, 5: 2, 31: 1}


def test_the_histogram_is_sorted_by_distance(audit):
    distances = {(1, 2): 9, (1, 3): 0, (2, 3): 4}
    assert list(audit.distance_histogram(distances)) == [0, 4, 9]


# ---------------------------------------------------------- refusals


def test_a_missing_hash_library_refuses_rather_than_improvising(audit, monkeypatch):
    """An unvalidated hand-rolled hash would be worse than no audit: it would
    produce a clean verdict nobody could trust."""
    import builtins

    real_import = builtins.__import__

    def no_imagehash(name, *args, **kwargs):
        if name == "imagehash":
            raise ImportError("no imagehash")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_imagehash)
    with pytest.raises(SystemExit) as caught:
        audit.require_imagehash()
    assert "pip install imagehash" in str(caught.value)


def test_a_partial_data_root_is_refused(audit, tmp_path):
    """Auditing a subset would report 'no near-duplicates' for pairs it never
    compared -- the most dangerous possible wrong answer here."""
    import pandas as pd

    index = pd.DataFrame([
        {"idx": 0, "relpath": "a/one.png", "class": "a", "sha1": "x", "excluded": False},
        {"idx": 1, "relpath": "a/two.png", "class": "a", "sha1": "y", "excluded": False},
    ])
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.png").write_bytes(b"x")

    with pytest.raises(SystemExit) as caught:
        audit.resolve_images(index, tmp_path)
    message = str(caught.value)
    assert "does not contain 1 of the 2" in message
    assert "two.png" in message


def test_excluded_images_are_not_audited(audit, tmp_path):
    """They are not in any fold, so they cannot leak into one."""
    import pandas as pd

    index = pd.DataFrame([
        {"idx": 0, "relpath": "a/one.png", "class": "a", "sha1": "x", "excluded": False},
        {"idx": 1, "relpath": "a/gone.png", "class": "a", "sha1": "y", "excluded": True},
    ])
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.png").write_bytes(b"x")

    found = audit.resolve_images(index, tmp_path)
    assert [idx for idx, _ in found] == [0]


def test_an_index_without_the_expected_columns_is_refused(audit, artifacts):
    import pandas as pd

    pd.DataFrame([{"file": "a.png"}]).to_csv(artifacts / "image_index.csv", index=False)
    with pytest.raises(SystemExit) as caught:
        audit.load_index(artifacts)
    assert "not the index" in str(caught.value)


# ------------------------------------------------------------- guarantees


def test_the_script_never_writes_folds_or_the_registry(audit):
    source = (REPO_ROOT / "scripts" / "09_duplicate_audit.py").read_text(encoding="utf-8")
    assert "append_record" not in source
    for forbidden in ("folds_path.write", "registry.append", "shutil.rmtree", "unlink("):
        assert forbidden not in source, "09 must change nothing: found %r" % forbidden


def test_both_remedies_are_offered_with_their_cost(audit):
    """If clusters straddle folds the answer is a decision, not an action."""
    assert "StratifiedGroupKFold" in audit.REMEDIES
    assert "165" in audit.REMEDIES          # the re-run cost, stated for both
    assert "THIS SCRIPT HAS CHANGED NOTHING" in audit.REMEDIES


def test_three_hashes_are_computed_with_one_named_primary(audit):
    assert set(audit.HASHES) == {"phash", "dhash", "average_hash"}
    assert audit.PRIMARY_HASH == "phash"
    assert audit.HASH_SIZE == 8
    assert audit.DEFAULT_THRESHOLD == 5


# ============================================================================
# The hash only PROPOSES. These four checks decide.
#
# A perceptual hash is a DCT over an 8x8 reduction, and a dynamometer card is a
# thin curve on a uniform white background, so almost all the low-frequency
# energy is shared corpus-wide whatever the class. A threshold borrowed from
# photographic work therefore measures shape family, not identity -- and on this
# corpus it flagged 116 pairs with exactly ONE at distance 0 and two thirds of
# the clusters carrying more than one label.
# ============================================================================


# -------------------------------------------------------------- timestamps


def test_both_filename_shapes_in_the_corpus_parse(audit):
    """692 of the 695 filenames carry a capture time in one of two shapes."""
    from datetime import datetime

    assert audit.parse_timestamp(
        "natural_flowing/Screenshot 2026-05-05 140648.png"
    ) == datetime(2026, 5, 5, 14, 6, 48)
    assert audit.parse_timestamp(
        "collide_pump_and_vibration/IMG_20260524_101625.jpg"
    ) == datetime(2026, 5, 24, 10, 16, 25)


def test_an_unparseable_name_returns_none_rather_than_guessing(audit):
    """Three files in the corpus are UUIDs. They are COUNTED, never invented."""
    assert audit.parse_timestamp(
        "natural_flowing/09300c24-3104-4285-ac84-16181e9aee2c.png"
    ) is None


def test_a_nonsense_date_is_not_accepted(audit):
    assert audit.parse_timestamp("x/Screenshot 2026-13-45 999999.png") is None


def test_the_time_spread_of_a_recapture_is_seconds(audit):
    from datetime import datetime

    times = {
        1: datetime(2026, 5, 5, 14, 6, 48),
        2: datetime(2026, 5, 5, 14, 6, 51),
    }
    spread = audit.cluster_time_spread([1, 2], times)
    assert spread["spread_s"] == 3.0
    assert spread["min_gap_s"] == 3.0
    assert spread["n_timed"] == 2


def test_two_survey_sessions_are_days_apart(audit):
    from datetime import datetime

    times = {1: datetime(2026, 5, 5, 14, 0, 0), 2: datetime(2026, 5, 8, 9, 0, 0)}
    assert audit.cluster_time_spread([1, 2], times)["spread_s"] > 86400


def test_an_untimed_member_does_not_break_the_spread(audit):
    from datetime import datetime

    times = {1: datetime(2026, 5, 5, 14, 0, 0), 2: None}
    spread = audit.cluster_time_spread([1, 2], times)
    assert spread["n_timed"] == 1
    assert spread["spread_s"] is None


def test_the_null_samples_same_class_pairs_only(audit):
    """Without the null, "these two are 40 minutes apart" means nothing."""
    from datetime import datetime, timedelta

    import pandas as pd

    base = datetime(2026, 5, 5, 12, 0, 0)
    index = pd.DataFrame([
        {"idx": i, "relpath": "c/%d.png" % i,
         "class": "a" if i < 5 else "b", "sha1": str(i), "excluded": False}
        for i in range(10)
    ])
    times = {i: base + timedelta(hours=i) for i in range(10)}

    gaps = audit.null_time_gaps(index, times, n_samples=200, seed=1)

    assert len(gaps) == 200
    assert gaps == sorted(gaps)
    # within class "a" the maximum gap is 4 h; within "b" also 4 h
    assert max(gaps) <= 4 * 3600


# ------------------------------------------------------------------ pixels


def test_the_verdict_thresholds_are_declared(audit):
    assert audit.SSIM_DUPLICATE == 0.98
    assert audit.SSIM_CERTAIN == 0.99


def test_ssim_is_not_hand_rolled(audit):
    """The pixel check decides whether 165 GPU re-runs happen. That is not a
    thing to settle with an unvalidated implementation."""
    import inspect

    source = inspect.getsource(audit.require_skimage)
    assert "refuses to hand-roll" in source
    assert "pip install scikit-image" in source


def test_pixel_similarity_uses_our_own_letterbox(audit):
    """"Identical after letterboxing" is the property that matters for a leak,
    because the letterbox is the transform the model sees."""
    import inspect

    source = inspect.getsource(audit.pixel_similarity)
    assert "load_letterboxed" in source
    assert "ssim" in source and "nrmse" in source


# ------------------------------------------- within-class vs between-class


def test_real_duplication_is_overwhelmingly_within_class(audit):
    import pandas as pd

    index = pd.DataFrame([
        {"idx": i, "relpath": "c/%d.png" % i,
         "class": "a" if i < 10 else "b", "sha1": str(i), "excluded": False}
        for i in range(20)
    ])
    class_of = {int(r["idx"]): r["class"] for _, r in index.iterrows()}
    pairs = [(0, 1), (2, 3), (4, 5), (10, 11)]        # all within class

    result = audit.class_contingency(pairs, class_of, index)

    assert result["within_class"] == 4
    assert result["between_class"] == 0
    assert result["observed_within_rate"] == 1.0
    assert result["enrichment"] > 2.0


def test_morphology_shows_up_as_a_base_rate_spread(audit):
    """If flagged pairs are spread across classes like the base rate, the hash
    is detecting class shape, not duplication."""
    import pandas as pd

    index = pd.DataFrame([
        {"idx": i, "relpath": "c/%d.png" % i,
         "class": "a" if i < 10 else "b", "sha1": str(i), "excluded": False}
        for i in range(20)
    ])
    class_of = {int(r["idx"]): r["class"] for _, r in index.iterrows()}
    pairs = [(0, 11), (1, 12), (2, 13), (3, 14)]      # all BETWEEN classes

    result = audit.class_contingency(pairs, class_of, index)

    assert result["within_class"] == 0
    assert result["observed_within_rate"] == 0.0
    assert result["enrichment"] == 0.0


# ----------------------------------------------------- threshold sensitivity


def test_the_sweep_reports_every_threshold_asked_for(audit):
    assert audit.SWEEP_THRESHOLDS == (0, 1, 2, 5)

    distances = {(1, 2): 0, (1, 3): 2, (2, 3): 5, (4, 5): 9}
    folds = [fold(0, 0, train=[1, 2], test=[3, 4, 5])]

    sweep = audit.threshold_sweep(distances, folds)

    assert [row["threshold"] for row in sweep] == [0, 1, 2, 5]
    assert sweep[0]["n_pairs"] == 1          # only the distance-0 pair
    assert sweep[3]["n_pairs"] == 3
    assert sweep[0]["n_clusters"] == 1


def test_a_conclusion_that_only_holds_at_5_is_visible_as_such(audit):
    """The whole point of the sweep: if nothing is flagged at 0-2 and the
    finding appears only at 5, it belongs to the number, not the data."""
    distances = {(1, 2): 5, (3, 4): 5}
    folds = [fold(0, 0, train=[1, 3], test=[2, 4])]

    sweep = audit.threshold_sweep(distances, folds)

    assert sweep[0]["n_pairs"] == 0 and sweep[0]["n_clusters_straddling"] == 0
    assert sweep[3]["n_pairs"] == 2 and sweep[3]["n_clusters_straddling"] == 2


# --------------------------------------------------------------- the verdict


def test_the_remedy_is_gated_on_the_pixel_and_time_evidence(audit):
    """Recommending 165 re-runs on Hamming distance alone is how a measurement
    of class morphology turns into a refrozen corpus."""
    source = (REPO_ROOT / "scripts" / "09_duplicate_audit.py").read_text(encoding="utf-8")

    assert "NO PAIR SURVIVES THE PIXEL CHECK" in source
    assert "NO REMEDY IS RECOMMENDED" in source
    assert "confirmed_straddling" in source
    assert "timestamps_agree" in source
    # the remedy text must sit behind all three
    remedy_at = source.index("STOPPING FOR A DECISION -- all three checks agree")
    gate_at = source.index("if not confirmed:")
    assert gate_at < remedy_at, "the remedy must be gated, not printed first"


def test_mixed_labels_are_called_out_as_the_hash_failing(audit):
    source = (REPO_ROOT / "scripts" / "09_duplicate_audit.py").read_text(encoding="utf-8")
    assert "MORE THAN ONE LABEL" in source
    assert "A true duplicate cannot have two labels" in source


def test_a_contact_sheet_is_produced_for_the_largest_clusters(audit):
    import inspect

    source = inspect.getsource(audit.contact_sheet)
    assert "n_clusters: int = 10" in source
    assert "MIXED" in source, "a mixed-label row must be labelled as such"


# ============================================================================
# The decoded-pixel hash: the check that was actually missing.
#
# File sha1 detects duplicate FILES. A re-encode changes every byte and leaves
# the decoded image identical, so the same card filed under two class
# directories slips through with two sha1s and two labels. A perceptual hash
# does not find these reliably either -- on line art over uniform white it
# floods the result with same-shape, different-card pairs.
#
# Hashing the decoded RGB buffer is exact: zero false positives by
# construction, and robust to re-encoding.
# ============================================================================


def a_card(size=(64, 48), shift=0):
    """A thin dark curve on white -- the shape of the real corpus."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse([5 + shift, 5, size[0] - 6, size[1] - 6], outline="black")
    return image


def test_a_reencode_keeps_the_decoded_hash(audit, tmp_path):
    """THE case file sha1 cannot reach. Same pixels, different bytes."""
    import hashlib

    original = tmp_path / "a.png"
    reencoded = tmp_path / "b.png"
    card = a_card()
    card.save(original, optimize=False, compress_level=0)
    card.save(reencoded, optimize=True, compress_level=9)

    file_a = hashlib.sha1(original.read_bytes()).hexdigest()
    file_b = hashlib.sha1(reencoded.read_bytes()).hexdigest()
    assert file_a != file_b, "the fixture must actually re-encode"

    assert audit.decoded_pixel_sha1(original) == audit.decoded_pixel_sha1(reencoded)


def test_a_format_change_keeps_the_decoded_hash(audit, tmp_path):
    png, bmp = tmp_path / "a.png", tmp_path / "a.bmp"
    card = a_card()
    card.save(png)
    card.save(bmp)
    assert audit.decoded_pixel_sha1(png) == audit.decoded_pixel_sha1(bmp)


def test_one_different_pixel_changes_the_hash(audit, tmp_path):
    """Exact means exact: this is not a near-duplicate detector."""
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    card = a_card()
    card.save(a)
    card.putpixel((0, 0), (254, 255, 255))
    card.save(b)
    assert audit.decoded_pixel_sha1(a) != audit.decoded_pixel_sha1(b)


def test_a_different_card_is_a_different_hash(audit, tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a_card().save(a)
    a_card(shift=3).save(b)
    assert audit.decoded_pixel_sha1(a) != audit.decoded_pixel_sha1(b)


def test_the_size_is_folded_in(audit, tmp_path):
    """Two buffers of equal length at different dimensions must not collide."""
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    from PIL import Image

    Image.new("RGB", (4, 6), "white").save(a)
    Image.new("RGB", (6, 4), "white").save(b)
    assert audit.decoded_pixel_sha1(a) != audit.decoded_pixel_sha1(b)


# ------------------------------------------------------------ grouping


def index_of(rows):
    import pandas as pd

    return pd.DataFrame(rows)


def test_excluded_images_are_still_scanned(audit, tmp_path):
    """Over all 695, not the clean 668: a group the existing rule already
    excluded must be distinguishable from one nothing has caught."""
    (tmp_path / "c").mkdir()
    for name in ("one.png", "two.png"):
        a_card().save(tmp_path / "c" / name)
    index = index_of([
        {"idx": 0, "relpath": "c/one.png", "class": "a", "sha1": "x", "excluded": True},
        {"idx": 1, "relpath": "c/two.png", "class": "b", "sha1": "y", "excluded": True},
    ])

    groups = audit.pixel_duplicate_groups(index, tmp_path, quiet=True)

    assert len(groups) == 1
    assert sorted(next(iter(groups.values()))) == [0, 1]


def test_a_singleton_is_not_a_group(audit, tmp_path):
    (tmp_path / "c").mkdir()
    a_card().save(tmp_path / "c" / "one.png")
    a_card(shift=4).save(tmp_path / "c" / "two.png")
    index = index_of([
        {"idx": 0, "relpath": "c/one.png", "class": "a", "sha1": "x", "excluded": False},
        {"idx": 1, "relpath": "c/two.png", "class": "a", "sha1": "y", "excluded": False},
    ])
    assert audit.pixel_duplicate_groups(index, tmp_path, quiet=True) == {}


# ------------------------------------------------------------ the description


def make_by_idx(rows):
    return {r["idx"]: r for r in rows}


def test_one_file_sha1_means_the_existing_rule_already_caught_it(audit):
    by_idx = make_by_idx([
        {"idx": 0, "class": "a", "sha1": "same", "excluded": True, "relpath": "a/0.png"},
        {"idx": 1, "class": "b", "sha1": "same", "excluded": True, "relpath": "b/1.png"},
    ])
    block = audit.describe_pixel_group([0, 1], by_idx, [], None)

    assert block["already_caught_by_file_sha1"] is True
    assert block["n_file_sha1s"] == 1
    assert block["all_members_excluded"] is True


def test_two_file_sha1s_means_a_reencode_nothing_caught(audit):
    """The defect this check exists for."""
    by_idx = make_by_idx([
        {"idx": 297, "class": "natural_flowing", "sha1": "aaa", "excluded": False,
         "relpath": "natural_flowing/uuid.png"},
        {"idx": 655, "class": "vibration", "sha1": "bbb", "excluded": False,
         "relpath": "vibration/Screenshot 2026-04-19 115549.png"},
    ])
    folds = [fold(0, f, train=[297], test=[655]) for f in range(3)]

    block = audit.describe_pixel_group([297, 655], by_idx, folds, None)

    assert block["already_caught_by_file_sha1"] is False
    assert block["n_file_sha1s"] == 2
    assert block["label_consistent"] is False
    assert block["labels"] == ["natural_flowing", "vibration"]
    assert block["n_folds_straddled"] == 3
    assert block["n_excluded_members"] == 0


def test_a_consistent_label_group_is_benign(audit):
    """Byte- or pixel-identical within one class is duplication, not a conflict."""
    by_idx = make_by_idx([
        {"idx": 0, "class": "a", "sha1": "x", "excluded": False, "relpath": "a/0.png"},
        {"idx": 1, "class": "a", "sha1": "y", "excluded": False, "relpath": "a/1.png"},
    ])
    block = audit.describe_pixel_group([0, 1], by_idx, [], None)
    assert block["label_consistent"] is True


# --------------------------------------------------------------- the verdict


def test_the_pixel_finding_states_the_direction_of_the_bias(audit):
    """Conflicting labels mean the straddle teaches one label and tests the
    other, so it DEPRESSES the reported numbers -- the conservative direction.
    Saying so is the difference between a documented residual and a scare."""
    source = (REPO_ROOT / "scripts" / "09_duplicate_audit.py").read_text(encoding="utf-8")
    assert "DIRECTION OF THE BIAS" in source
    assert "DEPRESSES the reported numbers" in source
    assert "NO REMEDY IS RECOMMENDED HERE" in source


def test_the_pixel_check_runs_over_the_full_index(audit):
    source = (REPO_ROOT / "scripts" / "09_duplicate_audit.py").read_text(encoding="utf-8")
    assert "pixel_duplicate_groups(index, data_root" in source, (
        "must be handed the full index, not the resolved 668"
    )
