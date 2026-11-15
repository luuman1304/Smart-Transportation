from types import SimpleNamespace

import pandas as pd
import pytest

from lastmile_data.lade_external import (
    TransitionModel,
    _round_robin_sample,
    normalized_edit_distance,
    normalized_footrule,
    normalized_kendall_loss,
    predict,
)


def test_permutation_metrics_have_correct_bounds_and_direction():
    actual = [1, 2, 3, 4]
    assert normalized_kendall_loss(actual, actual) == 0
    assert normalized_edit_distance(actual, actual) == 0
    assert normalized_footrule(actual, actual) == 0
    reversed_ = list(reversed(actual))
    assert normalized_kendall_loss(actual, reversed_) == 1
    assert normalized_footrule(actual, reversed_) == 1
    assert 0 <= normalized_edit_distance(actual, reversed_) <= 1


def test_balanced_sampler_is_deterministic_and_group_balanced():
    frame = pd.DataFrame({"group": ["a"] * 10 + ["b"] * 10 + ["c"] * 10, "value": range(30)})
    first = _round_robin_sample(frame, 11, 42)
    second = _round_robin_sample(frame, 11, 42)
    assert first.index.tolist() == second.index.tolist()
    counts = first.groupby("group").size()
    assert counts.max() - counts.min() <= 1
    assert len(first) == 11


def test_balanced_sampler_excludes_groups_below_common_quota():
    frame = pd.DataFrame({"group": ["sparse"] + ["a"] * 10 + ["b"] * 10, "value": range(21)})
    sampled = _round_robin_sample(frame, 12, 42)
    assert "sparse" not in set(sampled["group"])
    assert sampled.groupby("group").size().to_dict() == {"a": 6, "b": 6}


def test_regional_history_is_optional_and_prediction_is_a_permutation():
    training = pd.DataFrame([
        {"group": "x:1", "aoi_ids": [1, 2, 3], "aoi_types": [1, 1, 2]},
        {"group": "x:1", "aoi_ids": [1, 2, 4], "aoi_types": [1, 1, 2]},
    ])
    row = SimpleNamespace(
        group="x:1", aoi_ids=[1, 2, 3], aoi_types=[1, 1, 2],
        lngs=[0.0, 0.001, 0.002], lats=[0.0, 0.0, 0.0],
    )
    for regional in (False, True):
        model = TransitionModel.fit(training, use_regional_history=regional)
        result = predict(row, model)
        assert result[0] == 1
        assert sorted(result) == [1, 2, 3]


def test_sampler_rejects_oversized_request():
    with pytest.raises(ValueError):
        _round_robin_sample(pd.DataFrame({"group": ["a"]}), 2, 1)


def test_kendall_loss_matches_one_inversion():
    assert normalized_kendall_loss([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)
