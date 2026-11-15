import pytest

from lastmile_data.evaluate_soft_zone_prior import balanced_probabilities, soften_times


def synthetic_model():
    return {"station_infos": {
        "A": {"a": {"zone_list": ["X", "Y", "X", "Y"]}},
        "B": {"b": {"zone_list": ["X", "Z"]}},
    }}


def test_balanced_probabilities_average_stations_not_raw_route_counts():
    probabilities = balanced_probabilities(synthetic_model(), ["Q", "X", "Y", "Z"])
    assert probabilities["X", "Y"] > 0
    assert probabilities["X", "Z"] > 0
    assert all(probabilities["Q", right] == 1 for right in ("Q", "X", "Y", "Z"))


def test_soft_penalty_is_bounded_and_same_zone_is_neutral(monkeypatch):
    monkeypatch.setattr("lastmile_data.evaluate_soft_zone_prior.normalized_zones", lambda route: {"a": "X", "b": "Y", "c": "X"})
    routes = {"r": {"stops": {"a": {}, "b": {}, "c": {}}}}
    costs = {"r": {"a": {"a": 0, "b": 10, "c": 10}, "b": {"a": 10, "b": 0, "c": 10}, "c": {"a": 10, "b": 10, "c": 0}}}
    softened, rows = soften_times(routes, costs, synthetic_model(), alpha=.05)
    assert softened["r"]["a"]["c"] == 10
    assert rows[0]["max_added_seconds"] <= .5 + 1e-12
