import pandas as pd

from lastmile_data.lade_pairwise import Geometry, State, eligible_split, stable_fold


def test_stable_fold_is_deterministic_and_bounded():
    assert stable_fold("Shanghai:1") == stable_fold("Shanghai:1")
    assert 0 <= stable_fold("Shanghai:1") < 5


def test_common_time_split_is_disjoint(monkeypatch):
    rows = []
    for group in ("A:1", "B:2"):
        rows.extend({"group": group, "date": "09-01", "route_id": f"{group}:tr{i}"} for i in range(20))
        rows.extend({"group": group, "date": "10-20", "route_id": f"{group}:te{i}"} for i in range(10))
    train, test = eligible_split(pd.DataFrame(rows))
    assert len(train) == 40
    assert len(test) == 20
    assert set(train.route_id).isdisjoint(test.route_id)


def test_geometry_state_distance_ranks_are_unique():
    class Row:
        aoi_ids = [1, 2, 3]
        aoi_types = [0, 1, 2]
        lngs = [0.0, 0.01, 0.02]
        lats = [0.0, 0.0, 0.0]
    geometry = Geometry.from_row(Row())
    state = State.build(geometry, 1, [2, 3], None)
    assert state.distance_rank == {2: 0.0, 3: 1.0}
