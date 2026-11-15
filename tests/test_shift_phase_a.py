from lastmile_data.evaluate_shift_phase_a import bootstrap_ci, choose_groups, select_targets


def synthetic_routes():
    routes = {}
    for station in ("A", "B", "C", "D", "E", "F"):
        for day in range(1, 8):
            for index in range(12):
                rid = f"{station}-{day}-{index}"
                routes[rid] = {"station_code": station, "date_YYYY_MM_DD": f"2018-08-{day:02d}"}
    return routes


def test_group_and_target_selection_is_supported_deterministic_and_bounded():
    routes = synthetic_routes()
    eligible = set(routes)
    groups = choose_groups(routes, eligible)
    assert groups["station"] == ["A", "B", "C", "D", "E"]
    assert groups["temporal"] == ["2018-08-07", "2018-08-06", "2018-08-05", "2018-08-04", "2018-08-03"]
    first = select_targets(routes, eligible, groups)
    second = select_targets(routes, eligible, groups)
    assert first == second
    assert all(len(ids) == 12 for mode in first.values() for ids in mode.values())


def test_bootstrap_ci_is_reproducible_and_preserves_constant():
    assert bootstrap_ci([2.0] * 10) == [2.0, 2.0]
    assert bootstrap_ci([-1, 0, 1], repetitions=100) == bootstrap_ci([-1, 0, 1], repetitions=100)
