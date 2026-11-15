from lastmile_data.evaluate_shared_zone_grammar import overlap_diagnostic, pool_target_history


def test_pool_target_history_adds_only_cross_station_routes():
    model = {
        "station_infos": {"A": {"r1": {"zone_list": ["A-1.1A"]}}},
        "superSets": {"A": {"r1": {"superList": ["A-1"]}}},
        "clusterKeys": {},
    }
    pooled = pool_target_history(model, "B")
    assert pooled["station_infos"]["B"] == {"A:r1": {"zone_list": ["A-1.1A"]}}
    assert pooled["superSets"]["B"] == {"A:r1": {"superList": ["A-1"]}}
    assert "B" not in model["station_infos"]


def test_overlap_diagnostic_uses_only_zone_sets():
    routes = {"target": {"stops": {
        "s1": {"zone_id": "A-1.1A"}, "s2": {"zone_id": "A-1.1B"},
    }}}
    model = {"station_infos": {"A": {
        "history": {"zone_list": ["A-1.1A", "A-1.1C"]},
    }}}
    row = overlap_diagnostic(routes, model)[0]
    assert row["best_jaccard"] == 1 / 3
    assert row["zone_coverage"] == 1 / 2
