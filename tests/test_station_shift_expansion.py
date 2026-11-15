import json

from lastmile_data.evaluate_station_shift_expansion import EXPECTED_GROUPS, expansion_groups


def test_expansion_selects_all_supported_nonpilot_stations(monkeypatch, tmp_path):
    pilot = tmp_path / "protocol.json"
    pilot.write_text(json.dumps({"groups": {"station": ["P0", "P1", "P2", "P3", "P4"]}}))
    monkeypatch.setattr("lastmile_data.evaluate_station_shift_expansion.PILOT", pilot)
    routes = {}
    for station in [f"P{i}" for i in range(5)] + [f"E{i:02d}" for i in range(EXPECTED_GROUPS)]:
        for index in range(12):
            routes[f"{station}-{index}"] = {"station_code": station}
    groups = expansion_groups(routes, set(routes))
    assert groups["station"] == [f"E{i:02d}" for i in range(EXPECTED_GROUPS)]
