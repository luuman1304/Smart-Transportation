import unittest

from lastmile_data.build_residual_oof_pilot_v2 import residual_candidates_v2


class ResidualOOFV2Tests(unittest.TestCase):
    def test_v2_contains_only_anchor_and_one_hop_moves(self):
        anchor = ["D", "A", "B", "C", "E"]
        route = {"station_code": "S", "stops": {
            "D": {"type": "Station", "lat": 0, "lng": 0, "zone_id": None},
            "A": {"type": "Dropoff", "lat": 1, "lng": 1, "zone_id": "X-1.1A"},
            "B": {"type": "Dropoff", "lat": 1, "lng": 1, "zone_id": "X-1.1A"},
            "C": {"type": "Dropoff", "lat": 2, "lng": 2, "zone_id": "Y-2.2B"},
            "E": {"type": "Dropoff", "lat": 2, "lng": 2, "zone_id": "Y-2.2B"},
        }}
        costs = {a: {b: 0.0 if a == b else 10.0 for b in anchor} for a in anchor}
        result = residual_candidates_v2(anchor, route, costs, limit=12)
        self.assertEqual(result[0], anchor)
        self.assertEqual(len({tuple(value) for value in result}), len(result))
        self.assertGreater(len(result), 1)


if __name__ == "__main__":
    unittest.main()
