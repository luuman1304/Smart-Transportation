import unittest

from lastmile_data.build_residual_oof_pilot import residual_candidates


class ResidualOOFTests(unittest.TestCase):
    def test_candidates_valid_deterministic_and_budgeted(self):
        anchor = ["D", "A", "B", "C", "E"]
        route = {"station_code": "S", "stops": {
            "D": {"type": "Station", "lat": 0, "lng": 0, "zone_id": None},
            "A": {"type": "Dropoff", "lat": 1, "lng": 1, "zone_id": "X-1.1A"},
            "B": {"type": "Dropoff", "lat": 1, "lng": 1, "zone_id": "X-1.1A"},
            "C": {"type": "Dropoff", "lat": 2, "lng": 2, "zone_id": "Y-2.2B"},
            "E": {"type": "Dropoff", "lat": 2, "lng": 2, "zone_id": "Y-2.2B"},
        }}
        costs = {a: {b: 0.0 if a == b else 10.0 for b in anchor} for a in anchor}
        first = residual_candidates(anchor, route, costs, limit=12)
        second = residual_candidates(anchor, route, costs, limit=12)
        self.assertEqual(first, second)
        self.assertEqual(first[0], anchor)
        self.assertEqual(len({tuple(value) for value in first}), len(first))
        for sequence in first:
            self.assertEqual(sequence[0], "D")
            self.assertEqual(sorted(sequence), sorted(anchor))


if __name__ == "__main__":
    unittest.main()
