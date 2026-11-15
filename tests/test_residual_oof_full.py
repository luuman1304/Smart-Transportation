import copy
import json
import tempfile
import unittest
from pathlib import Path

from lastmile_data.build_residual_oof_full import _checkpoint_ids, _score_route
from lastmile_data.reproduce_jpt import official_score


class ResidualOOFFullTests(unittest.TestCase):
    def test_parallel_worker_matches_official_score(self):
        stops = ["D", "A", "B", "C"]
        actual = ["D", "A", "B", "C"]
        sequences = [actual, ["D", "B", "A", "C"]]
        costs = {
            a: {b: float((i + 1) * (j + 2)) for j, b in enumerate(stops)}
            for i, a in enumerate(stops)
        }
        route = {"date_YYYY_MM_DD": "2020-01-01", "departure_time_utc": "00:00:00", "stops": {
            stop: {"type": "Station" if stop == "D" else "Dropoff"} for stop in stops
        }}
        packages = {stop: {} for stop in stops}
        result = _score_route(("R", 0, actual, sequences, costs, route, packages))
        expected = [official_score(actual, sequence, copy.deepcopy(costs)) for sequence in sequences]
        self.assertEqual([row["official_score"] for row in result["rows"]], expected)

    def test_checkpoint_rejects_duplicate_route(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scores.jsonl"
            path.write_text(json.dumps({"route_id": "R"}) + "\n" + json.dumps({"route_id": "R"}) + "\n")
            with self.assertRaises(RuntimeError):
                _checkpoint_ids(path)


if __name__ == "__main__":
    unittest.main()
