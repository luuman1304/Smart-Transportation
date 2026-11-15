import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path

from lastmile_data.baselines import evaluate, nearest_neighbor, travel_time, two_opt, validate_sequence
from lastmile_data.benchmark import objects, run


class BaselineTests(unittest.TestCase):
    def test_directed_two_opt_matches_exhaustive_best_reversal(self):
        rng = random.Random(42)
        for _ in range(20):
            stops = list('ABCDE')
            costs = {a: {b: 0 if a == b else rng.randrange(1, 100) for b in stops} for a in stops}
            candidates = [stops] + [stops[:i] + list(reversed(stops[i:j+1])) + stops[j+1:]
                                    for i, j in itertools.combinations(range(1, 5), 2)]
            result = two_opt(stops, costs, 1)
            self.assertEqual(travel_time(result, costs), min(travel_time(s, costs) for s in candidates))
            validate_sequence(result, stops, 'A')

    def test_nearest_neighbor_tie_break(self):
        costs = {a: {b: int(a != b) for b in 'CBA'} for a in 'CBA'}
        self.assertEqual(nearest_neighbor(costs, 'A'), list('ABC'))

    def test_invalid_sequences(self):
        for sequence in (list('ABA'), list('BA'), list('A')):
            with self.assertRaises(ValueError):
                validate_sequence(sequence, list('AB'), 'A')

    def fixture(self):
        route = {'station_code': 'S1', 'route_score': 'High', 'date_YYYY_MM_DD': '2018-01-01',
                 'departure_time_utc': '00:00:00', 'stops': {'A': {'type': 'Station'}, 'B': {'type': 'Dropoff'}}}
        costs = {'A': {'A': 0, 'B': 10}, 'B': {'A': 20, 'B': 0}}
        packages = {'A': {}, 'B': {'p': {'planned_service_time_seconds': 5,
                    'time_window': {'start_time_utc': '2018-01-01 00:00:15', 'end_time_utc': '2018-01-01 00:00:20'}}}}
        return route, costs, packages

    def test_time_accounting_and_lateness(self):
        route, costs, packages = self.fixture()
        result = evaluate(list('AB'), route, packages, costs)
        self.assertEqual(result['travel_seconds'], 30)
        self.assertEqual(result['waiting_seconds'], 5)
        self.assertEqual(result['duration_seconds'], 40)
        self.assertTrue(result['time_windows_satisfied'])
        costs['A']['B'] = 25
        result = evaluate(list('AB'), route, packages, costs)
        self.assertEqual(result['package_lateness_seconds'], 5)
        self.assertEqual(result['late_packages'], 1)
        costs['A']['B'] = float('nan')
        with self.assertRaises(ValueError):
            evaluate(list('AB'), route, packages, costs)

    def test_stream_and_training_only_runner(self):
        route, costs, packages = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'almrrc2021-data-training' / 'model_build_inputs'
            source.mkdir(parents=True)
            for name, payload in [('route_data', route), ('package_data', packages),
                                  ('travel_times', costs), ('actual_sequences', {'actual': {'A': 0, 'B': 1}})]:
                (source / (name + '.json')).write_text(json.dumps({'RouteID_1': payload}))
            city = root / 'map.csv'
            city.write_text('station_code,metropolitan_group\nS1,Boston\n')
            # No evaluation directory exists: runner must complete without it.
            run(root, city, root / 'output', passes=2)
            summary = json.loads((root / 'output' / 'summary.json').read_text())
            self.assertEqual(summary['observed']['routes'], 1)
            folds = json.loads((root / 'output' / 'training_folds.json').read_text())
            self.assertEqual(folds['Boston']['validation'], ['RouteID_1'])
            self.assertEqual(folds['Boston']['train'], [])
            stream = root / 'stream.json'
            payload = {'large': 'x' * 1100000, 'small': {'x': 3}}
            stream.write_text(json.dumps(payload))
            self.assertEqual(dict(objects(stream)), payload)
            stream.write_text('{"a": 1,}')
            with self.assertRaises(ValueError):
                list(objects(stream))
