import unittest
import json
import tempfile
from pathlib import Path

from lastmile_data.prepare import prepare_directory
from lastmile_data.validation import (
    SplitLeakageError,
    assert_no_route_split_leakage,
    build_class_weights,
    find_route_split_leakage,
    summarize_imbalance,
    validate_canonical_records,
)


def route(route_id, city, split, quality):
    return {
        "route_id": route_id,
        "city": city,
        "source_split": split,
        "route_quality": quality,
    }


class ValidationTests(unittest.TestCase):
    def test_validation_normalizes_identifiers_and_labels(self):
        records = validate_canonical_records([route(" r1 ", "Seattle", "TRAINING", "HIGH")])
        self.assertEqual(records[0]["route_id"], "r1")
        self.assertEqual(records[0]["source_split"], "training")
        self.assertEqual(records[0]["route_quality"], "high")

    def test_duplicate_route_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate route_id"):
            validate_canonical_records([
                route("r1", "Seattle", "training", "high"),
                route("r1", "Seattle", "training", "low"),
            ])

    def test_unknown_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid route_quality"):
            validate_canonical_records([route("r1", "Seattle", "training", "unknown")])

    def test_split_leakage_is_detected(self):
        records = [
            route("r1", "Seattle", "training", "high"),
            route("r1", "Seattle", "evaluation", "high"),
        ]
        self.assertEqual(find_route_split_leakage(records), {"r1": {"training", "evaluation"}})
        with self.assertRaises(SplitLeakageError):
            assert_no_route_split_leakage(records)

    def test_class_weights_handle_imbalance_without_resampling(self):
        records = [
            route("r1", "Seattle", "training", "high"),
            route("r2", "Seattle", "training", "high"),
            route("r3", "Seattle", "training", "high"),
            route("r4", "Seattle", "training", "low"),
        ]
        report = summarize_imbalance(records)
        weights = build_class_weights(report)
        self.assertEqual(report.minority_class, "low")
        self.assertGreater(weights["low"], weights["high"])
        self.assertAlmostEqual(weights["high"] * 3 + weights["low"], 4.0)

    def test_prepare_directory_creates_canonical_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw" / "training"
            raw.mkdir(parents=True)
            (raw / "routes.json").write_text(
                json.dumps([{"route_id": "r1", "route_quality": "high", "stops": ["a", "b"]}]),
                encoding="utf-8",
            )
            city_map = root / "city_map.csv"
            city_map.write_text("route_id,city\nr1,Seattle\n", encoding="utf-8")
            output = root / "processed" / "routes.jsonl"
            summary = prepare_directory(raw.parent.parent, city_map, output)
            self.assertEqual(summary["records"], 1)
            record = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(record["source_split"], "training")
            self.assertEqual(record["city"], "Seattle")

    def test_prepare_directory_reads_amazon_keyed_route_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw" / "almrrc2021-data-training" / "model_build_inputs"
            raw.mkdir(parents=True)
            (raw / "route_data.json").write_text(
                json.dumps({
                    "RouteID_r1": {
                        "station_code": "DLA3",
                        "route_score": "High",
                        "stops": {"AA": {"type": "Station"}},
                    }
                }),
                encoding="utf-8",
            )
            (raw / "actual_sequences.json").write_text(
                json.dumps({"RouteID_r1": {"actual": ["AA"]}}),
                encoding="utf-8",
            )
            city_map = root / "station_city_map.csv"
            city_map.write_text(
                "station_code,metropolitan_group\nDLA3,Los Angeles\n",
                encoding="utf-8",
            )
            output = root / "processed" / "routes.jsonl"
            summary = prepare_directory(raw.parent.parent, city_map, output)
            self.assertEqual(summary["records"], 1)
            record = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(record["route_id"], "RouteID_r1")
            self.assertEqual(record["city"], "Los Angeles")
            self.assertEqual(record["route_quality"], "high")
            self.assertEqual(record["stop_sequence"], ["AA"])


if __name__ == "__main__":
    unittest.main()
