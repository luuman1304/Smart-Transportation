"""Fail-closed adapter from Amazon raw JSON/JSONL to canonical JSONL.

The public Amazon schema must be inspected before adding field aliases. This
adapter accepts common aliases but refuses to infer city labels from
anonymized coordinates. City metadata must come from a verified CSV mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

from .validation import assert_no_route_split_leakage, validate_canonical_records


ROUTE_ID_KEYS = ("route_id", "routeId", "Route_ID", "routeID", "id")
CITY_KEYS = ("city", "metro", "metropolitan_area")
QUALITY_KEYS = ("route_quality", "routeQuality", "quality", "route_score")
SEQUENCE_KEYS = ("stop_sequence", "stopSequence", "sequence", "observed_sequence")


def _first(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def read_city_map(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames:
            raise ValueError("city map must contain a header")
        if {"route_id", "city"}.issubset(rows.fieldnames):
            key_field, city_field = "route_id", "city"
        elif {"station_code", "metropolitan_group"}.issubset(rows.fieldnames):
            key_field, city_field = "station_code", "metropolitan_group"
        else:
            raise ValueError(
                "city map must contain route_id/city or "
                "station_code/metropolitan_group columns"
            )
        mapping: dict[str, str] = {}
        for row in rows:
            key = row[key_field].strip()
            city = row[city_field].strip().replace("_", " ")
            if not key or not city or key in mapping:
                raise ValueError(f"city map contains an empty or duplicate {key_field}")
            mapping[key] = city
    return mapping


def iter_json_payloads(path: Path) -> Iterator[Any]:
    """Yield JSON values from .json and .jsonl files."""

    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        return
    with path.open(encoding="utf-8") as handle:
        try:
            yield json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}") from exc


def iter_route_objects(payload: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, Mapping):
                raise ValueError("route list contains a non-object record")
            yield item
        return
    if not isinstance(payload, Mapping):
        raise ValueError("JSON payload must be an object or list")
    if payload and all(str(key).startswith("RouteID_") for key in payload):
        for route_id, route in payload.items():
            if not isinstance(route, Mapping):
                raise ValueError(f"route {route_id!r} is not an object")
            item = dict(route)
            item["route_id"] = route_id
            yield item
        return
    for key in ("routes", "data", "records", "instances"):
        nested = payload.get(key)
        if isinstance(nested, list):
            yield from iter_route_objects(nested)
            return
    yield payload


def source_split_from_path(path: Path) -> str:
    normalized = "/".join(part.lower() for part in path.parts)
    if "training" in normalized or "/train/" in normalized:
        return "training"
    if "evaluation" in normalized or "/eval/" in normalized or "test" in normalized:
        return "evaluation"
    raise ValueError(f"cannot infer official source split from path: {path}")


def canonicalize_route(
    obj: Mapping[str, Any],
    source_file: Path,
    city_map: Mapping[str, str],
    sidecars: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    route_id_value = _first(obj, ROUTE_ID_KEYS)
    if route_id_value is None:
        raise ValueError(f"route ID not found in {source_file}")
    route_id = str(route_id_value).strip()
    city_value = _first(obj, CITY_KEYS)
    station_code = str(obj.get("station_code", "")).strip()
    city = (
        str(city_value).strip()
        if city_value is not None
        else city_map.get(route_id, city_map.get(station_code, ""))
    )
    if not city:
        raise ValueError(
            f"city is missing for route_id={route_id!r}, station_code={station_code!r}; "
            "provide a verified city map"
        )
    sidecars = sidecars or {}
    sequence = _first(obj, SEQUENCE_KEYS)
    if sequence is None:
        sequence_record = sidecars.get("actual_sequences", {}).get(route_id, {})
        sequence = sequence_record.get("actual") if isinstance(sequence_record, Mapping) else None
    quality = _first(obj, QUALITY_KEYS)
    record = {
        "route_id": route_id,
        "city": city,
        "source_split": source_split_from_path(source_file),
        "route_quality": quality,
        "stop_sequence": sequence,
        "route_time": obj.get("route_time", obj.get("total_route_time")),
        "transit_time": obj.get("transit_time"),
        "service_time": obj.get("service_time"),
        "station_code": station_code,
        "route_date": obj.get("date_YYYY_MM_DD"),
        "departure_time_utc": obj.get("departure_time_utc"),
        "executor_capacity_cm3": obj.get("executor_capacity_cm3"),
        "stops": obj.get("stops"),
        "source_file": str(source_file),
    }
    return record


def prepare_directory(raw_dir: Path, city_map_path: Path, output_path: Path) -> dict[str, Any]:
    city_map = read_city_map(city_map_path)
    records: list[dict[str, Any]] = []
    all_json_files = sorted(
        path for path in raw_dir.rglob("*")
        if path.suffix.lower() in {".json", ".jsonl"}
    )
    files = [
        path for path in all_json_files
        if path.name in {"route_data.json", "eval_route_data.json"}
    ]
    if not files:
        files = [path for path in all_json_files if path.name == "routes.json"]
    if not files:
        raise FileNotFoundError(f"no JSON/JSONL files found under {raw_dir}")
    for path in files:
        parent = path.parent
        sidecars: dict[str, Mapping[str, Any]] = {}
        sidecar_names = {
            "actual_sequences": "actual_sequences.json"
            if source_split_from_path(path) == "training"
            else "eval_actual_sequences.json",
        }
        for kind, name in sidecar_names.items():
            sidecar_path = parent / name
            if not sidecar_path.exists() and source_split_from_path(path) == "evaluation":
                sidecar_path = parent.parent / "model_score_inputs" / name
            if sidecar_path.exists():
                payload = next(iter_json_payloads(sidecar_path))
                if not isinstance(payload, Mapping):
                    raise ValueError(f"sidecar must be an object: {sidecar_path}")
                sidecars[kind] = payload
        for payload in iter_json_payloads(path):
            for obj in iter_route_objects(payload):
                records.append(canonicalize_route(obj, path, city_map, sidecars))
    records = validate_canonical_records(records)
    assert_no_route_split_leakage(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"records": len(records), "source_files": len(files), "output": str(output_path)}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--city-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = prepare_directory(args.raw_dir, args.city_map, args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
